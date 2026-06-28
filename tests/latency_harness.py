"""Latency-aware closed-loop tracking evaluation harness (headless, deterministic).

Drives the *real* ``PTZController.step`` seam inside a tiny closed loop with a
configurable feedback **dead time** (sensor/inference latency) and proves that
tracking error and command oscillation grow as the delay grows — the way a
real PTZ head hunts when its corrections lag the subject.

Why this exists
---------------
The current ``PTZController`` is *reactive* (PD on the latest measured error,
no motion prediction).  A pure-integrator plant fed a delayed error is the
textbook recipe for dead-time-induced overshoot: the controller keeps pushing
on stale information and overshoots, then corrects and overshoots back.  This
harness makes that physics measurable so a later *predictive* phase can show it
recovers the lost margin (the same tests should then degrade *less* with delay).

Design
------
* **Controller seam** — ``PTZController.step((ex,ey),(vx,vy),h,active,t)`` returns
  a normalized ``(pan,tilt,zoom)`` rate in ``[-1,1]``.  We only use 1-D pan.
* **Error convention** — normalized ``(target - aim) / half_frame``; here ``aim``
  and ``target`` already live in that normalized space (range ~``[-1,1]``), so the
  composite error is simply ``target - aim``.
* **Plant** — a scalar ``aim`` integrating the pan command:
  ``aim += pan * slew_gain * dt``.  This *pure integrator* (no damping in the
  plant) is what turns dead time into overshoot.
* **Delay seam** — a ring/history of past composite errors; the controller is fed
  ``e_seen = history[max(0, k-d)]`` with ``d = round(delay_ms/1000/dt)`` and a
  finite-difference ``v_seen``.  **Scoring always uses the TRUE error**
  ``target - aim`` for the current step, never the delayed signal.

Determinism
-----------
Every tick is driven with an explicit ``t = k*dt`` (no wall clock), the config is
fixed, and there is no hardware, model, or thread — so results are bit-stable.

Calibration note
----------------
The baseline gains (``kp=0.6``, ``slew_gain=6.0``, ``rate_hz=20`` → loop
gain/tick ≈ 0.18) are picked so the ``delay=0`` loop is *stable but not
over-damped*: snappy enough that dead time bites.  At the spec's nominal
``slew_gain=2.0`` (loop gain ≈ 0.06) the loop is so heavily damped that delay
barely changes the error and the sensitivity guarantees can't hold — so the
harness default is bumped to ``6.0`` to preserve the intended physics (more
delay ⇒ worse tracking).  The sinusoid default ``omega`` is likewise raised
(slow waves are nearly insensitive to a few hundred ms of lag) so that a
realistic delay produces a measurable phase-lag growth.  The built-in predictive
lead is disabled in the baseline cfg (``lead_time_s=0``, ``lead_time_auto=False``)
so the harness measures the *non-predictive* controller in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from autoptz.config.models import PTZConfig
from autoptz.engine.ptz.controller import PTZController

# Reuse the deterministic config builder + plant-free backend from the existing
# synthetic-tracking fakes (no duplication of the PTZConfig defaults / MockBackend).
from tests.synthetic_tracking import MockBackend, make_cfg

__all__ = [
    "LoopResult",
    "baseline_cfg",
    "make_cfg",
    "MockBackend",
    "step_profile",
    "ramp_profile",
    "sinusoid_profile",
    "run_latency_loop",
    "rms",
    "steady_state_error",
    "overshoot",
    "settling_time",
    "sign_flips",
    "peak_abs_error",
]


# ── baseline config ─────────────────────────────────────────────────────────────


def baseline_cfg(*, kp: float = 0.6, **kw: object) -> PTZConfig:
    """The stable non-predictive baseline used by the harness.

    A snappy reactive PD: smoothing off, oscillation guard left to the caller,
    and every *predictive* / nonlinear feature that would mask or compensate the
    dead-time signal disabled (built-in lead, catch-up speed boost, safe zone,
    dead zones, slew limiter, auto zoom).  What's left is essentially
    ``pan = shape(clamp(kp * error))`` so the latency effect is unobscured.
    """
    defaults: dict[str, object] = {
        "kp": kp,
        "aim_smoothing": 0.0,
        "osc_guard": False,
        "lead_time_s": 0.0,
        "lead_time_auto": False,
        "catch_up_speed": 0.0,
        "safe_zone_enabled": False,
        "deadzone_x": 0.0,
        "deadzone_y": 0.0,
        "max_accel": 0.0,
        "auto_zoom": False,
    }
    defaults.update(kw)
    return make_cfg(**defaults)


# ── motion profiles (normalized-error space, range ~[-1, 1]) ─────────────────────


def step_profile(n: int, amp: float = 0.6, t_step: int = 5) -> list[float]:
    """A step: 0 until ``t_step``, then ``amp`` — the classic overshoot probe."""
    return [0.0 if i < t_step else amp for i in range(n)]


def ramp_profile(n: int, v: float = 0.04, x0: float = -0.6) -> list[float]:
    """A constant-velocity ramp ``x0 + v*i`` (clamped to ``[-1, 1]``)."""
    return [max(-1.0, min(1.0, x0 + v * i)) for i in range(n)]


def sinusoid_profile(n: int, amp: float = 0.5, omega: float = 2.0, dt: float = 0.05) -> list[float]:
    """A sine ``amp*sin(omega * i*dt)`` — phase lag from dead time shows here.

    ``omega`` is in rad/s and is deliberately higher than a leisurely sweep so a
    few hundred ms of feedback delay is a non-trivial fraction of the period (a
    very slow wave is almost insensitive to lag).
    """
    return [amp * math.sin(omega * i * dt) for i in range(n)]


# ── closed-loop result ──────────────────────────────────────────────────────────


@dataclass
class LoopResult:
    """Per-tick traces from one closed-loop run.

    ``error`` is the SCORED true error ``target - aim`` (not the delayed signal);
    ``seen`` is what the controller was actually fed (delayed); ``command`` is the
    returned pan rate.
    """

    t: list[float] = field(default_factory=list)
    target: list[float] = field(default_factory=list)
    aim: list[float] = field(default_factory=list)
    error: list[float] = field(default_factory=list)
    seen: list[float] = field(default_factory=list)
    command: list[float] = field(default_factory=list)


# ── the loop ─────────────────────────────────────────────────────────────────────


def run_latency_loop(
    target: list[float],
    *,
    delay_ms: float,
    cfg: PTZConfig | None = None,
    rate_hz: float = 20.0,
    slew_gain: float = 6.0,
    subject_height: float = 0.45,
) -> LoopResult:
    """Run the delayed closed loop and return the per-tick traces.

    The controller sees a *delayed* composite error; the plant integrates the
    returned pan command; scoring uses the *true* error for the current tick.

    Args:
        target: per-tick setpoint in normalized-error space (range ~[-1, 1]).
        delay_ms: feedback dead time; ``d = round(delay_ms/1000 / dt)`` ticks.
        cfg: controller config; defaults to :func:`baseline_cfg` (kp=0.6).
        rate_hz: control rate; ``dt = 1/rate_hz`` and ``t = k*dt`` per tick.
        slew_gain: plant integrator gain (normalized units of aim per unit
            command per second).
        subject_height: fed to ``step`` (auto-zoom is off, so it's inert here).
    """
    if cfg is None:
        cfg = baseline_cfg()

    dt = 1.0 / rate_hz
    d = round((delay_ms / 1000.0) / dt)
    n = len(target)

    controller = PTZController(MockBackend(), cfg, rate_hz=rate_hz)

    res = LoopResult()
    aim = 0.0
    history: list[float] = []  # composite = target - aim, per tick
    prev_seen = 0.0

    for k in range(n):
        tgt = target[k]
        composite = tgt - aim  # TRUE error this tick
        history.append(composite)

        # Delay seam: feed the controller a PAST error (dead time), with a
        # finite-difference velocity of that same delayed signal.
        e_seen = history[max(0, k - d)]
        v_seen = (e_seen - prev_seen) / dt
        prev_seen = e_seen

        pan, _tilt, _zoom = controller.step(
            (e_seen, 0.0), (v_seen, 0.0), subject_height, True, k * dt
        )

        # Plant: pure integrator driven by the command.
        aim += pan * slew_gain * dt

        true_error = tgt - aim  # scored AFTER the move
        res.t.append(k * dt)
        res.target.append(tgt)
        res.aim.append(aim)
        res.error.append(true_error)
        res.seen.append(e_seen)
        res.command.append(pan)

    return res


# ── metrics ──────────────────────────────────────────────────────────────────────


def rms(xs: list[float]) -> float:
    """Root-mean-square of a sequence (0.0 for an empty sequence)."""
    if not xs:
        return 0.0
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def steady_state_error(error: list[float], tail: int = 20) -> float:
    """RMS of the last ``tail`` error samples — the residual once transients fade."""
    if not error:
        return 0.0
    return rms(error[-tail:])


def overshoot(aim: list[float], setpoint: float) -> float:
    """How far ``aim`` overshot ``setpoint`` past it (0.0 if it never crossed).

    For a positive setpoint this is ``max(aim) - setpoint`` (clamped at 0); for a
    negative setpoint it's the symmetric ``setpoint - min(aim)``.
    """
    if not aim:
        return 0.0
    if setpoint >= 0.0:
        return max(0.0, max(aim) - setpoint)
    return max(0.0, setpoint - min(aim))


def settling_time(error: list[float], dt: float, band: float = 0.05) -> float:
    """Time (s) after which ``|error|`` stays within ``band`` for the rest of the run.

    Returns ``len*dt`` (never settles) when the error re-leaves the band before the
    end — i.e. the LAST exit from the band sets the settling instant.
    """
    n = len(error)
    if n == 0:
        return 0.0
    last_out = -1
    for i, e in enumerate(error):
        if abs(e) > band:
            last_out = i
    if last_out < 0:
        return 0.0  # already within band the whole time
    if last_out >= n - 1:
        return n * dt  # never settled
    return (last_out + 1) * dt


def sign_flips(command: list[float], eps: float = 1e-3) -> int:
    """Count sign changes in a command stream, ignoring near-zero (|cmd|<eps) ticks.

    A flip is counted when a non-trivial command's sign differs from the previous
    non-trivial command's sign; sustained zero-crossing chatter ⇒ high count.
    """
    flips = 0
    prev = 0
    for c in command:
        if abs(c) < eps:
            continue
        sign = 1 if c > 0.0 else -1
        if prev != 0 and sign != prev:
            flips += 1
        prev = sign
    return flips


def peak_abs_error(error: list[float]) -> float:
    """Largest absolute error over the run (divergence / NaN sentinel)."""
    if not error:
        return 0.0
    return max(abs(x) for x in error)
