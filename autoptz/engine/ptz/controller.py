"""Closed-loop PTZ motion controller.

Pipeline per tick:
  raw error → dead-zone → one-euro filter → PD + velocity feed-forward
            → clamp → response curve → backend.move_velocity()

  zoom: subject_height error → PD with hysteresis → backend zoom component

Coast-on-loss: when track_active goes False, hold last velocity for
coast_window_ms, then stop and enter SEARCHING state.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autoptz.config.models import PTZConfig
    from autoptz.engine.ptz.base import PTZBackend

log = logging.getLogger(__name__)

# Throttle for the control-loop tick failure WARNING: a persistent backend fault
# would otherwise emit a stack trace at the full control rate — at most one line
# per interval keeps it visible without flooding the log.
_TICK_WARN_INTERVAL_S = 5.0

# Stop-on-loss heartbeat (pump mode only): if the background loop is TRACKING but
# no fresh update() has arrived from the inference thread for longer than this, the
# loop commands a stop() so a stalled inference thread / dropped feed halts the
# camera instead of coasting on a frozen aim.  Bounded to a small floor and a few
# control periods, whichever is larger, so a momentarily slow producer doesn't trip
# it while a genuinely wedged one halts within a fraction of a second.
_HEARTBEAT_FLOOR_S = 0.3  # absolute minimum staleness before the heartbeat fires
_HEARTBEAT_PERIODS = 4.0  # …or this many control periods, whichever is larger
# Throttle for the heartbeat stop WARNING so a sustained producer stall logs once
# per interval rather than at the full control rate.
_HEARTBEAT_WARN_INTERVAL_S = 5.0

# Named framing presets → target subject-height fraction of the frame.
# A larger fraction means the subject fills more of the frame (tighter shot).
#   face           — head fills most of the frame (closeup)
#   head_shoulders — classic head-and-shoulders
#   upper_body     — waist-up (default)
#   full_body      — whole person in frame
#   wide           — person small in a wide establishing shot
_ZOOM_FRAMING_TARGETS = {
    "face": 0.80,
    "head_shoulders": 0.60,
    "upper_body": 0.45,
    "full_body": 0.30,
    "wide": 0.20,
    # Legacy names kept so an un-migrated config still resolves sanely.
    "tight": 0.60,
    "medium": 0.45,
}
_DEFAULT_ZOOM_FRAMING_TARGET = 0.45  # == upper_body
# Auto-zoom is deliberately slow + stable: it holds the crop across a wide band
# and only acts at the extremes, so it never "zooms out of nowhere".
_ZOOM_HEIGHT_EMA = 0.12  # heavy smoothing of subject height (de-spike bad frames)
_ZOOM_IN_BAND = 0.18  # zoom IN only when the subject is this much SMALLER than target
_ZOOM_TOO_CLOSE = 0.82  # safety: subject fills ≥82% of frame height → zoom OUT
_ZOOM_OUT_MARGIN = 0.30  # …or this much taller than the framing target, whichever first
_ZOOM_RATE_PER_S = 0.8  # max change in the normalized zoom command per second (slew)
_POWER = 1.2  # response-curve exponent (mild ease-in; 1.5 attenuated medium moves
# so much the follow felt sluggish — 1.2 stays smooth near centre but reaches
# useful speed much sooner for real movement)

# Oscillation guard: each frame-to-frame command sign-flip adds to a score that
# decays when motion is steady; the score scales a damping factor 1/(1+gain·score)
# applied to the command, so sustained hunting is progressively suppressed.
_OSC_DECAY = 0.6  # per-tick score decay (lower = forgets flips faster)
_OSC_GAIN = 0.6  # how hard the score damps the command
_OSC_MAX = 4.0  # score ceiling (caps the strongest damping)
_LEAD_MAX_S = 0.8  # hard cap on total lead time (avoid runaway extrapolation)
_ACCEL_LEAD_MAX = 0.3  # hard cap on the acceleration-term contribution to the aim
# Framing-box hold hysteresis: once parked inside the box, the subject must move
# this fraction *beyond* the box edge before following resumes — kills the
# start/stop chatter when they hover right on the boundary.
_HOLD_HYSTERESIS = 0.25
# Safe-zone CENTERING: the inner no-move deadband as a fraction of the zone's
# half-extent.  Inside it the camera holds (anti-jitter); beyond it the subject
# is eased back toward the zone centre (Center-Stage feel) instead of being
# frozen wherever they entered the oval.  Smaller → tighter centring.
_FRAMING_DEADBAND = 0.35
# Frame-edge guard: once the subject's measured error reaches this fraction of
# the half-frame (≈ near the screen edge), correct on the full offset toward the
# zone centre regardless of zone size — a big safe zone must never let the
# subject sit at/over the frame edge.
_EDGE_GUARD = 0.9


# ── one-euro filter ───────────────────────────────────────────────────────────


class OneEuroFilter:
    """Adaptive low-latency filter: low jitter on slow signals, low lag on fast ones.

    Reference: Casiez et al. 2012, "1€ Filter: A Simple Speed-based Low-pass
    Filter for Noisy Input in Interactive Systems".
    """

    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 0.007,
        dcutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x: float | None = None
        self._dx: float = 0.0
        self._last_t: float | None = None

    @staticmethod
    def _alpha(freq: float, cutoff: float) -> float:
        te = 1.0 / freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, t: float | None = None) -> float:
        if t is not None and self._last_t is not None and t > self._last_t:
            self.freq = 1.0 / (t - self._last_t)
        if t is not None:
            self._last_t = t

        if self._x is None:
            self._x = x
            return x

        # filtered derivative
        dx = (x - self._x) * self.freq
        alpha_d = self._alpha(self.freq, self.dcutoff)
        self._dx = alpha_d * dx + (1.0 - alpha_d) * self._dx

        # adaptive cutoff → filter value
        cutoff = self.mincutoff + self.beta * abs(self._dx)
        alpha = self._alpha(self.freq, cutoff)
        self._x = alpha * x + (1.0 - alpha) * self._x
        return self._x

    def reset(self, x: float = 0.0) -> None:
        self._x = None
        self._dx = 0.0
        self._last_t = None


# ── helpers ───────────────────────────────────────────────────────────────────


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _shape(x: float) -> float:
    """Non-linear response curve: ease-in (gentle near zero, full speed at ±1)."""
    return math.copysign(abs(x) ** _POWER, x) if x != 0.0 else 0.0


# Dynamic catch-up speed: the per-axis speed ceiling is scaled by 1 + gain·strength·
# min(1, |error|/ref), so the camera speeds up the further the subject is from the
# target framing and eases back to the configured speed near centre.
_DYN_GAIN = 1.5  # max extra-speed multiplier at full strength + far error (→ up to 2.5×)
_DYN_E_REF = 0.6  # aim error (fraction of half-frame) at which the boost saturates


def _catch_up_boost(error: float, strength: float) -> float:
    """Error-proportional speed multiplier (``>= 1.0``).

    ``error`` is the normalized aim error toward the setpoint (0 at centre, ~1 at
    the frame edge); ``strength`` is the user's catch-up control in ``[0, 1]``
    (0 disables → always ``1.0``).  Saturates at ``_DYN_E_REF`` so a far subject
    gets full catch-up speed without the boost running away.
    """
    if strength <= 0.0:
        return 1.0
    return 1.0 + _DYN_GAIN * strength * min(1.0, abs(error) / _DYN_E_REF)


def _zone_norm(dx: float, dy: float, hw: float, hh: float, roundness: float) -> float:
    """Normalized distance from a zone centre; ``<= 1.0`` means inside the zone.

    Blends an ellipse (``roundness=1`` → ``(dx/hw)²+(dy/hh)²``) and a rectangle
    (``roundness=0`` → ``max((dx/hw)², (dy/hh)²)``) so a configured "square" zone
    actually contains its corners instead of being treated as an ellipse.  Both
    forms equal 1.0 on the axis-aligned edge, so the boundary semantics are stable
    across ``roundness``.
    """
    hw = max(1e-6, hw)
    hh = max(1e-6, hh)
    nx = (dx / hw) ** 2
    ny = (dy / hh) ** 2
    r = _clamp(roundness, 0.0, 1.0)
    return r * (nx + ny) + (1.0 - r) * max(nx, ny)


def _smoothstep(t: float) -> float:
    """Classic cubic smoothstep: 0 at t=0, 1 at t=1, zero derivative at both ends."""
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _soft_deadband(d: float, dead: float, band: float = 0.0) -> float:
    """Zero within ``±dead``; nonlinear ramp over ``band``; linear excess beyond.

    With ``band=0`` (the default) this reproduces the original hard-knee
    behaviour: ``copysign(abs(d) - dead, d)`` for ``|d| > dead``.

    With ``band > 0`` a smoothstep ramps the effective slope from 0 → 1 over the
    width ``band`` just past the dead-zone edge, making the response C1-continuous
    at the edge (no slope discontinuity / "click" when the camera starts moving).
    Beyond ``dead + band`` the output is indistinguishable from the linear excess.

    Feeding this to the PD makes the steady state ``|error| ≤ dead`` — the subject
    settles *within* the deadband of the setpoint (i.e. centred), rather than the
    PD holding wherever they happened to enter a wide region.
    """
    if dead <= 0.0 and band <= 0.0:
        return d
    if abs(d) <= dead:
        return 0.0
    excess = abs(d) - dead
    if band > 0.0 and excess < band:
        # Nonlinear transition zone: scale the excess by a smoothstep so the
        # slope rises continuously from 0 at the dead-zone edge to 1 at band.
        effective = excess * _smoothstep(excess / band)
    else:
        effective = excess
    return math.copysign(effective, d)


def _slew(prev: float, target: float, max_step: float) -> float:
    """Rate-limit *acceleration* only: ramp speed UP gently, slow DOWN freely.

    Increasing the command magnitude is capped to *max_step* (smooth start, no
    "speeds out of nowhere"); decreasing it (toward zero — slowing or stopping) is
    unrestricted, so the camera stops promptly when the subject enters the box.
    """
    if max_step <= 0.0:
        return target
    if abs(target) <= abs(prev):
        return target  # decelerating / stopping → immediate
    return prev + _clamp(target - prev, -max_step, max_step)


# ── controller state ──────────────────────────────────────────────────────────


class ControllerState(Enum):
    IDLE = auto()  # no target assigned
    TRACKING = auto()  # active PD loop
    COASTING = auto()  # target LOST; holding last velocity for coast_window
    SEARCHING = auto()  # coast expired; stopped and waiting for re-ID


# ── shared update payload ─────────────────────────────────────────────────────


@dataclass
class _TrackPayload:
    error: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    subject_height: float = 0.0
    track_active: bool = False
    seq: int = 0


# ── controller ────────────────────────────────────────────────────────────────


class PTZController:
    """Rate-limited closed-loop PTZ controller.

    Usage (threaded):
        ctrl = PTZController(backend, cfg)
        ctrl.start()
        # from inference thread:
        ctrl.update(error=(ex, ey), velocity=(vx, vy), subject_height=h, track_active=True)
        ...
        ctrl.stop()   # sends backend.stop() before returning

    Usage (synchronous / tests):
        pan, tilt, zoom = ctrl.step(error, velocity, subject_height, track_active, t=0.0)
    """

    def __init__(
        self,
        backend: PTZBackend,
        cfg: PTZConfig,
        *,
        coast_window_ms: int = 1500,
        rate_hz: float = 20.0,
    ) -> None:
        self._backend = backend
        self._cfg = cfg
        self._coast_window_s = coast_window_ms / 1000.0
        self._rate_hz = rate_hz

        # per-axis one-euro filters (freq seed = rate_hz); cutoff/beta are set
        # from cfg.aim_smoothing by _apply_smoothing (called next + on re-acquire).
        self._filt_ex = OneEuroFilter(freq=rate_hz, mincutoff=1.0, beta=0.01)
        self._filt_ey = OneEuroFilter(freq=rate_hz, mincutoff=1.0, beta=0.01)
        self._apply_smoothing()

        # controller state
        self._state = ControllerState.IDLE
        self._coast_start: float = 0.0
        self._coast_pan: float = 0.0
        self._coast_tilt: float = 0.0
        self._search_start: float = 0.0  # when SEARCHING began (loss-recovery zoom-out)

        # PD derivative state
        self._prev_ex_f: float = 0.0
        self._prev_ey_f: float = 0.0
        self._last_t: float = -1.0

        # Previous (ego-corrected) target velocity, for the opt-in acceleration
        # term of the predictive lead (predict_accel_gain > 0).
        self._prev_vx: float = 0.0
        self._prev_vy: float = 0.0

        # PID integral accumulators (anti-windup clamped in _pd_step)
        self._int_ex: float = 0.0
        self._int_ey: float = 0.0

        # Oscillation-guard state (sign of last command per axis + flip score)
        self._prev_pan_sign: int = 0
        self._prev_tilt_sign: int = 0
        self._flip_score: float = 0.0

        # Slew-rate limiter state: the last emitted pan/tilt command, so the next
        # one can only move toward its target by ``max_accel * dt`` (smooth ramp).
        self._slew_pan: float = 0.0
        self._slew_tilt: float = 0.0

        # Framing-box hold latch (hysteresis): True while parked inside the box.
        self._holding: bool = False

        # Auto-zoom state: heavily-smoothed subject height + slewed zoom command,
        # so zoom is slow/stable and only reacts to sustained size changes.
        self._zoom_height_ema: float | None = None
        self._zoom_cmd: float = 0.0

        # Measured end-to-end loop latency (s), fed by the worker each tick; used
        # as extra lead so the aim anticipates the real pipeline delay.
        self._loop_latency_s: float = 0.0

        # last command sent (for rate-limit suppression)
        self._last_pan: float = 0.0
        self._last_tilt: float = 0.0
        self._last_zoom: float = 0.0

        # thread-safe payload from inference thread
        self._payload = _TrackPayload()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # True only while the background _loop is running; gates the stop-on-loss
        # heartbeat so the synchronous step() path (tests / inline mode) never trips
        # it (step() refreshes the payload itself, immediately before _tick).
        self._loop_running: bool = False

        # Stop-on-loss heartbeat (pump mode): monotonic time of the last update()
        # from the inference thread, and the staleness threshold past which the
        # background loop halts a still-TRACKING camera.  The threshold is a few
        # control periods or a small floor, whichever is larger.
        self._last_update_t: float = time.monotonic()
        self._heartbeat_stale_s = max(_HEARTBEAT_FLOOR_S, _HEARTBEAT_PERIODS / max(1.0, rate_hz))

        # Throttle for the control-loop tick failure WARNING (monotonic seconds).
        self._last_tick_warn_t = 0.0
        # Throttle for the heartbeat stop WARNING (monotonic seconds).
        self._last_heartbeat_warn_t = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> ControllerState:
        return self._state

    def status_snapshot(self, t: float | None = None) -> dict[str, float | str]:
        """Return state + recovery timing for operator-facing telemetry."""
        now = time.monotonic() if t is None else float(t)
        state = self._state.name.lower()
        coast_remaining = 0.0
        search_remaining = 0.0
        action = ""
        if self._state == ControllerState.COASTING:
            coast_remaining = max(0.0, self._coast_window_s - (now - self._coast_start))
            action = "holding"
        elif self._state == ControllerState.SEARCHING:
            window = float(getattr(self._cfg, "reacquire_window_s", 0.0))
            search_remaining = max(0.0, window - (now - self._search_start))
            zoom_out = float(getattr(self._cfg, "loss_zoom_out", 0.0))
            action = "zooming_out" if zoom_out > 0.0 and search_remaining > 0.0 else "standby"
        elif self._state == ControllerState.TRACKING:
            action = "tracking"
        return {
            "state": state,
            "action": action,
            "coast_remaining_s": coast_remaining,
            "search_remaining_s": search_remaining,
        }

    def set_loop_latency(self, seconds: float) -> None:
        """Report the measured capture+inference latency for latency-lead.

        Cheap to call every tick; clamped to a sane range so a spurious spike
        can't extrapolate the aim wildly.
        """
        self._loop_latency_s = _clamp(float(seconds), 0.0, _LEAD_MAX_S)

    def update_config(self, cfg: PTZConfig) -> None:
        """Swap in a new PTZConfig live (e.g. the Advanced tuning sliders).

        Only tuning *values* are read per tick from ``self._cfg``; the backend and
        coast window are fixed at construction, so this never needs a rebuild.
        """
        self._cfg = cfg
        self._apply_smoothing()

    def _apply_smoothing(self) -> None:
        """Map ``cfg.aim_smoothing`` (0..1) onto the one-euro filter parameters.

        Higher smoothing → lower cutoff (calmer, more lag); lower smoothing →
        higher cutoff + beta (snappier, anticipatory).  0.5 ≈ the original tuning.
        """
        s = _clamp(float(getattr(self._cfg, "aim_smoothing", 0.5)), 0.0, 1.0)
        mincutoff = max(0.2, 2.0 - 1.8 * s)
        beta = 0.005 + (1.0 - s) * 0.05
        for f in (self._filt_ex, self._filt_ey):
            f.mincutoff = mincutoff
            f.beta = beta

    def update(
        self,
        error: tuple[float, float],
        velocity: tuple[float, float],
        subject_height: float,
        track_active: bool,
    ) -> None:
        """Called from the inference thread with the latest tracking data."""
        with self._lock:
            self._payload = _TrackPayload(
                error=error,
                velocity=velocity,
                subject_height=subject_height,
                track_active=track_active,
                seq=self._payload.seq + 1,
            )
        # Heartbeat: record recency so the background loop can detect a stalled
        # producer and halt the camera (pump mode).  Monotonic so a wall-clock
        # adjustment can't make the feed look stale.
        self._last_update_t = time.monotonic()

    def step(
        self,
        error: tuple[float, float],
        velocity: tuple[float, float],
        subject_height: float,
        track_active: bool,
        t: float | None = None,
    ) -> tuple[float, float, float]:
        """Synchronous single tick (tests / manual drive).

        Updates internal state, calls backend, and returns (pan, tilt, zoom)."""
        with self._lock:
            self._payload = _TrackPayload(
                error=error,
                velocity=velocity,
                subject_height=subject_height,
                track_active=track_active,
                seq=self._payload.seq + 1,
            )
        return self._tick(t=t if t is not None else time.perf_counter())

    def goto_preset(self, idx: int) -> None:
        self._backend.goto_preset(idx)

    def save_preset(self, idx: int) -> None:
        self._backend.save_preset(idx)

    def start(self) -> None:
        """Start the background PTZ thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="ptz-ctrl", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and send stop to the backend."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._backend.stop()
        except Exception:
            pass

    def close(self) -> None:
        """Stop tracking and release backend resources."""
        self.stop()
        try:
            self._backend.close()
        except Exception:
            pass

    def __enter__(self) -> PTZController:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── background thread ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        interval = 1.0 / self._rate_hz
        # Reset recency on (re)start so a long-idle controller doesn't immediately
        # trip the heartbeat before the first update() of the new session arrives.
        self._last_update_t = time.monotonic()
        self._loop_running = True
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self._tick(t=t0)
            except Exception as exc:  # noqa: BLE001
                # Was a bare ``pass`` — a backend/compute fault silently froze the
                # PTZ with no log at all.  Surface it as a throttled WARNING so a
                # persistent control fault is visible without spamming the log at
                # the control rate.  The loop keeps running so motion can recover.
                now = time.monotonic()
                if now - self._last_tick_warn_t >= _TICK_WARN_INTERVAL_S:
                    self._last_tick_warn_t = now
                    log.warning("PTZ control tick failed (%s); continuing", exc, exc_info=True)
            elapsed = time.perf_counter() - t0
            self._stop_event.wait(max(0.0, interval - elapsed))
        self._loop_running = False
        # guarantee stop on thread exit
        try:
            self._backend.stop()
        except Exception:
            pass

    # ── tick (core logic) ─────────────────────────────────────────────────────

    def _tick(self, t: float) -> tuple[float, float, float]:
        # Stop-on-loss heartbeat (pump mode only): if we're driving the camera but
        # the inference thread has gone quiet (stalled / feed dropped) past the
        # staleness threshold, halt rather than coast on a frozen aim.  Gated on the
        # background loop running so the synchronous step() path never trips it.
        if self._loop_running and self._heartbeat_stalled():
            return self._heartbeat_stop()

        with self._lock:
            payload = self._payload

        pan_cmd, tilt_cmd, zoom_cmd = self._compute(payload, t)

        # only send if command changed meaningfully
        if (
            abs(pan_cmd - self._last_pan) > 1e-4
            or abs(tilt_cmd - self._last_tilt) > 1e-4
            or abs(zoom_cmd - self._last_zoom) > 1e-4
        ):
            self._backend.move_velocity(pan_cmd, tilt_cmd, zoom_cmd)
            self._last_pan = pan_cmd
            self._last_tilt = tilt_cmd
            self._last_zoom = zoom_cmd

        return pan_cmd, tilt_cmd, zoom_cmd

    def _heartbeat_stalled(self) -> bool:
        """True when actively driving but the producer feed has gone stale.

        Only relevant while TRACKING/COASTING (a camera that is actually moving or
        about to be stopped by coast logic): IDLE/SEARCHING are already halted, so a
        quiet feed there is expected and must not be flagged.
        """
        if self._state not in (ControllerState.TRACKING, ControllerState.COASTING):
            return False
        return (time.monotonic() - self._last_update_t) > self._heartbeat_stale_s

    def _heartbeat_stop(self) -> tuple[float, float, float]:
        """Halt the camera because the inference feed stalled; transition to IDLE.

        Drops to IDLE so when fresh updates resume the controller does a clean
        (re-)acquire instead of resuming a frozen mid-pan command, and sends a real
        backend stop (bypassing the change-suppression in ``_tick``) so the halt is
        guaranteed even if the last command was already zero.
        """
        now = time.monotonic()
        if now - self._last_heartbeat_warn_t >= _HEARTBEAT_WARN_INTERVAL_S:
            self._last_heartbeat_warn_t = now
            log.warning(
                "PTZ heartbeat: no tracking update for %.2fs (>%.2fs) — halting camera",
                now - self._last_update_t,
                self._heartbeat_stale_s,
            )
        self._state = ControllerState.IDLE
        self._last_pan = self._last_tilt = self._last_zoom = 0.0
        try:
            self._backend.stop()
        except Exception:
            pass
        return 0.0, 0.0, 0.0

    def _compute(self, p: _TrackPayload, t: float) -> tuple[float, float, float]:
        cfg = self._cfg

        # ── state transitions ─────────────────────────────────────────────────
        if p.track_active:
            if self._state != ControllerState.TRACKING:
                # WARM re-acquire when the target was lost only BRIEFLY (still in
                # the coast window): it's ~where it was, so keep the slew-limiter
                # speed (and hold latch) so motion resumes from the pre-loss speed
                # instead of re-ramping from a dead stop — that cold restart of the
                # slew is the visible "find" lurch in the find/lose/find cycle
                # during pans.  Everything else still resets (below), so there's no
                # stale wind-up or filter/derivative artifact.
                warm = self._state == ControllerState.COASTING

                # (re-)acquired target: reset filters (and re-apply smoothing in
                # case the tuning changed while we were searching)
                self._filt_ex.reset()
                self._filt_ey.reset()
                self._apply_smoothing()
                self._prev_ex_f = 0.0
                self._prev_ey_f = 0.0
                self._prev_vx = 0.0
                self._prev_vy = 0.0
                self._last_t = -1.0
                # Fresh acquire: drop accumulated integral + hunting history so a
                # stale wind-up or flip score can't kick the new target.
                self._int_ex = 0.0
                self._int_ey = 0.0
                self._prev_pan_sign = 0
                self._prev_tilt_sign = 0
                self._flip_score = 0.0
                self._zoom_height_ema = None
                self._zoom_cmd = 0.0
                if not warm:
                    # COLD acquire (fresh target, or recovery after a long loss via
                    # SEARCHING where the subject may be anywhere): also stop the
                    # slew limiter and release the hold so motion ramps from rest.
                    self._slew_pan = 0.0
                    self._slew_tilt = 0.0
                    self._holding = False
                self._state = ControllerState.TRACKING
        else:
            if self._state == ControllerState.TRACKING:
                # just lost target: enter coast
                self._state = ControllerState.COASTING
                self._coast_start = t
                self._coast_pan = self._last_pan
                self._coast_tilt = self._last_tilt
            elif self._state == ControllerState.COASTING:
                if t - self._coast_start >= self._coast_window_s:
                    self._state = ControllerState.SEARCHING
                    self._search_start = t
                    # Stop the inherited pan/tilt drift; SEARCHING then zooms out
                    # to widen the view and re-find the subject (below).
                    try:
                        self._backend.stop()
                    except Exception:
                        pass
            # IDLE: nothing to do

        # ── compute commands per state ────────────────────────────────────────
        if self._state == ControllerState.TRACKING:
            pan_cmd, tilt_cmd = self._pd_step(p, t)
            zoom_cmd = self._zoom_step(p.subject_height) if cfg.auto_zoom else 0.0
        elif self._state == ControllerState.COASTING:
            pan_cmd, tilt_cmd = self._coast_pan, self._coast_tilt
            zoom_cmd = 0.0
        elif self._state == ControllerState.SEARCHING:
            # Loss recovery: hold pan/tilt still and gently zoom OUT for a window
            # so the lost subject is more likely to re-enter the (wider) frame,
            # rather than freezing on a tight shot of empty space.
            pan_cmd = tilt_cmd = 0.0
            zoom_out = float(getattr(cfg, "loss_zoom_out", 0.0))
            window = float(getattr(cfg, "reacquire_window_s", 0.0))
            if zoom_out > 0.0 and (t - self._search_start) < window:
                zoom_cmd = -_clamp(zoom_out * cfg.max_zoom_speed, 0.0, 1.0)
            else:
                zoom_cmd = 0.0
        else:  # IDLE
            pan_cmd = tilt_cmd = zoom_cmd = 0.0

        return pan_cmd, tilt_cmd, zoom_cmd

    def _pd_step(self, p: _TrackPayload, t: float) -> tuple[float, float]:
        cfg = self._cfg
        ex, ey = p.error
        vx, vy = p.velocity

        # 0. FRAMING — remap the measured error onto the framing setpoint.  With a
        #    safe zone this drives the subject toward the ZONE CENTRE (eased, with
        #    an inner deadband + frame-edge guard) instead of freezing them
        #    anywhere inside the oval; without one it's the legacy per-axis dead
        #    zone around the frame centre.  ``holding`` (subject settled in the
        #    deadband) zeroes the feed-forward velocity too so a parked subject
        #    can't trigger micro-moves.  See ``_framing_error``.
        #
        # 1. Only while *following* (outside the deadband) do we project the aim
        #    forward by the lead time (+ measured loop latency when
        #    ``lead_time_auto``), so the camera leads real motion without nudging a
        #    stationary subject.  The velocity is ego-corrected, so this leads the
        #    subject's world motion, not the camera's own pan.
        ex, ey, holding = self._framing_error(ex, ey, cfg)
        if holding:
            vx = vy = 0.0
        else:
            lead = float(getattr(cfg, "lead_time_s", 0.0))
            if getattr(cfg, "lead_time_auto", True):
                lead += self._loop_latency_s
            lead = _clamp(lead, 0.0, _LEAD_MAX_S)
            if lead > 0.0:
                ex += vx * lead
                ey += vy * lead
                # Opt-in 2nd-order term: anticipate a subject starting/stopping by
                # projecting the *change* in velocity (½·a·lead²·gain), clamped so a
                # noisy acceleration spike can't fling the aim off the subject.
                g = float(getattr(cfg, "predict_accel_gain", 0.0))
                if g > 0.0 and self._last_t >= 0.0:
                    dt_v = max(t - self._last_t, 1e-3)
                    half_l2 = 0.5 * lead * lead * g
                    ax = (vx - self._prev_vx) / dt_v
                    ay = (vy - self._prev_vy) / dt_v
                    ex += _clamp(ax * half_l2, -_ACCEL_LEAD_MAX, _ACCEL_LEAD_MAX)
                    ey += _clamp(ay * half_l2, -_ACCEL_LEAD_MAX, _ACCEL_LEAD_MAX)

        # 2. one-euro filter
        ex_f = self._filt_ex(ex, t)
        ey_f = self._filt_ey(ey, t)

        # 3. PD derivative (finite difference of filtered error).  On the very
        #    first tick (no previous time) assume one nominal control interval so
        #    the derivative doesn't spike and the slew limiter below still allows a
        #    normal first step (a tiny dt would otherwise freeze the command at 0).
        dt = (t - self._last_t) if self._last_t >= 0.0 else (1.0 / max(1.0, self._rate_hz))
        dt = max(dt, 1e-6)
        dex = (ex_f - self._prev_ex_f) / dt
        dey = (ey_f - self._prev_ey_f) / dt

        # 3b. integral (opt-in, ki>0) with anti-windup: accumulate the filtered
        #     error, but clamp the accumulator so ki·∫ can't exceed a bounded
        #     contribution, and freeze it while the command is saturated (below).
        #     While holding (parked in the box) bleed it instead of accumulating,
        #     so it can't build up and kick when following resumes.
        ki = float(getattr(cfg, "ki", 0.0))
        if holding:
            self._int_ex *= 0.9
            self._int_ey *= 0.9
        elif ki > 0.0 and self._last_t >= 0.0:
            i_max = 0.5 / ki  # cap ki·∫ at ±0.5 of a normalized command
            self._int_ex = _clamp(self._int_ex + ex_f * dt, -i_max, i_max)
            self._int_ey = _clamp(self._int_ey + ey_f * dt, -i_max, i_max)
        i_term_x = ki * self._int_ex
        i_term_y = ki * self._int_ey

        # 4. PID + velocity feed-forward
        pan_raw = cfg.kp * ex_f + i_term_x + cfg.kd * dex + cfg.kv * vx
        tilt_raw = cfg.kp * ey_f + i_term_y + cfg.kd * dey + cfg.kv * vy

        # 4b. anti-windup back-off: if a command saturates, bleed the integral so
        #     it can't keep growing while the axis is already maxed out.
        if ki > 0.0:
            if abs(pan_raw) > 1.0:
                self._int_ex *= 0.5
            if abs(tilt_raw) > 1.0:
                self._int_ey *= 0.5

        # 5. per-camera speed ceiling + dynamic catch-up + clamp + response curve.
        #    The ceiling is scaled per-axis by an error-proportional boost: far
        #    from the setpoint the camera speeds up to catch the subject; near it
        #    the boost is ~1 so centred framing stays smooth and precise.
        catch = float(getattr(cfg, "catch_up_speed", 0.0))
        pan_cap = cfg.max_pan_speed * _catch_up_boost(ex_f, catch)
        tilt_cap = cfg.max_tilt_speed * _catch_up_boost(ey_f, catch)
        pan_cmd = _shape(_clamp(pan_raw * pan_cap, -1.0, 1.0))
        tilt_cmd = _shape(_clamp(tilt_raw * tilt_cap, -1.0, 1.0))

        # 5b. oscillation guard — damp the command when it keeps flipping sign
        #     (self-sustained hunting), easing back to full speed once it settles.
        if getattr(cfg, "osc_guard", True):
            damp = self._osc_damping(pan_cmd, tilt_cmd)
            pan_cmd *= damp
            tilt_cmd *= damp

        # 5c. slew-rate limit — cap acceleration so the head ramps smoothly toward
        #     the target speed instead of jumping there (no "speeds out of nowhere").
        max_accel = float(getattr(cfg, "max_accel", 0.0))
        if max_accel > 0.0:
            step = max_accel * dt
            pan_cmd = _slew(self._slew_pan, pan_cmd, step)
            tilt_cmd = _slew(self._slew_tilt, tilt_cmd, step)
        self._slew_pan = pan_cmd
        self._slew_tilt = tilt_cmd

        # 6. soft limits (velocity: clamp to zero if at boundary)
        lim = cfg.soft_limits
        if lim is not None:
            if pan_cmd > 0 and self._backend.get_position() is not None:
                pos = self._backend.get_position()
                if pos is not None and pos.pan >= lim.pan_max:
                    pan_cmd = 0.0
            if pan_cmd < 0 and self._backend.get_position() is not None:
                pos = self._backend.get_position()
                if pos is not None and pos.pan <= lim.pan_min:
                    pan_cmd = 0.0

        # 7. invert
        if cfg.invert_pan:
            pan_cmd = -pan_cmd
        if cfg.invert_tilt:
            tilt_cmd = -tilt_cmd

        self._prev_ex_f = ex_f
        self._prev_ey_f = ey_f
        # Store the raw measured velocity (not the hold-zeroed locals) so the next
        # tick's acceleration term reflects real subject dynamics.
        self._prev_vx, self._prev_vy = p.velocity
        self._last_t = t

        return pan_cmd, tilt_cmd

    def _framing_error(self, ex: float, ey: float, cfg: Any) -> tuple[float, float, bool]:
        """Map measured (frame-centre) error → control error toward the setpoint.

        Returns ``(control_ex, control_ey, holding)``.  The PD drives the control
        error to zero, so the *setpoint* is wherever the control error is zero:

        - **Safe zone ON** → setpoint is the zone CENTRE.  An inner deadband
          (``_FRAMING_DEADBAND`` of the zone) holds the camera still (anti-jitter);
          beyond it the subject is eased back toward the centre via a soft deadband
          (the Center-Stage feel — no freezing at the oval edge).  A frame-edge
          guard (``_EDGE_GUARD``) abandons the deadband near the screen edge so a
          *wide* zone can never strand the subject off-frame.  ``roundness`` shapes
          the zone between ellipse and rectangle so a "square" respects its corners.
        - **Safe zone OFF** → legacy per-axis dead-zone around the frame centre
          (hold inside, full error outside), unchanged.

        Hold uses the *measured* error and a hysteresis band so hovering on the
        boundary doesn't start/stop chatter.
        """
        if not getattr(cfg, "safe_zone_enabled", False):
            holding = self._update_hold(ex, ey, cfg)
            return (0.0, 0.0, True) if holding else (ex, ey, False)

        cx = float(getattr(cfg, "safe_zone_x", 0.0))
        cy = float(getattr(cfg, "safe_zone_y", 0.0))
        hw = max(1e-3, float(getattr(cfg, "safe_zone_w", 0.15)))
        hh = max(1e-3, float(getattr(cfg, "safe_zone_h", 0.22)))
        roundness = float(getattr(cfg, "safe_zone_roundness", 1.0))
        dx = ex - cx
        dy = ey - cy

        # Frame-edge guard: near the screen edge, drop the deadband/hold and drive
        # on the full offset toward the zone centre (a wide zone must not let the
        # subject sit at the edge).
        if abs(ex) >= _EDGE_GUARD or abs(ey) >= _EDGE_GUARD:
            self._holding = False
            return dx, dy, False

        dbx = _FRAMING_DEADBAND * hw
        dby = _FRAMING_DEADBAND * hh
        # Latch ON inside the inner deadband; release only past the OUTER zone edge
        # (×(1+hyst)) — a wide quiet band, then ease all the way to centre.
        inside_db = _zone_norm(dx, dy, dbx, dby, roundness) <= 1.0
        outside_zone = _zone_norm(dx, dy, hw, hh, roundness) > (1.0 + _HOLD_HYSTERESIS) ** 2
        if self._holding:
            if outside_zone:
                self._holding = False
        elif inside_db:
            self._holding = True
        if self._holding:
            return 0.0, 0.0, True

        band = float(getattr(cfg, "nonlinear_band", 0.0))
        return _soft_deadband(dx, dbx, band), _soft_deadband(dy, dby, band), False

    def _update_hold(self, ex: float, ey: float, cfg: Any) -> bool:
        """Return whether to HOLD inside the per-axis dead-zone, with hysteresis.

        Used only when the safe zone is OFF (the safe-zone case is handled by
        :meth:`_framing_error`).  Once holding, the subject must travel
        ``_HOLD_HYSTERESIS`` beyond the dead-zone edge to resume following — so
        hovering on the boundary doesn't start/stop chatter.
        """
        dzx = max(1e-6, float(getattr(cfg, "deadzone_x", 0.0)))
        dzy = max(1e-6, float(getattr(cfg, "deadzone_y", 0.0)))
        inside = abs(ex) <= dzx and abs(ey) <= dzy
        m = 1.0 + _HOLD_HYSTERESIS
        outside = abs(ex) > dzx * m or abs(ey) > dzy * m

        if self._holding:
            if outside:
                self._holding = False
        elif inside:
            self._holding = True
        return self._holding

    def _osc_damping(self, pan_cmd: float, tilt_cmd: float) -> float:
        """Return a 0<d≤1 damping factor that shrinks while the command hunts.

        Each per-axis sign flip versus the previous tick bumps a score that
        otherwise decays; the score maps to ``1/(1+gain·score)`` so sustained
        oscillation is damped hard and steady motion is left at full speed.
        """
        score = self._flip_score * _OSC_DECAY
        for cur, prev_attr in (
            (pan_cmd, "_prev_pan_sign"),
            (tilt_cmd, "_prev_tilt_sign"),
        ):
            sign = 0 if abs(cur) < 1e-3 else (1 if cur > 0.0 else -1)
            prev = getattr(self, prev_attr)
            if sign != 0 and prev != 0 and sign != prev:
                score += 1.0
            setattr(self, prev_attr, sign)
        self._flip_score = min(score, _OSC_MAX)
        return 1.0 / (1.0 + _OSC_GAIN * self._flip_score)

    def _zoom_step(self, subject_height: float) -> float:
        """Slow, stable auto-zoom: hold the crop; only act at the extremes.

        Zoom is the most jarring axis when it twitches (it shifts the whole frame,
        which makes tilt chase it), so this is deliberately sluggish: the subject
        height is heavily EMA-smoothed, the command is slew-limited, and there's a
        wide neutral band.  Zoom OUT is a *safety* — it only fires when the subject
        is genuinely too close (filling the frame); zoom IN only when they're well
        below the framing target.  Otherwise the crop is left alone.
        """
        cfg = self._cfg
        if subject_height <= 0.0:
            # No reliable size reading → ease any zoom to a stop, don't hold one.
            self._zoom_cmd = _slew(self._zoom_cmd, 0.0, _ZOOM_RATE_PER_S / max(1.0, self._rate_hz))
            return self._zoom_cmd

        # Heavy smoothing so a single bad detection frame can't jolt the zoom.
        if self._zoom_height_ema is None:
            self._zoom_height_ema = subject_height
        else:
            a = _ZOOM_HEIGHT_EMA
            self._zoom_height_ema = a * subject_height + (1.0 - a) * self._zoom_height_ema
        height = self._zoom_height_ema

        # The unified "Framing" control drives the target subject height.
        framing = getattr(cfg, "framing", None) or cfg.zoom_framing
        target = _ZOOM_FRAMING_TARGETS.get(framing, _DEFAULT_ZOOM_FRAMING_TARGET)
        # Zoom OUT past a WIDE band above the framing target (keeps a tighter crop
        # than the target before pulling back), but never past the absolute
        # too-close safety floor (catches very tight framings about to overflow).
        out_threshold = min(target + _ZOOM_OUT_MARGIN, _ZOOM_TOO_CLOSE)
        in_threshold = target - _ZOOM_IN_BAND

        if height > out_threshold:
            raw = -_clamp((height - out_threshold) * 1.5, 0.0, 1.0)  # zoom OUT
        elif height < in_threshold:
            raw = _clamp((in_threshold - height) * 1.5, 0.0, 1.0)  # zoom IN
        else:
            raw = 0.0  # wide neutral band → keep the crop

        target_cmd = _clamp(raw * cfg.max_zoom_speed, -1.0, 1.0)
        # Slew so zoom eases in/out and never jolts (rate is per second).
        self._zoom_cmd = _slew(
            self._zoom_cmd, target_cmd, _ZOOM_RATE_PER_S / max(1.0, self._rate_hz)
        )
        return self._zoom_cmd
