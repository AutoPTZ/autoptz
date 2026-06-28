"""Latency sensitivity evaluation for the (current, non-predictive) PTZController.

These tests pin the *physics* the harness is built to expose: a reactive PD on a
delayed error, integrated by a pure-integrator plant, tracks WORSE as the
feedback dead time grows.  The load-bearing assertion is Test B — error RMS is
monotone non-decreasing across delay and grows by >1.5x from 0 → 300 ms.  When a
later phase adds motion prediction, these same tests should degrade *less* with
delay, demonstrating the fix.

Run::

    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_synthetic_latency_eval.py -q
"""

from __future__ import annotations

import math

import pytest

from tests.latency_harness import (
    baseline_cfg,
    overshoot,
    peak_abs_error,
    ramp_profile,
    rms,
    run_latency_loop,
    settling_time,
    sign_flips,
    sinusoid_profile,
    steady_state_error,
    step_profile,
)

DELAYS_MS = [0, 50, 100, 200, 300]
RATE_HZ = 20.0
DT = 1.0 / RATE_HZ
N = 200

# Profiles keyed by name (the sinusoid's dt must match the loop's dt).
PROFILES = {
    "step": lambda: step_profile(N),
    "ramp": lambda: ramp_profile(N),
    "sinusoid": lambda: sinusoid_profile(N, dt=DT),
}


def _rms_error_over_delays(profile: list[float], *, kp: float, osc_guard: bool) -> list[float]:
    """RMS of the SCORED true error for each delay in ``DELAYS_MS``."""
    cfg = baseline_cfg(kp=kp, osc_guard=osc_guard)
    out: list[float] = []
    for d in DELAYS_MS:
        res = run_latency_loop(profile, delay_ms=d, cfg=cfg, rate_hz=RATE_HZ)
        out.append(rms(res.error))
    return out


# ── Test A — the loop stays bounded for every profile × delay ────────────────────


@pytest.mark.parametrize("profile_name", list(PROFILES))
@pytest.mark.parametrize("delay_ms", DELAYS_MS)
def test_loop_bounded(profile_name: str, delay_ms: int) -> None:
    """No divergence / NaN: peak error stays small and the tail residual is bounded."""
    profile = PROFILES[profile_name]()
    res = run_latency_loop(profile, delay_ms=delay_ms, cfg=baseline_cfg(), rate_hz=RATE_HZ)

    assert all(math.isfinite(e) for e in res.error), "error went non-finite (diverged)"
    assert peak_abs_error(res.error) < 2.0, "peak error exceeded the divergence bound"
    assert steady_state_error(res.error, tail=20) < 0.5, "tail residual too large"


# ── Test B — LOAD-BEARING latency sensitivity guarantee ──────────────────────────


@pytest.mark.parametrize("profile_name", ["step", "sinusoid"])
def test_error_monotone_in_delay(profile_name: str) -> None:
    """RMS error is monotone non-decreasing in delay AND rms[300] > 1.5 * rms[0].

    With the oscillation guard OFF (so it can't mask the latency signal) more dead
    time must mean more tracking error — the core premise the whole harness exists
    to demonstrate.
    """
    profile = PROFILES[profile_name]()
    rms_by_delay = _rms_error_over_delays(profile, kp=0.6, osc_guard=False)

    # Monotone non-decreasing across [0, 50, 100, 200, 300] ms.
    for lo, hi in zip(rms_by_delay, rms_by_delay[1:], strict=False):
        assert hi >= lo - 1e-9, f"RMS error not monotone in delay: {rms_by_delay}"

    # Sensitivity: 300 ms of dead time must inflate the error by >1.5x.
    assert rms_by_delay[-1] > rms_by_delay[0] * 1.5, (
        f"delay barely changed the error ({rms_by_delay}); harness not sensitive enough"
    )


# ── Test C — dead time induces command oscillation (sign flips) ──────────────────


def test_command_oscillation_grows_with_delay() -> None:
    """At a higher gain (kp=0.8) the step command goes from clean to hunting.

    Few/no flips with no delay; many more with 300 ms of dead time.
    """
    profile = step_profile(N)
    cfg = baseline_cfg(kp=0.8, osc_guard=False)
    flips = [
        sign_flips(run_latency_loop(profile, delay_ms=d, cfg=cfg, rate_hz=RATE_HZ).command)
        for d in DELAYS_MS
    ]

    assert flips[0] <= 1, f"clean (delay=0) step should not hunt: {flips}"
    assert flips[-1] > flips[0], f"delay should add command sign flips: {flips}"


# ── Test D — step overshoot is bounded but grows with delay ──────────────────────


def test_step_overshoot_bounded() -> None:
    """Overshoot stays physical (<1.0) for every delay on the baseline step."""
    profile = step_profile(N)
    cfg = baseline_cfg(kp=0.6, osc_guard=False)
    for d in DELAYS_MS:
        res = run_latency_loop(profile, delay_ms=d, cfg=cfg, rate_hz=RATE_HZ)
        ovr = overshoot(res.aim, 0.6)
        assert ovr < 1.0, f"overshoot unphysical at delay={d}: {ovr}"


def test_step_overshoot_grows_with_delay() -> None:
    """Dead time makes the integrator overshoot the setpoint: 300 ms > 0 ms."""
    profile = step_profile(N)
    cfg = baseline_cfg(kp=0.8, osc_guard=False)
    ov0 = overshoot(run_latency_loop(profile, delay_ms=0, cfg=cfg, rate_hz=RATE_HZ).aim, 0.6)
    ov300 = overshoot(run_latency_loop(profile, delay_ms=300, cfg=cfg, rate_hz=RATE_HZ).aim, 0.6)
    assert ov300 > ov0, f"overshoot did not grow with delay: {ov0} -> {ov300}"
    assert ov300 < 1.0, f"overshoot unphysical at 300 ms: {ov300}"


# ── Harness self-tests ───────────────────────────────────────────────────────────


def test_zero_delay_feeds_true_error() -> None:
    """With delay=0 the controller's SEEN error equals the TRUE composite error."""
    profile = step_profile(N)
    res = run_latency_loop(profile, delay_ms=0, cfg=baseline_cfg(), rate_hz=RATE_HZ)
    # seen[k] is history[k] = target[k] - aim_before_move[k]; the true composite at
    # tick k (before the move) is target[k] - aim[k-1]. Reconstruct and compare.
    for k in range(N):
        aim_before = 0.0 if k == 0 else res.aim[k - 1]
        true_composite = res.target[k] - aim_before
        assert res.seen[k] == pytest.approx(true_composite, abs=1e-12)


def test_nonzero_delay_lags_true_error() -> None:
    """A non-zero delay shifts the seen signal back by exactly ``d`` ticks."""
    profile = sinusoid_profile(N, dt=DT)
    res = run_latency_loop(profile, delay_ms=100, cfg=baseline_cfg(), rate_hz=RATE_HZ)
    d = round((100 / 1000.0) / DT)
    assert d == 2
    # seen[k] == composite history at k-d for k >= d (and clamped to 0 before that).
    # Composite[j] = target[j] - aim_before_move[j]; aim_before_move[j] = aim[j-1].
    for k in range(d, N):
        j = k - d
        aim_before_j = 0.0 if j == 0 else res.aim[j - 1]
        assert res.seen[k] == pytest.approx(res.target[j] - aim_before_j, abs=1e-12)


def test_metric_helpers() -> None:
    """Metric helpers on hand-crafted arrays."""
    assert rms([3.0, 4.0]) == pytest.approx(3.5355339059, abs=1e-9)
    assert rms([]) == 0.0
    assert sign_flips([1.0, -1.0, 1.0]) == 2
    assert sign_flips([0.5, 0.0, 0.4]) == 0  # same sign, zero ignored
    assert sign_flips([1.0, 5e-4, -1.0]) == 1  # |cmd|<1e-3 ignored, still one flip
    assert peak_abs_error([0.1, -0.7, 0.3]) == pytest.approx(0.7)
    assert overshoot([0.0, 0.5, 0.7, 0.6], 0.6) == pytest.approx(0.1)
    assert overshoot([0.0, -0.5, -0.7], -0.6) == pytest.approx(0.1)
    assert overshoot([0.0, 0.3, 0.5], 0.6) == 0.0  # never reached setpoint
    assert steady_state_error([1.0, 1.0, 0.0, 0.0], tail=2) == 0.0


def test_settling_time_helper() -> None:
    """settling_time = time after which |error| stays within band for the rest."""
    # Error leaves the band at i=0,1 then settles from i=2 onward → 2 ticks * dt.
    err = [0.5, 0.2, 0.01, 0.0, 0.0]
    assert settling_time(err, DT, band=0.05) == pytest.approx(2 * DT)
    # Within band the whole time → 0.0.
    assert settling_time([0.01, 0.0, 0.02], DT, band=0.05) == 0.0
    # Never settles (last sample out of band) → full duration.
    assert settling_time([0.0, 0.0, 0.9], DT, band=0.05) == pytest.approx(3 * DT)
