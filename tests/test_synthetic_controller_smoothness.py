"""Assertion group 4: controller command smoothness + preset monotonicity."""

from __future__ import annotations

import math

from autoptz.config.models import (
    SPEED_PROFILES,
    PTZConfig,
    TrackingSpeed,
    apply_speed_profile,
)
from autoptz.engine.ptz.controller import PTZController
from tests.synthetic_tracking import MockBackend, make_cfg

RATE_HZ = 20.0
DT = 1.0 / RATE_HZ


def _drive_constant(cfg: PTZConfig, error_x: float, ticks: int = 40) -> list[float]:
    """Drive a constant step error; return the per-tick pan command sequence."""
    ctrl = PTZController(MockBackend(), cfg, rate_hz=RATE_HZ)
    pans: list[float] = []
    t = 0.0
    for _ in range(ticks):
        pan, _tilt, _zoom = ctrl.step((error_x, 0.0), (0.0, 0.0), 0.45, True, t=t)
        pans.append(pan)
        t += DT
    return pans


def test_slew_bounds_command_acceleration() -> None:
    """With max_accel>0 the per-tick INCREASE in |pan| <= max_accel*dt (+eps).

    Deceleration toward zero is intentionally unbounded (the _slew contract), so
    only magnitude increases are checked.
    """
    max_accel = 2.0
    cfg = make_cfg(
        kp=1.0,
        aim_smoothing=0.0,
        safe_zone_enabled=False,
        osc_guard=False,
        max_accel=max_accel,
    )
    pans = _drive_constant(cfg, error_x=0.6, ticks=40)
    bound = max_accel * DT + 1e-6
    prev = 0.0
    for cur in pans:
        if abs(cur) > abs(prev):  # increasing magnitude → accel-limited
            assert abs(cur) - abs(prev) <= bound
        prev = cur


def test_no_sign_flip_jitter_at_steady_state() -> None:
    """A clean constant error must settle to a single-sign command (no hunting)."""
    cfg = make_cfg(
        kp=0.8,
        kd=0.0,
        aim_smoothing=0.0,
        safe_zone_enabled=False,
        max_accel=3.0,
    )
    pans = _drive_constant(cfg, error_x=0.5, ticks=40)
    tail = pans[-10:]  # settled region
    # No alternation: every consecutive pair shares sign (or is ~zero).
    for a, b in zip(tail, tail[1:], strict=False):
        if abs(a) > 1e-3 and abs(b) > 1e-3:
            assert math.copysign(1.0, a) == math.copysign(1.0, b)


def _settled_magnitude(speed: TrackingSpeed, error_x: float = 0.15) -> float:
    # error_x=0.15 keeps the slower presets below the response-curve clamp so the
    # CALM→SPORT ordering separates cleanly instead of all saturating at 1.0.
    base = PTZConfig(safe_zone_enabled=False, deadzone_x=0.0, deadzone_y=0.0)
    cfg = apply_speed_profile(base, speed)
    pans = _drive_constant(cfg, error_x=error_x, ticks=60)
    return abs(pans[-1])


def test_command_magnitude_monotone_across_presets() -> None:
    """Settled |pan| is non-decreasing CALM <= NORMAL <= FAST <= SPORT."""
    order = [
        TrackingSpeed.CALM,
        TrackingSpeed.NORMAL,
        TrackingSpeed.FAST,
        TrackingSpeed.SPORT,
    ]
    mags = [_settled_magnitude(s) for s in order]
    for lo, hi in zip(mags, mags[1:], strict=False):
        assert hi >= lo - 1e-6  # non-decreasing speed feel


def test_speed_profiles_are_monotone_source_of_truth() -> None:
    """Guard: the SPEED_PROFILES this suite relies on are themselves monotone."""
    order = [
        TrackingSpeed.CALM,
        TrackingSpeed.NORMAL,
        TrackingSpeed.FAST,
        TrackingSpeed.SPORT,
    ]
    pans = [SPEED_PROFILES[s]["max_pan_speed"] for s in order]
    kps = [SPEED_PROFILES[s]["kp"] for s in order]
    assert pans == sorted(pans)
    assert kps == sorted(kps)
