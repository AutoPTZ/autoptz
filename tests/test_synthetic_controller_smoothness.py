"""Assertion group 4: controller command smoothness."""

from __future__ import annotations

import math

from autoptz.config.models import PTZConfig
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
