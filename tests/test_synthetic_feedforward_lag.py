"""Assertion group 5: velocity feed-forward reduces follow lag."""

from __future__ import annotations

from autoptz.engine.ptz.controller import PTZController
from tests.synthetic_tracking import MockBackend, make_cfg

RATE_HZ = 20.0
DT = 1.0 / RATE_HZ


def _accumulate_command(kv: float, *, vx: float, error_x: float, ticks: int = 30) -> float:
    """Sum the pan command over a constant velocity+error drive.

    A larger positive sum == the camera leans harder in the motion direction ==
    less lag behind a constantly-moving subject.
    """
    cfg = make_cfg(
        kp=0.4,
        kd=0.0,
        kv=kv,
        aim_smoothing=0.0,
        safe_zone_enabled=False,
        osc_guard=False,
        max_accel=0.0,  # isolate FF from the slew ramp
    )
    ctrl = PTZController(MockBackend(), cfg, rate_hz=RATE_HZ)
    total = 0.0
    t = 0.0
    for _ in range(ticks):
        pan, _tilt, _zoom = ctrl.step((error_x, 0.0), (vx, 0.0), 0.45, True, t=t)
        total += pan
        t += DT
    return total


def test_feedforward_increases_lead_command() -> None:
    """With the same residual error + subject velocity, kv>0 commands more lead."""
    no_ff = _accumulate_command(kv=0.0, vx=0.6, error_x=0.2)
    ff = _accumulate_command(kv=0.4, vx=0.6, error_x=0.2)
    assert ff > no_ff


def test_feedforward_single_tick_matches_existing_contract() -> None:
    """Sanity tie-back to the existing per-tick FF behaviour (no regression)."""
    common = {
        "kp": 0.4,
        "kd": 0.0,
        "aim_smoothing": 0.0,
        "safe_zone_enabled": False,
        "osc_guard": False,
        "max_accel": 0.0,
    }
    ctrl_noff = PTZController(MockBackend(), make_cfg(kv=0.0, **common))
    ctrl_ff = PTZController(MockBackend(), make_cfg(kv=0.3, **common))
    pan_noff, _, _ = ctrl_noff.step((0.2, 0.0), (0.5, 0.0), 0.45, True, t=0.0)
    pan_ff, _, _ = ctrl_ff.step((0.2, 0.0), (0.5, 0.0), 0.45, True, t=0.0)
    assert pan_ff > pan_noff
