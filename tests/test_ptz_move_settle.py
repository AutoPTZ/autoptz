"""Move/settle policy: move with purpose, never twitch or reverse mid-move.

Field data (live NDI PTZ, 2026-11-26, 3.7 min armed): 32 moves, 9 shorter than
0.6 s and 6 started on <0.15 error — a single aim blip outside the zone released
the hold and re-centred the camera on it. Moves also reversed mid-flight when the
error crossed over (overshoot / blip) instead of settling.
"""

from __future__ import annotations

from autoptz.config.models import PTZConfig
from autoptz.engine.ptz.base import PTZBackend, PTZCaps
from autoptz.engine.ptz.controller import PTZController

DT = 1.0 / 30.0


class _Backend(PTZBackend):
    def __init__(self) -> None:
        super().__init__()
        self.caps = PTZCaps(continuous_pan_tilt=True, continuous_zoom=True)

    def move_velocity(self, pan, tilt, zoom=0.0):
        pass

    def stop(self):
        pass

    def goto_preset(self, idx):
        pass

    def save_preset(self, idx):
        pass

    def close(self):
        pass


def _ctrl(**kw) -> PTZController:
    return PTZController(_Backend(), PTZConfig(**kw), rate_hz=30.0)


def _run(ctrl: PTZController, errors, t0: float = 100.0) -> list[tuple[float, float]]:
    out = []
    t = t0
    for ex, ey in errors:
        pan, tilt, _ = ctrl.step((ex, ey), (0.0, 0.0), 0.4, track_active=True, t=t)
        out.append((pan, tilt))
        t += DT
    return out


def _settle(ctrl: PTZController) -> float:
    """Park the subject at centre so the hold latch is engaged; return next t."""
    _run(ctrl, [(0.0, 0.0)] * 15)
    return 100.0 + 15 * DT


def test_single_frame_blip_outside_zone_does_not_move_camera() -> None:
    ctrl = _ctrl()
    t = _settle(ctrl)
    # One sample well outside the zone (an aim jump), then back to centre.
    cmds = _run(ctrl, [(0.30, 0.0)] + [(0.0, 0.0)] * 20, t0=t)
    assert all(abs(p) < 1e-6 and abs(q) < 1e-6 for p, q in cmds)


def test_short_excursion_below_confirm_time_does_not_move() -> None:
    ctrl = _ctrl(move_confirm_ms=300)
    t = _settle(ctrl)
    blip = [(0.30, 0.0)] * 6  # 0.2 s < 0.3 s
    cmds = _run(ctrl, blip + [(0.0, 0.0)] * 20, t0=t)
    assert all(abs(p) < 1e-6 for p, _ in cmds)


def test_sustained_exit_moves_after_confirm_time() -> None:
    ctrl = _ctrl(move_confirm_ms=300)
    t = _settle(ctrl)
    cmds = _run(ctrl, [(0.30, 0.0)] * 30, t0=t)
    first_move = next(i for i, (p, _) in enumerate(cmds) if abs(p) > 1e-3)
    assert 0.25 <= first_move * DT <= 0.45
    assert cmds[-1][0] > 0.0


def test_large_offset_moves_immediately() -> None:
    ctrl = _ctrl(move_confirm_ms=300)
    t = _settle(ctrl)
    cmds = _run(ctrl, [(0.70, 0.0)] * 3, t0=t)
    assert abs(cmds[0][0]) > 1e-3 or abs(cmds[1][0]) > 1e-3


def test_move_settles_instead_of_reversing_on_small_crossover() -> None:
    ctrl = _ctrl(move_confirm_ms=0)
    t = _settle(ctrl)
    moving = _run(ctrl, [(0.40, 0.0)] * 20, t0=t)
    assert moving[-1][0] > 0.0
    # Error crosses to a small opposite offset (overshoot / blip): settle, don't reverse.
    after = _run(ctrl, [(-0.10, 0.0)] * 20, t0=t + 20 * DT)
    assert all(p >= -1e-6 for p, _ in after)


def test_genuine_reversal_beyond_zone_is_followed() -> None:
    ctrl = _ctrl(move_confirm_ms=0)
    t = _settle(ctrl)
    _run(ctrl, [(0.40, 0.0)] * 20, t0=t)
    after = _run(ctrl, [(-0.45, 0.0)] * 30, t0=t + 20 * DT)
    assert after[-1][0] < 0.0


def test_tilt_axis_gets_the_same_treatment() -> None:
    ctrl = _ctrl()
    t = _settle(ctrl)
    cmds = _run(ctrl, [(0.0, 0.40)] + [(0.0, 0.0)] * 20, t0=t)
    assert all(abs(q) < 1e-6 for _, q in cmds)


# ── tilt: only sustained vertical shifts ─────────────────────────────────────
# Live (2026-11-26, 82-94 s): the box bottom stayed put while the top swung
# 330→50→330→50→410 px on a ~3 s cycle (leaning over the keyboard / hands up);
# the camera chased it with small, late tilts — "moving up a bunch".


def _bob(n: int, amp: float = 0.27, period_s: float = 3.0, ex: float = 0.0):
    import math

    return [(ex, amp * math.sin(2 * math.pi * i * DT / period_s)) for i in range(n)]


def test_vertical_bobbing_does_not_tilt() -> None:
    ctrl = _ctrl(tilt_settle_ms=1500)
    t = _settle(ctrl)
    cmds = _run(ctrl, _bob(300), t0=t)  # 10 s of bobbing
    assert max(abs(q) for _, q in cmds) < 1e-6


def test_vertical_bobbing_during_a_pan_move_does_not_tilt() -> None:
    ctrl = _ctrl(tilt_settle_ms=1500, move_confirm_ms=0)
    t = _settle(ctrl)
    cmds = _run(ctrl, _bob(90, ex=0.45), t0=t)
    assert any(p > 0.05 for p, _ in cmds)  # pan follows the real horizontal offset
    assert max(abs(q) for _, q in cmds) < 1e-6


def test_sustained_vertical_shift_still_tilts() -> None:
    ctrl = _ctrl(tilt_settle_ms=1500, move_confirm_ms=300)
    t = _settle(ctrl)
    cmds = _run(ctrl, [(0.0, 0.45)] * 75, t0=t)  # sits/stands and stays: 2.5 s
    first = next(i for i, (_, q) in enumerate(cmds) if abs(q) > 1e-3)
    assert first * DT <= 1.0
    assert cmds[-1][1] > 0.0


def test_subject_near_frame_edge_tilts_immediately() -> None:
    ctrl = _ctrl(tilt_settle_ms=1500)
    t = _settle(ctrl)
    cmds = _run(ctrl, [(0.0, 0.95)] * 3, t0=t)
    assert any(abs(q) > 1e-3 for _, q in cmds)
