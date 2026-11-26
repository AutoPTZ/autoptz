"""Reject physically impossible target-aim jumps before they drive the PTZ.

Live (2026-11-26, 110.0→110.15 s): track 28's aim jumped +0.52 → -0.72
half-frames in one tick (the tracker re-assigned the id to another body while
the camera moved); the controller slammed pan to -0.55 chasing it, then lost
the subject. 4-6 such jumps per session, each a dramatic reversal.
"""

from __future__ import annotations

from autoptz.config.models import CameraConfig, SourceConfig
from autoptz.engine.camera_worker import CameraWorker
from autoptz.engine.runtime.messages import BBox, TrackInfo

DT = 1 / 30


def _worker() -> CameraWorker:
    cfg = CameraConfig(id="jump-1", name="J", source=SourceConfig(type="ndi", address="ndi://X"))
    w = CameraWorker("jump-1", cfg, lambda _m: None)
    w._ego_vel = (0.0, 0.0)
    return w


def _t(tid: int = 28) -> TrackInfo:
    return TrackInfo(track_id=tid, bbox=BBox(x1=0, y1=0, x2=10, y2=10), is_target=True)


def _feed(w, errs, t0=100.0, tid=28):
    out, t = [], t0
    for e in errs:
        out.append(w._aim_jump_ok(_t(tid), e, t))
        t += DT
    return out


def test_one_tick_teleport_is_held() -> None:
    w = _worker()
    res = _feed(w, [(0.50, 0.0)] * 5 + [(-0.72, 0.0)])
    assert res[:5] == [True] * 5
    assert res[5] is False


def test_new_position_confirmed_over_a_few_frames_is_accepted() -> None:
    w = _worker()
    res = _feed(w, [(0.50, 0.0)] * 5 + [(-0.72, 0.0)] * 4)
    assert res[5] is False
    assert res[-1] is True


def test_ordinary_motion_is_never_held() -> None:
    w = _worker()
    errs = [(0.02 * i, -0.01 * i) for i in range(40)]  # brisk but human
    assert all(_feed(w, errs))


def test_jump_explained_by_camera_motion_is_accepted() -> None:
    w = _worker()
    _feed(w, [(0.5, 0.0)] * 5)
    w._ego_vel = (-18.0, 0.0)  # camera swing shifts content -0.6 per tick
    assert w._aim_jump_ok(_t(), (-0.1, 0.0), 100.0 + 5 * DT) is True


def test_target_change_resets_the_guard() -> None:
    w = _worker()
    _feed(w, [(0.5, 0.0)] * 5, tid=28)
    assert w._aim_jump_ok(_t(31), (-0.7, 0.0), 100.0 + 5 * DT) is True
