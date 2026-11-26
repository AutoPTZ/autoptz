"""Head-anchored tilt: the PTZ tilt error must not jump when the aim blend does.

Field data (live NDI PTZ, 2026-11-26, camera static, subject standing): the
fused torso/box aim that drives tilt jumped >0.1 half-frame in 10% of samples
(p99 279 px) while the box top (head) moved at most 5.5 px. 12 of 26 camera
moves in that session started on such a jump.
"""

from __future__ import annotations

import numpy as np

from autoptz.config.models import CameraConfig, SourceConfig
from autoptz.engine.camera_worker import CameraWorker
from autoptz.engine.runtime.messages import BBox, TrackInfo

H, W = 1080, 1920
FRAME = np.zeros((H, W, 3), dtype=np.uint8)


def _worker(**ptz) -> CameraWorker:
    cfg = CameraConfig(id="tilt-1", name="Tilt", source=SourceConfig(type="ndi", address="ndi://X"))
    if ptz:
        cfg = cfg.model_copy(update={"ptz": cfg.ptz.model_copy(update=ptz)})
    w = CameraWorker("tilt-1", cfg, lambda _m: None)
    w._torso_stable_box = lambda _tid: None  # no pose cached → box-top reference
    return w


def _track(y1: float, y2: float, tid: int = 5) -> TrackInfo:
    return TrackInfo(track_id=tid, bbox=BBox(x1=800, y1=y1, x2=1100, y2=y2), is_target=True)


def test_fused_aim_flicker_with_still_head_barely_moves_tilt_error() -> None:
    w = _worker()
    outs = []
    t = 100.0
    for i in range(60):  # 2 s at 30 Hz
        # Box bottom flickers (occlusion) and the fused aim jumps ±0.25 with it…
        y2 = 1000.0 if i % 2 else 700.0
        raw_ey = 0.10 if i % 2 else 0.35
        (_ex, ey) = w._head_anchored_tilt(_track(300.0, y2), (0.0, raw_ey), FRAME, t)
        outs.append(ey)
        t += 1 / 30
    # Raw signal jumps 0.25 every tick; the tilt error may only drift slowly
    # toward the average framing — no tick-to-tick jumps.
    steps = [abs(b - a) for a, b in zip(outs, outs[1:], strict=False)]
    assert max(steps) < 0.01


def test_real_head_motion_passes_through_immediately() -> None:
    w = _worker()
    t = 100.0
    for _ in range(30):
        w._head_anchored_tilt(_track(300.0, 1000.0), (0.0, 0.2), FRAME, t)
        t += 1 / 30
    before = w._head_anchored_tilt(_track(300.0, 1000.0), (0.0, 0.2), FRAME, t)[1]
    # Subject sits down: the whole box drops 162 px (0.3 half-frame) in one tick.
    after = w._head_anchored_tilt(_track(462.0, 1162.0), (0.0, -0.1), FRAME, t + 1 / 30)[1]
    assert after - before < -0.27


def test_slow_framing_offset_is_still_honoured() -> None:
    """The fused aim's *average* placement (framing) is kept, just low-passed."""
    w = _worker()
    t = 100.0
    out = 0.0
    for _ in range(300):  # 10 s
        out = w._head_anchored_tilt(_track(300.0, 1000.0), (0.0, 0.25), FRAME, t)[1]
        t += 1 / 30
    assert abs(out - 0.25) < 0.01


def test_clipped_head_tilts_gently_up_regardless_of_aim_noise() -> None:
    """Live (392-408 s): head out of frame for 15 s, the fused aim jumped
    -0.7 ↔ +0.6 (even off-frame, -1.5) and tilt went up and down. With the head
    clipped, the only sensible move is a gentle, steady tilt UP to reveal it."""
    w = _worker()
    outs = []
    t = 100.0
    for i in range(30):
        raw = (0.0, 0.6 if i % 2 else -0.7)
        outs.append(w._head_anchored_tilt(_track(0.0, 600.0 + 10 * (i % 3)), raw, FRAME, t)[1])
        t += 1 / 30
    assert all(o > 0.2 for o in outs)  # always up, enough to leave the hold zone
    assert max(outs) - min(outs) < 1e-9  # and perfectly steady


def test_subject_filling_the_frame_holds_tilt() -> None:
    w = _worker()
    assert w._head_anchored_tilt(_track(0.0, 1075.0), (0.1, -0.7), FRAME, 100.0) == (0.1, 0.0)


def test_unclipping_resumes_head_anchoring_without_a_jump() -> None:
    w = _worker()
    w._head_anchored_tilt(_track(0.0, 600.0), (0.0, -0.7), FRAME, 100.0)
    first = w._head_anchored_tilt(_track(40.0, 640.0), (0.0, 0.3), FRAME, 100.0 + 1 / 30)
    # fresh start on the raw error once the head is visible
    assert first[0] == 0.0 and abs(first[1] - 0.3) < 1e-9


def test_new_target_track_starts_from_its_own_raw_error() -> None:
    w = _worker()
    t = 100.0
    for _ in range(30):
        w._head_anchored_tilt(_track(300.0, 1000.0, tid=5), (0.0, 0.2), FRAME, t)
        t += 1 / 30
    first = w._head_anchored_tilt(_track(500.0, 900.0, tid=9), (0.0, -0.3), FRAME, t)
    assert first == (0.0, -0.3)


def test_can_be_disabled() -> None:
    w = _worker(tilt_head_anchor=False)
    assert w._head_anchored_tilt(_track(300.0, 700.0), (0.1, 0.35), FRAME, 100.0) == (0.1, 0.35)
