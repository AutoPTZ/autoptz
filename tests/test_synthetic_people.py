"""Synthetic person silhouettes: visible, moving, detector-friendly."""

from __future__ import annotations

import os

import numpy as np
import pytest

from autoptz.engine.pipeline.ingest import SyntheticAdapter


def _read(addr: str, people=None, w=640, h=480):
    a = SyntheticAdapter(
        "cam-people", address=addr, width=w, height=h, target_fps=30.0, people=people
    )
    a._open()
    frames = [a._read_frame() for _ in range(6)]
    a._close()
    return [f for f in frames if f is not None]


class TestSyntheticPeople:
    def test_people_on_by_default_for_anim(self) -> None:
        frames = _read("anim")
        assert frames and frames[0].shape == (480, 640, 3)
        # People are skin/clothing coloured blobs distinct from the gradient bg:
        # at least one large connected non-background region exists.
        f = frames[0]
        # silhouettes are drawn brighter than the dim torso/leg fill threshold
        bright = f.max(axis=2) > 60
        assert bright.sum() > (0.02 * f.shape[0] * f.shape[1]), "no person-sized foreground"

    def test_people_can_be_disabled(self) -> None:
        # Explicit people=False restores the plain procedural scene (old behaviour).
        frames = _read("anim", people=False)
        assert frames and frames[0].shape == (480, 640, 3)

    def test_people_move_between_frames(self) -> None:
        frames = _read("anim")
        assert len(frames) >= 2
        assert not np.array_equal(frames[0], frames[-1])

    def test_phase_offset_decorrelates_cameras(self) -> None:
        a = SyntheticAdapter("cam-A", address="anim", width=320, height=240, target_fps=30.0)
        b = SyntheticAdapter("cam-B", address="anim", width=320, height=240, target_fps=30.0)
        a._open()
        b._open()
        for _ in range(3):
            fa = a._read_frame()
            fb = b._read_frame()
        a._close()
        b._close()
        assert fa is not None and fb is not None
        assert not np.array_equal(fa, fb), "cameras should not move in lockstep"


def test_motion_is_deterministic_per_camera_and_frame() -> None:
    # Same camera_id replayed from frame 0 yields byte-identical frames.
    a1 = SyntheticAdapter("cam-det1", address="anim", width=320, height=240, target_fps=30.0)
    a2 = SyntheticAdapter("cam-det1", address="anim", width=320, height=240, target_fps=30.0)
    a1._open()
    a2._open()
    fa = [a1._read_frame() for _ in range(5)]
    fb = [a2._read_frame() for _ in range(5)]
    a1._close()
    a2._close()
    for x, y in zip(fa, fb, strict=True):
        assert x is not None and y is not None
        assert np.array_equal(x, y), "same camera_id must be frame-for-frame reproducible"


def test_motion_is_lively_not_constant_velocity() -> None:
    # The old motion was a fixed ±0.32-amplitude sinusoid, so a silhouette never got
    # closer than ~15% of the frame width from the right edge.  Lively motion includes
    # an occasional smooth walk-off/return glide that pushes a person all the way off
    # the right edge.  Over a window covering one exit cycle (210 frames @ 30 fps = 7 s)
    # the rightmost foreground column therefore reaches the very edge of the frame —
    # something the gentle sinusoid can never do.
    w = 640
    a = SyntheticAdapter("cam-live", address="anim", width=w, height=480, target_fps=30.0)
    a._open()
    frames = [a._read_frame() for _ in range(210)]
    a._close()
    frames = [f for f in frames if f is not None]

    max_right = 0
    for f in frames:
        mask = f.max(axis=2) > 60
        cols = np.nonzero(mask.any(axis=0))[0]
        if cols.size:
            max_right = max(max_right, int(cols.max()))
    assert max_right >= w - 4, "a person should glide off the right edge (lively walk-off)"


def _real_detector():
    """Return a PersonDetector backed by a cached real model, or None."""
    from autoptz.engine.pipeline.detect import PersonDetector

    env = os.environ.get("AUTOPTZ_MODEL_PATH")
    if env and os.path.isfile(env):
        return PersonDetector(model_path=env, conf_threshold=0.25)
    try:
        from autoptz.engine.runtime.models import default_manager

        p = default_manager().ensure_detector(tier="auto", download=False)  # cached only
    except Exception:
        return None
    if not p or not os.path.isfile(str(p)):
        return None
    return PersonDetector(model_path=str(p), conf_threshold=0.25)


@pytest.mark.skipif(_real_detector() is None, reason="no cached real detector model")
def test_real_detector_finds_a_silhouette() -> None:
    det = _real_detector()
    assert det is not None
    a = SyntheticAdapter("cam-det", address="anim", width=1280, height=720, target_fps=30.0)
    a._open()
    got = 0
    for _ in range(8):  # detect_interval=1 → every frame
        f = a._read_frame()
        if f is None:
            continue
        if det.detect(f):
            got += 1
    a._close()
    assert got >= 1, "real detector saw no synthetic person across 8 frames"
