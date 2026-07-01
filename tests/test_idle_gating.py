"""Idle-headroom gating: the inference loop must not do follow-only work when
nothing is being tracked.

Two per-frame costs used to run unconditionally, every frame, even with all AI
services off:

* ``_update_ego_motion`` — sparse optical flow whose *only* consumer is the
  PTZ aim-velocity feed-forward (runs only while actively following).  It burned
  ~15% of a core computing a value nothing read when tracking was off.
* the async appearance hand-off — woke the appearance thread every frame to run
  ReID + face passes that both immediately no-op with face + ReID off.

These tests pin the gates: ego-motion runs iff ``tracking`` is on, and the
appearance hand-off fires iff ``face_recognition`` or ``reid`` is on.  Models are
never built (``_build_inference_stacks`` is stubbed) so the loop body runs with a
cheap synthetic frame source.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from autoptz.config.models import CameraConfig, SourceConfig
from autoptz.engine.camera_worker import CameraWorker

_ALL_OFF = {
    "detection": False,
    "tracking": False,
    "face_recognition": False,
    "pose": False,
    "reid": False,
}


class _PacedSource:
    """A trivial frame source paced ~120 fps so the loop gets fresh frame ids."""

    def __init__(self) -> None:
        self._frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def open(self) -> bool:
        return True

    def read(self) -> np.ndarray:
        time.sleep(1 / 120)
        return self._frame

    def close(self) -> None:
        pass


def _worker() -> CameraWorker:
    cfg = CameraConfig(
        id="cam-idle-1234abcd",
        name="Idle",
        source=SourceConfig(type="usb", address="usb://0"),
    )
    w = CameraWorker(
        "cam-idle-1234abcd", cfg, on_telemetry=lambda _m: None, frame_source=_PacedSource()
    )
    # Never build real models — we only care about the per-frame gating.
    w._build_inference_stacks = lambda: None  # type: ignore[method-assign]
    return w


def _run_briefly(w: CameraWorker, seconds: float = 0.6) -> None:
    w.start()
    try:
        time.sleep(seconds)
    finally:
        w.stop()


def test_egomotion_skipped_when_tracking_off() -> None:
    w = _worker()
    calls: list[int] = []
    w._update_ego_motion = lambda *a, **k: calls.append(1)  # type: ignore[method-assign]
    w.set_features(dict(_ALL_OFF))
    _run_briefly(w)
    assert calls == [], f"ego-motion ran {len(calls)}x with tracking off — idle headroom regressed"


def test_egomotion_runs_when_tracking_on() -> None:
    w = _worker()
    ran = threading.Event()
    w._update_ego_motion = lambda *a, **k: ran.set()  # type: ignore[method-assign]
    feats = dict(_ALL_OFF)
    feats["tracking"] = True
    w.set_features(feats)
    w.start()
    try:
        assert ran.wait(2.0), (
            "ego-motion never ran with tracking on — feed-forward would lose ego compensation"
        )
    finally:
        w.stop()


def test_appearance_not_published_when_face_and_reid_off() -> None:
    w = _worker()
    pubs: list[int] = []
    w._publish_appearance_input = lambda *a, **k: pubs.append(1)  # type: ignore[method-assign]
    w.set_features(dict(_ALL_OFF))
    _run_briefly(w)
    assert pubs == [], (
        f"appearance thread woken {len(pubs)}x with face+ReID off — idle headroom regressed"
    )


def test_appearance_published_when_face_on() -> None:
    w = _worker()
    pub = threading.Event()
    w._publish_appearance_input = lambda *a, **k: pub.set()  # type: ignore[method-assign]
    feats = dict(_ALL_OFF)
    feats["face_recognition"] = True
    w.set_features(feats)
    w.start()
    try:
        assert pub.wait(2.0), (
            "appearance input never published with face on — identity would never run"
        )
    finally:
        w.stop()
