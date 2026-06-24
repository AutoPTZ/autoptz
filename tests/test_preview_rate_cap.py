"""The preview ShmWriter push is rate-capped to save per-frame resize CPU."""

from __future__ import annotations

import numpy as np

from autoptz.config.models import CameraConfig, SourceConfig
from autoptz.engine import camera_worker as cw
from autoptz.engine.camera_worker import CameraWorker


class _RecordingShm:
    # _fit_frame reads .height/.width to decide whether to resize; match the
    # preview dims so a preview-sized test frame is passed straight through.
    height = cw._PREVIEW_H
    width = cw._PREVIEW_W

    def __init__(self) -> None:
        self.pushes = 0

    def push(self, frame: object) -> None:
        self.pushes += 1


def _worker() -> CameraWorker:
    config = CameraConfig(
        id="prevcap01abcd",
        name="Cam",
        source=SourceConfig(type="usb", address="usb://0"),
    )
    return CameraWorker("prevcap01abcd", config, lambda m: None)


def test_preview_push_is_rate_capped(monkeypatch) -> None:
    w = _worker()
    w._shm = _RecordingShm()  # type: ignore[assignment]
    clock = {"t": 1000.0}
    monkeypatch.setattr(cw.time, "monotonic", lambda: clock["t"])
    frame = np.zeros((cw._PREVIEW_H, cw._PREVIEW_W, 3), dtype=np.uint8)

    w._push_frame(frame)  # first push always lands
    assert w._shm.pushes == 1

    clock["t"] += 0.01  # < one preview period later → skipped
    w._push_frame(frame)
    assert w._shm.pushes == 1

    clock["t"] += cw._PREVIEW_PUSH_MIN_PERIOD_S  # a full period later → pushes again
    w._push_frame(frame)
    assert w._shm.pushes == 2


def test_preview_first_frame_always_pushes(monkeypatch) -> None:
    # With _last_preview_push_t seeded at 0.0, the very first frame must push
    # immediately regardless of the absolute clock value.
    w = _worker()
    w._shm = _RecordingShm()  # type: ignore[assignment]
    monkeypatch.setattr(cw.time, "monotonic", lambda: 9_999.0)
    w._push_frame(np.zeros((cw._PREVIEW_H, cw._PREVIEW_W, 3), dtype=np.uint8))
    assert w._shm.pushes == 1
