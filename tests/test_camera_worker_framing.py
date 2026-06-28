"""CameraWorker Center-Stage crop-rect telemetry wiring.

Verifies _framed_output records the active digital crop into
_last_digital_crop_rect (and clears it when Center Stage is inactive), and that
the rect is carried on the emitted TelemetryMsg.
"""

from __future__ import annotations

import numpy as np


def _bare_worker():
    """A CameraWorker instance with only the attributes _framed_output touches,
    built without running the capture thread (we never call .start())."""
    from autoptz.config.models import CameraConfig
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(id="cam-fr", name="Cam FR")
    return CameraWorker("cam-fr", cfg, on_telemetry=lambda m: None)


def test_framed_output_records_none_without_digital_backend() -> None:
    w = _bare_worker()
    w._ptz_backend = None  # no Center Stage
    w._last_digital_crop_rect = (1, 2, 3, 4)  # stale value must be cleared
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = w._framed_output(frame)
    assert out is frame  # passthrough unchanged
    assert w._last_digital_crop_rect is None


def test_framed_output_records_crop_rect_when_center_stage_active() -> None:
    from autoptz.engine.ptz.digital import DigitalPTZBackend

    w = _bare_worker()
    w._ptz_backend = DigitalPTZBackend()  # digital crop active
    # No target locked → framer eases toward the full frame on the first tick,
    # but _framed_output STILL applies (crops/scales) and records a rect.
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    w._framed_output(frame)
    rect = w._last_digital_crop_rect
    assert rect is not None
    x, y, cw, ch = rect
    assert 0 <= x and 0 <= y
    assert 0 < cw <= 1920 and 0 < ch <= 1080


def test_telemetry_carries_last_digital_crop_rect() -> None:
    from autoptz.engine.runtime.messages import HealthState, TelemetryMsg

    w = _bare_worker()
    w._last_digital_crop_rect = (100, 50, 600, 400)
    # The emit helper builds a TelemetryMsg; assert the field is wired through.
    captured: list[TelemetryMsg] = []
    w._on_telemetry = captured.append
    w._emit_telemetry(tracks=[], health=HealthState.OK, last_error=None)
    assert captured and captured[0].digital_crop_rect == (100, 50, 600, 400)
