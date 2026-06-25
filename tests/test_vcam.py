from __future__ import annotations

import numpy as np

from autoptz.engine.pipeline.vcam import VirtualCamSink, pick_interpolation


def test_sink_is_noop_without_pyvirtualcam(monkeypatch):
    import autoptz.engine.pipeline.vcam as v

    monkeypatch.setattr(v, "_probe_pyvirtualcam", lambda: False)
    sink = VirtualCamSink(640, 480)
    assert sink.available is False
    sink.send_bgr(np.zeros((480, 640, 3), dtype=np.uint8))  # must not raise
    sink.close()


class TestPickInterpolation:
    """B5 — choose a quality-appropriate interpolation by scale factor."""

    def test_upscale_uses_cubic(self):
        import cv2

        # Source smaller than destination → upscaling → high-quality cubic.
        assert pick_interpolation((640, 360), (1280, 720)) == cv2.INTER_CUBIC

    def test_downscale_uses_area(self):
        import cv2

        # Source larger than destination → downscaling → area (best for shrink).
        assert pick_interpolation((1920, 1080), (1280, 720)) == cv2.INTER_AREA

    def test_one_to_one_uses_linear(self):
        import cv2

        assert pick_interpolation((1280, 720), (1280, 720)) == cv2.INTER_LINEAR

    def test_mixed_scale_prefers_upscale_quality(self):
        import cv2

        # Width grows, height shrinks: treat as an upscale (favour sharpness).
        assert pick_interpolation((1000, 800), (1280, 720)) == cv2.INTER_CUBIC


class TestVcamRateGate:
    """B6 — the vcam send runs on its own ~30fps gate, independent of the
    ~20fps preview gate. The decision is a pure elapsed-vs-period check."""

    def test_due_when_period_elapsed(self):
        from autoptz.engine.camera_worker import _VCAM_PUSH_MIN_PERIOD_S, _push_due

        last = 100.0
        # Just under one period → not due.
        assert (
            _push_due(last + _VCAM_PUSH_MIN_PERIOD_S * 0.9, last, _VCAM_PUSH_MIN_PERIOD_S) is False
        )
        # At/after a full period → due.
        assert (
            _push_due(last + _VCAM_PUSH_MIN_PERIOD_S * 1.001, last, _VCAM_PUSH_MIN_PERIOD_S) is True
        )

    def test_vcam_faster_than_preview(self):
        from autoptz.engine.camera_worker import (
            _PREVIEW_PUSH_MIN_PERIOD_S,
            _VCAM_PUSH_MIN_PERIOD_S,
        )

        # The vcam must run at a higher rate (shorter period) than the preview.
        assert _VCAM_PUSH_MIN_PERIOD_S < _PREVIEW_PUSH_MIN_PERIOD_S

    def test_vcam_due_independently_of_preview(self):
        from autoptz.engine.camera_worker import (
            _PREVIEW_PUSH_MIN_PERIOD_S,
            _VCAM_PUSH_MIN_PERIOD_S,
            _push_due,
        )

        # At a moment one vcam-period after the last vcam push but BEFORE a full
        # preview period, the vcam is due while the preview is not — proving the
        # vcam is no longer throttled to the preview rate.
        t0 = 0.0
        now = t0 + _VCAM_PUSH_MIN_PERIOD_S
        assert now < t0 + _PREVIEW_PUSH_MIN_PERIOD_S  # within one preview period
        assert _push_due(now, t0, _VCAM_PUSH_MIN_PERIOD_S) is True
        assert _push_due(now, t0, _PREVIEW_PUSH_MIN_PERIOD_S) is False
