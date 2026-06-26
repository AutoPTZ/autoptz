"""Assertion group 6: detect+track latency with a CI-robust p95 bound."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

import autoptz.engine.pipeline.track as track_mod
from autoptz.engine.pipeline.detect import PersonDetector, make_synthetic_detector_session
from autoptz.engine.pipeline.track import Tracker

# Skip entirely on a runner the operator marks as too slow/contended for timing.
pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOPTZ_SKIP_PERF") == "1",
    reason="AUTOPTZ_SKIP_PERF=1 set",
)


def _slowdown() -> float:
    try:
        return max(1.0, float(os.environ.get("AUTOPTZ_TEST_SLOWDOWN", "1.0")))
    except ValueError:
        return 1.0


def _p95(samples: list[float]) -> float:
    s = sorted(samples)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def test_detect_plus_track_p95_latency_under_bound() -> None:
    pytest.importorskip("onnx")  # synthetic session builder needs onnx
    input_size = 640
    session = make_synthetic_detector_session(
        input_size=input_size,
        detections=[(280.0, 140.0, 360.0, 340.0, 0.9)],
    )
    detector = PersonDetector(_session=session, input_size=input_size, detect_interval=1)

    # Real tracker via the built-in IoU fallback so no boxmot install is required.
    orig = track_mod._BOXMOT_AVAILABLE
    track_mod._BOXMOT_AVAILABLE = False
    try:
        tracker = Tracker(min_hits=1)
        tracker._impl_pending = True
        tracker._impl = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Warm-up (exclude first-call allocation / session warm path).
        for _ in range(5):
            dets = detector.detect(frame)
            tracker.update(dets, frame, fps=30.0)

        samples: list[float] = []
        for _ in range(120):
            t0 = time.perf_counter()
            dets = detector.detect(frame)
            tracker.update(dets, frame, fps=30.0)
            samples.append((time.perf_counter() - t0) * 1000.0)  # ms
    finally:
        track_mod._BOXMOT_AVAILABLE = orig

    p95 = _p95(samples)
    # The synthetic ORT session does no real conv work, but letterbox + ORT call
    # + IoU assoc are real. 25 ms is very generous for a single 640 frame on CPU;
    # scaled up by the CI slowdown factor so a contended runner won't flake.
    bound_ms = 25.0 * _slowdown()
    assert p95 < bound_ms, f"p95 {p95:.2f} ms exceeded bound {bound_ms:.2f} ms"


def test_latency_helpers_are_sane() -> None:
    assert _p95([1.0, 2.0, 3.0, 4.0, 100.0]) == 100.0
    assert _slowdown() >= 1.0
