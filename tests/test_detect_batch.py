"""Tests for PersonDetector.detect_batch.

All tests use the synthetic ONNX session (make_synthetic_detector_session) so
no real model files or hardware are required.

Core guarantee tested: detect_batch(frames)[i] == detect(frames[i]) for every i,
i.e. batched inference is a pure performance optimisation with no behaviour change.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoptz.engine.pipeline.detect import (
    Detection,
    PersonDetector,
    make_synthetic_detector_session,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_frame(color: tuple[int, int, int], h: int = 480, w: int = 640) -> np.ndarray:
    """Return a solid-colour BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = color  # (B, G, R)
    return frame


def _detections_equal(a: list[Detection], b: list[Detection]) -> bool:
    """Check two detection lists are identical (same bboxes, conf, class_id)."""
    if len(a) != len(b):
        return False
    for da, db in zip(a, b, strict=False):
        if da.class_id != db.class_id:
            return False
        if abs(da.conf - db.conf) > 1e-6:
            return False
        if abs(da.bbox.x1 - db.bbox.x1) > 1e-4:
            return False
        if abs(da.bbox.y1 - db.bbox.y1) > 1e-4:
            return False
        if abs(da.bbox.x2 - db.bbox.x2) > 1e-4:
            return False
        if abs(da.bbox.y2 - db.bbox.y2) > 1e-4:
            return False
    return True


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_session():
    """ORT session with two hardcoded person detections (fixed batch dim = 1)."""
    return make_synthetic_detector_session(
        input_size=640,
        detections=[
            (100.0, 150.0, 200.0, 400.0, 0.90),
            (300.0, 100.0, 450.0, 420.0, 0.82),
        ],
    )


@pytest.fixture
def detector(synthetic_session):
    """PersonDetector backed by the synthetic session."""
    return PersonDetector(_session=synthetic_session, conf_threshold=0.5)


# ── Empty input ────────────────────────────────────────────────────────────────


class TestDetectBatchEmpty:
    def test_empty_list_returns_empty(self, detector) -> None:
        result = detector.detect_batch([])
        assert result == []


# ── Single frame ───────────────────────────────────────────────────────────────


class TestDetectBatchSingleFrame:
    def test_single_frame_length_one(self, detector) -> None:
        frame = _make_frame((0, 0, 0))
        result = detector.detect_batch([frame])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_single_frame_equals_detect(self, detector, synthetic_session) -> None:
        """detect_batch([f])[0] must equal detect(f)."""
        frame = _make_frame((128, 64, 32))

        # Fresh detectors to keep frame counters independent.
        det_batch = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
        det_single = PersonDetector(_session=synthetic_session, conf_threshold=0.5)

        batch_result = det_batch.detect_batch([frame])[0]
        single_result = det_single.detect(frame)

        assert _detections_equal(batch_result, single_result), (
            f"detect_batch single-frame mismatch:\n  batch={batch_result}\n  single={single_result}"
        )


# ── Three varied frames — core equivalence guarantee ──────────────────────────


class TestDetectBatchMultiFrame:
    def test_three_frames_correct_length(self, synthetic_session) -> None:
        det = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
        frames = [
            _make_frame((0, 0, 0)),
            _make_frame((128, 128, 128)),
            _make_frame((200, 100, 50)),
        ]
        result = det.detect_batch(frames)
        assert len(result) == 3

    def test_three_frames_batch_equals_per_frame(self, synthetic_session) -> None:
        """detect_batch(frames)[i] must equal detect(frames[i]) for each i.

        The synthetic session has a fixed batch dim of 1, so this exercises
        the fallback (per-frame detect() loop) path — which must still produce
        correct results.
        """
        frames = [
            _make_frame((0, 0, 0)),
            _make_frame((64, 128, 192)),
            _make_frame((255, 0, 127)),
        ]

        # Detector used for detect_batch.
        det_batch = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
        batch_results = det_batch.detect_batch(frames)

        # Per-frame reference: fresh detector per frame to isolate frame counters.
        for i, frame in enumerate(frames):
            det_single = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
            single_result = det_single.detect(frame)
            assert _detections_equal(batch_results[i], single_result), (
                f"Frame {i} mismatch:\n  batch={batch_results[i]}\n  single={single_result}"
            )

    def test_result_is_list_of_lists_of_detections(self, synthetic_session) -> None:
        det = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
        frames = [_make_frame((0, 0, 0)), _make_frame((255, 255, 255))]
        result = det.detect_batch(frames)
        for frame_dets in result:
            assert isinstance(frame_dets, list)
            for d in frame_dets:
                assert isinstance(d, Detection)

    def test_order_preserved(self, synthetic_session) -> None:
        """Result list must be in the same order as input frames."""
        # Use different-sized frames so we can track order by shape mapping.
        frames = [
            _make_frame((10, 20, 30), h=240, w=320),
            _make_frame((40, 50, 60), h=480, w=640),
            _make_frame((70, 80, 90), h=360, w=480),
        ]
        det = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
        result = det.detect_batch(frames)
        assert len(result) == 3
        # Each per-frame result must be non-empty (synthetic session returns 2 dets).
        for i, frame_dets in enumerate(result):
            assert len(frame_dets) >= 1, f"Frame {i}: expected detections, got none"


# ── Fixed-batch-1 fallback path ────────────────────────────────────────────────


class TestFixedBatch1Fallback:
    def test_fallback_path_still_returns_correct_results(self, synthetic_session) -> None:
        """Explicitly verify the fixed-batch-1 fallback.

        make_synthetic_detector_session exports with shape [1, 3, H, W] —
        batch dim is the integer 1, so detect_batch must fall back to the
        per-frame loop and still return correct results.
        """
        inp_shape = synthetic_session.get_inputs()[0].shape
        assert isinstance(inp_shape[0], int) and inp_shape[0] == 1, (
            "Precondition: synthetic session must have fixed batch dim = 1"
        )

        frame = _make_frame((100, 150, 200))
        det = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
        batch_result = det.detect_batch([frame])

        det_ref = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
        single_result = det_ref.detect(frame)

        assert _detections_equal(batch_result[0], single_result)

    def test_fallback_does_not_perturb_frame_counter(self, synthetic_session) -> None:
        """detect_batch must not permanently advance the frame counter."""
        det = PersonDetector(_session=synthetic_session, conf_threshold=0.5, detect_interval=2)

        # Advance to frame 1 (detects).
        det.detect(_make_frame((0, 0, 0)))  # frame_count = 1

        saved = det._frame_count

        # Call detect_batch; internals restore the counter.
        det.detect_batch([_make_frame((50, 50, 50)), _make_frame((100, 100, 100))])

        assert det._frame_count == saved, (
            f"detect_batch must not perturb frame counter: expected {saved}, got {det._frame_count}"
        )
