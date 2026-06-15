"""Unit tests for autoptz.engine.pipeline.detect.

All tests use synthetic ONNX models (built with onnx.helper) or mocked ORT
sessions — no real model files or hardware required.
"""
from __future__ import annotations

import numpy as np
import pytest

from autoptz.engine.pipeline.detect import (
    BBox,
    Detection,
    PersonDetector,
    _letterbox,
    _nms,
    _parse_raw_output,
    _ScaleInfo,
    _to_orig_coords,
    detections_to_numpy,
    make_synthetic_detector_session,
)

onnx = pytest.importorskip("onnx")


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_session():
    """ORT session returning two hardcoded person detections."""
    return make_synthetic_detector_session(
        input_size=640,
        detections=[
            (100.0, 150.0, 200.0, 400.0, 0.90),
            (300.0, 100.0, 450.0, 420.0, 0.82),
        ],
    )


@pytest.fixture
def black_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ── BBox ──────────────────────────────────────────────────────────────────────

class TestBBox:
    def test_properties(self) -> None:
        b = BBox(10, 20, 110, 120)
        assert b.cx == 60.0
        assert b.cy == 70.0
        assert b.w == 100.0
        assert b.h == 100.0
        assert b.area() == 10_000.0

    def test_iou_identical(self) -> None:
        b = BBox(0, 0, 100, 100)
        assert b.iou(b) == pytest.approx(1.0)

    def test_iou_no_overlap(self) -> None:
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 20, 30, 30)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_partial(self) -> None:
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 15, 15)
        assert 0 < a.iou(b) < 1

    def test_as_xyxy(self) -> None:
        b = BBox(1, 2, 3, 4)
        assert b.as_xyxy() == (1, 2, 3, 4)

    def test_zero_area(self) -> None:
        b = BBox(5, 5, 5, 5)
        assert b.area() == 0.0

    def test_iou_zero_area_safe(self) -> None:
        a = BBox(0, 0, 0, 0)
        b = BBox(0, 0, 10, 10)
        assert a.iou(b) == pytest.approx(0.0)


# ── Letterbox ─────────────────────────────────────────────────────────────────

class TestLetterbox:
    def test_square_frame_unchanged_ratio(self) -> None:
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        out, si = _letterbox(frame, 640)
        assert si.ratio == pytest.approx(1.0)
        assert si.pad_x == 0
        assert si.pad_y == 0

    def test_wide_frame_padded_vertically(self) -> None:
        # 1920×1080 → ratio = 640/1920 = 0.333; new_h = 360; pad_y = 140
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out, si = _letterbox(frame, 640)
        assert out.shape == (1, 3, 640, 640)
        assert si.ratio == pytest.approx(640 / 1920, rel=0.01)
        assert si.pad_y > 0
        assert si.pad_x == 0

    def test_tall_frame_padded_horizontally(self) -> None:
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        out, si = _letterbox(frame, 640)
        assert out.shape == (1, 3, 640, 640)
        assert si.pad_x > 0
        assert si.pad_y == 0

    def test_output_dtype_float32(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out, _ = _letterbox(frame, 640)
        assert out.dtype == np.float32

    def test_output_normalised_01(self) -> None:
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        out, _ = _letterbox(frame, 640)
        assert out.max() <= 1.0 + 1e-6


class TestToOrigCoords:
    def test_no_padding(self) -> None:
        si = _ScaleInfo(ratio=0.5, pad_x=0, pad_y=0, orig_h=480, orig_w=640)
        # bbox at (50,50,100,100) in 320×240 padded space → (100,100,200,200) orig
        b = _to_orig_coords(50, 50, 100, 100, si)
        assert b.x1 == pytest.approx(100.0)
        assert b.y1 == pytest.approx(100.0)

    def test_with_padding(self) -> None:
        # 640×480 frame → ratio=1.0 but padded vertically 80px each side
        si = _ScaleInfo(ratio=1.0, pad_x=0, pad_y=80, orig_h=480, orig_w=640)
        # bbox top at y=80 in padded space → y=0 in original
        b = _to_orig_coords(0, 80, 100, 560, si)
        assert b.y1 == pytest.approx(0.0)
        assert b.y2 == pytest.approx(480.0)

    def test_clamp_to_frame(self) -> None:
        si = _ScaleInfo(ratio=1.0, pad_x=0, pad_y=0, orig_h=100, orig_w=100)
        b = _to_orig_coords(-10, -10, 200, 200, si)
        assert b.x1 == 0.0
        assert b.y1 == 0.0
        assert b.x2 == 100.0
        assert b.y2 == 100.0


# ── NMS ───────────────────────────────────────────────────────────────────────

class TestNMS:
    def test_no_boxes(self) -> None:
        assert _nms(np.empty((0, 4), np.float32), np.empty(0, np.float32)) == []

    def test_single_box(self) -> None:
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        assert _nms(boxes, scores) == [0]

    def test_non_overlapping_both_kept(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        kept = _nms(boxes, scores)
        assert set(kept) == {0, 1}

    def test_high_overlap_lower_suppressed(self) -> None:
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        kept = _nms(boxes, scores, iou_thr=0.5)
        assert len(kept) == 1
        assert kept[0] == 0

    def test_scores_order_respected(self) -> None:
        # Higher-scoring box should win when overlapping
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=np.float32)
        scores = np.array([0.7, 0.95], dtype=np.float32)
        kept = _nms(boxes, scores, iou_thr=0.5)
        assert kept[0] == 1  # index of highest score


# ── _parse_raw_output ─────────────────────────────────────────────────────────

class TestParseRawOutput:
    INPUT_SIZE = 640

    def _parse(self, raw: np.ndarray, floor: float = 0.0) -> np.ndarray:
        return _parse_raw_output(raw, self.INPUT_SIZE, floor)

    def test_nms_free_6col(self) -> None:
        # [1, N, 6] in pixel coords
        raw = np.zeros((1, 3, 6), dtype=np.float32)
        raw[0, 0] = [10, 10, 100, 200, 0.9, 0]
        raw[0, 1] = [200, 10, 300, 200, 0.8, 0]
        raw[0, 2] = [0, 0, 0, 0, 0.05, 0]   # below floor
        result = self._parse(raw, floor=0.1)
        assert result.shape[1] == 6
        assert len(result) == 2

    def test_nms_free_5col_person_class_added(self) -> None:
        raw = np.zeros((1, 2, 5), dtype=np.float32)
        raw[0, 0] = [10, 10, 100, 200, 0.9]
        raw[0, 1] = [200, 10, 300, 200, 0.85]
        result = self._parse(raw)
        assert result.shape[1] == 6
        assert all(result[:, 5] == 0.0)

    def test_nms_free_normalised_coords_scaled_up(self) -> None:
        # coords in [0, 1] range → should be multiplied by input_size
        raw = np.zeros((1, 1, 6), dtype=np.float32)
        raw[0, 0] = [0.1, 0.1, 0.2, 0.5, 0.9, 0.0]
        result = self._parse(raw)
        assert result[0, 0] == pytest.approx(0.1 * self.INPUT_SIZE, rel=0.01)

    def test_yolov8_prenmms_format(self) -> None:
        # [1, 4+C, anchors] pre-NMS format — 2 persons
        n_anchors = 50
        n_classes = 1  # person-only
        raw = np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)
        # Set two strong detections (cx,cy,w,h,class_score)
        raw[0, :, 0] = [200, 200, 100, 200, 0.95]
        raw[0, :, 1] = [400, 300, 80, 150, 0.85]
        result = self._parse(raw, floor=0.5)
        assert result.shape[1] == 6
        assert len(result) >= 2

    def test_conf_floor_filters(self) -> None:
        raw = np.zeros((1, 4, 6), dtype=np.float32)
        raw[0, 0] = [10, 10, 100, 200, 0.9, 0]
        raw[0, 1] = [10, 10, 100, 200, 0.3, 0]
        raw[0, 2] = [10, 10, 100, 200, 0.05, 0]
        result = self._parse(raw, floor=0.5)
        assert len(result) == 1


# ── PersonDetector ────────────────────────────────────────────────────────────

class TestPersonDetector:
    def test_detect_returns_two_persons(self, synthetic_session, black_frame) -> None:
        det = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
        results = det.detect(black_frame)
        assert len(results) == 2
        assert all(isinstance(r, Detection) for r in results)
        assert all(r.class_id == 0 for r in results)
        assert all(r.conf >= 0.5 for r in results)

    def test_detect_ep_property(self, synthetic_session, black_frame) -> None:
        det = PersonDetector(_session=synthetic_session)
        assert "CPU" in det.ep

    def test_detect_interval_skips_frames(self, synthetic_session, black_frame) -> None:
        det = PersonDetector(_session=synthetic_session, detect_interval=3)
        # Frame 1: should detect
        r1 = det.detect(black_frame)
        assert len(r1) > 0
        # Frame 2: skip
        r2 = det.detect(black_frame)
        assert r2 == []
        # Frame 3: skip
        r3 = det.detect(black_frame)
        assert r3 == []
        # Frame 4 = 4 % 3 == 1: detect
        r4 = det.detect(black_frame)
        assert len(r4) > 0

    def test_detect_bbox_coords_in_frame_range(self, synthetic_session, black_frame) -> None:
        det = PersonDetector(_session=synthetic_session, conf_threshold=0.5)
        H, W = black_frame.shape[:2]
        for d in det.detect(black_frame):
            assert 0 <= d.bbox.x1 < d.bbox.x2 <= W + 1
            assert 0 <= d.bbox.y1 < d.bbox.y2 <= H + 1

    def test_high_conf_threshold_filters_weak(self) -> None:
        session = make_synthetic_detector_session(
            detections=[
                (10, 10, 100, 200, 0.3),   # below threshold
                (200, 10, 300, 200, 0.9),  # above threshold
            ]
        )
        det = PersonDetector(_session=session, conf_threshold=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 1
        assert results[0].conf >= 0.5

    def test_no_model_path_and_no_session_raises(self) -> None:
        with pytest.raises(ValueError, match="model_path or _session"):
            PersonDetector()

    def test_reset_restarts_frame_count(self, synthetic_session, black_frame) -> None:
        det = PersonDetector(_session=synthetic_session, detect_interval=2)
        det.detect(black_frame)  # frame 1
        det.detect(black_frame)  # frame 2 → skip
        det.reset()              # count back to 0
        # Next call is frame 1 again → should detect
        results = det.detect(black_frame)
        assert len(results) > 0


# ── detections_to_numpy ───────────────────────────────────────────────────────

class TestDetectionsToNumpy:
    def test_empty(self) -> None:
        arr = detections_to_numpy([])
        assert arr.shape == (0, 6)
        assert arr.dtype == np.float32

    def test_two_detections(self) -> None:
        dets = [
            Detection(BBox(10, 20, 100, 200), 0.9, 0),
            Detection(BBox(50, 60, 150, 250), 0.7, 0),
        ]
        arr = detections_to_numpy(dets)
        assert arr.shape == (2, 6)
        np.testing.assert_allclose(arr[0, :4], [10, 20, 100, 200])
        assert arr[0, 4] == pytest.approx(0.9)
        assert arr[0, 5] == 0.0

    def test_column_order(self) -> None:
        det = Detection(BBox(1, 2, 3, 4), conf=0.77, class_id=0)
        arr = detections_to_numpy([det])
        assert arr[0, 0] == 1.0  # x1
        assert arr[0, 1] == 2.0  # y1
        assert arr[0, 2] == 3.0  # x2
        assert arr[0, 3] == 4.0  # y2
        assert arr[0, 4] == pytest.approx(0.77)  # conf
        assert arr[0, 5] == 0.0  # class


# ── make_synthetic_detector_session ───────────────────────────────────────────

class TestMakeSyntheticSession:
    def test_default_detections(self) -> None:
        session = make_synthetic_detector_session()
        det = PersonDetector(_session=session, conf_threshold=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) >= 1

    def test_empty_detections(self) -> None:
        session = make_synthetic_detector_session(detections=[])
        det = PersonDetector(_session=session, conf_threshold=0.1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert results == []

    def test_custom_input_size(self) -> None:
        session = make_synthetic_detector_session(input_size=480)
        det = PersonDetector(_session=session, input_size=480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert isinstance(results, list)
