"""Unified pose-detector (one backbone → boxes + keypoints) parsing tests.

No real model: a fake ORT session returns crafted YOLO-pose output so we can
assert the box/keypoint decode, NMS, layout auto-detect, and coord mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoptz.engine.pipeline.pose_detect import _POSE_CHANNELS, PoseDetector

_SIZE = 640


class _IO:
    def __init__(self, name: str, shape: list | None = None) -> None:
        self.name = name
        self.shape = shape


class _FakeSession:
    """Minimal ORT-session stand-in returning a fixed raw output array."""

    def __init__(self, raw: np.ndarray) -> None:
        self._raw = raw
        self._inputs = [_IO("images", [1, 3, _SIZE, _SIZE])]
        self._outputs = [_IO("output0")]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _names, _feed):
        return [self._raw]


def _person(cx, cy, w, h, conf, kp_xy):
    """Build one 56-vector: box + conf + 17×(x,y,conf=1)."""
    vec = np.zeros(_POSE_CHANNELS, dtype=np.float32)
    vec[:5] = [cx, cy, w, h, conf]
    for i, (x, y) in enumerate(kp_xy):
        vec[5 + i * 3 + 0] = x
        vec[5 + i * 3 + 1] = y
        vec[5 + i * 3 + 2] = 1.0
    return vec


def _kp17(cx, cy):
    # 17 keypoints clustered near the box centre (values are arbitrary but mapped).
    return [(cx, cy - 60), *[(cx + i, cy + i) for i in range(16)]]


def _raw_channels_first(vectors: list[np.ndarray]) -> np.ndarray:
    arr = np.stack(vectors, axis=0)  # [anchors, 56]
    return arr.T[np.newaxis]  # [1, 56, anchors]


def _frame() -> np.ndarray:
    return np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8)


def _make(raw: np.ndarray) -> PoseDetector:
    return PoseDetector(_session=_FakeSession(raw), conf_threshold=0.25)


def test_decodes_box_and_keypoints() -> None:
    vec = _person(320, 320, 100, 300, 0.9, _kp17(320, 320))
    det = _make(_raw_channels_first([vec]))
    out = det.detect(_frame())
    assert len(out) == 1
    d = out[0]
    # box cx,cy=320,320 w,h=100,300 → x1,y1,x2,y2 (frame==input so 1:1)
    assert d.bbox.x1 == pytest.approx(270, abs=1)
    assert d.bbox.y1 == pytest.approx(170, abs=1)
    assert d.bbox.x2 == pytest.approx(370, abs=1)
    assert d.bbox.y2 == pytest.approx(470, abs=1)
    assert d.keypoints is not None and len(d.keypoints) == 17
    nose = d.keypoints[0]
    assert nose[0] == pytest.approx(320, abs=1)
    assert nose[1] == pytest.approx(260, abs=1)  # cy-60


def test_low_confidence_person_dropped() -> None:
    strong = _person(200, 200, 80, 200, 0.8, _kp17(200, 200))
    weak = _person(450, 450, 80, 200, 0.05, _kp17(450, 450))  # below floor
    det = _make(_raw_channels_first([strong, weak]))
    out = det.detect(_frame())
    assert len(out) == 1
    assert out[0].bbox.cx == pytest.approx(200, abs=2)


def test_transposed_layout_is_autodetected() -> None:
    vec = _person(320, 320, 100, 300, 0.9, _kp17(320, 320))
    arr = np.stack([vec], axis=0)[np.newaxis]  # [1, anchors, 56]
    det = _make(arr)
    out = det.detect(_frame())
    assert len(out) == 1 and out[0].keypoints is not None


def test_nms_merges_duplicate_boxes() -> None:
    a = _person(320, 320, 100, 300, 0.9, _kp17(320, 320))
    b = _person(322, 321, 102, 298, 0.85, _kp17(322, 321))  # heavy overlap
    det = _make(_raw_channels_first([a, b]))
    out = det.detect(_frame())
    assert len(out) == 1  # duplicate suppressed


def test_normalised_coords_scaled_to_pixels() -> None:
    # Same person but in 0..1 normalised space → must scale by input size.
    vec = _person(0.5, 0.5, 100 / _SIZE, 300 / _SIZE, 0.9, [(0.5, 0.4)] * 17)
    det = _make(_raw_channels_first([vec]))
    out = det.detect(_frame())
    assert len(out) == 1
    assert out[0].bbox.cx == pytest.approx(320, abs=2)
    assert out[0].keypoints[0][0] == pytest.approx(320, abs=2)


def _det(x1, y1, x2, y2, kp=True):
    from autoptz.engine.pipeline.detect import BBox, Detection

    keypoints = tuple((x1 + i, y1 + i, 1.0) for i in range(17)) if kp else None
    return Detection(bbox=BBox(x1, y1, x2, y2), conf=0.9, keypoints=keypoints)


def test_adapter_matches_overlapping_detection() -> None:
    from autoptz.engine.pipeline.pose_detect import UnifiedPoseAdapter

    dets = [_det(100, 100, 200, 400), _det(500, 100, 560, 300)]
    adapter = UnifiedPoseAdapter(lambda: dets, ep="CPU")
    kps = adapter.estimate(_frame(), (105, 105, 205, 405))  # overlaps first det
    assert kps is not None and len(kps) == 17
    assert kps[0].x == pytest.approx(100, abs=1)


def test_adapter_returns_none_when_no_overlap() -> None:
    from autoptz.engine.pipeline.pose_detect import UnifiedPoseAdapter

    adapter = UnifiedPoseAdapter(lambda: [_det(0, 0, 50, 50)], ep="CPU")
    assert adapter.estimate(_frame(), (400, 400, 500, 600)) is None


def test_adapter_skips_detections_without_keypoints() -> None:
    from autoptz.engine.pipeline.pose_detect import UnifiedPoseAdapter

    adapter = UnifiedPoseAdapter(lambda: [_det(100, 100, 200, 400, kp=False)], ep="CPU")
    assert adapter.estimate(_frame(), (100, 100, 200, 400)) is None


def test_detect_interval_skips_frames() -> None:
    vec = _person(320, 320, 100, 300, 0.9, _kp17(320, 320))
    det = PoseDetector(
        _session=_FakeSession(_raw_channels_first([vec])),
        conf_threshold=0.25,
        detect_interval=2,
    )
    assert len(det.detect(_frame())) == 1  # frame 1 runs
    assert det.detect(_frame()) == []  # frame 2 skipped
    assert len(det.detect(_frame())) == 1  # frame 3 runs
