"""Regression tests for the pose estimator's model-input-size handling.

The exported YOLO-pose ONNX has a FIXED input (640×640).  A previous default
letterboxed every crop to 256×256, so ORT rejected every ``run`` and
:meth:`PoseEstimator.estimate` silently returned ``None`` — pose produced no
keypoints (blank skeleton overlay, bbox-only "torso" aim that moved with the
box).  These tests pin the fix: the estimator letterboxes to the size the model
actually declares, and feeds ORT a tensor of exactly that size.

No real model is needed — a tiny fake ORT session stands in.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoptz.engine.pipeline.pose import PoseEstimator, _model_input_size


class _FakeInput:
    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape


class _FakeOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Minimal ORT-session stand-in that records the fed tensor's spatial size.

    ``run`` returns a YOLO-pose-shaped ``[1, 56, anchors]`` output with one
    high-confidence anchor so the parse path yields 17 keypoints.
    """

    def __init__(self, input_shape: list) -> None:
        self._inputs = [_FakeInput("images", input_shape)]
        self._outputs = [_FakeOutput("output0")]
        self.fed_hw: tuple[int, int] | None = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, feed):
        tensor = feed["images"]
        # tensor is [1, 3, H, W]
        self.fed_hw = (tensor.shape[2], tensor.shape[3])
        out = np.zeros((1, 56, 4), dtype=np.float32)
        out[0, 4, 0] = 0.99  # one confident anchor (channel 4 = person conf)
        return [out]


class TestModelInputSize:
    def test_static_square_shape(self) -> None:
        assert _model_input_size([1, 3, 640, 640], fallback=256) == 640

    def test_dynamic_axes_fall_back(self) -> None:
        assert _model_input_size([1, 3, "height", "width"], fallback=320) == 320

    def test_none_axes_fall_back(self) -> None:
        assert _model_input_size([1, 3, None, None], fallback=288) == 288

    def test_garbage_shape_falls_back(self) -> None:
        assert _model_input_size([], fallback=416) == 416


class TestPoseEstimatorRespectsModelSize:
    def test_input_size_taken_from_session(self) -> None:
        pe = PoseEstimator(_session=_FakeSession([1, 3, 640, 640]))
        assert pe.available
        assert pe._input_size == 640

    def test_non_default_static_size_is_used(self) -> None:
        pe = PoseEstimator(_session=_FakeSession([1, 3, 512, 512]))
        assert pe._input_size == 512

    def test_estimate_feeds_tensor_of_model_size(self) -> None:
        # The crux of the original bug: the tensor handed to ORT must match the
        # model's declared input size, or every run() raises and pose goes dark.
        session = _FakeSession([1, 3, 640, 640])
        pe = PoseEstimator(_session=session)
        rng = np.random.default_rng(1234)
        frame = (rng.random((720, 1280, 3)) * 255).astype(np.uint8)
        kps = pe.estimate(frame, (500.0, 200.0, 760.0, 680.0))
        assert session.fed_hw == (640, 640)
        assert kps is not None and len(kps) == 17

    def test_dynamic_model_uses_constructor_size(self) -> None:
        session = _FakeSession([1, 3, "h", "w"])
        pe = PoseEstimator(_session=session, input_size=384)
        rng = np.random.default_rng(5678)
        frame = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        pe.estimate(frame, (100.0, 50.0, 300.0, 400.0))
        assert session.fed_hw == (384, 384)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
