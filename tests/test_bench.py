"""Unit tests for autoptz.engine.runtime.bench."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from autoptz.engine.runtime.bench import (
    LatencyStats,
    time_session,
    zeros_for_session,
)


def _identity_session() -> ort.InferenceSession:
    """A trivial [1,4]->[1,4] float Identity model as an ORT CPU session."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "id", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def test_zeros_for_session_matches_input() -> None:
    sess = _identity_session()
    feeds = zeros_for_session(sess)
    assert set(feeds) == {"X"}
    assert feeds["X"].shape == (1, 4)
    assert feeds["X"].dtype == np.float32


def test_time_session_returns_positive_stats() -> None:
    sess = _identity_session()
    stats = time_session(sess, warmup=1, runs=5)
    assert isinstance(stats, LatencyStats)
    assert stats.runs == 5
    assert stats.median_ms > 0.0
    assert stats.p95_ms >= stats.median_ms
    assert stats.fps > 0.0


def test_latency_stats_to_dict_roundtrips() -> None:
    stats = LatencyStats(runs=5, median_ms=2.0, p95_ms=3.0, mean_ms=2.5, fps=500.0)
    d = stats.to_dict()
    assert d["runs"] == 5
    assert d["median_ms"] == 2.0
    assert d["fps"] == 500.0
