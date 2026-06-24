"""Unit tests for autoptz.engine.runtime.bench."""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from autoptz.engine.runtime import bench as bench_mod
from autoptz.engine.runtime.bench import (
    ACCEL_MIN_SPEEDUP,
    AccelReport,
    LatencyStats,
    measure_acceleration,
    run_acceleration_bench,
    time_session,
    verdict,
    zeros_for_session,
)
from autoptz.engine.runtime.inference import EP


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


def _save_identity_model(tmp_path: Path) -> Path:
    import onnx

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "id", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "identity.onnx"
    onnx.save(model, str(path))
    return path


def test_verdict_cpu_only_when_actual_is_cpu() -> None:
    assert verdict(EP.CPU.value, 1.0) == "cpu-only"
    # Even a high "speedup" is cpu-only if the EP actually in use is CPU.
    assert verdict(EP.CPU.value, 9.9) == "cpu-only"


def test_verdict_accelerated_above_threshold() -> None:
    assert verdict(EP.COREML.value, ACCEL_MIN_SPEEDUP) == "accelerated"
    assert verdict(EP.COREML.value, ACCEL_MIN_SPEEDUP + 1.0) == "accelerated"


def test_verdict_no_benefit_below_threshold() -> None:
    # Accelerator EP selected but not actually faster than CPU.
    assert verdict(EP.COREML.value, 1.0) == "no-benefit"
    assert verdict(EP.DIRECTML.value, 1.05) == "no-benefit"


def test_measure_acceleration_cpu_machine(tmp_path: Path) -> None:
    """Forcing CPU-only providers → chosen EP is CPU → verdict 'cpu-only'."""
    model_path = _save_identity_model(tmp_path)
    with patch(
        "autoptz.engine.runtime.inference._available_providers",
        return_value=frozenset({"CPUExecutionProvider"}),
    ):
        report = measure_acceleration(model_path, warmup=1, runs=3)
    assert isinstance(report, AccelReport)
    assert report.actual_ep == EP.CPU.value
    assert report.verdict == "cpu-only"
    assert report.accel.runs == 3
    assert report.cpu.runs == 3
    assert report.speedup > 0.0


def test_accel_report_summary_and_dict() -> None:
    stats = LatencyStats(runs=3, median_ms=2.0, p95_ms=3.0, mean_ms=2.5, fps=500.0)
    report = AccelReport(
        model="m.onnx",
        requested_ep=EP.COREML.value,
        actual_ep=EP.CPU.value,
        precision="fp32",
        speedup=1.0,
        verdict="cpu-only",
        accel=stats,
        cpu=stats,
    )
    s = report.summary()
    assert "cpu-only" in s.lower()  # verdict surfaced
    assert "CoreML" in s  # requested EP named (because it differs from actual)
    assert "CPU" in s  # actual EP labelled honestly
    d = report.to_dict()
    assert d["verdict"] == "cpu-only"
    assert isinstance(d["accel"], dict)


def test_run_acceleration_bench_no_model(monkeypatch, capsys) -> None:
    fake_mgr = types.SimpleNamespace(ensure_detector=lambda *a, **k: None, last_error="no model")
    monkeypatch.setattr(bench_mod, "default_manager", lambda: fake_mgr)
    code = run_acceleration_bench(tier="auto")
    assert code == 1
    assert "no detector model" in capsys.readouterr().out.lower()


def test_run_acceleration_bench_reports_and_writes_json(tmp_path, monkeypatch, capsys) -> None:
    model_path = _save_identity_model(tmp_path)
    fake_mgr = types.SimpleNamespace(ensure_detector=lambda *a, **k: str(model_path), last_error="")
    monkeypatch.setattr(bench_mod, "default_manager", lambda: fake_mgr)
    json_out = tmp_path / "bench.json"
    with patch(
        "autoptz.engine.runtime.inference._available_providers",
        return_value=frozenset({"CPUExecutionProvider"}),
    ):
        code = run_acceleration_bench(tier="auto", json_path=str(json_out), warmup=1, runs=3)
    assert code == 0
    out = capsys.readouterr().out
    assert "cpu-only" in out.lower()  # CI host has no accelerator
    data = json.loads(json_out.read_text())
    assert data["verdict"] == "cpu-only"
    assert data["actual_ep"] == "CPUExecutionProvider"
