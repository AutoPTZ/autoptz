"""Tests for InferencePool acceleration caching (A-2).

Covers:
- measure_detector_acceleration() happy path → state "done", verdict/summary exposed.
- Failure path → state "unavailable", verdict/summary empty.
- Unified-pose path → skipped (state stays "idle").
- Empty model path → skipped (state stays "idle").
"""

from __future__ import annotations

from autoptz.engine.runtime.bench import AccelReport, LatencyStats
from autoptz.engine.runtime.inference import EP

# ── helpers ──────────────────────────────────────────────────────────────────


def _fake_report(verdict_str: str = "accelerated", summary_str: str = "") -> AccelReport:
    """Build a real AccelReport with synthetic but valid latency stats."""
    stats = LatencyStats(runs=8, median_ms=2.0, p95_ms=3.0, mean_ms=2.2, fps=500.0)
    return AccelReport(
        model="yolo11n.onnx",
        requested_ep=EP.COREML.value,
        actual_ep=EP.COREML.value if verdict_str == "accelerated" else EP.CPU.value,
        precision="fp32",
        speedup=1.24 if verdict_str == "accelerated" else 0.98,
        verdict=verdict_str,
        accel=stats,
        cpu=stats,
    )


def _make_pool(unified_pose: bool = False, model_path: str = "/fake/model.onnx") -> object:
    """Return an InferencePool with pre-seeded detector state (no I/O)."""
    from autoptz.engine.pipeline.pool import InferencePool

    pool = InferencePool(allow_model_download=False)
    pool._unified_pose = unified_pose
    pool._detector_model_path = model_path
    # Simulate "detector already built" so measure_detector_acceleration can check state.
    pool._detector_built = True
    return pool


# ── pool: happy path ─────────────────────────────────────────────────────────


class TestMeasureDetectorAcceleration:
    def test_done_state_and_verdict(self, monkeypatch) -> None:
        from autoptz.engine.runtime import bench as bench_mod

        report = _fake_report("accelerated")
        monkeypatch.setattr(bench_mod, "measure_acceleration", lambda p, warmup, runs: report)
        pool = _make_pool()
        pool.measure_detector_acceleration()

        assert pool._accel_state == "done"
        assert pool.detector_accel_verdict() == "accelerated"
        assert pool.detector_accel_summary() != ""
        assert "CoreML" in pool.detector_accel_summary()

    def test_accel_summary_non_empty_on_done(self, monkeypatch) -> None:
        from autoptz.engine.runtime import bench as bench_mod

        report = _fake_report("accelerated")
        monkeypatch.setattr(bench_mod, "measure_acceleration", lambda p, warmup, runs: report)
        pool = _make_pool()
        pool.measure_detector_acceleration()
        assert pool.detector_accel_summary() != ""

    def test_no_benefit_verdict_exposed(self, monkeypatch) -> None:
        from autoptz.engine.runtime import bench as bench_mod

        report = _fake_report("no-benefit")
        monkeypatch.setattr(bench_mod, "measure_acceleration", lambda p, warmup, runs: report)
        pool = _make_pool()
        pool.measure_detector_acceleration()
        assert pool.detector_accel_verdict() == "no-benefit"

    def test_cpu_only_verdict_exposed(self, monkeypatch) -> None:
        from autoptz.engine.runtime import bench as bench_mod

        report = _fake_report("cpu-only")
        monkeypatch.setattr(bench_mod, "measure_acceleration", lambda p, warmup, runs: report)
        pool = _make_pool()
        pool.measure_detector_acceleration()
        assert pool.detector_accel_verdict() == "cpu-only"


# ── pool: failure path ────────────────────────────────────────────────────────


class TestMeasureDetectorAccelFailure:
    def test_unavailable_on_exception(self, monkeypatch) -> None:
        from autoptz.engine.runtime import bench as bench_mod

        monkeypatch.setattr(
            bench_mod,
            "measure_acceleration",
            lambda p, warmup, runs: (_ for _ in ()).throw(RuntimeError("ort broken")),
        )
        pool = _make_pool()
        pool.measure_detector_acceleration()  # must not raise
        assert pool._accel_state == "unavailable"
        assert pool.detector_accel_verdict() == ""
        assert pool.detector_accel_summary() == ""


# ── pool: skip guards ─────────────────────────────────────────────────────────


class TestMeasureDetectorAccelSkips:
    def test_skips_unified_pose(self, monkeypatch) -> None:
        from autoptz.engine.runtime import bench as bench_mod

        called = []
        monkeypatch.setattr(
            bench_mod,
            "measure_acceleration",
            lambda p, warmup, runs: called.append(p) or _fake_report(),
        )
        pool = _make_pool(unified_pose=True)
        pool.measure_detector_acceleration()
        assert not called, "measure_acceleration must not be called for unified-pose"
        assert pool._accel_state == "idle"

    def test_skips_empty_model_path(self, monkeypatch) -> None:
        from autoptz.engine.runtime import bench as bench_mod

        called = []
        monkeypatch.setattr(
            bench_mod,
            "measure_acceleration",
            lambda p, warmup, runs: called.append(p) or _fake_report(),
        )
        pool = _make_pool(model_path="")
        pool.measure_detector_acceleration()
        assert not called
        assert pool._accel_state == "idle"

    def test_skips_placeholder_model_path(self, monkeypatch) -> None:
        from autoptz.engine.runtime import bench as bench_mod

        called = []
        monkeypatch.setattr(
            bench_mod,
            "measure_acceleration",
            lambda p, warmup, runs: called.append(p) or _fake_report(),
        )
        pool = _make_pool(model_path="<yolo11-pose unified>")
        pool.measure_detector_acceleration()
        assert not called
        assert pool._accel_state == "idle"

    def test_skips_if_already_done(self, monkeypatch) -> None:
        from autoptz.engine.runtime import bench as bench_mod

        called = []
        monkeypatch.setattr(
            bench_mod,
            "measure_acceleration",
            lambda p, warmup, runs: called.append(p) or _fake_report(),
        )
        pool = _make_pool()
        pool._accel_state = "done"
        pool.measure_detector_acceleration()
        assert not called, "must not re-measure if state is already done"


# ── pool: initial state ───────────────────────────────────────────────────────


class TestPoolAccelInitialState:
    def test_summary_empty_before_measurement(self) -> None:
        from autoptz.engine.pipeline.pool import InferencePool

        pool = InferencePool(allow_model_download=False)
        assert pool.detector_accel_summary() == ""
        assert pool.detector_accel_verdict() == ""
        assert pool._accel_state == "idle"
