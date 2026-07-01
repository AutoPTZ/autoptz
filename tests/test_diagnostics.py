"""Diagnostics: latency telemetry, the latencyMs model role, and log control/export.

Covers the engine-logging + diagnostics work:
  - ``TelemetryMsg.latency_ms`` field (default + msgpack round-trip).
  - ``CameraListModel.LatencyMsRole`` / ``latencyMs`` role (int ms) wired from
    telemetry, plus the ``CameraRecord.latency_ms`` accessor.
  - ``EngineClient.setLogLevel`` adjusting the root logger + handler threshold.
  - ``EngineClient.copyLogsToClipboard`` / ``exportLogs`` returning/writing the
    full buffered log text, and ``LogListModel.dump_text``.

All tests use QCoreApplication (no display) so they run cleanly in CI.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import PySide6  # noqa: F401
import pytest


def _telemetry(camera_id: str, **kw):
    from autoptz.engine.runtime.messages import TelemetryMsg

    return TelemetryMsg(camera_id=camera_id, seq=1, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# TelemetryMsg.latency_ms
# ─────────────────────────────────────────────────────────────────────────────


class TestLatencyTelemetryField:
    def test_default_zero(self) -> None:
        msg = _telemetry("c")
        assert msg.latency_ms == 0.0

    def test_round_trips_msgpack(self) -> None:
        from autoptz.engine.runtime.messages import TelemetryMsg

        msg = _telemetry("c", latency_ms=12.5)
        restored = TelemetryMsg.from_msgpack(msg.to_msgpack())
        assert restored.latency_ms == pytest.approx(12.5)


class TestOptionalComponentDiagnostics:
    def test_optional_components_include_setup_details(self) -> None:
        from autoptz.engine.runtime.diagnostics import optional_components

        rows = optional_components()
        keys = {r["key"] for r in rows}
        assert {"detector", "reid", "pose", "face"}.issubset(keys)
        for row in rows:
            assert row["source"]
            assert row["path"]
            assert row["why"]
            assert row["managed"]
            assert row["network"]


class TestTrackerStatusNonBlocking:
    """tracker_status must probe with find_spec, never a heavy ``import boxmot``.

    Importing boxmot pulls in torch (multi-second); this probe runs on the GUI
    thread's Services-panel poll, so a real import there froze the UI at launch.
    """

    def test_reflects_module_presence(self, monkeypatch) -> None:
        from autoptz.engine.runtime import diagnostics as diag

        monkeypatch.setattr(diag, "_module_present", lambda name: name == "boxmot")
        row = diag.tracker_status()
        assert row["state"] == "ok"
        assert "boxmot" in row["detail"].lower()

        monkeypatch.setattr(diag, "_module_present", lambda name: False)
        assert diag.tracker_status()["state"] == "warn"

    def test_does_not_import_boxmot(self) -> None:
        import sys

        from autoptz.engine.runtime import diagnostics as diag

        had_boxmot = "boxmot" in sys.modules
        diag.tracker_status()
        if not had_boxmot:
            assert "boxmot" not in sys.modules, "tracker_status must not import boxmot"


class TestFaceStatusModelPresence:
    """face_status must reflect whether the model WEIGHTS are on disk, not just
    whether the insightface package imports.

    The "faces never save on Windows" symptom: an offline first-run has the
    package installed but no ``buffalo_l`` weights, so SCRFD/ArcFace never load.
    The old probe reported "ok" purely on package import — hiding the real cause.
    """

    def test_warns_when_package_present_but_model_missing(self, monkeypatch, tmp_path) -> None:
        from autoptz.engine.runtime import diagnostics as diag

        monkeypatch.setattr(diag, "_module_present", lambda name: name == "insightface")
        monkeypatch.setenv("INSIGHTFACE_HOME", str(tmp_path))  # empty → no weights
        row = diag.face_status()
        assert row["state"] == "warn"
        assert "model" in row["detail"].lower()
        assert str(tmp_path) in row["detail"]

    def test_ok_when_model_weights_present(self, monkeypatch, tmp_path) -> None:
        from autoptz.engine.runtime import diagnostics as diag

        monkeypatch.setattr(diag, "_module_present", lambda name: name == "insightface")
        models = tmp_path / "models" / "buffalo_l"
        models.mkdir(parents=True)
        (models / "det_10g.onnx").write_bytes(b"x")
        monkeypatch.setenv("INSIGHTFACE_HOME", str(tmp_path))
        row = diag.face_status()
        assert row["state"] == "ok"
        assert str(models) in row["detail"]

    def test_optional_components_face_path_uses_resolved_root(self, monkeypatch, tmp_path) -> None:
        from autoptz.engine.runtime import diagnostics as diag

        monkeypatch.setenv("INSIGHTFACE_HOME", str(tmp_path))
        face = next(row for row in diag.optional_components() if row["key"] == "face")
        assert face["path"] == str(tmp_path / "models")

    def test_off_when_package_absent(self, monkeypatch) -> None:
        from autoptz.engine.runtime import diagnostics as diag

        monkeypatch.setattr(diag, "_module_present", lambda name: False)
        assert diag.face_status()["state"] == "off"


class TestInferencePoolRelease:
    """Releasing a pooled model drops the cached instance + re-arms the build."""

    def test_release_resets_built_flags(self) -> None:
        from autoptz.engine.pipeline.pool import InferencePool

        pool = InferencePool(allow_model_download=False)
        # Simulate "already built" without loading real models.
        pool._detector = object()
        pool._detector_built = True
        pool._face = object()
        pool._face_built = True
        pool._pose = object()
        pool._pose_built = True

        pool.release_detector()
        pool.release_face()
        pool.release_pose()

        assert pool._detector is None and pool._detector_built is False
        assert pool._face is None and pool._face_built is False
        assert pool._pose is None and pool._pose_built is False


class _FakeSource:
    """Deterministic frame source yielding solid-colour frames."""

    def __init__(self, h: int = 240, w: int = 320) -> None:
        self.h, self.w = h, w

    def open(self) -> bool:
        return True

    def read(self):
        return np.full((self.h, self.w, 3), 100, dtype=np.uint8)

    def close(self) -> None:
        pass


class TestWorkerPopulatesLatency:
    def test_latency_ms_becomes_positive(self, qapp, wait_until) -> None:
        from autoptz.config.models import CameraConfig, SourceConfig
        from autoptz.engine.camera_worker import CameraWorker

        config = CameraConfig(
            id="latcam01abcd",
            name="Cam",
            source=SourceConfig(type="usb", address="usb://0"),
        )
        received: list = []
        lock = threading.Lock()

        def on_tel(m):
            with lock:
                received.append(m)

        worker = CameraWorker(
            "latcam01abcd",
            config,
            on_tel,
            frame_source=_FakeSource(),
            telemetry_hz=50.0,
        )
        worker.start()
        try:

            def _best_latency() -> float:
                with lock:
                    if received:
                        return max(m.latency_ms for m in received)
                return 0.0

            best = wait_until(
                _best_latency,
                timeout=3.0,
                interval=0.02,
                message="latency_ms never became positive with a live source",
            )
            assert best > 0.0, "latency_ms never became positive with a live source"
        finally:
            worker.stop()


# ─────────────────────────────────────────────────────────────────────────────
# CameraListModel.latencyMs role + CameraRecord.latency_ms accessor
# ─────────────────────────────────────────────────────────────────────────────


class TestLatencyModelRole:
    def _model_with_camera(self):
        from autoptz.ui.engine_client import CameraListModel, CameraRecord

        m = CameraListModel()
        m.add_camera(CameraRecord(camera_id="c1", source_uri="usb://0", display_name="Cam"))
        return m

    def test_role_present(self, qapp) -> None:
        m = self._model_with_camera()
        names = m.roleNames()
        assert b"latencyMs" in names.values()

    def test_role_defaults_zero(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel

        m = self._model_with_camera()
        assert m.data(m.index(0), CameraListModel.LatencyMsRole) == 0

    def test_role_reports_int_ms_from_telemetry(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel

        m = self._model_with_camera()
        m.update_telemetry(_telemetry("c1", latency_ms=18.7))
        val = m.data(m.index(0), CameraListModel.LatencyMsRole)
        assert val == 19  # rounded to whole ms
        assert isinstance(val, int)

    def test_record_accessor_rounds(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraRecord

        rec = CameraRecord(camera_id="c1", source_uri="usb://0", display_name="Cam")
        assert rec.latency_ms == 0
        rec.telemetry = _telemetry("c1", latency_ms=33.2)
        assert rec.latency_ms == 33


class TestSystemMetricsShape:
    def test_app_memory_percent_key_present(self) -> None:
        from autoptz.engine.runtime.diagnostics import system_metrics

        metrics = system_metrics()
        assert "app_mem_percent" in metrics
        assert isinstance(metrics["app_mem_percent"], int | float)

    def test_app_memory_is_honest_footprint_not_inflated_rss(self) -> None:
        # "App Mem" must reflect real memory pressure: macOS phys_footprint, else
        # RSS. Feed a FAKE proc with a fixed RSS so the fallback is deterministic —
        # a live process's RSS churns between two reads, which made the old
        # `mem == rss` assertion flaky on non-macOS CI.
        import os
        import sys
        import types

        from autoptz.engine.runtime.diagnostics import _proc_memory_bytes

        sentinel_rss = 7_000_000_003  # implausible exact value
        fake = types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=sentinel_rss))

        # Pass THIS process's pid: macOS reads its real phys_footprint for that pid.
        mem = _proc_memory_bytes(os.getpid(), fake)
        assert mem > 0
        if sys.platform == "darwin":
            # macOS reads the real proc_pid_rusage phys_footprint of the given pid,
            # so it must differ from the sentinel RSS fallback.
            assert mem != sentinel_rss
        else:
            # Non-macOS returns the proc's RSS straight through — now exact.
            assert mem == sentinel_rss

    def test_unreadable_pid_falls_back_to_rss(self) -> None:
        # A child pid the rusage call can't read (or any non-darwin path) must fall
        # back to the proc's RSS rather than raising or returning 0.
        import types

        from autoptz.engine.runtime.diagnostics import _proc_memory_bytes

        sentinel_rss = 4_242_424_247
        fake = types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=sentinel_rss))
        # An obviously-invalid pid: proc_pid_rusage fails → RSS fallback on every OS.
        mem = _proc_memory_bytes(-1, fake)
        assert mem == sentinel_rss


def _burn_cpu() -> None:  # pragma: no cover — runs in a spawned child
    """Tight loop that pins one core; the parent measures whether it's counted."""
    x = 0
    while True:
        x = (x * 1_000_003 + 7) % 2_147_483_647


class TestAppProcessTreeCpu:
    """``system_metrics`` must count child processes (process-per-camera workers,
    go2rtc) in App CPU/Mem — the GUI PID alone undercounts badly once cameras run
    in their own processes ('not accounting for the new process')."""

    def _reset(self, monkeypatch) -> None:
        from autoptz.engine.runtime import diagnostics as diag

        for name, val in (
            ("_PROC", None),
            ("_PRIMED", False),
            ("_CHILD_PROCS", {}),
            ("_CACHE", None),
            ("_CACHE_T", 0.0),
        ):
            monkeypatch.setattr(diag, name, val)

    def test_tree_cpu_primes_newcomers_then_counts_them(self, monkeypatch) -> None:
        from autoptz.engine.runtime import diagnostics as diag

        monkeypatch.setattr(diag, "_CHILD_PROCS", {})

        class FakeChild:
            def __init__(self, pid: int, value: float) -> None:
                self.pid, self.value, self.calls = pid, value, 0

            def cpu_percent(self, interval=None):  # noqa: ANN001
                self.calls += 1
                return self.value

        child = FakeChild(101, 40.0)

        class FakeParent:
            def children(self, recursive=False):  # noqa: ANN001
                return [child]

        parent = FakeParent()
        # First sight → primed only (0 baseline), contributes nothing this round.
        assert diag._app_tree_cpu(parent) == 0.0
        # Known on the next round → its real CPU is counted.
        assert diag._app_tree_cpu(parent) == 40.0
        assert child.calls == 2

    def test_tree_cpu_reads_the_primed_instance_across_calls(self, monkeypatch) -> None:
        # children() hands back a FRESH Process object each call (like psutil); the
        # real read must come from the cached/primed instance, not the fresh one —
        # else the cpu_percent baseline resets every call and CPU reads as ~0.
        from autoptz.engine.runtime import diagnostics as diag

        monkeypatch.setattr(diag, "_CHILD_PROCS", {})

        class FakeChild:
            def __init__(self, pid: int, value: float) -> None:
                self.pid, self.value, self.calls = pid, value, 0

            def cpu_percent(self, interval=None):  # noqa: ANN001
                self.calls += 1
                return self.value

        primed = FakeChild(101, 33.0)
        fresh = FakeChild(101, 999.0)
        batches = iter([[primed], [fresh]])

        class FakeParent:
            def children(self, recursive=False):  # noqa: ANN001
                return next(batches)

        parent = FakeParent()
        assert diag._app_tree_cpu(parent) == 0.0  # primes `primed`
        assert diag._app_tree_cpu(parent) == 33.0  # reads cached `primed`, not `fresh`
        assert primed.calls == 2
        assert fresh.calls == 0  # the fresh duplicate-pid object is never sampled

    def test_tree_cpu_drops_dead_children(self, monkeypatch) -> None:
        from autoptz.engine.runtime import diagnostics as diag

        monkeypatch.setattr(diag, "_CHILD_PROCS", {})

        class FakeChild:
            def __init__(self, pid: int) -> None:
                self.pid = pid

            def cpu_percent(self, interval=None):  # noqa: ANN001
                return 50.0

        child = FakeChild(7)
        batches = iter([[child], []])

        class FakeParent:
            def children(self, recursive=False):  # noqa: ANN001
                return next(batches)

        parent = FakeParent()
        diag._app_tree_cpu(parent)  # track + prime pid 7
        assert 7 in diag._CHILD_PROCS
        diag._app_tree_cpu(parent)  # pid 7 gone → pruned
        assert 7 not in diag._CHILD_PROCS

    def test_app_cpu_percent_counts_a_busy_child(self, monkeypatch) -> None:
        import multiprocessing as mp
        import time

        psutil = pytest.importorskip("psutil")
        from autoptz.engine.runtime import diagnostics as diag

        self._reset(monkeypatch)
        ncpu = psutil.cpu_count(logical=True) or 1
        one_core_share = 100.0 / ncpu

        proc = mp.get_context().Process(target=_burn_cpu, daemon=True)
        proc.start()
        try:
            diag.system_metrics()  # prime main + discover/prime the child (child → 0)
            time.sleep(1.2)  # exceed the 1 s cache TTL; let the child pin a core
            metrics = diag.system_metrics()  # recompute → child CPU now included
            assert metrics["available"]
            # Without the fix the parent is idle (~0%); with it, ≈ one core's share.
            assert metrics["app_cpu_percent"] > 0.4 * one_core_share, metrics["app_cpu_percent"]
        finally:
            proc.terminate()
            proc.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# EngineClient.setLogLevel
# ─────────────────────────────────────────────────────────────────────────────


class TestSetLogLevel:
    def test_adjusts_root_logger(self, qapp) -> None:
        from autoptz.ui.engine_client import EngineClient

        root = logging.getLogger()
        prev = root.level
        try:
            client = EngineClient()
            client.setLogLevel("DEBUG")
            assert root.level == logging.DEBUG
            client.setLogLevel("INFO")
            assert root.level == logging.INFO
        finally:
            root.setLevel(prev)

    def test_case_insensitive(self, qapp) -> None:
        from autoptz.ui.engine_client import EngineClient

        root = logging.getLogger()
        prev = root.level
        try:
            EngineClient().setLogLevel("debug")
            assert root.level == logging.DEBUG
        finally:
            root.setLevel(prev)

    def test_unknown_level_is_safe_noop(self, qapp) -> None:
        from autoptz.ui.engine_client import EngineClient

        root = logging.getLogger()
        prev = root.level
        root.setLevel(logging.WARNING)
        try:
            EngineClient().setLogLevel("NONSENSE")  # must not raise
            assert root.level == logging.WARNING  # unchanged
        finally:
            root.setLevel(prev)

    def test_adjusts_handler_threshold(self, qapp) -> None:
        from autoptz.ui.engine_client import EngineClient
        from autoptz.ui.log_bridge import LogListModel, QtLogHandler

        model = LogListModel(capacity=10)
        handler = QtLogHandler(model, level=logging.INFO)
        client = EngineClient()
        client.set_log_bridge(model, handler)
        root = logging.getLogger()
        prev = root.level
        try:
            client.setLogLevel("DEBUG")
            assert handler.level == logging.DEBUG
        finally:
            root.setLevel(prev)


# ─────────────────────────────────────────────────────────────────────────────
# Log copy / export
# ─────────────────────────────────────────────────────────────────────────────


class TestLogExport:
    def _bridged_client(self, qapp):
        from autoptz.ui.engine_client import EngineClient
        from autoptz.ui.log_bridge import LogListModel, QtLogHandler

        model = LogListModel(capacity=50)
        handler = QtLogHandler(model)
        client = EngineClient()
        client.set_log_bridge(model, handler)
        # Append rows directly (avoid depending on a running event loop).
        model.appendRow("INFO", "engine.cam", "camera started", "00:00:01.000")
        model.appendRow("ERROR", "engine.cam", "read failed", "00:00:02.000")
        return client, model

    def test_dump_text_contains_all_rows(self, qapp) -> None:
        from autoptz.ui.log_bridge import LogListModel

        model = LogListModel(capacity=10)
        model.appendRow("INFO", "a.b", "first", "00:00:00.000")
        model.appendRow("WARNING", "a.b", "second", "00:00:00.500")
        text = model.dump_text()
        assert "first" in text
        assert "second" in text
        assert "WARNING" in text
        assert len(text.splitlines()) == 2

    def test_copy_returns_full_buffer(self, qapp) -> None:
        client, _model = self._bridged_client(qapp)
        text = client.copyLogsToClipboard()
        assert "camera started" in text
        assert "read failed" in text

    def test_copy_without_bridge_returns_empty(self, qapp) -> None:
        from autoptz.ui.engine_client import EngineClient

        assert EngineClient().copyLogsToClipboard() == ""

    def test_export_writes_file(self, qapp, tmp_path) -> None:
        client, _model = self._bridged_client(qapp)
        target = tmp_path / "logs.txt"
        ok = client.exportLogs(str(target))
        assert ok is True
        contents = target.read_text(encoding="utf-8")
        assert "camera started" in contents
        assert "read failed" in contents

    def test_export_accepts_file_url(self, qapp, tmp_path) -> None:
        client, _model = self._bridged_client(qapp)
        target = tmp_path / "via_url.log"
        ok = client.exportLogs(target.as_uri())
        assert ok is True
        assert target.exists()

    def test_export_bad_path_returns_false(self, qapp) -> None:
        client, _model = self._bridged_client(qapp)
        # A path under a non-existent directory cannot be written.
        ok = client.exportLogs("/no/such/dir/AUTOPTZ/logs.txt")
        assert ok is False


# ─────────────────────────────────────────────────────────────────────────────
# _runtime_services: detector row accel verdict (A-2)
# ─────────────────────────────────────────────────────────────────────────────


class TestRuntimeServicesAccelVerdict:
    """Detector row in _runtime_services reflects the cached accel summary/verdict.

    Uses a fake pool that pre-exposes detector_accel_summary/verdict so no real
    ONNX session or model file is needed.  Calls _runtime_services() directly
    (no thread, no frame loop).
    """

    def _make_worker_with_pool(self, pool):
        from autoptz.config.models import CameraConfig, SourceConfig
        from autoptz.engine.camera_worker import CameraWorker

        config = CameraConfig(
            id="svctest01accd",
            name="Cam",
            source=SourceConfig(type="usb", address="usb://0"),
        )
        worker = CameraWorker(
            "svctest01accd",
            config,
            lambda m: None,
            frame_source=_FakeSource(),
        )
        worker.set_inference_pool(pool)
        return worker

    def test_detector_row_contains_accel_summary(self) -> None:
        import types

        fake_pool = types.SimpleNamespace(
            detector_model_name="yolo11n.onnx",
            detector_tier="auto",
            detector_ep="CoreMLExecutionProvider",
            detector_error="",
            detector_accel_summary=lambda: "CoreML · fp32 · 1.24× CPU (accelerated: GPU/accelerator is helping)",
            detector_accel_verdict=lambda: "accelerated",
        )
        worker = self._make_worker_with_pool(fake_pool)
        rows = worker._runtime_services()
        det_row = next(r for r in rows if r.key == "detector")
        assert "1.24×" in det_row.detail
        assert "CoreML" in det_row.detail

    def test_detector_row_confidence_is_verdict(self) -> None:
        import types

        fake_pool = types.SimpleNamespace(
            detector_model_name="yolo11n.onnx",
            detector_tier="auto",
            detector_ep="CoreMLExecutionProvider",
            detector_error="",
            detector_accel_summary=lambda: "CoreML · fp32 · 0.98× CPU (no-benefit: accelerator selected but no faster than CPU)",
            detector_accel_verdict=lambda: "no-benefit",
        )
        worker = self._make_worker_with_pool(fake_pool)
        rows = worker._runtime_services()
        det_row = next(r for r in rows if r.key == "detector")
        assert det_row.confidence == "no-benefit"

    def test_detector_row_no_accel_summary_when_not_ready(self) -> None:
        """When pool has not yet measured (summary=""), detail has no '·' suffix."""
        import types

        fake_pool = types.SimpleNamespace(
            detector_model_name="yolo11n.onnx",
            detector_tier="auto",
            detector_ep="CPUExecutionProvider",
            detector_error="",
            detector_accel_summary=lambda: "",
            detector_accel_verdict=lambda: "",
        )
        worker = self._make_worker_with_pool(fake_pool)
        rows = worker._runtime_services()
        det_row = next(r for r in rows if r.key == "detector")
        # Detail should just be the model name, no appended accel text.
        assert det_row.detail == "yolo11n.onnx"
        assert det_row.confidence == ""

    def test_detector_row_no_crash_when_pool_lacks_accel_attrs(self) -> None:
        """getattr guard: a pool without accel methods does not crash the tick."""
        import types

        # Old-style fake pool without accel attributes.
        fake_pool = types.SimpleNamespace(
            detector_model_name="yolo11n.onnx",
            detector_tier="auto",
            detector_ep="CPUExecutionProvider",
            detector_error="",
            # no detector_accel_summary / detector_accel_verdict
        )
        worker = self._make_worker_with_pool(fake_pool)
        rows = worker._runtime_services()  # must not raise
        det_row = next(r for r in rows if r.key == "detector")
        assert det_row.confidence == ""

    def test_face_row_failed_when_recognizer_unavailable(self) -> None:
        """A built-but-disabled recognizer must not look active in diagnostics."""
        import types

        worker = self._make_worker_with_pool(pool=None)
        worker._face = types.SimpleNamespace(
            recognizer=types.SimpleNamespace(
                available=False,
                last_error="ImportError: simulated insightface failure",
            ),
            service=object(),
        )

        services = worker._runtime_services()
        face = next(row for row in services if row.key == "face")
        assert face.enabled is True
        assert face.active is False
        assert face.state == "failed"
        assert "insightface failure" in face.detail

        timings = worker._stage_timings()
        stage = next(row for row in timings if row.key == "face")
        assert stage.status == "failed"
        assert "insightface failure" in stage.detail
