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
import sys
import threading
import time

import numpy as np
import PySide6  # noqa: F401
import pytest

# ── one QCoreApplication for the whole module ─────────────────────────────────


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtCore import QCoreApplication

    existing = QCoreApplication.instance()
    if existing is not None:
        yield existing
        return
    app = QCoreApplication(sys.argv[:1])
    yield app


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
        assert {"reid", "pose", "face"}.issubset(keys)
        for row in rows:
            assert row["source"]
            assert row["path"]
            assert row["network"]


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
    def test_latency_ms_becomes_positive(self, qapp) -> None:
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
            deadline = time.monotonic() + 3.0
            best = 0.0
            while time.monotonic() < deadline:
                with lock:
                    if received:
                        best = max(best, received[-1].latency_ms)
                if best > 0.0:
                    break
                time.sleep(0.02)
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
