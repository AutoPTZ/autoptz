"""W1 (app-wiring slice) tests.

Covers the Python app-wiring contracts the UI relies on:
- EngineClient.restartEngine() — stop then start.
- EngineClient.getSetting/setSetting round-trip through ConfigStore.
- EngineClient.scanUSBCameras() shape + Continuity label + in_use de-dup.
- SourceConfig.unique_id populated from the scan cache by addCamera().
- log_bridge: a logging record appends a row to LogListModel.

All tests use QCoreApplication (no display) so they run in CI.
"""

from __future__ import annotations

import PySide6  # noqa: F401
import pytest


@pytest.fixture()
def store(tmp_path):
    from autoptz.config.store import ConfigStore

    s = ConfigStore(db_path=tmp_path / "w1.db", debounce_s=0.0)
    yield s
    s.close()


def _client(store=None):
    from autoptz.ui.engine_client import EngineClient

    return EngineClient(store=store)


# ── restartEngine ─────────────────────────────────────────────────────────────


class _FakeSupervisor:
    active_ep = "CPU"

    def __init__(self) -> None:
        self.events: list[str] = []
        self.is_running = False

    def start(self) -> None:
        self.events.append("start")
        self.is_running = True

    def stop(self) -> None:
        self.events.append("stop")
        self.is_running = False


class TestRestartEngine:
    def test_restart_stops_then_starts(self, qapp) -> None:
        client = _client()
        sup = _FakeSupervisor()
        client.set_supervisor(sup)

        client.startEngine()
        assert client.engineRunning is True
        assert sup.events == ["start"]

        client.restartEngine()
        # restart = stop then start
        assert sup.events == ["start", "stop", "start"]
        assert client.engineRunning is True

    def test_restart_from_stopped_just_starts(self, qapp) -> None:
        client = _client()
        sup = _FakeSupervisor()
        client.set_supervisor(sup)

        client.restartEngine()  # was never running
        assert sup.events == ["start"]
        assert client.engineRunning is True


# ── getSetting / setSetting ───────────────────────────────────────────────────


class TestSettingsRoundTrip:
    def test_round_trip(self, qapp, store) -> None:
        client = _client(store=store)
        client.setSetting("overrideCols", 4)
        assert client.getSetting("overrideCols", 0) == 4

    def test_default_when_missing(self, qapp, store) -> None:
        client = _client(store=store)
        assert client.getSetting("does_not_exist", "fallback") == "fallback"

    def test_complex_value(self, qapp, store) -> None:
        client = _client(store=store)
        geom = {"x": 10, "y": 20, "w": 800, "h": 600}
        client.setSetting("window_geometry", geom)
        assert client.getSetting("window_geometry", {}) == geom

    def test_no_store_returns_default(self, qapp) -> None:
        client = _client(store=None)
        assert client.getSetting("k", 99) == 99
        client.setSetting("k", 1)  # must not raise


class TestOptionalComponentState:
    def test_ignore_state_round_trips_per_component(self, qapp, store) -> None:
        client = _client(store=store)
        client.setOptionalComponentIgnored("reid", True)
        ignored = {row["key"]: row["ignored"] for row in client.optionalComponents()}
        assert ignored["reid"] is True
        assert ignored.get("pose", False) is False

        client.setOptionalComponentIgnored("reid", False)
        ignored = {row["key"]: row["ignored"] for row in client.optionalComponents()}
        assert ignored["reid"] is False


# ── scanUSBCameras ────────────────────────────────────────────────────────────


def _patch_enumerate(monkeypatch, devices):
    import autoptz.engine.discovery.usb as usb_mod

    monkeypatch.setattr(usb_mod, "enumerate_cameras", lambda: devices, raising=False)


class TestScanUSBCameras:
    def test_shape_and_continuity_label(self, qapp, monkeypatch) -> None:
        _patch_enumerate(
            monkeypatch,
            [
                {
                    "name": "FaceTime HD",
                    "unique_id": "uid-builtin",
                    "index": 0,
                    "is_continuity": False,
                },
                {"name": "iPhone", "unique_id": "uid-phone", "index": 1, "is_continuity": True},
            ],
        )
        client = _client()
        rows = client.scanUSBCameras()

        assert len(rows) == 2
        for row in rows:
            assert set(row.keys()) == {
                "name",
                "uri",
                "unique_id",
                "in_use",
                "is_continuity",
                "source_label",
            }

        assert rows[0] == {
            "name": "FaceTime HD",
            "uri": "usb://0",
            "unique_id": "uid-builtin",
            "in_use": False,
            "is_continuity": False,
            "source_label": "USB",
        }
        # Continuity Camera gets a label + the flag (drives the menu tooltip).
        assert "Continuity Camera" in rows[1]["name"]
        assert rows[1]["uri"] == "usb://1"
        assert rows[1]["in_use"] is False
        assert rows[1]["is_continuity"] is True

    def test_in_use_dedup_by_unique_id(self, qapp, monkeypatch, store) -> None:
        devices = [
            {"name": "Cam A", "unique_id": "uid-A", "index": 0, "is_continuity": False},
            {"name": "Cam B", "unique_id": "uid-B", "index": 1, "is_continuity": False},
        ]
        _patch_enumerate(monkeypatch, devices)
        client = _client(store=store)

        # First scan caches the uri→unique_id map.
        client.scanUSBCameras()
        # Add Cam A (usb://0) — its unique_id should be persisted from the cache.
        cam_id = client.addCamera("usb://0", "Cam A")
        rec = client.get_camera(cam_id)
        assert rec is not None
        assert rec.camera_config.source.unique_id == "uid-A"

        # Re-scan: Cam A is now in use, Cam B is not.
        _patch_enumerate(monkeypatch, devices)
        rows = client.scanUSBCameras()
        by_uri = {r["uri"]: r for r in rows}
        assert by_uri["usb://0"]["in_use"] is True
        assert by_uri["usb://1"]["in_use"] is False

    def test_fallback_when_enumerate_unavailable_returns_empty(self, qapp, monkeypatch) -> None:
        """When the enumeration backend can't even be imported, return ``[]``.

        No more phantom "Camera 0-3": offering indices that may not open is the
        source of the out-of-bound errors and operator confusion.
        """
        import autoptz.engine.discovery.usb as usb_mod

        def _boom():
            raise RuntimeError("no backend")

        monkeypatch.setattr(usb_mod, "enumerate_cameras", _boom, raising=False)
        client = _client()
        rows = client.scanUSBCameras()
        assert rows == []

    def test_names_are_plain_strings(self, qapp, monkeypatch) -> None:
        """QML must only ever see plain-string names — never objects/functions."""

        class Weird:
            def __str__(self) -> str:
                return "Weird Cam"

        _patch_enumerate(
            monkeypatch,
            [
                {"name": Weird(), "unique_id": "uid-x", "index": 0, "is_continuity": False},
            ],
        )
        client = _client()
        rows = client.scanUSBCameras()
        assert len(rows) == 1
        assert isinstance(rows[0]["name"], str)
        assert rows[0]["name"] == "Weird Cam"

    def test_empty_enumeration_returns_empty(self, qapp, monkeypatch) -> None:
        """enumerate_cameras returning no real devices → empty scan result."""
        _patch_enumerate(monkeypatch, [])
        client = _client()
        assert client.scanUSBCameras() == []


# ── scanNDISources / ndiAvailable ──────────────────────────────────────────────


class _FakeFinder:
    """Stand-in for cyndilib.finder.Finder using the iter_sources() API."""

    sources = ["STUDIO (CAM 1)", "STUDIO (CAM 2)"]

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def iter_sources(self):
        return list(self.sources)


def _install_fake_cyndilib(monkeypatch) -> None:
    import sys
    import time
    import types

    pkg = types.ModuleType("cyndilib")
    finder_mod = types.ModuleType("cyndilib.finder")
    finder_mod.Finder = _FakeFinder
    monkeypatch.setitem(sys.modules, "cyndilib", pkg)
    monkeypatch.setitem(sys.modules, "cyndilib.finder", finder_mod)
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)  # skip the 2s settle


class TestScanNDISources:
    def test_ndi_available_false_without_cyndilib(self, qapp, monkeypatch) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        assert _client().ndiAvailable() is False

    def test_ndi_available_true_when_spec_found(self, qapp, monkeypatch) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
        assert _client().ndiAvailable() is True

    def test_scan_returns_empty_without_cyndilib(self, qapp, monkeypatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _no_cyndilib(name, *a, **k):
            if name.startswith("cyndilib"):
                raise ImportError("no cyndilib")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_cyndilib)
        assert _client().scanNDISources() == []

    def test_scan_parses_iter_sources(self, qapp, monkeypatch) -> None:
        _install_fake_cyndilib(monkeypatch)
        rows = _client().scanNDISources()
        assert rows == [
            {"name": "STUDIO (CAM 1)", "uri": "ndi://STUDIO (CAM 1)"},
            {"name": "STUDIO (CAM 2)", "uri": "ndi://STUDIO (CAM 2)"},
        ]

    def test_scan_dedupes_repeated_names(self, qapp, monkeypatch) -> None:
        _install_fake_cyndilib(monkeypatch)
        monkeypatch.setattr(_FakeFinder, "sources", ["A", "A", "B"])
        rows = _client().scanNDISources()
        assert [r["name"] for r in rows] == ["A", "B"]


# ── addCamera unique_id population ─────────────────────────────────────────────


class TestAddCameraUniqueId:
    def test_non_usb_has_no_unique_id(self, qapp) -> None:
        client = _client()
        cam_id = client.addCamera("rtsp://host/stream", "RTSP Cam")
        rec = client.get_camera(cam_id)
        assert rec.camera_config.source.type == "rtsp"
        assert rec.camera_config.source.unique_id is None

    def test_usb_without_scan_cache_is_none(self, qapp) -> None:
        client = _client()
        cam_id = client.addCamera("usb://2", "Unknown USB")
        rec = client.get_camera(cam_id)
        assert rec.camera_config.source.type == "usb"
        assert rec.camera_config.source.unique_id is None


# ── SourceConfig field ─────────────────────────────────────────────────────────


class TestSourceConfigField:
    def test_unique_id_default_none(self) -> None:
        from autoptz.config.models import SourceConfig

        assert SourceConfig().unique_id is None

    def test_unique_id_round_trips_json(self) -> None:
        from autoptz.config.models import SourceConfig

        src = SourceConfig(type="usb", address="usb://0", unique_id="uid-X")
        restored = SourceConfig.model_validate_json(src.model_dump_json())
        assert restored.unique_id == "uid-X"


# ── log bridge ─────────────────────────────────────────────────────────────────


class TestLogBridge:
    def test_handler_appends_row(self, qapp) -> None:
        import logging

        from autoptz.ui.log_bridge import LogListModel, QtLogHandler

        model = LogListModel(capacity=10)
        handler = QtLogHandler(model)
        logger = logging.getLogger("test.w1.logbridge")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            logger.warning("hello %s", "world")

            rows = model.rows()
            assert len(rows) == 1
            assert rows[0]["level"] == "WARNING"
            assert rows[0]["logger"] == "test.w1.logbridge"
            assert rows[0]["message"] == "hello world"
            assert rows[0]["ts"]  # non-empty timestamp
        finally:
            logger.removeHandler(handler)

    def test_ring_buffer_caps_rows(self, qapp) -> None:
        import logging

        from autoptz.ui.log_bridge import LogListModel, QtLogHandler

        model = LogListModel(capacity=3)
        handler = QtLogHandler(model)
        logger = logging.getLogger("test.w1.ring")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        try:
            for i in range(6):
                logger.info("msg-%d", i)
            rows = model.rows()
            assert len(rows) == 3
            # Oldest evicted; newest retained.
            assert rows[0]["message"] == "msg-3"
            assert rows[-1]["message"] == "msg-5"
        finally:
            logger.removeHandler(handler)

    def test_clear_empties_model(self, qapp) -> None:
        from autoptz.ui.log_bridge import LogListModel

        model = LogListModel(capacity=10)
        model.appendRow("INFO", "a.b", "x", "00:00:00.000")
        assert model.rowCount() == 1
        model.clear()
        assert model.rowCount() == 0
