"""Phase 7: UI engine-client and model tests.

All tests use QCoreApplication (no display needed) so they run cleanly in CI.
Tests that require QGuiApplication / real display are skipped unless the
AUTOPTZ_GUI_TESTS env var is set.
"""
from __future__ import annotations

import os
import sys

import pytest

# Skip the whole module gracefully if PySide6 is not installed
PySide6 = pytest.importorskip("PySide6")

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
    # deliberately do not call app.quit() — other module fixtures still need it


# ── helpers ───────────────────────────────────────────────────────────────────


def _client(qapp):  # noqa: ANN001 — fixture type varies
    from autoptz.ui.engine_client import EngineClient
    return EngineClient()


def _make_telemetry(camera_id: str, fps: float = 25.0):
    from autoptz.engine.runtime.messages import TelemetryMsg
    return TelemetryMsg(camera_id=camera_id, seq=1, fps=fps)


# ─────────────────────────────────────────────────────────────────────────────
# CameraRecord
# ─────────────────────────────────────────────────────────────────────────────


class TestCameraRecord:
    def test_shm_name_derived_from_id(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraRecord
        rec = CameraRecord("abc123def456", "usb://0", "Cam")
        assert rec.shm_name == "cam_abc123de_preview"

    def test_fps_zero_without_telemetry(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraRecord
        rec = CameraRecord("id1", "usb://0", "Cam")
        assert rec.fps == 0.0

    def test_fps_from_telemetry(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraRecord
        rec = CameraRecord("id1", "usb://0", "Cam")
        rec.telemetry = _make_telemetry("id1", fps=30.0)
        assert rec.fps == pytest.approx(30.0)

    def test_health_ok_without_telemetry(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraRecord
        rec = CameraRecord("id1", "usb://0", "Cam")
        assert rec.health == "ok"

    def test_tracks_empty_without_telemetry(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraRecord
        rec = CameraRecord("id1", "usb://0", "Cam")
        assert rec.tracks_as_list() == []

    def test_ptz_defaults_without_telemetry(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraRecord
        rec = CameraRecord("id1", "usb://0", "Cam")
        p = rec.ptz_as_dict()
        assert p["pan"] == 0.0
        assert p["zoom"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CameraListModel
# ─────────────────────────────────────────────────────────────────────────────


class TestCameraListModel:
    def _model(self):
        from autoptz.ui.engine_client import CameraListModel
        return CameraListModel()

    def _rec(self, cid: str, name: str = "Cam"):
        from autoptz.ui.engine_client import CameraRecord
        return CameraRecord(cid, "usb://0", name)

    def test_empty_on_init(self, qapp) -> None:
        m = self._model()
        assert m.rowCount() == 0

    def test_add_camera_increases_row_count(self, qapp) -> None:
        m = self._model()
        m.add_camera(self._rec("c1"))
        assert m.rowCount() == 1

    def test_add_two_cameras(self, qapp) -> None:
        m = self._model()
        m.add_camera(self._rec("c1"))
        m.add_camera(self._rec("c2"))
        assert m.rowCount() == 2

    def test_add_duplicate_is_noop(self, qapp) -> None:
        m = self._model()
        m.add_camera(self._rec("c1"))
        m.add_camera(self._rec("c1"))  # duplicate
        assert m.rowCount() == 1

    def test_remove_camera(self, qapp) -> None:
        m = self._model()
        m.add_camera(self._rec("c1"))
        result = m.remove_camera("c1")
        assert result is True
        assert m.rowCount() == 0

    def test_remove_nonexistent_returns_false(self, qapp) -> None:
        m = self._model()
        assert m.remove_camera("ghost") is False

    def test_data_camera_id_role(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel
        m = self._model()
        m.add_camera(self._rec("cam-xyz", "Test"))
        idx = m.index(0)
        assert m.data(idx, CameraListModel.CameraIdRole) == "cam-xyz"

    def test_data_display_name_role(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel
        m = self._model()
        m.add_camera(self._rec("c1", "Studio Cam"))
        idx = m.index(0)
        assert m.data(idx, CameraListModel.DisplayNameRole) == "Studio Cam"

    def test_data_fps_zero_initially(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel
        m = self._model()
        m.add_camera(self._rec("c1"))
        assert m.data(m.index(0), CameraListModel.FpsRole) == 0.0

    def test_update_telemetry_updates_fps(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel
        m = self._model()
        m.add_camera(self._rec("c1"))
        msg = _make_telemetry("c1", fps=29.97)
        m.update_telemetry(msg)
        assert m.data(m.index(0), CameraListModel.FpsRole) == pytest.approx(29.97)

    def test_update_telemetry_unknown_camera_is_safe(self, qapp) -> None:
        m = self._model()
        msg = _make_telemetry("no-such-id")
        m.update_telemetry(msg)  # must not raise

    def test_swap_cameras(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel
        m = self._model()
        m.add_camera(self._rec("c1"))
        m.add_camera(self._rec("c2"))
        m.swapCameras("c1", "c2")
        assert m.data(m.index(0), CameraListModel.CameraIdRole) == "c2"
        assert m.data(m.index(1), CameraListModel.CameraIdRole) == "c1"

    def test_swap_same_id_is_noop(self, qapp) -> None:
        m = self._model()
        m.add_camera(self._rec("c1"))
        m.swapCameras("c1", "c1")  # must not raise or corrupt
        assert m.rowCount() == 1

    def test_move_camera(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel
        m = self._model()
        m.add_camera(self._rec("c1"))
        m.add_camera(self._rec("c2"))
        m.add_camera(self._rec("c3"))
        m.moveCamera("c3", 0)
        assert m.data(m.index(0), CameraListModel.CameraIdRole) == "c3"

    def test_role_names_present(self, qapp) -> None:
        m = self._model()
        names = m.roleNames()
        assert b"cameraId"       in names.values()
        assert b"displayName"    in names.values()
        assert b"trackingEnabled" in names.values()
        assert b"tracks"         in names.values()
        assert b"fps"            in names.values()

    def test_invalid_index_returns_none(self, qapp) -> None:
        from PySide6.QtCore import QModelIndex

        from autoptz.ui.engine_client import CameraListModel
        m = self._model()
        assert m.data(QModelIndex(), CameraListModel.CameraIdRole) is None

    def test_out_of_range_index_returns_none(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel
        m = self._model()
        idx = m.index(99)
        assert m.data(idx, CameraListModel.CameraIdRole) is None


# ─────────────────────────────────────────────────────────────────────────────
# EngineClient
# ─────────────────────────────────────────────────────────────────────────────


class TestEngineClient:
    def test_zero_cameras_on_init(self, qapp) -> None:
        c = _client(qapp)
        assert c.camera_count == 0

    def test_add_camera_returns_uuid(self, qapp) -> None:
        import uuid
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam 1")
        assert len(cid) == 36  # standard UUID string length
        uuid.UUID(cid)         # raises if invalid

    def test_add_camera_increments_count(self, qapp) -> None:
        c = _client(qapp)
        c.addCamera("usb://0", "Cam 1")
        c.addCamera("rtsp://x", "Cam 2")
        assert c.camera_count == 2

    def test_get_camera_after_add(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://1", "Studio")
        rec = c.get_camera(cid)
        assert rec is not None
        assert rec.camera_id == cid
        assert rec.source_uri == "usb://1"
        assert rec.display_name == "Studio"

    def test_get_camera_unknown_id_returns_none(self, qapp) -> None:
        c = _client(qapp)
        assert c.get_camera("does-not-exist") is None

    def test_remove_camera(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        c.removeCamera(cid)
        assert c.camera_count == 0
        assert c.get_camera(cid) is None

    def test_remove_nonexistent_is_safe(self, qapp) -> None:
        c = _client(qapp)
        c.removeCamera("ghost-id")  # must not raise

    def test_enable_tracking(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        assert c.get_camera(cid).tracking_enabled is False
        c.enableTracking(cid, True)
        assert c.get_camera(cid).tracking_enabled is True

    def test_disable_tracking(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        c.enableTracking(cid, True)
        c.enableTracking(cid, False)
        assert c.get_camera(cid).tracking_enabled is False

    def test_enable_tracking_unknown_id_safe(self, qapp) -> None:
        c = _client(qapp)
        c.enableTracking("ghost", True)  # must not raise

    def test_set_target(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("rtsp://x", "Cam")
        assert c.get_camera(cid).target_track_id is None
        c.setTarget(cid, 42)
        assert c.get_camera(cid).target_track_id == 42

    def test_set_target_unknown_id_safe(self, qapp) -> None:
        c = _client(qapp)
        c.setTarget("ghost", 1)  # must not raise

    def test_clear_target(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("rtsp://x", "Cam")
        c.setTarget(cid, 7)
        c.clearTarget(cid)
        assert c.get_camera(cid).target_track_id is None

    def test_push_telemetry_updates_fps(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        msg = _make_telemetry(cid, fps=30.0)
        c.push_telemetry(msg)
        assert c.get_camera(cid).fps == pytest.approx(30.0)

    def test_push_telemetry_unknown_camera_safe(self, qapp) -> None:
        c = _client(qapp)
        msg = _make_telemetry("no-such-id")
        c.push_telemetry(msg)  # must not raise

    def test_drain_commands_add_camera(self, qapp) -> None:
        from autoptz.engine.runtime.messages import CmdKind
        c = _client(qapp)
        c.addCamera("usb://0", "Cam")
        cmds = c.drain_commands()
        assert len(cmds) == 1
        assert cmds[0].kind == CmdKind.ADD_CAMERA

    def test_drain_commands_clears_queue(self, qapp) -> None:
        c = _client(qapp)
        c.addCamera("usb://0", "Cam")
        c.drain_commands()
        assert c.drain_commands() == []

    def test_drain_includes_enable_tracking(self, qapp) -> None:
        from autoptz.engine.runtime.messages import CmdKind
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        c.enableTracking(cid, True)
        cmds = c.drain_commands()
        kinds = [cmd.kind for cmd in cmds]
        assert CmdKind.ENABLE_TRACKING in kinds

    def test_drain_includes_set_target(self, qapp) -> None:
        from autoptz.engine.runtime.messages import SetTargetCmd
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        c.setTarget(cid, 5)
        cmds = c.drain_commands()
        target_cmds = [cmd for cmd in cmds if isinstance(cmd, SetTargetCmd)]
        assert len(target_cmds) == 1
        assert target_cmds[0].track_id == 5

    def test_drain_includes_remove_camera(self, qapp) -> None:
        from autoptz.engine.runtime.messages import CmdKind
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        c.removeCamera(cid)
        cmds = c.drain_commands()
        kinds = [cmd.kind for cmd in cmds]
        assert CmdKind.REMOVE_CAMERA in kinds

    def test_camera_model_property(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel
        c = _client(qapp)
        assert isinstance(c.cameraModel, CameraListModel)

    def test_camera_model_reflects_add(self, qapp) -> None:
        c = _client(qapp)
        c.addCamera("usb://0", "Cam")
        assert c.cameraModel.rowCount() == 1

    def test_camera_model_reflects_remove(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        c.removeCamera(cid)
        assert c.cameraModel.rowCount() == 0

    def test_ptz_nudge_queues_command(self, qapp) -> None:
        from autoptz.engine.runtime.messages import PtzNudgeCmd
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        c.ptzNudge(cid, 0.5, 0.0, 0.0)
        cmds = c.drain_commands()
        nudge = next((cmd for cmd in cmds if isinstance(cmd, PtzNudgeCmd)), None)
        assert nudge is not None
        assert nudge.pan_speed == pytest.approx(0.5)

    def test_signal_emitted_on_add(self, qapp) -> None:
        c = _client(qapp)
        received = []
        c.cameraAdded.connect(received.append)
        cid = c.addCamera("usb://0", "Cam")
        assert received == [cid]

    def test_signal_emitted_on_remove(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        received = []
        c.cameraRemoved.connect(received.append)
        c.removeCamera(cid)
        assert received == [cid]

    def test_telemetry_signal_emitted(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        received = []
        c.telemetryUpdated.connect(received.append)
        msg = _make_telemetry(cid, fps=15.0)
        c.push_telemetry(msg)
        assert received == [cid]


# ─────────────────────────────────────────────────────────────────────────────
# ShmFrameProvider (no display — only tests that don't need rendering)
# ─────────────────────────────────────────────────────────────────────────────

_gui_tests = pytest.mark.skipif(
    os.environ.get("AUTOPTZ_GUI_TESTS") != "1"
    and not (sys.platform == "darwin"),
    reason="requires display or AUTOPTZ_GUI_TESTS=1",
)


class TestShmFrameProvider:
    @pytest.fixture(scope="class")
    def guiapp(self):
        from PySide6.QtGui import QGuiApplication
        existing = QGuiApplication.instance()
        if existing is not None:
            yield existing
            return
        app = QGuiApplication(sys.argv[:1])
        yield app

    @_gui_tests
    def test_placeholder_returned_for_unknown_camera(self, guiapp) -> None:
        from PySide6.QtCore import QSize

        from autoptz.ui.providers import ShmFrameProvider
        p = ShmFrameProvider()
        img = p.requestImage("no-such-cam", QSize(), QSize(640, 360))
        assert not img.isNull()

    @_gui_tests
    def test_reads_frame_from_shm_writer(self, guiapp) -> None:
        import numpy as np
        from PySide6.QtCore import QSize

        from autoptz.engine.runtime.shm import ShmWriter
        from autoptz.ui.providers import ShmFrameProvider

        shm_name = "test_prov_cam"
        h, w = 360, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = [0, 128, 255]  # distinctive BGR color

        p = ShmFrameProvider()
        with ShmWriter(shm_name, h, w) as writer:
            writer.push(frame)
            p.attach("test-cam", shm_name, h, w)
            img = p.requestImage("test-cam", QSize(), QSize(w, h))
        p.detach("test-cam")

        assert not img.isNull()
        assert img.width() == w
        assert img.height() == h

    @_gui_tests
    def test_detach_removes_reader(self, guiapp) -> None:
        from PySide6.QtCore import QSize

        from autoptz.ui.providers import ShmFrameProvider

        p = ShmFrameProvider()
        # Attach to something that doesn't exist — should silently fail
        p.attach("cam-x", "nonexistent_shm_region", 360, 640)
        p.detach("cam-x")
        # After detach, falls back to placeholder
        img = p.requestImage("cam-x", QSize(), QSize())
        assert not img.isNull()
