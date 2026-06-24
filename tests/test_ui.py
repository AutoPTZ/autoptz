"""UI engine-client and model tests.

All tests use QCoreApplication (no display needed) so they run cleanly in CI.
Widget smoke tests run in an offscreen subprocess so they do not conflict with
the module-level QCoreApplication used by the headless model tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import PySide6  # noqa: F401
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _run_ui_smoke(
    code: str,
    *,
    cwd,
    env,
    timeout: int = 30,
    attempts: int = 3,
) -> subprocess.CompletedProcess:
    """Run a Qt widget-smoke child process, retrying ONLY on a signal crash.

    Headless Qt occasionally segfaults during teardown on macOS CI
    (``returncode == -11``). That is an environment flake, not a test failure:
    a failed assertion in the child exits with code 1 (>= 0), which we never
    retry. Only a negative returncode (killed by a signal) triggers a retry.
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    for _ in range(max(0, attempts - 1)):
        if result.returncode >= 0:
            return result
        # Child was killed by a signal (e.g. SIGSEGV during Qt teardown) — flake; retry.
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    return result


def _client(qapp):  # noqa: ANN001 — fixture type varies
    from autoptz.ui.engine_client import EngineClient

    return EngineClient()


def _make_telemetry(camera_id: str, fps: float = 25.0):
    from autoptz.engine.runtime.messages import TelemetryMsg

    return TelemetryMsg(camera_id=camera_id, seq=1, fps=fps)


class _PreflightErrorSignal:
    def __init__(self, owner):
        self._owner = owner

    def emit(self, message: str) -> None:
        self._owner.errors.append(message)


class _PreflightClient:
    def __init__(self) -> None:
        self.starts = 0
        self.errors: list[str] = []
        self.errorOccurred = _PreflightErrorSignal(self)

    def startEngine(self) -> None:
        self.starts += 1


class _PreflightSignal:
    def __init__(self) -> None:
        self._slots = []

    def connect(self, slot) -> None:
        self._slots.append(slot)

    def disconnect(self, slot) -> None:
        if slot in self._slots:
            self._slots.remove(slot)

    def emit(self, value: bool) -> None:
        for slot in list(self._slots):
            slot(value)


class _PreflightBridge:
    def __init__(self) -> None:
        self.resolved = _PreflightSignal()


class _ExplodingSignal:
    def connect(self, _slot) -> None:
        pass

    def emit(self, _value: bool) -> None:
        raise RuntimeError("emit failed")


class _ExplodingBridge:
    def __init__(self) -> None:
        self.resolved = _ExplodingSignal()


class TestAboutLinks:
    def test_public_profile_links(self, qapp) -> None:
        from autoptz.ui.widgets.dialogs.about import GITHUB_URL, LINKEDIN_URL

        assert GITHUB_URL == "https://github.com/AutoPTZ/autoptz"
        assert LINKEDIN_URL == "https://www.linkedin.com/in/stevenson-chittumuri/"


class TestMacOSCameraPreflight:
    def test_starts_when_camera_access_authorized(self, monkeypatch) -> None:
        import autoptz.ui.app as app_mod

        fake = types.ModuleType("AVFoundation")
        fake.AVMediaTypeVideo = "video"

        class _CaptureDevice:
            @staticmethod
            def authorizationStatusForMediaType_(_media):
                return 3

        fake.AVCaptureDevice = _CaptureDevice
        monkeypatch.setattr(app_mod.sys, "platform", "darwin")
        monkeypatch.setitem(sys.modules, "AVFoundation", fake)
        client = _PreflightClient()

        app_mod._start_engine_after_macos_camera_preflight(client, object())

        assert client.starts == 1
        assert client.errors == []

    def test_reports_denied_without_starting(self, monkeypatch) -> None:
        import autoptz.ui.app as app_mod

        fake = types.ModuleType("AVFoundation")
        fake.AVMediaTypeVideo = "video"

        class _CaptureDevice:
            @staticmethod
            def authorizationStatusForMediaType_(_media):
                return 2

        fake.AVCaptureDevice = _CaptureDevice
        monkeypatch.setattr(app_mod.sys, "platform", "darwin")
        monkeypatch.setitem(sys.modules, "AVFoundation", fake)
        client = _PreflightClient()

        app_mod._start_engine_after_macos_camera_preflight(client, object())

        assert client.starts == 0
        # Message wording differs for packaged ("denied") vs source runs ("from
        # source"), but both must surface a camera-access error and not start.
        assert "Camera access" in client.errors[-1]

    def test_requests_access_then_starts_when_granted(self, monkeypatch) -> None:
        import autoptz.ui.app as app_mod

        bridge = _PreflightBridge()
        fake = types.ModuleType("AVFoundation")
        fake.AVMediaTypeVideo = "video"

        class _CaptureDevice:
            @staticmethod
            def authorizationStatusForMediaType_(_media):
                return 0

            @staticmethod
            def requestAccessForMediaType_completionHandler_(_media, handler):
                handler(True)

        fake.AVCaptureDevice = _CaptureDevice
        monkeypatch.setattr(app_mod.sys, "platform", "darwin")
        monkeypatch.setitem(sys.modules, "AVFoundation", fake)
        client = _PreflightClient()

        app_mod._start_engine_after_macos_camera_preflight(client, bridge)

        assert client.starts == 1
        assert client.errors == []

    def test_permission_callback_exceptions_do_not_escape_to_tcc(self, monkeypatch) -> None:
        import autoptz.ui.app as app_mod

        bridge = _ExplodingBridge()
        fake = types.ModuleType("AVFoundation")
        fake.AVMediaTypeVideo = "video"

        class _CaptureDevice:
            @staticmethod
            def authorizationStatusForMediaType_(_media):
                return 0

            @staticmethod
            def requestAccessForMediaType_completionHandler_(_media, handler):
                handler(True)

        fake.AVCaptureDevice = _CaptureDevice
        monkeypatch.setattr(app_mod.sys, "platform", "darwin")
        monkeypatch.setitem(sys.modules, "AVFoundation", fake)

        app_mod._start_engine_after_macos_camera_preflight(_PreflightClient(), bridge)

    def test_permission_result_is_delivered_on_qt_event_loop(self, monkeypatch, qapp) -> None:
        from PySide6.QtCore import QObject, Signal, Slot

        import autoptz.ui.app as app_mod

        handlers = []
        fake = types.ModuleType("AVFoundation")
        fake.AVMediaTypeVideo = "video"

        class _CaptureDevice:
            @staticmethod
            def authorizationStatusForMediaType_(_media):
                return 0

            @staticmethod
            def requestAccessForMediaType_completionHandler_(_media, handler):
                handlers.append(handler)

        class _QtBridge(QObject):
            resolved = Signal(bool)

            @Slot(bool)
            def resolve(self, granted: bool) -> None:
                self.resolved.emit(bool(granted))

        fake.AVCaptureDevice = _CaptureDevice
        monkeypatch.setattr(app_mod.sys, "platform", "darwin")
        monkeypatch.setitem(sys.modules, "AVFoundation", fake)
        client = _PreflightClient()
        bridge = _QtBridge()

        app_mod._start_engine_after_macos_camera_preflight(client, bridge)
        handlers[0](True)

        assert client.starts == 0
        qapp.processEvents()
        assert client.starts == 1
        assert client.errors == []


class TestCameraTileFramingHelpers:
    def test_framing_snap_threshold(self, qapp) -> None:
        from autoptz.ui.widgets.camera_tile import _snap_center_axis

        assert _snap_center_axis(0.039) == 0.0
        assert _snap_center_axis(-0.04) == 0.0
        assert _snap_center_axis(0.041) == pytest.approx(0.041)


class TestElideKeepingPct:
    """On-video labels must keep the trailing 'NN%' instead of eliding it away."""

    class _FM:
        # 10px per character; elidedText keeps as many chars as fit + an ellipsis.
        def horizontalAdvance(self, s: str) -> int:
            return len(s) * 10

        def elidedText(self, s: str, _mode, px: int) -> str:
            if len(s) * 10 <= px:
                return s
            n = max(0, int(px) // 10 - 1)
            return s[:n] + "…"

    def test_unchanged_when_it_fits(self) -> None:
        from autoptz.ui.widgets.tile_helpers import elide_keeping_pct

        assert elide_keeping_pct(self._FM(), "Target: Al 85%", 1000) == "Target: Al 85%"

    def test_percentage_preserved_when_cramped(self) -> None:
        from autoptz.ui.widgets.tile_helpers import elide_keeping_pct

        out = elide_keeping_pct(self._FM(), "Target: Alexander 85%", 120)
        assert out.endswith("85%")

    def test_plain_elide_without_percentage(self) -> None:
        from autoptz.ui.widgets.tile_helpers import elide_keeping_pct

        out = elide_keeping_pct(self._FM(), "Target: Alexander", 80)
        assert out.endswith("…") and not out.endswith("%")


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

    def test_tracking_status_defaults_without_telemetry(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraRecord

        rec = CameraRecord("id1", "usb://0", "Cam")
        assert rec.tracking_status_as_dict()["state"] == "idle"

    def test_tracking_status_as_dict(self, qapp) -> None:
        from autoptz.engine.runtime.messages import TelemetryMsg, TrackingStatusInfo
        from autoptz.ui.engine_client import CameraRecord

        rec = CameraRecord("id1", "usb://0", "Cam")
        rec.telemetry = TelemetryMsg(
            camera_id="id1",
            seq=0,
            tracking_status=TrackingStatusInfo(
                state="coasting",
                headline="Target lost - holding position",
                action="holding",
                remaining_s=0.7,
                severity="warning",
            ),
        )
        out = rec.tracking_status_as_dict()
        assert out["headline"] == "Target lost - holding position"
        assert out["remaining_s"] == pytest.approx(0.7)

    def test_tracks_emit_name_and_id_not_uuid(self, qapp) -> None:
        # Regression: the bbox/target label must show the display NAME, with the
        # gallery id carried separately as identity_id (not the UUID as the label).
        from autoptz.engine.runtime.messages import BBox, TelemetryMsg, TrackInfo
        from autoptz.ui.engine_client import CameraRecord

        rec = CameraRecord("id1", "usb://0", "Cam")
        rec.telemetry = TelemetryMsg(
            camera_id="id1",
            seq=0,
            width=1000,
            height=500,
            tracks=[
                TrackInfo(
                    track_id=7,
                    bbox=BBox(x1=100, y1=50, x2=300, y2=450),
                    identity="Person 3",
                    identity_id="uuid-abc",
                    confidence=0.8,
                    is_target=True,
                )
            ],
        )
        out = rec.tracks_as_list()
        assert len(out) == 1
        t = out[0]
        assert t["identity"] == "Person 3"  # display name
        assert t["identity_id"] == "uuid-abc"  # stable id for enroll/target
        # bbox normalised to 0..1 by frame dims
        assert t["bbox"]["x1"] == pytest.approx(0.1)
        assert t["bbox"]["y2"] == pytest.approx(0.9)

    def test_tracks_emit_normalized_target_aim(self, qapp) -> None:
        from autoptz.engine.runtime.messages import BBox, TelemetryMsg, TrackInfo
        from autoptz.ui.engine_client import CameraRecord

        rec = CameraRecord("id1", "usb://0", "Cam")
        rec.telemetry = TelemetryMsg(
            camera_id="id1",
            seq=0,
            width=1000,
            height=500,
            tracks=[
                TrackInfo(
                    track_id=7,
                    bbox=BBox(x1=100, y1=50, x2=300, y2=450),
                    is_target=True,
                    aim_x=250,
                    aim_y=125,
                    aim_source="pose",
                )
            ],
        )
        aim = rec.tracks_as_list()[0]["aim"]
        assert aim == {"x": pytest.approx(0.25), "y": pytest.approx(0.25), "source": "pose"}


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
        assert b"cameraId" in names.values()
        assert b"displayName" in names.values()
        assert b"trackingEnabled" in names.values()
        assert b"tracks" in names.values()
        assert b"fps" in names.values()

    def test_resolution_and_dropped_frames_roles_present(self, qapp) -> None:
        """FROZEN role names the QML Camera Info panel binds to."""
        m = self._model()
        names = m.roleNames()
        assert b"resolution" in names.values()
        assert b"droppedFrames" in names.values()

    def test_resolution_role_defaults_empty(self, qapp) -> None:
        from autoptz.ui.engine_client import CameraListModel

        m = self._model()
        m.add_camera(self._rec("c1"))
        assert m.data(m.index(0), CameraListModel.ResolutionRole) == ""
        assert m.data(m.index(0), CameraListModel.DroppedFramesRole) == 0

    def test_update_telemetry_sets_resolution_and_dropped(self, qapp) -> None:
        from autoptz.engine.runtime.messages import TelemetryMsg
        from autoptz.ui.engine_client import CameraListModel

        m = self._model()
        m.add_camera(self._rec("c1"))
        msg = TelemetryMsg(
            camera_id="c1",
            seq=2,
            width=1920,
            height=1080,
            dropped_frames=4,
        )
        m.update_telemetry(msg)
        assert m.data(m.index(0), CameraListModel.ResolutionRole) == "1920x1080"
        assert m.data(m.index(0), CameraListModel.DroppedFramesRole) == 4

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
        uuid.UUID(cid)  # raises if invalid

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

    def test_enable_tracking_emits_tracking_changed(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        seen = []
        c.trackingChanged.connect(seen.append)
        c.enableTracking(cid, True)
        assert seen == [cid]

    def test_disable_tracking(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        c.enableTracking(cid, True)
        c.enableTracking(cid, False)
        assert c.get_camera(cid).tracking_enabled is False

    def test_enable_tracking_unknown_id_safe(self, qapp) -> None:
        c = _client(qapp)
        c.enableTracking("ghost", True)  # must not raise

    def test_feature_overrides_are_session_only_and_reset_on_start(self, qapp) -> None:
        c = _client(qapp)
        c.setFeatureEnabled("pose", False)
        assert c.features()["pose"] is False
        c.resetFeatureOverrides()
        assert all(c.features().values())

    def test_detector_model_tier_persists(self, qapp, tmp_path) -> None:
        from autoptz.config.store import ConfigStore
        from autoptz.ui.engine_client import EngineClient

        db = tmp_path / "cfg.db"
        store = ConfigStore(db_path=db, debounce_s=0)
        c = EngineClient(store=store)
        c.setDetectorModelTier("balanced")
        assert c.getDetectorModelTier() == "balanced"
        store.close()

        store2 = ConfigStore(db_path=db, debounce_s=0)
        c2 = EngineClient(store=store2)
        try:
            assert c2.getDetectorModelTier() == "balanced"
        finally:
            store2.close()

    def test_running_missing_detector_tier_without_auto_download_is_rejected(
        self, qapp, monkeypatch
    ) -> None:
        import autoptz.engine.runtime.models as models

        class FakeManager:
            def app_model_statuses(self):
                return [
                    {"key": "detector_fast", "cached": True},
                    {"key": "detector_balanced", "cached": False},
                ]

        c = _client(qapp)
        c._engine_running = True
        c._detector_model_tier = "fast"
        errors = []
        c.errorOccurred.connect(errors.append)
        monkeypatch.setattr(models, "default_manager", lambda: FakeManager())

        c.setDetectorModelTier("balanced")

        assert c.getDetectorModelTier() == "fast"
        assert errors
        assert "not downloaded" in errors[-1]

    def test_overlay_prediction_toggle_persists(self, qapp, tmp_path) -> None:
        from autoptz.config.store import ConfigStore
        from autoptz.ui.engine_client import EngineClient

        db = tmp_path / "cfg.db"
        store = ConfigStore(db_path=db, debounce_s=0)
        c = EngineClient(store=store)
        assert c.overlays()["prediction"] is False
        c.setOverlay("prediction", True)
        assert c.overlays()["prediction"] is True
        c.setOverlay("unknown", True)
        assert "unknown" not in c.overlays()
        store.close()

        store2 = ConfigStore(db_path=db, debounce_s=0)
        c2 = EngineClient(store=store2)
        try:
            assert c2.overlays()["prediction"] is True
        finally:
            store2.close()

    def test_set_target_fps_enqueues_single_fps_command(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("usb://0", "Cam")
        c.drain_commands()
        c.setTargetFps(cid, 30.0)
        kinds = [cmd.kind.value for cmd in c.drain_commands()]
        assert kinds == ["set_target_fps"]

    def test_set_target(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("rtsp://x", "Cam")
        assert c.get_camera(cid).target_track_id is None
        c.setTarget(cid, 42)
        rec = c.get_camera(cid)
        assert rec.target_track_id == 42
        assert rec.camera_config.target.mode == "manual"
        assert rec.camera_config.target.identity_id is None

    def test_manual_target_clears_identity_target(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("rtsp://x", "Cam")
        c.setTargetIdentity(cid, "person-1")
        c.setTarget(cid, 42)
        rec = c.get_camera(cid)
        assert rec.target_track_id == 42
        assert rec.camera_config.target.mode == "manual"
        assert rec.camera_config.target.identity_id is None

    def test_set_target_unknown_id_safe(self, qapp) -> None:
        c = _client(qapp)
        c.setTarget("ghost", 1)  # must not raise

    def test_identity_target_clears_manual_target(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("rtsp://x", "Cam")
        c.setTarget(cid, 42)
        c.setTargetIdentity(cid, "person-1")
        rec = c.get_camera(cid)
        assert rec.target_track_id is None
        assert rec.camera_config.target.mode == "identity"
        assert rec.camera_config.target.identity_id == "person-1"

    def test_clear_target(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("rtsp://x", "Cam")
        c.setTarget(cid, 7)
        c.clearTarget(cid)
        assert c.get_camera(cid).target_track_id is None

    def test_clear_target_clears_manual_and_identity_target(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("rtsp://x", "Cam")
        c.setTarget(cid, 7)
        c.setTargetIdentity(cid, "person-1")
        c.drain_commands()
        c.clearTarget(cid)
        rec = c.get_camera(cid)
        assert rec.target_track_id is None
        assert rec.camera_config.target.mode == "off"
        assert rec.camera_config.target.identity_id is None
        kinds = [cmd.kind.value for cmd in c.drain_commands()]
        assert "set_target" in kinds
        assert "set_target_identity" in kinds

    def test_clear_target_and_stop_disables_tracking_and_clears_target(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("rtsp://x", "Cam")
        c.setTarget(cid, 7)
        c.enableTracking(cid, True)
        seen_tracking = []
        seen_target = []
        c.trackingChanged.connect(seen_tracking.append)
        c.targetChanged.connect(seen_target.append)
        c.clearTargetAndStop(cid)
        rec = c.get_camera(cid)
        assert rec.tracking_enabled is False
        assert rec.target_track_id is None
        assert rec.camera_config.target.mode == "off"
        assert seen_tracking == [cid]
        assert seen_target == [cid]

    def test_stop_tracking_keeps_target_lock(self, qapp) -> None:
        c = _client(qapp)
        cid = c.addCamera("rtsp://x", "Cam")
        c.setTarget(cid, 7)
        c.enableTracking(cid, True)
        c.enableTracking(cid, False)
        rec = c.get_camera(cid)
        assert rec.tracking_enabled is False
        assert rec.target_track_id == 7

    def test_move_camera_persisted_saves_order(self, qapp, tmp_path) -> None:
        from autoptz.config.store import ConfigStore
        from autoptz.ui.engine_client import EngineClient

        store = ConfigStore(db_path=tmp_path / "cfg.db", debounce_s=0)
        try:
            c = EngineClient(store=store)
            c1 = c.addCamera("usb://0", "A")
            c2 = c.addCamera("usb://1", "B")
            c3 = c.addCamera("usb://2", "C")
            c.moveCameraPersisted(c3, 0)
            assert c.cameraModel.camera_ids() == [c3, c1, c2]
            assert store.get_setting("camera_order") == [c3, c1, c2]
        finally:
            store.close()

        store2 = ConfigStore(db_path=tmp_path / "cfg.db", debounce_s=0)
        try:
            c_loaded = EngineClient(store=store2)
            assert c_loaded.cameraModel.camera_ids() == [c3, c1, c2]
        finally:
            store2.close()

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
# ShmFrameSource — shm → QImage bridge for the Qt Widgets camera tiles
# ─────────────────────────────────────────────────────────────────────────────


class TestShmFrameSource:
    def test_unknown_camera_returns_none(self) -> None:
        from autoptz.ui.frames import ShmFrameSource

        assert ShmFrameSource().latest_qimage("no-such-cam") is None

    def test_self_healing_reads_after_writer_appears(self) -> None:
        """attach() BEFORE the writer exists → None until it appears, then a frame.

        This is the old blank-preview regression: an attach that races ahead of
        the writer's segment must still serve real frames once it shows up.
        """
        import time
        import uuid

        import numpy as np

        from autoptz.engine.runtime.shm import ShmWriter
        from autoptz.ui.frames import ShmFrameSource

        h, w = 16, 24
        cid = "cam-" + uuid.uuid4().hex[:8]
        shm_name = f"fstest_{uuid.uuid4().hex[:8]}"
        src = ShmFrameSource()
        src.attach(cid, shm_name, h, w)
        assert src.latest_qimage(cid) is None  # writer absent → no frame yet

        writer = None
        try:
            writer = ShmWriter(shm_name, h, w)
            writer.push(np.full((h, w, 3), 200, dtype=np.uint8))  # BGR grey
            real = None
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                img = src.latest_qimage(cid)
                if img is not None and img.width() == w and img.height() == h:
                    real = img
                    break
                time.sleep(0.02)
            assert real is not None, "frame source never served the real frame"
            px = real.pixelColor(w // 2, h // 2)
            assert (px.red(), px.green(), px.blue()) == (200, 200, 200)
        finally:
            src.detach(cid)
            if writer is not None:
                writer.close()

    def test_detach_clears_intent(self) -> None:
        from autoptz.ui.frames import ShmFrameSource

        src = ShmFrameSource()
        src.attach("cam-x", "nonexistent_shm_region", 8, 8)
        src.detach("cam-x")
        assert src.latest_qimage("cam-x") is None
        src.detach_all()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Camera wall aspect geometry
# ─────────────────────────────────────────────────────────────────────────────


class TestCameraWallAspectLayout:
    def test_fit_tile_preserves_16x9(self, qapp) -> None:
        from autoptz.ui.widgets.camera_wall import _fit_16x9_tile

        w, h = _fit_16x9_tile(1280, 720, 2, 2)
        assert (w / h) == pytest.approx(16 / 9)

    def test_fit_tile_accounts_for_columns_rows(self, qapp) -> None:
        from autoptz.ui.widgets.camera_wall import _fit_16x9_tile

        w2, h2 = _fit_16x9_tile(1280, 720, 2, 2)
        w3, h3 = _fit_16x9_tile(1280, 720, 3, 2)
        assert w3 < w2
        assert h3 < h2

    def test_fixed_layout_placeholder_count(self, qapp) -> None:
        from autoptz.ui.widgets.camera_wall import _placeholder_count

        assert _placeholder_count("2x2", used=1, cols=2, rows=2) == 3
        assert _placeholder_count("3x2", used=4, cols=3, rows=2) == 2

    def test_auto_layout_has_no_placeholders(self, qapp) -> None:
        from autoptz.ui.widgets.camera_wall import _placeholder_count

        assert _placeholder_count("auto", used=1, cols=2, rows=2) == 0

    def test_drop_index_from_rects_insert_before_after(self, qapp) -> None:
        from autoptz.ui.widgets.camera_wall import _drop_index_from_rects

        order = ["a", "b", "c"]
        rects = {
            "a": (0, 0, 100, 56),
            "b": (110, 0, 100, 56),
            "c": (220, 0, 100, 56),
        }
        assert _drop_index_from_rects(order, "a", 120, 20, rects) == 0
        assert _drop_index_from_rects(order, "a", 205, 20, rects) == 1
        assert _drop_index_from_rects(order, "a", 310, 20, rects) == 2


class TestCameraTileHelpers:
    def test_target_button_label(self, qapp) -> None:
        from autoptz.ui.widgets.camera_tile import _format_target_button_label

        assert _format_target_button_label("Anyone") == "Track: Anyone ▾"
        assert _format_target_button_label("Alice") == "Track: Alice ▾"
        assert _format_target_button_label("ID 4") == "Track: ID 4 ▾"

    def test_context_menu_action_labels(self, qapp) -> None:
        from autoptz.ui.widgets.camera_tile import _context_menu_action_labels

        assert _context_menu_action_labels(
            person=True,
            current_target=False,
            has_target=False,
            tracking=False,
        ) == ["Save Face / Name Person…", "Set Target", "Set Target and Track"]
        assert _context_menu_action_labels(
            person=True,
            current_target=True,
            has_target=True,
            tracking=False,
        ) == ["Save Face / Name Person…", "Track", "Clear"]
        assert _context_menu_action_labels(
            person=True,
            current_target=True,
            has_target=True,
            tracking=True,
        ) == ["Save Face / Name Person…", "Stop Tracking", "Clear"]
        assert _context_menu_action_labels(
            person=False,
            current_target=False,
            has_target=True,
            tracking=False,
        ) == ["Track", "Clear"]
        assert _context_menu_action_labels(
            person=False,
            current_target=False,
            has_target=True,
            tracking=True,
        ) == ["Stop Tracking", "Clear"]

    def test_upper_body_bbox_crops_lower_body(self, qapp) -> None:
        from autoptz.ui.widgets.camera_tile import _upper_body_bbox

        out = _upper_body_bbox({"x1": 0.2, "y1": 0.1, "x2": 0.8, "y2": 0.9})
        assert out["x1"] == pytest.approx(0.2)
        assert out["x2"] == pytest.approx(0.8)
        assert out["y2"] == pytest.approx(0.596)

    def test_norm_bbox_contains(self, qapp) -> None:
        from autoptz.ui.widgets.camera_tile import _norm_bbox_contains

        box = {"x1": 0.2, "y1": 0.1, "x2": 0.8, "y2": 0.9}
        assert _norm_bbox_contains(box, 0.5, 0.5)
        assert not _norm_bbox_contains(box, 0.1, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Widgets shell smoke test (MainWindow + panels)
# ─────────────────────────────────────────────────────────────────────────────


class TestWidgetsShell:
    def test_mainwindow_constructs_and_routes_selection(self, tmp_path) -> None:
        code = f"""
import os
import sys
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from autoptz.config.store import ConfigStore
from autoptz.ui.engine_client import EngineClient
from autoptz.ui.frames import ShmFrameSource
from autoptz.ui.log_bridge import LogListModel
from autoptz.ui.widgets import MainWindow
app = QApplication(sys.argv[:1])
client = EngineClient(store=ConfigStore(db_path=Path({str(tmp_path / "cfg.db")!r}), debounce_s=0))
win = MainWindow(client, log_model=LogListModel(), frame_source=ShmFrameSource())
try:
    assert set(win._docks) == {{"properties", "camera_info", "people", "services", "logs"}}
    cid = client.addCamera("usb://0", "Cam")
    win._on_camera_selected(cid)
    assert win._properties._camera_id == cid
    assert win._camera_info._camera_id == cid
    win._wall._layout = "2x2"
    win._wall.resize(800, 520)
    win._wall._grid_host.resize(800, 450)
    win._wall._reflow()
    assert len(win._wall._empty_slots) >= 3
finally:
    win.close()
"""
        env = dict(os.environ)
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        result = _run_ui_smoke(code, cwd=Path(__file__).resolve().parents[1], env=env, timeout=30)
        assert result.returncode == 0, result.stderr or result.stdout

    def test_status_logs_button_controls_logs_dock(self, tmp_path) -> None:
        code = f"""
import os
import sys
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from autoptz.config.store import ConfigStore
from autoptz.ui.engine_client import EngineClient
from autoptz.ui.frames import ShmFrameSource
from autoptz.ui.log_bridge import LogListModel
from autoptz.ui.widgets import MainWindow
app = QApplication(sys.argv[:1])
client = EngineClient(store=ConfigStore(db_path=Path({str(tmp_path / "cfg.db")!r}), debounce_s=0))
win = MainWindow(client, log_model=LogListModel(), frame_source=ShmFrameSource())
try:
    win.show()
    app.processEvents()
    dock = win._docks["logs"]
    button = win._status._logs_btn
    assert button is not None
    assert dock.isVisible()
    assert button.isChecked()

    button.click()
    app.processEvents()
    assert not dock.isVisible()
    assert not button.isChecked()

    button.click()
    app.processEvents()
    assert dock.isVisible()
    assert button.isChecked()

    dock.hide()
    app.processEvents()
    assert not button.isChecked()
finally:
    win.close()
"""
        env = dict(os.environ)
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        result = _run_ui_smoke(code, cwd=Path(__file__).resolve().parents[1], env=env, timeout=30)
        assert result.returncode == 0, result.stderr or result.stdout

    def test_theme_does_not_strip_mainwindow_desktop_chrome(self, tmp_path) -> None:
        code = f"""
import os
import sys
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from autoptz.config.store import ConfigStore
from autoptz.ui.engine_client import EngineClient
from autoptz.ui.frames import ShmFrameSource
from autoptz.ui.log_bridge import LogListModel
from autoptz.ui.theme import ThemeController
from autoptz.ui.widgets import MainWindow
app = QApplication(sys.argv[:1])
client = EngineClient(store=ConfigStore(db_path=Path({str(tmp_path / "cfg.db")!r}), debounce_s=0))
theme = ThemeController(app, client)
win = MainWindow(client, log_model=LogListModel(), frame_source=ShmFrameSource(), theme=theme)
try:
    win.show()
    app.processEvents()
    flags = win.windowFlags()
    assert not flags & Qt.WindowType.FramelessWindowHint, int(flags)
    assert not flags & Qt.WindowType.NoDropShadowWindowHint, int(flags)
    assert flags & Qt.WindowType.WindowTitleHint, int(flags)
    assert flags & Qt.WindowType.WindowSystemMenuHint, int(flags)
    assert flags & Qt.WindowType.WindowMinimizeButtonHint, int(flags)
    assert flags & Qt.WindowType.WindowMaximizeButtonHint, int(flags)
    assert flags & Qt.WindowType.WindowCloseButtonHint, int(flags)
    assert not win.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
finally:
    win.close()
"""
        env = dict(os.environ)
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        result = _run_ui_smoke(code, cwd=Path(__file__).resolve().parents[1], env=env, timeout=30)
        assert result.returncode == 0, result.stderr or result.stdout

    def test_services_panel_does_not_force_tall_main_window(self, tmp_path) -> None:
        code = f"""
import os
import sys
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from autoptz.config.store import ConfigStore
from autoptz.ui.engine_client import EngineClient
from autoptz.ui.frames import ShmFrameSource
from autoptz.ui.log_bridge import LogListModel
from autoptz.ui.widgets import MainWindow
app = QApplication(sys.argv[:1])
client = EngineClient(store=ConfigStore(db_path=Path({str(tmp_path / "cfg.db")!r}), debounce_s=0))
win = MainWindow(client, log_model=LogListModel(), frame_source=ShmFrameSource())
try:
    win.resize(1320, 820)
    win.show()
    app.processEvents()
    assert win.height() <= 900
    assert win.minimumSizeHint().height() <= 650
finally:
    win.close()
"""
        env = dict(os.environ)
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        result = _run_ui_smoke(code, cwd=Path(__file__).resolve().parents[1], env=env, timeout=30)
        assert result.returncode == 0, result.stderr or result.stdout
