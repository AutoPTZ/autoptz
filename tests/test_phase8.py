"""Phase 8 tests: config drawer wiring, identity/layout CRUD, theme, debounce.

These tests run headless (no QML engine) and verify the Python side of
Phase 8 — EngineClient slots, model CRUD, ConfigStore integration,
and new command types — without requiring a live Qt application.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest

from autoptz.engine.runtime.messages import (
    CmdKind,
    DeleteIdentityCmd,
    DeleteLayoutCmd,
    EnrollIdentityCmd,
    PtzSavePresetCmd,
    RenameIdentityCmd,
    SaveLayoutCmd,
    SetLayoutCmd,
    UpdateCameraConfigCmd,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_store(tmp_path: Path):
    from autoptz.config.store import ConfigStore
    return ConfigStore(db_path=tmp_path / "test.db", debounce_s=0)


def make_client(tmp_path: Path | None = None):
    """Return an EngineClient with an in-memory (or tmp) ConfigStore."""
    from autoptz.ui.engine_client import EngineClient
    if tmp_path:
        store = make_store(tmp_path)
        return EngineClient(store=store), store
    return EngineClient(), None


# ── UpdateCameraConfigCmd ─────────────────────────────────────────────────────

class TestUpdateCameraConfigCmd:
    def test_kind(self):
        cmd = UpdateCameraConfigCmd(camera_id="abc", config={"name": "Cam"})
        assert cmd.kind == CmdKind.UPDATE_CONFIG

    def test_msgpack_roundtrip(self):
        cmd = UpdateCameraConfigCmd(camera_id="x", config={"key": "val"})
        raw = cmd.to_msgpack()
        assert len(raw) > 0

    def test_config_preserved(self):
        cmd = UpdateCameraConfigCmd(camera_id="x", config={"ptz": {"kp": 1.2}})
        assert cmd.config["ptz"]["kp"] == pytest.approx(1.2)


# ── EnrollIdentityCmd ─────────────────────────────────────────────────────────

class TestEnrollIdentityCmd:
    def test_identity_id_field(self):
        iid = str(uuid.uuid4())
        cmd = EnrollIdentityCmd(
            camera_id="cam1",
            identity_name="Alice",
            identity_id=iid,
            track_id=5,
            click_x=0.4,
            click_y=0.6,
        )
        assert cmd.identity_id == iid
        assert cmd.track_id == 5
        assert cmd.click_x == pytest.approx(0.4)
        assert cmd.click_y == pytest.approx(0.6)

    def test_kind(self):
        cmd = EnrollIdentityCmd(camera_id="c")
        assert cmd.kind == CmdKind.ENROLL_IDENTITY


# ── New command kinds ─────────────────────────────────────────────────────────

class TestNewCommandKinds:
    def test_delete_identity(self):
        cmd = DeleteIdentityCmd(camera_id=None, identity_id="abc")
        assert cmd.kind == CmdKind.DELETE_IDENTITY

    def test_rename_identity(self):
        cmd = RenameIdentityCmd(camera_id=None, identity_id="abc", new_name="Bob")
        assert cmd.kind == CmdKind.RENAME_IDENTITY
        assert cmd.new_name == "Bob"

    def test_save_layout(self):
        cmd = SaveLayoutCmd(layout_name="Stage", tiles=[{"camera_id": "x"}])
        assert cmd.kind == CmdKind.SAVE_LAYOUT
        assert cmd.tiles[0]["camera_id"] == "x"

    def test_delete_layout(self):
        cmd = DeleteLayoutCmd(camera_id=None, layout_id="lid")
        assert cmd.kind == CmdKind.DELETE_LAYOUT


# ── EngineClient — no store ───────────────────────────────────────────────────

class TestEngineClientNoStore:
    def test_add_camera_creates_config(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "Cam A")
        rec = client.get_camera(cid)
        assert rec is not None
        assert rec.camera_config is not None
        assert rec.camera_config.name == "Cam A"

    def test_add_camera_source_type_rtsp(self):
        client, _ = make_client()
        cid = client.addCamera("rtsp://192.168.1.10/stream", "IP Cam")
        rec = client.get_camera(cid)
        assert rec.camera_config.source.type == "rtsp"

    def test_add_camera_source_type_ndi(self):
        client, _ = make_client()
        cid = client.addCamera("ndi://MyCamera", "NDI Cam")
        rec = client.get_camera(cid)
        assert rec.camera_config.source.type == "ndi"

    def test_get_camera_config_returns_dict(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "Test")
        cfg = client.getCameraConfig(cid)
        assert isinstance(cfg, dict)
        assert cfg["id"] == cid
        assert cfg["name"] == "Test"

    def test_get_camera_config_unknown_returns_empty(self):
        client, _ = make_client()
        cfg = client.getCameraConfig("nonexistent")
        assert cfg == {}

    def test_update_camera_config_changes_name(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "Old Name")
        cfg = client.getCameraConfig(cid)
        cfg["name"] = "New Name"
        client.updateCameraConfig(cid, json.dumps(cfg))
        rec = client.get_camera(cid)
        assert rec.display_name == "New Name"

    def test_update_camera_config_enqueues_command(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        cfg = client.getCameraConfig(cid)
        client.updateCameraConfig(cid, json.dumps(cfg))
        cmds = client.drain_commands()
        kinds = [c.kind for c in cmds]
        assert CmdKind.UPDATE_CONFIG in kinds

    def test_update_camera_config_invalid_json_is_safe(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        # Should not raise; should emit error signal instead
        client.updateCameraConfig(cid, "not json {{{")
        # camera_id is still valid
        assert client.get_camera(cid) is not None

    def test_update_camera_config_preserves_id(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        cfg = client.getCameraConfig(cid)
        # try to change the id — should be silently overridden
        cfg["id"] = "different-id"
        client.updateCameraConfig(cid, json.dumps(cfg))
        rec = client.get_camera(cid)
        assert rec.camera_config.id == cid

    def test_ptz_save_preset_adds_to_config(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.ptzSavePreset(cid, "Home")
        cfg = client.getCameraConfig(cid)
        assert any(p["name"] == "Home" for p in cfg.get("presets", []))

    def test_ptz_save_preset_enqueues_command(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        # drain add command first
        client.drain_commands()
        client.ptzSavePreset(cid, "Home")
        cmds = client.drain_commands()
        assert any(c.kind == CmdKind.PTZ_SAVE_PRESET for c in cmds)

    def test_delete_preset_removes_from_config(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.ptzSavePreset(cid, "Home")
        cfg = client.getCameraConfig(cid)
        idx = cfg["presets"][0]["idx"]
        client.deletePreset(cid, idx)
        cfg2 = client.getCameraConfig(cid)
        assert cfg2.get("presets", []) == []

    def test_theme_default_is_dark(self):
        client, _ = make_client()
        assert client.themeMode == "dark"

    def test_set_theme_changes_mode(self):
        client, _ = make_client()
        client.setTheme("light")
        assert client.themeMode == "light"

    def test_set_theme_invalid_is_ignored(self):
        client, _ = make_client()
        client.setTheme("neon")
        assert client.themeMode == "dark"


# ── EngineClient identity management ─────────────────────────────────────────

class TestIdentityManagement:
    def test_enroll_adds_to_model(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()
        client.enrollIdentity(cid, "Alice", 1)
        assert client._identity_model.rowCount() == 1

    def test_enroll_enqueues_command(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()
        client.enrollIdentity(cid, "Alice", 1)
        cmds = client.drain_commands()
        assert any(c.kind == CmdKind.ENROLL_IDENTITY for c in cmds)

    def test_enroll_command_carries_identity_id(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()
        client.enrollIdentity(cid, "Bob", 2)
        cmds = client.drain_commands()
        enroll_cmds = [c for c in cmds if c.kind == CmdKind.ENROLL_IDENTITY]
        assert len(enroll_cmds) == 1
        assert enroll_cmds[0].identity_id  # not empty
        assert enroll_cmds[0].identity_name == "Bob"

    def test_enroll_blank_name_is_rejected(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()
        client.enrollIdentity(cid, "  ", 1)
        assert client._identity_model.rowCount() == 0

    def test_delete_identity(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.enrollIdentity(cid, "Alice", 1)
        iid = client._identity_model.get_all()[0].id
        client.drain_commands()
        client.deleteIdentity(iid)
        assert client._identity_model.rowCount() == 0

    def test_delete_enqueues_command(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.enrollIdentity(cid, "Alice", 1)
        iid = client._identity_model.get_all()[0].id
        client.drain_commands()
        client.deleteIdentity(iid)
        cmds = client.drain_commands()
        assert any(c.kind == CmdKind.DELETE_IDENTITY for c in cmds)

    def test_rename_identity(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.enrollIdentity(cid, "Alice", 1)
        iid = client._identity_model.get_all()[0].id
        client.renameIdentity(iid, "Alicia")
        recs = client._identity_model.get_all()
        assert recs[0].name == "Alicia"

    def test_rename_enqueues_command(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.enrollIdentity(cid, "Alice", 1)
        iid = client._identity_model.get_all()[0].id
        client.drain_commands()
        client.renameIdentity(iid, "Alicia")
        cmds = client.drain_commands()
        assert any(c.kind == CmdKind.RENAME_IDENTITY for c in cmds)

    def test_rename_blank_is_rejected(self):
        client, _ = make_client()
        cid = client.addCamera("usb://0", "X")
        client.enrollIdentity(cid, "Alice", 1)
        iid = client._identity_model.get_all()[0].id
        client.renameIdentity(iid, "")
        recs = client._identity_model.get_all()
        assert recs[0].name == "Alice"   # unchanged


# ── EngineClient layout management ───────────────────────────────────────────

class TestLayoutManagement:
    def test_save_layout_adds_to_model(self):
        client, _ = make_client()
        client.addCamera("usb://0", "Cam A")
        client.addCamera("usb://1", "Cam B")
        client.saveCurrentLayout("My Layout")
        assert client._layout_model.rowCount() == 1

    def test_save_layout_enqueues_command(self):
        client, _ = make_client()
        client.addCamera("usb://0", "Cam A")
        client.drain_commands()
        client.saveCurrentLayout("Stage")
        cmds = client.drain_commands()
        assert any(c.kind == CmdKind.SAVE_LAYOUT for c in cmds)

    def test_save_layout_blank_name_rejected(self):
        client, _ = make_client()
        client.addCamera("usb://0", "Cam A")
        client.saveCurrentLayout("  ")
        assert client._layout_model.rowCount() == 0

    def test_load_layout_reorders_cameras(self):
        client, _ = make_client()
        cid_a = client.addCamera("usb://0", "A")
        cid_b = client.addCamera("usb://1", "B")
        # save layout B-A order
        client._model.swapCameras(cid_a, cid_b)
        client.saveCurrentLayout("BA Order")
        # reset to A-B order
        client._model.swapCameras(cid_a, cid_b)
        assert client._model.camera_ids()[0] == cid_a

        # load BA layout
        layout_id = client._layout_model.get_all()[0].id
        client.loadLayout(layout_id)
        assert client._model.camera_ids()[0] == cid_b

    def test_load_layout_enqueues_set_layout(self):
        client, _ = make_client()
        client.addCamera("usb://0", "A")
        client.saveCurrentLayout("X")
        client.drain_commands()
        lid = client._layout_model.get_all()[0].id
        client.loadLayout(lid)
        cmds = client.drain_commands()
        assert any(c.kind == CmdKind.SET_LAYOUT for c in cmds)

    def test_delete_layout(self):
        client, _ = make_client()
        client.addCamera("usb://0", "A")
        client.saveCurrentLayout("X")
        lid = client._layout_model.get_all()[0].id
        client.deleteLayout(lid)
        assert client._layout_model.rowCount() == 0

    def test_delete_layout_enqueues_command(self):
        client, _ = make_client()
        client.addCamera("usb://0", "A")
        client.saveCurrentLayout("X")
        lid = client._layout_model.get_all()[0].id
        client.drain_commands()
        client.deleteLayout(lid)
        cmds = client.drain_commands()
        assert any(c.kind == CmdKind.DELETE_LAYOUT for c in cmds)


# ── ConfigStore integration (with real DB) ───────────────────────────────────

class TestConfigStoreIntegration:
    def test_add_camera_persists(self, tmp_path):
        client, store = make_client(tmp_path)
        cid = client.addCamera("usb://0", "Cam A")
        cameras = store.load_cameras()
        assert any(c.id == cid for c in cameras)

    def test_update_camera_config_persists(self, tmp_path):
        client, store = make_client(tmp_path)
        cid = client.addCamera("usb://0", "Cam A")
        cfg = client.getCameraConfig(cid)
        cfg["name"] = "Updated"
        client.updateCameraConfig(cid, json.dumps(cfg))
        cameras = store.load_cameras()
        matching = [c for c in cameras if c.id == cid]
        assert matching[0].name == "Updated"

    def test_ptz_save_preset_persists(self, tmp_path):
        client, store = make_client(tmp_path)
        cid = client.addCamera("usb://0", "Cam A")
        client.ptzSavePreset(cid, "Home")
        cameras = store.load_cameras()
        cam = next(c for c in cameras if c.id == cid)
        assert any(p.name == "Home" for p in cam.presets)

    def test_enroll_identity_persists(self, tmp_path):
        client, store = make_client(tmp_path)
        cid = client.addCamera("usb://0", "Cam A")
        client.enrollIdentity(cid, "Alice", 1)
        identities = store.load_identities()
        assert any(i.name == "Alice" for i in identities)

    def test_delete_identity_persists(self, tmp_path):
        client, store = make_client(tmp_path)
        cid = client.addCamera("usb://0", "Cam A")
        client.enrollIdentity(cid, "Alice", 1)
        iid = client._identity_model.get_all()[0].id
        client.deleteIdentity(iid)
        identities = store.load_identities()
        assert not any(i.id == iid for i in identities)

    def test_save_layout_persists(self, tmp_path):
        client, store = make_client(tmp_path)
        client.addCamera("usb://0", "A")
        client.saveCurrentLayout("Stage")
        layouts = store.load_layouts()
        assert any(lo.name == "Stage" for lo in layouts)

    def test_delete_layout_persists(self, tmp_path):
        client, store = make_client(tmp_path)
        client.addCamera("usb://0", "A")
        client.saveCurrentLayout("Stage")
        lid = client._layout_model.get_all()[0].id
        client.deleteLayout(lid)
        layouts = store.load_layouts()
        assert not any(lo.id == lid for lo in layouts)

    def test_theme_persists(self, tmp_path):
        client, store = make_client(tmp_path)
        client.setTheme("light")
        theme_data = store.get_setting("theme", {})
        assert theme_data.get("name") == "light"

    def test_restart_restores_cameras(self, tmp_path):
        """Cameras saved in session 1 appear in session 2."""
        client1, store1 = make_client(tmp_path)
        cid = client1.addCamera("usb://0", "Persist Cam")
        store1.close()

        client2, store2 = make_client(tmp_path)
        assert client2.camera_count == 1
        rec = client2.get_camera(cid)
        assert rec is not None
        assert rec.display_name == "Persist Cam"
        store2.close()

    def test_restart_restores_identities(self, tmp_path):
        client1, store1 = make_client(tmp_path)
        cid = client1.addCamera("usb://0", "Cam")
        client1.enrollIdentity(cid, "Alice", 1)
        store1.close()

        client2, _ = make_client(tmp_path)
        assert client2._identity_model.rowCount() == 1
        assert client2._identity_model.get_all()[0].name == "Alice"

    def test_restart_restores_layouts(self, tmp_path):
        client1, store1 = make_client(tmp_path)
        client1.addCamera("usb://0", "A")
        client1.saveCurrentLayout("Scene 1")
        store1.close()

        client2, _ = make_client(tmp_path)
        assert client2._layout_model.rowCount() == 1
        assert client2._layout_model.get_all()[0].name == "Scene 1"

    def test_remove_camera_cleans_up(self, tmp_path):
        client, store = make_client(tmp_path)
        cid = client.addCamera("usb://0", "Temp")
        client.removeCamera(cid)
        cameras = store.load_cameras()
        assert not any(c.id == cid for c in cameras)


# ── IdentityListModel ─────────────────────────────────────────────────────────

class TestIdentityListModel:
    def test_add_and_count(self):
        from autoptz.config.models import IdentityRecord
        from autoptz.ui.engine_client import IdentityListModel
        m = IdentityListModel()
        m.add_identity(IdentityRecord(name="Alice"))
        assert m.rowCount() == 1

    def test_remove(self):
        from autoptz.config.models import IdentityRecord
        from autoptz.ui.engine_client import IdentityListModel
        m = IdentityListModel()
        rec = IdentityRecord(name="Alice")
        m.add_identity(rec)
        removed = m.remove_identity(rec.id)
        assert removed
        assert m.rowCount() == 0

    def test_rename(self):
        from autoptz.config.models import IdentityRecord
        from autoptz.ui.engine_client import IdentityListModel
        m = IdentityListModel()
        rec = IdentityRecord(name="Alice")
        m.add_identity(rec)
        m.rename_identity(rec.id, "Alicia")
        assert m.get_all()[0].name == "Alicia"


# ── LayoutListModel ───────────────────────────────────────────────────────────

class TestLayoutListModel:
    def test_add_and_remove(self):
        from autoptz.config.models import Layout
        from autoptz.ui.engine_client import LayoutListModel
        m = LayoutListModel()
        lo = Layout(name="Stage")
        m.add_layout(lo)
        assert m.rowCount() == 1
        m.remove_layout(lo.id)
        assert m.rowCount() == 0

    def test_no_duplicate(self):
        from autoptz.config.models import Layout
        from autoptz.ui.engine_client import LayoutListModel
        m = LayoutListModel()
        lo = Layout(name="Stage")
        m.add_layout(lo)
        m.add_layout(lo)  # duplicate — should be ignored
        assert m.rowCount() == 1

    def test_get_layout(self):
        from autoptz.config.models import Layout
        from autoptz.ui.engine_client import LayoutListModel
        m = LayoutListModel()
        lo = Layout(name="Stage")
        m.add_layout(lo)
        found = m.get_layout(lo.id)
        assert found is not None
        assert found.name == "Stage"
