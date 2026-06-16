"""Unit tests for autoptz.config.models and autoptz.config.store.

All tests use an in-memory or tmp_path SQLite database — no filesystem
side-effects and no hardware needed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from autoptz.config.models import (
    CURRENT_SCHEMA_VERSION,
    AppConfig,
    CameraConfig,
    HardwarePrefs,
    IdentityRecord,
    Layout,
    PanTiltZoomLimits,
    PTZConfig,
    PTZPreset,
    ReconnectConfig,
    SourceConfig,
    TargetConfig,
    ThemeConfig,
    TilePlacement,
    TrackingConfig,
)
from autoptz.config.store import ConfigStore

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path: Path) -> ConfigStore:
    """Fresh ConfigStore backed by a tmp-path SQLite file (debounce disabled)."""
    return ConfigStore(db_path=tmp_path / "test.db", debounce_s=0)


@pytest.fixture
def cam() -> CameraConfig:
    return CameraConfig(
        name="Front Camera",
        source=SourceConfig(type="rtsp", address="rtsp://192.168.1.10/stream1", fps=30.0),
        enabled=True,
        tracking=TrackingConfig(tracker="botsort", detect_interval=3),
        ptz=PTZConfig(backend="ndi", max_pan_speed=0.4),
        presets=[
            PTZPreset(camera_id="placeholder", idx=0, name="Home", pan=0.0, tilt=0.0, zoom=0.0),
        ],
        target=TargetConfig(mode="manual"),
        reconnect=ReconnectConfig(backoff_initial_s=2.0, backoff_max_s=60.0),
    )


# ── Model validation ───────────────────────────────────────────────────────────

class TestModels:
    def test_camera_id_is_uuid(self, cam: CameraConfig) -> None:
        import uuid
        uuid.UUID(cam.id)  # raises if not a valid UUID

    def test_camera_name_blank_raises(self) -> None:
        with pytest.raises(Exception, match="name"):
            CameraConfig(name="   ")

    def test_source_fps_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SourceConfig(fps=0.0)
        with pytest.raises(ValidationError):
            SourceConfig(fps=999.0)

    def test_tracking_detect_interval_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TrackingConfig(detect_interval=0)

    def test_ptz_preset_name_blank_raises(self) -> None:
        with pytest.raises(Exception, match="name"):
            PTZPreset(camera_id="x", idx=0, name="  ")

    def test_layout_name_blank_raises(self) -> None:
        with pytest.raises(Exception, match="name"):
            Layout(name="")

    def test_identity_name_blank_raises(self) -> None:
        with pytest.raises(Exception, match="name"):
            IdentityRecord(name="")

    def test_app_config_defaults(self) -> None:
        cfg = AppConfig()
        assert cfg.schema_version == CURRENT_SCHEMA_VERSION
        assert cfg.cameras == []
        assert isinstance(cfg.theme, ThemeConfig)
        assert isinstance(cfg.hardware, HardwarePrefs)

    def test_with_camera(self) -> None:
        cfg = AppConfig()
        cam = CameraConfig(name="A")
        cfg2 = cfg.with_camera(cam)
        assert len(cfg2.cameras) == 1
        assert cfg2.cameras[0].id == cam.id

    def test_with_camera_replaces_by_id(self, cam: CameraConfig) -> None:
        cfg = AppConfig().with_camera(cam)
        updated = CameraConfig(id=cam.id, name="Updated")
        cfg2 = cfg.with_camera(updated)
        assert len(cfg2.cameras) == 1
        assert cfg2.cameras[0].name == "Updated"

    def test_without_camera(self, cam: CameraConfig) -> None:
        cfg = AppConfig().with_camera(cam)
        cfg2 = cfg.without_camera(cam.id)
        assert cfg2.cameras == []

    def test_pan_tilt_zoom_limits(self) -> None:
        limits = PanTiltZoomLimits(pan_min=-0.5, pan_max=0.5)
        ptz = PTZConfig(soft_limits=limits)
        assert ptz.soft_limits is not None
        assert ptz.soft_limits.pan_min == -0.5

    def test_tile_placement_defaults(self) -> None:
        tile = TilePlacement(camera_id="abc")
        assert tile.x == 0 and tile.visible is True

    def test_hardware_prefs_max_workers_bounds(self) -> None:
        with pytest.raises(ValidationError):
            HardwarePrefs(max_workers=0)
        with pytest.raises(ValidationError):
            HardwarePrefs(max_workers=100)


# ── Store: bootstrap & schema version ─────────────────────────────────────────

class TestStoreBootstrap:
    def test_schema_version_set_on_init(self, store: ConfigStore) -> None:
        assert store._get_schema_version() == CURRENT_SCHEMA_VERSION

    def test_idempotent_bootstrap(self, tmp_path: Path) -> None:
        """Opening the same DB twice must not raise or reset the version."""
        p = tmp_path / "idempotent.db"
        s1 = ConfigStore(db_path=p, debounce_s=0)
        s1.close()
        s2 = ConfigStore(db_path=p, debounce_s=0)
        assert s2._get_schema_version() == CURRENT_SCHEMA_VERSION
        s2.close()

    def test_empty_db_migrated_to_current(self, tmp_path: Path) -> None:
        """A DB with schema_version=0 (simulated old DB) must migrate cleanly."""
        p = tmp_path / "old.db"
        import sqlite3
        conn = sqlite3.connect(str(p))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value JSON NOT NULL)"
        )
        conn.execute(
            "INSERT INTO app_settings(key,value) VALUES ('schema_version','0')"
        )
        conn.commit()
        conn.close()

        store = ConfigStore(db_path=p, debounce_s=0)
        assert store._get_schema_version() == CURRENT_SCHEMA_VERSION
        store.close()


# ── Store: camera CRUD ─────────────────────────────────────────────────────────

class TestCameraStore:
    def test_save_and_reload_camera(self, store: ConfigStore, cam: CameraConfig) -> None:
        """Create → persist → reload must round-trip identity."""
        store.save_camera(cam)
        loaded = store.load_cameras()
        assert len(loaded) == 1
        reloaded = loaded[0]
        assert reloaded.id == cam.id
        assert reloaded.name == cam.name
        assert reloaded.source.address == cam.source.address
        assert reloaded.tracking.detect_interval == cam.tracking.detect_interval
        assert reloaded.ptz.backend == cam.ptz.backend

    def test_simulated_restart(self, tmp_path: Path, cam: CameraConfig) -> None:
        """CameraConfig must survive a store close-and-reopen (simulated restart)."""
        p = tmp_path / "restart.db"
        s1 = ConfigStore(db_path=p, debounce_s=0)
        s1.save_camera(cam)
        s1.close()

        s2 = ConfigStore(db_path=p, debounce_s=0)
        cameras = s2.load_cameras()
        s2.close()

        assert len(cameras) == 1
        assert cameras[0].id == cam.id
        assert cameras[0].name == cam.name
        assert cameras[0].source.fps == cam.source.fps

    def test_upsert_updates_name(self, store: ConfigStore, cam: CameraConfig) -> None:
        store.save_camera(cam)
        updated = CameraConfig(
            id=cam.id, name="Rear Camera", source=cam.source, enabled=cam.enabled,
        )
        store.save_camera(updated)
        cameras = store.load_cameras()
        assert len(cameras) == 1
        assert cameras[0].name == "Rear Camera"

    def test_delete_camera(self, store: ConfigStore, cam: CameraConfig) -> None:
        store.save_camera(cam)
        store.delete_camera(cam.id)
        assert store.load_cameras() == []

    def test_presets_round_trip(self, store: ConfigStore, cam: CameraConfig) -> None:
        store.save_camera(cam)
        loaded = store.load_cameras()[0]
        assert len(loaded.presets) == len(cam.presets)
        assert loaded.presets[0].name == cam.presets[0].name

    def test_presets_replaced_on_upsert(self, store: ConfigStore, cam: CameraConfig) -> None:
        """Re-saving with different presets must replace, not accumulate."""
        store.save_camera(cam)
        new_preset = PTZPreset(camera_id=cam.id, idx=1, name="Stage Left")
        cam2 = CameraConfig(
            id=cam.id, name=cam.name, source=cam.source,
            presets=[new_preset],
        )
        store.save_camera(cam2)
        loaded = store.load_cameras()[0]
        assert len(loaded.presets) == 1
        assert loaded.presets[0].name == "Stage Left"

    def test_multiple_cameras(self, store: ConfigStore) -> None:
        cams = [CameraConfig(name=f"Cam {i}") for i in range(3)]
        for c in cams:
            store.save_camera(c)
        loaded = store.load_cameras()
        assert len(loaded) == 3

    def test_invalid_row_quarantined_not_fatal(
        self, store: ConfigStore, cam: CameraConfig
    ) -> None:
        """A corrupted JSON blob must quarantine the row without crashing load_cameras."""
        store.save_camera(cam)
        # Corrupt a second row directly in the DB
        assert store._conn is not None
        store._conn.execute(
            "INSERT INTO cameras(id,name,config,enabled,updated_at) "
            "VALUES ('bad-id','Bad','{invalid json}',1,'2024-01-01')"
        )
        store._conn.commit()

        cameras = store.load_cameras()
        # The valid row loads; the corrupt one is quarantined
        assert len(cameras) == 1
        assert cameras[0].id == cam.id
        assert len(store.quarantine) == 1


# ── Store: settings ────────────────────────────────────────────────────────────

class TestSettings:
    def test_get_missing_returns_default(self, store: ConfigStore) -> None:
        assert store.get_setting("nonexistent", "fallback") == "fallback"

    def test_set_and_get(self, store: ConfigStore) -> None:
        store.set_setting("foo", {"bar": 42})
        assert store.get_setting("foo") == {"bar": 42}

    def test_overwrite_setting(self, store: ConfigStore) -> None:
        store.set_setting("k", 1)
        store.set_setting("k", 2)
        assert store.get_setting("k") == 2


# ── Store: layout CRUD ─────────────────────────────────────────────────────────

class TestLayoutStore:
    def test_save_and_reload_layout(self, store: ConfigStore, cam: CameraConfig) -> None:
        layout = Layout(
            name="Main Show",
            tiles=[TilePlacement(camera_id=cam.id, x=0, y=0, w=2, h=2)],
        )
        store.save_layout(layout)
        layouts = store.load_layouts()
        assert len(layouts) == 1
        assert layouts[0].name == "Main Show"
        assert layouts[0].tiles[0].camera_id == cam.id

    def test_delete_layout(self, store: ConfigStore) -> None:
        layout = Layout(name="Temp")
        store.save_layout(layout)
        store.delete_layout(layout.id)
        assert store.load_layouts() == []


# ── Store: identity CRUD ───────────────────────────────────────────────────────

class TestIdentityStore:
    def test_save_and_reload_identity(self, store: ConfigStore) -> None:
        identity = IdentityRecord(
            name="Alice",
            embeddings=[b"\x00\x01\x02\x03", b"\xff\xfe\xfd"],
            thumbnail=b"\xde\xad\xbe\xef",
        )
        store.save_identity(identity)
        loaded = store.load_identities()
        assert len(loaded) == 1
        assert loaded[0].name == "Alice"
        assert len(loaded[0].embeddings) == 2
        assert loaded[0].thumbnail == b"\xde\xad\xbe\xef"

    def test_delete_identity_cascades_embeddings(self, store: ConfigStore) -> None:
        identity = IdentityRecord(name="Bob", embeddings=[b"\x01\x02"])
        store.save_identity(identity)
        store.delete_identity(identity.id)
        assert store.load_identities() == []
        assert store._conn is not None
        rows = store._conn.execute("SELECT * FROM identity_embeddings").fetchall()
        assert rows == []


# ── Store: debounced writes ────────────────────────────────────────────────────

class TestDebounce:
    def test_debounced_write_eventually_persists(self, tmp_path: Path, cam: CameraConfig) -> None:
        store = ConfigStore(db_path=tmp_path / "db.db", debounce_s=0.1)
        store.save_camera_debounced(cam)
        # Before debounce fires: may or may not be persisted yet (implementation-dependent)
        time.sleep(0.25)  # well past the 0.1 s window
        cameras = store.load_cameras()
        store.close()
        assert len(cameras) == 1
        assert cameras[0].id == cam.id

    def test_flush_writes_immediately(self, tmp_path: Path, cam: CameraConfig) -> None:
        store = ConfigStore(db_path=tmp_path / "db2.db", debounce_s=10.0)
        store.save_camera_debounced(cam)
        store.flush()
        cameras = store.load_cameras()
        store.close()
        assert len(cameras) == 1

    def test_repeated_debounced_calls_coalesce(self, tmp_path: Path) -> None:
        """Ten rapid saves for the same camera → exactly one row in the DB."""
        store = ConfigStore(db_path=tmp_path / "db3.db", debounce_s=0.15)
        for i in range(10):
            cam = CameraConfig(id="fixed-id", name=f"Version {i}")
            store.save_camera_debounced(cam)
        store.flush()
        cameras = store.load_cameras()
        store.close()
        assert len(cameras) == 1
        assert cameras[0].name == "Version 9"


# ── Store: AppConfig convenience ───────────────────────────────────────────────

class TestAppConfig:
    def test_load_empty_db_returns_defaults(self, store: ConfigStore) -> None:
        cfg = store.load_app_config()
        assert cfg.cameras == []
        assert isinstance(cfg.theme, ThemeConfig)

    def test_save_and_reload_app_config(self, store: ConfigStore, cam: CameraConfig) -> None:
        cfg = AppConfig(
            theme=ThemeConfig(name="light"),
            hardware=HardwarePrefs(model_tier="small"),
            cameras=[cam],
        )
        store.save_app_config(cfg)
        reloaded = store.load_app_config()
        assert reloaded.theme.name == "light"
        assert reloaded.hardware.model_tier == "small"
        assert len(reloaded.cameras) == 1
        assert reloaded.cameras[0].id == cam.id


# ── Store: JSON export / import ────────────────────────────────────────────────

class TestExportImport:
    def test_export_import_equality(self, store: ConfigStore, cam: CameraConfig) -> None:
        """export_show → import_show must round-trip camera equality."""
        layout = Layout(name="Main", tiles=[TilePlacement(camera_id=cam.id)])
        store.save_camera(cam)
        store.save_layout(layout)

        bundle = store.export_show()
        assert bundle["schema_version"] == CURRENT_SCHEMA_VERSION

        # Fresh store — import into it
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            s2 = ConfigStore(db_path=Path(d) / "import.db", debounce_s=0)
            count = s2.import_show(bundle)
            assert count == 1
            cameras2 = s2.load_cameras()
            layouts2 = s2.load_layouts()
            s2.close()

        assert len(cameras2) == 1
        assert cameras2[0].id == cam.id
        assert cameras2[0].name == cam.name
        assert cameras2[0].source.address == cam.source.address
        assert len(layouts2) == 1
        assert layouts2[0].name == "Main"

    def test_export_import_with_identities(self, store: ConfigStore) -> None:
        identity = IdentityRecord(
            name="Carol",
            embeddings=[b"\xAA\xBB\xCC"],
            thumbnail=b"\x01",
        )
        store.save_identity(identity)
        bundle = store.export_show(include_identities=True)
        assert "identities" in bundle
        assert len(bundle["identities"]) == 1

        import tempfile
        with tempfile.TemporaryDirectory() as d:
            s2 = ConfigStore(db_path=Path(d) / "ids.db", debounce_s=0)
            s2.import_show(bundle)
            ids2 = s2.load_identities()
            s2.close()

        assert len(ids2) == 1
        assert ids2[0].name == "Carol"
        assert ids2[0].embeddings[0] == b"\xAA\xBB\xCC"

    def test_import_merge_preserves_existing(
        self, tmp_path: Path, cam: CameraConfig
    ) -> None:
        """merge=True must not delete cameras already in the target store."""
        p = tmp_path / "merge.db"
        store = ConfigStore(db_path=p, debounce_s=0)

        existing = CameraConfig(name="Existing")
        store.save_camera(existing)

        new_cam = CameraConfig(name="Imported")
        bundle = {"schema_version": CURRENT_SCHEMA_VERSION, "cameras": [json.loads(new_cam.model_dump_json())], "layouts": []}
        store.import_show(bundle, merge=True)

        cameras = store.load_cameras()
        store.close()
        ids = {c.id for c in cameras}
        assert existing.id in ids
        assert new_cam.id in ids

    def test_import_invalid_camera_quarantined(
        self, store: ConfigStore, cam: CameraConfig
    ) -> None:
        """Invalid camera blobs in the import bundle must quarantine, not raise."""
        bundle = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "cameras": [
                json.loads(cam.model_dump_json()),
                {"id": "bad", "name": "", "source": "NOT_AN_OBJECT"},  # invalid
            ],
            "layouts": [],
        }
        count = store.import_show(bundle)
        assert count == 1  # only the valid camera imported
        assert len(store.quarantine) >= 1

    def test_export_is_json_serialisable(self, store: ConfigStore, cam: CameraConfig) -> None:
        """export_show() output must serialise to JSON without error."""
        store.save_camera(cam)
        bundle = store.export_show()
        json.dumps(bundle)  # must not raise


# ── Store: event log ───────────────────────────────────────────────────────────

class TestEventLog:
    def test_log_event_stored(self, store: ConfigStore) -> None:
        store.log_event("CAM_LOST", "Connection lost", camera_id="cam-1", level="error")
        assert store._conn is not None
        rows = store._conn.execute("SELECT * FROM events").fetchall()
        assert len(rows) == 1
        assert rows[0]["code"] == "CAM_LOST"
        assert rows[0]["level"] == "error"

    def test_log_event_without_camera(self, store: ConfigStore) -> None:
        store.log_event("STARTUP", "App started")
        assert store._conn is not None
        rows = store._conn.execute("SELECT * FROM events").fetchall()
        assert rows[0]["camera_id"] is None
