"""SQLite-backed config store with migration runner and JSON export/import.

Design decisions
----------------
- One SQLite file per user at the platform config dir (§6.3).
- ``CameraConfig`` is stored as a JSON blob so schema evolution (adding a
  new sub-field with a default) never requires a SQL column migration.
- ``ptz_presets``, ``identities``, ``identity_embeddings``, and ``layouts``
  get their own tables so they can be queried/indexed independently.
- ``schema_version`` is a single row in ``app_settings``; the migration runner
  applies numbered upgrade functions in order, each in its own transaction.
- Invalid rows (failing pydantic validation) are quarantined to an
  ``_quarantine`` list and logged — the rest of the config still loads.
- Debounced writes: callers can call ``save_camera_debounced()``; the store
  delays the actual write by ``debounce_s`` seconds and coalesces repeats.
  Flush with ``flush()`` on clean shutdown.
- JSON export/import ("show file") is a self-contained dict that can be
  round-tripped across machines.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autoptz.config.models import (
    CURRENT_SCHEMA_VERSION,
    AppConfig,
    CameraConfig,
    HardwarePrefs,
    IdentityRecord,
    Layout,
    PTZPreset,
    ThemeConfig,
)

log = logging.getLogger(__name__)

# ── Platform config-dir resolution ────────────────────────────────────────────


def default_config_dir() -> Path:
    """Return the platform-appropriate directory for the AutoPTZ config DB."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        import os

        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        import os

        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "AutoPTZ"


def default_db_path() -> Path:
    return default_config_dir() / "autoptz.db"


# ── DDL (schema version 1) ────────────────────────────────────────────────────

_DDL_V1 = """
CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL    -- JSON stored as text; JSON column type gives NUMERIC affinity
);

CREATE TABLE IF NOT EXISTS cameras (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    config     JSON NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ptz_presets (
    id             TEXT PRIMARY KEY,
    camera_id      TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    idx            INTEGER NOT NULL,
    name           TEXT NOT NULL,
    pan            REAL NOT NULL DEFAULT 0,
    tilt           REAL NOT NULL DEFAULT 0,
    zoom           REAL NOT NULL DEFAULT 0,
    native_preset  INTEGER,
    UNIQUE(camera_id, idx)
);

CREATE TABLE IF NOT EXISTS identities (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    thumbnail  BLOB,
    enabled    INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS identity_embeddings (
    id          TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS identity_photos (
    id          TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    image       BLOB NOT NULL,
    idx         INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS layouts (
    id    TEXT PRIMARY KEY,
    name  TEXT NOT NULL,
    tiles JSON NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS events (
    id        TEXT PRIMARY KEY,
    ts        TEXT NOT NULL,
    camera_id TEXT,
    level     TEXT NOT NULL DEFAULT 'info',
    code      TEXT NOT NULL DEFAULT '',
    message   TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_events_ts         ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_camera_id  ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_ptz_camera        ON ptz_presets(camera_id);
CREATE INDEX IF NOT EXISTS idx_emb_identity      ON identity_embeddings(identity_id);
CREATE INDEX IF NOT EXISTS idx_photo_identity    ON identity_photos(identity_id);
"""

# ── Migration registry ────────────────────────────────────────────────────────
# Each entry: (target_version, callable(conn) -> None)
# The runner applies all entries whose target_version > current schema_version.


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    """Apply the v1 schema to an older DB.

    _bootstrap() already runs CREATE TABLE IF NOT EXISTS for all tables, so
    this migration only needs to exist so the runner bumps schema_version.
    Future schema changes (ALTER TABLE, new tables) go in subsequent entries.
    """
    conn.executescript(_DDL_V1)


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    """Add the ``identities.enabled`` column (idempotent).

    Fresh DBs already have the column from the updated v1 DDL; an older DB that
    predates it gets it added here.  ``ADD COLUMN`` on an existing column raises
    ``OperationalError`` which we swallow so the migration is safe to re-run.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(identities)")}
    if "enabled" not in cols:
        conn.execute("ALTER TABLE identities ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1")


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    """Shorten the stale-track coast default for existing camera configs.

    Only configs still carrying the old default ``1500`` are migrated. A user
    who intentionally changed the value keeps their setting.
    """
    rows = conn.execute("SELECT id, config FROM cameras").fetchall()
    for row in rows:
        try:
            data = json.loads(row["config"])
            tracking = data.setdefault("tracking", {})
            if int(tracking.get("coast_window_ms", 1500)) != 1500:
                continue
            tracking["coast_window_ms"] = 300
            conn.execute(
                "UPDATE cameras SET config=? WHERE id=?",
                (json.dumps(data), row["id"]),
            )
        except Exception:  # noqa: BLE001
            log.debug("v3 coast-window migration skipped camera %s", row["id"], exc_info=True)


def _migrate_to_v4(conn: sqlite3.Connection) -> None:
    """Add the tracking aim-body mode to existing camera configs."""
    rows = conn.execute("SELECT id, config FROM cameras").fetchall()
    for row in rows:
        try:
            data = json.loads(row["config"])
            tracking = data.setdefault("tracking", {})
            if "aim_body_mode" in tracking:
                continue
            tracking["aim_body_mode"] = "torso"
            conn.execute(
                "UPDATE cameras SET config=? WHERE id=?",
                (json.dumps(data), row["id"]),
            )
        except Exception:  # noqa: BLE001
            log.debug("v4 aim-body-mode migration skipped camera %s", row["id"], exc_info=True)


_MIGRATIONS: list[tuple[int, Any]] = [
    # (target_version, upgrade_fn(conn) -> None)
    (1, _migrate_to_v1),
    (2, _migrate_to_v2),
    (3, _migrate_to_v3),
    (4, _migrate_to_v4),
]


# ── ConfigStore ───────────────────────────────────────────────────────────────


class ConfigStore:
    """Persistent config store backed by a SQLite database.

    Args:
        db_path:     Path to the SQLite file.  Created (with parent dirs) if absent.
        debounce_s:  Seconds to wait before flushing a debounced write.  Set to 0
                     to write synchronously on every call (useful in tests).
    """

    def __init__(
        self,
        db_path: Path | None = None,
        debounce_s: float = 0.5,
    ) -> None:
        self._path = db_path or default_db_path()
        self._debounce_s = debounce_s
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

        # Debounce state: camera_id → (CameraConfig, timer)
        self._pending: dict[str, tuple[CameraConfig, threading.Timer]] = {}
        self._pending_lock = threading.Lock()

        # Rows that failed pydantic validation
        self.quarantine: list[dict[str, Any]] = []

        self._open()
        self._bootstrap()
        self._migrate()

    # ── Connection management ──────────────────────────────────────────────────

    def _open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        self._conn = conn

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        assert self._conn is not None
        with self._lock, self._conn:
            yield self._conn

    def close(self) -> None:
        """Flush pending debounced writes and close the connection."""
        self.flush()
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ── Bootstrap & migrations ─────────────────────────────────────────────────

    def _bootstrap(self) -> None:
        """Create all tables if this is a fresh DB, then ensure schema_version exists."""
        with self._tx() as conn:
            conn.executescript(_DDL_V1)
            # Insert schema_version = 1 only if it doesn't exist yet
            conn.execute(
                "INSERT OR IGNORE INTO app_settings(key, value) VALUES (?, ?)",
                ("schema_version", json.dumps(CURRENT_SCHEMA_VERSION)),
            )

    def _migrate(self) -> None:
        """Run any pending migrations in version order."""
        current = self._get_schema_version()
        pending = [(v, fn) for v, fn in _MIGRATIONS if v > current]
        if not pending:
            return
        for target_version, fn in sorted(pending, key=lambda x: x[0]):
            log.info("Migrating DB schema %d → %d", current, target_version)
            with self._tx() as conn:
                fn(conn)
                conn.execute(
                    "UPDATE app_settings SET value=? WHERE key='schema_version'",
                    (json.dumps(target_version),),
                )
            current = target_version
        log.info("DB schema is now at version %d", current)

    def _get_schema_version(self) -> int:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT value FROM app_settings WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            return 0
        val = row["value"]
        # Defensive: a legacy DB with NUMERIC-affinity column may return int directly.
        if isinstance(val, int | float):
            return int(val)
        return int(json.loads(val))

    # ── app_settings ──────────────────────────────────────────────────────────

    def get_setting(self, key: str, default: Any = None) -> Any:
        assert self._conn is not None
        row = self._conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        val = row["value"]
        if isinstance(val, str):
            return json.loads(val)
        return val  # already a Python value (NUMERIC affinity in legacy DBs)

    def set_setting(self, key: str, value: Any) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO app_settings(key, value) VALUES (?,?)",
                (key, json.dumps(value)),
            )

    # ── Camera CRUD ───────────────────────────────────────────────────────────

    def save_camera(self, cam: CameraConfig) -> None:
        """Upsert a CameraConfig (including its embedded presets JSON)."""
        config_json = cam.model_dump_json()
        now = datetime.now(UTC).replace(tzinfo=None).isoformat()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cameras(id, name, config, enabled, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cam.id, cam.name, config_json, int(cam.enabled), now),
            )
            # Sync ptz_presets table for this camera (replace all)
            conn.execute("DELETE FROM ptz_presets WHERE camera_id=?", (cam.id,))
            for preset in cam.presets:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ptz_presets
                        (id, camera_id, idx, name, pan, tilt, zoom, native_preset)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        preset.id,
                        cam.id,
                        preset.idx,
                        preset.name,
                        preset.pan,
                        preset.tilt,
                        preset.zoom,
                        preset.native_preset,
                    ),
                )

    def save_camera_debounced(self, cam: CameraConfig) -> None:
        """Schedule a write that fires after ``debounce_s`` seconds.

        Repeated calls for the same camera_id within the window coalesce into
        one write — useful for high-frequency slider-drag events.
        """
        if self._debounce_s <= 0:
            self.save_camera(cam)
            return

        with self._pending_lock:
            existing = self._pending.get(cam.id)
            if existing:
                _, timer = existing
                timer.cancel()
            timer = threading.Timer(self._debounce_s, self._flush_one, args=(cam.id,))
            self._pending[cam.id] = (cam, timer)
            timer.start()

    def _flush_one(self, camera_id: str) -> None:
        with self._pending_lock:
            entry = self._pending.pop(camera_id, None)
        if entry:
            self.save_camera(entry[0])

    def flush(self) -> None:
        """Write all pending debounced cameras immediately."""
        with self._pending_lock:
            pending = list(self._pending.items())
            self._pending.clear()
        for _cid, (cam, timer) in pending:
            timer.cancel()
            self.save_camera(cam)

    def load_cameras(self) -> list[CameraConfig]:
        """Load all cameras; quarantine rows that fail validation."""
        assert self._conn is not None
        rows = self._conn.execute("SELECT config FROM cameras ORDER BY rowid").fetchall()
        cameras: list[CameraConfig] = []
        for row in rows:
            try:
                cam = CameraConfig.model_validate_json(row["config"])
                cameras.append(cam)
            except Exception as exc:
                log.warning("Quarantined invalid camera row: %s", exc)
                try:
                    raw = json.loads(row["config"])
                except Exception:
                    raw = {"raw": row["config"]}
                self.quarantine.append({"error": str(exc), "data": raw})
        return cameras

    def delete_camera(self, camera_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM cameras WHERE id=?", (camera_id,))
            conn.execute("DELETE FROM ptz_presets WHERE camera_id=?", (camera_id,))

    # ── Preset CRUD ───────────────────────────────────────────────────────────

    def load_presets(self, camera_id: str) -> list[PTZPreset]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM ptz_presets WHERE camera_id=? ORDER BY idx",
            (camera_id,),
        ).fetchall()
        presets: list[PTZPreset] = []
        for row in rows:
            try:
                presets.append(PTZPreset(**dict(row)))
            except Exception as exc:
                log.warning("Quarantined invalid preset row: %s", exc)
                self.quarantine.append({"error": str(exc), "data": dict(row)})
        return presets

    # ── Layout CRUD ───────────────────────────────────────────────────────────

    def save_layout(self, layout: Layout) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO layouts(id, name, tiles) VALUES (?,?,?)",
                (layout.id, layout.name, layout.model_dump_json()),
            )

    def load_layouts(self) -> list[Layout]:
        assert self._conn is not None
        rows = self._conn.execute("SELECT * FROM layouts ORDER BY rowid").fetchall()
        layouts: list[Layout] = []
        for row in rows:
            try:
                data = json.loads(row["tiles"]) if isinstance(row["tiles"], str) else row["tiles"]
                # tiles JSON may be the full Layout blob (if stored as model_dump_json)
                # or just the tiles list — handle both.
                if isinstance(data, dict):
                    layouts.append(Layout.model_validate(data))
                else:
                    layouts.append(Layout(id=row["id"], name=row["name"], tiles=data))
            except Exception as exc:
                log.warning("Quarantined invalid layout row: %s", exc)
                self.quarantine.append({"error": str(exc), "data": dict(row)})
        return layouts

    def delete_layout(self, layout_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM layouts WHERE id=?", (layout_id,))

    # ── Identity CRUD ─────────────────────────────────────────────────────────

    def save_identity(self, identity: IdentityRecord) -> None:
        now = datetime.now(UTC).replace(tzinfo=None).isoformat()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO
                    identities(id, name, thumbnail, enabled, created_at, updated_at)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    identity.id,
                    identity.name,
                    identity.thumbnail,
                    int(identity.enabled),
                    identity.created_at.isoformat(),
                    now,
                ),
            )
            conn.execute("DELETE FROM identity_embeddings WHERE identity_id=?", (identity.id,))
            for i, vec in enumerate(identity.embeddings):
                emb_id = f"{identity.id}:{i}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO identity_embeddings
                        (id, identity_id, vector, source, created_at)
                    VALUES (?,?,?,?,?)
                    """,
                    (emb_id, identity.id, vec, "", now),
                )
            # Candidate profile photos (the recognition crops the user re-picks from).
            conn.execute("DELETE FROM identity_photos WHERE identity_id=?", (identity.id,))
            for i, img in enumerate(identity.thumbnails):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO identity_photos
                        (id, identity_id, image, idx, created_at)
                    VALUES (?,?,?,?,?)
                    """,
                    (f"{identity.id}:p{i}", identity.id, img, i, now),
                )

    def load_identities(self) -> list[IdentityRecord]:
        assert self._conn is not None
        id_rows = self._conn.execute("SELECT * FROM identities ORDER BY created_at").fetchall()
        records: list[IdentityRecord] = []
        for row in id_rows:
            try:
                emb_rows = self._conn.execute(
                    "SELECT vector FROM identity_embeddings WHERE identity_id=? ORDER BY id",
                    (row["id"],),
                ).fetchall()
                embeddings = [bytes(r["vector"]) for r in emb_rows]
                photo_rows = self._conn.execute(
                    "SELECT image FROM identity_photos WHERE identity_id=? ORDER BY idx",
                    (row["id"],),
                ).fetchall()
                thumbnails = [bytes(r["image"]) for r in photo_rows]
                row_keys = row.keys()
                enabled = bool(row["enabled"]) if "enabled" in row_keys else True
                records.append(
                    IdentityRecord(
                        id=row["id"],
                        name=row["name"],
                        thumbnail=bytes(row["thumbnail"]) if row["thumbnail"] else None,
                        thumbnails=thumbnails,
                        embeddings=embeddings,
                        enabled=enabled,
                        labeled=True,  # only labeled identities are ever persisted
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
            except Exception as exc:
                log.warning("Quarantined invalid identity row: %s", exc)
                self.quarantine.append({"error": str(exc), "data": dict(row)})
        return records

    def delete_identity(self, identity_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM identities WHERE id=?", (identity_id,))
            conn.execute("DELETE FROM identity_embeddings WHERE identity_id=?", (identity_id,))
            conn.execute("DELETE FROM identity_photos WHERE identity_id=?", (identity_id,))

    def delete_identity_photo(self, identity_id: str, index: int) -> bool:
        """Delete a single candidate photo (``identity_photos`` row) by position.

        Removes the row at ``idx == index`` for *identity_id*, then re-packs the
        remaining photos' ``idx`` / row ids to stay contiguous (mirroring the
        ``f"{id}:p{i}"`` id scheme :meth:`save_identity` writes).  Embeddings are
        untouched — photos and embeddings are independent (see
        :meth:`IdentityService.remove_thumbnail`).  Returns ``True`` when a photo
        was removed, ``False`` when the index is out of range.

        Note: the service already re-persists the whole record via
        :meth:`save_identity` after a removal, so this is the explicit
        single-photo store path (and keeps the store usable on its own).
        """
        now = datetime.now(UTC).replace(tzinfo=None).isoformat()
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT image FROM identity_photos WHERE identity_id=? ORDER BY idx",
                (identity_id,),
            ).fetchall()
            if not (0 <= index < len(rows)):
                return False
            images = [bytes(r["image"]) for r in rows]
            images.pop(index)
            conn.execute("DELETE FROM identity_photos WHERE identity_id=?", (identity_id,))
            for i, img in enumerate(images):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO identity_photos
                        (id, identity_id, image, idx, created_at)
                    VALUES (?,?,?,?,?)
                    """,
                    (f"{identity_id}:p{i}", identity_id, img, i, now),
                )
            return True

    # ── Event log ─────────────────────────────────────────────────────────────

    def log_event(
        self,
        code: str,
        message: str,
        camera_id: str | None = None,
        level: str = "info",
    ) -> None:
        import uuid as _uuid

        now = datetime.now(UTC).replace(tzinfo=None).isoformat()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO events(id, ts, camera_id, level, code, message)
                VALUES (?,?,?,?,?,?)
                """,
                (str(_uuid.uuid4()), now, camera_id, level, code, message),
            )
            # Rolling: keep at most 10 000 events
            conn.execute(
                "DELETE FROM events WHERE id NOT IN "
                "(SELECT id FROM events ORDER BY ts DESC LIMIT 10000)"
            )

    # ── AppConfig convenience ─────────────────────────────────────────────────

    def load_app_config(self) -> AppConfig:
        """Reconstruct a full AppConfig from the DB."""
        cameras = self.load_cameras()
        raw_theme = self.get_setting("theme", {})
        raw_hw = self.get_setting("hardware", {})
        try:
            theme = ThemeConfig.model_validate(raw_theme)
        except Exception:
            theme = ThemeConfig()
        try:
            hardware = HardwarePrefs.model_validate(raw_hw)
        except Exception:
            hardware = HardwarePrefs()
        active_layout_id = self.get_setting("active_layout_id", "")
        return AppConfig(
            cameras=cameras,
            theme=theme,
            hardware=hardware,
            active_layout_id=active_layout_id,
        )

    def save_app_config(self, config: AppConfig) -> None:
        """Persist top-level AppConfig fields and all cameras."""
        self.set_setting("theme", config.theme.model_dump())
        self.set_setting("hardware", config.hardware.model_dump())
        self.set_setting("active_layout_id", config.active_layout_id)
        for cam in config.cameras:
            self.save_camera(cam)

    # ── JSON export / import ("show file") ────────────────────────────────────

    def export_show(
        self,
        *,
        include_identities: bool = False,
    ) -> dict[str, Any]:
        """Export cameras, presets, and layouts to a serialisable dict.

        The result is self-contained and portable across machines.  Pass it to
        ``import_show()`` on another instance to restore the config.

        Args:
            include_identities: If True, include identity names + embedding blobs.
                                 Omitted by default because blobs can be large.
        """
        cameras = self.load_cameras()
        layouts = self.load_layouts()
        out: dict[str, Any] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "exported_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "cameras": [json.loads(c.model_dump_json()) for c in cameras],
            "layouts": [json.loads(lo.model_dump_json()) for lo in layouts],
        }
        if include_identities:
            identities = self.load_identities()
            out["identities"] = [
                {
                    "id": i.id,
                    "name": i.name,
                    "thumbnail": i.thumbnail.hex() if i.thumbnail else None,
                    "thumbnails": [t.hex() for t in i.thumbnails],
                    "embeddings": [e.hex() for e in i.embeddings],
                    "enabled": i.enabled,
                    "created_at": i.created_at.isoformat(),
                    "updated_at": i.updated_at.isoformat(),
                }
                for i in identities
            ]
        return out

    def import_show(
        self,
        data: dict[str, Any],
        *,
        merge: bool = False,
    ) -> int:
        """Import cameras and layouts from an exported show dict.

        Args:
            data:  Dict produced by ``export_show()``.
            merge: If False (default), existing cameras/layouts are replaced.
                   If True, existing rows are preserved and new ones are added.

        Returns the number of cameras imported.
        """
        if not isinstance(data, dict):
            raise ValueError("import_show: expected a dict")

        if not merge:
            with self._tx() as conn:
                conn.execute("DELETE FROM ptz_presets")
                conn.execute("DELETE FROM cameras")
                conn.execute("DELETE FROM layouts")

        imported = 0
        for raw_cam in data.get("cameras", []):
            try:
                cam = CameraConfig.model_validate(raw_cam)
                self.save_camera(cam)
                imported += 1
            except Exception as exc:
                log.warning("import_show: skipping invalid camera %r: %s", raw_cam.get("id"), exc)
                self.quarantine.append({"error": str(exc), "data": raw_cam})

        for raw_layout in data.get("layouts", []):
            try:
                layout = Layout.model_validate(raw_layout)
                self.save_layout(layout)
            except Exception as exc:
                log.warning(
                    "import_show: skipping invalid layout %r: %s", raw_layout.get("id"), exc
                )
                self.quarantine.append({"error": str(exc), "data": raw_layout})

        for raw_id in data.get("identities", []):
            try:
                embeddings = [bytes.fromhex(e) for e in raw_id.get("embeddings", [])]
                thumbnail = bytes.fromhex(raw_id["thumbnail"]) if raw_id.get("thumbnail") else None
                thumbnails = [bytes.fromhex(t) for t in raw_id.get("thumbnails", [])]
                identity = IdentityRecord(
                    id=raw_id["id"],
                    name=raw_id["name"],
                    embeddings=embeddings,
                    thumbnail=thumbnail,
                    thumbnails=thumbnails,
                    enabled=bool(raw_id.get("enabled", True)),
                    labeled=True,
                    created_at=datetime.fromisoformat(
                        raw_id.get("created_at", datetime.now(UTC).replace(tzinfo=None).isoformat())
                    ),
                    updated_at=datetime.fromisoformat(
                        raw_id.get("updated_at", datetime.now(UTC).replace(tzinfo=None).isoformat())
                    ),
                )
                self.save_identity(identity)
            except Exception as exc:
                log.warning("import_show: skipping invalid identity %r: %s", raw_id.get("id"), exc)
                self.quarantine.append({"error": str(exc), "data": raw_id})

        return imported
