# 06 — Persistence & Config

v1 persists **nothing** (in‑memory module globals; Save/Open are unwired stubs). v2 remembers
everything: cameras, per‑camera settings, PTZ presets, layouts, identities, and window state.

## 6.1 Storage strategy

- **SQLite** is the source of truth (single file, transactional, queryable). One DB per user
  profile at the platform config dir:
  - macOS: `~/Library/Application Support/AutoPTZ/autoptz.db`
  - Windows: `%APPDATA%\AutoPTZ\autoptz.db`
- **JSON import/export** for portability/backup/sharing a setup between machines (a "show file").
- **pydantic** models define the schema; a tiny migration runner (`schema_version`) upgrades old
  DBs. Config objects validate on load; invalid rows are quarantined, not crashed on.
- **Models/large blobs** (ONNX/CoreML, compiled TRT/CoreML caches) live on disk under the app data
  dir, referenced by path — not in the DB.

## 6.2 Config models (pydantic) — shape, not final code

```python
class AppConfig:
    schema_version: int
    theme: ThemeConfig
    active_layout_id: str
    hardware: HardwarePrefs        # forced EP override, model tier, max workers
    cameras: list[CameraConfig]

class CameraConfig:
    id: str                        # stable UUID — the ONLY camera handle used anywhere
    name: str
    source: SourceConfig           # type=USB|RTSP|ONVIF|NDI, address, creds, substream, fps
    enabled: bool
    tracking: TrackingConfig
    ptz: PTZConfig
    presets: list[PTZPreset]
    target: TargetConfig           # mode=identity|manual, identity_id?, default_on_start
    reconnect: ReconnectConfig

class TrackingConfig:
    tracker: Literal["botsort","deepocsort","bytetrack"]
    detect_interval: int
    reid_enabled: bool
    reid_threshold_hi: float; reid_threshold_lo: float
    coast_window_ms: int
    face_confirm: bool
    quality_floor: Literal["auto","high","balanced","low"]

class PTZConfig:
    backend: Literal["auto","ndi","visca_ip","visca_usb","onvif"]
    address: str | None            # IP/port/serial/NDI name as applicable
    max_pan_speed: float; max_tilt_speed: float; max_zoom_speed: float
    invert_pan: bool; invert_tilt: bool
    deadzone_x: float; deadzone_y: float
    kp: float; kd: float; kv: float
    auto_zoom: bool; zoom_framing: Literal["tight","medium","wide"]
    soft_limits: PanTiltZoomLimits | None

class PTZPreset:
    idx: int; name: str
    pan: float; tilt: float; zoom: float    # absolute, normalized
    native_preset: int | None               # if device stores it natively

class IdentityRecord:
    id: str; name: str
    embeddings: list[bytes]        # ArcFace vectors (blob); averaged template + samples
    thumbnail: bytes | None
    created_at, updated_at

class Layout:
    id: str; name: str
    tiles: list[TilePlacement]     # camera_id, x, y, w, h, z, visible
```

## 6.3 Tables (SQLite)

```
app_settings(key TEXT PK, value JSON)                  -- theme, active layout, hardware prefs
cameras(id PK, name, config JSON, enabled, updated_at) -- one row per camera (CameraConfig)
ptz_presets(id PK, camera_id FK, idx, name, pan, tilt, zoom, native_preset)
identities(id PK, name, thumbnail BLOB, created_at, updated_at)
identity_embeddings(id PK, identity_id FK, vector BLOB, source, created_at)
layouts(id PK, name, tiles JSON)
events(id PK, ts, camera_id, level, code, message)     -- health/audit log (rolling)
```

> Storing `CameraConfig`/`TrackingConfig`/`PTZConfig` as a validated JSON blob in one column keeps
> migrations easy while still allowing top‑level indexed columns (`id`, `enabled`). Presets and
> embeddings get their own tables because they're queried/scanned independently.

## 6.4 Identity / embedding store

- ArcFace embeddings stored as float32 blobs in `identity_embeddings`. Maintain an in‑memory matrix
  + cosine search (`numpy`, or `faiss-cpu` if galleries grow) loaded by the `IdentityService`.
- Versioned: bump an `identity_gallery_version` so workers reload the gallery when it changes
  (replaces v1's pickle‑file `watchdog` hack with an explicit, race‑free broadcast).
- Enrollment writes new rows + a thumbnail; deletion is a transaction.

## 6.5 Lifecycle

- **Startup:** load `AppConfig`; restore window + active layout; for each `enabled` camera, the
  supervisor spawns a worker with its `CameraConfig` (and, if set, recalls its `default_on_start`
  preset / `home`).
- **Live edits:** every UI change → `UpdateCameraConfig`/`SetLayout` command → engine applies +
  `ConfigStore` persists (debounced writes for slider drags).
- **Shutdown:** persist window/layout, stop PTZ, flush events. A crash loses at most the last
  debounced edit, never the whole setup.
- **Export/Import:** "Save Show As…" writes a JSON bundle (cameras + presets + layouts, optionally
  identities); "Open Show…" validates and loads it. This is the real implementation of v1's stub
  File menu.
</content>
