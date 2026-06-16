# Changelog

All notable changes to AutoPTZ are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — v2.0.0a0

### Added — Phase 1: Config & persistence

- **`autoptz/config/models.py`** — Frozen pydantic models for the full config
  hierarchy: `AppConfig`, `CameraConfig`, `SourceConfig`, `TrackingConfig`,
  `PTZConfig`, `PanTiltZoomLimits`, `PTZPreset`, `TargetConfig`,
  `ReconnectConfig`, `HardwarePrefs`, `ThemeConfig`, `TilePlacement`,
  `Layout`, `IdentityRecord`.  All models are immutable (frozen) and
  UUID-addressed — never by list position or global state.
- **`autoptz/config/store.py`** — `ConfigStore`: SQLite-backed persistence
  with WAL mode and FK enforcement.  Key features:
  - Schema from §6.3: `app_settings`, `cameras`, `ptz_presets`, `identities`,
    `identity_embeddings`, `layouts`, `events` tables.
  - `schema_version` migration runner: numbered upgrade functions applied in
    order, each in its own transaction.  A DB with `schema_version=0` is
    automatically migrated to the current version on first open.
  - Platform config-dir resolution: `~/Library/Application Support/AutoPTZ/`
    on macOS, `%APPDATA%\AutoPTZ\` on Windows, `~/.config/AutoPTZ/` on Linux.
  - Debounced writes (`save_camera_debounced`): coalesces rapid slider-drag
    saves; `flush()` on clean shutdown.
  - JSON export/import (`export_show` / `import_show`): self-contained "show
    file" portable across machines; `merge=True` preserves existing rows.
  - Invalid rows quarantined to `store.quarantine`, not fatal.
- **`tests/test_config.py`** — 44 unit tests covering model validation,
  bootstrap/migration, camera CRUD (simulated restart), debounced writes,
  AppConfig round-trip, JSON export/import (equality, merge, identity blobs,
  invalid-row quarantine), and event logging.
- SQLite is stdlib; no new runtime dependency. `pydantic` already listed.

### Added — Phase 5: PTZ backends + closed-loop controller

- **`autoptz/engine/ptz/base.py`** — Full rewrite:
  - `PTZCaps` dataclass: capability flags (continuous pan/tilt/zoom, absolute moves,
    native presets, position query, per-axis speed ceilings).
  - `PTZState` dataclass: normalized position snapshot (pan/tilt [-1,1], zoom [0,1]).
  - `PTZBackend` ABC: `move_velocity`, `move_absolute` (optional), `stop`, `get_position`,
    `goto_preset`, `save_preset`, `close`; context-manager support.
  - Shared VISCA byte helpers used by both serial and IP backends:
    `visca_pantilt_cmd`, `visca_zoom_cmd`, `visca_stop_cmd`, `visca_zoom_stop_cmd`,
    `visca_preset_set_cmd`, `visca_preset_recall_cmd`.
- **`autoptz/engine/ptz/visca_usb.py`** — `ViscaUSBBackend`: pyserial VISCA/serial
  backend.  Normalized [-1,1] → VISCA speed bytes (0x01–0x18 pan, 0x01–0x14 tilt,
  0x01–0x07 zoom).  Drains pending ACK bytes before each write to prevent buffer
  stall.  Native preset memory commands (81 01 04 3F 01/02 MM FF).
- **`autoptz/engine/ptz/visca_ip.py`** — `ViscaIPBackend`: VISCA-over-TCP backend
  with two wire formats: ``"sony"`` (8-byte header per Sony VISCA-over-IP spec) and
  ``"raw"`` (plain bytes over TCP; default; compatible with PTZOptics/BirdDog/Lumens).
  Implements `get_position()` via `PanTiltPosInq` + `ZoomPosInq` inquiries; returns
  `None` on cameras that don't answer.
- **`autoptz/engine/ptz/ndi_ptz.py`** — `NDIPTZBackend`: cyndilib NDI PTZ receiver
  backend (`recv_ptz_pan_tilt_speed`, `recv_ptz_zoom_speed`, `recv_ptz_preset_store`,
  `recv_ptz_preset_recall`).  Optional — raises `ImportError` with install instructions
  if cyndilib is absent.  Caller owns the receiver lifetime.
- **`autoptz/engine/ptz/onvif_ptz.py`** — `ONVIFPTZBackend`: ONVIF PTZ via
  `onvif-zeep`.  Implements `ContinuousMove`, `AbsoluteMove`, `Stop`, `GetStatus`,
  `GotoPreset`, `SetPreset`.  Auto-selects first media profile; capability flags
  probed from `GetConfigurationOptions`.  Optional — raises `ImportError` if package
  is absent.
- **`autoptz/engine/ptz/controller.py`** — `PTZController`: rate-limited closed-loop
  controller (default 20 Hz PTZ thread):
  - `OneEuroFilter`: adaptive low-pass filter (Casiez et al. 2012); per-axis for pan
    and tilt error.  Low lag on fast motion, low jitter on slow motion.
  - Control pipeline per tick: error → dead-zone (elliptical, per-axis) → one-euro
    filter → PD + velocity feed-forward (`Kp·e + Kd·ė + Kv·v`) → per-camera speed
    ceiling → clamp [-1,1] → ease-in response curve (|x|^1.5) → `move_velocity()`.
  - Zoom controller: proportional error on `subject_height` vs framing target
    (tight=0.65, medium=0.45, wide=0.25) with ±5 % hysteresis band.
  - Coast-on-loss: when `track_active` goes False, holds last velocity for
    `coast_window_ms`, then `backend.stop()` and enters `SEARCHING` state.
  - Filters reset on target re-acquisition to avoid derivative spikes.
  - Rate-limit suppression: skips `move_velocity()` if command unchanged within ±1e-4.
  - Reliable `stop()` on all exit paths: thread join + `backend.stop()` in
    `try/finally`; background thread also calls `backend.stop()` on clean exit.
  - `step()` method for synchronous use in tests (injected timestamps, no thread).
  - `update()` method for inference-thread → PTZ-thread handoff (lock-protected).
  - Context manager: `__enter__` starts thread, `__exit__` calls `close()`.
- **`tests/test_ptz.py`** — 68 unit tests covering:
  - `PTZCaps` / `PTZState` dataclass defaults.
  - All VISCA byte helpers (stop, zoom stop, pan/tilt command encoding, preset cmds).
  - `OneEuroFilter`: pass-through init, convergence to constant, low-lag on step,
    reset, timestamp-free mode.
  - Helper `_clamp` and `_shape` (ease-in, sign preservation, unity).
  - Controller math: zero error, positive/negative error direction, feed-forward
    increases command, dead-zone suppression, dead-zone pass-through, invert pan/tilt,
    max-speed scaling, clamp to ±1, response curve, commands sent to backend.
  - Controller state machine: IDLE → TRACKING → COASTING → SEARCHING; re-acquisition;
    coast sends last velocity; coast expiry calls `backend.stop()`.
  - Zoom controller: auto_zoom off, tall/short subject, in-band no-zoom, speed clamp.
  - Presets: `goto_preset` / `save_preset` dispatch; multiple slots.
  - Thread lifecycle: start/stop, `close()` closes backend, context manager,
    stop-without-start, double-start idempotency, thread delivers commands.
  - Rate-limit and smoothing behaviour.
- `requirements/base.txt`: added `onvif-zeep==0.2.12` (optional ONVIF dep).

### Added — Phase 7: UI — camera wall + live preview

- **`autoptz/ui/engine_client.py`** — Typed QObject wrapper over the engine
  command/telemetry contract.  In-process today; swappable for WebSocket later
  without touching QML.
  - `CameraRecord` dataclass: per-camera mutable state (tracking flag, target
    ID, last telemetry, SHM geometry).  `shm_name` derived automatically from
    `camera_id`.  Properties: `fps`, `health`, `tracks_as_list()`,
    `ptz_as_dict()`.
  - `CameraListModel(QAbstractListModel)`: 11 roles (`CameraIdRole` through
    `ShmHeightRole`).  `add_camera`, `remove_camera`, `update_telemetry`.
    `@Slot swapCameras(id_a, id_b)` and `moveCamera(id, idx)` for drag-reorder
    via `layoutAboutToBeChanged/layoutChanged`.
  - `EngineClient(QObject)`: signals `cameraAdded`, `cameraRemoved`,
    `telemetryUpdated`, `errorOccurred`; slots `addCamera`, `removeCamera`,
    `enableTracking`, `setTarget`, `clearTarget`, `ptzNudge`, `ptzGoToPreset`.
    `push_telemetry()` thread-safe (called from engine thread).
    `drain_commands()` returns and clears the pending command deque.
- **`autoptz/ui/providers/__init__.py`** — `ShmFrameProvider(QQuickImageProvider)`:
  bridges SHM frames to QML `Image` sources.  `attach/detach/detach_all` manage
  per-camera `ShmReader` instances.  `requestImage` strips the `?r=N` cache-
  buster suffix before looking up the reader; returns a placeholder QImage when
  no frame is available.
- **`autoptz/ui/app.py`** — `run()` entry point: constructs `QGuiApplication`,
  `EngineClient`, and `ShmFrameProvider`; registers the image provider as
  `"frame"`; exposes `engineClient` to QML; loads `CameraWall.qml`.
- **`autoptz/ui/qml/CameraWall.qml`** — `ApplicationWindow` with dark palette,
  header toolbar (camera count + Add Camera button), collapsible left rail, and
  a `GridView` camera wall.  Auto-column count (`⌈√count⌉`), 16:9 cell aspect
  ratio.  Drag-reorder via `DragHandler` + `DropArea` → `swapCameras()`.
  Add Camera dialog with URI + display name fields.
- **`autoptz/ui/qml/CameraTile.qml`** — Per-camera tile with:
  - Live video preview via `image://frame/<id>?r=<tick>` (10 Hz cache-bust timer).
  - Person bounding-box `Repeater` (normalized [0,1] coords).  Target box
    highlighted in green; click any box to `setTarget`.
  - Centre reticle crosshair + dead-zone ellipse (Canvas, visible when tracking).
  - FPS/health chip (color-coded: green ≥20 fps, yellow ≥10, red otherwise).
  - State banner (`RECONNECTING`, `ERROR`, `STOPPED`, `SEARCHING`).
  - Bottom bar: status label + tracking toggle `Switch`.
  - Arrow-key PTZ nudge + Space-key tracking toggle (keyboard shortcuts when
    tile is selected).
- **`tests/test_ui.py`** — 53 tests (all passing without a display):
  - `CameraRecord`: `shm_name` derivation, `fps`/`health`/`tracks_as_list()`
    defaults and with telemetry.
  - `CameraListModel`: CRUD, duplicate guard, role data, telemetry update,
    swap/move operations, invalid-index safety.
  - `EngineClient`: `addCamera`/`removeCamera`, `enableTracking`, `setTarget`,
    `clearTarget`, `ptzNudge`, `ptzGoToPreset`, `push_telemetry` updates model,
    `drain_commands` returns and clears queue, command ordering, thread-safe
    telemetry, concurrent push.
  - `ShmFrameProvider` tests gated by `AUTOPTZ_GUI_TESTS=1` (needs display).
- Total tests: **302**.

### Added — Phase 3: Detection + tracking core

- **`autoptz/engine/pipeline/detect.py`** — `PersonDetector` wrapping an ONNX
  Runtime session via the Phase 0 EP factory.  Key features:
  - `detect_interval`: run inference only every N frames (default 1); returns
    `[]` on skipped frames so the tracker can coast on Kalman prediction.
  - `_letterbox()`: aspect-preserving resize with grey padding (114) +
    BGR→RGB→CHW→float32/255 normalisation.
  - Auto-detects NMS-free `[1,N,5|6]` output (YOLOv10/YOLO26 style) vs
    pre-NMS `[1, 4+C, anchors]` (YOLOv8 style); handles normalised and pixel
    coordinate spaces.
  - Vectorised greedy NMS for the pre-NMS path.
  - `detections_to_numpy()`: converts `list[Detection]` to `[N,6]` float32
    array in BoxMOT's expected format.
  - `make_synthetic_detector_session()`: builds a tiny ONNX model from
    `onnx.helper` (Constant-node output) for CI tests without model files.
- **`autoptz/engine/pipeline/track.py`** — `Tracker` wrapping BoxMOT with a
  four-state lifecycle:
  - States: `TENTATIVE → CONFIRMED` (after `min_hits` consecutive matches)
    `→ LOST` (coasting on Kalman prediction within `coast_window` seconds)
    `→ REMOVED` (coast window expired).
  - Default tracker: BoT-SORT with camera-motion compensation (CMC).
    Selectable via `TrackerType`: `BOTSORT | DEEPOCSORT | BYTETRACK`.
  - Velocity estimation from consecutive bbox-centre deltas `(vx, vy)`.
  - Lazy BoxMOT instantiation on first `update()` call (fps-dependent
    `max_age` = `coast_window × fps`).
  - Dependency injection via `_impl` parameter; BoxMOT is optional and
    degrades with a clear `ImportError` message if not installed.
- **`tools/bench/track_clip.py`** — CLI benchmark tool:
  - Runs a full detect + track pipeline on any video file.
  - Supports real YOLO26 ONNX models (`--model`) or `--synthetic` mode (no
    model file; uses `make_synthetic_detector_session`).
  - Reports: avg detect fps, avg total fps, unique track IDs, heuristic
    ID-switch count, stable tracks (≥1 s), avg active tracks/frame,
    occlusion recoveries.
  - Optional `--output`: writes annotated video with colour-coded bboxes
    (green = CONFIRMED, yellow = TENTATIVE, blue = LOST).
- **`tests/test_detect.py`** — 43 unit tests covering: `BBox` properties and
  IoU, `_letterbox` shapes/dtype/normalisation, `_to_orig_coords`,
  `_nms` (5 cases), `_parse_raw_output` (5 format cases), `PersonDetector`
  (7 tests), `detections_to_numpy` (3 tests), `make_synthetic_detector_session`
  (3 tests).
- **`tests/test_track.py`** — 19 unit tests covering: tracker basics,
  full lifecycle (TENTATIVE→CONFIRMED→LOST→REMOVED→re-acquired), age/hits
  counters, velocity, `TrackerType` enum, BoxMOT unavailability, edge cases
  (8-column output, empty frame, multiple lost tracks, fps-dependent coast).
- `requirements/base.txt`: added `onnx==1.17.0` and `boxmot>=10.0.91`.

### Added — Phase 2: Ingest adapters + continuous discovery

- **`autoptz/engine/pipeline/ingest.py`** — `SourceAdapter` ABC with
  target-fps pacing, stall detection (configurable timeout), and exponential
  reconnect backoff (1 s → 2 → 4 … 30 s).  Concrete adapters:
  - `USBAdapter` — OpenCV `VideoCapture` with platform backend
    (AVFoundation / MSMF / V4L2).
  - `RTSPAdapter` — PyAV (FFmpeg) with HW decode hints (VideoToolbox on
    macOS, D3D11VA on Windows, NVDEC/CUVID on Linux); falls back to
    `cv2.VideoCapture` if PyAV is not installed.
  - `NDIAdapter` — cyndilib `FrameSyncReceiver`; gracefully absent if
    cyndilib / NDI SDK runtime are not installed.
  All adapters write BGR frames into an injected `ShmWriter` (resizing to
  fit), expose a thread-safe `status` property, and run their capture loop
  in a daemon thread.
- **`autoptz/engine/discovery/ndi.py`** — `NDIDiscovery`: cyndilib
  `Finder` polled at a configurable interval; fires `on_change` callbacks
  with `("added"|"removed", NDISource)`.  No-ops gracefully without NDI.
- **`autoptz/engine/discovery/usb.py`** — `USBDiscovery`: cross-platform
  polling via `cv2.VideoCapture` index probing; on Linux also hooks
  `pyudev` for sub-second hot-plug events.
- **`autoptz/engine/discovery/onvif.py`** — `ONVIFDiscovery`: WS-Discovery
  multicast using `wsdiscovery`; device removal detected after a miss
  threshold (3 consecutive absent scans). No-ops without `wsdiscovery`.
- **`autoptz/engine/pipeline/go2rtc.py`** — optional `Go2RTCGateway`:
  launches a `go2rtc` subprocess, writes a config, health-checks the API,
  and exposes stable `rtsp://localhost:{port}/{name}` URLs.
- **`tools/ingest_probe.py`** — CLI tool to probe a single USB/RTSP/NDI
  source or run all discovery services for a fixed duration; useful for
  manual acceptance testing.
- **`tests/test_ingest.py`** — unit tests for all three adapters with
  mocked cv2 / PyAV / cyndilib; includes stall → reconnect timing test
  and ShmWriter delivery / resize tests.
- **`tests/test_discovery.py`** — unit tests for USB add/remove, NDI
  add/remove, ONVIF add + miss-threshold removal, and graceful degradation
  when optional packages are absent.
- `requirements/base.txt`: added `av==14.4.0` (PyAV) and
  `wsdiscovery==3.1.0`; cyndilib is noted in `requirements/macos.txt`
  pending NDI SDK installation.

### Added — Phase 0: Foundations & scaffolding

- **`autoptz/` package skeleton** matching the target architecture from
  `docs/v2-rework/01-target-architecture.md`: `engine/`, `config/`, `ui/`,
  `assets/`, `models/`, plus `tools/bench/` placeholder.
- **`requirements/base.txt`** (all platforms), **`requirements/gpu-nvidia.txt`**
  (TensorRT/CUDA EP), **`requirements/macos.txt`** (CoreML EP notes),
  **`requirements/dev.txt`** (lint, type-check, test).
- **`autoptz/engine/runtime/inference.py`** — `make_session()` + `get_best_ep()`:
  ONNX Runtime session factory that selects CoreML → TensorRT → CUDA →
  DirectML → OpenVINO → CPU based on platform and available providers, with
  explicit fallback logging and `HardwarePrefs.force_ep` override.
- **`autoptz/engine/runtime/shm.py`** — `ShmWriter` / `ShmReader`: latest-wins
  triple-buffered shared-memory frame ring buffer; torn-read protection via
  sequence-number fence; no hot-path locks.
- **`autoptz/engine/runtime/messages.py`** — pydantic + msgpack typed schemas
  for telemetry (`TelemetryMsg`) and commands (`AddCameraCmd`, `SetTargetCmd`,
  `PtzNudgeCmd`, etc.). All commands carry a stable `camera_id` UUID.
- **`python -m autoptz --selftest`**: prints chosen EP, round-trips a synthetic
  frame through an shm buffer in a subprocess, and round-trips telemetry +
  command messages via msgpack.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): matrix over
  `macos-14` (arm64) and `windows-latest` (x64); runs ruff, mypy, pytest,
  and `--selftest`.
- **`pyproject.toml`** with ruff, mypy, and pytest configuration.

### Notes

- v1 (`views/`, `logic/`, `libraries/`, `shared/`) is untouched and still
  runnable from `startup.py`.
- Placeholder modules log a `# TODO(phase-N):` comment so the next phase
  prompt knows exactly where to continue.

---

## Legacy (v1)

See git history on `main` for v1 changes prior to the v2 rework.
