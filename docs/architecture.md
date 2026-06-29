# Architecture

AutoPTZ 2.2 targets one production runtime: the Qt Widgets UI plus supervised
camera workers using shared model ownership, source-agnostic latest-wins frames,
and fixed-zoom tracking by default. The UI never blocks on capture, inference, or
PTZ I/O, and workers never touch Qt.

Legacy process-per-camera and model-server experiments still exist as explicit
developer/Labs env paths, but they are not the normal product architecture. See
`docs/engineering/retired-experiments.md` and
`docs/release/2.2.0-reliability-gates.md` before promoting either path.

```
┌────────────────────────── UI process (PySide6, GUI thread) ──────────────────────────┐
│  app.run()  →  MainWindow  ──┬── EngineClient (typed command/telemetry bridge)         │
│                              ├── ShmFrameSource (reads preview frames from SHM)         │
│                              ├── UpdateManager (GitHub Releases check, off-thread)      │
│                              └── ConfigStore (SQLite settings + cameras)                │
│        ▲ telemetry / preview                              │ commands                    │
└────────┼─────────────────────────────────────────────────┼─────────────────────────────┘
         │                                                  ▼
┌────────┴───────────────────── Supervisor (owns + supervises workers) ──────────────────┐
│  start() → applies hardware prefs to env → starts one CameraWorker per camera           │
└────────┬───────────────────────────────────────────────────────────────────────────────┘
         ▼  (capture + inference/control threads per camera; shared model ownership)
┌────────────────────────────── CameraWorker ─────────────────────────────────────────────┐
│  capture thread:  FrameSource.read() → SHM preview → hand newest frame to inference       │
│  inference thread: detect → track → reID recover → pose → aim → PTZ controller → backend  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Per-frame data flow

```
BGR frame
  └─ PersonDetector.detect()      detect.py     YOLO11 ONNX via ONNX Runtime (best EP)
     └─ Tracker.update()          track.py      BoT-SORT/ByteTrack (boxmot) or IoU fallback
        └─ BodyReID.recover()     reid.py       OSNet appearance re-bind of the target
        └─ PoseEstimator + framing pose.py/framing.py   torso/upper-body aim anchor
           └─ aim fusion + smoothing  (camera_worker)   pose anchor ⊕ bbox, EMA smoothing
              └─ PTZController.step()  ptz/controller.py  predict → deadzone → one-euro →
                                                          PD + velocity FF → response curve
                 └─ backend.move_velocity()  ptz/*.py     VISCA-USB / VISCA-IP / ONVIF / NDI
```

## Package map

| Path | Responsibility |
| --- | --- |
| `autoptz/config/` | Immutable pydantic models (`models.py`) + SQLite-backed `ConfigStore` (`store.py`). |
| `autoptz/engine/runtime/` | `inference.py` (EP selection + tuned ORT sessions), `models.py` (model fetch/export), `messages.py` (typed commands/telemetry), `shm.py` (shared-memory frame ring). |
| `autoptz/engine/pipeline/` | `detect.py`, `track.py`, `reid.py`, `pose.py`, `framing.py`, `identify.py`, `ingest.py`, `avf_capture.py`, `pool.py`. |
| `autoptz/engine/ptz/` | `controller.py` (smoothing/PD/zoom state machine) + backends `visca_usb.py`, `visca_ip.py`, `onvif_ptz.py`, `ndi_ptz.py`, `factory.py`. |
| `autoptz/engine/discovery/` | USB/network/NDI camera discovery. |
| `autoptz/engine/identity/` | Identity gallery service (faces/ReID embeddings). |
| `autoptz/engine/worker/` | Worker support modules extracted from `camera_worker`: `frame_source.py` (FrameSource + fps pacing + source construction) and `stacks.py` (ML capability probes + detector/face stack builders). |
| `autoptz/engine/camera_worker.py` | The `CameraWorker` itself — capture + inference threads, command handling, telemetry. |
| `autoptz/engine/supervisor.py` | Starts/supervises one worker (threads) per camera; publishes hardware prefs to the environment before start. |
| `autoptz/ui/` | `app.py` (entry), `engine_client.py` (Qt bridge) + `list_models.py` (camera/identity/layout models), `branding.py`, `update_manager.py`, `theme.py`, `frames.py`, `log_bridge.py`, `widgets/` (main window, camera wall/tiles, panels, dialogs; pure helpers split into `tile_helpers.py` / `properties_helpers.py`). |
| `autoptz/update/` | `checker.py` parses GitHub Releases; `installer.py` downloads and launches the matching OS asset. |

## Key conventions

- **Cameras are addressed by stable UUID** everywhere — never by list position.
- **Config objects are frozen** pydantic models, safe to pass between processes.
- **Nothing hard-fails on a missing model/dep** — the pipeline degrades to
  live-preview-only and logs an actionable one-time message.
- **Hardware prefs reach workers via environment** (`AUTOPTZ_FORCE_EP`,
  `AUTOPTZ_PRECISION`, `AUTOPTZ_ORT_INTRA_THREADS`), set by the supervisor and
  read by `inference.prefs_from_env()` — production worker threads read them
  in-process, and explicit Labs process modes inherit them.
- **EP selection is centralized** in `engine/runtime/inference.py`; see
  [Performance](performance.md).
