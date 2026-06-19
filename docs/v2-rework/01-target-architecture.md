# 01 — Target Architecture

## Guiding principles

1. **One owner per camera.** Each camera is a self‑contained session that owns *all* of its
   state (frames, tracks, identities, PTZ handle, tuning). No cross‑camera global mutable state.
   This is the single most important change vs v1 and directly fixes the "data shifted to the
   wrong camera" bug.
2. **Engine ≠ UI.** A headless **Engine** does capture + inference + tracking + PTZ. A thin
   **UI** renders previews and sends commands. They communicate over a narrow, typed interface so
   either can be swapped (and so a remote/headless mode is possible later).
3. **Move heavy work off the GUI thread and out of the hot path.** The GUI never decodes,
   never runs inference, never drives PTZ. It reads a preview frame and overlays metadata.
4. **Zero‑copy where it counts.** Hand large frames between processes via **shared memory ring
   buffers**, never by pickling through a `Manager().list()`.
5. **Don't run every model on every frame.** Detection at a steady cadence, ReID on demand, face
   every ~0.5–1 s, pose only for the actively‑tracked subject. (See `03-vision-pipeline.md`.)

## Process / thread topology

```
┌──────────────────────────────────────────────────────────────────────────┐
│ UI process (PySide6 + Qt Quick/QML) — main app                             │
│   • CameraWall (QML): drag/reorder/resize tiles, themes, layouts           │
│   • Per-camera config drawer, PTZ preset bar, global identity manager      │
│   • Subscribes to EngineClient telemetry; renders preview frames           │
│   • Sends commands (add/remove camera, set target, enable tracking, PTZ)   │
└───────────────▲───────────────────────────────────────────┬──────────────┘
                │ telemetry (small msgpack msgs) + shm frame  │ commands
                │                                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Engine supervisor process                                                  │
│   • CameraManager: spawn/stop CameraWorker per camera, health/restart      │
│   • DiscoveryService(s): NDI find, ONVIF WS-Discovery, USB hot-plug        │
│   • IdentityService: shared face/ReID gallery (read-mostly, versioned)     │
│   • ConfigStore (SQLite): persist + broadcast config changes               │
└───────────────┬───────────────────────────┬───────────────┬──────────────┘
                │ spawn                       │ spawn          │ spawn
        ┌───────▼────────┐          ┌────────▼───────┐  ┌─────▼────────┐
        │ CameraWorker 1 │          │ CameraWorker 2 │  │ CameraWorker N│
        │ (one OS proc)  │          │                │  │               │
        │  Ingest → Decode → Detect → Track(+ReID) → Identify(face)       │
        │   → Pose(zoom) → PTZ control loop                              │
        │  writes: preview frame → shared memory; telemetry → queue      │
        └────────────────┘          └────────────────┘  └──────────────┘
```

### Why a process per camera (done correctly this time)

v1 already used multiprocessing but spread one camera across 3 processes that shared raw frames
by pickling. v2 inverts this: **one process per camera does the whole pipeline in‑process**, so
frames never cross a process boundary during inference. Within a worker, use 2–3 threads
(or asyncio tasks) for ingest vs. inference vs. PTZ I/O, coordinated by a small bounded
queue / latest‑frame slot. Python's GIL is not a problem here because the heavy work (decode,
ORT inference, OpenCV) releases the GIL.

Workers are isolated: a camera that hangs or crashes is restarted by the supervisor without
touching the others. This is also what makes scaling predictable — N cameras = N bounded workers.

## Inter‑process contracts

### Frames (Engine → UI): shared memory, latest‑wins

Each worker owns a small **double/triple‑buffered shared‑memory** region
(`multiprocessing.shared_memory.SharedMemory`) holding the most recent annotated preview frame
(downscaled, e.g. 640‑wide, BGR or JPEG). The UI maps it read‑only and blits it into the QML
tile. No locks on the hot path beyond an atomic "ready buffer index" + frame sequence number.
Preview cadence is decoupled from inference (UI can pull at its own 30–60 Hz; inference can run
slower).

### Telemetry (Engine → UI): small typed messages

`msgpack` (or `pydantic`+JSON) messages over a `multiprocessing.Queue` / local socket:
`{camera_id, seq, ts, fps, tracks:[{track_id, bbox, identity, confidence, is_target}],
 ptz:{pan,tilt,zoom,moving,backend,state}, health:{state, last_error}}`.
These drive overlays, status chips, and the config panels — **never** raw frames.

### Commands (UI → Engine): typed RPC

A small typed command bus (local socket or `Queue`), e.g.:
`AddCamera`, `RemoveCamera`, `UpdateCameraConfig`, `SetTarget(track_id|identity)`,
`EnableTracking(bool)`, `PtzNudge/PtzGoToPreset/PtzSavePreset`, `EnrollIdentity`, `SetLayout`.
Commands are idempotent and addressed by **stable `camera_id`** (a UUID), never by "the currently
active widget."

> **Transport choice:** start with in‑process `multiprocessing` Queues + shared memory for the
> local desktop app. Define the contracts behind an `EngineClient` interface so the same UI can
> later talk to a remote engine over **WebSocket/gRPC** without UI changes (stretch goal: remote
> control / headless server).

## Core modules (target package layout)

```
autoptz/
  engine/
    supervisor.py          # CameraManager, lifecycle, health, restart
    camera_worker.py       # the per-camera pipeline entrypoint (one process)
    pipeline/
      ingest.py            # source adapters: USB, RTSP/ONVIF, NDI (+ reconnect)
      detect.py            # YOLO26 person detector (ORT EP-agnostic)
      track.py             # BoxMOT tracker wrapper (BoT-SORT / DeepOCSORT / ByteTrack)
      reid.py              # OSNet embeddings + gallery matching
      identify.py          # InsightFace face detect+embed, bind track→identity
      pose.py              # RTMPose (optional, for zoom framing)
      framing.py           # target-selection, dead-zone, velocity, zoom controller
    ptz/
      base.py              # PTZBackend interface
      ndi_ptz.py  visca_ip.py  visca_usb.py  onvif_ptz.py
      controller.py        # closed-loop motion controller (shared by all backends)
    discovery/
      ndi.py  onvif.py  usb.py     # continuous discovery services
    identity/
      service.py  store.py         # enrollment, gallery, vector index
    runtime/
      inference.py         # ORT session factory: pick EP per platform/hardware
      shm.py               # shared-memory frame ring buffer helpers
      messages.py          # typed telemetry/command schemas (pydantic/msgpack)
  config/
    models.py              # pydantic config models (AppConfig, CameraConfig, ...)
    store.py               # SQLite + JSON load/save/migrations
  ui/                      # PySide6 + QML
    app.py  engine_client.py
    qml/                   # CameraWall.qml, CameraTile.qml, ConfigDrawer.qml, ...
    providers/             # QQuickImageProvider / shm → QImage bridge
  assets/  models/         # bundled ONNX/CoreML model files
  startup.py               # thin launcher (replaces v1 startup.py)
```

## Data‑flow per worker (one camera, steady state)

```
loop (ingest thread):   grab frame → HW decode → push to latest-frame slot
loop (inference thread): take latest frame
   ├─ every Nd frames:  detect persons (YOLO26)
   ├─ always:           tracker.update(detections, frame)   # motion + ReID assoc
   ├─ on new/ambiguous track: reid.embed() → gallery match
   ├─ every ~0.5–1s on target: face detect+embed → bind/confirm identity
   ├─ if zoom enabled:  pose on target bbox → desired zoom level
   ├─ framing:          choose target → error(x,y), velocity → controller
   └─ render preview (downscale + overlays) → shared memory; emit telemetry
loop (ptz thread):       consume controller setpoints → backend pan/tilt/zoom (rate-limited)
```

This separation (ingest / inference / ptz) keeps PTZ command latency low and independent of the
inference frame rate, which is central to "extremely real‑time" smoothness.

## Failure handling & health

- **Source drop:** ingest adapter detects stalled reads, backs off, reconnects; worker reports
  `health.state = reconnecting`. Optionally front sources with **go2rtc** so reconnection and
  protocol normalization happen outside the engine.
- **Worker crash:** supervisor restarts with the persisted `CameraConfig`; UI shows a transient
  "restarting" state. PTZ is commanded to stop on worker exit.
- **Model/EP failure:** inference factory falls back CPU‑ward (e.g., TensorRT → CUDA → CPU/
  OpenVINO) and logs the downgrade so the UI can surface "running on CPU."
</content>
