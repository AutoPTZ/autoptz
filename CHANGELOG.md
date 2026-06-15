# Changelog

All notable changes to AutoPTZ are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — v2.0.0a0

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
