# 09 — Implementation Prompts (for the executing model)

Copy‑paste prompts, one per phase, designed for a *different, cheaper model* (e.g. Sonnet) to
execute without re‑deriving the whole plan. Each prompt is self‑contained but references the design
docs in this folder. Run them **in order**; each assumes the previous phase merged.

**Shared preamble — prepend to every prompt:**

> You are implementing AutoPTZ v2 on branch `dev/v2-architecture-rework`. The full design lives in
> `docs/v2-rework/` — read `01-target-architecture.md` and `02-technology-stack.md` first, then the
> doc named in this task. Keep the legacy v1 app (`views/`, `logic/`, `libraries/`, `shared/`)
> runnable; build the new code under the `autoptz/` package. Address cameras and PTZ devices by
> stable UUID only — never via global "current active" state (that was v1's core bug). Do no heavy
> work on the GUI thread. Add unit tests for every new module and a short note to `CHANGELOG.md`.
> Work on a feature branch off `dev/v2-architecture-rework`, open a PR when the acceptance criteria
> pass, and stop for review. Ask before adding a heavyweight dependency not already named in
> `02-technology-stack.md`.

---

## Prompt — Phase 0: Foundations & scaffolding
Implement Phase 0 from `08-execution-roadmap.md`.
1. Create the package skeleton from `01-target-architecture.md` (`autoptz/engine/...`, `config/`,
   `ui/`, `assets/`, `models/`, plus `tools/bench/` placeholder).
2. Create `requirements/base.txt` and extras `requirements/gpu-nvidia.txt`, `requirements/macos.txt`
   matching `02-technology-stack.md`; pin versions; document install per platform in the package
   README.
3. Implement `autoptz/engine/runtime/inference.py`: `make_session(model_path, prefer=...)` returning
   an ONNX Runtime session using the best available EP (CoreML → TensorRT → CUDA → DirectML →
   OpenVINO → CPU), with explicit fallback + structured logging of the chosen EP, and a forced‑EP
   override from `HardwarePrefs`.
4. Implement `autoptz/engine/runtime/shm.py` (a latest‑wins, double/triple‑buffered shared‑memory
   frame ring buffer: writer + reader, sequence numbers, no hot‑path locks) and
   `autoptz/engine/runtime/messages.py` (pydantic/msgpack telemetry + command schemas).
5. Add a `python -m autoptz --selftest` entry that prints the chosen EP, writes a synthetic frame to
   an shm buffer, reads it back in another process, and round‑trips one telemetry + one command msg.
6. Add GitHub Actions CI (macos‑arm64, windows‑x64): install, ruff/flake8, mypy, pytest.
**Done when:** `--selftest` passes locally and CI is green on both OSes.

## Prompt — Phase 1: Config & persistence
Implement Phase 1 from `08-execution-roadmap.md` per `06-persistence-and-config.md`.
- Build `autoptz/config/models.py` (pydantic: AppConfig, CameraConfig, SourceConfig, TrackingConfig,
  PTZConfig, PTZPreset, TargetConfig, ReconnectConfig, IdentityRecord, Layout) and
  `autoptz/config/store.py` (SQLite tables from §6.3; CRUD; `schema_version` migration runner;
  platform config‑dir resolution; debounced writes; JSON export/import "show file").
- Tests: create→persist→reload a CameraConfig; export→import equality; migration from empty/older DB;
  invalid row is quarantined, not fatal.
**Done when:** all persistence tests pass and a CameraConfig survives a simulated restart.

## Prompt — Phase 2: Ingest adapters + continuous discovery
Implement Phase 2 from `08-execution-roadmap.md` (see `02` ingest section, `01` ingest module).
- `autoptz/engine/pipeline/ingest.py`: a `SourceAdapter` interface + `USBAdapter`, `RTSPAdapter`
  (FFmpeg/PyAV with platform HW decode: VideoToolbox/NVDEC/D3D11VA), `NDIAdapter` (cyndilib,
  frame‑sync). Common features: target‑fps pacing, stalled‑read detection, reconnect with backoff,
  writes frames into an shm buffer, exposes status.
- `autoptz/engine/discovery/`: `ndi.py` (cyndilib finder w/ change callbacks), `onvif.py`
  (WS‑Discovery), `usb.py` (OS device‑change / Qt `videoInputsChanged`). Continuously emit
  add/remove events (NOT startup‑only).
- Optional: a `go2rtc` launcher/health wrapper that normalizes RTSP and exposes stable local URLs.
- Tests/mocks: simulate a stalled stream → adapter reconnects; simulate device add/remove → event
  fires. Provide a manual `tools/ingest_probe.py` to view one source.
**Done when:** live add/remove of USB and NDI sources is observed; RTSP auto‑reconnects after a drop.

## Prompt — Phase 3: Detection + tracking core
Implement Phase 3 from `08-execution-roadmap.md` per `03-vision-pipeline.md` §3.2–3.3.
- `autoptz/engine/pipeline/detect.py`: YOLO26 person detector over the ORT factory; configurable
  input size + `detect_interval`; returns `[bbox, conf]`. Bundle `yolo26n`/`yolo26s` ONNX (+ CoreML).
- `autoptz/engine/pipeline/track.py`: wrap BoxMOT; default BoT‑SORT with camera‑motion compensation;
  selectable DeepOCSORT/ByteTrack; expose track lifecycle (`tentative/confirmed/lost/removed`) and a
  configurable coast window.
- `tools/bench/track_clip.py`: run on a recorded clip, report ID stability and fps.
**Done when:** on a fast walk‑across + 1 s occlusion test clip, IDs stay stable in the common case at
the tier's target cadence.

## Prompt — Phase 4: ReID + identity (re‑identification fix)
Implement Phase 4 from `08-execution-roadmap.md` per `03-vision-pipeline.md` §3.4–3.5.
- `autoptz/engine/pipeline/reid.py`: OSNet ONNX embeddings; per‑track template ring + EMA; gallery
  cosine match with hysteresis (θ_hi/θ_lo); the run‑policy (new/ambiguous tracks + periodic on
  target only).
- `autoptz/engine/pipeline/identify.py` + `autoptz/engine/identity/{service,store}.py`: InsightFace
  SCRFD+ArcFace; enrollment (capture N crops, average template); SQLite gallery; versioned reload
  (replace v1's pickle/watchdog with an explicit gallery‑version broadcast).
- Implement the **recovery rule**: target lost → embed new tracks → match to target template AND
  enrolled identity → re‑bind old track_id/identity; reject interlopers.
- Tests: "interloper crosses in front" clip → target re‑acquired, not swapped; enroll→bind correct
  track; gallery version bump triggers reload.
**Done when:** the occlusion/crossing recovery test passes and face enrollment binds the right track.

## Prompt — Phase 5: PTZ backends + closed‑loop controller
Implement Phase 5 from `08-execution-roadmap.md` per `04-ptz-control.md`.
- `autoptz/engine/ptz/base.py` (`PTZBackend` + `PTZCaps`), and backends `ndi_ptz.py`, `visca_ip.py`,
  `visca_usb.py` (refactor v1's working serial VISCA), `onvif_ptz.py`. Normalized `[-1,1]` control;
  capability flags; absolute position + presets where supported (else store absolute pos in SQLite).
- `autoptz/engine/ptz/controller.py`: dead‑zone → one‑euro filter → PD + velocity feed‑forward →
  clamp/response‑curve → backend; coast‑on‑loss → search; zoom controller with hysteresis; per‑camera
  gains; rate‑limited PTZ thread; reliable `stop()` on all exit paths.
- Tests with a mock backend: step/ramp targets produce smooth, lead‑compensated commands; preset
  save/recall; loss → coast → stop.
**Done when:** mock + (if available) real PTZ shows smooth velocity‑aware tracking and working presets.

## Prompt — Phase 6: Camera worker + supervisor
Implement Phase 6 from `08-execution-roadmap.md` per `01-target-architecture.md`.
- `autoptz/engine/camera_worker.py`: one process; threads for ingest / inference / ptz; runs
  detect→track→reid→identify→(pose)→framing→controller; writes annotated preview to shm + telemetry
  to a queue; consumes commands; owns ALL its state (no module globals).
- `autoptz/engine/pipeline/framing.py` (target selection, dead‑zone, velocity, one‑euro) and
  `autoptz/engine/pipeline/pose.py` (RTMPose on target for zoom; optional).
- `autoptz/engine/supervisor.py`: CameraManager (spawn/stop/restart/health), hosts discovery +
  identity + config services; stops PTZ on worker exit.
- Headless integration test: start 3 workers from configs; verify independent tracking and **zero
  cross‑camera leakage**; kill one → it restarts.
**Done when:** the headless 3‑camera test passes and the v1 "wrong camera" bug cannot reproduce.

## Prompt — Phase 7: UI — camera wall + live preview
Implement Phase 7 from `08-execution-roadmap.md` per `05-ui-ux.md`.
- `autoptz/ui/app.py`, `engine_client.py` (typed wrapper over the command/telemetry contract;
  in‑process now, swappable for WebSocket later), `providers/` (shm→QImage/`QQuickImageProvider`).
- QML: `CameraWall.qml` (drag‑reorder/resize grid), `CameraTile.qml` (preview + telemetry overlays:
  boxes, target highlight, identity, dead‑zone, reticle, FPS/quality chip; quick track toggle;
  click‑box‑to‑target). Wire `AddCamera`/`RemoveCamera`/`SetTarget`/`EnableTracking`.
**Done when:** you can add cameras, see live overlaid previews, reorder tiles, pick a target by
clicking a box, and toggle tracking — all through the engine contract, no GUI‑thread stalls.

## Prompt — Phase 8: UI — config, presets, identities, layouts, themes
Implement Phase 8 from `08-execution-roadmap.md` per `05-ui-ux.md` + `06-persistence-and-config.md`.
- `ConfigDrawer.qml` (Source/Tracking/PTZ/Presets/Tuning with live sliders), preset bar,
  `IdentityManager.qml` (enroll/rename/delete, capture from a chosen camera), layout save/load,
  theme tokens (single theme file; remove any leftover hard‑coded stylesheet).
- Wire all edits to `UpdateCameraConfig`/`SetLayout`/`EnrollIdentity`; debounce slider writes.
**Done when:** every persisted setting is editable in‑app and survives restart; enroll → track‑by‑
identity works end to end; named layouts restore tile positions.

## Prompt — Phase 9: Performance, scaling & benchmark harness
Implement Phase 9 from `08-execution-roadmap.md` per `03-vision-pipeline.md` §3.8 and `07`.
- Add per‑stage latency metrics to the worker and a per‑worker **auto‑degrade** policy (drop pose →
  lower detect rate → switch BoT‑SORT→ByteTrack) that keeps the frame budget; surface the active
  quality level in telemetry/UI.
- Build `tools/bench/`: synthetic/recorded N‑camera load measuring glass‑to‑PTZ latency, sustained
  fps, and "max cameras at quality level X"; print a recommended hardware tier.
- Optional: shared/batched ORT session across same‑GPU workers.
**Done when:** bench meets the `07` tier table within tolerance on a reference machine and
auto‑degrade engages under overload instead of dropping frames.

## Prompt — Phase 10: Packaging, signing, installers
Implement Phase 10 from `08-execution-roadmap.md` per `02` packaging + `07.5`.
- macOS: PyInstaller/py2app spec → notarized `.app` in `.dmg`; bundle NDI runtime, ONNX/CoreML
  models, and a first‑run CoreML/TRT cache step.
- Windows: PyInstaller spec → Inno Setup/MSIX; default DirectML build + optional CUDA/TensorRT
  variant; bundle NDI runtime + models.
- CI publishes installers; add a first‑run EP self‑check + model presence check.
**Done when:** clean‑machine installs on Windows 11 and macOS launch, discover a camera, and track.

## Prompt — Phase 11: Cutover & cleanup
Implement Phase 11 from `08-execution-roadmap.md`.
- Repoint `startup.py` to the `autoptz` app; after verifying parity, remove/relocate v1 `views/`,
  `logic/`, `libraries/`, `shared/`; update root `README.md`, `requirements.txt`, screenshots, and
  the feature list; tag the last v1 commit for archival.
- Add a short user migration note (re‑add sources; re‑enroll faces — embedding format changed).
**Done when:** the repo builds and runs only the v2 path with updated docs; v1 is archived under a tag.

---

### Tips for the executing model
- When a real device isn't available, build against the **mock backends/adapters** and recorded
  clips the earlier phases add; never block a phase on hardware.
- Keep each PR focused on one phase; do not refactor v1 except where a phase explicitly says to.
- If a model export (YOLO26/OSNet/SCRFD/ArcFace/RTMPose) is missing for a target EP, fall back to the
  alternative named in `02-technology-stack.md` and note it in the PR rather than inventing one.
</content>
