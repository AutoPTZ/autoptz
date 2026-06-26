# Changelog

All notable changes to AutoPTZ are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.2.0-rc6] — 2026-06-26

> Pre-release for testing. Headline: a **major multi-camera CPU reduction**
> (validated on real cameras at ~2.3×) plus colored logging and two camera-handling
> fixes, on top of the reliability/PTZ/install work from the rc5 line. **Please
> validate on your real cameras** and report back before this becomes stable 2.2.0.
> Known next step: with several cameras the engine is now CPU-light but still
> Python-GIL-bound on per-frame work (some inference frames are skipped) — a
> process-per-camera path and an on-device benchmark are in progress.

### Performance

- **Multi-camera CPU cut ~2.3× — the dominant cost was ONNX Runtime thread-pool
  spin-wait, not inference.** Profiling a live 4-camera session showed ORT worker
  threads *busy-spinning* between intermittent runs (detect every Nth frame,
  pose/face a few Hz) as the single largest CPU consumer. Spinning is now disabled
  on every model session, and the insightface (face) sessions — which bypassed the
  thread cap and ran cores-wide — are capped and de-spun too. Measured on an M4 Pro
  with 4 real cameras and all services on: **~930% → ~411% CPU (≈66% → ≈29% of the
  machine)**, and the second-to-second CPU *bursts* are gone.
- **CPU/BLAS thread pools are bounded to the per-camera budget** (OMP / OpenBLAS /
  MKL / NumExpr / torch), so several camera workers no longer oversubscribe the CPU.

### Added

- **Colored logging.** The console colors each level (info green, warning yellow,
  errors bold red) and gives every camera a stable color so multi-camera output is
  easy to scan; the in-app Logs panel tints messages per-camera too. Auto-disables
  when output is piped/redirected (honors `NO_COLOR`).
- **Synthetic camera source** (`source type: synthetic`) for headless multi-camera
  testing with no physical camera or OS camera permission, plus `AUTOPTZ_DB_PATH`
  (isolated profile) and `AUTOPTZ_SKIP_CAMERA_PREFLIGHT` (start the engine with no
  local camera, for NDI/RTSP/synthetic or headless runs).
- **Unified "Tracking Speed" preset** (Calm / Normal / Fast / Sport) with a
  nonlinear dead-band for steadier framing.
- **Center Stage** gained multi-person **group framing** (auto-widens to keep
  everyone in shot), shot-size-aware headroom and subtle lead-room, plus a
  dead-zone hold, calmer zoom, sharper upscale, and a 30 fps virtual-camera output.
- **Off-thread PTZ command pump** (opt-in via `AUTOPTZ_PTZ_PUMP`) that emits motion
  at a fixed rate off the inference hot path, with a stop-on-loss heartbeat.
- **OpenVINO is auto-selected for Intel CPU / iGPU / Arc** systems at install.

### Changed

- **Torch-free default install.** The heavy tracking (boxmot) and export
  (ultralytics) stacks are now opt-in extras (`--with-tracking` / `--with-export`
  / `--full`); the default install is leaner.

### Fixed

- **USB cameras are identified by their stable device id, not the volatile
  `usb://<index>`** — fixes enabling/disabling or deleting the *wrong* camera after
  the USB enumeration order shifts.
- **Center Stage now zooms in on far subjects** instead of leaving a distant person
  small in frame (the minimum-crop floor is now per-framing).
- **PTZ stop-on-loss:** an ONVIF dead-man's-switch timeout and a VISCA
  halt-on-reconnect stop a runaway pan when the target/transport drops.
- **Crash-safe worker threads** — a failed capture/inference thread is surfaced and
  auto-restarted instead of dying silently, and hot-path errors are visible.
- **Honest precision reporting** per execution provider / host (no more misleading
  "fp16" where the host runs fp32).
- Accelerator auto-selection is deterministic from detected host info (fixes a
  Linux-only CI failure).

### Security

- **The updater verifies the downloaded installer (SHA-256) over a pinned-TLS
  connection before launching it.**

## [2.2.0-rc5] — 2026-06-25

> Pre-release for testing — PTZ tracking responsiveness + USB-PTZ and updater fixes.

### Added

- **Auto-detect USB (VISCA-over-serial) PTZ cameras** with a configurable baud rate,
  so a UVC camera with a companion VISCA serial port is driven without manual setup.
- **Dynamic, error-proportional catch-up tracking speed** — the controller speeds up
  to close a large framing error and eases off as it closes, for snappier yet stable
  following.

### Fixed

- Update flow: distinct check states, a loading indicator, and Intel-Mac TLS +
  architecture safety.

## [2.2.0-rc4] — 2026-06-25

> Pre-release for testing — fixes the three issues reported on rc3 (safe-zone
> centering, PTZ pan jitter, and Windows face-recognition diagnosability +
> offline model bundling). **Please validate on your real cameras** — especially
> physical-PTZ panning and Windows face enrollment — and report back before this
> becomes the stable 2.2.0.

### Fixed

- **Safe zone now centers the subject (was: froze them at the oval edge).** The
  PTZ controller's framing setpoint is the safe-zone **center**, not the frame
  center: a subject inside the zone is now gently eased toward the middle (Center
  Stage-style) instead of being held wherever they happened to enter the oval. A
  **frame-edge guard** keeps a wide zone from ever stranding the subject at the
  screen edge, and a configured **square** zone now respects its corners (the
  `Roundness` control) instead of always behaving like an ellipse.
- **Smoother PTZ tracking while the camera pans.** Ego-motion compensation is now
  window-matched, so the controller no longer injects the camera's own pan speed
  into its feed-forward on the frames between ego measurements (the main cause of
  hunting). A brief target loss now resumes from the pre-loss speed instead of
  cold-restarting the motion ramp (the "find" lurch). Note: this fixes the
  control-loop side of the jitter; a detector/tracker dropping the box during very
  fast pans is a separate item still under investigation.
- **Windows face recognition failures are now diagnosable, and the model can ship
  offline.** A failed insightface/model load no longer fails silently — the
  Services panel reports the real reason (and "model not downloaded" when the
  `buffalo_l` weights are absent) instead of a misleading "ok". The face pack is
  now provisioned via `python -m tools.fetch_models` and bundled into installers,
  so a fresh **offline** machine can enroll faces. (If face recognition is failing
  for a non-network reason — e.g. a dependency/ABI break — the Services panel now
  surfaces it so it can be fixed directly.)

## [2.2.0-rc3] — 2026-06-24

> Pre-release for testing — adds **Center Stage** software auto-framing on top of
> the rc2 CPU-performance and tracking work. Please validate on your real cameras
> and report back before this becomes the stable 2.2.0.

### Added

- **Center Stage** — software auto-framing (digital PTZ) for cameras without
  motorised PTZ: a single **Center Stage** toggle in the PTZ panel digitally
  crops, zooms, and pans to keep the selected target framed — a head-and-shoulders
  shot that follows them and holds steady through normal track re-association
  instead of snapping back to the full frame. Set how tightly it frames with the
  **Framing** dropdown (Face / Head & shoulders / Upper body / Full body), live.
  An optional **Virtual camera output** publishes the framed crop to Zoom/OBS
  (needs a system virtual-camera driver) and disconnects cleanly when turned off.
  The raw PTZ transport selector now lives under PTZ → Advanced, since most users
  use Center Stage or the auto-probe.

### Changed

- **Less CPU oversubscription from OpenCV** — OpenCV's internal thread pool
  (resize/letterbox and the per-frame ego-motion optical flow) is now capped to
  the same per-camera budget as the ONNX Runtime sessions instead of defaulting
  to every core. On Linux/Windows the exact count is honoured; on macOS's GCD
  OpenCV backend (which ignores a positive count) the cap forces single-threaded
  only under heavy multi-camera load.
- **ReID runs on the GPU** — OSNet appearance ReID now uses the Apple GPU (`mps`)
  or CUDA when available instead of always the CPU, cutting the per-frame
  appearance cost (and the GIL stall it caused on Macs). Set
  `AUTOPTZ_REID_DEVICE=cpu` to force the previous behaviour.
- **Verify/force the CoreML GPU path** — `AUTOPTZ_COREML_UNITS`
  (`ALL` / `CPUAndGPU` / `CPUOnly`) lets Intel + AMD Macs (e.g. iMac Pro Xeon +
  Vega 56) check whether CoreML is actually using the discrete GPU or silently
  falling back to the CPU: run `--bench` with `CPUOnly` and again with `ALL` and
  compare. Defaults to `ALL`.

### Fixed

- **Named-target tracking no longer drifts to the wrong person** — a target
  selected by identity (name), once lost, could be re-bound by appearance ReID to
  the next visually-similar person, then any track. It now re-binds only to a
  face-confirmed match of the *same* identity; otherwise it keeps searching.
- **The camera no longer chases an occluded subject** — when the target is
  suddenly covered, its box collapses to the visible part (e.g. the legs); the
  drive loop now coasts on a sudden box-height collapse instead of following the
  shrinking box down, and resumes when the subject reappears at full size.

### Removed

- Removed the per-camera **"Confirm with face recognition"** toggle
  (`tracking.face_confirm`). It was never wired to engine behaviour — face
  recognition is controlled by the global **Face recognition** module switch
  (Services) — so the checkbox only persisted and displayed itself. Existing
  configs that still carry the key load fine (it is ignored).

### Internal

- Consolidated the `AUTOPTZ_UNIFIED_POSE` env flag into a single resolver
  (`autoptz.engine.runtime.flags.env_unified_pose`); the inference pool and the
  worker stacks now share one source of truth instead of two duplicate readers.

## [2.2.0-rc2] — 2026-06-24

> Pre-release for testing — builds on rc1 with CPU-performance, tracking, and
> diagnostics work. Please validate on your real cameras and report back before
> this becomes the stable 2.2.0.

### Added

- **Live GPU-acceleration verdict in the Services panel** — the detector row now
  shows whether your GPU/accelerator is actually beating the CPU (e.g. *CoreML ·
  1.24× CPU (accelerated)*), measured once in the background at startup.
- **OpenVINO auto-install for Intel** — `tools/install.py` now auto-selects the
  OpenVINO ONNX Runtime wheel on Intel CPU/iGPU Linux machines (faster than the
  stock CPU provider).
- **Opt-in fused target-association** — a new `TargetAssociator` fuses motion,
  ReID, pose, and identity into one confidence-scored keep/switch/hold decision
  with explicit anti-ID-switch hysteresis. **Off by default**
  (`tracking.use_target_associator`); enable it to try the new logic and report back.
- **Batched detector inference** primitive (`detect_batch`) — foundation for a
  future multi-camera GPU batching path.

### Changed

- **Lower CPU usage and fewer CPU spikes with all subservices running** — face,
  ReID, and pose no longer fire on the same tick (phase-staggered), and their
  cadence eases off automatically when the machine is over its frame budget.
- **CoreML compile-cache** — the CoreML model compiles once per machine and is
  cached, cutting cold-start and per-camera warmup on Apple Silicon and Intel Macs.
- **Lighter preview path** — the camera-tile preview is capped at ~20 fps (it's a
  monitoring view), saving a per-frame resize on higher-fps sources.
- **Scene-adaptive ReID** — the recovery match threshold tightens automatically
  when people in frame look alike (reducing wrong re-locks); it never loosens
  below your configured value.

### Fixed

- CI test reliability: deflaked the macOS app-memory test, the Qt widget-smoke
  subprocess tests, and the ReID-throttle test.

## [2.2.0-rc1] — 2026-06-24

> Pre-release for testing — please validate on your real cameras and report back
> before this becomes the stable 2.2.0.

### Added

- **GPU acceleration check (`--bench`)** — run `python -m autoptz --bench` to see
  whether your GPU/accelerator is actually faster than the CPU, with a plain verdict
  (*accelerated* / *no-benefit* / *cpu-only*). It times the auto-selected execution
  provider (CoreML, CUDA/TensorRT, DirectML, OpenVINO) against a CPU baseline, so you
  can tell when, e.g., CoreML on an Intel Mac is silently running on the CPU.
- **CoreML compile-cache** — the CoreML model is compiled once per machine and cached,
  cutting cold-start on Apple Silicon and Intel Macs (and per-camera warmup when
  running several cameras).
- **Inference-hang watchdog** — if detection inference stalls, the camera now holds
  position and shows a *degraded — inference stalled* status instead of driving the
  PTZ toward a frozen target; it recovers automatically when inference resumes.
- **Automatic camera recovery** — if a camera's processing thread crashes, the engine
  detects it and restarts that camera automatically (with backoff) instead of letting
  it go silently dark.
- **PTZ auto-reconnect** — VISCA over IP and USB now reconnect automatically after a
  network blip or a USB re-enumeration, instead of staying dead until you restart.

### Fixed

- CI reliability: deflaked the macOS app-memory test and the Qt widget-smoke subprocess
  tests so continuous integration is no longer intermittently red.

## [2.1.0] — 2026-06-23

### Added

- **Models ship inside the installer** — the detector tiers (Fast / Balanced /
  Accurate) and the pose model are now bundled with the app, so detection and
  pose work on first launch with **no download, network, or setup**. They appear
  in the Model Manager as *Included* and can't be removed.
- **Model Manager** (Engine → Models) — review every AutoPTZ-managed model with
  per-row status, download or remove the ones you want, pick the active detector
  tier (un-downloaded tiers are greyed out), with live download progress.
- **Seamless in-app updates** — the updater shows a **download progress bar**, and
  on Windows installs with **no setup wizard** (just a progress window) then
  **relaunches AutoPTZ automatically**, the closest to a "it just updated"
  experience.
- **Copy selected log lines** — select a range of rows in the Logs panel and copy
  just those with `Ctrl`/`Cmd`+`C` or right-click → *Copy Selected*; the toolbar
  button is now *Copy All*.
- **Quick-collapse side panels** — hide/show the left (Properties) and right
  (Camera Info / People / Services) panels from the View menu, the status-bar
  ◧ / ◨ buttons, or `Ctrl+Alt+[` / `Ctrl+Alt+]`.
- **Startup banner** — a progress banner steps through engine/camera startup.
- **Signed + notarized macOS builds** — opt-in locally via `MACOS_SIGN_IDENTITY`
  (and notary credentials), and automatic in the release workflow once the signing
  secrets are configured; unsigned otherwise.
- **Inference EP is shown again** — the status bar and Camera Info display the
  active execution provider (CoreML / DirectML / CPU …); it was previously blank
  on every platform because the value was never populated from telemetry.
- **Module state badges** — each Services module shows an **ON / OFF /
  UNAVAILABLE** badge (the bare checkbox was too easy to misread), with an
  **Enable all** reset.
- **Low-latency manual PTZ** — joystick/D-pad nudges apply on the capture thread
  instead of waiting for the next inference pass, so manual moves feel immediate.
- **Opt-in acceleration-aware PTZ lead** — `predict_accel_gain` (default off, in
  PTZ tuning) anticipates a subject *starting or stopping* for sharper following.

### Changed

- **Module switches persist and skip loading when off** — Detection / Tracking /
  Face / Pose / ReID choices are remembered across launches, and a disabled
  module's model is **never built at startup**, so turning one off genuinely
  reclaims its CPU and memory instead of resetting to all-on every launch.
- **Auto detector tier always works** — Auto is never greyed out or shown as "not
  downloaded" (it uses the bundled Fast model and auto-upgrades to a heavier tier
  once you download one), with clearer labels; if a selected heavier tier's file
  goes missing at runtime the engine falls back to the bundled default instead of
  going silent.
- **Lighter preview path** — the live video repaints with the cheap transform
  (smoothing/anti-aliasing only on HUD overlays), the preview converter does one
  fewer full-frame copy per frame (`Format_BGR888`), and ORT reserves a core for
  the capture/UI threads.
- **NDI cameras decode lighter on the CPU** — the receiver now requests the
  source's native format (usually 8-bit UYVY) instead of asking the NDI SDK to
  convert every frame to BGRA, so AutoPTZ does a single YUV→BGR pass instead of
  the SDK's conversion *plus* an alpha strip. The frame reader dispatches on the
  actual format (UYVY/UYVA/NV12/I420/YV12/BGRA/RGBA…) and falls back to the old
  universal BGRA path automatically for the rare 16-bit sources. Set
  `AUTOPTZ_NDI_COLOR_FORMAT=bgra` to force the previous behaviour.
- **Idle preview costs almost nothing** — each camera tile now repaints only when
  a new frame arrives or its state changes, instead of on every timer tick. On an
  Apple-Silicon Mac an idle / no-signal tile dropped from ~12% CPU to ~2–3%; live
  streams still update every frame.
- **Stalled cameras stop spinning the CPU** — when a source goes offline or
  returns no frames, the capture loop backs its retry off (up to 0.5 s) instead of
  polling at ~100 Hz, while a manual PTZ command still wakes it instantly so the
  joystick stays responsive during a reconnect.
- **Tracking-off cameras spin up less work** — the inference loop ran camera
  ego-motion (per-frame optical flow) and woke the appearance thread on *every*
  frame even with nothing to track. Both are now gated on what's actually on
  (ego-motion only while tracking, appearance only with Face/ReID on), so a camera
  with services off no longer burns a chunk of a core computing values nothing
  reads. Measured on a synthetic 1080p60 source with services off, the worker's
  idle CPU dropped from ~47% to ~29% of one core, with no change to follow
  behaviour.
- **Experimental: process-per-camera mode** (`AUTOPTZ_PROCESS_PER_CAMERA=1`, off by
  default) — runs each camera in its own OS process for true multi-core
  parallelism (GIL bypass) under many cameras, instead of threads in the UI
  process. Frames already cross via shared memory; commands/telemetry/identity
  cross via process queues. Each process loads its own models, so it trades RAM
  for parallelism — hence opt-in. Marked experimental: the plumbing is tested, but
  the throughput/RAM/real-camera behaviour wants validation on a multi-camera rig.
- **Face recognition uses far less memory** — it now runs on CPU and loads only
  the two models AutoPTZ uses (SCRFD detection + ArcFace embedding) instead of the
  full insightface pack on CoreML. Measured on macOS, AutoPTZ's real memory for two
  live cameras with all AI on dropped from ~1.1 GB to ~440 MB. Face runs only a few
  Hz on one target, so CPU costs nothing perceptible.

- **Torch-free model acquisition** — when a model isn't bundled, AutoPTZ downloads
  a prebuilt ONNX from the project's `models-v1` release instead of exporting via
  ultralytics. ultralytics is now a source-only fallback and is no longer shipped
  in installers (smaller builds; fixes the "ultralytics not installed" failure on
  packaged Windows/macOS/Linux).
- **Disabling a subsystem frees its model** — turning off Detection / Face / Pose /
  ReID (or removing a model) now unloads the in-memory session and reclaims its
  memory, instead of leaving it running until a restart.
- **The detector is the foundation** — face/pose/ReID and tracking are gated on a
  present detector model (greyed with a clear reason when it's missing), and a
  missing model degrades to live preview instead of silently still drawing boxes.
- Services panel: compact, higher-contrast model summary + read-only active tier.
- Model Manager dialog: flat list rows, dynamic tier enable/disable, bigger window.
- Detector-tier UI labels simplified to **Auto / Fast / Balanced / Accurate**.
- Menu bar consolidated from six top-level menus to four — Panels and Layouts moved
  under **View**; update controls grouped under **Help → Updates** (with a new
  startup auto-check toggle).
- Removed the unused per-camera `model_tier` and the redundant `aim_region` config
  fields; the worker reads the unified `framing` control directly.

### Fixed

- **Windows updates are visible and relaunch the app** — the silent install used
  `/VERYSILENT` (no window at all) and the installer's launch step was flagged
  `skipifsilent`, so an in-app update showed no installer/progress *and* never
  reopened AutoPTZ — you couldn't tell it had finished. It now installs with
  `/SILENT` (a progress window, still no wizard) and the installer relaunches
  AutoPTZ itself when done.
- **UI no longer hangs while cameras launch** — service probes use `find_spec`
  instead of importing boxmot/torch on the GUI thread, worker startup runs off the
  GUI thread (no lock held during camera open), and the detector never silently
  downloads/exports on the inference thread.
- **Detection/ReID stop when you delete models or disable them** — the shared
  inference sessions are released, so boxes stop being drawn and ReID stops; a
  re-downloaded model resumes automatically.
- Enabling Detection no longer triggers a surprise model download when automatic
  downloads are off.
- Linux CI no longer fails when a native library aborts at interpreter shutdown
  after the tests have already passed.
- **Deleting and downloading models works on Windows** — model files are no longer
  held open by onnxruntime when you remove or re-download them (the engine releases
  its sessions first, then mutates the cache, with a short retry for transient
  antivirus/handle locks). This previously failed on Windows while working on
  macOS/Linux, because only Windows refuses to delete an open file.
- **Pose "downloaded but not working" now explains itself** — a pose model that's
  present but whose runtime session fails to load is reported with the concrete
  reason (and a bundled pose model is recognized), instead of silently producing
  no keypoints.
- **USB camera menu stays current** — the device list refreshes in the background
  so a plugged/unplugged camera appears the first time you open the menu, and each
  row's checkmark now reflects the live camera list (it read a stale scan cache, so
  it lagged when toggling and could show cameras checked with none on the wall).
- **Accurate camera-permission message when running from source** (macOS) — a
  source run can't trigger the camera consent prompt (the bare Python binary has no
  usage entitlement; macOS attributes camera use to the launching terminal), so the
  message now says to grant the terminal or use the packaged app, instead of the
  misleading "denied" that only applies to the packaged build.
- **"App Mem" now reflects real memory** (macOS) — the status bar reported RSS,
  which counts memory-mapped model/framework files that are clean and reclaimable
  and inflated the figure ~3× (it looked like AutoPTZ was eating gigabytes it
  wasn't). It now reports `phys_footprint` — the same number Activity Monitor's
  "Memory" column shows — falling back to RSS off-macOS.
- **Selecting log lines is visible again** — selected rows in the Logs panel had no
  highlight (a per-item style was overriding the selection colour), so "select rows
  → Copy Selected" looked broken; selected rows now highlight properly.
- **Smoother Properties section expand/collapse** — the collapsible sections no
  longer stutter mid-motion or pop at the start/end: the content is pinned to one
  exact height per frame and animated to its true measured size.

### Dependencies

- onnxruntime 1.20.1 → 1.24.1 (required to load the IR v13 models onnx 1.21 writes),
  onnx 1.17.0 → 1.21.0, pytest 8.3.5 → 9.0.3, Pillow 11.2.1 → 12.2.0,
  msgpack 1.1.0 → 1.2.1.
- Installers no longer bundle `ultralytics` (and the `matplotlib`/`pandas` it
  dragged in); models are bundled or downloaded prebuilt instead.

## [2.0.0] — 2026-06-21

First stable release of the v2 architecture: a native Qt Widgets app with a
multi-threaded, multi-camera real-time tracking engine.

### Added

- **Multi-camera engine** — one threaded worker per camera (capture + inference
  threads), supervised, with shared-memory preview and typed command/telemetry.
- **Vision pipeline** — YOLO11 person detection (ONNX Runtime), BoT-SORT/ByteTrack
  tracking with an IoU fallback, OSNet appearance ReID re-acquisition, YOLO11
  pose, and face-recognition identity binding.
- **PTZ control** — motion prediction, one-euro smoothing, PD + velocity
  feed-forward, adjustable framing safe-zone, auto-zoom, and loss recovery, over
  VISCA-USB, VISCA-IP, ONVIF, and NDI backends.
- **Cross-platform acceleration** — automatic execution-provider selection with
  per-EP tuning: CoreML MLProgram (Apple ANE/GPU incl. AMD on Intel Macs),
  TensorRT FP16 + persistent engine cache, CUDA, DirectML, OpenVINO, CPU; full
  graph optimization and CPU-oversubscription-aware thread capping. Optional
  `gpu-nvidia` / `gpu-directml` / `openvino` requirement sets. Opt-in INT8
  detector quantization and drop-in RT-DETR support.
- **In-app updates** — checks GitHub Releases on startup and via
  **Help → Updates**; when a newer build exists it can download the matching OS
  asset and launch the installer. Stable releases by default, with an opt-in for
  pre-releases (beta/RC) and a skip-this-version option.
- **Installers** — macOS `.dmg`, Windows Inno Setup `.exe`, Linux AppImage, built
  and published by a tag-triggered release workflow.
- **Tooling** — `tools/bench/ep_compare.py` (per-EP latency) and `track_clip.py`
  (pipeline + ID-stability metrics); `packaging/make_icons.py`.

### Changed

- UI rebuilt as native Qt Widgets (PySide6), replacing the QML interface.
- Single source of truth for the version (`autoptz.__version__`); `pyproject.toml`
  and the UI read it dynamically.
- Documentation rewritten from scratch (README + `docs/` + CONTRIBUTING).

### Removed

- All config schema-migration and legacy-value compatibility code — 2.0.0 starts
  from a clean config schema (delete any old app-data database).

### Developer

- Repo-wide `ruff` lint + format; CI on macOS/Windows/Linux with a strict mypy
  gate on the typed core; pre-commit hooks.
- The four largest modules split into focused submodules:
  `camera_worker` → `engine/worker/{frame_source,stacks}`, `engine_client` →
  `ui/list_models`, `camera_tile` → `ui/widgets/tile_helpers`, `properties_panel`
  → `ui/widgets/properties_helpers`.

[Unreleased]: https://github.com/AutoPTZ/autoptz/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/AutoPTZ/autoptz/releases/tag/v2.0.0
