# Changelog

All notable changes to AutoPTZ are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Models ship inside the installer** — the detector tiers (Fast / Balanced /
  Accurate) and the pose model are now bundled with the app, so detection and
  pose work on first launch with **no download, network, or setup**. They appear
  in the Model Manager as *Included* and can't be removed.
- **Model Manager** (Engine → Models) — review every AutoPTZ-managed model with
  per-row status, download or remove the ones you want, pick the active detector
  tier (un-downloaded tiers are greyed out), with live download progress.
- **Seamless in-app updates** — the updater now shows a **download progress bar**,
  and on Windows installs **silently** (no setup wizard) and restarts, the
  closest to a "it just updated" experience.
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

### Changed

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
