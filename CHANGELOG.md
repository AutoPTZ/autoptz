# Changelog

All notable changes to AutoPTZ are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/).

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

[2.0.0]: https://github.com/AutoPTZ/autoptz/releases/tag/v2.0.0
