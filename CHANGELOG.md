# Changelog

All notable changes to AutoPTZ are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — v2.0.0a0

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
