# AutoPTZ v2

AI-driven PTZ camera tracking — multi-camera, stable IDs, no cross-camera state bugs.

## Project layout

```
autoptz/           Python package (all v2 code lives here)
  config/          Pydantic models + SQLite config store
  engine/          Inference pipeline, PTZ backends, discovery
  ui/              PySide6 + QML user interface
  ui/qml/          QML files (CameraWall, CameraTile, ConfigDrawer, …)
requirements/      Dependency files split by purpose
  ui.txt           Minimal deps to run the UI (no ML stack)
  base.txt         Full deps including onnxruntime, boxmot, av
  macos.txt        macOS / Apple Silicon extras
  dev.txt          Test + lint tools
tests/             pytest test suite (360 tests)
docs/v2-rework/    Architecture design docs (01 – 11)
tools/             Bench/profiling scripts
pyproject.toml     Project metadata + tool config
```

## Quick start (macOS / Apple Silicon)

```bash
# 1. Clone
git clone <repo-url>
cd autoptz

# 2. Create a venv at the PROJECT ROOT (important — not inside autoptz/)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install UI dependencies (fast — no ML stack needed yet)
pip install -r requirements/ui.txt
pip install -e .

# 4. Launch
python -m autoptz
```

> **Important:** always activate the venv from the project root (`autoptz/`),
> not from inside `autoptz/autoptz/`. Both directories happen to be named
> `autoptz` which can be confusing. The one that contains `pyproject.toml` is
> the project root.

## Full ML stack install

Only needed once you want live detection + tracking (requires onnxruntime, boxmot, PyAV):

```bash
pip install -r requirements/base.txt -r requirements/macos.txt
pip install -e .
```

## Run

```bash
# Launch the UI
python -m autoptz

# Same thing, explicit UI submodule
python -m autoptz.ui

# Foundation selftest (verifies ORT EP + shared-memory + messaging)
python -m autoptz --selftest
```

## Test

```bash
pip install -r requirements/dev.txt
pytest                       # all tests
pytest tests/test_phase8.py  # Phase 8 (config/identity/layout)
pytest -k "not detect and not track and not ingest"  # skip ML tests
```

## Development

```bash
# Lint
ruff check autoptz tests

# Type-check
mypy autoptz

# Format
ruff format autoptz tests
```

## Config / data location

AutoPTZ stores its database at the platform config dir — no config files to edit:

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/AutoPTZ/autoptz.db` |
| Windows | `%APPDATA%\AutoPTZ\autoptz.db` |

All camera settings, presets, layouts, and identities are in this single SQLite file.
Export/import via **File → Save Show / Open Show** (saves a portable JSON bundle).

## Packaging (native apps)

Build AutoPTZ into a native, correctly-named app via PyInstaller. The macOS
build produces a real `.app` bundle whose `Info.plist` declares
`CFBundleName=AutoPTZ` — **this is what makes the macOS app menu read "AutoPTZ"
instead of "Python".** All build assets live under `packaging/`. Full details
(model/NDI bundling, CI) are in `docs/v2-rework/11-packaging-and-distribution.md`.

### macOS (Apple Silicon)

```bash
bash packaging/build_macos.sh          # → dist/AutoPTZ.app (unsigned)
```

The bundle is already named correctly. Sign + notarize **on your Mac** (needs
your Apple **Developer ID Application** certificate + **Team ID** — find them
with `security find-identity -v -p codesigning`):

```bash
codesign --deep --force --options runtime --timestamp \
  --entitlements packaging/entitlements.plist \
  --sign "Developer ID Application: <YOUR NAME> (<TEAM_ID>)" dist/AutoPTZ.app

ditto -c -k --keepParent dist/AutoPTZ.app dist/AutoPTZ.zip
xcrun notarytool store-credentials AUTOPTZ_NOTARY \
  --apple-id "<you@example.com>" --team-id "<TEAM_ID>" \
  --password "<app-specific-password>"
xcrun notarytool submit dist/AutoPTZ.zip --keychain-profile AUTOPTZ_NOTARY --wait
xcrun stapler staple dist/AutoPTZ.app
```

### Windows (x64)

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
# → dist\AutoPTZ\AutoPTZ.exe  (DirectML EP by default; -OneFile for a single exe)
```

Sign with `signtool` (Windows SDK) + your code-signing certificate. For NVIDIA
GPUs, install `requirements/gpu-nvidia.txt` (CUDA/TensorRT EPs) and rebuild.

> The YOLO11 detector ONNX **auto-downloads on first run**; pre-bundle it offline
> with `python -m tools.fetch_models --cache-dir autoptz/models` before building.
> The NDI runtime is a system library (not pip) — drop it in `packaging/ndi/` to
> bundle it.
