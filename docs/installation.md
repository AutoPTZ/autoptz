# Installation

## Pre-built installers (recommended)

Download the latest build for your OS from the
[Releases page](https://github.com/AutoPTZ/autoptz/releases):

- **macOS** — `AutoPTZ-<version>-macos-arm64.dmg` (Apple Silicon) or
  `…-macos-x86_64.dmg` (Intel). Open it and drag **AutoPTZ** to Applications.
  Signed + notarized releases open normally. If you build it yourself unsigned,
  the first launch needs: right-click the app → **Open** → **Open** (or System
  Settings → Privacy & Security → *Open Anyway*).
- **Windows** — `AutoPTZ-<version>-windows-x64-setup.exe`. Run it; it installs
  Start-menu/desktop shortcuts and an uninstaller. SmartScreen may warn on the
  unsigned installer — **More info → Run anyway**.
- **Linux** — `AutoPTZ-<version>-linux-x86_64.AppImage`. `chmod +x` it and run.

The app checks GitHub Releases on startup and from **Help → Updates → Check Now…**.
When a newer version exists, AutoPTZ downloads the matching asset for your OS,
starts it, and closes so the installer/new AppImage can finish. If that release
does not include your OS asset, AutoPTZ opens the release page instead.

## From source

Requires **Python 3.12+**.

```bash
git clone https://github.com/AutoPTZ/autoptz
cd autoptz
python3.12 -m venv .venv            # at the repo root, NOT inside autoptz/
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python tools/install.py --editable
python -m autoptz
```

The default `python tools/install.py` is **torch-free** (~2–3 GB lighter):
detection runs on ONNX Runtime, multi-object tracking uses the built-in
lightweight IoU tracker, and detector/pose weights provision via the
prebuilt-ONNX download. The two PyTorch-heavy fallbacks are opt-in extras:

```bash
python tools/install.py --editable                 # lean, torch-free default
python tools/install.py --full --editable          # + tracking + export extras
python tools/install.py --with-tracking --editable # boxmot only
python tools/install.py --with-export --editable   # ultralytics only
```

- **tracking** (`requirements/tracking.txt`, `boxmot`) — occlusion-robust
  BoT-SORT/DeepOCSORT/ByteTrack trackers and the OSNet ReID backend used for
  body-appearance recovery after occlusion.
- **export** (`requirements/export.txt`, `ultralytics`) — the YOLO11 `.pt` →
  ONNX export fallback, used only when the prebuilt ONNX download is unreachable.

- `requirements/base.txt` — torch-free core: ONNX Runtime, OpenCV, PySide6,
  PyAV, insightface, PTZ libs, plus OS-specific camera helpers through pip
  environment markers.
- `requirements/tracking.txt` / `requirements/export.txt` — optional torch
  extras (above); `--dev` and CI install both automatically (the test suite
  needs them).
- `requirements/ui.txt` — UI-only (no ML stack), for quick UI work.
- `requirements/dev.txt` — pytest, ruff, mypy (plus the tracking + export extras).
- `tools/install.py` — one readable install entry point that selects the right
  profile and prevents multiple `onnxruntime*` wheels from coexisting.

### Accelerators

The installer defaults to safe local choices: CoreML through the base wheel on
macOS, DirectML on Windows, NVIDIA on Linux when `nvidia-smi` is present, and
CPU otherwise. Review or override it with:

```bash
python tools/install.py --dry-run
python tools/install.py --accelerator cpu --editable
python tools/install.py --accelerator directml --editable   # Windows
python tools/install.py --accelerator nvidia --editable     # Windows/Linux
python tools/install.py --accelerator openvino --editable
```

Manual accelerator installs are still possible: install `requirements/base.txt`,
uninstall all `onnxruntime*` packages, then install exactly one of
`requirements/gpu-nvidia.txt`, `requirements/gpu-directml.txt`, or
`requirements/openvino.txt`.

### Platform notes

- **macOS** — `requirements/base.txt` installs PyObjC AVFoundation packages via
  markers, so native capture can bind cameras by stable uniqueID. NDI support is
  provided by the `cyndilib` package from `requirements/base.txt`.
- **Windows** — DirectML is the default GPU path because it works without CUDA.
  Force `--accelerator nvidia` only on machines with CUDA 12.x + cuDNN 9.x, and
  TensorRT 10.x if you want TensorRT.
- **Linux** — install Qt's system libs: `libegl1 libgl1 libxkbcommon0
  libdbus-1-3` and the `libxcb-*` set (see `docs/building.md`).

### Model setup

Release builds may start without bundled model weights. Use **Engine →
Models...** or run `python -m tools.fetch_models` to cache the detector
tiers and pose model before going offline. Use **Engine → Models...** to
delete AutoPTZ-managed detector/pose files from the local cache.
AutoPTZ does not silently fetch a missing detector tier when you switch models
unless **Automatically download a missing detector tier when I select it** is
enabled in that window.

The Services panel labels why each model is needed and disables feature controls
whose required model/dependency is missing. Face recognition and ReID model packs
are managed by their upstream packages and are not bundled or deleted by AutoPTZ
by default; review upstream model licenses before redistributing them. See
[`NOTICE.md`](../NOTICE.md).

## Verify

```bash
python -m autoptz --selftest --log-level INFO
```

Prints the selected execution provider and exercises the shared-memory + message
plumbing.
