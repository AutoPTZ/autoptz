# Installation

## Pre-built installers (recommended)

Download the latest build for your OS from the
[Releases page](https://github.com/AutoPTZ/autoptz/releases):

- **macOS** — `AutoPTZ-<version>-macos-arm64.dmg`. Open it and drag **AutoPTZ**
  to Applications. The builds are currently **unsigned**, so the first launch
  needs: right-click the app → **Open** → **Open** (or System Settings → Privacy
  & Security → *Open Anyway*).
- **Windows** — `AutoPTZ-<version>-windows-x64-setup.exe`. Run it; it installs
  Start-menu/desktop shortcuts and an uninstaller. SmartScreen may warn on the
  unsigned installer — **More info → Run anyway**.
- **Linux** — `AutoPTZ-<version>-linux-x86_64.AppImage`. `chmod +x` it and run.

The app checks GitHub Releases on startup and from **Help -> Check for Updates...**.
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

- `requirements/base.txt` — full stack: ONNX Runtime, OpenCV, PySide6, PyAV,
  ultralytics, boxmot, insightface, PTZ libs, plus OS-specific camera helpers
  through pip environment markers.
- `requirements/ui.txt` — UI-only (no ML stack), for quick UI work.
- `requirements/dev.txt` — pytest, ruff, mypy.
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
  markers, so native capture can bind cameras by stable uniqueID. NDI
  (`cyndilib`) still needs the NDI SDK runtime first.
- **Windows** — DirectML is the default GPU path because it works without CUDA.
  Force `--accelerator nvidia` only on machines with CUDA 12.x + cuDNN 9.x, and
  TensorRT 10.x if you want TensorRT.
- **Linux** — install Qt's system libs: `libegl1 libgl1 libxkbcommon0
  libdbus-1-3` and the `libxcb-*` set (see `docs/building.md`).

## Verify

```bash
python -m autoptz --selftest --log-level INFO
```

Prints the selected execution provider and exercises the shared-memory + message
plumbing.
