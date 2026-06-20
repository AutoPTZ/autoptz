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

The app checks GitHub Releases on startup and from **Help → Check for Updates…**
and opens the download page when a newer version exists.

## From source

Requires **Python 3.12+**.

```bash
git clone https://github.com/AutoPTZ/autoptz
cd autoptz
python3.12 -m venv .venv            # at the repo root, NOT inside autoptz/
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements/base.txt
pip install -e .
python -m autoptz
```

- `requirements/base.txt` — full stack: ONNX Runtime, OpenCV, PySide6, PyAV,
  ultralytics, boxmot, insightface, PTZ libs.
- `requirements/ui.txt` — UI-only (no ML stack), for quick UI work.
- `requirements/dev.txt` — pytest, ruff, mypy.

### Accelerators

Install **one** accelerator wheel in place of the base CPU `onnxruntime`
(uninstall the previous one first):

```bash
# NVIDIA (Windows/Linux): TensorRT + CUDA
pip uninstall -y onnxruntime && pip install -r requirements/gpu-nvidia.txt
# AMD/Intel GPU (Windows): DirectML
pip uninstall -y onnxruntime && pip install -r requirements/gpu-directml.txt
# Intel CPU/iGPU (any OS): OpenVINO
pip uninstall -y onnxruntime && pip install -r requirements/openvino.txt
```

macOS needs nothing extra — CoreML ships in the base wheel. See
[Performance](performance.md).

### Platform notes

- **macOS** — `requirements/macos.txt` adds pyobjc frameworks for native
  AVFoundation capture + friendly camera names. NDI (`cyndilib`) needs the NDI
  SDK runtime and is commented out by default.
- **Windows** — `pygrabber` gives friendly camera names; install
  `requirements/gpu-nvidia.txt` for CUDA/TensorRT (needs CUDA 12.x + cuDNN 9.x,
  TensorRT 10.x).
- **Linux** — install Qt's system libs: `libegl1 libgl1 libxkbcommon0
  libdbus-1-3` and the `libxcb-*` set (see `docs/building.md`).

## Verify

```bash
python -m autoptz --selftest --log-level INFO
```

Prints the selected execution provider and exercises the shared-memory + message
plumbing.
