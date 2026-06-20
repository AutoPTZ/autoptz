<div align="center">

<img src="autoptz/assets/AutoPTZLogo.png" alt="AutoPTZ" width="140" />

# AutoPTZ

**AI-driven PTZ camera tracking — detect people, lock onto a target, and move the camera to follow them automatically.**

[Installation](docs/installation.md) · [Configuration](docs/configuration.md) · [Performance](docs/performance.md) · [Building](docs/building.md) · [Architecture](docs/architecture.md) · [Troubleshooting](docs/troubleshooting.md)

</div>

---

AutoPTZ is a cross-platform desktop app (native Qt Widgets / PySide6) that runs
a real-time vision pipeline per camera — **detect → track → re-identify → pose →
aim → drive PTZ** — and sends smooth pan/tilt/zoom commands so a PTZ camera keeps
the chosen person framed. It is built for live production: multi-camera, stable
target identity across occlusions, and graceful degradation when a model or
device is missing (it always keeps live preview).

## Highlights

- **Multi-camera** — each camera runs its own worker; identities stay stable per
  camera with no cross-camera state bugs.
- **Identity-gated tracking** — click a person to target them; optional face
  recognition + appearance ReID re-bind the right person after occlusions.
- **Smooth PTZ control** — motion prediction, one-euro smoothing, PD + velocity
  feed-forward, an adjustable framing "safe zone", auto-zoom, and loss recovery.
- **Runs anywhere, fast** — ONNX Runtime picks the best accelerator per platform
  (Apple CoreML, NVIDIA TensorRT/CUDA, Windows DirectML, Intel OpenVINO, CPU)
  with per-EP tuning (FP16, persistent TensorRT engine cache, full graph
  optimization). See [Performance](docs/performance.md).
- **PTZ backends** — VISCA over USB, VISCA over IP, ONVIF, and NDI.
- **In-app updates** — checks GitHub Releases and points you at the new build.

## Quick start (from source)

Requires **Python 3.12+**.

```bash
git clone https://github.com/AutoPTZ/autoptz
cd autoptz

# Create a venv at the PROJECT ROOT (not inside autoptz/)
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Full stack (detection + tracking + UI):
pip install -r requirements/base.txt
pip install -e .

python -m autoptz                # launch the app
python -m autoptz --selftest     # verify the foundations and exit
```

The first launch downloads the detector model (YOLO11) into the platform app-data
dir; without it the app still runs in live-preview-only mode.

### Picking your accelerator

The standard `requirements/base.txt` ships CPU ONNX Runtime (plus Apple CoreML on
macOS arm64). For a GPU, install **one** accelerator wheel in its place:

| Platform / GPU            | Install                                                                 |
| ------------------------- | ---------------------------------------------------------------------- |
| Apple Silicon / Intel Mac | nothing extra — CoreML ships in the base wheel                          |
| NVIDIA (Win/Linux)        | `pip install -r requirements/gpu-nvidia.txt` (TensorRT → CUDA)         |
| AMD / Intel GPU (Windows) | `pip install -r requirements/gpu-directml.txt`                         |
| Intel CPU/iGPU (any OS)   | `pip install -r requirements/openvino.txt`                             |

Only one `onnxruntime*` wheel can be installed at a time — see
[Performance](docs/performance.md).

## Installers

Pre-built installers are published on the
[Releases page](https://github.com/AutoPTZ/autoptz/releases): a macOS `.dmg`, a
Windows installer (`.exe`), and a Linux `AppImage`. To build them yourself see
[docs/building.md](docs/building.md).

## Documentation

| Doc | What's in it |
| --- | --- |
| [Installation](docs/installation.md)   | From source + pre-built installers, per platform. |
| [Configuration](docs/configuration.md) | Every tuning knob: model tier, detect interval, framing, smoothing, PTZ gains. |
| [Performance](docs/performance.md)     | Cross-platform device/precision matrix + the `ep_compare` benchmark. |
| [Building](docs/building.md)           | PyInstaller bundles → DMG / Windows installer / AppImage. |
| [Architecture](docs/architecture.md)   | Module map and the per-frame data flow. |
| [Troubleshooting](docs/troubleshooting.md) | Common issues (no boxes, wrong camera, slow tracking). |
| [Contributing](CONTRIBUTING.md)        | Dev setup, lint/type/test gates, branch policy. |

## License

See [LICENSE.md](LICENSE.md).
