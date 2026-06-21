# Requirements

Most installs should use the detector script from the repo root:

```bash
python tools/install.py --editable
python tools/install.py --dev --editable      # contributors
python tools/install.py --dry-run             # inspect the exact pip commands
```

The individual files stay small so installs remain reviewable. `base.txt` uses
normal pip environment markers for OS-specific packages such as macOS
AVFoundation support. GPU selection needs `tools/install.py` or an explicit
accelerator file because pip requirements cannot inspect your GPU hardware.

| File | When to install |
| --- | --- |
| `base.txt` | Always — full app: inference, tracking, PTZ, UI. Ships CPU ONNX Runtime (+ CoreML on macOS) and OS-specific camera helpers via markers. |
| `gpu-nvidia.txt` | NVIDIA GPU (Windows/Linux) — TensorRT + CUDA. Replaces the CPU `onnxruntime`. |
| `gpu-directml.txt` | AMD/Intel GPU on Windows — DirectML. Replaces the CPU `onnxruntime`. |
| `openvino.txt` | Intel CPU/iGPU (any OS) — OpenVINO. Replaces the CPU `onnxruntime`. |
| `dev.txt` | Contributors — ruff, mypy, pytest. |
| `packaging.txt` | Building installers — PyInstaller (see [docs/building.md](../docs/building.md)). |
| `ui.txt` | UI-only iteration — minimal, no ML stack. |

> Only **one** `onnxruntime*` wheel can be installed at a time. To switch
> accelerators, `pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
> onnxruntime-openvino` first, then install the one you want.
