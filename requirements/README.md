# Requirements

Install **`base.txt`** plus the **one** accelerator file for your hardware. The
others are situational. See [docs/installation.md](../docs/installation.md).

| File | When to install |
| --- | --- |
| `base.txt` | Always — full app: inference, tracking, PTZ, UI. Ships CPU ONNX Runtime (+ CoreML on macOS). |
| `gpu-nvidia.txt` | NVIDIA GPU (Windows/Linux) — TensorRT + CUDA. Replaces the CPU `onnxruntime`. |
| `gpu-directml.txt` | AMD/Intel GPU on Windows — DirectML. Replaces the CPU `onnxruntime`. |
| `openvino.txt` | Intel CPU/iGPU (any OS) — OpenVINO. Replaces the CPU `onnxruntime`. |
| `macos.txt` | macOS extras — native AVFoundation capture + friendly camera names. |
| `dev.txt` | Contributors — ruff, mypy, pytest. |
| `packaging.txt` | Building installers — PyInstaller (see [docs/building.md](../docs/building.md)). |
| `ui.txt` | UI-only iteration — minimal, no ML stack. |

> Only **one** `onnxruntime*` wheel can be installed at a time. To switch
> accelerators, `pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
> onnxruntime-openvino` first, then install the one you want.
