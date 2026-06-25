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

The **default install is torch-free** (~2–3 GB lighter): detection runs on ONNX
Runtime, multi-object tracking uses the built-in lightweight IoU tracker, and
detector/pose weights provision via the prebuilt-ONNX download. The two
torch-heavy fallbacks are opt-in extras:

```bash
python tools/install.py --editable                 # lean, torch-free default
python tools/install.py --full --editable          # + tracking + export extras
python tools/install.py --with-tracking --editable # + boxmot tracking only
python tools/install.py --with-export --editable   # + ultralytics export only
```

`--dev` (and CI) install both extras automatically because the test suite needs
them — `dev.txt` references `tracking.txt` and `export.txt`.

| File | When to install |
| --- | --- |
| `base.txt` | Always — torch-free core: ONNX inference, IoU tracking, PTZ, UI. Ships CPU ONNX Runtime (+ CoreML on macOS) and OS-specific camera helpers via markers. |
| `requirements.txt` | Compatibility entry point for tools that expect the conventional filename, including GitHub Dependency Graph. Includes `base.txt`. |
| `tracking.txt` | Optional extra — `boxmot` adds BoT-SORT/DeepOCSORT/ByteTrack + OSNet ReID for occlusion-robust tracking. Pulls in PyTorch. `--with-tracking` / `--full`. |
| `export.txt` | Optional extra — `ultralytics` adds the YOLO11 `.pt` → ONNX export fallback (used only when the prebuilt download is unreachable). Pulls in PyTorch. `--with-export` / `--full`. |
| `gpu-nvidia.txt` | NVIDIA GPU (Windows/Linux) — TensorRT + CUDA. Replaces the CPU `onnxruntime`. |
| `gpu-directml.txt` | AMD/Intel GPU on Windows — DirectML. Replaces the CPU `onnxruntime`. |
| `openvino.txt` | Intel CPU/iGPU (any OS) — OpenVINO. Replaces the CPU `onnxruntime`. |
| `dev.txt` | Contributors — ruff, mypy, pytest. Also pulls in `tracking.txt` + `export.txt` (the test suite needs them). |
| `packaging.txt` | Building installers — PyInstaller (see [docs/building.md](../docs/building.md)). |
| `ui.txt` | UI-only iteration — minimal, no ML stack. |

> Only **one** `onnxruntime*` wheel can be installed at a time. To switch
> accelerators, `pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
> onnxruntime-openvino` first, then install the one you want.
