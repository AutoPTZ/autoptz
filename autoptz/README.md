# AutoPTZ v2 — package README

AutoPTZ v2 is an AI-driven PTZ camera tracking system built on ONNX Runtime,
BoxMOT, InsightFace, PySide6/QML, and shared-memory inter-process
communication. This document covers installation per platform.

## Requirements

- Python 3.12+
- See `requirements/` for pinned versions.

---

## macOS (Apple Silicon, M-series) — recommended path

```bash
# 1. Create and activate a venv
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install base + macOS extras
pip install -r requirements/base.txt -r requirements/macos.txt

# 3. (Optional) Install NDI SDK runtime from https://ndi.video/tools/ndi-downloads/
#    then: pip install cyndilib>=0.0.10

# 4. Verify CoreML EP is available (expected on Apple Silicon)
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# → ['CoreMLExecutionProvider', 'CPUExecutionProvider']

# 5. Install autoptz in editable mode
pip install -e . --no-deps

# 6. Run selftest
python -m autoptz --selftest
```

**Inference acceleration:** the standard `onnxruntime` wheel on macOS arm64
ships with the CoreML Execution Provider, which routes computation through
Apple's Neural Engine (ANE) and GPU. No extra packages required.
VideoToolbox hardware decode is used automatically by OpenCV/FFmpeg.

---

## Windows x64 — CPU / Intel iGPU

```bash
# 1. Create and activate a venv
python -m venv .venv && .venv\Scripts\activate

# 2. Install base dependencies
pip install -r requirements/base.txt

# 3. Install autoptz
pip install -e . --no-deps

# 4. Verify available EPs
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# 5. Run selftest
python -m autoptz --selftest
```

The `DirectML` EP (AMD/Intel/NVIDIA via D3D12) and `OpenVINO` EP are selected
automatically if the corresponding runtimes are installed.

---

## Windows x64 — NVIDIA GPU (TensorRT / CUDA)

```bash
# Prerequisites: CUDA 12.x + cuDNN 9.x + TensorRT 10.x installed from NVIDIA

# 1. Base venv and install
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements/base.txt

# 2. Swap in the GPU onnxruntime build
pip uninstall onnxruntime -y
pip install -r requirements/gpu-nvidia.txt

# 3. Install autoptz
pip install -e . --no-deps

# 4. Verify TensorRT/CUDA EPs appear
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# → ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider', ...]

# 5. Run selftest
python -m autoptz --selftest
```

TRT engine caches are written to the model directory on first run and reused
on subsequent runs.

---

## Linux x64 — NVIDIA GPU

Same as Windows NVIDIA path above, but use `source .venv/bin/activate`.
HW decode uses NVDEC via FFmpeg/OpenCV.

---

## Development install

```bash
pip install -r requirements/base.txt -r requirements/dev.txt
pip install -e . --no-deps
ruff check autoptz/ tests/
mypy autoptz/engine/runtime/ tests/
pytest tests/ -v
```

---

## Selftest output (expected)

```
AutoPTZ v2 — selftest
==================================================
[1] EP selection:         CoreMLExecutionProvider
[2] SHM ring buffer:      OK  wrote seq=0 shape=[360, 640, 3]  reader got seq=0 sum=... ✓
[3] Telemetry round-trip: OK  ... bytes ✓
[4] Command round-trip:   OK  ... bytes ✓
==================================================
All selftest checks passed.
```
