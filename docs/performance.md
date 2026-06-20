# Performance & cross-platform acceleration

The whole product is real-time, so inference speed and stability matter. AutoPTZ
runs detection/pose through **ONNX Runtime**, which selects the best **execution
provider (EP)** for your hardware and tunes the session per EP.

## What's automatic

`engine/runtime/inference.py:make_session()` builds every session with:

- **Full graph optimization** (`ORT_ENABLE_ALL`).
- **Per-EP acceleration options:**
  - **CoreML** → `MLProgram` + `MLComputeUnits=ALL` (Apple Neural Engine / GPU,
    including the AMD GPU on Intel Macs via Metal).
  - **TensorRT** → FP16 + a **persistent engine cache** so the multi-minute
    engine build happens once, not every launch, plus a timing cache.
  - **CUDA** → cuDNN heuristic conv-algo search.
  - **DirectML** → device selection. **OpenVINO** → `AUTO` device, FP16.
- **Thread capping** — intra-op threads default to `cores ÷ cameras` so several
  camera workers don't oversubscribe the CPU.
- **Safe fallback** — a provider that rejects its options is retried bare, and any
  GPU failure downgrades to CPU. Each step is logged with the effective EP +
  precision, surfaced in the **Camera Info** panel and the **About** dialog.

## Choosing an accelerator

Install **one** `onnxruntime*` wheel (they conflict — uninstall the previous one
first):

| Target | Install | EP order | Precision |
| --- | --- | --- | --- |
| Apple Silicon | base wheel | CoreML → CPU | FP16 (MLProgram) |
| Intel Mac + AMD GPU | base wheel | CoreML → CPU | FP16 |
| Windows / Linux + NVIDIA | `requirements/gpu-nvidia.txt` | TensorRT → CUDA → CPU | FP16 + engine cache |
| Windows + AMD/Intel GPU | `requirements/gpu-directml.txt` | DirectML → CPU | FP16 |
| Intel CPU/iGPU (any OS) | `requirements/openvino.txt` | OpenVINO → CPU | FP16 |
| CPU only (any OS) | base wheel | CPU | FP32 |

Overrides (per the supervisor → env wiring): `AUTOPTZ_FORCE_EP`,
`AUTOPTZ_PRECISION` (`auto`/`fp32`/`fp16`), `AUTOPTZ_ORT_INTRA_THREADS`. These are
also exposed as `HardwarePrefs` (`force_ep`, `precision`, `intra_op_threads`).

## Tuning for stability

For the smoothest tracking on constrained machines (see
[Configuration](configuration.md) for all knobs):

- Lower the **model tier** (`nano` is default and fastest) and/or input size.
- Raise **detect interval** (run detection every N frames; the Kalman tracker
  interpolates between) or let **quality floor = auto** adapt it to the frame
  budget.
- Cap **source fps** to what the camera + accelerator can sustain.

## Benchmarking your machine

`tools/bench/ep_compare.py` times every available EP through the *real*
`make_session` factory, so you measure exactly what the app uses:

```bash
python tools/bench/ep_compare.py --runs 50           # auto-resolves the cached model
python tools/bench/ep_compare.py --model path.onnx --precision fp16
```

It prints requested-vs-actual EP, mean/median latency, and FPS. Run it twice on
NVIDIA to confirm the TensorRT engine cache makes the second launch fast.

`tools/bench/track_clip.py` benchmarks the full detect + track pipeline (and
ID-stability metrics) on a recorded clip.

## Roadmap (evaluate, then enable)

These are deliberately **not** on by default for 2.0.0 — they either need
per-hardware validation or are a net win only for specific setups. Use the bench
harness to evaluate on your target hardware before enabling:

- **INT8 quantization** — dynamic INT8 helps transformer-heavy graphs but is
  marginal for YOLO's conv-dominated network and can cost accuracy without a
  static calibration set; evaluate per model/footage.
- **RT-DETR / alternative detectors** — NMS-free, anchor-free; promising on GPU.
  Drop in via `AUTOPTZ_MODEL_PATH` once validated on your footage.
- **Multi-camera batched inference** — batching frames across cameras that share
  one EP raises GPU utilization; benefits scale with camera count and need
  end-to-end latency validation on the real fleet.
