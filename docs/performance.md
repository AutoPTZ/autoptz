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
`AUTOPTZ_PRECISION` (`auto`/`fp32`/`fp16`/`int8`), `AUTOPTZ_ORT_INTRA_THREADS`.
These are also exposed as `HardwarePrefs` (`force_ep`, `precision`,
`intra_op_threads`).

### INT8 (opt-in)

Setting `precision = "int8"` runs a **dynamically-quantized** detector
(`ModelManager.ensure_detector_int8` caches a `*.int8.onnx` ~¼ the FP32 size). It
can speed up CPU inference but, for YOLO's conv-heavy graph, the win is modest and
it can cost a little accuracy — so it's **opt-in**, not a default. Measure it on
your footage first: `python tools/bench/ep_compare.py --precision int8`. It falls
back to FP32 automatically if quantization fails.

### RT-DETR (drop-in)

The detector's pre-NMS parser already understands RT-DETR's (NMS-free) COCO
output. To try it, export an RT-DETR ONNX (`rtdetr-l`/`rtdetr-x` are in the tier
map for `tools.fetch_models`) and point `AUTOPTZ_MODEL_PATH` at it. Benchmark vs
YOLO11 with `ep_compare`/`track_clip` on your hardware before switching.

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

## Multi-camera throughput & the batching roadmap

Today the inference pool builds **one** detector/pose/face model for the whole
app and shares it across cameras; ONNX Runtime's `run()` is thread-safe, so every
camera thread calls the same session concurrently. That already removes the
per-camera model duplication and keeps the accelerator busy.

**True batched inference** (collecting frames from N cameras into one
`run()` with batch=N) is the next step but is deliberately deferred — it requires:

1. Re-exporting the model with a **dynamic batch axis** (current exports are
   fixed `[1,3,640,640]`, which the letterbox/parser rely on).
2. A central **batch scheduler** that trades a little latency (waiting to fill a
   batch) for throughput, replacing the simple concurrent-call model.
3. End-to-end latency validation on a real multi-camera GPU fleet.

It's a throughput optimization on top of an already-shared model, not a missing
capability — evaluate it against your fleet's latency budget before adopting.
