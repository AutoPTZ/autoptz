# 07 — System Requirements & Scaling

These are **engineering estimates** for planning and must be confirmed by the benchmark harness
(Phase 9 deliverable). They assume the v2 compute‑budget policy from `03-vision-pipeline.md`:
detection at a steady cadence (not every frame on CPU), ReID/face on demand and at low rate, pose
only for the zoomed target, hardware‑accelerated decode, and per‑worker auto‑degrade under load.

## 7.1 What costs what

A "tracked camera" runs the full pipeline (detect + track + ReID + face + optional pose + PTZ
loop). A "preview‑only camera" just decodes + displays (cheap). The cost drivers, in order:

1. **AI inference** (detection dominates; ReID/face/pose are bursty) — the usual bottleneck.
2. **Video decode** — cheap on a GPU/Apple VideoToolbox; non‑trivial on CPU at high res. Use
   sub‑streams for AI.
3. **Memory / VRAM** — model weights + frame buffers per worker.
4. **Network** — IP cameras at ~4–12 Mb/s each; ensure NIC/switch headroom (prefer wired GbE).

Rule of thumb from research: `max_cameras = min(decode capacity, inference capacity, memory BW)`.
On modern NVIDIA, a single NVDEC decodes 100s of 1080p streams, so **inference is the limit**, not
decode. On Apple Silicon, unified memory + VideoToolbox makes mid/high tiers punch above their
spec.

## 7.2 Hardware tiers (target: 1080p sources, ~25–30 fps preview, AI at policy cadence)

| Tier | Example hardware | Inference path | Tracked cameras (est.) | Notes |
|---|---|---|---|---|
| **Entry / CPU‑only** | 8‑core laptop (i7‑12700H / Ryzen 7 / **M2**) | ORT CPU / OpenVINO / CoreML | **2–4** tracked + several preview‑only | Use YOLO26n @ 480, ByteTrack (ReID off), face `buffalo_s` @1 Hz, pose off |
| **Apple Silicon — mid** | **M2/M3 Pro** | CoreML EP (ANE+GPU) | **4–6** | VideoToolbox decode; unified memory helps |
| **Apple Silicon — high** | **M3 Max / M4 Max** (38 TOPS ANE, big GPU) | CoreML EP | **8–12** | best perf/watt; great fit for the brief |
| **NVIDIA — entry** | RTX 4060 / 4060 Ti (8–16 GB) | TensorRT EP + NVDEC | **6–10** | YOLO26s INT8/FP16; ReID on |
| **NVIDIA — mid** | RTX 4070 / 4080 (12–16 GB) | TensorRT EP + NVDEC | **12–20** | headroom for DeepOCSORT + pose |
| **NVIDIA — high** | RTX 4090 (24 GB) | TensorRT EP + NVDEC | **24–40+** | inference‑bound, not decode‑bound; keep tensor util ≤ ~60% when sharing NVDEC/NVENC |

These scale roughly linearly with detection cadence and model size: halving `detect_interval` or
moving YOLO26s→n materially increases camera count; enabling pose/DeepOCSORT on every camera
decreases it. The engine's auto‑degrade keeps things real‑time by trading quality for headroom.

## 7.3 Minimum / recommended specs

**Minimum (a few cameras):**
- CPU: 8 cores (Apple M1/M2 or Intel 12th‑gen/Ryzen 5000+).
- RAM: 16 GB. Disk: SSD, 5 GB free for app + models. OS: Windows 11 / macOS 13+.
- GPU: not required (CPU/OpenVINO/CoreML path), but any modern GPU/iGPU helps.

**Recommended (8+ cameras, smooth, ReID on):**
- Apple: M3 Max / M4 Max, 32–64 GB unified memory; **or**
- Windows: 8+ core CPU, 32 GB RAM, **RTX 4070+ (12 GB VRAM)**, NVMe SSD, wired GbE.
- For 16+ IP cameras: 32–64 GB RAM, RTX 4080/4090 (16–24 GB), 2.5 GbE/10 GbE if many high‑bitrate
  streams.

**Per‑camera resource budget (planning figures, validate):**
- ~0.5–1.0 GB RAM per tracked worker (models + buffers; shared model weights reduce this).
- ~0.6–1.2 GB VRAM per tracked worker on GPU tiers (less with shared engines / INT8).
- Network: size for `Σ camera_bitrate`; prefer the camera **sub‑stream** for AI to cut bandwidth
  and decode cost; use the main stream only for the focused/preview tile.

## 7.4 Levers that change the camera count

- **Detect cadence** (`detect_interval`) — biggest single lever on CPU tiers.
- **Model size** — YOLO26n vs s; OSNet x0_25 vs ain_x1_0; buffalo_s vs l.
- **ReID on/off** and **face rate** — turn down for more cameras.
- **Pose/auto‑zoom** — off on entry tiers (use bbox height).
- **Tracker** — ByteTrack (cheapest) → BoT‑SORT → DeepOCSORT (priciest).
- **Resolution for AI** — 480 vs 640 input; sub‑stream vs main.
- **Shared model sessions** — one ORT session shared across workers on the same GPU (batched)
  reduces VRAM and improves utilization at high camera counts (an advanced Phase 9 optimization).

## 7.5 Cross‑platform decode/accel matrix

| | macOS (Apple Silicon) | Windows + NVIDIA | Windows CPU/iGPU |
|---|---|---|---|
| Decode | VideoToolbox | NVDEC | D3D11VA / QSV / CPU |
| Inference EP | CoreML (ANE+GPU) | TensorRT → CUDA | DirectML / OpenVINO / CPU |
| NDI | cyndilib + NDI runtime | cyndilib + NDI runtime | cyndilib + NDI runtime |
| Packaging | notarized .app/.dmg | Inno/MSIX (+ optional CUDA build) | same Windows installer |

## 7.6 Benchmark harness (deliverable)

Ship `tools/bench/` that, given a synthetic or recorded multi‑camera load, measures per‑stage
latency, end‑to‑end glass‑to‑PTZ latency, sustained fps, and "max cameras at quality level X" on
the current machine, and prints a recommended tier. This turns the estimates above into measured,
per‑machine numbers and guards against regressions.
</content>
