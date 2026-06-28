# Multi-camera streaming/scaling benchmark

Reproduces the A/B that decided the **shared model-server** architecture
(`AUTOPTZ_MODEL_SERVER`). It drives the *real* `Supervisor` + engine over N
independent NDI sources at 1080p30 and reports delivered fps, steady-state
frame-drops, end-to-end latency, and App/System CPU + RAM — so threaded vs
per-process vs model-server can be compared on the axes that matter for
tracking smoothness (latency) and scale (CPU/RAM at high camera counts).

## Pieces

| file | role |
|------|------|
| `ndi_sender.py`        | Broadcasts N independent NDI sources from a separate process (so the receiver's CPU reflects only the app's receive+inference load). Logs achieved per-sender fps. |
| `ndi_receiver.py`      | Drives the real `Supervisor` over the N `ndi://` sources, pumps the engine, and prints a `RESULT_JSON` summary (fps / drops / e2e latency / CPU / RAM). Prints `MS_ENGAGED True` when the model-server actually spawned. |
| `proto_model_server.py`| Standalone prototype that first proved the architecture's properties (one model set, GIL-free capture, ANE-bound detection) outside the full engine. Kept as a minimal reference. |

## Run an A/B

Senders run in their own process; start them first, then the receiver. The
receiver's mode is chosen by env var:

```bash
# threaded (baseline)
python tools/bench/scaling/ndi_sender.py 16 1920 1080 30 90 &
python tools/bench/scaling/ndi_receiver.py 16 full 18 16

# one-process-per-camera
AUTOPTZ_PROCESS_PER_CAMERA=1 python tools/bench/scaling/ndi_receiver.py 16 full 18 60

# shared model-server (the scalable one)
AUTOPTZ_MODEL_SERVER=1 python tools/bench/scaling/ndi_receiver.py 16 full 18 40
```

`ndi_receiver.py <N> <profile> <run_s> <warmup_s>` — `profile` is `full`
(detect+track+control) or `streams` (ingest only). Give per-process / model-
server runs a longer warmup; spawned children load models before steady state.

## Measured result (Apple Silicon, yolo11s, NDI 1080p30)

| N=16 | fps/cam | drops/s | e2e latency | sys CPU | RAM |
|------|---------|---------|-------------|---------|-----|
| threaded     | 29.7\* | 232.9 | 1716 ms | 51.6% | 5.6 GB |
| per-process  | 14.8   | 202.8 | 94 ms   | 100%  | 15.7 GB |
| model-server | **30.0** | **9.7** | **54 ms** | 49.4% | 11.4 GB |

\* threaded "fps" counts frames *arriving*; the 1.7 s e2e latency + 233 drops/s
show the pipeline running ~1.7 s behind — the cause of the "bouncing" aim. Only
the model-server holds full 30 fps with near-real-time latency at 16 cameras,
at half the per-process CPU and without its RAM cliff.
