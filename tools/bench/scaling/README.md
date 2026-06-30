# Multi-camera streaming/scaling benchmark

Reproduces the A/B for the **shared model-server** architecture candidate
(`AUTOPTZ_MODEL_SERVER`). It drives the *real* `Supervisor` + engine over N
independent NDI sources at 1080p30 and reports delivered fps, steady-state
app-induced drops, source/SDK drop counters, NDI FourCC/copy/convert cost,
end-to-end latency, and App/System CPU + RAM — so threaded vs
model-server can be compared on the axes that matter for
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

# shared model-server candidate
AUTOPTZ_MODEL_SERVER=1 python tools/bench/scaling/ndi_receiver.py 16 full 18 40
```

`ndi_receiver.py <N> <profile> <run_s> <warmup_s> [json_path]` — `profile` is `full`
(detect+track+control) or `streams` (ingest only). Give model-server runs a
longer warmup; spawned children load models before steady state.

## Measure detection health, not just capture

`ndi_receiver.py` reports `detect_active_pct` / `detect_ms_median` / `stall_s_max`,
and the model-server logs `served=N/s` under `AUTOPTZ_MS_DIAG=1`. **Always check one
of these.** The fps / drops / e2e numbers come from the *capture* path, which runs on
its own thread — so a model-server that is serving **zero** detections (e.g. a wiring
bug where readers never attach) still shows great fps/latency. The first model-server
benchmark looked perfect (30 fps, 48 ms) while `served=0`: it was measuring capture
with detection dead. Trust `served/s > 0` (or `detect_active_pct ≈ 100`) as the proof
detection actually ran.

For the 2.2 release gate, `drops_steady_window` is the same value as
`app_induced_drops_steady_window`. `source_drop_est_steady_window` and
`ndi_dropped_video_steady_window` are still emitted because they help diagnose source
pacing, NDI SDK queue loss, and conversion/copy regressions, but they are not the
same as app-induced capture drops.

## Measured result — detection VERIFIED alive (Apple Silicon, yolo11s, NDI 1080p30)

| N=16 | fps/cam | drops/s | e2e latency | sys CPU | RAM | detect served/s |
|------|---------|---------|-------------|---------|-----|-----------------|
| threaded     | 29.8\* | 237 | 1635 ms | 43% | 5.3 GB | — |
| per-process  | 4.2 💥 | 807 | 580 ms  | (under-counted) | 13.7 GB | — |
| model-server | **23.8** | 146 | **54 ms** | 59% | **10.9 GB** | ~7–18 |

\* threaded "fps" counts frames *arriving*; the 1.6 s e2e latency + 237 drops/s show
the pipeline ~1.6 s behind — the cause of the "bouncing" aim. The model-server is the
best-scaling option (low, stable latency; no RAM cliff; graceful degradation vs
per-process's 4.2 fps collapse) — but it does **not** hold 30 fps at 16 cams, and
detection is sparse (~0.5 det/cam/s, the single ANE under-fed by per-camera process
overhead), so **predictive tracking is required** for smooth aim at high camera counts.
See `docs/research/streaming-tracking-redesign.md`.
