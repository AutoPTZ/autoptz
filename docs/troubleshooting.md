# Troubleshooting

Run with logs to see what's happening:

```bash
python -m autoptz --log-level INFO
```

The in-app **Logs** panel shows the same stream; **Camera Info** shows the active
execution provider + precision.

## No detection boxes / "live-preview-only"

The detector model couldn't be loaded. The log says why. Common causes:

- **First run, no network** — the YOLO11 ONNX downloads on first use. Pre-fetch
  offline: `python -m tools.fetch_models`, or set `AUTOPTZ_MODEL_PATH` to an
  existing ONNX, or `AUTOPTZ_MODEL_URL` to a mirror.
- **`onnxruntime`/`cv2` missing** — you installed `requirements/ui.txt` only.
  Run `python tools/install.py --editable`.

Tracking still works once a model is present; boxmot is optional (the tracker
falls back to a built-in IoU tracker).

## Running on CPU when a GPU is present

`make_session` logs the requested vs actual EP and any downgrade. Usual fixes:

- You have the wrong `onnxruntime` wheel. Only one can be installed — see
  [Installation](installation.md) and reinstall the right accelerator wheel.
- **NVIDIA** — the CUDA EP needs CUDA 12.x + cuDNN 9.x; the TensorRT EP needs
  TensorRT 10.x on the machine.
- Force/inspect with `AUTOPTZ_FORCE_EP=...` and confirm in **Camera Info**.

## First TensorRT launch is slow

TensorRT builds an engine on first run (can take minutes). It's cached
persistently afterward — the second launch is fast. Run
`python tools/bench/ep_compare.py` twice to confirm.

## Wrong camera opens / camera names are generic (macOS)

Run `python tools/install.py --editable` so the PyObjC AVFoundation packages from
`requirements/base.txt` are present. That lets cameras open by stable `uniqueID`
instead of OpenCV's divergent index order.

## Tracking is laggy or jittery

See [Configuration](configuration.md). Quick levers:

- **Laggy follow** — lower `aim_smoothing`, raise `lead_time_s` or `kp`.
- **Jittery** — raise `aim_smoothing`, enlarge the framing safe zone.
- **CPU-bound** — drop the detector tier, raise `detect_interval` (or leave
  `quality_floor=auto`), cap source `fps`.

## App menu shows "Python" (macOS, source run)

Cosmetic — only the packaged `.app` (with `CFBundleName=AutoPTZ`) fixes the menu
title. Build it with `bash packaging/build_macos.sh`.

## Reset everything

Delete the app-data dir (`~/Library/Application Support/AutoPTZ`,
`%APPDATA%\AutoPTZ`, or `~/.config/AutoPTZ`) to clear cameras, settings, and the
model cache. 2.0.0 uses a fresh config schema with no migration from older dev
builds, so an incompatible old database should be removed if you hit load errors.
