# AUTOPTZ_* environment flags

Every ``AUTOPTZ_*`` environment variable the app reads, in one place. Verified by
grep against the source tree — if a flag isn't listed here, it isn't read
anywhere in `autoptz/`.

Categories:

- **Experimental Features dialog** — managed by **Help → Experimental Features…**.
  Persisted selections are applied to `os.environ` by the supervisor at the next
  engine start (`Supervisor._apply_experimental_env`; the registry lives in
  `autoptz/engine/runtime/experimental_flags.py`). The dialog groups these into
  sections (Experiments / Devices & tuning / Model overrides / Diagnostics).
  Setting any of them directly in the shell also works — the dialog only clobbers
  a key it actually persisted, never an operator-exported var it never touched,
  and it prunes keys it no longer recognises.
- **Operations override** — for advanced/offline/CI deployment, deliberately kept
  out of the dialog. Safe to set by the operator; the app runs fine without them.
- **Internal/dev** — set BY the app itself (derived from config, not meant to be
  hand-set) or intended for CI/local development and debugging only.

## Experimental Features dialog (Help → Experimental Features…)

**Experiments**

| Name | Meaning | Default |
| --- | --- | --- |
| `AUTOPTZ_UNIFIED_POSE` | Use one YOLO11-pose backbone for boxes + keypoints instead of a separate detector and pose pass. | `0` (off) |
| `AUTOPTZ_ASYNC_APPEARANCE` | Run face recognition and appearance ReID on their own thread, overlapping inference. | `1` (on) |
| `AUTOPTZ_PTZ_PUMP` | Drive PTZ commands from a dedicated background loop instead of inline on the inference thread. | `0` (off) |
| `AUTOPTZ_PTZ_SERIAL_AUTOPROBE` | Scan serial ports for a companion VISCA control port when a USB PTZ camera opens. | `1` (on) |
| `AUTOPTZ_TRUE_LATENCY_LEAD` | Lead the PTZ aim by the measured whole-pipeline dead time instead of just ingest+inference latency. | `0` (off) |
| `AUTOPTZ_MODEL_SERVER` | Run one shared detection server process that every camera delegates to (best-scaling mode for many cameras; self-healing — auto-respawn, fast-fail, local-detector fallback). Still experimental. | `0` (off) |

**Devices & tuning**

| Name | Meaning | Default |
| --- | --- | --- |
| `AUTOPTZ_REID_DEVICE` | Force the OSNet appearance (ReID) model onto a specific device (`cpu`/`mps`/`cuda`). | `""` (auto) |
| `AUTOPTZ_COREML_UNITS` | Target for the CoreML execution provider on Apple/Intel-Mac builds. | `""` (ALL) |
| `AUTOPTZ_NDI_COLOR_FORMAT` | Color format requested from NDI sources (`fastest`/`bgra`). | `fastest` |

**Model overrides**

| Name | Meaning | Default |
| --- | --- | --- |
| `AUTOPTZ_MODEL_PATH` | Use this detector ONNX file verbatim; skips download/export. | unset (auto-managed) |
| `AUTOPTZ_POSE_MODEL_PATH` | Use this pose ONNX file verbatim. | unset (auto-managed) |
| `AUTOPTZ_MODEL_URL` | Mirror/base URL to fetch a prebuilt detector ONNX from (air-gapped/offline installs); accepts a `{stem}`/`{model}` placeholder. | built-in release export URL |

**Diagnostics**

| Name | Meaning | Default |
| --- | --- | --- |
| `AUTOPTZ_MS_DIAG` | Verbose diagnostic logging for the shared detection server. | `0` (off) |
| `AUTOPTZ_SYNTH_DEBUG` | Verbose logging for the synthetic-camera ingest path (bench/Mark tooling). | `0` (off) |

## Operations override

Deliberately kept out of the dialog — they change library thread pools, the
profile location, launch behaviour, or the update source, which are operator/CI
concerns rather than per-run tuning.

| Name | Meaning | Default |
| --- | --- | --- |
| `AUTOPTZ_MODEL_URL_<STEM>` | Per-model override of `AUTOPTZ_MODEL_URL` for one specific weight (e.g. `AUTOPTZ_MODEL_URL_YOLO11M`). | unset |
| `AUTOPTZ_FACE_MODEL` | InsightFace model pack name used for face recognition (also the pack the Model Manager downloads/removes). | `buffalo_l` |
| `AUTOPTZ_DB_PATH` | Override the ConfigStore SQLite path. | platform user-data dir |
| `AUTOPTZ_UPDATE_REPO` | Override the GitHub repo the in-app updater checks for releases. | the AutoPTZ repo |
| `AUTOPTZ_NO_MODEL_EXPORT` | Disable the Ultralytics/Torch ONNX export fallback; used by CI and locked-down installs where `torch` isn't available. | unset (export allowed) |
| `AUTOPTZ_SKIP_CAMERA_PREFLIGHT` | Skip the startup camera-availability preflight (NDI/RTSP-only or headless runs). | unset (preflight runs) |

## Internal/dev

Set by the app itself, or only used by CI / the Mark harness / logging.

| Name | Meaning | Default |
| --- | --- | --- |
| `AUTOPTZ_FORCE_EP` | Force a specific ONNX Runtime execution provider. **Set BY the supervisor** at engine start from the hardware/config selection; hand-set only to override for one run. | unset (auto) |
| `AUTOPTZ_PRECISION` | Force inference precision (`auto`/`fp32`/`fp16`/`int8`). **Set BY the supervisor** from config; hand-set only to override. | `auto` |
| `AUTOPTZ_ORT_INTRA_THREADS` | ONNX Runtime intra-op thread cap. **Set BY the supervisor** at engine start from the detected core count; hand-set only for benchmarking. | unset (auto) |
| `AUTOPTZ_CV2_THREADS` | OpenCV thread cap. **Set BY the supervisor** alongside `AUTOPTZ_ORT_INTRA_THREADS` to prevent thread-pool oversubscription. | unset (auto) |
| `AUTOPTZ_MARK_GT` | Enable AutoPTZ Mark's ground-truth synthetic-camera scoring path. | unset (off) |
| `AUTOPTZ_MARK_NO_AUTOSTART` | Skip Mark's engine auto-start (dev/test harness convenience). | unset (off) |
| `AUTOPTZ_START_MARK` | Launch straight into AutoPTZ Mark instead of the normal app on startup. | unset (off) |
| `AUTOPTZ_NO_COLOR` / `NO_COLOR` | Disable ANSI color in log output. | unset (color on when a TTY) |
| `AUTOPTZ_FORCE_COLOR` | Force ANSI color in log output even when not a TTY (e.g. piped CI logs). | unset (off) |

> **Why the four `FORCE_EP` / `PRECISION` / `ORT_INTRA_THREADS` / `CV2_THREADS`
> vars aren't in the Experimental dialog:** the supervisor already writes them at
> engine start from the hardware/performance selection. Surfacing them in the
> dialog would create a second writer for the same variable and let the two drift,
> so they stay operator-only.

`AUTOPTZ_PROCESS_PER_CAMERA` no longer exists as a code path: the standalone
model-per-child experiment it gated is retired (superseded by the shared detection
server, `AUTOPTZ_MODEL_SERVER`) and its compatibility alias was removed. See
`docs/engineering/retired-experiments.md`.
