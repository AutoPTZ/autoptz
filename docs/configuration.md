# Configuration

Settings live in a per-user SQLite database in the platform app-data dir
(`~/Library/Application Support/AutoPTZ` on macOS, `%APPDATA%\AutoPTZ` on Windows,
`~/.config/AutoPTZ` on Linux). Per-camera settings (source, tracking, PTZ) are edited
in the **Properties** panel; app-wide settings (hardware/EP and the shared detector
model) are in the **Services** panel. This page documents what each knob does.
Defaults are the validated, broadcast-sane starting point; one concept = one control.

## Source

| Setting | Default | Notes |
| --- | --- | --- |
| `type` | `usb` | `usb`, `rtsp`, `onvif`, or `ndi`. |
| `address` | — | USB index, RTSP/ONVIF URL, or NDI name. |
| `fps` | `30` | Target capture rate; cap it to what the camera + accelerator sustain. |

## Tracking & detection

| Setting | Default | Notes |
| --- | --- | --- |
| `tracker` | `botsort` | `botsort`, `deepocsort`, or `bytetrack` (falls back to a built-in IoU tracker if boxmot is absent). |
| `tracking_mode` | `stable` | `stable` holds the target through occlusions via appearance ReID; `responsive` follows the freshest track with less delay. |
| `detect_interval` | `1` | Run detection every N frames; the tracker interpolates in between. Higher = cheaper. |
| `quality_floor` | `auto` | `auto` adapts the detect interval to the frame budget; `high`/`balanced`/`low` pin it. |
| `reid_threshold_hi` / `lo` | `0.60` / `0.35` | Hysteresis for appearance re-acquisition (enter / maintain lock). |
| `coast_window_ms` | `300` | How long a lost track is coasted before it's dropped. |
| `framing` | `upper_body` | The single "shot composition" control. Sets the vertical aim point and (mirrored into `zoom_framing`) the auto-zoom tightness: `face`, `head_shoulders`, `upper_body`, `full_body`. |
| `aim_body_mode` | `torso` | `torso` ignores arms/limbs (steadier aim); `full_silhouette` uses the whole box. |
| `min_detection_size_frac` | `0.05` | Drop people smaller than this fraction of frame height (ignore distant specks). |
| `face_confirm` | `false` | Require a face match before binding a target. |

## PTZ control

| Setting | Default | Notes |
| --- | --- | --- |
| `max_pan_speed` / `max_tilt_speed` / `max_zoom_speed` | `0.5` / `0.5` / `0.3` | Per-axis speed ceilings (0–1). |
| `kp` / `kd` / `kv` | `0.6` / `0.05` / `0.1` | Proportional, derivative, and velocity feed-forward gains. |
| `lead_time_s` | `0.15` | Motion prediction: project the aim point forward by this much using measured velocity. |
| `aim_smoothing` | `0.5` | 0 = snappiest, 1 = smoothest (maps to the one-euro filter cutoff). |
| `safe_zone_enabled` | `true` | While the subject stays inside the framing box the PTZ holds still. |
| `safe_zone_x/y/w/h` | centred, `0.15`×`0.22` | Box centre offset + half-extents (fraction of half-frame). |
| `safe_zone_roundness` | `1.0` | 0 = rectangle … 1 = full oval. |
| `deadzone_x` / `deadzone_y` | `0.05` | Per-axis circular deadzone (used when the safe zone is off). |
| `auto_zoom` | `true` | Drive zoom to keep the subject at the framing target height. |
| `zoom_framing` | `upper_body` | Auto-zoom target height: `face`, `head_shoulders`, `upper_body`, `full_body`, or `wide`. Mirrors `framing`; `wide` is the one extra (looser) option. |
| `loss_zoom_out` / `reacquire_window_s` | `0.25` / `4.0` | On loss, gently zoom out to widen the view and re-find the subject. |
| `soft_limits` | none | Optional pan/tilt/zoom travel clamps. |

## Hardware

| Setting | Default | Notes |
| --- | --- | --- |
| `force_ep` | auto | Pin an execution provider (e.g. `CoreMLExecutionProvider`); falls back to auto if unavailable. |
| `precision` | `auto` | `auto`/`fp32`/`fp16`/`int8`. Accelerator EPs use FP16 unless forced to fp32; CPU is always fp32; `int8` runs a quantized detector (see [Performance](performance.md)). |
| `detector_model_tier` | `auto` | Detector model shared by all cameras (Services panel). `auto` picks Fast (YOLO11n); `fast`/`balanced`/`medium` map to YOLO11n/s/m — bigger detects better but costs more. Shown in the UI as Auto / Fast / Balanced / Accurate. |
| `max_workers` | `4` | Parallel camera workers (also informs the per-worker thread cap). |
| `intra_op_threads` | auto | Override ORT intra-op threads per worker (auto = cores ÷ cameras). |

See [Performance](performance.md) for how these interact with your accelerator.

## Environment overrides

| Variable | Effect |
| --- | --- |
| `AUTOPTZ_MODEL_PATH` | Use this detector ONNX verbatim (skip download/export). |
| `AUTOPTZ_MODEL_URL` / `AUTOPTZ_MODEL_URL_<STEM>` | Mirror to fetch a prebuilt ONNX (air-gapped/offline). |
| `AUTOPTZ_NO_MODEL_EXPORT` | Disable Ultralytics/Torch ONNX export fallback; useful for CI and locked-down installs. |
| `AUTOPTZ_POSE_MODEL_PATH` | Use this pose ONNX verbatim. |
| `AUTOPTZ_FORCE_EP` / `AUTOPTZ_PRECISION` / `AUTOPTZ_ORT_INTRA_THREADS` | Hardware prefs (set automatically from config by the supervisor). |
| `AUTOPTZ_UPDATE_REPO` | Override the GitHub repo the updater checks. |
