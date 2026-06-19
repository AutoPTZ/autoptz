# 00 — Current State & Goals

## What AutoPTZ v1 is

A Python + PySide6 desktop app that displays live camera feeds and physically moves PTZ
cameras to keep a chosen person centered. Sources: USB (OpenCV `VideoCapture`), NewTek NDI
(`ndi-python`), and RTSP (described as "under development"). PTZ control via NDI PTZ, USB VISCA
serial, and network VISCA (`visca-over-ip`). Tracking is a mix of `face_recognition` (dlib HOG +
ResNet embeddings), MediaPipe Pose, and a dlib correlation tracker.

### v1 architecture (as built)

```
startup.py
  └─ AutoPTZ_MainWindow (views/homepage/main_window.py)
       ├─ menu/source discovery: findNDISources(), findHardwareSources()  ← runs ONCE at startup
       ├─ FlowLayout of CameraWidget tiles (views/widgets/camera_widget.py)
       └─ global state in shared/constants.py:
            CURRENT_ACTIVE_CAM_WIDGET, CURRENT_ACTIVE_PTZ_DEVICE,
            IN_USE_USB_PTZ_DEVICES, RUNNING_HARDWARE_CAMERA_WIDGETS, NDI_SOURCE_LIST

CameraWidget (a QLabel) spawns, per camera, 3 processes via multiprocessing:
  • run_camera_stream      → appends full frames into manager.list() (shared, pickled)
  • run_facial_recognition → face_recognition every 240th frame, busy-loop (no sleep)
  • run_body_pose_estimation → MediaPipe Pose model_complexity=2, busy-loop (no sleep)
  A QTimer at ~30fps runs draw_on_frame() ON THE GUI THREAD: pulls queues, runs the dlib
  correlation tracker, fuses face+pose boxes, computes centroid, and drives PTZ inline.
```

## Confirmed shortcomings (verified in the code, matching the brief)

1. **Stagnant, non‑customizable, "ugly" UI.** `main_window.py` builds the manual‑control tab
   with hard‑coded pixel geometry (`setGeometry(QRect(...))` everywhere). Tiles live in a
   `FlowLayout` sized to `screen_width // 3`; you cannot drag‑reorder, resize, group, or save a
   layout. Styling is a single hard‑coded stylesheet string in `constants.py`. **No presets of
   any kind.**
2. **Detection/discovery only at startup.** `findNDISources()` and `findHardwareSources()` are
   called once in `__init__`. NDI discovery blocks for `8000ms + time.sleep(5)`. There is no
   hot‑plug, no reconnect, no periodic rescan. A camera that drops or appears later is invisible.
3. **No persisted state.** All runtime state is in‑memory module globals. The File ▸ Save / Open /
   Save As actions are unwired stubs. Nothing — cameras, PTZ assignments, tracked identities,
   layout, tuning — survives a restart.
4. **Multi‑camera data crossing.** Identity and PTZ selection flow through *module‑level mutable
   globals* (`CURRENT_ACTIVE_CAM_WIDGET`, `CURRENT_ACTIVE_PTZ_DEVICE`). With several cameras these
   get crossed → "data shifted to the wrong camera." Frames are shared through a `Manager().list()`
   which **pickles every numpy frame across the process boundary** — expensive and racy.
5. **Fragile tracking.** The dlib correlation tracker drifts, is slow, and has no motion model, so
   fast movers escape it. Face recognition runs only every 240 frames (~8 s at 30 fps), so a lost
   subject is re‑found slowly if at all.
6. **Re‑identification is rough and face‑only.** Re‑acquisition relies solely on
   `face_recognition`. If the subject turns away, is occluded, or someone crosses in front, there
   is no appearance (body) ReID to recover the *correct* track — hence the "hacks to overcome it."
7. **Does not scale.** Per camera: 3 processes + a `Manager` proxy + MediaPipe `complexity=2`
   (the heaviest model) running flat‑out in a busy‑loop, + face recognition busy‑looping with no
   sleep, + all tracking/drawing on the single GUI thread. 5 cameras ≈ 15+ processes thrashing CPU.
8. **No GPU acceleration.** dlib (CPU), MediaPipe (CPU), MobileNetSSD caffemodel (CPU). Nothing
   uses CUDA, TensorRT, CoreML/ANE, DirectML, or OpenVINO. MediaPipe Pose is also **single‑person**,
   so multi‑person body tracking was never really supported.
9. **Bang‑bang PTZ motion.** `ptz_control()` picks one of 8 directions from the centroid vs a fixed
   elliptical dead‑zone. There is no velocity model, no smoothing, no lead/prediction, and no
   auto‑zoom — so motion looks jerky and cannot "keep relative speed" with a moving subject.

## v2 goals (success criteria)

- **G1 — Real‑time & smooth.** Sub‑frame‑budget latency from detection to PTZ command; smoothed,
  velocity‑aware motion that keeps pace with a moving subject; no GUI‑thread stalls.
- **G2 — Works with nearly all sources.** USB, IP/RTSP (and ONVIF), and NDI — with continuous
  discovery, hot‑plug, and automatic reconnect (no startup‑only checks).
- **G3 — Robust tracking & re‑ID.** Survive fast motion, occlusion, and crossings. Re‑identify by
  **body appearance (ReID) + face**, not face alone. Always keep following the *right* person.
- **G4 — Smart framing.** Auto‑zoom in/out to maintain a target framing with hysteresis; widen on
  loss to re‑acquire.
- **G5 — Per‑camera config & remembered state.** Every camera has its own persisted settings,
  PTZ backend, presets, tracking tuning, and identity bindings. Everything survives restarts.
- **G6 — Scales with multiple cameras.** No global crossing; each camera owns its state. Scales
  from a few cameras on CPU to many on a GPU / M‑series Mac (see `07-system-requirements-scaling.md`).
- **G7 — Great, customizable UI.** Drag‑reorder/resize/group camera tiles, save layouts, quick
  enable/disable tracking, simple click‑to‑configure, PTZ presets, theming.
- **G8 — Cross‑platform.** First‑class Windows and macOS; signed installers; CPU‑capable, GPU‑ and
  Apple‑Silicon‑accelerated.

## Explicit non‑goals (v2 scope guardrails)

- Not a full NVR/recording suite (no long‑term video storage; preview + optional clip export only).
- Not a cloud service; it is a local desktop app (an optional remote control plane is a stretch goal).
- Linux is "best effort," not a release target (the engine should remain Linux‑compatible).
</content>
