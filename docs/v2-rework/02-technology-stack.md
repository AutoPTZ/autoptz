# 02 — Technology Stack (researched)

Each choice below lists the **pick**, the **why**, and the **alternative considered**. Research
sources are listed at the bottom. Versions are targets as of mid‑2026 — pin exact versions during
Phase 0.

## Language & app core

- **Pick: Python 3.12.** Keep it. The whole value of this project lives in the Python CV
  ecosystem (OpenCV, Ultralytics, InsightFace, BoxMOT, ONNX Runtime, NDI/ONVIF bindings). A
  rewrite in Rust/C++ would throw away that leverage for marginal gains the per‑process model
  already recovers.
- **Alternative:** Rust/C++ engine with Python bindings. Rejected as the default; revisit only if
  a specific hot loop proves GIL‑bound after profiling (decode and ORT both release the GIL).

## Desktop UI

- **Pick: PySide6 (Qt 6.7+) with Qt Quick / QML** for the camera wall and panels. QML uses a
  GPU scene graph → smooth tiles, cheap drag‑reorder/resize/animation, and easy theming, while
  the heavy CV stays in Python worker processes. Video reaches QML via a shared‑memory →
  `QImage`/`QQuickImageProvider` bridge.
- **Why not stay on QWidgets:** v1's QWidget + hand‑placed geometry is exactly the "stagnant,
  non‑customizable" UI we're replacing. QML is built for fluid, reorderable, themeable layouts.
- **Alternative: web UI via Tauri/Electron + Python backend over WebSocket.** Rejected as the
  *primary* path because piping many live video tiles into a webview adds real cost and a second
  runtime. **But** the engine exposes a clean command/telemetry contract, so a remote web
  dashboard remains a viable later add‑on (stretch goal).

## Inference runtime (the cross‑platform "CPU‑or‑GPU" backbone)

- **Pick: ONNX Runtime with per‑platform Execution Providers (EPs).** One model format, many
  backends, chosen at runtime by a small factory (`engine/runtime/inference.py`):
  | Platform / hardware | EP | Notes |
  |---|---|---|
  | Apple Silicon (M‑series) | **CoreML EP** (ANE+GPU) | also ship native `.mlpackage` where it wins; VideoToolbox for decode |
  | Windows + NVIDIA | **TensorRT EP** (best) → **CUDA EP** | NVDEC for decode; cache TRT engines per machine |
  | Windows CPU / iGPU (Intel/AMD) | **DirectML EP** or **OpenVINO EP** | broad GPU coverage without CUDA |
  | CPU fallback (any) | **OpenVINO / ORT CPU** | guarantees "runs on CPU" |
- **Why:** delivers the brief's "ideally works on CPU, scales with GPU / M‑series" with a single
  codebase and graceful fallback. ONNX Runtime is also what InsightFace already runs on.
- **Alternative:** framework‑native (PyTorch CUDA + CoreML + OpenVINO separately). More moving
  parts, more platform branches. ORT EPs unify it.

## Person detection

- **Pick: YOLO26** (Ultralytics, released Jan 2026) — `yolo26n`/`yolo26s` for the person class.
  NMS‑free end‑to‑end, faster CPU inference, better small‑object accuracy, and first‑class export
  to CoreML / OpenVINO / TensorRT / ONNX / TFLite — perfect for the multi‑EP plan above.
- **Why over v1's MobileNetSSD caffemodel:** far higher accuracy and speed, modern export targets,
  unified with pose (YOLO26‑pose) if we want a single backbone.
- **Alternatives:** YOLO11 (still excellent, fall back if a YOLO26 export target is immature for a
  given EP); RT‑DETR / RF‑DETR (strong accuracy, heavier — keep as a "max accuracy" option).

## Multi‑object tracking

- **Pick: BoxMOT** trackers behind one wrapper, selectable per camera:
  - **BoT‑SORT** (default) — Kalman motion + **camera‑motion compensation** + optional appearance
    ReID. Handles a moving subject and a moving (panning) camera, which a PTZ rig always is.
  - **DeepOCSORT** — for crowded / heavy‑occlusion scenes (adaptive appearance fusion).
  - **ByteTrack** — lightweight, no‑ReID fallback for CPU‑bound / low‑resource tiers.
- **Why over v1's dlib correlation tracker:** dlib has no motion model and drifts; these trackers
  add a Kalman predictor (survives fast motion + short gaps) and appearance association (survives
  occlusion and crossings). Camera‑motion compensation matters specifically because PTZ pans.
- **Alternative:** Ultralytics built‑in BoT‑SORT/ByteTrack (simpler) — fine for a first cut;
  BoxMOT gives more tracker choices and pluggable ReID.

## Person re‑identification (ReID)

- **Pick: OSNet** (`osnet_x0_25` for speed, `osnet_ain_x1_0` for cross‑domain robustness) via
  BoxMOT / torchreid, exported to ONNX. Produces a body appearance embedding so a person who was
  occluded, turned away, or crossed by someone else gets the **same track ID** back.
- **Why:** this is the missing half of v1's "re‑ID was rough and face‑only." Body ReID works when
  the face isn't visible; face confirms identity when it is. Together they keep the camera on the
  *correct* person.
- **Run policy:** embed only on new/ambiguous tracks and periodically on the target — not every
  box every frame (keeps it cheap). See `03-vision-pipeline.md`.

## Face recognition (identity binding)

- **Pick: InsightFace** — **SCRFD** detector + **ArcFace** recognition (`buffalo_l` on capable
  hardware, `buffalo_s`/`buffalo_sc` on edge/CPU), on ONNX Runtime.
- **Role change vs v1:** face recognition **binds a track to a named identity**; it is *not* the
  per‑frame tracker. Run it at ~1–3 Hz on the target to (re)confirm "this track is Alice," then let
  the tracker + ReID carry the frames in between. This is dramatically cheaper and more stable than
  v1's `face_recognition` every 240 frames trying to do everything.
- **Why over `face_recognition`/dlib HOG:** SCRFD+ArcFace is markedly more accurate, GPU/ANE
  accelerated via ORT, and handles pose/lighting far better.
- **Gallery/matching:** cosine similarity over stored ArcFace embeddings; small index is fine
  (`numpy`/`faiss-cpu`). Persist embeddings in SQLite (see `06`).

## Pose (for smart zoom only)

- **Pick: RTMPose** (ONNX/TensorRT) — multi‑person, ~90+ FPS on CPU for the `-m` model, 400+ FPS on
  a modest GPU. Used to estimate head/shoulder/hip keypoints of the *target* to drive framing
  (headroom, head‑to‑waist ratio) for auto‑zoom.
- **Why over MediaPipe Pose:** v1's MediaPipe Pose is **single‑person** and was pinned to
  `model_complexity=2` (heaviest) running flat‑out per camera. RTMPose is multi‑person and far more
  efficient, and exports to all our EPs.
- **Cheaper alternative:** YOLO26‑pose (reuse the detector backbone) or just use bbox height for
  zoom on low tiers (pose optional).

## Video ingest

- **Pick:** per‑source adapters with **hardware decode**:
  - **RTSP/IP:** FFmpeg (PyAV or OpenCV‑FFMPEG) with HW decode — NVDEC (NVIDIA),
    VideoToolbox (macOS), D3D11VA/QSV (Windows). Prefer the camera's **sub‑stream** for AI and the
    main stream for preview/record where available.
  - **USB:** OpenCV/FFmpeg with platform backends (AVFoundation/macOS, MSMF/DirectShow/Windows).
  - **NDI:** **cyndilib** (see below).
- **Optional front‑end gateway: go2rtc.** A tiny local server that normalizes RTSP/RTMP/USB/WebRTC,
  auto‑reconnects, and exposes stable local endpoints. Using it offloads reconnection/protocol
  quirks from the engine and directly fixes v1's "only checks at startup."
- **Alternative:** GStreamer pipelines (powerful but heavier to install/ship cross‑platform).

## NDI

- **Pick: cyndilib** (Cython NDI bindings, maintained, prebuilt wheels) — supports frame‑sync,
  PTZ, tally, metadata. Frame‑sync yields clean frames while PTZ/metadata share the receiver.
- **Why over v1's `ndi-python`:** better maintained, faster (Cython), frame‑sync, cleaner PTZ.
- **Note:** requires the NewTek/NDI SDK runtime installed/bundled per platform.

## PTZ control

- **Pick: a `PTZBackend` interface** with four implementations behind one closed‑loop controller:
  - **NDI PTZ** (cyndilib): speed + absolute + presets.
  - **VISCA over IP** (`visca-over-ip` / extend): inquiry for absolute pan/tilt/zoom, presets.
  - **ONVIF PTZ** (`python-onvif-zeep` / `pyptz`): `ContinuousMove`, `AbsoluteMove`, presets —
    widest IP‑camera compatibility, the biggest coverage win for "nearly all IP cameras."
  - **USB VISCA serial** (keep v1's working serial path, refactored).
- **Why:** v1 supports NDI/USB/network‑VISCA but with bang‑bang motion and no presets/absolute
  position. ONVIF adds huge device coverage; absolute+presets enable remembered states and instant
  recall. The shared controller (see `04-ptz-control.md`) adds velocity, smoothing, lead, and zoom.

## Persistence & config

- **Pick: SQLite** (stdlib `sqlite3` or SQLAlchemy) for structured state — cameras, per‑camera
  settings, PTZ presets, layouts, identities/embeddings — plus **JSON import/export** for
  portability/backup. Config modeled with **pydantic** for validation and safe migrations.
- **Why:** v1 has *no* persistence. SQLite gives transactional, queryable, single‑file state that
  survives restarts; JSON gives human‑readable export/sharing.

## Packaging & CI

- **macOS:** PyInstaller (or py2app) → signed + notarized `.app` in a `.dmg`; arm64 (and
  universal2 if needext). Bundle NDI runtime + CoreML models.
- **Windows:** PyInstaller → Inno Setup / MSIX installer; bundle the correct ORT EP (DirectML by
  default; optional CUDA/TensorRT build for NVIDIA users). Bundle NDI runtime.
- **CI:** GitHub Actions matrix (`macos-arm64`, `windows-x64`) building artifacts + smoke tests.

## Summary table

| Concern | v1 | v2 pick |
|---|---|---|
| UI | QWidgets, fixed geometry, no layouts/presets | PySide6 **Qt Quick/QML**, reorderable wall, presets, themes |
| Detection | MobileNetSSD caffemodel (CPU) | **YOLO26** via ONNX Runtime EPs |
| Tracking | dlib correlation tracker | **BoT‑SORT / DeepOCSORT / ByteTrack** (BoxMOT) |
| ReID | none (face only) | **OSNet** appearance embeddings |
| Face | `face_recognition` (dlib HOG) | **InsightFace SCRFD + ArcFace** |
| Pose/zoom | MediaPipe Pose (single‑person, complexity 2) | **RTMPose** (multi‑person) / YOLO26‑pose, zoom optional |
| Inference accel | none (CPU) | **ONNX Runtime EPs**: CoreML / TensorRT / CUDA / DirectML / OpenVINO |
| RTSP ingest | "under development" | FFmpeg HW decode (+ optional **go2rtc** gateway) |
| NDI | `ndi-python` | **cyndilib** (frame‑sync, PTZ) |
| PTZ | NDI/USB/net‑VISCA, bang‑bang | NDI/VISCA‑IP/USB + **ONVIF**, closed‑loop + presets + absolute |
| State | in‑memory globals (none persisted) | **SQLite + JSON**, pydantic config |
| Concurrency | 3 procs/cam + Manager().list() frame pickling | **1 worker proc/cam**, shared‑memory frames |

---

### Sources

- Ultralytics tracking (BoT‑SORT/ByteTrack): https://docs.ultralytics.com/modes/track
- BoxMOT (DeepOCSORT/BoT‑SORT/OCSORT/ByteTrack + ReID): https://github.com/mikel-brostrom/boxmot
- YOLO26 overview: https://blog.roboflow.com/yolo26/ • https://docs.ultralytics.com/models • https://docs.ultralytics.com/integrations/coreml
- OSNet / Torchreid: https://github.com/KaiyangZhou/deep-person-reid
- InsightFace (SCRFD + ArcFace, ONNX): https://github.com/deepinsight/insightface
- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/ • DirectML: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html • TensorRT/CUDA: https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/
- RTMPose: https://arxiv.org/pdf/2303.07399 • https://openmmlab.medium.com/rtmpose-the-all-in-one-real-time-pose-estimation-solution-for-application-and-research-6404f17cd52f
- go2rtc: https://github.com/AlexxIT/go2rtc • https://go2rtc.org/
- cyndilib (NDI): https://cyndilib.readthedocs.io/en/latest/overview.html • NDI PTZ/frame‑sync: https://docs.ndi.video/docs/sdk/ndi-recv
- PTZ libs: https://pypi.org/project/visca-over-ip/ • https://github.com/misterhay/VISCA-IP-Controller • https://pypi.org/project/pyptz/ (ONVIF/VAPIX/SUNAPI)
- Apple Silicon CoreML/ANE: https://blog.roboflow.com/putting-the-new-m4-macs-to-the-test/
- NVDEC / GPU stream sizing: https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_Overview.html • https://forums.developer.nvidia.com/t/how-many-streams-can-be-decoded-when-the-gpu-is-running-ai-models/307975
- Desktop framework comparison: https://peerlist.io/jagss/articles/tauri-vs-electron-a-deep-technical-comparison
</content>
