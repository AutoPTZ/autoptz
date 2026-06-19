# 08 — Execution Roadmap

Ordered, dependency‑aware phases. Each phase is independently testable and leaves `main`‑mergeable
work. The matching copy‑paste prompts for an implementing model are in
`09-implementation-prompts.md` (one per phase). Keep v1 runnable until Phase 11 cutover.

**Conventions**
- Work on `dev/v2-architecture-rework` (already created) with a feature branch per phase.
- Every phase ends with: tests pass, a short `CHANGELOG` note, and a demo/verification step.
- Prefer adding the new `autoptz/` package alongside v1; do not delete v1 files until Phase 11.

---

### Phase 0 — Foundations & scaffolding
**Goal:** new package skeleton, dependency baseline, platform inference factory, CI.
- Create the `autoptz/` package layout from `01-target-architecture.md`.
- Pin dependencies; create `requirements/` (base, plus `gpu-nvidia`, `macos` extras).
- Implement `engine/runtime/inference.py`: ORT session factory that picks EP per platform
  (CoreML / TensorRT / CUDA / DirectML / OpenVINO / CPU) with fallback + logging.
- Implement `engine/runtime/shm.py` (shared‑memory frame ring buffer) and
  `engine/runtime/messages.py` (typed telemetry/command schemas).
- GitHub Actions matrix (macos‑arm64, windows‑x64): lint, type‑check, unit tests.
**Acceptance:** `python -m autoptz --selftest` prints the chosen EP and round‑trips a frame through
a shm buffer and a telemetry message; CI green on both OSes.

### Phase 1 — Config & persistence
**Goal:** SQLite + pydantic config; load/save/migrate; JSON export/import.
- Implement `config/models.py` and `config/store.py` per `06-persistence-and-config.md`.
- Migration runner + `schema_version`; platform config‑dir resolution.
**Acceptance:** create/edit/persist a `CameraConfig`; restart reloads it; export→import round‑trips;
migration test from an empty/older DB passes.

### Phase 2 — Ingest adapters + continuous discovery
**Goal:** get frames from all source types, with reconnect and live discovery (no startup‑only).
- `engine/pipeline/ingest.py`: USB, RTSP/ONVIF (FFmpeg HW decode), NDI (cyndilib) adapters with a
  common interface + stalled‑read detection + reconnect/backoff.
- `engine/discovery/`: NDI find (callbacks), ONVIF WS‑Discovery, USB hot‑plug. Emit add/remove
  events continuously.
- (Optional) go2rtc integration path for RTSP normalization.
**Acceptance:** plugging/unplugging a USB cam and starting/stopping an NDI source updates the source
list live; a dropped RTSP stream auto‑reconnects; frames land in a shm buffer at target fps.

### Phase 3 — Detection + tracking core
**Goal:** YOLO26 person detection + BoxMOT tracking producing stable track IDs.
- `engine/pipeline/detect.py` (YOLO26 via ORT) and `engine/pipeline/track.py` (BoT‑SORT default;
  DeepOCSORT/ByteTrack selectable). Camera‑motion compensation enabled.
- Bench: stable IDs through fast motion and short gaps on a recorded clip.
**Acceptance:** on a test video, IDs persist through a fast walk‑across and a 1 s occlusion without
ID swaps in the common case; runs at the tier's target cadence.

### Phase 4 — ReID + identity (the re‑identification fix)
**Goal:** body ReID recovery + face identity binding.
- `engine/pipeline/reid.py` (OSNet embeddings, gallery, hysteresis matching, run‑policy).
- `engine/pipeline/identify.py` + `engine/identity/` (InsightFace SCRFD+ArcFace, enrollment, gallery
  store, versioned reload).
- Implement the recovery rule from `03-vision-pipeline.md` (re‑bind after occlusion/crossing).
**Acceptance:** scripted "someone walks in front of the target" clip → target is re‑acquired (same
identity) rather than locking onto the interloper; enrolling a face binds the correct track.

### Phase 5 — PTZ backends + closed‑loop controller
**Goal:** smooth, velocity‑aware motion + presets + absolute position across all backends.
- `engine/ptz/` backends (ndi, visca_ip, visca_usb, onvif) behind `PTZBackend`; refactor v1's
  working serial VISCA into `visca_usb.py`.
- `engine/ptz/controller.py`: dead‑zone, one‑euro smoothing, PD + velocity feed‑forward, clamps,
  coast‑on‑loss, zoom controller; per‑camera gains.
- Presets + absolute recall persisted via `ptz_presets`.
**Acceptance:** on real/emulated PTZ, tracking is visibly smooth and leads a moving subject; saving
and recalling a preset works; loss triggers coast→search; `stop()` is reliable on every exit path.

### Phase 6 — Camera worker + supervisor (wire the pipeline)
**Goal:** one process per camera running ingest→detect→track→reid→identify→pose→framing→PTZ, plus
the supervisor lifecycle.
- `engine/camera_worker.py` (threads: ingest / inference / ptz; writes shm preview + telemetry).
- `engine/supervisor.py` (spawn/stop/health/restart; owns discovery + identity + config services).
- `engine/pipeline/framing.py` + `engine/pipeline/pose.py` (RTMPose, zoom on target).
**Acceptance:** start 3+ cameras headless; each tracks independently with **no cross‑camera state
leakage** (the v1 "wrong camera" bug cannot reproduce); kill a worker → supervisor restarts it.

### Phase 7 — UI: camera wall + live preview
**Goal:** QML wall rendering preview frames + telemetry overlays; select/track/toggle.
- `ui/app.py`, `ui/engine_client.py`, `ui/providers/` (shm→QImage), `ui/qml/CameraWall.qml`,
  `CameraTile.qml`.
- Drag‑reorder/resize tiles; click‑to‑select; click‑a‑box to target; enable/disable tracking.
**Acceptance:** add cameras from the UI, see live previews with overlays, reorder tiles, pick a
target by clicking, toggle tracking — all driven through the engine command/telemetry contract.

### Phase 8 — UI: config, presets, identities, layouts, themes
**Goal:** full per‑camera drawer, preset bar, identity manager, saved layouts, theming.
- `ConfigDrawer.qml` (Source/Tracking/PTZ/Presets/Tuning), preset bar, `IdentityManager.qml`,
  layout save/load, theme tokens. Wire to `UpdateCameraConfig`/`SetLayout`/`EnrollIdentity`.
**Acceptance:** every persisted setting is editable in‑app and survives restart; enroll→track‑by‑
identity works end to end; named layouts restore tile positions.

### Phase 9 — Performance, scaling & benchmark harness
**Goal:** hit the tier targets in `07`; add auto‑degrade and the bench tool.
- Per‑stage latency metrics + per‑worker auto‑degrade (drop pose → lower detect rate → switch
  tracker) to hold real‑time.
- `tools/bench/`: measure glass‑to‑PTZ latency, sustained fps, max cameras per quality level.
- Optional: shared/batched ORT session across workers on one GPU.
**Acceptance:** bench reports meet the tier table within tolerance on a reference machine; no
GUI‑thread stalls under N‑camera load; auto‑degrade visibly engages instead of dropping frames.

### Phase 10 — Packaging, signing, installers
**Goal:** shippable signed apps on both OSes.
- macOS: PyInstaller/py2app → notarized `.app`/`.dmg`; bundle NDI runtime + models + CoreML caches.
- Windows: PyInstaller → Inno/MSIX; default DirectML build + optional CUDA/TensorRT build; bundle
  NDI runtime + models.
- CI publishes artifacts; first‑run model/download + EP self‑check.
**Acceptance:** clean‑machine installs on Windows 11 and macOS launch, discover a camera, and track
without a dev environment.

### Phase 11 — Cutover & cleanup
**Goal:** make v2 the app; retire v1.
- Point `startup.py` at `autoptz` (or replace it); remove v1 `views/`, `logic/`, `libraries/`,
  `shared/` once parity is verified; update `README.md`, `requirements.txt`, screenshots.
- Migration note for any v1 users (re‑add sources; re‑enroll faces — formats differ).
**Acceptance:** repo builds and runs only the v2 path; docs updated; v1 removed or archived under a
tag.

---

## Milestone groupings (if you prefer fewer checkpoints)
- **M1 “Headless tracking works” = Phases 0–6** (engine end‑to‑end, no UI polish).
- **M2 “Usable app” = Phases 7–8** (full UI + persistence UX).
- **M3 “Ship it” = Phases 9–11** (perf, packaging, cutover).

## Cross‑cutting requirements (apply to every phase)
- **Stable IDs everywhere** — address cameras/PTZ by UUID, never by "current active widget."
- **No work on the GUI thread** — decode/inference/PTZ only in the engine.
- **Tests** — unit tests per module; integration tests on recorded clips; a smoke test in CI.
- **Telemetry/logging** — structured logs + an `events` table; surface EP/quality in the UI.
- **Keep it real‑time** — if a change risks the frame budget, it must be behind the auto‑degrade
  policy or a per‑camera setting.
</content>
