# 10 — UX Overhaul & Engine Completion (v2.1 plan)

> **Purpose.** A cohesive, self-contained implementation plan to be executed by an
> implementing model **one phase at a time**. It builds on the P0+P1 work already in the
> working tree (engine wiring + first UI cleanups) and turns AutoPTZ into a usable,
> polished app. Read `00-current-state-and-goals.md`, `03-vision-pipeline.md`,
> `05-ui-ux.md`, and `06-persistence-and-config.md` for background.

## How to use this document
- Each phase below is independently shippable and ends with: **tests green**, a short
  `CHANGELOG.md` note, and a manual verification step.
- Each phase has a copy-paste **PROMPT** block written for a fresh model with no memory of
  this conversation — it names files and acceptance criteria so it can act cold.
- **Branch:** work on `dev/v2-architecture-rework`. Do **not** create per-phase branches;
  do **not** commit unless the user asks.
- **Run tests** with the repo venv: `.venv/bin/python -m pytest -q` (the venv now has the
  full base+dev stack). Keep all existing tests green (388 at time of writing).
- **Invariants (unchanged):** cameras addressed by stable UUID; no global "current active"
  state; no heavy work on the GUI thread; graceful degrade when an optional dep/model is
  missing.

---

## Current state (what exists now)

**Built & working**
- Engine *primitives*: `engine/pipeline/ingest.py`, `detect.py`, `track.py`;
  `engine/ptz/*` backends + `controller.py`; `engine/runtime/{shm,messages,inference}.py`;
  `config/{models,store}.py` (SQLite, `get_setting`/`set_setting`, layouts, identities).
- Engine *orchestration* (P0): `engine/camera_worker.py` (threaded `CameraWorker`) +
  `engine/supervisor.py` (`Supervisor`), wired in `ui/app.py`; `EngineClient` lifecycle API
  `engineRunning`/`engineEp`/`startEngine()`/`stopEngine()`/`engineStateChanged`, thread-safe
  `push_telemetry`. Live preview + fps work even with no ML stack.
- UI (P0+P1): native macOS menu bar; in-window engine status pill + Start/Stop; sidebar =
  selected-camera config only; `ConfigDrawer` tabs contextual to PTZ capability; on-screen
  D-pad on tiles (pauses tracking); inverted column shortcut. `SourceBrowser.qml` deleted.

**Stubs (still `TODO`, 7 lines each):** `engine/pipeline/reid.py`, `engine/pipeline/identify.py`,
`engine/identity/service.py` (and `identity/store.py`), `engine/pipeline/{pose,framing}.py`.

### Confirmed bugs & root causes (fix these in the phases noted)
1. **Blank navy preview** *(Phase 1)* — `ShmFrameProvider.attach()` does `ShmReader(name…)`
   which raises because the worker creates its `ShmWriter` *inside its thread* **after**
   `app.py` fires `providerAttachRequested`. The failed attach stores no reader and there is
   **no retry**, so `requestImage` returns the navy placeholder forever.
   (`ui/providers/__init__.py:49`, `ui/app.py:64`, `engine/camera_worker.py:468`.)
2. **Camera name listed twice** *(Phase 4)* — a discovered USB source stays under
   **Cameras ▸ USB** even after it is added, and also appears under **Cameras ▸ Active
   Cameras** (`ui/qml/CameraWall.qml:82,122`). In-use sources must drop out of the add list.
3. **macOS app menu says "Python"** *(Phase 3 interim, Phase 10 real fix)* — unbundled
   `python -m autoptz` makes macOS use the process name. Proper fix = `.app` bundle with
   `CFBundleName=AutoPTZ`.
4. **Continuity Camera / wrong USB device & name** *(Phase 4)* — `scanUSBCameras()` maps
   `system_profiler` order to `usb://0,1,2…` (`ui/engine_client.py:808`), but OpenCV's capture
   index order differs and an iPhone Continuity Camera shifts indices. Replace with
   AVFoundation enumeration keyed on stable `uniqueID`.
5. **Manual `usb://`/`ndi://` entry is pointless** *(Phase 4)* — USB & NDI are discoverable;
   only RTSP/ONVIF should accept a typed address.
6. **Manual PTZ has no effect on hardware** *(Phase 6)* — there is no `PTZBackend` factory
   from `PTZConfig`, so `ptzNudge` reaches the worker but moves nothing.

### Locked product decisions (from the user)
- **Engine auto-starts on launch**, with manual Start / Stop / **Restart**.
- **Identities live in a top "People" view** (full-width gallery opened from the status
  strip), not a sidebar setting.
- **Tracking gate = soft.** Show a warning when no identity is registered, but always allow
  manual click-a-box to follow, **and** allow picking a *registered* identity by name so the
  engine starts tracking them **when found**.
- **Config = Auto-first.** Source caps / detector / tracker / PTZ backend default to *Auto*;
  the drawer shows essentials only with an **Advanced** disclosure of collapsible groups +
  `?` help tooltips.
- **Faces = continuous auto-harvest** into an "Unlabeled people" tray to name / merge / enable.

---

## Recommended libraries & models (researched June 2026)

| Need | Library / model | Notes |
|------|-----------------|-------|
| Person detection | **Ultralytics YOLO11** (`yolo11n.pt`/`yolo11s.pt`) → ONNX | Production pick; YOLO12/26 attention layers are unstable & slow on CPU. Export `format=onnx, nms=False, dynamic=False, opset=12` (our `detect.py` does NMS; avoids the batched-NMS export bug). |
| Face detect+embed | **`insightface`** `FaceAnalysis("buffalo_l")` | SCRFD detector + ArcFace **512-d** embeddings; **auto-downloads** to `~/.insightface/models` on first `prepare()`. OSS pack licensing: contact `recognition-oss-pack@insightface.ai`. |
| Body re-ID + trackers | **`boxmot`** (already in `base.txt`) | OSNet ReID + BoT-SORT/DeepOCSORT/ByteTrack; weights auto-download on first use. |
| ONNX inference | **`onnxruntime`** (CoreML/TensorRT/CUDA/DirectML/CPU) | Already wired in `engine/runtime/inference.py`. |
| macOS camera enum | **`pyobjc-framework-AVFoundation`** | `AVCaptureDeviceDiscoverySession` → per-device `localizedName` + stable `uniqueID`; identifies Continuity Camera. |
| Windows camera enum | **`pygrabber`** (DirectShow) | Friendly device names on Windows (parity with macOS). |
| App packaging | **PyInstaller** or **py2app** | macOS `.app` + `Info.plist` `CFBundleName=AutoPTZ` (fixes the "Python" menu); notarize/sign in Phase 10. |
| Logging viewer | stdlib `logging` + Qt | A `logging.Handler` → Qt signal → QML model; no new dep. |

Add to `requirements/`: `ultralytics` (export-time/optional), `insightface`, `onnx`,
`pyobjc-framework-AVFoundation` (macOS), `pygrabber` (Windows). `boxmot`, `onnxruntime`, `av`
already present.

---

## Phases

### Phase 0 — Dependencies & model bootstrap
**Goal.** Detection works out of the box; one place owns model discovery/download.
- Add deps above to `requirements/{base,macos,gpu-*}.txt` as appropriate (keep ML optional).
- New `engine/runtime/models.py` — a `ModelManager`:
  - Cache dir `~/Library/Application Support/AutoPTZ/models` (platform-aware, reuse
    `config/store.py` dir logic).
  - `ensure_detector() -> Path|None`: returns a YOLO11 ONNX path; if missing, download the
    `.pt` via `ultralytics` and export to ONNX (`nms=False`), or fetch a prebuilt ONNX. Honor
    `AUTOPTZ_MODEL_PATH` override (already read in `camera_worker._resolve_model_path`).
  - Non-fatal + logged if `ultralytics`/network is unavailable (live-preview-only still works).
- `tools/fetch_models.py` — CLI to pre-download/export all models offline.
- Wire `camera_worker._resolve_model_path()` to call `ModelManager.ensure_detector()`.
**Acceptance.** With deps installed and network access, first engine start downloads/export
YOLO11 ONNX once, caches it, and tiles show person boxes. Offline → graceful live-preview.
```
PROMPT — Phase 0
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches, no commits).
Add a model bootstrap so person detection works out of the box. Create autoptz/engine/runtime/models.py with a
ModelManager that resolves/downloads a YOLO11 person-detection ONNX into the platform app-data models dir
(reuse the dir logic in autoptz/config/store.py), exporting from ultralytics (YOLO("yolo11n.pt").export(
format="onnx", nms=False, dynamic=False, opset=12)) when only the .pt is available, and honoring the
AUTOPTZ_MODEL_PATH env override. Everything must degrade gracefully (log + return None) if ultralytics or the
network is missing — never raise into the engine. Wire autoptz/engine/camera_worker.py:_resolve_model_path to
use it. Add a tools/fetch_models.py CLI to pre-fetch offline. Add ultralytics + onnx to requirements/base.txt
(optional/ML). Add unit tests with the network/export mocked. Keep `pytest -q` green. Read detect.py to match
the expected ONNX output format (NMS-free [1,N,6] preferred).
```

### Phase 1 — Engine auto-start, restart & the preview fix
**Goal.** Launch → engine running → **you see live video**; manual Start/Stop/Restart.
- **Fix the preview race** (`ui/providers/__init__.py`): make attach lazy + self-healing.
  Add `register(camera_id, shm_name, h, w)` that only records intent; in `requestImage`, if
  there's no reader yet, *try* to open a `ShmReader` each call (cheap) and cache on success —
  so it doesn't matter whether the writer existed at attach time. Also have
  `CameraWorker.start()` create the `ShmWriter` **before** the thread starts (so the segment
  exists ASAP). Keep `detach`/`detach_all`.
- **Auto-start**: `EngineClient` gains `restartEngine()` and an `autostart` path; `app.py`
  calls `startEngine()` after load (deferred via `QTimer.singleShot(0,…)` so the window shows
  first). Persist last engine on/off in `app_settings` (`store.set_setting("engine_running", …)`)
  and restore it; default **on**.
- Menu + status strip: add **Restart Engine**; the status pill already shows running/EP.
**Acceptance.** Cold launch with a USB camera added → within ~1–2 s the tile shows live
video and a non-zero fps, no manual step. Stop/Start/Restart all work and update the pill.
```
PROMPT — Phase 1
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
Two things. (1) Fix the blank-navy preview: in autoptz/ui/providers/__init__.py make ShmFrameProvider
self-healing — add register(camera_id, shm_name, h, w) that records intent without opening, and in
requestImage lazily try ShmReader(shm_name,h,w) on each call until it succeeds (cache it), returning the
placeholder meanwhile; keep detach/detach_all. In autoptz/engine/camera_worker.py create the ShmWriter inside
start() (before the thread) so the segment exists immediately, and have the supervisor emit the attach/register
request after start(). (2) Engine auto-start: add restartEngine() to EngineClient and make autoptz/ui/app.py
start the engine after the window loads (QTimer.singleShot(0, client.startEngine)); persist/restore the last
on/off state via ConfigStore.set_setting/get_setting("engine_running"), default ON; add a "Restart Engine"
item to the Engine menu in autoptz/ui/qml/CameraWall.qml. Read ui/app.py, ui/providers/__init__.py,
engine/camera_worker.py, engine/supervisor.py first. Keep `pytest -q` green and add a headless test that a
worker+register makes requestImage return a real frame after the writer appears.
```

### Phase 2 — Logging viewer
**Goal.** "Extensive logging to view" inside the app.
- `ui/log_bridge.py`: a `logging.Handler` subclass that marshals records (level, logger,
  message, ts) to the GUI thread via a Qt signal; a `LogListModel` (ring-buffered, e.g. last
  2000) exposed to QML; install the handler on the root logger in `app.py`.
- `ui/qml/LogConsole.qml`: opened from a status-strip **Logs** button (and Window menu),
  monospaced list, color by level, level filter, text search, autoscroll/pause, copy, clear.
- Set sensible default levels; a `--log-level` already exists in `__main__.py`.
**Acceptance.** Start/stop engine, add/remove camera, trigger an error → entries appear live,
filterable by level, copyable.
```
PROMPT — Phase 2
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
Add an in-app log viewer. Create autoptz/ui/log_bridge.py: a logging.Handler that emits each record to the GUI
thread through a Qt signal, plus a QAbstractListModel (ring buffer ~2000 rows; roles level/logger/message/ts)
exposed to QML. Install it on the root logger in autoptz/ui/app.py and expose the model as a context property.
Create autoptz/ui/qml/LogConsole.qml — a themed console (monospace, color per level, level filter, search,
autoscroll toggle, copy, clear) and open it from a new "Logs" button in the top status strip of CameraWall.qml
(and a Window/View menu item). Use Theme tokens. Keep `pytest -q` green; add a headless test that emitting a log
record appends a row to the model.
```

### Phase 3 — App identity, About & state persistence
**Goal.** It's "AutoPTZ" not "Python"; About lives in the app menu; the app remembers itself.
- **App name (interim):** set `app.setApplicationDisplayName("AutoPTZ")`; attempt a runtime
  `CFBundleName` set via PyObjC when available (best-effort; real fix is Phase 10 bundling —
  document the limitation).
- **About:** a `Platform.MenuItem { role: AboutRole }` (lands under the app menu on macOS)
  opening an About dialog (version, author, EP, links) — reuse `SettingsPanel`'s About tab.
- **State persistence** via `app_settings`: window geometry, `overrideCols`, theme,
  `selectedCameraId`, engine on/off, last People/Logs visibility — save on change/close,
  restore on launch. Ensure **named layouts** (already in `SettingsPanel`) round-trip tile
  order; add "Save Layout" affordance to the status strip.
**Acceptance.** Quit & relaunch restores window size, columns, theme, selection, and engine
state; the macOS app menu reads "AutoPTZ" (display name) with a working **About AutoPTZ**.
```
PROMPT — Phase 3
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
(1) In autoptz/ui/app.py set QGuiApplication.setApplicationDisplayName("AutoPTZ") and, if PyObjC is importable,
best-effort set the bundle name at runtime (document that the definitive fix is the .app bundle in Phase 10).
(2) Add an About item using Qt.labs.platform MenuItem role AboutRole in CameraWall.qml that opens an About
dialog (reuse SettingsPanel's About tab content). (3) Persist & restore app state through ConfigStore
set_setting/get_setting: window x/y/w/h, overrideCols, theme mode, selectedCameraId, engine_running. Save on
change and on close; restore after load. Verify named layouts save/restore tile order (SettingsPanel Layouts
tab + EngineClient.saveCurrentLayout/loadLayout). Keep `pytest -q` green; add tests for the settings round-trip.
```

### Phase 4 — Source discovery & camera identity
**Goal.** Reliable, name-correct discovery; no pointless manual URIs; per-camera About.
- **macOS USB enum** via `pyobjc-framework-AVFoundation` (`AVCaptureDeviceDiscoverySession`,
  builtin+external+continuity types): return `{name, unique_id, index}`; flag Continuity
  Camera; **open by enumeration index that matches OpenCV's AVFoundation order** (verify) and
  store `unique_id` in `SourceConfig` for stable re-binding. Windows: `pygrabber` names.
  Replace `EngineClient.scanUSBCameras()`'s `system_profiler` path.
- **Remove manual USB/NDI address entry** — USB & NDI are pick-from-discovery only; keep the
  RTSP/ONVIF dialog (Phase-1 P1 already added it). USB/NDI submenus list discovered devices.
- **De-dupe**: once a source is added, hide it from the "add" submenu (compare by
  `unique_id`/uri against active cameras) — fixes the name-twice bug.
- **Per-camera About/Info**: a tile context action + a section opening camera metadata —
  display name, source type & sanitized address, PTZ backend/EP, target vs **actual** fps,
  resolution (from shm/probe), health, uptime, identity bound, shm name. Add any missing
  fields to `TelemetryMsg` (e.g. `width`,`height`) sparingly.
**Acceptance.** USB list shows correct macOS names (incl. labeled Continuity Camera), opens
the right device, never lists an already-added camera; per-camera About shows live stats.
```
PROMPT — Phase 4
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
Fix camera discovery & identity. (1) Rewrite EngineClient.scanUSBCameras() (autoptz/ui/engine_client.py) to use
pyobjc-framework-AVFoundation on macOS: AVCaptureDeviceDiscoverySession over builtin+external+continuity video
device types, returning {name, unique_id, index}; label Continuity Camera; fall back to the current behavior if
PyObjC is absent. Persist unique_id in SourceConfig (autoptz/config/models.py) for stable rebinding, and prefer
opening USB by the AVFoundation-ordered index (verify it matches cv2.CAP_AVFOUNDATION ordering). On Windows use
pygrabber for names. (2) In CameraWall.qml, drop any manual usb://ndi:// text entry (USB/NDI are discovery-only;
keep the RTSP/ONVIF dialog) and hide a discovered source from the USB/NDI submenu once a camera with the same
unique_id/uri is active (fixes the camera-name-listed-twice bug at lines ~82 and ~122). (3) Add a per-camera
"Camera Info" view (name, source type+sanitized address, PTZ backend, EP, target vs actual fps, resolution,
health, uptime, bound identity, shm name); add width/height to TelemetryMsg if needed. Keep `pytest -q` green;
mock PyObjC in tests.
```

### Phase 5 — ConfigDrawer: Auto-first, collapsible groups, help
**Goal.** Simplify; "Auto" everywhere; collapsible groups; `?` help.
- Reusable `CollapsibleSection.qml` (header + expand/collapse, remembers state) and
  `HelpHint.qml` (a `?` that shows a tooltip/popover).
- Default everything to **Auto**: source caps (fps/resolution probed), detector (model auto),
  tracker (`auto`→BoT-SORT), PTZ backend (`auto`-detect). Drawer shows only essentials
  (name, source, target identity, a few toggles); an **Advanced** disclosure reveals
  collapsible groups: *Detection*, *Tracking & ReID*, *PTZ tuning*, *Presets*.
- Add `?` help to: sub-stream, source fps cap, ReID, detect-interval, tracker choice,
  backends, dead-zone, Kp/Kd/Kv, zoom framing, quality floor. Keep contextual tab logic
  (USB-without-PTZ hides PTZ groups; already added by P1's Agent 3) and the `_set`/debounce
  mechanism intact.
**Acceptance.** A new USB camera shows a short, friendly drawer; **Advanced** reveals tidy
collapsible groups; every jargon control has a working `?` explanation; Auto requires no edits.
```
PROMPT — Phase 5
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
Redesign autoptz/ui/qml/ConfigDrawer.qml to be Auto-first. Create reusable autoptz/ui/qml/CollapsibleSection.qml
(themed header + expand/collapse, persists open/closed) and HelpHint.qml (a "?" icon → tooltip/popover). Default
detector/tracker/PTZ backend/source caps to "auto" (valid values per autoptz/config/models.py; tracker default
maps to botsort, PTZ "auto"). Show only essentials by default (name, source, target identity, key toggles) with
an "Advanced" disclosure that contains CollapsibleSection groups: Detection, Tracking & ReID, PTZ tuning,
Presets. Add HelpHint help text to sub-stream, fps cap, ReID, detect interval, tracker, PTZ backend, dead-zone,
Kp/Kd/Kv, zoom framing, quality floor. Preserve the required props (cameraId, theme) and the _set/_reload/
debounce mechanism, and keep PTZ groups hidden when the camera has no PTZ. Verify with pyside6-qmllint.
```

### Phase 6 — PTZ backend factory + control loop
**Goal.** Manual joystick **and** auto-tracking physically move the camera.
- `engine/ptz/factory.py`: `build_backend(PTZConfig) -> PTZBackend|None` mapping
  `visca_usb`/`visca_ip`/`ndi`/`onvif`, and `auto` → probe (NDI PTZ if NDI source, else
  ONVIF/VISCA by address); graceful `None` if unconfigured/unavailable.
- `CameraWorker`: construct a `PTZController` (existing) around the backend; auto mode drives
  it from the current target track's error each tick; `ptz_nudge` enters a **manual-override**
  window that suspends auto until idle; report real `PTZState` in telemetry.
- Reliable `stop()` on all paths (controller already supports this).
**Acceptance.** On a real/emulated PTZ camera: manual D-pad moves it and auto-tracking leads a
moving target smoothly; releasing the D-pad resumes auto; `stop()` always halts motion.
```
PROMPT — Phase 6
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
Make PTZ actually move. Create autoptz/engine/ptz/factory.py: build_backend(config.ptz) returning the right
PTZBackend (visca_usb/visca_ip/ndi/onvif) or None, with "auto" probing (NDI source→NDI PTZ, else ONVIF/VISCA by
address); never raise. In autoptz/engine/camera_worker.py, when a backend is available wrap it in the existing
PTZController (engine/ptz/controller.py): in auto mode feed the current target track's normalized center error
each tick; ptz_nudge() should start a short manual-override window that suspends auto control and then resumes;
publish the real PTZState (pan/tilt/zoom/moving/state) in TelemetryMsg. Ensure stop()/close() always halts the
backend. Read base.py, controller.py, the four backends, and camera_worker.py first. Keep `pytest -q` green and
add tests with a fake backend asserting nudge→move_velocity, auto error→command direction, and override→resume.
```

### Phase 7 — Tile UX: PTZ control + tracking control rethink
**Goal.** Replace the "small/childish" joystick and the disliked idle/track toggle.
- **PTZ control**: a larger, professional control — either a polished joystick puck with a
  visible deflection ring + speed, or a clean D-pad+zoom cluster — revealed via a "Manual"
  control mode (button or hover), sized for real use, themed. Keep press-and-hold continuous
  motion and the tracking-pause behavior; route arrow keys through it.
- **Tracking control (replace the bottom Switch)**: an identity-aware affordance:
  - If a registered identity is selectable → a "Follow ▸ [Name]" picker; engine starts
    tracking when that identity appears ("track when found").
  - Click a person box → follow that track directly (always allowed).
  - **Soft gate**: if no identity is registered, show a non-blocking warning chip
    ("No registered ID — following by box only"), but don't block.
  - Show the current target's name + thumbnail while tracking; a clear Stop.
**Acceptance.** The manual control feels substantial; you can start tracking by clicking a
box or by choosing a registered name; with no identities you still can track-by-box and see a
soft warning; the old idle/enable Switch is gone.
```
PROMPT — Phase 7
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
Redesign autoptz/ui/qml/CameraTile.qml. (1) Replace the small D-pad with a larger, professional manual PTZ
control (a joystick puck with a deflection ring + speed indication, or a clean D-pad+zoom cluster), revealed via
a "Manual" mode toggle, themed via Theme tokens, sized for real use. Keep hold-to-move (~18 Hz ptzNudge) and the
"pause tracking while manual, resume after ~1.5 s idle" behavior; route arrow keys through it. (2) Remove the
bottom-bar idle/track Switch and replace it with an identity-aware tracking control: a "Follow ▸ [Name]" picker
populated from engineClient.identityModel (enabled identities) that calls a target-by-identity slot; clicking a
person box still follows that track; if no identity is registered show a non-blocking warning chip but still
allow box-tracking (soft gate). While tracking, show the target's name + thumbnail and a Stop control. Use only
existing EngineClient methods plus any added in Phase 8 (setTargetIdentity); verify with pyside6-qmllint.
```

### Phase 8 — Face identity engine (the Phase-4 vision work)
**Goal.** Real identities: auto-harvest, enroll, recognize, re-ID, target-by-identity.
- `engine/pipeline/identify.py`: InsightFace `FaceAnalysis("buffalo_l")` (auto-download),
  detect faces in target/whole frame, 512-d embeddings; match against the gallery.
- `engine/identity/service.py` + `store.py`: gallery CRUD backed by `identities` /
  `identity_embeddings` (already in `config/store.py`), enroll (add embeddings + thumbnail),
  rename, delete, **enable/disable**, **merge** (fold B's embeddings/thumbnail into A),
  versioned reload.
- `engine/pipeline/reid.py`: OSNet (via boxmot) body embeddings + hysteresis matching to
  recover the *right* track after occlusion/crossing (per `03-vision-pipeline.md`).
- **Continuous auto-harvest**: the worker periodically grabs a good face crop for unmatched
  faces, creates an **unlabeled** identity (auto-name "Person N") with a thumbnail, and pushes
  it to the UI; matched faces annotate the track's `identity`/`confidence` in telemetry.
- **Target-by-identity**: new `EngineClient.setTargetIdentity(camera_id, identity_id)` +
  `SetTargetIdentityCmd`; the worker sets the target when that identity is detected.
- **Thumbnails to UI**: serve via an `image://identity/<id>` provider or a small base64 field
  on the identity model.
**Acceptance.** Running the engine populates an Unlabeled tray with face thumbnails; labeling
+ enabling one lets "Follow ▸ Name" lock on when they appear; someone crossing in front no
longer steals the lock (ReID recovers); merge combines two into one.
```
PROMPT — Phase 8
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
Implement the face/ReID identity engine (currently 7-line stubs). engine/pipeline/identify.py: wrap insightface
FaceAnalysis("buffalo_l") (auto-downloads to ~/.insightface/models; CPU ctx fallback), detect faces + 512-d
ArcFace embeddings, match vs gallery with a threshold. engine/identity/service.py + store.py: gallery over the
existing identities/identity_embeddings tables in config/store.py — enroll(add embeddings+thumbnail), rename,
delete, enable/disable, merge(foldInto), versioned reload. engine/pipeline/reid.py: OSNet body embeddings via
boxmot with hysteresis matching to recover the correct track after occlusion (see docs/v2-rework/03). In
camera_worker.py: continuously harvest a good face crop for unmatched faces → create an "unlabeled" identity
(auto-name) with a thumbnail and surface it; annotate matched tracks with identity+confidence in TelemetryMsg.
Add setTargetIdentity(camera_id, identity_id) to EngineClient + SetTargetIdentityCmd in messages.py; the worker
targets that identity when detected. Serve thumbnails to the UI (image://identity/<id> provider or base64 role).
Everything must degrade gracefully without insightface/model/network. Keep `pytest -q` green with mocks; add
tests for enroll/match/merge/enable-disable and target-by-identity.
```

### Phase 9 — People view (top gallery) + soft tracking gate
**Goal.** The "more apparent" identities surface, per the locked decision (top view).
- A **People** button in the top status strip opens a full-width **PeopleView.qml** overlay:
  - **Enrolled** grid: thumbnail, name (inline rename), enable/disable, delete, "cameras
    following" indicator, multi-select **Merge**.
  - **Unlabeled tray**: auto-harvested faces with a quick name field → promotes to enrolled;
    discard.
  - Shows which identity each camera is currently tracking.
- Wire to Phase-8 service slots; thumbnails from the Phase-8 provider.
- **Soft gate**: surfaced in the tile (Phase 7) and here — informational, never blocking.
**Acceptance.** Open People → see live unlabeled faces, name/merge/enable them, and they
immediately become selectable as "Follow ▸ Name" targets on tiles.
```
PROMPT — Phase 9
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
Create autoptz/ui/qml/PeopleView.qml — a full-width gallery overlay opened from a new "People" button in the top
status strip of CameraWall.qml. Sections: (a) Enrolled — thumbnail + inline-rename + enable/disable toggle +
delete + a "following on N cameras" indicator + multi-select Merge (calls the Phase-8 merge slot); (b) Unlabeled
tray — auto-harvested faces (from identityModel where unlabeled) with a quick name field that promotes to
enrolled, and a discard. Show per-camera current target identity. Use engineClient.identityModel + the Phase-8
slots (enroll/rename/delete/enable/disable/merge) and identity thumbnails (image://identity/<id>). Remove the
duplicate Identities surface from the SettingsPanel window (People view replaces it). Theme everything; verify
with pyside6-qmllint.
```

### Phase 10 — Packaging & signed apps (incl. the real app-name fix)
**Goal.** Shippable, correctly-named apps.
- macOS: PyInstaller/py2app `.app` with `Info.plist` `CFBundleName=AutoPTZ` +
  `CFBundleDisplayName` (this is the definitive "AutoPTZ not Python" fix) + camera-usage
  string `NSCameraUsageDescription`; bundle NDI runtime, models, CoreML caches; notarize+sign.
- Windows: PyInstaller → installer; DirectML default + optional CUDA/TensorRT; bundle NDI +
  models. First-run model download + EP self-check.
**Acceptance.** Clean-machine install on macOS & Windows launches, shows "AutoPTZ" in the
menu/Dock, discovers a camera, and tracks — no dev environment.
```
PROMPT — Phase 10
Repo /Users/prince/development/autoptz, branch dev/v2-architecture-rework (no new branches/commits).
Package AutoPTZ. macOS: a PyInstaller (or py2app) spec producing AutoPTZ.app with Info.plist CFBundleName and
CFBundleDisplayName = "AutoPTZ" (this is the real fix for the app menu showing "Python"), NSCameraUsageDescription,
bundled NDI runtime + models dir + CoreML cache; add a notarize/sign step. Windows: a PyInstaller build →
installer, DirectML default with optional CUDA/TensorRT variant, bundled NDI + models. Add a first-run EP
self-check and model download. Document build commands in README. CI should produce artifacts for macOS-arm64
and windows-x64.
```

---

## Cross-cutting requirements (every phase)
- Stable UUIDs; no GUI-thread heavy work; graceful degrade when a dep/model/camera is absent.
- Tests per module (headless, mock ML/PyObjC); keep the suite green; short `CHANGELOG.md` note.
- Surface EP/quality/health in the UI; structured logs (now viewable via Phase 2).
- Keep v1 references out; this is v2-only.

## Suggested grouping
- **M1 "It runs & you see it"** = Phases 0–4 (auto-start, live video, logs, app identity,
  correct discovery).
- **M2 "Control & tune"** = Phases 5–7 (Auto-first config, real PTZ, redesigned tile).
- **M3 "Who to follow"** = Phases 8–9 (identity engine + People view).
- **M4 "Ship"** = Phase 10.

## Committed additions (review round 2 — confirmed by the user)
Folded into scope:
- **Framing presets** (head-and-shoulders / full-body) — extend `PTZConfig.zoom_framing` into named presets the auto-zoom targets (Phases 5–7).
- **Target-lock HUD** on the tracked person — name + lock ring + lead/direction indicator (Phase 7).
- **First-run onboarding checklist** — add camera → start engine → enroll a face (Phase 9 / late, once People exists).
- **Keyboard shortcut scheme** — select camera, follow target, recall preset, toggle manual, engine control (woven through the QML phases).
- **Manual speed presets + gamepad** support for the manual PTZ control (Phase 7).
- **Face data retention = labeled-only persisted**; the unlabeled tray is discarded on quit (Phase 8 store policy).
- **One target per camera** at a time (Phase 8 engine + controller).

Declined for now: cross-camera hand-off, tally/on-air, clip capture.

Pending: the user also picked an unspecified "something else" operator feature — to be captured.
