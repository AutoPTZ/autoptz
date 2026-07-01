# Retired Experiments

This file records experiments that were removed from the normal product path.
Future agents should read this before reintroducing a similar architecture.

## 2026-06-29 - Mark subprocess relaunch and `--mark`

### Problem Tried
AutoPTZ Mark originally had a separate subprocess-style launch path. The app
could be started with a Mark-specific mode, and helper functions built a relaunch
argv that included `--mark`.

### Why It Failed
The product now treats Mark as an in-process Labs benchmark. Keeping a second
entry path made the lifecycle harder to reason about: normal AutoPTZ could be
suspended, Mark could run with its own isolated engine, and then the main window
could resume. A stale `--mark` compatibility path made tests and docs imply there
were still two application modes.

### Removed
- `autoptz.__main__` no longer accepts `--mark`.
- `autoptz.ui.app.run()` no longer has a Mark compatibility mode.
- `autoptz.ui.mark_session.relaunch_argv()` and `relaunch()` were removed.
- Tests that asserted the deprecated relaunch argv shape were removed.

### Replacement
Use the in-app AutoPTZ Mark flow from the UI, or `--benchmark` for headless Mark
throughput runs. Mark session settings remain persisted through `mark_session`.

### Reconsideration Criteria
Only reintroduce an external Mark launcher if it has a distinct executable,
distinct process contract, and release tests for start, stop, crash cleanup, and
return-to-main behavior. Do not restore `--mark` as a silent no-op.

## 2026-06-29 - User-facing tracking speed presets

### Problem Tried
The UI exposed Calm, Normal, Fast, and Sport tracking speed presets. Each preset
rewrote controller tuning values such as pan speed, tilt speed, gain, smoothing,
acceleration, and catch-up speed.

### Why It Failed
The presets pushed controller engineering choices onto users. In practice, users
want AutoPTZ to track reliably without knowing whether a failure is caused by
speed, gain, smoothing, catch-up, or prediction. Multiple visible speed choices
also made it harder to define one release-quality default for Intel Macs and
CPU-only Windows machines.

### Removed
- `TrackingSpeed`, `SPEED_PROFILES`, and `apply_speed_profile`.
- `PTZConfig.tracking_speed`.
- The Properties-panel Tracking Speed segmented control.
- The dedicated speed-preset tests.

### Replacement
The controller keeps internal tuning fields for compatibility and tests, but the
normal UI no longer exposes speed presets or the Advanced tracking tuning group.
2.2 work should move toward one adaptive controller that uses measured frame age,
target error, target velocity, PTZ send latency, and bounded acceleration.

### Reconsideration Criteria
Do not reintroduce user speed presets unless production telemetry proves one
adaptive controller cannot handle a supported camera class. If a control returns,
it must be a camera capability/profile selected automatically or during setup,
not a day-to-day tracking choice.

## 2026-06-30 - User-editable framing box / safe-zone controls

### Problem Tried
The normal camera tile and Properties panel exposed a movable/resizable framing
box. The same `safe_zone_*` config fields also feed the PTZ controller's internal
quiet zone: the controller holds still while the target is near center, resumes
only after hysteresis is crossed, and eases the target back toward center.

### Why It Failed
The visible editor made an internal control primitive look like a product
feature. Users could reasonably think they were drawing the person's bounding
box or solving head/body framing manually. That conflicts with the 2.2 product
goal: one adaptive follow mode that decides speed and framing automatically.

It also added more places where stale code could affect tracking: tile drag
state, fast config pushes, Properties sliders, and tooltip copy all implied that
operators should tune a control that is supposed to be automatic. For bobbing,
removing the quiet zone itself would be the wrong fix; the quiet zone is the
deadband/hysteresis that prevents small detector jitter from commanding PTZ.
The failed part was making that deadband user-editable in normal workflow.

### Removed
- The off-layout Properties-panel `Advanced tracking` builder and its gain,
  speed, catch-up, prediction, and safe-zone sliders.
- The fast framing-box config push timer and center/reset helpers.
- Camera-tile drag, resize, hover-cursor, live-state, and persist methods for
  the framing box.
- The remaining passive dashed quiet-zone/crosshair overlay in the camera tile.
- Normal UI text that described dragging/resizing the box.

### Replacement
Keep the internal `safe_zone_*` fields as controller compatibility/config values,
but treat them as automatic internals. Normal UI must not draw a safe-zone box,
circle, crosshair, or other center-zone indicator: users should see the tracked
person marker and result, not a controller primitive. Debug overlays may be
added only behind a developer-only diagnostic flag with tests proving they do
not appear in production UI.
The person detector bounding box remains separate: it is evidence for detection,
tracking, association, and framing estimates; raw bbox geometry must not directly
command PTZ when it is stale, degenerate, or shape-jumpy.

### Reconsideration Criteria
Do not reintroduce a normal user-editable center zone unless real hardware tests
prove AutoPTZ cannot adapt without operator-set camera geometry. If it returns,
it needs a setup-only calibration flow with pass/fail validation, not a live
drag box in the main tracking surface. Do not remove the internal deadband unless
it is replaced by another tested hysteresis/hold mechanism that prevents
micro-corrections and bobbing.

## 2026-06-29 - Normal Help-menu Experimental Features entry

### Problem Tried
The app exposed a normal Help-menu `Experimental Features...` dialog with engine
flags and new-camera tracking defaults. This made implementation probes look like
supported product modes.

### Why It Failed
AutoPTZ 2.2 needs one reliable production behavior, not a menu of runtime
research switches. A normal user should not have to decide whether unified pose,
PTZ pump, true latency lead, ReID device selection, CoreML unit selection, or NDI
receive color format is the correct deployment answer. Those are release-gate
questions for Mark artifacts and maintainer tools.

### Removed
- The `Help -> Experimental Features...` menu action.
- Normal app routing to `ExperimentalFeaturesDialog`.

### Replacement
AutoPTZ Mark remains the visible Labs benchmark because it exercises realistic
deployments and emits validation artifacts. Engine flags stay as dev/benchmark
inputs only: set by environment or explicit test harnesses, never presented as a
normal product mode. A dev/benchmark flag must either be promoted to an automatic
default after evidence, or deleted and documented here.

### Reconsideration Criteria
Do not restore a generic experimental menu. If a setting is important enough for
users, make it part of a focused setup flow with a safe default and platform
evidence. If it is only for validation, keep it in Mark/headless tools or env.

## 2026-06-29 - Process-per-camera in the normal Experimental Features UI

### Problem Tried
`AUTOPTZ_PROCESS_PER_CAMERA` was exposed as a normal experimental toggle to bypass
the Python GIL by running one full camera worker process per camera.

### Why It Failed
It can improve Python parallelism but duplicates model stacks and raises memory
and CPU pressure. That tradeoff is too sharp for the normal UI, especially when
the 2.2 target is eight reliable 1080p30 streams on mixed hardware.

### Removed
- `AUTOPTZ_PROCESS_PER_CAMERA` is no longer listed in the normal Experimental
  Features dialog.
- `AUTOPTZ_PROCESS_PER_CAMERA` is now ignored by the env parser.
- The supervisor no longer falls back to model-per-child children when the shared
  model-server queues are absent; it uses the normal threaded worker instead.

### Replacement
No standalone replacement. Production defaults to the normal threaded worker with
shared in-process model ownership. The only remaining process-worker path is the
separate `AUTOPTZ_MODEL_SERVER` candidate, where camera children delegate detector
work to one shared server instead of loading their own model set.

### Reconsideration Criteria
Do not reintroduce the model-per-child flag. If process isolation is needed, it
must use shared model ownership and pass the 6-camera and 8-camera gates without
RAM cliffs, orphaned child processes, or unbounded CPU growth.

## 2026-06-29 - Model-server in the normal Experimental Features UI

### Problem Tried
`AUTOPTZ_MODEL_SERVER` was exposed as a normal experimental toggle for per-camera
processes delegating detector work to one shared model-server process.

### Why It Failed
The implementation is promising because it avoids the model-per-child RAM cliff,
but it is not yet a production scheduler contract. It still needs deterministic
fake-NDI gates, source preflight, clean dynamic membership, Windows/macOS shutdown
proof, and 30-minute 8x1080p30 validation with zero steady-state app-induced
capture drops.

The 2026-06-30 short fake-NDI artifact showed why this remains a candidate, not
a default: `full` profile median latency improved from about 2268 ms in the
default threaded path to about 42 ms with `AUTOPTZ_MODEL_SERVER=1`, but RAM rose
to about 6.4 GB and child logs showed InsightFace still initializing per camera.
Detector ownership is centralized enough to fix latency; face/appearance
ownership is not yet centralized enough to ship.

### Removed
- `AUTOPTZ_MODEL_SERVER` is no longer listed in the normal Experimental Features
  dialog.

### Replacement
Keep it only as an explicit env-driven release-gate candidate while the production
architecture gets a source-agnostic capture plane and CPU-safe scheduler. It is
not a user-facing feature and must be deleted or promoted based on Mark artifacts,
not preference.

### Reconsideration Criteria
Only promote after it passes the same release gates as production: 6 and 8 fake
NDI streams, CPU/RAM stability, clean shutdown, and no app-induced capture drops.
