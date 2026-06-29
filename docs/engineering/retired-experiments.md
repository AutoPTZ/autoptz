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

## 2026-06-29 - Process-per-camera in the normal Experimental Features UI

### Problem Tried
`AUTOPTZ_PROCESS_PER_CAMERA` was exposed as a normal experimental toggle to bypass
the Python GIL by running one worker process per camera.

### Why It Failed
It can improve Python parallelism but duplicates model stacks and raises memory
and CPU pressure. That tradeoff is too sharp for the normal UI, especially when
the 2.2 target is eight reliable 1080p30 streams on mixed hardware.

### Removed
- `AUTOPTZ_PROCESS_PER_CAMERA` is no longer listed in the normal Experimental
  Features dialog.

### Replacement
The env flag remains a developer/Labs path only. Production defaults to shared
model ownership inside one supported runtime path.

### Reconsideration Criteria
Only promote a process mode if AutoPTZ Mark proves it beats the production path
on 6-camera and 8-camera gates without RAM cliffs, orphaned child processes, or
unbounded CPU growth.

## 2026-06-29 - Model-server in the normal Experimental Features UI

### Problem Tried
`AUTOPTZ_MODEL_SERVER` was exposed as a normal experimental toggle for per-camera
processes delegating detector work to one shared model-server process.

### Why It Failed
The implementation is promising but still a Labs architecture. It uses fixed
1080p shared-memory slots and serial detector requests. That is not yet a
general production scheduler contract for all source types and resolutions.

### Removed
- `AUTOPTZ_MODEL_SERVER` is no longer listed in the normal Experimental Features
  dialog.

### Replacement
Keep it as an explicit env-driven Labs path while the production architecture
gets a source-agnostic capture plane and CPU-safe scheduler.

### Reconsideration Criteria
Only promote after it passes the same release gates as production: 6 and 8 fake
NDI streams, CPU/RAM stability, clean shutdown, and no app-induced capture drops.
