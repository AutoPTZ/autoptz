# Streaming + Tracking Redesign — Research & Architecture Report

> Status: **research / proposal — review before code.** Scope: fix (1) random
> frame drops with multiple NDI streams and (2) unstable PTZ tracking on NDI
> cameras (bouncing / lag / wrong‑way moves). Targets macOS + Windows + Linux as
> equal first‑class platforms. Greenfield/any‑stack was on the table; the honest
> finding (below) is that the bottleneck is **architecture, not language**.
>
> Method: 5 parallel codebase analyses + 7 web‑research tracks + a 7‑claim
> adversarial verification pass. Where a "load‑bearing" assumption was checked and
> changed, it is called out as **[verified]**. Sources are linked inline.

---

## 0. Primary-source refresh (2026-06-29)

These are the current load-bearing facts for the 2.2 ingest and scheduler work.
They are intentionally explicit so future agents do not reinterpret the
performance plan as "just use multiprocessing" or "just switch languages."

- **NDI receive format:** NDI's SDK documentation says
  `NDIlib_recv_color_format_fastest` returns buffers in the format the SDK
  processes internally, without conversion before delivery, and is the best
  performance / lower-latency receive path. On most no-alpha sources this is
  UYVY; `allow_video_fields` is effectively true in this mode.
  Source: https://docs.ndi.video/all/developing-with-ndi/sdk/ndi-recv
- **NDI receive loop:** NDI's performance guide explicitly recommends
  `NDIlib_recv_color_format_fastest` for receiving, separate receive waits for
  audio/video where needed, and a reasonable `NDIlib_recv_capture_v3` timeout
  instead of zero-timeout polling. This directly supports changing AutoPTZ's NDI
  path from Python-paced polling to blocking receive with measured pacing.
  Source:
  https://docs.ndi.video/all/developing-with-ndi/sdk/performance-and-implementation
- **BGRA/BGRX cost:** The same NDI performance guide warns that BGRA/BGRX incur a
  memory-bandwidth and conversion penalty. For AutoPTZ, that means BGRA is a UI
  boundary format, not the production capture format. Capturing in BGRA just to
  convert/copy again later is release-blocking waste.
  Source:
  https://docs.ndi.video/all/developing-with-ndi/sdk/performance-and-implementation
- **cyndilib mapping:** cyndilib exposes the same NDI receive format choices;
  its docs describe `fastest` as the best-performance receive mode and show the
  usual no-alpha result as UYVY. AutoPTZ should keep the actual FourCC in
  `SourceHealth` rather than hiding it behind "BGR frame arrived."
  Source: https://cyndilib.readthedocs.io/en/latest/reference/wrapper/ndi_recv.html
- **Receiver diagnostics are first-class, not optional:** the NDI receive SDK
  exposes receiver performance and queue-depth APIs for dropped/dequeued frames
  and pending queues. AutoPTZ's Python wrapper does not expose every native field
  today, so the production contract must record every available backend counter
  and avoid treating "no counter exposed" as "healthy."
  Source: https://docs.ndi.video/all/developing-with-ndi/sdk/ndi-recv
- **Timestamp/timecode are the cheap duplicate detector:** NDI Analysis documents
  per-video-frame timestamp and timecode fields in 100 ns units and uses them to
  compare sender vs receiver frame timing. AutoPTZ should use those scalar source
  stamps, when exposed by the wrapper, to count duplicate/stale FrameSync returns
  without hashing full 1080p frames on the capture path.
  Source: https://docs.ndi.video/all/using-ndi/utilities/analysis
- **ONNX Runtime CPU thread pools:** ORT's default
  `intra_op_num_threads = 0` creates intra-op worker threads up to the number of
  physical CPU cores per session, with spinning enabled by default. If AutoPTZ
  creates a session per camera or per process on CPU-only hosts, it can multiply
  full-core thread pools and create CPU oversubscription even when Python itself
  is not the immediate bottleneck. Production must centralize model ownership or
  explicitly cap session threads and disable/limit spinning on CPU profiles.
  Source: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- **Python GIL reality:** Python's threading docs still state that only one
  thread executes Python bytecode at a time in normal CPython builds, while
  threads remain appropriate for I/O-bound work. Python's FAQ recommends
  processes or C extensions that release the GIL for CPU work. For AutoPTZ this
  means: keep receive threads I/O/native-heavy, move Python CPU work off capture,
  and avoid per-camera process duplication unless memory and ORT thread budgets
  are proven.
  Sources:
  https://docs.python.org/3/library/threading.html and
  https://docs.python.org/3/_sources/faq/library.rst.txt

Implementation consequences for 2.2:

- The 8-stream gate measures **app-induced capture drops**, not detector skips.
  Inference cadence may be reduced on CPU-only hosts, but capture must still
  drain every source except during add/remove-source transitions.
- NDI capture must expose actual FourCC, SDK buffer handoff time, format
  conversion time, final contiguous-copy time, delivered fps, duplicate/stale
  counts, and receiver/backend counters in `SourceHealth`.
  Native NDI performance counters (`total_*` / `dropped_*`) are not equivalent to
  AutoPTZ's `frames_dropped_est`: native dropped frames mean the receiver had
  frames available that the app did not dequeue fast enough, while the estimate
  compares observed delivered cadence against source cadence when the wrapper
  exposes no direct dropped-frame signal.
- The production receiver must prefer `fastest` / native YCbCr receive and
  convert once at the boundary that actually needs BGR/RGB. Preview and model
  input may choose different downscale/convert points, but capture must not pay
  repeated full-frame conversion costs.
- Per-camera processes are not a production answer by themselves. They bypass
  the GIL for Python bytecode, but they also multiply model memory and ORT thread
  pools. The default 2.2 scheduler should use shared model ownership plus
  explicit CPU cadence/thread caps; Labs can keep process/model-server variants
  only when Mark artifacts show a net win.

---

## 1. Executive summary

Both problems are real and both are **fixable without a rewrite**, but the popular
one‑line diagnoses are each only half right:

- **"NDI frame drops are the GIL."** **[verified: PARTIAL → mostly *secondary*.]**
  The drops are first a *consumer‑throughput* problem: AutoPTZ does **2–3
  full‑frame CPU passes per NDI frame in Python** on the capture thread
  ([`ingest.py:1063‑1073`](../../autoptz/engine/pipeline/ingest.py), color‑convert
  at `:902‑918`), and requests the **heavy `BGRX_BGRA`** color format (`:998`)
  which forces the SDK to convert YUV→BGRA for every frame. NDI keeps only a
  *short internal queue* and **silently drops** whatever the consumer can't drain
  in time ([NDI recv docs](https://docs.ndi.video/all/developing-with-ndi/sdk/ndi-recv)).
  OBS and Resolume — both C++, no GIL — drop frames on 3+ NDI sources for the same
  reason. The GIL is a **second‑order ceiling** that only dominates *once heavy
  per‑stream Python work (detector post‑proc, tracker, paint) is stacked on top* —
  which AutoPTZ does, three threads per camera in one interpreter. So the fix is
  **both**: cut the per‑frame copy/convert cost *and* get the heavy pipeline off
  the shared GIL.

- **"NDI tracking bounces because NDI is laggy."** **[verified: the latency is
  real but it is *not mostly the NDI link*.]** Full‑bandwidth NDI is ~16 ms (one
  field) and NDI|HX3 ~≤100 ms; only **NDI|HX2 long‑GOP** adds 100–300 ms
  ([NDI latency docs](https://docs.ndi.video/all/developing-with-ndi/advanced-sdk/using-h.264-h.265-and-aac-codecs/latency-of-compressed-streams)).
  The dead time that actually destabilizes the loop is the **whole pipeline**:
  capture → detector inference → **command transport (VISCA/ONVIF SOAP)** →
  **motor actuation** → next frame. AutoPTZ's controller is sophisticated (one‑euro
  filter, PID + velocity feed‑forward, latency‑lead, oscillation guard, slew,
  coast/search — [`controller.py`](../../autoptz/engine/ptz/controller.py)) but it
  is fed a **latency that excludes the command round‑trip and actuation** (only
  `ingest_ms + inference_ms`, [`camera_worker.py:2698`](../../autoptz/engine/camera_worker.py)),
  runs **inline on a jittery inference thread** so its `dt`‑based derivative and
  the one‑euro frequency estimate are corrupted by cadence jitter, and has **no
  predictive target estimator** — it leads with `velocity × under‑measured‑latency`
  off a noisy single‑frame velocity. That is the textbook recipe for dead‑time
  oscillation. **[verified: SUPPORTED]** — "raw faster tracking does not fix
  oscillation; predict the target forward by the measured dead time and damp the
  loop critically."

**Headline recommendation:** a **disciplined, Frigate‑style redesign of the
existing thin‑Python‑over‑native‑core**, not a greenfield rewrite. Concretely:
decouple **capture** (per‑source, receive‑only, drop‑oldest) from **inference**
(a **shared model‑server**, not one model per process) from **control** (a
fixed‑rate predictive loop), all connected by the zero‑copy shared memory AutoPTZ
already has. Add a **predictive (Kalman/alpha‑beta) target estimator** fed the
**true measured end‑to‑end latency**, and switch to a **fixed‑rate velocity
control loop** decoupled from frame arrival. A selective **native (Rust/C) ingest
shim** is an optional later win for deterministic tail latency — but **[verified:
PARTIAL]** the evidence (Frigate is Python‑cored and does exactly this workload)
says language is *not* the lever; architecture is.

---

## 2. Root‑cause analysis

### 2.1 Frame drops on multiple NDI streams

| # | Cause | Evidence | Class |
|---|-------|----------|-------|
| 1 | 2–3 full‑frame Python/numpy passes per NDI frame on the capture thread (`np.asarray`→`reshape`→`cvtColor`→`ascontiguousarray`) | [`ingest.py:1063‑1073`, `:902‑918`](../../autoptz/engine/pipeline/ingest.py) | consumer cost |
| 2 | Requests `BGRX_BGRA` → SDK does a hidden full‑frame YUV→BGRA convert/frame; a single 16‑bit frame *permanently* downgrades a source to the heavy `bgra` path | `ingest.py:847‑867, :998, :1079‑1086` | consumer cost |
| 3 | Frame copied into shm by `frame.ravel()` + slice‑assign, with a `cv2.resize` first if not 720p | [`shm.py:148`](../../autoptz/engine/runtime/shm.py), `ingest.py:311‑315` | consumer cost |
| 4 | NDI receive is a **Python‑paced poll** (`capture_video()` latest‑snapshot) gated by a `time.sleep`; a late poll re‑reads or skips, and the SDK's short queue overflows silently | `ingest.py:1057`, [`frame_source.py:100‑129`](../../autoptz/engine/worker/frame_source.py) | architecture |
| 5 | All N cameras' capture+inference+appearance threads share **one GIL**; per‑frame Python glue can't run in parallel | `camera_worker.py:738/2622/2932`; process isolation is opt‑in only | GIL ceiling |
| 6 | A transient GIL‑starved miss is treated like a real stall → backoff *slows* the poll (making the next miss more likely) → reconnect storms | `camera_worker.py:2723‑2735`, `ingest.py:276‑285` | feedback trap |

**The mechanism is consumer back‑pressure, amplified by the GIL.** Each NDI
receiver keeps a short queue and drops what isn't drained in real time; the
capture thread spends its budget on color conversion + copies + (under load) waits
on the GIL while an inference thread holds it, so the queue overflows. Fixing the
copy chain and getting receive off the heavy path removes most drops *before* any
parallelism work; parallelism removes the rest once the AI pipeline is the limiter.

### 2.2 PTZ tracking bounce / lag / wrong‑way on NDI

| # | Cause | Evidence | Effect |
|---|-------|----------|--------|
| 1 | Lead/feed‑forward latency **excludes** NDI transport + command round‑trip + actuation (only local `ingest+inference`) | `camera_worker.py:1406, :2698`; `controller.py:738‑743` | under‑leads → chases stale error → overshoot |
| 2 | Control runs **inline on the inference thread**, whose cadence jitters (detector decimation, GIL); the PD derivative `(e‑e_prev)/dt` and the one‑euro `freq=1/Δt` are corrupted by that jitter | `camera_worker.py:1368`; `controller.py:765‑768, :131‑132` | jitter reads as oscillation |
| 3 | Velocity is **pixels/frame, no `dt` scaling**, and is re‑fed **stale detections** on skip frames; the fallback IoU tracker has **no Kalman predict** | [`track.py:234, :510‑513`](../../autoptz/engine/pipeline/track.py); `camera_worker.py:3769‑3778` | velocity wrong under jitter |
| 4 | Ego‑motion measured every 3rd frame; aim velocity **holds the previous value** off‑cadence → ~3–9 Hz feed‑forward | `camera_worker.py:1469‑1476, :1517‑1523` | stale anticipation |
| 5 | **Wrong‑way**: NDI negates pan internally and there's also `invert_pan`; **no position feedback** on NDI to detect a sign/gain error — only the unstable visual loop can | [`ndi_ptz.py:18‑19,150,172‑173`](../../autoptz/engine/ptz/ndi_ptz.py); `controller.py:826‑841` | sign error → drives away, loses subject |
| 6 | Every `move_velocity` is a **blocking transport call inline** (VISCA `sendall`, ONVIF SOAP `ContinuousMove`); on GIL‑bound NDI threads a slow send stalls that camera's whole tick — **couples both pains** | `controller.py:590`; [`visca_ip.py:126‑169`](../../autoptz/engine/ptz/visca_ip.py), `onvif_ptz.py:131‑144` | widens control cadence |
| 7 | The fixed‑rate **PTZ pump** (the structural fix) is **default OFF** (`AUTOPTZ_PTZ_PUMP=0`); commands emit inline from the jittery thread | `camera_worker.py:209‑225` | the cure is dormant |

**This is dead‑time‑dominated control.** **[verified: SUPPORTED]** Pure transport
delay subtracts phase linearly with frequency, erodes phase margin, and forces gain
down or the loop oscillates; cranking gain/rate makes it *worse*
([phase‑margin / dead‑time control](https://pmc.ncbi.nlm.nih.gov/articles/PMC11398195/)).
The remedies are well‑established: **predict the target forward by the measured
dead time** (Kalman/alpha‑beta), **command velocity not position**, **decouple the
command rate from the frame rate**, and **damp critically** (P‑dominant,
filtered‑D, Ki≈0). Integral action is *actively dangerous* under dead time (windup
while waiting on stale feedback).

---

## 3. Current architecture — strengths and excess

**Genuinely good and worth keeping:** the lock‑free triple‑buffered shared‑memory
frame ring ([`shm.py`](../../autoptz/engine/runtime/shm.py)); the controller's
anti‑jitter toolkit (one‑euro, slew, deadband, osc‑guard, coast/search); the ONNX
+ EP‑fallback inference layer; the process‑per‑camera scaffolding already exists.
AutoPTZ is **already a thin‑Python‑over‑native‑core design** — the redesign hardens
it, it doesn't start over.

**Excess / over‑engineering a redesign should shed** (from the codebase analysis):

- **`CameraWorker` is a 5,133‑line god‑class** — ~160 methods, ~129 instance
  attributes (80 set in `__init__`), owning capture, inference, appearance, ~13
  pipeline subsystems, PTZ, identity, telemetry, quality governance and watchdogs
  in one object guarded by ~6 ad‑hoc locks. This is the single biggest obstacle to
  fixing the latency loop in one place.
- **Telemetry is over‑built**: a fat `TelemetryMsg` with ~15 nested diagnostic
  model‑lists (`runtime_services`, `stage_timings`, `quality_state`, `model_switch`,
  `ground_truth`, …) is assembled ~10×/s/camera and marshaled to the GUI thread —
  most is diagnostics‑panel data that belongs on a separate low‑rate channel.
- **Two ingest architectures coexist**: `SourceAdapter` has its own capture thread
  + shm writer + pacing (`ingest.py:213‑294`) that the worker **never uses** —
  `_AdapterFrameSource` re‑implements pacing instead. Dead code, real confusion.
- **Two target‑decision systems**: `TargetAssociator` is fully built+tested but
  wired **OFF** by default; the legacy heuristic it was meant to replace still runs.
- **Two pose paths**: a per‑crop second ONNX forward (`pose.py`) *and* a unified
  keypoints path (`pose_detect.py`) — redundant compute when unified is available.
- **Inert thread‑cap env vars**: OMP/BLAS/MKL/NUMEXPR are published in‑process where
  the file's own docstring admits they're ineffective after import (`flags.py:93‑99`).
- **Mark/bench scaffolding in the production hot path**: synthetic source +
  ground‑truth + transcode cache live in `ingest.py`/`TelemetryMsg`, enlarging the
  surface that must be reasoned about for zero‑copy guarantees.
- **Three near‑duplicate model‑lifecycle methods** on the supervisor
  (`release`/`rebuild`/`apply_model_cache_changed`, `supervisor.py:768‑832`); an
  **unlocked `_workers` iteration** (`supervisor.py:463`) racing mutation elsewhere.

---

## 4. How leading PTZ/streaming systems stay stable (reference)

| System | What it does | Transplantable lesson |
|--------|--------------|------------------------|
| **OBS Studio** | Dedicated graphics thread at fixed FPS; async sources push timestamped frames into per‑source queues; encode on a 3rd thread; GPU color convert ([backend‑design](https://docs.obsproject.com/backend-design)) | Separate capture / process / output; schedule by **absolute timestamp**, never "latest arrived"; bounded drop‑oldest queue (avoid the [runaway‑latency bug](https://github.com/obsproject/obs-studio/discussions/11142)) |
| **DistroAV (obs‑ndi)** | **One dedicated receiver thread per NDI source** doing recv→convert→handoff only ([3.1‑ndi‑source](https://deepwiki.com/DistroAV/DistroAV/3.1-ndi-source)) | Never multiplex NDI receives on one thread; receive‑only on the capture thread |
| **obs‑face‑tracker** | PID **+ integrator** with **dead band + nonlinear band + integral attenuation on loss**, detection on its own thread via a circular buffer ([properties‑ptz](https://github.com/norihiro/obs-face-tracker/blob/main/doc/properties-ptz.md)) | A battle‑tested, directly portable control law + the deadband/attenuation anti‑jitter stack |
| **glikely/obs‑ptz** | Control transport entirely separate from video; per‑protocol sockets/serial; "lockout live moves" ([obs‑ptz](https://github.com/glikely/obs-ptz)) | Send commands on their own thread; gate abrupt moves |
| **Frigate NVR** | **Process per camera** (FFmpeg) writing raw frames to **POSIX shared memory**; only metadata tuples on the queue; **drop‑on‑full** ("skipped FPS") ([frame‑processing/SHM](https://deepwiki.com/blakeblackshear/frigate/4.2-frame-processing-and-shared-memory)) | The canonical Python multi‑camera escape from the GIL — and proof a Python core suffices |
| **Commercial (PTZOptics/Panasonic/AVer/Axis)** | Track **on‑camera** (no network round‑trip); slow continuous small‑correction loop (~500 ms), deadzone, **continuous‑velocity** moves; Axis uses absolute **radar** (feed‑forward, no visual loop) ([PTZOptics Move](https://ptzoptics.com/move-se/), [Axis radar](https://www.axis.com/products/axis-radar-autotracking-for-ptz)) | Low‑gain critically‑damped continuous‑velocity control; deadzone; where possible feed‑forward from an absolute estimate, not pixel‑error chasing |
| **Huddly / Logitech** | AI **digital crop/pan** within a fixed wide sensor — no motors, no overshoot ([Huddly L1](https://www.huddly.com/conference-cameras/l1/)) | Prefer the **digital crop** for fine framing (instant, no inertia); reserve mechanical PTZ for coarse moves |
| **Robotics / IBVS** | Fast inner loop (>200 Hz) interpolating slow vision (3–5 Hz); **Kalman/EKF predict‑ahead**; **feed‑forward** on target rate; Smith predictor for known dead time ([IBVS review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11280684/), [PVT++ ICCV'23](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_PVT_A_Simple_End-to-End_Latency-Aware_Visual_Tracking_Framework_ICCV_2023_paper.pdf)) | **Decouple command rate from sensor rate**; predict over measured latency; latency‑aware evaluation |
| **Frigate autotrack / Roboflow** | Center error → **velocity vector**, 10–20 Hz loop, EMA, deadzone ~0.5% FOV, accel/jerk limits; PID `Ki≈0`, raise `Kp` to overshoot then add `Kd` ([Frigate #20903](https://github.com/blakeblackshear/frigate/issues/20903), [Roboflow PTZ](https://blog.roboflow.com/control-ptz-camera-computer-vision/)) | Concrete, shipping velocity‑loop + tuning recipe |

---

## 5. Recommended architecture

A pipeline of **single‑responsibility stages** connected by explicit single‑writer
queues + the existing zero‑copy SHM, with **three independent clocks**: capture
(source rate), inference (decimated), control (fixed rate). Cross‑platform by
construction (all native deps below exist on macOS/Windows/Linux).

```
            per source                shared (1 per machine)         per source, fixed-rate
  ┌──────────────────────┐      ┌───────────────────────────┐     ┌───────────────────────┐
  │ Capture worker       │ shm  │ Model server (1 ORT/face  │     │ Control loop (20-60Hz)│
  │  • recv UYVY/fastest  ├─────▶│  set, batched inference)  ├────▶│  • Kalman predict     │
  │  • convert once → shm │ ring │  • detect/track/reid/pose │ shm │    forward by measured│
  │  • drop-oldest        │      │  • emits (state,ts,cov)   │ /q  │    dead-time          │
  │  • recv_get_perf      │      └───────────────────────────┘     │  • velocity command   │
  └──────────────────────┘                                        │  • send on own thread │
        (own GIL/proc)                  (own GIL/proc)             └───────────────────────┘
```

### 5.1 Streaming / intake (fixes the drops)

1. **Receive `NDIlib_recv_color_format_fastest` (UYVY)**, not BGRA — NDI is
   memory‑bandwidth bound; convert to BGR **once**, ideally straight into the shm
   destination buffer (`cvtColor` with a preallocated `dst` that is the ring slot).
   Collapse the 3‑copy chain to one.
2. **One receive‑only thread/process per source**: capture + free into a **single
   latest‑frame slot (drop‑oldest)**; do *no* color/inference work there. Use a
   blocking capture with a **non‑zero timeout** (yields the GIL in cyndilib's
   Cython) instead of the busy `time.sleep` poll.
3. **Make drops observable**: surface `NDIlib_recv_get_performance` (dropped count)
   and `get_queue` depth per source as telemetry — never fail silently.
4. **Do not use `NDIlib_framesync` in the tracking path** — it re‑clocks and
   inserts/duplicates frames, adding latency and feeding the tracker stale frames.
5. **Decouple reconnect from transient misses**: a `None` on a healthy receiver
   must not slow the poll; only a true `stall_timeout` gap reconnects, and
   reconnects are jittered to avoid synchronized discovery storms.
6. **Run inference on a bandwidth‑lowest / lower‑res proxy** where available; pull
   full bandwidth only for display.

### 5.2 Parallelism (fixes multi‑stream scaling) — **[verified, refined: do we even need per‑process?]**

**Short answer: probably not as a default, and not in its current form.** Today's
`AUTOPTZ_PROCESS_PER_CAMERA` showed **no real‑world benefit** because it solves the
*secondary* cause (the GIL) at high cost — it **duplicates the entire model set per
child** (RAM + more total CPU as N full pipelines run at once) and adds IPC, while
the *primary* cause (per‑frame copy/convert) is untouched (each child still does the
heavy BGRA convert). Until [PR #134] even fixed the CPU accounting it *looked* like
nothing was using the CPU the children were burning. It targeted GIL contention
before the cheaper, bigger win (cut per‑frame work + batch inference) was taken.

The redesign therefore converges on **ONE concurrency model, not a "normal vs
per‑process" fork**, and the cures in §5.1/§5.3 are threading‑agnostic — they fix
every camera the same way regardless of thread vs process:

- **Default = single process.** Capture is a **receive‑only thread per source**
  doing *only* native work (NDI recv + one `cvtColor` into the shm slot) — both
  **release the GIL**, so threads give true parallelism here once the per‑frame
  Python glue is gone. **[Phase 0 finding — resolved]** cyndilib actually *holds*
  the GIL during capture (no `with nogil:` on its call chain), **but** AutoPTZ uses
  the *non‑blocking* FrameSync snapshot (a microsecond‑scale memcpy of the latest
  frame, not a blocking network wait) and the expensive `cv2.cvtColor` releases the
  GIL — so the GIL bite on NDI capture today is negligible and **capture can stay
  threads**. (Critical corollary: do **not** "fix" this by switching to cyndilib's
  blocking `RecvThread`/`Receiver.receive()` — that path holds the GIL across the
  blocking receive and would be strictly worse.) The thread‑vs‑process decision is
  therefore driven by *whole‑pipeline* GIL contention, not capture.
- **Inference = a single shared, batched stage.** ORT already runs ops on native
  threads; the GIL‑bound part is the Python glue (NMS, letterbox, tracker update).
  Run that glue **once per N‑camera batch**, not once per camera‑thread, and the
  per‑camera GIL contention largely disappears — the DeepStream/Frigate pattern.
  This caps RAM at **one** model set (vs one‑per‑child today).
- **Put the capture↔inference↔control boundaries behind one `Worker` Protocol** so
  the boundary can be a thread *or* a process as a **measured, internal choice —
  never a user‑facing mode.**
- **Per‑process is demoted to two narrow, measured roles** (always with the shared
  model‑server, never model‑per‑child): (a) **fault isolation** — a flaky NDI
  source / buggy driver that segfaults can't take down the GUI (a reliability win,
  the only durable reason to keep it); (b) **last‑resort parallelism** — only if
  Phase‑0 profiling shows residual *Python‑glue* contention that batching can't
  remove. **Decision rule:** default to threads + shared batched inference; promote
  a stage to a process only when a profiler proves it's GIL‑bound on Python work
  that can't be batched away, or when a source is crash‑prone.
- **Free‑threaded CPython is NOT the path today [verified: SUPPORTED]**: OpenCV has
  no `cp313t` wheels (limited‑API blocker, [opencv/opencv#27933](https://github.com/opencv/opencv-python/issues/1029)),
  and onnxruntime **re‑enables the GIL on import** on 3.13t
  ([microsoft/onnxruntime#26780]). Target **3.14t** as a *future* bet, gated on
  `sys._is_gil_enabled()` staying false and every C‑ext declaring `Py_mod_gil`.
- Recompute the per‑camera **thread budget on every add/remove** (today it's
  start‑only) and drop the inert OMP/BLAS env vars in‑process.

### 5.3 Control (fixes the bounce) — the keystone

1. **Decouple command rate from frame rate.** Make the fixed‑rate pump
   (`controller._loop`) the **default** (validate, then flip `AUTOPTZ_PTZ_PUMP`
   on). The inference stage only `update()`s the latest target state; a steady
   20–60 Hz loop owns command emission with a constant `dt` (which fixes the
   jittered derivative and one‑euro frequency by construction).
2. **Predictive target estimator.** Timestamp every detection; run a
   nearly‑constant‑velocity **Kalman or alpha‑beta** filter; command toward
   `pos + vel·T_d` (+ `½·a·T_d²` only when a maneuver is detected, IMM‑style). This
   is the single highest‑leverage fix. **Fold the dead‑time compensation into the
   Kalman predict** rather than a separate Smith predictor — **[verified]** the
   Smith predictor is fragile to *jittery* delay (AutoPTZ's latency varies across
   the band), whereas predicting forward by the **measured per‑frame age** is robust.
3. **Measure the *true* end‑to‑end dead time** — capture/NDI timestamp → detection
   → command‑applied (including command transport + actuation), per source — and
   feed *that* to the predictor, not `ingest+inference`. cyndilib exposes frame
   timestamps.
4. **Velocity (continuous‑move) control** at the inner‑loop rate with an explicit
   stop; **critically‑damped** PID (P‑dominant, **filtered** D, **Ki≈0**), with
   `Kp` lower and `Kd` higher as measured latency rises.
5. **Confidence‑gated adaptive deadband + hysteresis** (shrink when the predictor
   is confident, widen when uncertain) — kills micro‑hunting without adding its own
   dead‑time limit cycle.
6. **Startup direction/gain self‑calibration**: command a small known pan, observe
   the image‑shift sign/magnitude, auto‑correct `invert_pan`/gain. This directly
   kills the **wrong‑way‑and‑lose‑the‑subject** failure that today only the
   unstable visual loop can catch (NDI has no position feedback).
7. **Backend sends on a dedicated non‑blocking thread**; **dead‑man stop on NDI**;
   stop‑on‑loss heartbeat active in **all** modes (not just pump). This also
   *decouples the two pains* — blocking sends no longer stall GIL‑bound NDI ticks.
8. **Units fix in the tracker**: express all motion in **world/seconds** (divide
   every centre delta by the real `dt` between the two boxes), give the fallback
   IoU tracker a **constant‑velocity Kalman predict** on skip frames, and tag each
   box with measured‑vs‑predicted + age so the controller leads proportional to
   staleness instead of trusting a frozen box.
9. **Digital‑first framing**: prefer the existing center‑stage **digital crop** for
   fine framing (instant, no inertia, no overshoot) and reserve mechanical PTZ for
   coarse repositioning — the Huddly/Logitech lesson.

### 5.4 Decomposition + stack — **[verified]**

- Split the god `CameraWorker` along its pipeline seams (**Ingest, Detect+Track,
  Appearance, Aim/Control, Output, Telemetry**) behind one explicit **Worker
  Protocol/ABC**, so the supervisor stops duck‑typing with scattered `hasattr`
  and the thread/process implementations can't drift.
- **Two telemetry channels**: a tiny hot frame (tracks + aim + PTZ state at preview
  rate) and a separate low‑rate diagnostics frame.
- **Stack: keep a Python core.** **[verified: PARTIAL/overstated → keep Python.]**
  Greenfield‑any‑stack was offered, but Frigate (Python‑cored, multi‑process + SHM
  + native FFmpeg/ONNX) is direct existence‑proof that a Python core does this exact
  workload; the cited Rust‑beats‑Python benchmarks pit Rust against *pure‑Python
  pixel loops* nobody ships. A wholesale Rust/C++ rewrite forfeits the Python AI
  ecosystem (ultralytics, insightface, boxmot) and multiplies CI/build surface for
  a win the evidence attributes to *architecture, not language*. **Recommended:** a
  disciplined Python redesign now; **optionally** a thin **native (Rust/C via PyO3)
  ingest shim** later (NDI recv + UYVY→BGR + shm write in one GIL‑released pass) for
  deterministic tail latency where it measurably matters.
- **Inference accel**: standardize on ONNX + onnxruntime; ship a **per‑platform EP
  wheel** (TensorRT on NVIDIA, OpenVINO on Intel CPU/GPU/NPU, CoreML/ANE on Apple,
  CPU fallback) — they're mutually exclusive, so the installer picks one. On
  Windows target **Windows ML** (auto‑EP, GA late‑2025) over maintenance‑mode
  DirectML. Use the EP's **latency** hint (the loop is latency‑sensitive). For real
  H.264/HEVC (RTSP/USB), keep decode on‑GPU (NVDEC→CUDA / VideoToolbox→Metal);
  **NDI decode is CPU‑side inside the SDK** and can't be offloaded — the GPU win
  there is only the convert/resize/inference tail.

---

## 6. Phased migration plan (review before code; validate each on real cameras)

Ordered so the **highest‑ROI, lowest‑risk** wins ship first and each phase is
independently shippable + reversible. Phases 0–3 fix the *user‑visible* pains
without any architectural rewrite.

- **Phase 0 — Instrumentation (can't fix what you can't measure).** Per‑source
  `recv_get_performance` drop counter + queue depth in telemetry; true end‑to‑end
  latency probe (frame‑ts → command‑applied); a latency‑aware tracking eval harness
  (PVT++ "evaluate as run") extending the existing synthetic suite.
- **Phase 1 — Streaming quick wins (most of the drops).** UYVY/fastest color;
  collapse the copy chain (convert once into the shm dst); receive‑only capture
  thread with latest‑frame slot + drop‑oldest + blocking timeout; decouple
  reconnect/backoff from transient misses; jitter reconnects.
- **Phase 2 — Control quick wins (most of the bounce).** Flip the fixed‑rate pump
  on; feed the controller the **full measured latency**; move backend sends to a
  dedicated non‑blocking thread; dead‑man stop + heartbeat in all modes; express
  tracker velocity in /sec with real `dt`.
- **Phase 3 — Predictive control (the real cure).** Kalman/alpha‑beta predict‑ahead
  with measured dead time; CV Kalman predict on skip frames; startup direction/gain
  self‑calibration; confidence‑gated adaptive deadband.
- **Phase 4 — Concurrency convergence (this is where per‑process is resolved).**
  Single process by default: native capture threads + **one shared, batched
  inference** stage. **Retire** the model‑per‑child `AUTOPTZ_PROCESS_PER_CAMERA`;
  replace it with a `Worker` Protocol whose boundary is a *measured* thread/process
  choice, not a user mode. Keep process isolation only for fault‑isolation /
  profiled residual contention — always with the shared model‑server. Dynamic
  thread re‑budget; drop inert env caps.
- **Phase 5 — Decomposition + cleanup.** Split `CameraWorker` into staged
  components behind a Worker Protocol; two telemetry channels; delete the excess
  (dead ingest arch, Mark scaffolding out of the hot path, collapse the three
  model‑lifecycle methods, fix the unlocked `_workers` iteration, retire the
  redundant pose path / unused associator decision).
- **Phase 6 — Optional native shim + future bets.** Rust/C NDI ingest shim via
  PyO3 for deterministic tail latency; GStreamer/FFmpeg per‑OS HW decode for RTSP;
  re‑evaluate free‑threaded **3.14t** when OpenCV + onnxruntime certify.

---

## 7. Risks, trade‑offs, and explicit "don'ts"

- **Don't do a wholesale greenfield rewrite.** **[verified]** The bottleneck is
  architecture; a rewrite forfeits the Python AI ecosystem and multiplies risk for
  a marginal language win. Harden the existing thin‑Python‑over‑native core.
- **Don't replicate the model per camera process.** Use a shared model‑server
  (RAM + GPU‑occupancy), not the current per‑child model set.
- **Don't use NDI framesync in the tracking path** (adds latency, dup/stale frames).
- **Don't bet on free‑threaded Python yet** (OpenCV wheels missing; onnxruntime
  re‑enables the GIL on 3.13t). Target 3.14t as a guarded future bet.
- **Don't use a naive Smith predictor** for AutoPTZ's *jittery* latency — fold the
  prediction into the Kalman state (predict forward by measured age).
- **Don't rely on the integral term under dead time** — it winds up while waiting
  on stale feedback; keep `Ki≈0` with anti‑windup if used at all.
- **Reducing real latency still matters** — prediction compensates residual delay;
  it is not a substitute for a lower‑latency capture path / faster detector.
- **Validate on the user's real NDI cameras** — dead time and mount dynamics differ
  per camera/network and can't be confirmed from CI (consistent with project policy).

---

## 7.5 Measured benchmark findings (2026‑06‑28, Apple Silicon, 1080p30)

A two‑process NDI benchmark (separate sender fleet → the real `NDIAdapter` +
`Supervisor`, i.e. Mark's `source="ndi"` path) and a raw inference benchmark
**measured** the following on the user's Mac. These supersede the earlier
*assumptions* where they differ.

**1. The NDI receive path is not the bottleneck.** Capture‑only holds 30 fps with
~0 drops to 16×1080p30 at ~14% CPU.

**2. The bottleneck is the GIL, then the accelerator.**

| N | mode | App CPU | fps/cam | drops/s | e2e |
|---|------|---------|---------|---------|-----|
| 8 | threaded | 13% | 16.9 | 105 | 780 ms |
| 8 | per‑process | 48% | 26.7 | 27 | 42 ms |
| 16 | threaded | 18% | ~30* | 233 | 1746 ms |
| 16 | per‑process | 69% | 11.8 | 360 | 15 GB RAM, CPU pegged |

Per‑process gives big GIL relief at 8 cams but **collapses at 16** — model‑per‑child
doesn't scale (RAM + ANE thrash). **Don't make it default; gate it by cores/RAM.**

**3. ⭐ The Neural Engine is a fixed ~20 detections/sec shared resource.** Raw
yolo11s on CoreML/ANE: ~54 ms/frame at batch 1, and **batching barely helps** —
batch‑8 is only 1.19× the per‑frame rate (the ANE processes the batch ~serially).

| batch | ms/call | frames/s | speedup |
|-------|---------|----------|---------|
| 1 | 54 | 18.4 | 1.00× |
| 8 | 363 | 22.0 | 1.19× |
| 16 | 755 | 21.2 | 1.15× |

**Consequence — the §5.2 architecture is corrected:** the scalable design is **not
"batch for speed"** (the ANE won't batch) — it is a **single accelerator
*scheduler*** that owns the ANE and fairly budgets the fixed ~20 infer/sec across
(a) cameras and (b) model types (detector/face/pose), while capture + tracking +
control run in parallel on the GIL‑light threads. One model set (no RAM cliff), the
scarce accelerator at 100% with fair scheduling (no thrash cliff), GIL free for the
rest. It degrades *smoothly* as cameras are added (each gets a smaller detection
slice) instead of hitting a cliff.

**4. The hidden second ANE consumer.** `AUTOPTZ_ASYNC_APPEARANCE=0` **doubled**
throughput at 8 cams (16.9→30 fps, 105→0 drops/s) because face‑SCRFD runs a full
inference pass that contends for the same ANE budget as the detector. The scheduler
must budget face/pose against the detector; until then, the async appearance default
deserves a re‑think. *(Caveat: benchmark frames have no faces, but SCRFD scans the
whole frame regardless, so the cost is real; validate with people‑content.)*

**5. CoreML‑units / unified‑pose / NDI‑color flags made no throughput difference**
when GIL‑bound — confirming the GIL (not inference speed or color format) is the cap
until the architecture changes.

**Orthogonal lever (raise the ceiling, independent of architecture):** a
faster/quantized detector (`yolo11n`, INT8 — the code already has `quantize_dynamic`)
buys more detections/sec directly. This is also the honest answer to "rewrite in
another stack": **the ceiling is the accelerator, not the language** — a Rust core
would hit the same ANE ceiling. Language buys deterministic tail latency + lower RAM,
not detection throughput.

**Correction on the ceiling:** the *production* cached yolo11s does **~56–62
detections/sec** on this ANE (measured in the prototype below), not ~20 — the earlier
figure used a freshly‑exported dynamic model that was slower. 3× more budget.

### 7.6 Tracking-control conclusion: bbox is evidence, not the PTZ target truth

Primary tracking literature supports bounding boxes as a practical detection and
association primitive, not as an unconditional control command:

- [SORT](https://arxiv.org/abs/1602.00763) models boxes with motion prediction and
  assignment; it is explicitly a simple baseline, not a PTZ control policy.
- [ByteTrack](https://arxiv.org/abs/2110.06864) improves association by using
  lower-confidence detections carefully rather than discarding them, which reinforces
  the point that weak boxes need classification and association logic.
- [BoT-SORT](https://arxiv.org/abs/2206.14651) adds camera-motion compensation and
  stronger association because box position alone is not stable enough in moving
  camera scenes.
- The [1 Euro filter](https://www.researchgate.net/publication/254005010_1_Filter_A_Simple_Speed-based_Low-pass_Filter_for_Noisy_Input_in_Interactive_Systems)
  is the right kind of low-latency jitter suppression for aim signals, but it does
  not decide whether a measurement is trustworthy.

AutoPTZ's implication for 2.2 is:

- A bbox remains the universal fallback because it exists for every detector and
  every platform.
- The controller should not blindly follow the raw bbox center. Pose/fused torso aim
  can improve the target point when it is already available and owned by the same
  track, but pose must stay optional because it is another inference/service cost.
- Before PTZ sees a target, the worker must classify the evidence: lost/held boxes,
  degenerate boxes, too-small boxes, collapsed occlusion boxes, and one-frame
  low-overlap teleports are treated as no PTZ evidence.
- A low-overlap jump is accepted only after repeated consistent evidence, or when
  trusted pose/body evidence confirms it. This prevents a one-frame tracker teleport
  from causing the camera to over-correct, while still allowing real reacquire.
- Lost target behavior should hold/stop by default. Search and zoom-out remain Labs
  until they prove they do not create bobbing or surprise reframing.

### 7.7 ⭐ Validated: in‑process scheduler FAILS, multi‑process model‑server SCALES

Two architectures were built behind flags and **measured** end‑to‑end on real NDI +
the real detector:

**(a) In‑process inference scheduler (`AUTOPTZ_INFERENCE_SCHEDULER`) — NEGATIVE.**
One scheduler thread owns detection for all cameras. A/B (engagement confirmed): no
change at all — N=8 16.9→16.6 fps, N=16 29.8→29.8, drops/CPU/e2e identical. **It stays
single‑process, so the GIL still caps throughput** regardless of how the work is
reorganized. (Kept flag‑gated + tested; the scheduling logic is reusable in (b)/native.)

**(b) Multi‑process model‑server (N camera processes + 1 shared model‑server) —
SCALES.** Prototype measured at 1080p30:

| N=16 | Total CPU | Capture fps | Detect/s | **RAM** | Outcome |
|------|-----------|-------------|----------|---------|---------|
| threaded | 18% | 29.8 (decimated) | — | 5.4 GB | GIL‑bound, 232 drops/s |
| per‑process | 69% (pegged) | 11.8 | — | **15 GB** | collapsed |
| **mp model‑server** | **26%** | **30+ (all 16)** | 56/s shared | **6.3 GB** | **scales** |

One model set in the server → **no RAM cliff** (6.3 GB vs 15 GB); capture parallel
across processes (GIL escaped) → all 16 cameras alive at 30 fps; CPU efficient.
Detection is the fixed ANE resource, fairly shared (~3.5/s/camera at N=16 → smoothness
must come from **predictive tracking**, Phase 3). **Conclusion: the GIL is escaped by
*process* parallelism, and the RAM cliff by a *shared* model‑server. This is the
scalable architecture; a native rewrite is now an optional optimization (≈5 GB less
RAM + deterministic latency from removing the per‑process Python interpreter overhead),
not a requirement.**

---

## 8. Appendix — cross‑platform decode + inference matrix

| Platform | Real‑codec decode (RTSP/USB) | Inference EP | Notes |
|----------|------------------------------|--------------|-------|
| macOS (Apple Silicon) | VideoToolbox → IOSurface/Metal (zero‑copy) | CoreML / **ANE** (bundled in the ORT wheel) | ANE ~½ the power of GPU; NDI decode stays CPU‑side |
| Windows | D3D11VA / NVDEC / QSV via FFmpeg | **Windows ML** (auto: TensorRT/OpenVINO/QNN) > DirectML | DirectML in maintenance mode; WinML GA late‑2025 |
| Linux | VAAPI / NVDEC via FFmpeg/GStreamer | TensorRT (NVIDIA) / OpenVINO (Intel) / CPU | DeepStream is NVIDIA‑only → breaks cross‑platform |
| All | **GStreamer `appsink max-buffers=1 drop=true`** with per‑OS HW decoders behind one pipeline string | ONNX + onnxruntime, one API, per‑platform EP wheel | NDI: `recv_color_format_fastest`, per‑source thread; decode is CPU‑side in the SDK |

**Verification ledger (what the adversarial pass changed):** GIL‑as‑primary‑cause
→ **refuted** (secondary ceiling; consumer cost first). NDI latency destabilizes
the loop → **supported**, but the dead time is the *whole pipeline*, not the NDI
link (full NDI/HX3 ≈ 16 ms). Free‑threaded Python ready → **refuted for 3.13t**.
Commercial smoothness via on‑camera + predictive damping → **partial** (NDI
mis‑attributed; principle sound). Per‑stream‑process + zero‑copy SHM is "standard"
→ **partial** (decouple capture per‑stream but *centralize/batch inference*).
Predict + dead‑time‑compensate + critically‑damp → **supported**. Native core more
appropriate than Python → **partial/overstated** (Frigate is the counter‑proof).
