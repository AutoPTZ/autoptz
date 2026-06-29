# AutoPTZ — Master Improvement Plan & Spec

> **Historical document, 2026-06-29:** this rc5-era plan is no longer the source
> of truth for 2.2.0 reliability work. The body below is preserved for context,
> but tracking speed presets and normal Experimental UI access to process/model-
> server scaling are retired. Use `docs/engineering/retired-experiments.md` and
> `docs/release/2.2.0-reliability-gates.md` for the current release direction.

> **Historical note from the original plan:** it consolidated three deep reviews
> into one place. It should no longer be used as the canonical implementation
> plan for 2.2.0.
> Version reviewed: **2.2.0-rc5** · Last updated: **2026-06-25**

---

## 0. Read this first (the whole thing in 6 sentences)

AutoPTZ is **well-built** — its per-camera vision pipeline and its PTZ control math are already as good as or better than the commercial products (OBSBOT, Center Stage, obs-face-tracker). What stops it from *feeling* like a finished product is **not the math** — it's **three layers around the math**: (1) the way work is scheduled across threads, (2) a thin "director" layer above the controller, and (3) reliability, packaging, and licensing rough edges. The good news: most of the high-impact fixes are **small and low-risk**, and a few of them (especially one keystone change) fix several problems at once. The plan below is sequenced so the **cheap, visible quality wins and the safety fixes ship first**, the **structural keystone** lands next, and the **bigger features and refactors** follow. There is exactly **one "do this and three things get better" change** — moving PTZ onto a fixed-rate, off-thread loop — and it is the spine of the plan.

**If you do nothing else, do these five (all small, all high-value):**
1. Wrap the worker threads so a crash can't silently leak the camera (safety).
2. Verify the auto-updater download before running it (security).
3. Make the camera **stop** when a PTZ cable/connection drops (safety — today it can run away).
4. Ship a single **"Tracking Speed" preset** (Calm/Normal/Fast/Sport) over the six existing knobs.
5. Give the Center-Stage crop a **dead-zone** so small movements stop jittering the frame.

---

## 1. The roadmap on one screen

Priority: **P0** = do now · **P1** = next · **P2** = then · **P3** = later/roadmap. Effort: S (<1 day) / M (days) / L (1–2+ weeks).

| # | Initiative | Why it matters | P | Effort |
|---|---|---|---|---|
| **PHASE A — Safety & Truth (ship first; all small, low-risk)** |
| A1 | Guard the worker thread loops (try/except + finally) | A crash silently kills the thread and leaks the camera + shared memory | P0 | S |
| A2 | Verify the updater download (checksum + certifi TLS) | Today it downloads and *runs* an installer with no integrity check (RCE risk) | P0 | S |
| A3 | VISCA stop-on-loss + ONVIF dead-man's-switch | A PTZ cable drop mid-pan keeps the camera moving for up to 30 s | P0 | S–M |
| A4 | Make hot-path failures visible (DEBUG/`pass` → throttled WARN) | Today a dead camera shows an empty log; you can't diagnose it | P0 | S |
| A5 | Cap OMP/MKL/torch thread pools | The known CPU-spike variance; the existing caps miss these pools | P0 | S |
| A6 | Honest precision/acceleration labels | The UI claims "fp16/GPU" on hardware that's actually running CPU FP32 | P0 | S |
| **PHASE B — Perceived-quality quick wins (days, mostly pure-Python, low risk)** |
| B1 | Unified **"Tracking Speed" preset** | One dial instead of six expert sliders — the single biggest "finished product" signal | P1 | S |
| B2 | **Nonlinear band** at the dead-zone edge | Removes the one rough edge in the otherwise-smooth control | P1 | S |
| B3 | Center-Stage **dead-zone / hold band** | Stops the crop jittering with every detector wobble | P1 | S |
| B4 | Split zoom smoothing (size ≠ position) | Kills "zoom breathing" in the digital framer | P1 | S |
| B5 | Better upscaler (CUBIC/LANCZOS4) | The Center-Stage crop is soft because it upscales with LINEAR | P1 | S |
| B6 | Decouple virtual-cam output from the 20 fps preview cap | The virtual camera is throttled to ≤20 fps; real cameras output 30 | P1 | S |
| **PHASE C — The keystone + latency (the structural smoothness fix)** |
| C1 | **Fixed-rate, off-thread PTZ command pump** ⭐ | Fixes micro-stutter, lowers latency, AND enables A3's stop-on-loss — one change, three payoffs | P1 | M |
| C2 | Center-Stage **critically-damped spring + real dt** | Gentle, frame-rate-independent ease-in/out (the Apple feel) | P1 | M |
| C3 | Tighten frame-ready wake (50 ms → ~10 ms) + count true latency in the predictor | Cuts tail latency; makes the lead accurate | P2 | S |
| C4 | Per-camera speed calibration | "Same config" feels twitchy on fast cameras, laggy on slow ones | P2 | M |
| **PHASE D — Real-product features** |
| D1 | **Multi-person group framing** (union-bbox) | The marquee Center-Stage feature AutoPTZ lacks: zoom out to include everyone | P2 | M |
| D2 | Shot-size-aware headroom + optional lead-room | Makes framing read as *composed*, not just *centered* | P2 | S–M |
| D3 | Auto-select OpenVINO for Intel CPU/iGPU/Arc | Real speedups for ~4 of 7 hardware targets that get nothing today | P2 | M |
| D4 | Torch-free default install | The default pulls ~2–3 GB of torch that inference never uses | P2 | S–M |
| **PHASE E — Bundled "AutoPTZ Camera" virtual device** |
| E1 | `VCamBackend` abstraction + **retire pyvirtualcam (license)** | `pyvirtualcam` is GPL-2.0-only → conflicts with AGPL-3.0 | P2 | M |
| E2 | Linux: first-class v4l2loopback detect-and-instruct | Can't bundle a kernel module; guide the user instead | P2 | M |
| E3 | Windows DirectShow filter → (later) MF source | A named "AutoPTZ Camera" in Zoom/Teams without installing OBS | P3 | L |
| E4 | macOS CMIO Camera Extension | The clean macOS path; biggest signing prerequisite | P3 | L |
| **PHASE F — Architecture debt (larger, test-guarded; do incrementally)** |
| F1 | Decompose `camera_worker.py` (4,713 lines) around an immutable `FrameResult` | The central maintainability risk; ~13 responsibilities in one class | P2 | L |
| F2 | Make the target a real state machine (enum + transitions) | Today it's a string flag mutated from 9 sites across 2 threads | P2 | M |
| F3 | Collapse redundant optionality (sync/async appearance, etc.) | Three parallel code paths maintained without proportional value | P3 | M |
| F4 | Type the config/command seams + dedupe bbox geometry | A typo'd nested config key silently drops; bbox math is duplicated | P3 | S–M |
| F5 | Split the UI god-files (engine_client/properties/tile) | 1,800–2,200-line UI files | P3 | L |
| **PHASE G — Commercialization & hygiene (cross-cutting)** |
| G1 | Resolve pretrained-weight license landmines | insightface, OSNet/MSMT17, ultralytics weights block a paid SKU | P2 | M+legal |
| G2 | Swap abandoned `onvif-zeep` → `onvif-zeep-async`; pin deps | A 2018-abandoned dependency on a core PTZ path | P2 | M |
| G3 | Sign Windows + Linux artifacts | macOS is signed/notarized; Win/Linux are not | P3 | M |
| G4 | Decide: process-per-camera (commit & validate, or shelf it) | Half-built; the only real >4-camera scaling story | P3 | M |
| — | "Random underscore" report | **Not a code bug** (verified) — see §6 | — | — |

---

## 2. Why this plan — the one idea that organizes everything

Two independent deep-research passes plus firsthand code reading converged on the same conclusion:

> **AutoPTZ's per-tick control and signal-processing is already competitive-or-better.** It has one-euro filtering, PID + velocity feed-forward, latency-aware look-ahead, slew limiting, an oscillation guard, ego-motion compensation, error-proportional catch-up, and a Center-Stage-style safe zone. **Do not spend more effort on the filter math** — the returns are gone.

What's missing lives in **three layers around that math**:

1. **Scheduling / threading (structural).** The PTZ control loop and the Center-Stage framer both run at the *variable* inference/preview frame rate, and the PTZ command *send blocks the inference thread*. → micro-stutter under load, extra latency, and a safety gap when a connection drops. **One change fixes all of this: C1.**
2. **The "director" layer is thin.** No unified speed dial, no multi-person group framing, no shot-size composition. This is the difference between "centered" and "directed" — and most of it is cheap (B1, D1, D2).
3. **Reliability, packaging & licensing rough edges.** Updater security, VISCA runaway-on-disconnect, torch bloat, weights/`pyvirtualcam` license conflicts, unsigned Windows/Linux builds.

Everything in §1 ladders into one of those three. The sequencing front-loads the **cheap, visible, low-risk** wins (Phases A–B), then the **keystone** (C1), then features and debt.

### The keystone, stated once (C1)
The controller's background `_loop()` (`controller.py:467`, built at 20 Hz) **is never started** — the worker drives `step()` *inline, once per inference frame* (`camera_worker.py:1294`). Consequences and the single fix:

- Command rate = inference FPS (10–30 Hz, jittering with CPU) → **micro-stutter**. Real trackers run a fixed 50–60 Hz loop and interpolate between detections.
- `move_velocity()` — a blocking serial write or **ONVIF SOAP round-trip (5–50+ ms)** — runs on the inference hot path → **added latency + a slow camera stalls detection**.
- There's no clean place to enforce "no fresh command in N ms → send stop," which is exactly what A3 (stop-on-loss) needs.

**Fix:** start the existing `_loop()` (or a dedicated PTZ timer), feed it via the already-built `update()` (lock-safe), keep `set_loop_latency()` on the vision thread. The tick body is identical, so it's low-risk. **This is the spine of the plan** — it's why C1, the latency items, and the A3 safety fix are all really one effort.

---

## 3. The initiatives — what / why / how / rating

> Format per item: **Why** (the problem + evidence) · **What/How** (the change, with file:line) · **Rating** (Impact / Effort / Risk).

### Phase A — Safety & Truth

**A1 — Guard the worker thread loops.**
*Why:* `_run` (`camera_worker.py:2457`) guards only `source.read()`; `_open_resources()` and the rest of the body have no top-level guard and no `finally`, so an unhandled exception skips `_close_resources()` (line 2580) → leaked capture handle + leaked SHM segment. `_inference_loop` has the same gap.
*How:* wrap each loop body in `try/except Exception: log.exception(...) + emit ERROR telemetry`; put `_close_resources()` / `_stop_appearance_thread()` in a `finally`; add a per-iteration guard so one bad frame doesn't kill the thread.
*Rating:* Impact **H** · Effort **S** · Risk **L**.

**A2 — Verify the updater download (security).**
*Why:* `update/installer.py` downloads an installer and launches it with **no signature/checksum check**, and the download uses bare `urllib` with **no certifi context** (the checker was explicitly fixed for this). With unsigned Win/Linux builds there's no OS backstop → RCE/MITM exposure.
*How:* publish a per-asset SHA-256 (or signed manifest) in each release; verify before launch; reuse the checker's `_ssl_context()` in `download_update`.
*Rating:* Impact **H** · Effort **S–M** · Risk **L**.

**A3 — VISCA stop-on-loss + ONVIF dead-man's-switch (safety).**
*Why:* the VISCA backends *silently swallow* sends during reconnect backoff (`visca_usb.py:113`, `visca_ip.py:136`), and VISCA continuous-move runs **until an explicit stop** — so a cable/socket drop *mid-pan* keeps the camera panning for up to the **30 s** backoff cap. The inference-stall watchdog doesn't catch transport-only drops.
*How:* issue `stop()` on the backend `connected` True→False edge (both backends already expose `connected`); set the ONVIF `ContinuousMove` `Timeout` field (≈1–2× the command interval) so the camera self-stops; wrap ONVIF `move_velocity` in try/except + a socket timeout. **Best delivered through C1's pump** as a "no fresh command in N ms → stop" heartbeat.
*Rating:* Impact **H (safety)** · Effort **S–M** · Risk **L**.

**A4 — Make hot-path failures visible.**
*Why:* the failure mode today is "silently does the wrong thing." The PTZ control tick swallows *all* errors at no log level (`controller.py:473`); supervisor pump, `_push_frame`, and detect/track failures log only at DEBUG (invisible at the default INFO).
*How:* promote hot-path swallows to rate-limited `warning(..., exc_info=True)` + a `last_error` in telemetry so the UI can show "tracking stopped: <reason>."
*Rating:* Impact **M–H** · Effort **S** · Risk **L**.

**A5 — Cap the remaining thread pools.**
*Why:* ORT-intra-op and OpenCV are capped, but OMP / MKL / OpenBLAS / torch / ORT-inter-op are not — each defaults to all cores × per camera. This is the likely residual cause of the documented CPU-spike variance, worst on the OpenVINO path.
*How:* in `_apply_hardware_env`, also export `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `torch.set_num_threads`, OpenVINO `inference_num_threads` to the per-camera budget — **before** the libraries import.
*Rating:* Impact **H** · Effort **S** · Risk **L**.

**A6 — Honest precision/acceleration labels.**
*Why:* the UI reports "fp16" for EPs that actually run FP32 (CoreML-on-Intel-Mac, DirectML, OpenVINO-CPU), and the docs claim CoreML uses the AMD GPU/ANE on Intel Macs (it doesn't — it runs CPU FP32).
*How:* gate the fp16 label on whether the EP genuinely accelerates on *this* host; reword the Intel-Mac docs; emit "no GPU EP — running CPU" on Linux+AMD.
*Rating:* Impact **M–H (trust)** · Effort **S** · Risk **L**.

### Phase B — Perceived-quality quick wins

**B1 — Adaptive PTZ controller, no visible speed preset.**
*Why:* exposing Calm/Normal/Fast/Sport moved responsibility to the operator and did
not fix the real failure mode: the same camera needs different speed depending on
target error, target velocity, loop latency, zoom/framing, and whether target
evidence is trustworthy.
*How:* keep speed presets retired. The controller adapts internally from measured
error, ego-corrected target velocity, loop latency, bounded acceleration, and the
framing deadband. Lost, stale, collapsed, degenerate, or one-frame-teleported boxes
are treated as no PTZ evidence.
*Rating:* Impact **H** · Effort **S-M** · Risk **M** (movement feel must be tested on hardware).

**B2 — Nonlinear band at the dead-zone edge.**
*Why:* AutoPTZ's dead-band exit is a slope discontinuity (full proportional gain engages immediately past the band) — the one rough edge in otherwise-smooth motion. obs-face-tracker does exactly this fix.
*How:* in `_soft_deadband` (`controller.py:186`), ramp the post-band slope from 0→full over a configurable width.
*Rating:* Impact **M–H** · Effort **S** · Risk **L**.

**B3 — Center-Stage dead-zone / hold band.**
*Why:* the digital framer (`digital_framer.py`) recomputes the crop from the raw bbox every frame with no dead-zone, so every detector wobble shifts the frame. Apple holds still until you cross a threshold.
*How:* before easing, snap the target to the current crop center while the subject stays within a band (~3–5% of crop dim), with hysteresis. ~10 lines in `_step`/`desired_crop`.
*Rating:* Impact **H** · Effort **S** · Risk **L**. *(Biggest "looks like Center Stage" win.)*

**B4 — Split zoom smoothing.**
*Why:* the framer eases crop *size* with the same 0.18 EMA as position, so detector-height flicker causes visible "zoom breathing" (worst in `full_silhouette` mode).
*How:* give size its own much-slower time constant + a size dead-band; ease out faster than in.
*Rating:* Impact **M–H** · Effort **S** · Risk **L**.

**B5 — Better upscaler.**
*Why:* the crop is upscaled to 720p with `INTER_LINEAR` (`camera_worker.py:3294`) → soft. (Also serves the virtual-camera output, B6/E.)
*How:* `INTER_CUBIC` (or `LANCZOS4`) when upscaling, `INTER_AREA` when downscaling. Also fix the secondary resize in `vcam.py:57`.
*Rating:* Impact **M** · Effort **S** · Risk **L**.

**B6 — Decouple virtual-cam output from the 20 fps preview cap.**
*Why:* the vcam send sits inside the ≤20 fps preview gate (`camera_worker.py:72,3330`), while real cameras output a steady 30 fps. (Verified defect.)
*How:* drive the vcam on its own ~30 fps clock and repeat the last frame so consumers see continuous video; convert to NV12 once (what Zoom/Teams want).
*Rating:* Impact **M** · Effort **S** · Risk **L**.

### Phase C — The keystone + latency

**C1 — Fixed-rate, off-thread PTZ command pump. ⭐** See §2 for the full rationale. *Impact **H** · Effort **M** · Risk **M**.* (Validate on real cameras.)

**C2 — Critically-damped spring + real dt in the Center-Stage framer.**
*Why:* the framer is a first-order EMA with no `dt`, so it's frame-rate-dependent and "draggy then drifty," not the settled Apple ease.
*How:* track crop center + size with a ζ=1 critically-damped spring integrated with the real inter-frame `dt` (`time.monotonic()` already read at `:3278`); tune for ~0.6–1.0 s settle.
*Rating:* Impact **M–H** · Effort **M** · Risk **M**.

**C3 — Tighten wake + fix predictor latency.**
*Why:* the inference wake caps at 50 ms (`_frame_ready.wait(0.05)`), and `_latency_ms` excludes the PTZ send + sensor/decode age, so the lead is systematically short.
*How:* drop the wait to ~10 ms; add the measured send time + SHM `ts_ns` frame age into `set_loop_latency`.
*Rating:* Impact **M** · Effort **S** · Risk **L–M**.

**C4 — Per-camera speed calibration.**
*Why:* gains are absolute, so the same config is twitchy on a fast 300°/s camera and laggy on a slow 60°/s one; VISCA's 24/20-step quantization compounds it.
*How:* use the existing-but-unused `PTZCaps.{pan,tilt,zoom}_speed_max` (`base.py:23`) to scale output per backend; optional measured-°/s calibration.
*Rating:* Impact **M–H** · Effort **M** · Risk **M**.

### Phase D — Real-product features

**D1 — Multi-person group framing.** *Why:* single-target only; Center Stage/OBSBOT/Jabra all zoom out to include everyone. *How:* in `_current_digital_target`, when >1 trusted person is present, return the **union bbox** and let `desired_crop` widen (per-preset `max_frac` cap). Digital path first (no hardware). *Impact **H** · Effort **M** · Risk **M**.*

**D2 — Shot-size-aware headroom + lead-room.** *Why:* fixed `headroom=0.10`; framing is centered, not composed. *How:* tie headroom to the framing preset; offset the setpoint in the direction of travel using the ego-corrected velocity. *Impact **M–H** · Effort **S–M** · Risk **L**.*

**D3 — Auto-select OpenVINO for Intel.** *Why:* OpenVINO is the right path for Intel CPU/iGPU/Arc (4 of 7 targets) but is only auto-chosen on a narrow Linux case; `--accelerator openvino` even hard-fails the install on macOS. *How:* broaden `_auto_accelerator` (match Intel CPU any-OS + Arc/Iris/UHD by PCI vendor 8086); gate `openvino` off macOS; use device-aware precision. *Impact **H (4 targets)** · Effort **M** · Risk **L–M**.*

**D4 — Torch-free default install.** *Why:* `ultralytics` + `boxmot` in `base.txt` drag ~2–3 GB of torch, but inference is 100% ONNX Runtime and the prebuilt-ONNX + IoU-fallback already make the default work. *How:* move `ultralytics`→`requirements/export.txt`, `boxmot`→`requirements/tracking.txt`; ship prebuilt ONNX for all tiers. *Impact **H** · Effort **S–M** · Risk **L**.*

### Phase E — Bundled "AutoPTZ Camera" virtual device

> Goal: a named "AutoPTZ Camera" shows up in Zoom/Teams/Meet **without** the user installing OBS. Today `vcam.py` only targets a pre-existing OBS/v4l2loopback and no-ops otherwise. Recommended strategy: **hybrid** — own driver where it pays off, depend on the kernel module on Linux. **Tier-1 output-quality fixes (B5/B6) ship first and are pure Python.**

**E1 — `VCamBackend` abstraction + retire `pyvirtualcam` (license). ⚠️**
*Why:* **`pyvirtualcam` is GPL-2.0-ONLY and AutoPTZ is AGPL-3.0** — the FSF says those can't be combined in one work, and importing it into the app is exactly that. (OBS/v4l2loopback/akvcam are GPL-2.0-**or-later** → compatible; only `pyvirtualcam` is the problem.) This is a live conflict independent of the bundling work.
*How:* introduce `VCamBackend{advertise, send, close}` over `_push_frame`; keep `VirtualCamSink` as the Linux backend + "use the user's existing OBS" fallback only; the hybrid backends (E3/E4) retire it on macOS/Windows.
*Rating:* Impact **H (license)** · Effort **M** · Risk **L**.

**E2 — Linux v4l2loopback detect-and-instruct.** *Why:* a kernel module can't be bundled in an AppImage (DKMS + kernel headers). *How:* detect `/dev/video*` loopback; if absent, show the copy-paste `modprobe v4l2loopback exclusive_caps=1 card_label="AutoPTZ Camera"` + a Recheck button; existing pyvirtualcam path drives it. *Impact **M** · Effort **M** · Risk **L**.*

**E3 — Windows DirectShow filter (then optional MF).** *Why:* a self-contained Win10+11 named device; DirectShow is what OBS uses and is broadly visible. *How:* a COM `.ax`/DLL reading the engine's **shared-memory ring** (reusable on Win/Linux), `regsvr32`-registered in the existing admin Inno installer; ship **both 32- and 64-bit**. MF `MFCreateVirtualCamera` (Win11 22000+) later for first-party quality. *Impact **H (Windows UX)** · Effort **L** · Risk **M**.*

**E4 — macOS CMIO Camera Extension.** *Why:* Apple removed the old DAL path in macOS 14, so this is the only clean macOS route; you already sign+notarize, so the pipeline foundation exists. *How:* a Swift `.appex` (macOS 13+) fed via the **CMIO sink-stream + IOSurface `CVPixelBuffer`** (⚠️ **not** a shared-memory ring — the extension runs as `_cmiodalassistants`, so App-Group shm is a dead end); add the `com.apple.developer.system-extension.install` entitlement + provisioning profile (absent today). *Impact **H (macOS UX)** · Effort **L** · Risk **M–H (signing prerequisites)**.*

### Phase F — Architecture debt (test-guarded, incremental)

**F1 — Decompose `camera_worker.py`** (4,713 lines, ~13 responsibilities) around an immutable `FrameResult` + a thin orchestrator, into collaborators (`governor`, `telemetry`, `target_tracker`, `reid_recovery`, `pose_aim`, `identity_flow`, `ptz_driver`). **Add integration tests for the loops first.** *Impact **H** · Effort **L** · Risk **M**.*

**F2 — Real target FSM** (enum + transition table) replacing the string status mutated from 9 sites across two threads. *Impact **H** · Effort **M** · Risk **M**.*

**F3 — Collapse redundant optionality** (pick async appearance, make the pool the only model path). *Impact **M–H** · Effort **M** · Risk **M**.*

**F4 — Type the config/command seams** (kill the typed→dict→typed round-trip; one typed command dispatch) + extract `geometry.py` (dedupe bbox math). *Impact **M** · Effort **S–M** · Risk **L**.*

**F5 — Split the UI god-files** (`engine_client.py` 2,191; `properties_panel.py` 1,864; `camera_tile.py` 1,860). *Impact **M** · Effort **L** · Risk **L–M**.*

### Phase G — Commercialization & hygiene

**G1 — Resolve pretrained-weight license landmines.** insightface `buffalo_l` (non-commercial — flagged), **OSNet `osnet_x0_25_msmt17.pt` (MSMT17 non-commercial — unflagged)**, Ultralytics AGPL weights, plus the `pyvirtualcam` GPL-only (E1) and NDI attribution. *Why:* AGPL on your code doesn't grant rights to third-party weights → blocks a paid SKU. *How:* flag OSNet now; for commercial, license/replace per weight (YOLOX/RT-DETR/RTMPose Apache backbones; dlib/MediaPipe face tier). *Impact **H (legal)** · Effort **M + legal**.*

**G2 — Retire `onvif-zeep` (abandoned 2018) → `onvif-zeep-async`; pin deps + hash-lock.** *Impact **M** · Effort **M** · Risk **L–M**.*

**G3 — Sign Windows (Authenticode) + GPG-sign the AppImage** (macOS already done). *Impact **M** · Effort **M** · Risk **L**.*

**G4 — Decide on process-per-camera:** commit + validate (the only real >4-camera path) or shelve it behind a clearly-experimental flag — don't carry it half-live. *Impact **H if multi-cam matters** · Effort **M (decision)**.*

---

## 4. Spec detail for the first things we'd build (Phases A–C)

These are the items you'd start on; the rest are specified enough above to plan against.

### Spec — C1: Fixed-rate off-thread PTZ command pump *(the keystone)*
- **Current:** `camera_worker._drive_ptz_auto` calls `ctrl.step(...)` inline on the inference thread (`camera_worker.py:1294`); `controller._loop()` exists but is never started.
- **Target:** the controller runs its own ~30–50 Hz loop; the inference thread only *publishes* the latest tracking payload.
- **Change:**
  1. In the worker's PTZ init, call `ctrl.start()` (launches `_loop`); on teardown, `ctrl.stop()` (already sends `backend.stop()`).
  2. Replace inline `ctrl.step(error, velocity, subject_height, track_active)` calls with `ctrl.update(...)` (lock-safe, already implemented at `controller.py:387`).
  3. Keep `ctrl.set_loop_latency(...)` on the inference thread each tick.
  4. Add to `_loop`/`_tick`: a heartbeat — if no `update()` arrived in N ms (and state is TRACKING), command `stop()` (this is A3's transport-drop safety net, now trivial).
- **Acceptance:** PTZ command cadence is steady and independent of inference FPS (log/telemetry the send rate); a deliberately-slowed ONVIF camera no longer raises `_latency_ms` on the inference loop; yanking the PTZ connection mid-pan stops the camera within N ms.
- **Validation:** run on the real cameras (USB-VISCA, IP-VISCA, ONVIF, NDI); confirm no regression in tracking tests (update any test asserting on `step()` return values).
- **Risk/rollback:** medium — gate behind a flag (`AUTOPTZ_PTZ_PUMP=1`) during validation; the inline path remains until proven.

### Spec — B1: Adaptive PTZ controller with hidden expert tuning
- **Remove permanently:** `TrackingSpeed`, `SPEED_PROFILES`, user-facing speed
  segmented controls, and normal-workflow PTZ tuning choices.
- **Control source:** the PTZ loop computes speed from target error, target
  velocity, measured frame/loop age, bounded acceleration, and camera limits.
- **Evidence gate:** only fresh usable target evidence can set `track_active=True`.
  Lost, held, degenerate, too-small, collapsed, off-frame, or one-frame-teleported
  boxes hold/stop the controller until repeated evidence confirms the target.
- **Acceptance:** no speed choice in the normal UI; a moving subject speeds up and
  slows down automatically; a single bbox teleport never drives PTZ; a repeated
  consistent new box is accepted and reacquired.
- **Risk:** medium — requires real PTZ validation because camera motor dynamics
  differ across NDI, VISCA, ONVIF, and digital backends.

### Spec — B3: Center-Stage dead-zone / hold band
- **Current:** `DigitalFramer.frame_for` → `desired_crop(bbox,…)` recomputes the crop from the raw bbox every frame; `_step` eases with a fixed 0.18 EMA.
- **Change:** before computing the desired crop, compare the subject center to the current crop center; if within a `deadzone_frac` (default ~0.04 of crop dim), **reuse the held center** (don't move). Add hysteresis: once moving, keep following until the subject re-enters a tighter inner band. Apply the same idea to size (pairs with B4).
- **Acceptance:** a stationary subject with normal detector jitter produces a *still* frame (no sub-pixel drift); a real move still recomposes smoothly.
- **Risk:** low — self-contained in `digital_framer.py`; tunable; falls back to current behavior at `deadzone_frac=0`.

### Spec — A3: VISCA stop-on-loss + ONVIF dead-man's-switch
- **VISCA:** expose the existing `connected` flag to the worker; on a `True→False` transition (or when a send is swallowed in backoff), command a single best-effort `stop()`; treat `connected==False` as "motion state unknown → stop, don't continue."
- **ONVIF:** set `request.Timeout` on every `ContinuousMove` to ~1–2× the command interval (camera self-stops if the next command doesn't arrive); wrap `move_velocity` in try/except + a zeep socket timeout.
- **Best path:** implement the heartbeat in C1's pump (transport-agnostic) and let it cover all backends.
- **Acceptance:** physically disconnect a VISCA camera mid-pan → it stops within N ms (not 30 s). Validate per backend on real hardware.
- **Risk:** low.

---

## 5. Suggested milestones (how it sequences)

- **M1 "Safe & honest" (≈1 week):** A1–A6. *Net: the security hole, the silent-failure class, the runaway-camera safety bug, and the false hardware claims are all closed — all low-risk.*
- **M2 "Feels finished" (≈1 week):** B1–B6. *Net: one speed dial, no dead-band lurch, a still Center-Stage frame, sharper output, 30 fps virtual cam — visible quality with near-zero algorithmic risk.*
- **M3 "Smooth & solid" (≈1–2 weeks, validate on real cameras):** C1 keystone · C2 spring · C3 latency · C4 calibration. *Net: steady low-latency motion and consistent feel across cameras.*
- **M4 "Director features" (≈2 weeks):** D1 group framing · D2 composition · D3 OpenVINO · D4 torch-free install.
- **M5 "Bundled camera + commercialization":** E1 (retire pyvirtualcam) + E2 Linux UX first; E3/E4 native devices as a larger track; G1–G3 in parallel.
- **Ongoing:** F1–F5 architecture debt, done in test-guarded slices between feature work; G4 decision when multi-camera scale becomes a real requirement.

---

## 6. The "random underscore" — resolved: not a code bug

Investigated to ground truth: an agent pulled **all 18 real GitHub release bodies as raw bytes** and ran each through the **actual** Qt `setMarkdown()` (the call the update dialog makes). Every body renders clean — **zero stray underscores**. The original lead was a **WebFetch hallucination** (it reported `VISCA_USB`/`error_proportional`; the real rc5 body has 0 underscores / 7 hyphens). The cancel button, version string, and all labels are clean too.

**Most likely what was seen:** a literal intraword token (`detect_batch`, `x86_64`, `Format_BGR888`) in an auto-generated release-notes bullet — *correct* output, not a defect.
**Do:** (preferred) backtick/rephrase such identifiers in PR titles; (optional defensive) `model_manager.py:126` → `str(row.get("name") or self.key.replace("_"," ").title())`.
**Don't:** switch `setMarkdown`→`setPlainText` (kills bold/links to fix a non-bug). If you saw it on a specific screen, a screenshot pins it instantly — every code path swept is clean.

---

## 7. What to preserve (do NOT "improve" these)

The pipeline-stage layer, the shared `InferencePool`, the typed `messages.py`/SHM transport, the thin supervisor, the latest-wins frame handoff, the latency-aware prediction wiring, the one-place thread-capping, the transparent `configured→effective(reason)` UX, the macOS signing/notarization, and the per-tick control math (one-euro + PID + FF + slew + osc-guard + ego-comp + catch-up). These are genuinely right — the whole plan is designed to build *around* them.

---

## 8. Decisions that change the plan (your call)

1. **Commercial (paid) SKU?** If yes, G1 (weights licenses) + E1 (`pyvirtualcam`) become release blockers and move up.
2. **How many cameras on one machine?** If 5+, G4 (process-per-camera) and A5 (thread caps) matter much more.
3. **Which OS matters most for the bundled camera?** macOS (E4) is the cleanest tech but the biggest signing lift; Windows (E3) reaches the most users. The Tier-1 quality fixes (B5/B6) help all of them regardless.

---

## Appendix — detailed evidence (deep-dive docs)

- [`review/2026-06-24-architecture-overview.md`](review/2026-06-24-architecture-overview.md) — full two-pass architecture review, per-dimension grades, hardware-target matrix.
- [`review/2026-06-24-improvement-plan.md`](review/2026-06-24-improvement-plan.md) — the architecture/perf/packaging plan with per-item pros/cons.
- [`review/2026-06-25-ptz-framing-realproduct-plan.md`](review/2026-06-25-ptz-framing-realproduct-plan.md) — PTZ control/transport, auto-zoom, Center-Stage parity, competitive research, and the full bundled-virtual-camera analysis.

Every claim in this plan was verified against source (and, for hardware/OS/licensing claims, against primary web sources) by an adversarial second pass.
