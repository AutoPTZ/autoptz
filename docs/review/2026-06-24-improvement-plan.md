# AutoPTZ — In-Depth Improvement Plan (Ranked, Rated, with Trade-offs)

> Version: **2.2.0-rc5**  ·  Date: **2026-06-24**
> Companion: [`2026-06-24-architecture-overview.md`](2026-06-24-architecture-overview.md) (state-of-the-system + the two-pass comparison).
> This document is the **action plan**. Every item carries pros/cons and Impact / Effort / Risk ratings, an overall recommendation score (★1–5), and the research findings it closes.

## How to read the ratings

- **Impact** = user-visible value + risk reduction. **Effort** = S (<1 day) / M (days) / L (1–2+ weeks). **Risk** = chance the change introduces regressions.
- **★ score** = overall "do this" priority = roughly (Impact × applicability) ÷ (Effort × Risk). ★★★★★ = do it now; ★★ = worthwhile when convenient.
- **Confidence** reflects how strongly the two research passes + source adjudication support the finding.

---

## Tier 0 — Correctness & Safety (do first; cheap, high risk-reduction)

### 0.1 — Guard the worker thread loops (try/except + finally) ★★★★★
**Problem.** `_run` (camera_worker.py:2457) guards only `source.read()`; `_open_resources()` and the rest of the body have no top-level guard, and there is no `finally`. An unhandled exception kills the thread and **skips `_close_resources()` (line 2580)** → leaked `cv2.VideoCapture` handle + leaked SHM segment; `_inference_loop` has the same gap (skips `_stop_appearance_thread`). *Adjudicated against source — confirmed.*
**Recommendation.** Wrap each loop body in `try/except Exception: log.exception(...) + emit ERROR telemetry`, and put `_close_resources()` / `_stop_appearance_thread()` in a `finally`. Add a per-iteration guard so one bad frame doesn't kill the thread.
| | |
|---|---|
| **Pros** | Converts silent thread-death + resource leak into either recovery or a visible error; tiny, localized change. |
| **Cons** | A persistently-throwing loop could spin logging — throttle the warning. |
| **Impact** High · **Effort** S · **Risk** Low · **Confidence** High (Pass B1, source-verified) |

### 0.2 — Verify the auto-updater download before executing it ★★★★★ (SECURITY)
**Problem.** `update/installer.py` downloads an installer and launches it (`subprocess.Popen`/`os.startfile`/`open`) with **no signature or checksum check**; the download uses bare `urllib.request.urlopen` with **no certifi SSL context** (unlike the checker, which was explicitly fixed for the frozen-build trust-store gap). With unsigned Windows/Linux artifacts there is no OS backstop either → RCE/MITM exposure.
**Recommendation.** Publish a per-asset SHA-256 (or a signed manifest) in each GitHub release; verify it before launch. Reuse the checker's `_ssl_context()` in `download_update`. Long-term: verify Authenticode/notarization of the downloaded binary.
| | |
|---|---|
| **Pros** | Closes the most serious security hole; the SSL fix is one line. |
| **Cons** | Requires the release pipeline to emit and publish hashes (small CI change). |
| **Impact** High · **Effort** S–M · **Risk** Low · **Confidence** High (Pass A5) |

### 0.3 — Make hot-path failures visible (DEBUG/`pass` → throttled WARNING + telemetry) ★★★★☆
**Problem.** The failure mode of this system is "silently does the wrong thing." The PTZ control tick swallows *all* errors at no log level (`controller.py:473`, confirmed); supervisor pump failures, `_push_frame` SHM failures, and detect/track failures log only at DEBUG (invisible at the default INFO). An operator sees a frozen box / dead camera with an empty log.
**Recommendation.** Promote the hot-path swallows to rate-limited `warning(..., exc_info=True)` and add a `last_error` to telemetry for detect/track/PTZ so the UI can show "tracking stopped: <reason>".
| | |
|---|---|
| **Pros** | Makes degradation diagnosable; no behavior change, just visibility. |
| **Cons** | Must rate-limit to avoid log spam on a persistent fault. |
| **Impact** Med–High · **Effort** S · **Risk** Low · **Confidence** High (Pass B1) |

### 0.4 — Unlink SHM on child `terminate()` in process mode ★★★☆☆
**Problem.** In process-per-camera mode, a child killed via `terminate()` never runs `_close_resources()`, and the parent never unlinks the pair → `/dev/shm` leak across restart cycles (`process_worker.py:349`).
**Recommendation.** After `proc.terminate()`, have the parent call `unlink_shared_memory_pair(shm_name)`.
| **Impact** Med · **Effort** S · **Risk** Low · **Confidence** High — *only relevant if process mode is pursued (see 3.4).* |

---

## Tier 1 — Latency reduction (high user-visible value)

### 1.1 — Move the PTZ send off the control thread ★★★★★  *(the #1 latency win)*
**Problem.** `ctrl.step()` is driven inline on the inference thread, so `backend.move_velocity()` — a blocking serial/IP/**ONVIF SOAP (5–50+ ms)** call — stalls the entire detect→track→PTZ loop and inflates the measured latency that feeds prediction. *Both passes found this; adjudicated: the controller's background `_loop` (controller.py:467) already exists to own the send and is simply never started.*
**Recommendation.** Start the controller's background loop (`rate_hz=20` already there) and feed it via the existing `ctrl.update()` instead of inline `ctrl.step()`. The send no longer blocks the control loop; worst-case ONVIF/IP stalls vanish from the critical path and stop corrupting the next tick's `dt`.
| | |
|---|---|
| **Pros** | Removes 5–50+ ms of blocking + its jitter from every tick; the machinery is already written and unused; decouples control cadence from backend I/O latency. |
| **Cons** | Must keep feeding `set_loop_latency`/state each tick; tests that assert on `step()` return values need updating; one more running thread per camera. |
| **Impact** High · **Effort** M · **Risk** Med · **Confidence** High (Pass A2 + B2, source-verified) |

### 1.2 — Tighten the inference wake timeout (50 ms → ~10 ms) ★★★★☆
**Problem.** `_frame_ready.wait(0.05)` caps wake latency at 50 ms; on low-fps sources or an unlucky clear/set race the inference thread idles up to 50 ms on a frame that's already available (camera_worker.py:2597/2610).
**Recommendation.** Drop the timeout to ~10 ms (commands still drain on fall-through, just 5× more often — negligible CPU).
| **Impact** Med (tail latency; helps low-fps most) · **Effort** S · **Risk** Low · **Confidence** High (Pass B2) |

### 1.3 — Fix the predictor's latency accounting + reduce stacked smoothing ★★★☆☆
**Problem.** `_latency_ms` = ingest + detect/track only — it **excludes the PTZ send and the sensor/decode age**, so the lead-time is systematically short (worst on ONVIF/RTSP). Separately, the aim path is low-passed three times (EMA `_smooth_aim` → controller one-euro → EMA velocity), stacking phase lag the lead must overcome.
**Recommendation.** Add the measured send time + an estimate of capture/decode age (the SHM header already carries `ts_ns`) into the latency fed to `set_loop_latency`. Let the controller's one-euro be the primary position filter; reduce or deadband the upstream `_smooth_aim` EMA.
| | |
|---|---|
| **Pros** | More accurate lead → less perceived lag and overshoot; uses data already measured. Lead is capped (`_LEAD_MAX_S`) so over-lead risk is bounded. |
| **Cons** | Smoothing changes need re-tuning against real footage to avoid reintroducing jitter. |
| **Impact** Med · **Effort** S–M · **Risk** Med · **Confidence** Med–High (Pass B2) |

---

## Tier 2 — Cross-platform performance & correctness

### 2.1 — Cap the remaining thread pools (OMP/MKL/OpenBLAS/torch + ORT inter-op) ★★★★★
**Problem.** ORT intra-op and OpenCV are capped, but OMP, MKL/OpenBLAS, torch, and ORT inter-op are not — each defaults to all cores, multiplied per camera. This is the likely residual cause of the documented CPU-spike variance, and it's worst on the OpenVINO path (its own TBB pool is uncapped) — exactly where OpenVINO is recommended.
**Recommendation.** In `_apply_hardware_env`, also publish `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `torch.set_num_threads`, and OpenVINO `inference_num_threads`/`NUM_STREAMS` to the same per-camera budget. **These must be set before the libraries import**, so set them before the first `import cv2/numpy/torch/onnxruntime` (already correct for the spawned process path).
| | |
|---|---|
| **Pros** | Closes the last oversubscription holes across **all 7 targets**; directly attacks the known CPU variance; cheap. |
| **Cons** | Env-before-import ordering must be enforced for the in-process threaded path. |
| **Impact** High · **Effort** S · **Risk** Low · **Confidence** High (Pass A3 + B3) |

### 2.2 — Auto-select OpenVINO for Intel + fix the openvino-on-mac install bug ★★★★☆
**Problem.** OpenVINO is the correct path for Intel CPU/iGPU/Arc (4 of 7 targets) but is only auto-chosen on Linux + a brittle name-match. Separately, `--accelerator openvino` is accepted on macOS and **hard-fails the install** (no macOS wheel).
**Recommendation.** Broaden `_auto_accelerator`: any Intel CPU (any OS) and any Intel GPU (Arc/Iris/UHD, match PCI vendor 8086, not just the string "intel") → OpenVINO. Gate `--accelerator openvino` to raise a clear error on Darwin.
| | |
|---|---|
| **Pros** | Real speedups for a large share of users; removes a hard install failure; OpenVINO beats the stock CPU EP on Intel and is the only real Intel-GPU path. |
| **Cons** | OpenVINO is Windows/Linux-only (no mac wheel), so Intel Macs realistically stay on an honest CPU path. Needs device-aware precision (FP16 only when a GPU/NPU device is actually present; FP32 on CPU). |
| **Impact** High (4 targets) · **Effort** M · **Risk** Low–Med · **Confidence** High (Pass A3 + B3) |

### 2.3 — Tell the truth about precision and Intel-Mac/Linux-AMD acceleration ★★★★☆
**Problem.** "fp16" is reported for EPs that run FP32 (CoreML-on-Intel, DirectML, OpenVINO-CPU); the docs claim CoreML routes to the AMD GPU/ANE on Intel Macs (it doesn't); Linux+AMD silently runs on CPU with no GPU path.
**Recommendation.** Gate the fp16 label on whether the EP genuinely accelerates on *this* host (e.g. CoreML only claims fp16 on arm64). Reword the Intel-Mac docs to "CPU is the realistic path; the GPU may be used for some ops." Emit an explicit "no GPU EP available — running CPU" notice on Linux+AMD. Surface the TensorRT cold-start build as a one-time status and verify the engine cache survives app updates.
| **Impact** Med–High (trust/correctness) · **Effort** S · **Risk** Low · **Confidence** High (Pass A3 + B3) |

### 2.4 — Route int8 through the shared pool; serialize DirectML `Run()` ★★★☆☆
**Problem.** `AUTOPTZ_PRECISION=int8` is a **silent no-op** through the shared inference pool (only the per-worker `stacks.py` path quantizes). And DirectML requires single-threaded `Run()`, which the shared-session concurrent-call model violates on Windows multi-cam (undefined behavior).
**Recommendation.** Move the int8 quantization step into the pool's model resolution and gate it to CPU/OpenVINO-CPU EPs (it deopts on GPU EPs). For DirectML, serialize `Run()` under a lock or give DML one session per camera thread.
| **Impact** Med · **Effort** M · **Risk** Med · **Confidence** High (Pass B3) |

---

## Tier 3 — Architecture, readability & simplicity (medium-term, larger)

> **Sequencing rule:** add integration tests for the worker loops **before** decomposing them. The most complex, concurrent code currently has no regression net.

### 3.1 — Decompose `camera_worker.py` around an immutable `FrameResult` ★★★★☆
**Problem.** 4,713 lines, ~13 responsibilities, ~120 methods, mypy-excluded; per-frame outputs live as ~30 mutable `self._last_*` fields read across threads.
**Recommendation.** Introduce an immutable `FrameResult{tracks, target_state, aim, faces, pose, timings}` produced once per frame by a thin orchestrator, and lift the responsibilities into collaborators both passes independently proposed: `worker/{geometry, governor, telemetry, target_tracker, reid_recovery, pose_aim, identity_flow, ptz_driver}`. Do the most cohesive, already-tested clusters first (governor, telemetry), then the target/reid/pose unit. Residual `CameraWorker` ≈ 300–800 lines of thread + tick orchestration.
| | |
|---|---|
| **Pros** | Each collaborator becomes unit-testable; kills most of the cross-thread shared-mutable-state surface; recovers static typing; halves onboarding cost. |
| **Cons** | Large; must be done in test-guarded slices; touches the hottest code. |
| **Impact** High (maintainability) · **Effort** L (in M slices) · **Risk** Med · **Confidence** High (Pass A1 + B1 + B4 consensus) |

### 3.2 — Replace the implicit target-lock "FSM" with a real one ★★★★☆
**Problem.** The target status is a bare string mutated by direct assignment from 8–9 sites across two threads; contradictory states (e.g. `locked` with `target_id is None`) are reachable with no invariant. Highest-fear area for a new contributor (~1–2 weeks to safely modify).
**Recommendation.** A `TargetState` enum + an explicit `transition(event) → state` table, owned by a single `TargetTracker` collaborator, serialized at one point. Fold the heuristic guards and the `TargetAssociator` into the one path (the associator is the clean, typed one — make it authoritative and delete the heuristic fork).
| **Impact** High (correctness + legibility) · **Effort** M · **Risk** Med · **Confidence** High (Pass B1 + B4) |

### 3.3 — Collapse the redundant optionality axes ★★★☆☆
**Problem.** Three parallel code paths are maintained without proportional value: **sync vs async appearance**, **threads vs process-per-camera**, **pooled vs per-worker model builders**.
**Recommendation.** (a) Pick **async appearance** and delete the sync path — this eliminates the cross-thread target-state lock + the `_appearance_guarded` decorator. (b) Make the **pool** the single production model path; keep per-worker builders only as a test seam. (c) See 3.4 for the process path.
| | |
|---|---|
| **Pros** | Removes the most fragile concurrency duplication; shrinks the surface a refactor must preserve. |
| **Cons** | Deleting a fallback path needs a deprecation check that no one depends on it. |
| **Impact** Med–High · **Effort** M · **Risk** Med · **Confidence** High (Pass B4) |

### 3.4 — Decide on process-per-camera: commit-and-validate, or shelve ★★★☆☆ *(decision needed)*
**Problem.** A full process-per-camera path exists but is opt-in, "EXPERIMENTAL," unvalidated on real multi-camera hardware, can't propagate unlabeled cross-camera face harvests, and is a maintenance tax on every worker method-surface change. The GIL makes it the *only* real >4-camera scaling story.
**Recommendation (decision):** if multi-camera (5+) on one machine is a target, **commit**: validate on a real rig, close the unlabeled-face-propagation gap, accept the per-camera RAM cost, and make it the supported N-camera path. If not, **shelve** it behind a clearly-experimental flag and stop carrying it half-live. Don't leave it in the current limbo.
| **Impact** High (if multi-cam matters) · **Effort** M (decision) / L (validate) · **Risk** Med · **Confidence** High (Pass A2 + B4) |

### 3.5 — Type the config/command seams + dedupe geometry ★★★☆☆
**Problem.** A typed `CameraConfig` is serialized to a dict and back (`json.loads(model_dump_json())`) to cross an in-process boundary, then re-validated; commands are re-stringified into `("kind", payload)` tuples that silently no-op on a typo. Bbox math is duplicated (`camera_worker._bbox_iou` ≡ `detect.BBox.iou`).
**Recommendation.** Type the command field as `CameraConfig` (pydantic is already picklable/msgpackable); dispatch one typed command union. Extract `engine/geometry.py` and delete the duplicate bbox helpers; make `BBox` the single home.
| **Impact** Med · **Effort** S–M · **Risk** Low · **Confidence** High (Pass A1 + B4) |

### 3.6 — Split the UI god-files ★★★☆☆
**Problem.** `engine_client.py` (2,191) mixes transport + ConfigStore CRUD + 3 list-models + PTZ presets + identity + model-download + layout; `properties_panel.py` (1,864) and `camera_tile.py` (1,860) carry 380-line form-builders and inline paint-time prediction math.
**Recommendation.** Split `EngineClient` into transport + `CameraRepository` + `PtzPresetController` + `IdentityController` + `ModelController`; sectionize the properties form; extract `MotionPredictor`/`FramingBoxEditor`/`PoseOverlayRenderer` helpers from the tile.
| **Impact** Med · **Effort** L · **Risk** Low–Med · **Confidence** High (Pass A1 + A5) |

---

## Tier 4 — Packaging, licensing & dependencies

### 4.1 — Make the default install torch-free ★★★★☆
**Problem.** `ultralytics` + `boxmot` in `base.txt` drag **~2–3 GB of torch** into every default install, although inference is 100% ONNX Runtime and the prebuilt-ONNX-download + IoU-fallback tracker already make the default fully functional without torch.
**Recommendation.** Move `ultralytics` → a `requirements/export.txt` extra and `boxmot` → a `requirements/tracking.txt` extra; ship/host prebuilt ONNX for every detector tier (the `AUTOPTZ_MODEL_URL` hook already exists). Default install = onnxruntime-only; the extras restore BoT-SORT/OSNet/export.
| | |
|---|---|
| **Pros** | Multi-GB → few-hundred-MB default; removes the largest AGPL + MSMT17 weights surface from the base; faster, cheaper installs. |
| **Cons** | The lean install loses BoT-SORT occlusion robustness + OSNet recovery unless the extra is added (both already gated behind graceful probes). |
| **Impact** High · **Effort** S–M · **Risk** Low · **Confidence** High (Pass A4 + B5) |

### 4.2 — Resolve the pretrained-weight license landmines (pre-commercialization) ★★★★☆ *(release-blocker for a paid SKU)*
**Problem.** AGPL on your code does not grant rights to third-party weights. Three+ landmines ship/auto-download by default: **InsightFace `buffalo_l`** (non-commercial — flagged), **OSNet `osnet_x0_25_msmt17.pt`** (MSMT17 non-commercial — **unflagged**), **Ultralytics YOLO11 weights** (AGPL), plus the **NDI runtime** attribution requirement.
**Recommendation.** Flag the OSNet weights now (as InsightFace already is). For a commercial SKU: license the InsightFace pack or add a clean-license face tier (dlib/MediaPipe); replace OSNet weights with a permissively-trained set or disable ReID; switch the default detector/pose to Apache-licensed weights (YOLOX/RT-DETR for detection, RTMPose for pose) or buy the Ultralytics Enterprise license; add the required "NDI®" attribution in About.
| **Impact** High (legal) · **Effort** Legal + M · **Risk** Low (technical) · **Confidence** High (Pass A4 + B5) |

### 4.3 — Retire stale PTZ deps; pin everything ★★★☆☆
**Problem.** `onvif-zeep` (last release **2018**) is on a core PTZ path; `visca-over-ip` is flagged inactive; `boxmot` is pinned `>=10.0.91` (9 majors behind 19.x); several deps float with `>=` and there's no lockfile.
**Recommendation.** Swap `onvif-zeep` → `onvif-zeep-async` (MIT, maintained) behind the existing `onvif_ptz.py` shim; vendor or fork `visca-over-ip`; pin `boxmot` to a tested version; add an exact-version, hash-locked constraints file.
| **Impact** Med · **Effort** M · **Risk** Low–Med · **Confidence** High (Pass A4 + B5) |

### 4.4 — Sign Windows + Linux artifacts; add a per-OS EP assertion ★★★☆☆
**Problem.** macOS is fully signed/notarized; **Windows + Linux artifacts are unsigned** (SmartScreen/Gatekeeper friction + no OS integrity backstop for the updater). The PyInstaller spec bundles whatever EP/torch is in the build venv with no guard.
**Recommendation.** Add Authenticode signing (Windows) and a GPG-signed AppImage (Linux); add a spec assertion that the venv's onnxruntime variant matches the target OS; copy the Linux CPU-torch pin to the macOS/Windows CPU builds.
| **Impact** Med · **Effort** M · **Risk** Low · **Confidence** High (Pass A5 + B3) |

---

## Tier 5 — UX & differentiating features

### 5.1 — Surface silent action failures to the user ★★★★☆
**Problem.** Click-to-track, the master Track toggle, and config edits catch exceptions to **DEBUG only** — invisible at the default capture level, so edits can silently fail to persist.
**Recommendation.** Route these failures to the status bar / a toast, not just the debug log.
| **Impact** Med–High · **Effort** S · **Risk** Low · **Confidence** High (Pass A5) |

### 5.2 — Fix the identity-gating seam + click-to-track discoverability ★★★☆☆
**Problem.** The People panel disclaims tracking ("assigned per camera") while the actual gate lives in per-camera Properties — users look in the wrong place. Click-to-track has no persistent affordance (hidden behind hover overlays).
**Recommendation.** Add a "Track on camera…" action directly on People-panel cards that drives the per-camera picker; add a persistent first-tile hint ("Click a person to track").
| **Impact** Med–High · **Effort** M · **Risk** Low · **Confidence** High (Pass A5) |

### 5.3 — Add "recall preset / zoom-out on target-lost" ★★★★☆ *(best feature value/effort)*
**Problem.** On lock loss the camera coasts/centers but doesn't return to a known-good shot. Competitors (obs-face-tracker) do.
**Recommendation.** On the existing lost/coast state, optionally recall a saved PTZ preset or zoom out — you already have presets + the lost state, so this is mostly wiring.
| **Impact** Med–High · **Effort** S–M · **Risk** Low · **Confidence** Med–High (Pass B5) |

### 5.4 — Evaluate RTMPose (Apache, fast CPU) to replace YOLO11-pose ★★★☆☆
**Problem.** YOLO11-pose carries the AGPL-weights issue and is heavier on weak hardware.
**Recommendation.** Benchmark RTMPose (Apache-2.0, ~90 FPS CPU via ONNX) or MediaPipe Pose — pose is already consumed as ONNX, so the integration surface is small. Solves the AGPL-weights + weak-hardware-latency goals together.
| **Impact** Med · **Effort** M · **Risk** Med · **Confidence** Med–High (Pass B5) |

### 5.5 — Multi-person auto-switching / "AI director" ★★★☆☆ *(strongest differentiator)*
**Problem.** Single-target lock only; competitors (OBSBOT AI Director, Logitech smart-switching) auto-compose across people.
**Recommendation.** Add a switching policy layer on top of the existing per-track identity + pose + framing — the foundation is already there; only the director policy is missing.
| **Impact** High (differentiation) · **Effort** M–L · **Risk** Med · **Confidence** Med (Pass B5) |

---

## Recommended sequencing (a pragmatic roadmap)

**Sprint 1 — Safety & truth (≈1 week, all S, mostly mechanical):**
0.1 thread guards · 0.2 updater verification + TLS · 0.3 visible errors · 2.1 thread-pool caps · 2.3 honest precision/acceleration labels · 1.2 wake timeout. *Net: closes the security hole, the silent-failure class, and the CPU-variance + false-claim issues — all low risk.*

**Sprint 2 — Latency & install (≈1–2 weeks):**
1.1 PTZ send off the control thread (the headline latency win) · 1.3 predictor latency + smoothing · 2.2 OpenVINO auto-select + mac gate · 4.1 torch-free default install. *Net: the tracking feels tighter and the install gets dramatically lighter.*

**Sprint 3 — Decisions & dep hygiene (≈1 week + legal):**
3.4 process-per-camera decision · 4.2 weights-license plan · 4.3 retire `onvif-zeep` + pins · 4.4 Windows/Linux signing · 2.4 int8/DirectML. *Net: removes the commercialization blockers and the stale-dep risk.*

**Sprint 4+ — The big refactor (test-guarded, incremental):**
3.1 decompose `camera_worker` (tests first) · 3.2 target FSM · 3.3 collapse optionality · 3.5 typed seams + geometry · 3.6 UI god-files. *Net: pays down the central maintainability debt without touching the parts that are already right.*

**Opportunistic / product-led:**
5.1 visible failures (pull into Sprint 1 if cheap) · 5.2 identity seam · 5.3 preset-recall-on-lost · 5.4 RTMPose · 5.5 AI director.

---

## What NOT to do (preserve — see Overview §7)

Do not "refactor" the pipeline-stage layer, the `InferencePool`, the typed `messages.py` transport, the thin supervisor, the latest-wins frame handoff, the latency-aware prediction wiring, the one-place thread-capping, the transparent-degradation UX, or the macOS signing/notarization. These are the parts that are genuinely right; the plan above is explicitly designed to build *around* them.

---

## Top 10, if you only do ten things

1. **0.1** Guard the worker thread loops (S, ★★★★★)
2. **0.2** Verify the updater download (S–M, ★★★★★, security)
3. **1.1** PTZ send off the control thread (M, ★★★★★, latency)
4. **2.1** Cap the remaining thread pools (S, ★★★★★)
5. **0.3** Make hot-path failures visible (S, ★★★★☆)
6. **4.1** Torch-free default install (S–M, ★★★★☆)
7. **2.2/2.3** Auto-OpenVINO for Intel + honest precision labels (M, ★★★★☆)
8. **3.1 + 3.2** Decompose `camera_worker` around `FrameResult` + a real target FSM (L, ★★★★☆)
9. **4.2** Resolve the weights-license landmines before commercializing (legal+M, ★★★★☆)
10. **5.3** Preset-recall-on-target-lost (S–M, ★★★★☆, best feature ROI)
