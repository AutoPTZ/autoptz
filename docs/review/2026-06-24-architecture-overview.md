# AutoPTZ — In-Depth Architecture & Quality Overview

> Version reviewed: **2.2.0-rc5**  ·  Date: **2026-06-24**
> Method: two independent deep-research passes (10 specialist agents total), cross-compared, with three contested findings adjudicated directly against source.
> Companion document: [`2026-06-24-improvement-plan.md`](2026-06-24-improvement-plan.md) (the ranked, rated action plan).

---

## 1. One-paragraph verdict

AutoPTZ is **engineered well beyond what its size and solo/AI-assisted origin would suggest**. The load-bearing skeleton — a pure, typed, gracefully-degrading pipeline-stage layer; a shared cross-camera inference pool; a typed message/shared-memory transport; and a thin command-routing supervisor — is genuinely good and should be preserved as-is. The product is also unusually *self-aware*: it measures its own loop latency and feeds it to the motion predictor, caps thread pools to fight oversubscription, and surfaces `configured → effective(reason)` state instead of silently overriding the user. The weaknesses are concentrated and addressable: one **4,713-line god-class** (`camera_worker.py`) that holds ~13 responsibilities and three redundant "optionality" axes; a set of **cross-platform acceleration claims that don't match reality** on Intel Macs and Intel/AMD GPUs; a **blocking PTZ send on the control thread** that is the single biggest reducible latency source; **missing top-level exception guards** on the worker threads that turn any unhandled error into a silent thread death plus resource leak; an **unverified auto-updater** that executes a downloaded installer without integrity checks; and **pretrained-weight license landmines** plus **torch in the default install** that matter the moment this is commercialized. None of these are architectural dead-ends — they are a well-scoped backlog on top of a sound design.

**Overall grade: B / B−** — a solid, maintainable, real-time system with a clear, high-leverage improvement path.

---

## 2. What AutoPTZ is (ground truth, verified in code)

A cross-platform desktop app (native **PySide6 / Qt Widgets**, OBS-style) that runs a real-time per-camera vision pipeline — **capture → detect → track → re-identify → pose → aim → drive PTZ** — and sends smooth pan/tilt/zoom so a camera keeps a chosen person framed. It supports multi-camera, identity-gated tracking (click or face-recognize a specific person), four PTZ transports (VISCA-USB, VISCA-IP, ONVIF, NDI) plus a digital "Center Stage" virtual-camera output, and cross-platform GPU acceleration via ONNX Runtime execution providers.

### Real runtime topology (corrects the docs)

The architecture doc says "two threads per camera." **The actual default is three** — the appearance/ReID+face thread has been default-on since async-appearance was added:

```
ONE OS process (default)
├─ GUI thread (Qt) ............ widgets, SHM preview readers, EngineClient bridge, supervisor pump
└─ per camera: CameraWorker
   ├─ capture thread  (_run) ............... source.read → SHM preview push → hand newest frame to inference
   ├─ inference thread (_inference_loop) ... detect → track → target-lock → pose → ego-motion → aim → ctrl.step → PTZ
   └─ appearance thread (_appearance_loop) . OSNet ReID recovery (~4 Hz) + InsightFace identify (~4 Hz)
Shared singletons: InferencePool (detector/face/pose sessions) · IdentityService (gallery)

A process-per-camera path (process_worker.py) exists but is opt-in, EXPERIMENTAL, and unvalidated on real multi-camera hardware.
```

Frame handoff between stages is a **single-slot, latest-wins mailbox** (lock + Event + monotonic frame-id) applied consistently three times (capture→inference, inference→appearance, writer→UI). This is the right real-time pattern: a slow stage never stalls a fast one, and stale frames are dropped rather than queued. **This is one of the best decisions in the codebase.**

### Package map (what lives where)

| Layer | Path | Health |
|---|---|---|
| Pipeline stages (pure, typed) | `engine/pipeline/{detect,track,reid,pose,identify,framing,egomotion,associator,ingest,avf_capture,pool}.py` | **Excellent** — small, single-responsibility, well-tested |
| Runtime infra | `engine/runtime/{inference,models,messages,shm,diagnostics,flags,bench}.py` | **Strong** — centralized EP selection, typed messages, lock-free SHM |
| PTZ backends | `engine/ptz/{controller,visca_usb,visca_ip,onvif_ptz,ndi_ptz,digital,factory}.py` | Good — clean factory + degradation |
| Orchestration | `engine/camera_worker.py` (**4,713 lines**), `supervisor.py` (1,079), `process_worker.py` (460), `worker/{frame_source,stacks}.py` | **Mixed** — supervisor is a clean router; camera_worker is the god-class |
| UI | `ui/engine_client.py` (**2,191**), `widgets/{properties_panel(1,864),camera_tile(1,860),main_window(1,367),...}` | **Mixed** — transparent UX, but several god-files |
| Config / update | `config/{models,store}.py`, `update/{checker,installer}.py` | Good — frozen pydantic + SQLite; updater has a security gap |

---

## 3. Cross-comparison of the two research passes

The value of running two independent passes is in **where they agreed (high confidence)**, **where only one looked (coverage)**, and **where they disagreed (adjudicated)**.

### 3.1 Strong consensus — both passes, multiple agents (treat as high-confidence)

1. **`camera_worker.py` is THE central maintainability problem** — 4,713 lines, ~13 responsibilities, ~120+ methods, mypy-excluded. Both passes independently produced near-identical decomposition proposals (target-lock, reid-recovery, pose-aim, identity-flow, ptz-driver, governor, telemetry).
2. **The pipeline/pool/messages/supervisor skeleton is genuinely good** and must be preserved.
3. **Blocking PTZ `move_velocity` on the inference thread is the #1 latency landmine**, and the fix (start the controller's already-built background loop) is sitting unused. *Adjudicated against source — confirmed.*
4. **Thread oversubscription is only half-solved** — ORT intra-op + OpenCV are capped, but OMP / MKL / OpenBLAS / torch / ORT-inter-op are not. This is the likely residual cause of the known CPU-spike variance.
5. **CoreML on Intel Macs is a false promise** — the code/docs claim GPU/Metal/ANE + FP16, but ORT CoreML on Intel Macs effectively runs **CPU FP32** (often slower than the plain CPU EP). Verified against ONNX Runtime issue tracker by both passes.
6. **OpenVINO is under-selected** — never auto-chosen for Intel CPU/iGPU/Arc except a narrow Linux + name-match-"intel" case, despite being the correct path for ~4 of the 7 hardware targets.
7. **Precision is mis-reported** — "fp16" is shown for EPs that actually run FP32 (CoreML-on-Intel, DirectML, OpenVINO-CPU), so the Camera-Info/About panels state a precision the hardware never runs.
8. **torch ships in the default install** (via `ultralytics` + `boxmot` in `base.txt`) even though inference is 100% ONNX Runtime — a multi-GB cost the common user never benefits from.
9. **Pretrained-weight license landmines** — InsightFace `buffalo_l` (flagged in-repo), **OSNet `osnet_x0_25_msmt17.pt` (MSMT17 = non-commercial, UNflagged)**, and Ultralytics AGPL weights. The OSNet one was independently surfaced by both passes as the overlooked "second landmine."
10. **`onvif-zeep` is abandoned (last release 2018)** → swap to the maintained `onvif-zeep-async`.
11. **The onnxruntime/onnx pinning is fragile-looking but actually correct and durable** — both passes verified the IR-13 coupling and that the Intel-mac `1.23.2` split is justified (ORT dropped macOS x86_64 after 1.23).
12. **The target-lock "FSM" is an implicit tangle** — a bare string status mutated from 8–9 sites across two threads, not a real state machine.
13. **Transparent graceful degradation is genuinely implemented** end-to-end (every ML dep optional at import; `configured → effective(reason)` surfaced).
14. **The GIL is the scalability ceiling** (~4–8 cameras) for the in-process design; process-per-camera is the escape hatch but multiplies RAM and regresses cross-camera identity harvesting.

### 3.2 Single-pass coverage — found by only one pass (the reason to run two)

**Pass B (adversarial / verify-in-code lens) uniquely surfaced:**
- **Missing top-level exception guards** on `_run` / `_inference_loop` → silent thread death + leaked cv2 capture and SHM segment. *Adjudicated — confirmed real.* (Severity high; partly mitigated by supervisor respawn, but the SHM leak is real.)
- **int8 is a silent no-op through the shared inference pool** — only the per-worker `stacks.py` path quantizes; the pooled default never calls it.
- **DirectML concurrent `Run()` violates its "one Run at a time" constraint** with a shared session across camera threads (undefined behavior on Windows multi-cam).
- **`--accelerator openvino` is accepted on macOS and hard-fails the install** (no macOS wheel for `onnxruntime-openvino`).
- **The predictor under-leads** because `_latency_ms` excludes the PTZ send and the sensor/decode age; plus **stacked low-pass filtering** (EMA → one-euro → EMA velocity) adds phase lag.
- **Silent error-swallowing on hot paths** — the PTZ control tick does `except Exception: pass` at *no* log level (controller.py:473, confirmed); supervisor pump failures log only at DEBUG.

**Pass A (constructive / breadth lens) uniquely surfaced:**
- **SECURITY (high): the auto-updater downloads and executes an installer with no signature/checksum verification**, and the download path skips the certifi TLS context the checker was explicitly fixed to use → RCE/MITM exposure, worsened by unsigned Windows/Linux artifacts.
- **`cert.p12` in the repo root is correctly gitignored and untracked** — checked and clean (a non-finding worth recording so it isn't re-raised).
- **UX seams** — click-to-track has no persistent affordance (hidden behind hover overlays); the People panel disclaims tracking while the actual gate lives per-camera in Properties (users look in the wrong place).
- **PyAV bundles GPL FFmpeg** (fine under AGPL, but blocks future relicensing).

### 3.3 The one genuine disagreement — adjudicated

| Claim | Pass A2 | Pass B1 | Verdict (read the source) |
|---|---|---|---|
| SHM torn-read safety | "textbook-correct lock-free SPSC" | "defeated by slot rotation; 8-byte seq reads can tear" | **A2 is right.** `seq` is the first 8 bytes of each slot; `slot_size` (24 B header + 2,764,800 B frame) is 8-byte aligned, so seq reads are atomic on any 64-bit target. With 3 slots the writer must emit 3 frames to lap a slot mid-copy, which `seq_pre != seq_post` catches (shm.py:223-235). B1's "may return a 1-frame-stale frame" is the intended latest-wins semantics, not a tear. **Action: a one-line doc caveat, not a fix.** |

This is the two-pass method working as designed: an adversarial flag, resolved by direct inspection, with a small residual note rather than a wasted fix.

---

## 4. Dimension-by-dimension assessment (with ratings)

Ratings: **A** = excellent / preserve · **B** = good, minor gaps · **C** = adequate, real gaps · **D** = weak.

| Dimension | Rating | Summary |
|---|---|---|
| **Core architecture skeleton** (stages, pool, messages, supervisor) | **A−** | Pure typed stages, shared pool, latest-wins handoff, thin router. The thing to build *around*, not against. |
| **Code organization / readability** | **C** | One 4,713-line god-class + three 1,800–2,200-line UI files. Pipeline layer is exemplary; orchestration and UI are not. |
| **Simplicity** | **C+** | Three redundant optionality axes (sync/async appearance, threads/process-per-camera, pool/per-worker builders) and typed→dict→typed config round-trips inflate the surface. |
| **Concurrency correctness** | **B−** | Lock discipline is clean (no deadlock, disjoint locks, latest-wins handoff). But missing thread-loop guards and pervasive DEBUG/`pass` swallowing make the failure mode "silently does the wrong thing." |
| **Real-time latency design** | **B+** | Excellent: decoupled threads, deadline-accurate pacing, prediction tied to *measured* latency, stage phasing to flatten jitter. Held back by the inline blocking PTZ send and a systematic under-lead. |
| **Performance / HW acceleration** | **B** | Centralized EP selection, persistent TRT/CoreML caches, thread caps from one place — strong. Undercut by Intel-Mac false promise, OpenVINO under-selection, fp16 mis-report, incomplete thread caps, int8 pool-bypass. |
| **Cross-platform correctness** | **B−** | NVIDIA + Apple Silicon paths are right and optimal. Intel Mac (iGPU/AMD) and Linux+AMD silently fall to CPU; openvino-on-mac install bug; DML concurrent-Run UB. |
| **Subsystem / package fit** | **B+** | ONNX-Runtime spine is the best call in the project; most picks are right. `onvif-zeep` stale; `boxmot` pin 9 majors behind; `visca-over-ip` inactive. |
| **Licensing posture (commercial)** | **C** | Three weights landmines + torch-in-default. Fine for AGPL open use; a release-blocker set for a paid SKU. |
| **End-user UX** | **B** | Transparent degradation + status visibility are model-grade. Discoverability of the two core gestures and the identity-gating seam are the friction. |
| **Packaging / distribution** | **B** | macOS is production-grade (signed, notarized, stapled). Windows/Linux unsigned; torch bloat; no per-OS EP assertion in the spec. |
| **Security** | **C+** | One high finding (unverified update exec + non-certifi download). Secrets hygiene is otherwise clean (cert.p12 gitignored). |
| **Testing vs risk** | **C+** | Pipeline stages thoroughly unit-tested; the god-files and *all* concurrency paths (3-thread contention, SHM under a concurrent writer, clean-shutdown join) are untested — coverage is inverted vs risk. |

---

## 5. Performance across the seven hardware targets (synthesized)

| # | Target | EP actually selected | Real behavior | Verdict |
|---|---|---|---|---|
| 1 | Intel Mac + Intel iGPU | CoreML→CPU, reports "fp16" | No ANE on Intel; CoreML partitions back to **CPU FP32**, often slower than plain CPU EP | **Weak / mislabeled.** Should auto-use OpenVINO (but no mac wheel → realistically honest CPU). |
| 2 | Intel Mac + AMD GPU (Radeon Pro) | CoreML→CPU, reports "fp16" | AMD GPU **not reliably used**; CoreML→Metal is op-dependent and skews to CPU for conv-heavy YOLO | **Weak / over-claimed.** Drop the "Metal routing to AMD" claim; keep the `AUTOPTZ_COREML_UNITS` escape hatch. |
| 3 | Windows CPU-only | DirectML wheel → CPU EP, FP32 | DML with no DX12 GPU falls to CPU; OpenVINO would beat the stock CPU EP on Intel | **Partial.** Recommend/auto OpenVINO for CPU-only Intel Windows. |
| 4 | Windows + AMD/Intel GPU | DirectML, reports "fp16" | Correct EP and uses the GPU — but FP32 model (DML doesn't auto-convert); concurrent `Run()` is UB | **Good EP, wrong precision label, latent concurrency bug.** |
| 4L | **Linux + AMD GPU** | base CPU wheel (no auto match) | **No packaged GPU path** (ROCm EP not on PyPI, DML is Windows-only) → silent CPU | **Silent gap.** Emit an explicit "no GPU EP — running CPU" notice. |
| 5 | Windows/Linux + NVIDIA | TensorRT→CUDA→CPU, FP16 + engine cache | Optimal. Caveat: multi-minute cold-start build; cache invalidated by an ORT/TRT version bump (i.e. an app update) | **Best path.** Surface the cold-start; verify cache survives updates. |
| 6 | Intel GPU (Arc/iGPU) | Win: DirectML; Linux: OpenVINO only if name matches "intel" | OpenVINO (FP16) usually beats DirectML (FP32) on Arc; name-match is brittle (Arc shows as "DG2"/"Alchemist") | **Partial.** Prefer OpenVINO for Intel GPUs on both OSes; broaden detection. |
| 7 | Apple Silicon | CoreML MLProgram + ComputeUnits=ALL, FP16 | Strong. For YOLO11 conv graphs CoreML often settles on **GPU (Metal)**, not the ANE; "uses the ANE" is optimistic | **Good.** Soften the ANE claim; bench-confirm dispatch. |

**Net:** the two strong targets (NVIDIA, Apple Silicon) are genuinely well-served. The Intel-everything and Linux+AMD targets are where performance is left on the table *and* the UI tells the user something untrue about it.

---

## 6. End-to-end latency profile (synthesized, default config)

- **Critical path (what AutoPTZ controls):** inference wake (~1 ms, ≤50 ms worst) → detect (8–40 ms on detect frames) → track (2–15 ms) → target-lock (<1) → pose (amortized off detect frames) → ego-motion (3–8 ms, 1/3 frames) → aim+smooth+controller (<1) → **PTZ send (serial 2–5 / IP 1–10 / ONVIF 5–50+ ms, blocking, inline)**.
- **Typical AutoPTZ-induced latency:** ~28 ms on a detect+send tick. **End-to-end photons→wire ≈ 45–90 ms** typical including sensor/decode; **140+ ms** worst case (50 ms wake + 40 ms detect + 50 ms ONVIF).
- **How it's hidden:** the system measures `_latency_ms` and feeds it to the predictor's lead-time — sophisticated, and rare. **But** `_latency_ms` excludes the PTZ send and sensor/decode age, so the lead is *systematically a bit short*, worst on ONVIF/RTSP.
- **Biggest reducible sources:** (1) move the blocking PTZ send off the control thread (the controller's background loop already exists), (2) drop the 50 ms `_frame_ready` wake timeout, (3) feed true latency into the predictor and reduce the stacked smoothing.

---

## 7. What is genuinely excellent (preserve — do not "improve" these)

1. **The pipeline-stage layer** — pure, typed dataclass I/O, graceful degradation, model-owning-or-pool-injected. Rebuild the orchestrator *around* this.
2. **The shared `InferencePool`** — one detector/face/pose set for all cameras, per-camera tracker only; lock-free detector, fine-grained locks for the non-reentrant face/pose wrappers.
3. **Typed `messages.py` + msgpack/SHM transport** — the reason the process boundary is cheap.
4. **The thin supervisor** — clean `_route`→`_on_*` dispatch, heavy `worker.start()` outside the lock, single-writer registry. The pattern `camera_worker` should adopt.
5. **Latest-wins frame handoff** (×3) — decouples capture FPS from inference cost; no backlog growth under load.
6. **Latency-aware prediction** — measuring loop latency and feeding it to the lead-time is something most commercial products hardcode.
7. **Thread-cap-from-one-place** with the macOS GCD-backend nuance handled — directly attacks the documented CPU-spike root cause.
8. **Transparent degradation + `configured → effective(reason)`** — model-grade honesty in the UX.
9. **macOS packaging** — signed, notarized, stapled, correct entitlements, CI-enforced.
10. **The `--bench` / "measure on your hardware" philosophy** — the right antidote to EP-performance optimism, and honestly documented.

---

## 8. The headline backlog (full detail in the plan)

1. **Safety:** wrap the worker thread loops in `try/except + finally`; verify the updater download; promote silent hot-path errors to throttled warnings.
2. **Latency:** move the PTZ send off the control thread; tighten the wake timeout; fix the predictor's latency accounting.
3. **Cross-platform:** cap the remaining thread pools; auto-select OpenVINO for Intel; fix the precision labels and the Intel-Mac/Linux-AMD honesty; fix the openvino-on-mac install bug and the int8 pool-bypass.
4. **Architecture:** decompose `camera_worker.py` around an immutable `FrameResult` + a real target FSM; pick one appearance mode; type the config/command seams.
5. **Packaging/licensing:** make torch optional (torch-free default); resolve the three weights landmines; retire `onvif-zeep`; sign Windows/Linux.
6. **UX/features:** surface silent failures; fix the identity-gating seam and click-to-track discoverability; add preset-recall-on-lost; consider multi-person auto-switching.

See [`2026-06-24-improvement-plan.md`](2026-06-24-improvement-plan.md) for each item with pros/cons, impact/effort/risk ratings, and a recommended sequencing.
