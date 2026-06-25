# AutoPTZ — "Make It Feel Like a Real Product": PTZ Motion, Auto-Zoom / Center-Stage, & Reliability

> Date: **2026-06-25** · Version reviewed: **2.2.0-rc5**
> Method: a 12-agent workflow (6 dimensions × analyze→adversarially-verify) on PTZ control, PTZ transport, auto-zoom/framing, the digital Center-Stage path, competitive parity, and the "stray underscore" report. Every load-bearing claim was checked against source; the reviewer also read `controller.py`, `digital_framer.py`, `vcam.py`, and the framing methods firsthand.
> Companions: [`2026-06-24-architecture-overview.md`](2026-06-24-architecture-overview.md), [`2026-06-24-improvement-plan.md`](2026-06-24-improvement-plan.md). The bundled-virtual-camera section is appended separately when that workflow lands.

---

## 0. The one insight that organizes everything

**AutoPTZ's per-tick control and signal-processing math is already competitive-or-better than OBSBOT, obs-face-tracker, and Apple Center Stage.** It has one-euro filtering, PID + velocity feed-forward, latency-aware look-ahead (with an opt-in 2nd-order term), slew/accel limiting, an oscillation guard, ego-motion compensation, error-proportional catch-up, and a Center-Stage-style safe zone with deadband + hysteresis. Independent competitive research confirmed: **do not spend more effort on the low-level filter math — the returns are gone.**

What stops it from *feeling* like a finished product is **three layers above the math**:

1. **Cadence & threading (structural).** The control loop and the digital framer both run at the *variable* inference/preview frame rate, not a fixed high rate, and the PTZ command send *blocks the inference thread*. → micro-stutter under load, added latency, and reliability gaps.
2. **The "director" layer is thin.** No unified speed dial, no multi-person group framing, no shot-size-aware composition, no debounced switching. This is the difference between "centered" and "directed."
3. **The digital Center-Stage path is a single first-order EMA** — no deadzone, no spring, no `dt`, no multi-person, soft upscale, ≤20 fps.

Everything below ladders into those three.

---

## 1. PTZ movement: how speed actually scales (verified)

Per axis, the exact chain (`controller.py:588-722`, all constants confirmed):

```
pixel error → normalized [-1,1]  (camera_worker._track_error:2071)
  → framing remap to safe-zone centre, deadband ≈0.05–0.08, edge-guard at 0.9  (_framing_error:724)
  → predictive lead:  e += v·(lead_time_s 0.15 + measured_latency), capped 0.8s  (:610)
  → one-euro filter → e_f                                                          (:630)
  → PID+FF:  raw = kp·e_f + ki·∫ + kd·dė + kv·v   (defaults 0.6/0/0.05/0.1)         (:659)
  → speed cap × catch-up:  cap = max_speed(0.7) · (1 + 1.5·catch·min(1,|e_f|/0.6))  (:675)
  → cmd = shape(clamp(raw·cap, ±1)),  shape(x)=sign·|x|^1.2                          (:677)
  → oscillation-guard damp · slew-limit(accel-up only, decel-free) · invert          (:683-713)
```

So commanded speed is **dead-banded near centre → roughly linear-with-ease-in mid-range → super-linear once catch-up engages → hard-saturated at the edge**. Catch-up multiplies the ceiling by up to **2.5×** (default ~1.9× at the frame edge) and saturates at `|error| ≥ 0.6`. It is bounded — no runaway — but error drives the command through three coupled paths (`kp·e`, `kv·v`, and the boost), so it can compound into overshoot when a fast subject suddenly stops.

**Output is hardware-agnostic** (normalized `[-1,1]`; each backend maps to its own range) — but the gains are **absolute**, so the *same config feels twitchy on a fast 300°/s camera and laggy on a slow 60°/s one*. VISCA further **quantizes to 24 pan / 20 tilt / 7 zoom discrete steps**, so sub-step smoothness is lost on VISCA regardless of how smooth the controller is.

---

## 2. The keystone fix (helps smoothness, latency, AND reliability at once)

### ★★★★★ K1 — Run PTZ on a fixed-rate, off-thread command pump
**Confirmed three independent ways** (ptz-control F1, ptz-transport rec1, and the earlier latency pass). The controller's background `_loop()` (`controller.py:467`, built at 20 Hz) **is never started** — the worker drives `step()` *inline, once per inference frame* (`camera_worker.py:1294`). Consequences:
- Command update rate = inference FPS (≈10–30 Hz, **jitters with CPU load**) → visible micro-stutter. A real PTZ tracker emits at a steady 50–60 Hz and interpolates between detections.
- `backend.move_velocity()` — a blocking serial/TCP write or an **ONVIF SOAP HTTP round-trip (5–50+ ms)** — runs on the inference hot path, so a slow camera stalls detection.

**Fix:** start `controller._loop()` (or a dedicated PTZ `QTimer`), feed it via the already-built `update()` (lock-safe), keep `set_loop_latency()` on the vision thread. The tick body is identical, so it's low-risk.

| Pros | Cons |
|---|---|
| Steady command cadence → removes micro-stutter; lower latency (send off hot path); a hung camera stalls only the pump; **enables a heartbeat watchdog** that fixes the runaway-on-disconnect safety gap below | One more thread per camera; tests asserting on `step()` returns need updating; must keep feeding latency each tick |

**Impact High · Effort M · Risk M.** This is the single highest-leverage change in the whole review — it's also the latency fix from the architecture pass, so it pays off twice.

---

## 3. PTZ reliability & safety (real-product robustness)

### ★★★★★ R1 — Stop-on-loss for VISCA + ONVIF dead-man's switch *(safety)*
**Verified runaway:** the VISCA USB/IP backends *silently swallow* sends while inside the reconnect backoff (`visca_usb.py:113`, `visca_ip.py:136`), and VISCA continuous-move runs **until an explicit stop**. So if the cable/socket drops *mid-pan*, the camera **keeps panning for up to the 30 s backoff cap** with no stop. The only existing halt is the inference-stall watchdog (`camera_worker.py:4572`), which keys off inference liveness and **does not fire on a transport-only drop** (inference keeps ticking). ONVIF never sets the `ContinuousMove` `Timeout` field, so it relies entirely on the camera's vendor default.
- **Fix:** issue `stop()` on the backend `connected` True→False edge (both backends already expose `connected`); set ONVIF `Timeout` ≈1–2× the command interval (hardware self-stop); wrap ONVIF `move_velocity` in try/except + a socket timeout so a hung camera can't block. Best delivered *through* K1's pump as a "no fresh command in N ms → send stop" heartbeat. **Impact High (safety) · Effort S–M.**

### ★★★☆☆ R2 — Per-camera speed calibration
Absolute gains make the same config twitchy on fast heads and laggy on slow ones (and VISCA quantization compounds it). The `PTZCaps.{pan,tilt,zoom}_speed_max` fields **already exist** (`base.py:23-25`) but are **never read** by the controller — wire them to scale the final command per backend, plus an optional measured-°/s calibration step. **Impact Med–High · Effort M.**

### ★★★☆☆ R3 — Three small motion-quality fixes
- **Decouple max-speed from loop stiffness** (`controller.py:677`): apply `shape`/clamp to the normalized PD output *first*, then multiply by `effective_speed`. Today lowering "max speed" also softens the loop. Makes the knob honest. *(S)*
- **Decel ramp** (`_slew:200`): deceleration is currently *instantaneous* → the head can cut from full motion to a dead stop in one tick ("clunk"). Add an asymmetric (faster-down) decel limit while *following*; keep the prompt stop only when the hold latch triggers. *(S)*
- **VISCA low-speed dither + widen tilt to `0x17`** (`base.py:122,128`): reduces low-speed micro-jerk from coarse quantization. *(S, VISCA-only)*

### (verified non-issues — do NOT spend effort here)
- The "manual nudge bypasses `_ptz_lock`" race was **refuted** — every caller of `_drive_ptz_nudge` holds `_ptz_lock` (`:876`, `:975`); all five backend-access paths are serialized.
- Coast/search loss handling, the change-only send gate, and the oscillation guard are sound.

---

## 4. Auto-zoom & Center-Stage framing (what users see on the virtual cam)

There are **two separate zoom paths**: the physical-PTZ `_zoom_step` (deliberately sluggish, a 0.48-wide neutral band — it "barely engages," which is *why* the digital path was written) and the **digital `DigitalFramer`** that produces the actual Center-Stage crop. The digital path is a **single fixed-weight (0.18) EMA on the crop rectangle with no `dt`, no deadzone, no spring, single-person, and `INTER_LINEAR` upscale, capped at ≤20 fps**. Verbatim parity gaps vs Apple Center Stage (verified):

| Center-Stage behavior | AutoPTZ today | Fix |
|---|---|---|
| Hold/deadzone so small moves don't move the frame | **None** — every bbox-jitter pixel shifts the crop (digital_framer.py:46,104) | **Positional deadzone + hysteresis** |
| Gentle eased ease-in/out, settle | First-order EMA, frame-rate-dependent (no `dt`) | **Critically-damped spring (ζ=1) with real `dt`**, ~0.6–1.0 s settle |
| Stable zoom (no breathing) | Crop *height* eased by the same EMA as position → breathes with detector height (worst in `full_silhouette`) | **Separate, slower size smoothing + size deadband** |
| Zoom out to include multiple people | **Single target only** (camera_worker.py:3296) | **Union-bbox group framing** (digital path first) |
| Subtle lead toward motion | **None** — centers on current bbox | **Velocity lead** (mirror the controller's lead concept) |
| Clean upscale | `INTER_LINEAR` (camera_worker.py:3294) | **`INTER_CUBIC`/`LANCZOS4`** on upscale (also fix the 2nd resize in `vcam.py:57`) |

### Ranked (re-ranked by the verifier on verified ROI)
- **★★★★★ Z1 — Positional deadzone / hold band** in the digital framer. ~10 lines, no new math, the single biggest "looks like Center Stage" win. *(S)*
- **★★★★☆ Z2 — Critically-damped spring + real `dt`** (replace the EMA). Gentle ease-in/out, frame-rate-independent. *(M; `time.monotonic()` already read at `:3278`/`:3329`)*
- **★★★★☆ Z3 — Separate slower zoom smoothing + size hysteresis** → kills zoom breathing. *(S)*
- **★★★★☆ Z4 — Higher-quality upscale** (CUBIC/LANCZOS4). Cheap, visibly sharper. *(S)*
- **★★★★☆ Z5 — Multi-person union framing.** The marquee missing *feature* vs every competitor; do it in the digital path (no hardware). *(M)*
- **★★★☆☆ Z6 — Robustness:** digital occlusion/collapse guard (mirror `_target_box_collapsed`), reset/fast-ease on target-id switch (today it glides across empty space between subjects), hold-last-shot option on loss (today it always zooms to full frame), and an empty-crop "pop" guard (`camera_worker.py:3292` can momentarily output the full frame). *(S each)*
- **★★★☆☆ Z7 — Cleanups:** fix the dead config-resolve branch (`controller.py:844` reads a non-existent `cfg.framing` → physical zoom can silently mis-size on non-UI configs); delete the dead `DigitalPTZBackend.crop_rect` (keep the class — it's the Center-Stage sentinel). *(S)*

---

## 5. The "director" layer — what makes it read as a finished product

Competitive research (Center Stage, OBSBOT, Jabra, Logitech, Teams, obs-face-tracker) found AutoPTZ matches or beats them on per-tick math; the gaps are all *above* the controller:

- **★★★★★ D1 — Unified "Tracking Speed" preset (Calm / Normal / Fast / Sport).** Every winning product hides its tuning behind *one* dial that co-varies gain + max-speed + smoothing + accel + catch-up. AutoPTZ already owns all six knobs in `PTZConfig` — this is a pure config-layer enum→tuple map, **zero control-loop risk**, and it directly satisfies the "one concept = one control" project principle. Highest value-per-hour in the review. *(S)*
- **★★★★☆ D2 — Nonlinear band between deadzone and linear gain.** obs-face-tracker does exactly this; AutoPTZ's deadband exit is a slope discontinuity (full proportional gain engages immediately past the band). Ramp the post-band slope from 0→full over a configurable width in `_soft_deadband` (`controller.py:186`). *(S)*
- **★★★☆☆ D3 — Shot-size-aware headroom + optional lead-room.** Tie headroom to the framing preset and offset the setpoint in the subject's direction of travel (classic "nose room") using the already-ego-corrected velocity. Moves framing from *centered* to *composed*. *(S–M)*
- **★★★☆☆ D4 — Debounced target switching** (min-hold + recency) — gated on multi-target/group support landing first. *(S–M)*
- **★★☆☆☆ D5 — Commit/settle cadence** — *demoted by the verifier*: medium effort, behaviorally risky, and largely redundant with the existing `_holding` latch + hysteresis. Only pursue if continuous-chase is still visible on real cameras after D1/D2.
- **Roadmap — D6 — Audio-direction (active-speaker) cue.** How Jabra/OBSBOT/Logitech/Teams pick *who* to frame. High effort + mic-array dependency; meeting-use-case only. Roadmap, not parity.

---

## 6. The "random underscore in the install/update prompt" — verdict: **no code bug**

This was investigated to ground truth: the agent pulled **all 18 real GitHub release bodies as raw bytes** (`gh api`) and ran each through the **actual** PySide6 `QTextDocument.setMarkdown()` — the exact call the update dialog makes (`update_dialog.py:71`). Result:

- **Every release body renders clean** — zero stray underscores. Intraword tokens (`detect_batch`, `x86_64`, `Format_BGR888`) render their underscore *literally and correctly*; the only italics come from intended `*asterisk*` emphasis.
- The earlier hypothesis (a stray `_` from the release-notes markdown) is **disproven**. The lead also came from a **WebFetch hallucination**: the summarizer reported `VISCA_USB`/`error_proportional`/`v2.2.0_rc4`, but the real `v2.2.0-rc5` body contains **zero underscores / seven hyphens**.
- Secondary sources re-scanned clean: the `QProgressDialog("", …)` cancel button (`main_window.py:865`, removed by `setCancelButton(None)`), the normalized version string (`2.2.0rc5`), all status-bar/dialog labels, and the Inno Setup script.

**Most likely what was actually seen:** a literal intraword identifier (e.g. `detect_batch`, `x86_64`) in an auto-generated release-notes bullet — which is *correct* output, not a defect.

**Recommendation:**
- **Do NOT** switch `setMarkdown`→`setPlainText` (it would kill bold/links/bullets to "fix" a non-bug).
- **Editorial (preferred):** backtick or rephrase raw snake_case identifiers in PR titles so the notes read cleanly.
- **Optional defensive hardening** at `model_manager.py:126` (a latent edge, not the reported symptom): `self.display_name = str(row.get("name") or self.key.replace("_", " ").title())` so a future name-less model row can't surface a raw key.
- If you saw the underscore somewhere *specific* (a screenshot/which screen), that pins it instantly — the code paths above are all clean, so it would be a surface this sweep didn't enumerate.

---

## 7. Recommended sequencing

**Slice 1 — quick, high-perceived-quality wins (all S, low risk):** D1 unified speed preset · D2 nonlinear band · Z1 digital deadzone · Z3 split zoom smoothing · Z4 better upscale. *Net: the product immediately feels more "finished" with near-zero algorithmic risk.*

**Slice 2 — the keystone + safety (M, validate on real cameras):** K1 fixed-rate off-thread pump · R1 stop-on-loss + ONVIF dead-man's switch · Z2 spring+`dt`. *Net: smoother steady motion, lower latency, and the runaway-on-disconnect hazard closed.*

**Slice 3 — director features (M):** Z5 multi-person union framing · D3 headroom/lead-room · R2 per-camera calibration · Z6 robustness guards. *Net: closes the marquee Center-Stage feature gap and the per-camera feel inconsistency.*

**Slice 4 — polish / roadmap:** R3 motion-quality trio · Z7 cleanups · D4 debounced switching · D5 commit-settle (only if needed) · D6 audio cue (roadmap).

All PTZ-motion changes must be validated on the real cameras (per project policy) before merge — controller behavior can't be fully judged from tests.

---

## 8. Bundled virtual camera — ship "AutoPTZ Camera" so it works without OBS

Today AutoPTZ ships **no** virtual camera: `vcam.py` wraps optional `pyvirtualcam`, which only *targets a pre-existing device* (OBS Virtual Camera on Windows/macOS, v4l2loopback on Linux) and is a silent no-op if none is installed. Every consumer product that "just works" (OBSBOT, Camo, EpocCam, NVIDIA Broadcast, mmhmm) **ships its own OS-level driver — none piggyback on OBS.** This is a per-OS native-driver + signing + licensing project. All claims below are primary-source-verified.

### 8.1 The headline licensing finding (act on this regardless of the rest)

> **`pyvirtualcam` is GPL-2.0-ONLY, and AutoPTZ is AGPL-3.0 — that is a live, present license conflict.** Per the FSF, GPL-2.0-*only* cannot be combined with AGPL-3.0 in one work, and importing `pyvirtualcam` into the AGPL app is exactly that combination. (Verified: `pyvirtualcam`'s LICENSE is bare GPLv2 with no "or later", and its `setup.py` classifier is `GPLv2`, not `GPLv2+`.)
>
> By contrast, **OBS Studio, v4l2loopback, and akvcam are all GPL-2.0-OR-LATER** — upgradeable to v3 and therefore **AGPL-3.0-compatible** (the initial research mis-flagged these as risky; GitHub's "GPL-2.0" sidebar label is the family tag, not "only"). So the third-party *drivers* are licensing-cheap; the *dependency you already ship* is the expensive one.

**Action:** audit/retire `pyvirtualcam`. The hybrid plan below naturally removes it on macOS/Windows (own backends); on Linux, talk to v4l2loopback directly (it's or-later) or isolate `pyvirtualcam` as a separately-licensed optional shim. This dovetails with the weights/license landmines already tracked in the architecture review.

### 8.2 Per-OS mechanism (verified)

| OS | Recommended tech | OS floor | Frame transport | Signing | Effort |
|---|---|---|---|---|---|
| **macOS** | **CMIO Camera Extension** (`.appex`, replaces the removed DAL plugin) | **macOS 13+** (Ventura) | **CMIO sink-stream**: host pushes **IOSurface-backed `CVPixelBuffer`** (NV12/BGRA) — IOSurface is mandatory or consumers show nothing. **NOT** a shared-memory ring (see 8.4). | Developer-ID + notarization + a **provisioning profile** with `com.apple.developer.system-extension.install` + App Group + camera entitlements. AutoPTZ already signs+notarizes, so the pipeline exists; the entitlement/profile do **not** (absent from `entitlements.plist`). | **L** |
| **Windows (broad)** | **DirectShow source filter** (COM `.ax`/DLL, `regsvr32` at install; what OBS uses) | **Win10 + 11** | A separate COM server reads the engine's **shared-memory ring**; deliver **NV12**. Ship **both 32- and 64-bit** DLLs (a 64-bit-only filter is invisible to 32-bit consumers). | Authenticode-sign to avoid SmartScreen (not mandatory to function); no kernel-driver signing. | **L** |
| **Windows 11 (optional, later)** | **`MFCreateVirtualCamera`** (Media Foundation) | **Win11 22000+ only** | Registered custom media source (CLSID) the Frame Server pulls; **created by a runtime call**, not pure `regsvr32` (so *not* "available before first launch" like DShow). | Sign the media-source DLL; admin for All-Users/System lifetime. | **L** (not cheaper than DShow) |
| **Linux** | **v4l2loopback** (status quo) or **PipeWire** node | kernel 5.0+ | Engine writes `/dev/videoN` directly (no process boundary) — `pyvirtualcam` path works | none | **M** integrate; **cannot bundle** a kernel module in an AppImage → detect-and-instruct (`modprobe v4l2loopback exclusive_caps=1 card_label="AutoPTZ Camera"`) |

### 8.3 Recommended strategy — the hybrid

**Own CMIO extension (macOS) + DirectShow→MF (Windows) + depend-on-v4l2loopback (Linux), keeping the current no-op as the graceful floor.** macOS is where the status quo is genuinely broken (Apple removed the DAL plugin in macOS 14, so even the OBS-piggyback path is fragile) *and* where AutoPTZ already owns the exact signing/notarization machinery the extension needs — highest payoff-to-feasibility. A credible faster fallback: **bundle OBS's virtual-cam component** (now confirmed GPL-2.0-or-later → low license risk) as a separate program, trading the "AutoPTZ installed OBS bits" perception for not writing native code.

### 8.4 Integration plan (verified, with one refutation)

- **Reuse the existing SHM ring** (`shm.py`, lock-free triple-buffer, `=QQII` header) as the engine→device transport on **Windows + Linux** — its byte layout is plainly parseable by a native reader.
- **macOS shared-memory is a DEAD END (refuted):** a notarized CMIO extension runs as the `_cmiodalassistants` role account, **not** the GUI user, so an App-Group container resolves to *different paths* on each side. macOS must use the **CMIO sink-stream + IOSurface `CVPixelBuffer`** path (XPC handshake), not a shared file.
- **Output-quality fixes — do these now (pure Python, no native code, ship immediately):**
  1. **Decouple the vcam send from the 20 fps preview gate.** Verified defect: `_push_frame` (and thus the vcam send) is throttled to ≤20 fps by `_PREVIEW_PUSH_FPS` (`camera_worker.py:72,3330`), while real cameras output a steady **30 fps**. Drive the vcam on its own 30 fps clock and **repeat the last frame** so consumers see continuous video.
  2. **Better upscaler** (`camera_worker.py:3294`): `INTER_CUBIC`/`LANCZOS4` when upscaling a crop, `INTER_AREA` when downscaling. *(Same fix as Z4 — it serves both preview and vcam.)*
  3. **Convert to NV12 once** in the engine (Zoom/Teams/Meet prefer NV12; 1280×720@30 is the safe baseline).
- **Lifecycle:** install the device **persistently** (survives app close, so users can pick "AutoPTZ Camera" before launching), but create/destroy only the *stream* with `vcam_out`; the current "release device on disable so consumers disconnect" behavior (`camera_worker.py:3345`) is correct — keep it. Handle macOS extension **deactivation on uninstall** (a known OBS pain point) and a "no signal" placeholder when the engine is idle/crashed.

### 8.5 Ranked rollout (verifier's tiers)

- **Tier 1 — now, pure Python, zero platform risk:** decouple vcam fps (30 + repeat-last) · CUBIC/LANCZOS4 upscale · NV12 once · surface configured→effective when no device. *Immediate quality win; ships today.*
- **Tier 2 — medium, no native toolchain:** a `VCamBackend` abstraction over `_push_frame` (keep `VirtualCamSink` as the Linux backend + OBS fallback) · first-class Linux detect-and-instruct UX · **audit/retire the `pyvirtualcam` GPL-2.0-only dependency**.
- **Tier 3 — high, new native toolchain + signing (gate behind Tier 1–2):** Windows **DirectShow filter first** (broadest reach, no Win11 cliff, fits the existing admin Inno installer) → Win11 **MF** source → macOS **CMIO extension last** (new Swift target + the missing `system-extension.install` entitlement/provisioning profile + Apple-gated approval UX + macOS 13+ floor).

**Net:** the device work is real native engineering, but the **Tier-1 output-quality fixes are free and overlap the Center-Stage smoothness work in §4**, and the **`pyvirtualcam` license conflict is worth resolving on its own merits** whether or not the bundled device is built.
