# 03 — Vision Pipeline: Detection / Tracking / ReID / Face / Pose

This is the heart of the rework — it must fix: fast movers escaping, losing the subject on
occlusion / when someone crosses in front, and face‑only re‑identification. The design separates
**three jobs that v1 conflated**:

1. **Track** — keep a stable per‑person box across frames (motion model). Cheap, every frame.
2. **Re‑identify** — recover the *same* track after occlusion/crossing using **body appearance**
   (ReID) and **face**. On demand.
3. **Identify** — bind a track to a *named* person (Alice) via face. Low rate.

> Key idea: the **tracker** owns continuity; **ReID** owns recovery; **face** owns naming. v1
> tried to make face recognition do all three, which is why it was fragile.

## 3.1 Per‑frame data flow (one camera worker)

```
frame ──► (Nd-cadence) YOLO26 person detection ──► detections[bbox, conf]
                                  │
                                  ▼
                 BoT-SORT/DeepOCSORT update(detections, frame)
                   • Kalman predict (handles fast motion + 1–N frame gaps)
                   • camera-motion compensation (PTZ is always panning)
                   • appearance association via OSNet ReID embeddings
                                  │
                                  ▼
                 tracks[track_id, bbox, velocity, age, state]
                                  │
        ┌─────────────────────────┼───────────────────────────────┐
        ▼                         ▼                                ▼
  (≈1–3 Hz on target)     (on new/ambiguous track)         (zoom enabled, on target)
  InsightFace             OSNet embed → gallery match       RTMPose keypoints
  detect+embed → bind     → restore prior track_id          → desired framing/zoom
  track_id ↔ identity
        └─────────────────────────┬───────────────────────────────┘
                                  ▼
                         Target selection + Framing
                       (error_x, error_y, target_velocity, zoom_error)
                                  ▼
                         PTZ closed-loop controller  (see 04)
```

## 3.2 Detection

- Model: **YOLO26n/s**, person class only, via ORT (EP per platform).
- **Cadence `Nd`:** run detection every frame on GPU tiers; on CPU tiers every 2–3 frames and let
  the tracker's Kalman predictor interpolate. Expose `detect_interval` per camera.
- **Input size:** 640 default; 480 on CPU tiers. Letterbox once; reuse for tracker.
- Confidence/NMS: YOLO26 is NMS‑free; keep a low‑confidence floor so BoT‑SORT's second‑association
  (à la ByteTrack) can still use weak boxes during occlusion.

## 3.3 Tracking

- Library: **BoxMOT**. Default **BoT‑SORT** with ReID on; **DeepOCSORT** for crowded venues;
  **ByteTrack** (no ReID) for the CPU tier. Tracker type is a per‑camera setting.
- Motion model: Kalman filter → predicts position during short detection gaps and **leads fast
  movers**. This alone fixes most "moved a little too fast" failures.
- **Camera‑motion compensation (CMC):** essential because the PTZ head pans/tilts/zooms while
  tracking. BoT‑SORT's CMC (e.g., ECC/optical‑flow) keeps track boxes aligned to the world as the
  frame moves. When the controller commands a large move, feed the known pan/tilt velocity as a
  prior to CMC to reduce drift.
- Track lifecycle: `tentative → confirmed → lost(coasting) → removed`. Keep `lost` tracks alive for
  a configurable **coast window** (e.g., 1–2 s) so an occluded subject can be re‑attached instead
  of spawning a new ID.

## 3.4 Re‑identification (occlusion & crossings)

- Model: **OSNet** (ONNX), produces a 512‑d appearance embedding per crop.
- **When to embed (cost control):**
  - On every *newly created* track (to check it against the gallery of recently‑lost tracks).
  - On *ambiguous associations* (two tracks contest one detection, or IoU is low after a gap).
  - Periodically on the **target** track (e.g., every ~1 s) to keep its template fresh.
  - **Not** on every box every frame.
- **Gallery:** a short‑term ring of embeddings per track (e.g., last K crops, EMA template) plus
  the enrolled long‑term identity templates. Match by cosine similarity with a hysteresis
  threshold (enter > θ_hi, maintain > θ_lo) to avoid flicker.
- **Recovery rule (the "someone walked in front" case):**
  1. Target track goes `lost`. Controller enters *coast* (hold last velocity briefly), then *search*
     (stop or widen zoom).
  2. New tracks appear → embed each → compare against the target's template **and** the named
     identity's enrolled templates.
  3. Best match above θ_hi (and, if a face becomes visible, confirmed by ArcFace) → **re‑bind the
     old `track_id`/identity to the new track**. Resume following.
  4. If an *interloper* (different person) is the strongest box near center, ReID/face prevents
     locking onto them — you only re‑acquire a match to the *target's* templates.

## 3.5 Identity binding (face)

- Models: **InsightFace SCRFD** (detect) + **ArcFace** (embed). Run at **1–3 Hz** on the target
  region (and opportunistically on faces near the center), not every frame.
- Binding: cosine‑match the face embedding to the enrolled gallery → assign `identity` to the
  track. Once bound, the track keeps the identity even when the face is not visible (ReID + motion
  carry it).
- Enrollment: capture N face crops per person across angles; store averaged ArcFace embedding(s)
  in SQLite (see `06`). This replaces v1's pickle‑file‑watched encodings.
- **Target can be chosen two ways:** by **identity** ("always follow Alice" → engine finds the
  track bound to Alice) or by **track** (click a box in the UI → follow that track_id). Identity
  targeting is what makes multi‑camera "follow this person wherever they are" possible.

## 3.6 Pose & smart zoom

- Model: **RTMPose** (or YOLO26‑pose), run only on the **target** bbox when zoom is enabled.
- Use head/shoulder/hip keypoints to compute desired framing: keep **headroom**, frame head‑to‑
  waist at a target ratio (e.g., subject height ≈ 45–60% of frame, configurable: "tight" / "medium"
  / "wide").
- **Zoom controller:** error = (desired_height − current_subject_height). Apply a **hysteresis
  band** (don't zoom for small errors) + rate limit + max/min zoom clamp so it doesn't "hunt."
- **Loss behavior:** on `lost`/`search`, **zoom out** to widen the field of view and re‑acquire,
  then zoom back to the framing preset once re‑bound.
- Low tier: skip pose; derive subject height from the tracker bbox.

## 3.7 Target selection & framing math

- `frame_center = (W/2, H/2)`. `target_point` = a stable point on the target (e.g., face center if
  visible, else upper‑torso point from bbox: `cx, top + 0.25*h`).
- `error = target_point − frame_center`. A **dead‑zone** (configurable ellipse) around center
  suppresses micro‑corrections (replaces v1's fixed 95×60 ellipse, now per‑camera).
- **Velocity feed‑forward (keep relative speed):** estimate target velocity from the Kalman state;
  add a lead term so the camera matches a moving subject instead of always lagging. (Details &
  controller in `04-ptz-control.md`.)
- Smoothing: pass `error` and `velocity` through a **one‑euro filter** (low latency, low jitter)
  before the controller.

## 3.8 Compute‑budget policy (so it scales)

Per camera, per second (tunable per hardware tier):
| Stage | GPU tier | CPU tier |
|---|---|---|
| Detect (YOLO26) | 15–30 Hz | 5–10 Hz |
| Track update | every frame | every frame (cheap) |
| ReID (OSNet) | on‑demand + ~1 Hz on target | on‑demand only, smallest model |
| Face (InsightFace) | 2–3 Hz on target | ~1 Hz, `buffalo_s` |
| Pose (RTMPose) | target only, 10–15 Hz | off (use bbox) |

The engine measures per‑stage latency and **auto‑degrades cadence** when a worker exceeds its
frame budget (drop pose → lower detect rate → switch BoT‑SORT→ByteTrack) so real‑time smoothness
is preserved under load. Surface the active "quality level" in telemetry/UI.

## 3.9 Models to bundle

Ship ONNX (and CoreML where it wins) for: YOLO26n + YOLO26s, OSNet x0_25 (+ optional ain_x1_0),
SCRFD + ArcFace (buffalo_s + buffalo_l), RTMPose‑m (+ optional ‑s). Provide a small **model
manager** that selects the model variant per hardware tier and caches compiled TRT/CoreML
artifacts per machine.
</content>
