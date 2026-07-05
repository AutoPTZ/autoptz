# AutoPTZ Mark — simulation fidelity

*What the bundled Mark scenes actually measure, what they can't, and when a
rendered 3D PTZ environment would be worth building.*

AutoPTZ Mark adds fake cameras one at a time and measures how many the machine can
run smoothly, with the real pipeline running on a built-in scene. This note is an
honest read on what those scenes prove — so a green Mark verdict is not mistaken
for evidence it can't give.

## What the built-in clips exercise

The source is either a **built-in clip** (real decoded video, the recommended
default) or **live NDI sources** on the network. The clip library
([`autoptz/ui/mark_session.py`](../../autoptz/ui/mark_session.py), `CLIP_LIBRARY`)
covers the pipeline's main capabilities:

| Clip | Native | Exercises |
|------|--------|-----------|
| **Crowd Crossing** (`crowd`, default) | 720p / 30 | Tracking + re-ID through people crossing and occluding each other |
| **Pedestrians** (`pedestrians`) | 1080p / 30 | Sustained multi-person tracking at HD |
| **Cinematic People** (`cinematic_24`) | 1080p / 24 | Center Stage framing at 24 fps |
| **Cinematic People** (`cinematic_60`) | 1080p / 60 | Center Stage framing at high frame rate |
| **Faces** (`faces`) | 720p / 30 | Face detection + recognition |

The Mark transcode grid fabricates upscaled and frame-duplicated variants
(720p→1080p→4k, 24/30→60) so a scene can be measured at resolutions/fps it wasn't
recorded at; those synthetic variants are labelled honestly in the pre-flight UI
("(upscaled)", "(frame-duplicated)"). They stress **decode + inference + framing
throughput** at the requested size, which is exactly what the "how many cameras
can this machine run" verdict needs.

## What the clips cannot simulate

A fixed-viewpoint recorded clip is decoded and fed to the pipeline. It is a
faithful load test, but it is **not** the physical camera loop:

- **Real PTZ ego-motion.** The clips never pan/tilt/zoom, so the egomotion / ego-gate
  stack (which separates subject motion from camera-induced frame shift) is not
  exercised. A Mark run says nothing about how well physical PTZ holds a subject
  while the camera itself is moving.
- **Parallax and optical zoom artifacts.** Focus breathing, exposure/white-balance
  shifts on zoom, and depth parallax as the lens moves are absent from a flat clip.
- **Closed-loop control latency.** The true PTZ loop is *command → motor motion →
  observed pixel change*. A clip has no actuator, so lead-time, slew limiting, and
  oscillation behaviour are never closed against real motion — only measured as
  telemetry against a scripted scene.
- **Network behaviour.** The live-NDI source option covers some of this (real SDK
  buffering and drops); the clip path does not.

In short: clips are honest for **throughput and scaling** verdicts. They are **not**
evidence of **PTZ control quality**.

## When a rendered 3D PTZ environment would pay off

A 3D scene with a virtual PTZ camera that actually pans/tilts/zooms on command
would add exactly the things clips cannot:

- **Closed-loop control validation** — the rendered camera moves when the controller
  commands it, so aim latency, overshoot, oscillation, and reacquire can be measured
  end to end.
- **Ground-truth ego-motion** — known camera pose per frame gives a reference for the
  egomotion/ego-gate stack instead of inferring correctness.
- **Repeatable regression scenes** — deterministic renders make control-quality
  regressions reproducible in CI in a way live cameras never are.

That is a substantial new subsystem (a renderer, a virtual-camera-motion model, and
ground-truth plumbing). It is worth building **only when a tracking-quality gate
demands closed-loop evidence** — see the control-quality direction in the PTZ-parity
work and `docs/MASTER-PLAN.md`. Until then, the current clips remain the right tool
for throughput/scaling, and real cameras remain the check for control quality.

A concrete, cheaper-first design for that closed-loop rig — a 2D "virtual PTZ over
a wide canvas" that reuses the existing synthetic-source and PTZ-backend seams,
before any 3D renderer — is written up in
[virtual-ptz-simulation.md](virtual-ptz-simulation.md).
