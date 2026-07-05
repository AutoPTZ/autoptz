# Virtual PTZ simulation — closing the control loop without hardware

*Design note. Motivated by a concrete, current gap: the PTZ-framing-parity work
(physical PTZ now frames like Center Stage, including group framing) drives real
motors on a code path that can only be validated on a physical PTZ camera. There
is no repeatable, CI-able way to test PTZ **control quality** today.*

## The gap

AutoPTZ Mark and the bundled clips are honest for **throughput/scaling** — "how
many cameras can this machine run" — but they are **fixed-viewpoint recorded
video**. They never pan, tilt, or zoom, so they cannot exercise the physical PTZ
control loop at all (see [mark-simulation-fidelity.md](mark-simulation-fidelity.md)).
Everything that makes physical PTZ hard — aim latency, overshoot, oscillation,
reacquire after occlusion, ego-motion separation, and now group-framing
composition on motors — is invisible to the current test surface. The only way to
see it is to point a real PTZ camera at a real scene and watch.

That means a change like the shared framing target (group aim on physical PTZ)
ships with unit tests that prove the *math* but no automated proof of the
*behavior*. A virtual PTZ environment is the missing piece: a scene whose camera
actually moves when the controller commands it, so the loop is **closed** and
**repeatable**.

## What a virtual PTZ environment gives you

A rendered scene + a virtual camera that responds to pan/tilt/zoom commands lets a
test:

- **Close the loop.** Command → the viewport actually moves → the next frame
  reflects it → the controller reacts. This is the one thing clips can't do.
- **Score control quality deterministically.** With known ground-truth subject
  position and camera pose per frame, a run yields hard numbers: time-to-center,
  overshoot %, settle time, oscillation count, reacquire time, and — for the new
  group path — how well the framed union tracks the true group centroid.
- **Regress in CI.** Deterministic renders make control-quality regressions
  reproducible without a lab, so a tuning change that adds oscillation fails a
  test instead of a demo.
- **Exercise ego-motion.** A moving virtual camera with known pose is exactly the
  reference the egomotion/ego-gate stack never gets from a fixed clip.

## Recommended path: start 2D, not 3D

A full 3D renderer is a large subsystem and is **not** the cheapest way to close
the loop. The control loop only needs "the observed image moves in response to a
PTZ command." That can be done in 2D first:

**Phase 1 — 2D virtual PTZ over a wide canvas (cheap, high value).**
A synthetic frame source holds a **wide scene** — a panorama, a large rendered
canvas, or a 4K/8K video — and emits a **cropped viewport** of it. A new virtual
PTZ backend (sibling of the existing `DigitalPTZBackend`) does not move motors; it
**moves the crop viewport**: pan/tilt shift the viewport center, zoom changes the
viewport size. Move one or more synthetic "people" across the wide scene on a
known path. Now:
- The controller commands the virtual backend → the viewport moves → the person's
  apparent position in the emitted frame changes → the pipeline detects it → the
  controller reacts. The loop is closed.
- Ground truth is free: we know the person's canvas position and the viewport, so
  we know the true center error every frame.
- No renderer, no GPU, no hardware — it reuses the synthetic-source and PTZ-backend
  seams that already exist (`autoptz/engine/pipeline/ingest.py` synthetic adapter,
  `autoptz/engine/ptz/` backends). Latency/actuation can be injected by delaying
  when a command takes effect on the viewport, so lead-time and slew behavior are
  testable too.

This Phase-1 rig directly validates the PR2 group-aim path: put two "people" on
the canvas, enable group framing, and assert the viewport centers on their
midpoint and holds without oscillating.

**Phase 2 — rendered 3D scene (later, if needed).**
A real 3D scene adds parallax, perspective foreshortening, optical-zoom artifacts
(focus breathing), and true 6-DoF camera pose — things a 2D canvas approximates
but doesn't reproduce. Build this only when Phase 1's 2D fidelity is provably the
limiting factor (e.g., ego-motion tests need real parallax). It's a renderer + a
camera-motion model + ground-truth export — a distinct project.

## When to build it

- **Now-ish (Phase 1)** is justified: the PTZ-parity work just added motor
  behavior with no closed-loop test, and the 2D rig is cheap because it reuses
  existing seams. It converts "validate on real cameras" from a manual demo into
  an automated gate for the parts that don't need real optics.
- **Phase 2 (3D)** waits for a tracking-quality gate that specifically needs
  parallax/optical realism.

## Non-goals

- This is **not** a replacement for a final real-camera check — real lenses,
  motors, and networks still surface issues no sim will. It's a way to catch
  control-quality regressions early and repeatably, and to shrink what has to be
  verified by hand on hardware.
- It is **not** part of the current release; it's the recommended next investment
  for making physical-PTZ control quality testable.
