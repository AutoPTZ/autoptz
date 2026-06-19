# AutoPTZ v2 — Rework Plan

This folder is the **complete plan of action** for rebuilding AutoPTZ into an extremely
real‑time, cross‑platform (Windows + macOS), multi‑camera PTZ tracking application that
scales from CPU‑only laptops to NVIDIA GPU workstations and M‑series Macs.

It was produced after a full review of the existing v1 codebase and heavy online research
into the 2025/2026 state of the art for detection, tracking, re‑identification, face
recognition, pose, inference runtimes, video ingest, and PTZ control.

> **Who executes this?** These documents are written to be handed to a *different,
> cheaper/faster model* (e.g. Sonnet) for implementation. Each document is self‑contained.
> `08-execution-roadmap.md` breaks the work into ordered phases, and
> `09-implementation-prompts.md` contains copy‑paste, self‑contained task prompts — one per
> phase — that an implementing agent can run without re‑reading the whole repo.

## Reading order

| # | Document | Purpose |
|---|----------|---------|
| 00 | [Current state & goals](00-current-state-and-goals.md) | What v1 does, why it falls short, what v2 must achieve |
| 01 | [Target architecture](01-target-architecture.md) | Engine/UI split, per‑camera process model, data flow |
| 02 | [Technology stack](02-technology-stack.md) | Researched stack + rationale + alternatives |
| 03 | [Vision pipeline: detection / tracking / ReID / face / pose](03-vision-pipeline.md) | The core tracking + re‑identification design |
| 04 | [PTZ control](04-ptz-control.md) | Unified PTZ backends, closed‑loop motion, auto‑zoom, presets |
| 05 | [UI / UX](05-ui-ux.md) | Customizable camera wall, per‑camera config, presets, themes |
| 06 | [Persistence & config](06-persistence-and-config.md) | SQLite + JSON, remembered states, schemas |
| 07 | [System requirements & scaling](07-system-requirements-scaling.md) | Hardware tiers, cameras‑per‑machine, Windows + macOS |
| 08 | [Execution roadmap](08-execution-roadmap.md) | Phased milestones with acceptance criteria |
| 09 | [Implementation prompts](09-implementation-prompts.md) | Self‑contained per‑phase prompts for the executing model |

## One‑paragraph summary of the plan

Keep Python as the core (the CV ecosystem investment is large), but **split the app into a
headless multi‑process Engine and a thin Qt Quick (QML) UI**. Each camera gets its own OS
process that owns *all* of its state — ingest → hardware‑decoded frames → YOLO26 person
detection → BoT‑SORT/DeepOCSORT tracking with OSNet ReID → InsightFace identity binding →
optional RTMPose for smart zoom → closed‑loop PTZ control — and writes only a preview frame
(via shared memory) and small telemetry to the UI. This eliminates v1's global mutable state
(the cause of "data shifted to the wrong camera"), its per‑frame‑on‑the‑GUI‑thread tracking,
and its frame‑pickling overhead. Inference runs through **ONNX Runtime with per‑platform
execution providers** (CoreML on Apple Silicon, TensorRT/CUDA on NVIDIA, DirectML/OpenVINO on
Windows CPU/iGPU) so it runs on CPU and scales with a GPU or an M‑series chip. Everything is
persisted to SQLite + JSON so cameras, layouts, PTZ presets, per‑camera tracking settings, and
identities are remembered across launches.
</content>
</invoke>
