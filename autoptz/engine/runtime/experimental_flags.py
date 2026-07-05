"""Curated AUTOPTZ_* flags surfaced in the Experimental Features dialog.

Single source of truth for env values that may be persisted from the dialog and
then published by :func:`autoptz.engine.supervisor.Supervisor._apply_experimental_env`
before workers spawn.  Flags are grouped by ``section`` (Experiments, Devices &
tuning, Model overrides, Diagnostics) and read at engine start, so a restart
applies changes.

Deliberately NOT listed here (they stay env-only, documented in docs/flags.md):
the supervisor-managed hardware vars (``AUTOPTZ_FORCE_EP`` / ``AUTOPTZ_PRECISION``
/ ``AUTOPTZ_ORT_INTRA_THREADS`` / ``AUTOPTZ_CV2_THREADS`` — a hardware-prefs path
already writes them, so a second writer here would drift), plus launch-only
(``AUTOPTZ_SKIP_CAMERA_PREFLIGHT``), test-harness, and logging-cosmetic vars.

Each ``default`` is the value that means "engine default" — when the persisted
selection equals it, the supervisor leaves the env var UNSET so the existing
in-code fallback runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentalFlag:
    """One managed experimental env flag for dev/benchmark tools.

    ``section`` groups the flag under a heading in the dialog.  ``kind`` selects
    the editor: ``bool`` → checkbox, ``choice`` → dropdown (from ``choices``),
    ``text`` → free-text line edit, ``path`` → line edit with a Browse button.
    """

    env_key: str
    label: str
    description: str
    default: str
    kind: Literal["bool", "choice", "text", "path"]
    choices: tuple[str, ...]
    restart_required: bool
    section: str = "Experiments"


# Ordered for display.  ``default`` strings mirror the real in-code fallbacks
# (see camera_worker / process_worker / reid / inference / ingest / ptz.factory).
EXPERIMENTAL_FLAGS: tuple[ExperimentalFlag, ...] = (
    ExperimentalFlag(
        env_key="AUTOPTZ_UNIFIED_POSE",
        label="Unified pose detector",
        description=(
            "Use one YOLO11-pose backbone that emits person boxes AND keypoints "
            "in a single pass, instead of a separate detector plus a pose pass. "
            "Falls back to the plain detector if the pose model can't be built."
        ),
        default="0",
        kind="bool",
        choices=(),
        restart_required=True,
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_ASYNC_APPEARANCE",
        label="Async appearance pass",
        description=(
            "Run face recognition and appearance ReID on their own thread so the "
            "heavy appearance work overlaps inference instead of stalling it. "
            "On by default; turn off to diagnose appearance-thread issues."
        ),
        default="1",
        kind="bool",
        choices=(),
        restart_required=True,
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_PTZ_PUMP",
        label="Background PTZ command pump",
        description=(
            "Drive PTZ commands from a dedicated background loop instead of "
            "inline on the inference thread, to keep aim latency steady under "
            "load. Experimental — validate on real cameras before relying on it."
        ),
        default="0",
        kind="bool",
        choices=(),
        restart_required=True,
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_PTZ_SERIAL_AUTOPROBE",
        label="Auto-probe USB PTZ serial port",
        description=(
            "Scan serial ports for a companion VISCA control port when a USB PTZ "
            "camera opens, so pan/tilt/zoom and the camera menu work without "
            "manual setup. On by default; turn off if the scan stalls startup."
        ),
        default="1",
        kind="bool",
        choices=(),
        restart_required=True,
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_REID_DEVICE",
        label="ReID compute device",
        description=(
            "Force the OSNet appearance (ReID) model onto a specific device. "
            "Auto picks the best available (Apple mps / CUDA, else CPU); pin "
            "'cpu' if an OSNet op misbehaves on the GPU."
        ),
        default="",
        kind="choice",
        choices=("", "cpu", "mps", "cuda"),
        restart_required=True,
        section="Devices & tuning",
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_COREML_UNITS",
        label="CoreML compute units (macOS)",
        description=(
            "Target for the CoreML execution provider on Apple/Intel-Mac builds. "
            "Auto uses ALL; 'CPUOnly' measures whether the GPU helps, 'CPUAndGPU' "
            "pins the discrete GPU, 'CPUAndNeuralEngine' targets the Apple Neural "
            "Engine. Invalid values fall back to ALL."
        ),
        default="",
        kind="choice",
        choices=("", "ALL", "CPUAndGPU", "CPUOnly", "CPUAndNeuralEngine"),
        restart_required=True,
        section="Devices & tuning",
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_TRUE_LATENCY_LEAD",
        label="True end-to-end latency lead",
        description=(
            "Lead the PTZ aim by the MEASURED whole-pipeline dead time (capture "
            "age + command send + configured actuation estimate) instead of just "
            "the ingest+inference latency. Off by default; the decomposition is "
            "always measured for telemetry, but only steers the lead when on."
        ),
        default="0",
        kind="bool",
        choices=(),
        restart_required=True,
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_NDI_COLOR_FORMAT",
        label="NDI receive color format",
        description=(
            "Which color format to request from NDI sources. 'fastest' takes the "
            "SDK's cheapest native format (lighter CPU); 'bgra' forces the SDK's "
            "BGRA conversion as an escape hatch for misbehaving sources."
        ),
        default="fastest",
        kind="choice",
        choices=("fastest", "bgra"),
        restart_required=True,
        section="Devices & tuning",
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_MODEL_SERVER",
        label="Shared detection server (multi-camera)",
        description=(
            "Run one shared detection server process that every camera delegates "
            "to, instead of each camera loading its own model set. Validated as "
            "the best-scaling mode for many cameras (54 ms end-to-end at 16 "
            "cameras vs. 1.6 s for the threaded path) and now self-healing: the "
            "server auto-respawns on crash, fails fast during an outage instead "
            "of hanging cameras, and cameras fall back to their own local "
            "detector after repeated failures. Still experimental — off by "
            "default; validate on your hardware before relying on it."
        ),
        default="0",
        kind="bool",
        choices=(),
        restart_required=True,
    ),
    # ── Model overrides ──────────────────────────────────────────────────────
    ExperimentalFlag(
        env_key="AUTOPTZ_MODEL_PATH",
        label="Detector model file",
        description=(
            "Use a specific detector ONNX file instead of the managed, tier-based "
            "download. Leave empty to use the model chosen in Manage Models."
        ),
        default="",
        kind="path",
        choices=(),
        restart_required=True,
        section="Model overrides",
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_POSE_MODEL_PATH",
        label="Pose model file",
        description=(
            "Use a specific pose ONNX file instead of the managed download. Leave "
            "empty to use the bundled/downloaded pose model."
        ),
        default="",
        kind="path",
        choices=(),
        restart_required=True,
        section="Model overrides",
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_MODEL_URL",
        label="Model download base URL",
        description=(
            "Override the base URL prebuilt model ONNX files are downloaded from. "
            "Leave empty to use the built-in release URL."
        ),
        default="",
        kind="text",
        choices=(),
        restart_required=True,
        section="Model overrides",
    ),
    # ── Diagnostics ──────────────────────────────────────────────────────────
    ExperimentalFlag(
        env_key="AUTOPTZ_MS_DIAG",
        label="Shared-server diagnostics logging",
        description=(
            "Emit periodic shared-detection-server health logs (queue depth, "
            "round-trip timing) to help diagnose model-server issues. Off by default."
        ),
        default="0",
        kind="bool",
        choices=(),
        restart_required=True,
        section="Diagnostics",
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_SYNTH_DEBUG",
        label="Synthetic source debug logging",
        description=(
            "Emit extra logging from the synthetic/test frame source. Only useful "
            "when running against synthetic input; off by default."
        ),
        default="0",
        kind="bool",
        choices=(),
        restart_required=True,
        section="Diagnostics",
    ),
)
