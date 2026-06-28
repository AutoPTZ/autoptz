"""Curated experimental AUTOPTZ_* flags + experimental TrackingConfig defaults.

Single source of truth shared by the Experimental Features dialog (to build the
rows) and :func:`autoptz.engine.supervisor.Supervisor._apply_experimental_env`
(to publish the chosen values into ``os.environ`` before workers spawn).  Adding
a flag here automatically exposes it in the UI and applies it at engine start.

Each ``default`` is the value that means "engine default" — when the persisted
selection equals it, the supervisor leaves the env var UNSET so the existing
in-code fallback runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentalFlag:
    """One toggle-able experimental env flag surfaced in the UI."""

    env_key: str
    label: str
    description: str
    default: str
    kind: Literal["bool", "choice"]
    choices: tuple[str, ...]
    restart_required: bool


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
        env_key="AUTOPTZ_PROCESS_PER_CAMERA",
        label="One process per camera",
        description=(
            "Run each camera worker in its own OS process to sidestep the Python "
            "GIL across cameras. Big win at a few cameras, but each child duplicates "
            "the full model set so it does NOT scale (heavy RAM, collapses at ~16). "
            "Experimental, off by default."
        ),
        default="0",
        kind="bool",
        choices=(),
        restart_required=True,
    ),
    ExperimentalFlag(
        env_key="AUTOPTZ_INFERENCE_SCHEDULER",
        label="Centralized inference scheduler",
        description=(
            "Route all cameras' detection through ONE shared scheduler that owns the "
            "accelerator (fair across cameras), instead of each camera thread running "
            "the detector inline. Aims to scale without the per-process RAM cost. "
            "Experimental, off by default."
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
    ),
)


# The 4 experimental TrackingConfig bool fields, surfaced as app-level defaults
# applied to NEWLY added cameras.  Defaults MUST mirror
# ``autoptz.config.models.TrackingConfig`` exactly.
TRACKING_DEFAULT_FIELDS: tuple[tuple[str, str, str, bool], ...] = (
    (
        "unified_pose",
        "Unified pose (new cameras)",
        "Default new cameras to the unified one-backbone pose detector.",
        False,
    ),
    (
        "use_target_associator",
        "Fused target associator (new cameras)",
        "Default new cameras to the fused keep/switch associator (motion + "
        "appearance + identity + pose) instead of the heuristic path.",
        False,
    ),
    (
        "stage_spread",
        "Stage-spread inference (new cameras)",
        "Never run the detector and pose pass on the same frame, so heavy ticks "
        "don't stack into one slow frame. On by default.",
        True,
    ),
    (
        "group_framing",
        "Group framing (new cameras)",
        "When several people are present with no locked target, frame the whole "
        "group instead of one subject (Center Stage digital crop only).",
        False,
    ),
)
