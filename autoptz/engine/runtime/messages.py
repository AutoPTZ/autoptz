"""Typed telemetry and command schemas for Engine↔UI communication.

Serialization: pydantic models round-trip through msgpack for compact
cross-process transport.  JSON is also available via .model_dump_json().
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Self

import msgpack
from pydantic import BaseModel, Field

# ── shared primitives ─────────────────────────────────────────────────────────


class BBox(BaseModel):
    """Pixel-space bounding box (top-left, bottom-right)."""

    x1: float
    y1: float
    x2: float
    y2: float


# ── telemetry (Engine → UI) ───────────────────────────────────────────────────


class TrackInfo(BaseModel):
    track_id: int
    bbox: BBox
    identity: str | None = None  # display NAME ("Person 3" / "Alice"), or None
    identity_id: str | None = None  # stable gallery id (for enroll / target), or None
    confidence: float = 0.0
    is_target: bool = False
    # True while the track is coasting (LOST: no fresh detection this frame).  The
    # UI fades the box and the worker stops driving the PTZ toward a stale box.
    lost: bool = False
    # Estimated box velocity in pixels/frame, surfaced so the UI can draw a motion
    # prediction indicator (a lead vector / ghost box).
    vx: float = 0.0
    vy: float = 0.0
    # The current target aim point in frame pixels. Present for the active target
    # when the worker has a live frame; the UI draws the tracking dot here so it
    # matches PTZ control instead of the geometric box center.
    aim_x: float | None = None
    aim_y: float | None = None
    aim_source: str = ""  # "pose", "fused", "bbox", "silhouette", or ""


class FaceBox(BaseModel):
    """A detected face for the optional face-recognition overlay (pixel-space)."""

    bbox: BBox
    identity: str | None = None  # matched display NAME, or None when unknown
    score: float = 0.0  # match cosine (0 when unmatched)


class PoseKeypoint(BaseModel):
    """One COCO-17 pose keypoint (pixel-space) for the optional pose overlay."""

    x: float
    y: float
    conf: float = 0.0


class PTZState(BaseModel):
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    moving: bool = False
    backend: str = ""
    state: str = "idle"


class TrackingStatusInfo(BaseModel):
    """Operator-facing explanation of the current target-tracking state."""

    state: str = "idle"
    headline: str = ""
    detail: str = ""
    action: str = ""
    remaining_s: float = 0.0
    severity: str = "info"


class HealthState(str, Enum):
    OK = "ok"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


class HealthInfo(BaseModel):
    state: HealthState = HealthState.OK
    last_error: str | None = None
    # Observability: seconds since the last completed inference tick while
    # tracking is active (0.0 when not applicable).  Additive — older UIs ignore
    # it; diagnostics surface it next to the inference-stall watchdog state.
    inference_stall_age_s: float = 0.0


class RuntimeServiceInfo(BaseModel):
    """One configured/runtime service row for transparent diagnostics."""

    key: str = ""
    name: str = ""
    scope: str = "camera"  # "global" or "camera"
    configured: str = ""
    enabled: bool = True
    active: bool = False
    state: str = "idle"  # active / disabled / warming / stale / idle
    detail: str = ""
    model: str = ""
    tier: str = ""
    backend: str = ""
    ep: str = ""
    confidence: str = ""


class StageTimingInfo(BaseModel):
    """Rolling runtime timing for a stage such as ingest/detect/track/face."""

    key: str = ""
    name: str = ""
    status: str = "idle"  # active / disabled / warming / stale / idle
    last_ms: float = 0.0
    avg_ms: float = 0.0
    p95_ms: float = 0.0
    cadence: str = ""
    fresh: bool = False
    age_ms: float = 0.0
    budget_pct: float = 0.0
    detail: str = ""


class QualityStateInfo(BaseModel):
    """Effective adaptive quality state and operator-facing reason."""

    floor: str = "auto"
    active: str = "auto"
    reason: str = ""
    detector_tier: str = ""
    detector_model: str = ""
    tracker: str = ""
    detect_interval: int = 1


class SwitchStateInfo(BaseModel):
    """Current or most-recent hot-swap state for detector/tracker changes."""

    kind: str = ""
    state: str = "idle"  # idle / warming / active / failed / rolled_back
    from_value: str = ""
    to_value: str = ""
    active_value: str = ""
    reason: str = ""
    ts: float = 0.0
    error: str = ""


class RuntimeEventInfo(BaseModel):
    """Recent runtime event shown in diagnostics/history."""

    ts: float = Field(default_factory=time.time)
    kind: str = ""
    level: str = "info"
    message: str = ""


class TelemetryMsg(BaseModel):
    """Emitted by each CameraWorker at ~telemetry_hz (default 10 Hz)."""

    camera_id: str
    seq: int
    ts: float = Field(default_factory=time.time)
    fps: float = 0.0
    ep: str = ""  # active inference EP (e.g. "CoreMLExecutionProvider")
    # Source frame dimensions (the *ingested* resolution, not the preview shm
    # size) — fed to the UI's Camera Info panel.  0 until a frame is read.
    width: int = 0
    height: int = 0
    # Active Center-Stage digital crop ``(x, y, w, h)`` in FULL-frame pixels, or
    # None when Center Stage is not driving the painted preview this tick. The UI
    # uses it to re-normalize overlay boxes/aim into the cropped+scaled preview
    # so the detection/track/face/pose overlays follow the digital crop instead
    # of floating over the full frame. None ⇒ overlays keep full-frame
    # normalization (unchanged legacy behavior).
    digital_crop_rect: tuple[int, int, int, int] | None = None
    # Cumulative count of frame read() misses / decode failures since start —
    # surfaced as "dropped frames" in Camera Info.
    dropped_frames: int = 0
    # Per-frame processing latency in milliseconds (ingest read + detect + track
    # wall time for the most recent frame).  Surfaced as "Latency" in Camera Info
    # and the live stats overlay.  0.0 until the first frame is processed.
    latency_ms: float = 0.0
    # Per-stage breakdown of the latency above (milliseconds, most recent frame).
    # Surfaced in Camera Info ▸ Performance so the operator can see which stage is
    # the bottleneck.  ``detect_ms``/``track_ms`` are 0.0 on frames that skip
    # detection (detect_interval) until the next detect tick.
    ingest_ms: float = 0.0
    detect_ms: float = 0.0
    track_ms: float = 0.0
    # Face recognition + pose stage wall time (milliseconds, most recent run).
    # These run async/throttled (not every frame), so the value reflects the most
    # recent run and stays put between runs.  Surfaced in the tile "?" badge and
    # Camera Info ▸ Performance so the operator sees the per-subsystem cost that
    # explains the fps cliff (e.g. capture 30 → +detection 23 → +face 19).
    face_ms: float = 0.0
    pose_ms: float = 0.0
    # True once at least one real frame has been read + pushed to the preview shm.
    # The UI gates its "No Signal" overlay on this (not on fps, which lags a full
    # second, nor on the preview Image load, which can latch on a placeholder).
    streaming: bool = False
    # Trusted source fps ceiling (0.0 = unknown). Low current/default stream-rate
    # readings are not reported as caps, so the UI does not present them as the
    # camera's real maximum.
    source_fps_cap: float = 0.0
    # Effective requested capture/inference rate and derived frame budget. These
    # make "30 fps ≈ 33 ms/frame" explicit in diagnostics instead of overloading
    # vague load labels.
    target_fps: float = 0.0
    frame_budget_ms: float = 0.0
    # Layered runtime transparency. These are additive: older UI paths keep using
    # the scalar ms/model fields, newer diagnostics read the structured rows.
    runtime_services: list[RuntimeServiceInfo] = Field(default_factory=list)
    stage_timings: list[StageTimingInfo] = Field(default_factory=list)
    quality_state: QualityStateInfo = Field(default_factory=QualityStateInfo)
    model_switch: SwitchStateInfo | None = None
    tracker_switch: SwitchStateInfo | None = None
    runtime_events: list[RuntimeEventInfo] = Field(default_factory=list)
    tracks: list[TrackInfo] = Field(default_factory=list)
    # Optional overlay payloads — populated only when the matching subsystem ran
    # this tick (faces a few Hz, pose for the single target).  The UI draws them
    # only when the operator enables the corresponding overlay toggle.
    faces: list[FaceBox] = Field(default_factory=list)
    pose: list[PoseKeypoint] = Field(default_factory=list)  # target subject only
    ptz: PTZState = Field(default_factory=PTZState)
    tracking_status: TrackingStatusInfo = Field(default_factory=TrackingStatusInfo)
    health: HealthInfo = Field(default_factory=HealthInfo)

    def to_msgpack(self) -> bytes:
        return bytes(msgpack.packb(self.model_dump(), use_bin_type=True))

    @classmethod
    def from_msgpack(cls, data: bytes) -> Self:
        raw: dict[str, Any] = msgpack.unpackb(data, raw=False)
        return cls(**raw)


# ── commands (UI → Engine) ────────────────────────────────────────────────────


class CmdKind(str, Enum):
    ADD_CAMERA = "add_camera"
    REMOVE_CAMERA = "remove_camera"
    UPDATE_CONFIG = "update_config"
    SET_TARGET = "set_target"
    SET_TARGET_IDENTITY = "set_target_identity"
    ENABLE_TRACKING = "enable_tracking"
    SET_TARGET_FPS = "set_target_fps"
    PTZ_NUDGE = "ptz_nudge"
    PTZ_GO_TO_PRESET = "ptz_go_to_preset"
    PTZ_SAVE_PRESET = "ptz_save_preset"
    PTZ_SAVE_PRESET_SLOT = "ptz_save_preset_slot"
    PTZ_RECALL_PRESET_SLOT = "ptz_recall_preset_slot"
    PTZ_HOME = "ptz_home"
    PTZ_MENU = "ptz_menu"
    SET_FEATURES = "set_features"
    ENROLL_IDENTITY = "enroll_identity"
    DELETE_IDENTITY = "delete_identity"
    RENAME_IDENTITY = "rename_identity"
    SET_LAYOUT = "set_layout"
    SAVE_LAYOUT = "save_layout"
    DELETE_LAYOUT = "delete_layout"


class BaseCommand(BaseModel):
    """All commands carry a stable UUID camera_id so the engine never uses
    global "current active" state."""

    cmd_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: CmdKind
    camera_id: str | None = None  # None for supervisor-level commands
    ts: float = Field(default_factory=time.time)

    def to_msgpack(self) -> bytes:
        return bytes(msgpack.packb(self.model_dump(), use_bin_type=True))

    @classmethod
    def from_msgpack(cls, data: bytes) -> Self:
        raw: dict[str, Any] = msgpack.unpackb(data, raw=False)
        return cls(**raw)


class AddCameraCmd(BaseCommand):
    kind: CmdKind = CmdKind.ADD_CAMERA
    source_uri: str = ""
    display_name: str = ""


class RemoveCameraCmd(BaseCommand):
    kind: CmdKind = CmdKind.REMOVE_CAMERA


class SetTargetCmd(BaseCommand):
    kind: CmdKind = CmdKind.SET_TARGET
    track_id: int | None = None
    identity: str | None = None  # named identity takes priority over track_id


class SetTargetIdentityCmd(BaseCommand):
    """Target a *named identity* on one camera ("track when found").

    The worker keeps the (single) target locked to whichever track is bound to
    ``identity_id`` whenever that identity is detected.  ``identity_id=None``
    clears identity targeting (manual / box targeting takes over).
    """

    kind: CmdKind = CmdKind.SET_TARGET_IDENTITY
    identity_id: str | None = None


class EnableTrackingCmd(BaseCommand):
    kind: CmdKind = CmdKind.ENABLE_TRACKING
    enabled: bool = True


class SetTargetFpsCmd(BaseCommand):
    """Change one camera's capture/detection pacing fps **live**.

    The worker re-paces its running ingest adapter (and detection cadence
    follows) without an engine restart, so lowering ``fps`` actually reduces
    capture/detect work.  Clamped to the source's detected hardware ceiling.
    """

    kind: CmdKind = CmdKind.SET_TARGET_FPS
    fps: float = 30.0


class SavePtzPresetCmd(BaseCommand):
    """Store the camera's current PTZ position into hardware preset ``slot``.

    Wired straight to the backend's ``save_preset(slot)``; a no-PTZ / ``None``
    backend makes it a safe no-op on the worker side.
    """

    kind: CmdKind = CmdKind.PTZ_SAVE_PRESET_SLOT
    slot: int = 0


class RecallPtzPresetCmd(BaseCommand):
    """Recall hardware PTZ preset ``slot`` on the camera.

    Wired straight to the backend's ``goto_preset(slot)``; a no-PTZ / ``None``
    backend makes it a safe no-op on the worker side.
    """

    kind: CmdKind = CmdKind.PTZ_RECALL_PRESET_SLOT
    slot: int = 0


class PtzNudgeCmd(BaseCommand):
    kind: CmdKind = CmdKind.PTZ_NUDGE
    pan_speed: float = 0.0
    tilt_speed: float = 0.0
    zoom_speed: float = 0.0


class PtzHomeCmd(BaseCommand):
    """Send the camera to its home position (backend ``home()``; safe no-op when
    the backend can't home)."""

    kind: CmdKind = CmdKind.PTZ_HOME


class PtzMenuCmd(BaseCommand):
    """Toggle the camera's on-screen (OSD) menu (backend ``osd_menu()``; safe
    no-op when unsupported)."""

    kind: CmdKind = CmdKind.PTZ_MENU


class SetFeaturesCmd(BaseCommand):
    """Global ML-subsystem on/off switches (broadcast: ``camera_id`` is None).

    ``features`` carries the boolean flags keyed by: ``detection``, ``tracking``,
    ``face_recognition``, ``pose``.  The supervisor applies it to every worker so
    heavy subsystems can be disabled live for performance.
    """

    kind: CmdKind = CmdKind.SET_FEATURES
    features: dict[str, bool] = Field(default_factory=dict)


class PtzGoToPresetCmd(BaseCommand):
    kind: CmdKind = CmdKind.PTZ_GO_TO_PRESET
    preset_name: str = ""


class PtzSavePresetCmd(BaseCommand):
    kind: CmdKind = CmdKind.PTZ_SAVE_PRESET
    preset_name: str = ""


class UpdateCameraConfigCmd(BaseCommand):
    kind: CmdKind = CmdKind.UPDATE_CONFIG
    config: dict[str, Any] = Field(default_factory=dict)


class EnrollIdentityCmd(BaseCommand):
    kind: CmdKind = CmdKind.ENROLL_IDENTITY
    identity_name: str = ""
    identity_id: str = ""  # pre-allocated by UI so store and engine share the same key
    track_id: int | None = None
    # Optional normalized frame-space click point (0..1). When present, the
    # worker enrolls the face nearest this exact click instead of whichever face
    # first maps to the track.
    click_x: float | None = None
    click_y: float | None = None


class DeleteIdentityCmd(BaseCommand):
    kind: CmdKind = CmdKind.DELETE_IDENTITY
    identity_id: str = ""


class RenameIdentityCmd(BaseCommand):
    kind: CmdKind = CmdKind.RENAME_IDENTITY
    identity_id: str = ""
    new_name: str = ""


class SetLayoutCmd(BaseCommand):
    kind: CmdKind = CmdKind.SET_LAYOUT
    layout_name: str = ""


class SaveLayoutCmd(BaseCommand):
    kind: CmdKind = CmdKind.SAVE_LAYOUT
    layout_name: str = ""
    tiles: list[dict[str, Any]] = Field(default_factory=list)


class DeleteLayoutCmd(BaseCommand):
    kind: CmdKind = CmdKind.DELETE_LAYOUT
    layout_id: str = ""


AnyCommand = (
    AddCameraCmd
    | RemoveCameraCmd
    | UpdateCameraConfigCmd
    | SetTargetCmd
    | SetTargetIdentityCmd
    | EnableTrackingCmd
    | SetTargetFpsCmd
    | PtzNudgeCmd
    | PtzGoToPresetCmd
    | PtzSavePresetCmd
    | SavePtzPresetCmd
    | RecallPtzPresetCmd
    | PtzHomeCmd
    | PtzMenuCmd
    | SetFeaturesCmd
    | EnrollIdentityCmd
    | DeleteIdentityCmd
    | RenameIdentityCmd
    | SetLayoutCmd
    | SaveLayoutCmd
    | DeleteLayoutCmd
    | BaseCommand
)
