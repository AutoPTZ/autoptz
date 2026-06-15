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
    identity: str | None = None
    confidence: float = 0.0
    is_target: bool = False


class PTZState(BaseModel):
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    moving: bool = False
    backend: str = ""
    state: str = "idle"


class HealthState(str, Enum):
    OK = "ok"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    STOPPED = "stopped"


class HealthInfo(BaseModel):
    state: HealthState = HealthState.OK
    last_error: str | None = None


class TelemetryMsg(BaseModel):
    """Emitted by each CameraWorker at ~telemetry_hz (default 10 Hz)."""
    camera_id: str
    seq: int
    ts: float = Field(default_factory=time.time)
    fps: float = 0.0
    ep: str = ""  # active inference EP (e.g. "CoreMLExecutionProvider")
    tracks: list[TrackInfo] = Field(default_factory=list)
    ptz: PTZState = Field(default_factory=PTZState)
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
    ENABLE_TRACKING = "enable_tracking"
    PTZ_NUDGE = "ptz_nudge"
    PTZ_GO_TO_PRESET = "ptz_go_to_preset"
    PTZ_SAVE_PRESET = "ptz_save_preset"
    ENROLL_IDENTITY = "enroll_identity"
    SET_LAYOUT = "set_layout"


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


class EnableTrackingCmd(BaseCommand):
    kind: CmdKind = CmdKind.ENABLE_TRACKING
    enabled: bool = True


class PtzNudgeCmd(BaseCommand):
    kind: CmdKind = CmdKind.PTZ_NUDGE
    pan_speed: float = 0.0
    tilt_speed: float = 0.0
    zoom_speed: float = 0.0


class PtzGoToPresetCmd(BaseCommand):
    kind: CmdKind = CmdKind.PTZ_GO_TO_PRESET
    preset_name: str = ""


class PtzSavePresetCmd(BaseCommand):
    kind: CmdKind = CmdKind.PTZ_SAVE_PRESET
    preset_name: str = ""


class EnrollIdentityCmd(BaseCommand):
    kind: CmdKind = CmdKind.ENROLL_IDENTITY
    identity_name: str = ""
    track_id: int | None = None


class SetLayoutCmd(BaseCommand):
    kind: CmdKind = CmdKind.SET_LAYOUT
    layout_name: str = ""


AnyCommand = (
    AddCameraCmd
    | RemoveCameraCmd
    | SetTargetCmd
    | EnableTrackingCmd
    | PtzNudgeCmd
    | PtzGoToPresetCmd
    | PtzSavePresetCmd
    | EnrollIdentityCmd
    | SetLayoutCmd
    | BaseCommand
)
