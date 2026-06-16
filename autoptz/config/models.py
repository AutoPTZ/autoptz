"""Pydantic config models for AutoPTZ v2.

All config objects are immutable (frozen) pydantic models so they can be
passed safely between threads and processes without defensive copying.

Cameras are addressed by stable UUID (`CameraConfig.id`) everywhere — never
by list position or a "current active" global.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ── Helpers ───────────────────────────────────────────────────────────────────

def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ── Hardware / EP prefs ───────────────────────────────────────────────────────

class HardwarePrefs(BaseModel, frozen=True):
    force_ep: str | None = None        # "CoreMLExecutionProvider", etc.; None = auto
    model_tier: Literal["nano", "small", "medium", "large"] = "nano"
    max_workers: int = Field(default=4, ge=1, le=32)


# ── Theme ─────────────────────────────────────────────────────────────────────

class ThemeConfig(BaseModel, frozen=True):
    name: Literal["dark", "light", "system"] = "dark"
    accent: str = "#3d9bff"            # hex colour token


# ── Source ────────────────────────────────────────────────────────────────────

class SourceConfig(BaseModel, frozen=True):
    type: Literal["usb", "rtsp", "onvif", "ndi"] = "usb"
    address: str = ""                  # index (USB), URL (RTSP/ONVIF), name (NDI)
    username: str = ""
    password: str = ""
    substream: bool = False
    fps: float = Field(default=30.0, gt=0.0, le=240.0)


# ── Reconnect ─────────────────────────────────────────────────────────────────

class ReconnectConfig(BaseModel, frozen=True):
    backoff_initial_s: float = Field(default=1.0, gt=0.0)
    backoff_max_s: float = Field(default=30.0, gt=0.0)
    stall_timeout_s: float = Field(default=5.0, gt=0.0)


# ── Tracking ──────────────────────────────────────────────────────────────────

class TrackingConfig(BaseModel, frozen=True):
    tracker: Literal["botsort", "deepocsort", "bytetrack"] = "botsort"
    detect_interval: int = Field(default=1, ge=1, le=30)
    reid_enabled: bool = False
    reid_threshold_hi: float = Field(default=0.70, ge=0.0, le=1.0)
    reid_threshold_lo: float = Field(default=0.45, ge=0.0, le=1.0)
    coast_window_ms: int = Field(default=1500, ge=0)
    face_confirm: bool = False
    quality_floor: Literal["auto", "high", "balanced", "low"] = "auto"


# ── PTZ ───────────────────────────────────────────────────────────────────────

class PanTiltZoomLimits(BaseModel, frozen=True):
    pan_min: float = -1.0
    pan_max: float = 1.0
    tilt_min: float = -1.0
    tilt_max: float = 1.0
    zoom_min: float = 0.0
    zoom_max: float = 1.0


class PTZConfig(BaseModel, frozen=True):
    backend: Literal["auto", "ndi", "visca_ip", "visca_usb", "onvif"] = "auto"
    address: str | None = None
    max_pan_speed: float = Field(default=0.5, ge=0.0, le=1.0)
    max_tilt_speed: float = Field(default=0.5, ge=0.0, le=1.0)
    max_zoom_speed: float = Field(default=0.3, ge=0.0, le=1.0)
    invert_pan: bool = False
    invert_tilt: bool = False
    deadzone_x: float = Field(default=0.05, ge=0.0, le=0.5)
    deadzone_y: float = Field(default=0.05, ge=0.0, le=0.5)
    kp: float = Field(default=0.6, ge=0.0)
    kd: float = Field(default=0.05, ge=0.0)
    kv: float = Field(default=0.1, ge=0.0)
    auto_zoom: bool = True
    zoom_framing: Literal["tight", "medium", "wide"] = "medium"
    soft_limits: PanTiltZoomLimits | None = None


class PTZPreset(BaseModel, frozen=True):
    id: str = Field(default_factory=_new_id)
    camera_id: str = ""
    idx: int
    name: str
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    native_preset: int | None = None

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("PTZPreset.name must not be blank")
        return v


# ── Target / framing intent ───────────────────────────────────────────────────

class TargetConfig(BaseModel, frozen=True):
    mode: Literal["identity", "manual", "off"] = "off"
    identity_id: str | None = None
    default_on_start: int | None = None   # preset idx to recall on startup


# ── Camera ────────────────────────────────────────────────────────────────────

class CameraConfig(BaseModel, frozen=True):
    id: str = Field(default_factory=_new_id)
    name: str
    source: SourceConfig = Field(default_factory=SourceConfig)
    enabled: bool = True
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    ptz: PTZConfig = Field(default_factory=PTZConfig)
    presets: list[PTZPreset] = Field(default_factory=list)
    target: TargetConfig = Field(default_factory=TargetConfig)
    reconnect: ReconnectConfig = Field(default_factory=ReconnectConfig)

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("CameraConfig.name must not be blank")
        return v


# ── Layout ────────────────────────────────────────────────────────────────────

class TilePlacement(BaseModel, frozen=True):
    camera_id: str
    x: int = 0
    y: int = 0
    w: int = 1
    h: int = 1
    z: int = 0
    visible: bool = True


class Layout(BaseModel, frozen=True):
    id: str = Field(default_factory=_new_id)
    name: str
    tiles: list[TilePlacement] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Layout.name must not be blank")
        return v


# ── Identity ──────────────────────────────────────────────────────────────────

class IdentityRecord(BaseModel, frozen=True):
    id: str = Field(default_factory=_new_id)
    name: str
    embeddings: list[bytes] = Field(default_factory=list)
    thumbnail: bytes | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("IdentityRecord.name must not be blank")
        return v


# ── Top-level app config ──────────────────────────────────────────────────────

CURRENT_SCHEMA_VERSION = 1


class AppConfig(BaseModel, frozen=True):
    schema_version: int = CURRENT_SCHEMA_VERSION
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    active_layout_id: str = ""
    hardware: HardwarePrefs = Field(default_factory=HardwarePrefs)
    cameras: list[CameraConfig] = Field(default_factory=list)

    def with_camera(self, cam: CameraConfig) -> AppConfig:
        """Return a new AppConfig with *cam* upserted (replace by id or append)."""
        cameras = [c for c in self.cameras if c.id != cam.id] + [cam]
        return self.model_copy(update={"cameras": cameras})

    def without_camera(self, camera_id: str) -> AppConfig:
        """Return a new AppConfig with the given camera removed."""
        return self.model_copy(
            update={"cameras": [c for c in self.cameras if c.id != camera_id]}
        )
