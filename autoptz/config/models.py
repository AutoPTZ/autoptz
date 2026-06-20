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

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Helpers ───────────────────────────────────────────────────────────────────


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# ── Hardware / EP prefs ───────────────────────────────────────────────────────


class HardwarePrefs(BaseModel, frozen=True):
    force_ep: str | None = None  # "CoreMLExecutionProvider", etc.; None = auto
    model_tier: Literal["nano", "small", "medium", "large"] = "nano"
    max_workers: int = Field(default=4, ge=1, le=32)
    # Inference precision. "auto" lets each accelerator pick (GPU EPs use FP16);
    # "fp32"/"fp16" force it. CPU always runs FP32.
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    # Cap ORT intra-op threads per camera worker (None = auto: cores ÷ cameras, so
    # several cameras don't oversubscribe the CPU). Advanced override.
    intra_op_threads: int | None = Field(default=None, ge=1, le=256)


# ── Theme ─────────────────────────────────────────────────────────────────────


class ThemeConfig(BaseModel, frozen=True):
    name: Literal["dark", "light", "system"] = "dark"
    accent: str = "#3d9bff"  # hex colour token


# ── Source ────────────────────────────────────────────────────────────────────


class SourceConfig(BaseModel, frozen=True):
    type: Literal["usb", "rtsp", "onvif", "ndi"] = "usb"
    address: str = ""  # index (USB), URL (RTSP/ONVIF), name (NDI)
    unique_id: str | None = None  # stable device id (USB: AVFoundation uniqueID)
    source_label: str = ""  # friendly kind ("Built-in"/"Continuity Camera"/…)
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
    # User-facing target-retention policy and the single per-camera ReID control.
    # ``stable`` uses appearance ReID (gated by the global "reid" feature) to hold
    # the selected target through occlusions/crossings; ``responsive`` follows the
    # freshest track with no ReID hold and less delay.  (Replaces the old separate
    # ``reid_enabled`` flag, which contradicted this mode; legacy configs that
    # still carry ``reid_enabled`` simply ignore it.)
    tracking_mode: Literal["stable", "responsive"] = "stable"
    detect_interval: int = Field(default=1, ge=1, le=30)
    # Appearance (OSNet) re-acquisition thresholds.  ``hi`` enters a recovery
    # lock, ``lo`` maintains it (hysteresis).  Tuned down from the original
    # 0.70/0.45 — body crops rarely clear 0.70, so recovery used to never fire;
    # 0.60/0.35 re-binds the right person while still rejecting interlopers.
    reid_threshold_hi: float = Field(default=0.60, ge=0.0, le=1.0)
    reid_threshold_lo: float = Field(default=0.35, ge=0.0, le=1.0)
    coast_window_ms: int = Field(default=300, ge=0)
    face_confirm: bool = False
    quality_floor: Literal["auto", "high", "balanced", "low"] = "auto"
    # Which part of the detected person box the PTZ aims at (vertically). The
    # horizontal aim is always the box centre; this picks how high up the box the
    # vertical aim point sits, so the camera frames the face / upper body / whole
    # person.  See ``AIM_REGION_FRACTION`` for the actual fractions used by the
    # controller.  (True arm/leg exclusion would need pose keypoints; this is the
    # pragmatic bbox-fraction approximation.)
    aim_region: Literal["face", "head_shoulders", "upper_body", "full_body"] = "upper_body"
    # Unified "Framing" preset — the single user-facing control that drives BOTH
    # where the camera aims (``aim_region``) and how tightly it zooms
    # (``ptz.zoom_framing``).  The 4 names line up 1:1 with both legacy controls,
    # so consumers read ``framing`` directly and map through the matching constant
    # (``AIM_REGION_FRACTION`` for aim, the controller's zoom targets for zoom).
    framing: Literal["face", "head_shoulders", "upper_body", "full_body"] = "upper_body"
    # Whether pose-aware aiming ignores arms/limbs and tracks the torso, or uses
    # the whole detection silhouette. Torso is the default because it prevents an
    # extended arm from pulling the PTZ aim point away from the person.
    aim_body_mode: Literal["torso", "full_silhouette"] = "torso"
    # Ignore people whose detection-box HEIGHT is smaller than this fraction of the
    # frame height — drops distant specks so the engine doesn't chase/save every
    # far-away person.  0.0 disables the gate.
    min_detection_size_frac: float = Field(default=0.05, ge=0.0, le=1.0)


# Vertical aim point as a fraction of the person-box height measured from the TOP
# of the box (0.0 = top edge, 1.0 = bottom edge).  Consumed by the camera worker's
# ``_track_error`` to decide where inside the detection the PTZ should centre.
AIM_REGION_FRACTION: dict[str, float] = {
    "face": 0.10,  # head / face
    "head_shoulders": 0.22,
    "upper_body": 0.38,  # head + torso, robust to arm/leg motion (default)
    "full_body": 0.50,  # geometric centre (legacy behaviour)
}


# ── PTZ ───────────────────────────────────────────────────────────────────────


class PanTiltZoomLimits(BaseModel, frozen=True):
    pan_min: float = -1.0
    pan_max: float = 1.0
    tilt_min: float = -1.0
    tilt_max: float = 1.0
    zoom_min: float = 0.0
    zoom_max: float = 1.0


# Named auto-zoom framing presets (subject-height fractions live in the
# controller).  Legacy values "tight"/"medium"/"wide" are accepted and migrated
# to the nearest preset for backward compatibility with older stored configs.
ZoomFraming = Literal["face", "head_shoulders", "upper_body", "full_body", "wide"]

_LEGACY_ZOOM_FRAMING: dict[str, str] = {
    "tight": "head_shoulders",  # tightest legacy framing → head & shoulders
    "medium": "upper_body",  # legacy medium → upper body (≈0.45, same target)
    # "wide" maps to itself (still a valid value)
}


class PtzPresetSlot(BaseModel, frozen=True):
    """UI metadata for one quick-recall PTZ preset slot.

    The actual pan/tilt/zoom lives in the camera's hardware preset memory (driven
    by the backend ``save_preset``/``goto_preset``); this only carries what the UI
    shows in the PTZ section: a short ``label`` and a ``thumbnail`` — a base64 /
    data-URI snapshot of the camera view captured when the preset was saved
    (``None`` until a snapshot is taken).  ``thumbnail`` is a *string* (not raw
    bytes) so it round-trips cleanly through the JSON-serialised camera config.
    """

    label: str = ""
    thumbnail: str | None = None


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
    # Motion prediction: project the aim point forward by this many seconds using
    # the target's measured velocity, so the camera anticipates motion instead of
    # always chasing where the subject *was* (which reads as laggy following).
    lead_time_s: float = Field(default=0.15, ge=0.0, le=1.0)
    # Aim smoothing 0..1 (0 = most responsive, 1 = smoothest).  Maps to the
    # one-euro filter's minimum cutoff inside the controller; 0.5 ≈ the original.
    aim_smoothing: float = Field(default=0.5, ge=0.0, le=1.0)
    # Framing box: an adjustable rounded dead-zone around frame-centre.  While the
    # subject stays inside the box the PTZ holds still; the camera only moves to
    # keep them within it.  ``safe_zone_w`` / ``safe_zone_h`` are the box's
    # half-width / half-height as a fraction of the half-frame (0.15 ≈ 15% either
    # side of centre), drawn as a draggable, resizable overlay on the tile.  On by
    # default so new cameras show the framing region; the legacy single-radius
    # circle (``safe_zone_radius``) is migrated into the box by the validator.
    safe_zone_enabled: bool = True
    safe_zone_x: float = Field(default=0.0, ge=-0.9, le=0.9)
    safe_zone_y: float = Field(default=0.0, ge=-0.9, le=0.9)
    safe_zone_w: float = Field(default=0.15, ge=0.03, le=0.9)
    safe_zone_h: float = Field(default=0.22, ge=0.03, le=0.9)
    # Corner roundness of the framing box, 0 = sharp rectangle … 1 = full oval.
    # Defaults to a full oval (the framing region reads as a soft ellipse).
    safe_zone_roundness: float = Field(default=1.0, ge=0.0, le=1.0)
    safe_zone_radius: float = Field(default=0.12, ge=0.02, le=0.6)
    # Loss recovery: when the target is lost past the coast window, gently zoom
    # OUT (this speed) for up to ``reacquire_window_s`` to widen the view and
    # re-find the subject, instead of just stopping dead.
    loss_zoom_out: float = Field(default=0.25, ge=0.0, le=1.0)
    reacquire_window_s: float = Field(default=4.0, ge=0.0, le=30.0)
    auto_zoom: bool = True
    zoom_framing: ZoomFraming = "upper_body"
    soft_limits: PanTiltZoomLimits | None = None
    # Quick-recall hardware preset slots, shown in the Properties → PTZ section as
    # label + snapshot tiles.  Maps a slot index (0-based, 0..5) to a
    # :class:`PtzPresetSlot` (label + thumbnail); an absent slot is "empty" (no
    # preset saved).  This is UI metadata only — the actual position lives in the
    # camera's PTZ hardware preset memory, which the backend ``save_preset(slot)``
    # / ``goto_preset(slot)`` drive.  Round-trips through JSON (keys are coerced
    # back to ``int`` on load; legacy plain-string labels are migrated).
    preset_slots: dict[int, PtzPresetSlot] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _seed_box_from_radius(cls, data: object) -> object:
        """Seed the framing-box half-extents from the legacy circle radius.

        Configs saved before the box fields existed carry only
        ``safe_zone_radius``; map it onto ``safe_zone_w``/``safe_zone_h`` (a
        square box of the same size) so prior tuning carries over.  Newer configs
        that already specify the box keep their own values.
        """
        if isinstance(data, dict):
            r = data.get("safe_zone_radius")
            if r is not None:
                data.setdefault("safe_zone_w", r)
                data.setdefault("safe_zone_h", r)
        return data

    @field_validator("zoom_framing", mode="before")
    @classmethod
    def _migrate_legacy_framing(cls, v: object) -> object:
        """Accept legacy ``tight``/``medium`` and migrate to a named preset."""
        if isinstance(v, str):
            return _LEGACY_ZOOM_FRAMING.get(v, v)
        return v

    @field_validator("preset_slots", mode="before")
    @classmethod
    def _coerce_preset_slots(cls, v: object) -> object:
        """Coerce slot keys back to ``int`` and values to :class:`PtzPresetSlot`.

        Accepts the new shape (``{slot: {label, thumbnail}}``), an already-built
        ``PtzPresetSlot``, and the legacy ``{slot: "label"}`` plain-string form
        (migrated to ``PtzPresetSlot(label=...)``).  JSON has no int keys, hence
        the key coercion.
        """
        if isinstance(v, dict):
            out: dict[int, PtzPresetSlot] = {}
            for key, val in v.items():
                try:
                    k = int(key)
                except (TypeError, ValueError):
                    continue
                if isinstance(val, PtzPresetSlot):
                    out[k] = val
                elif isinstance(val, str):
                    out[k] = PtzPresetSlot(label=val)
                elif isinstance(val, dict):
                    out[k] = PtzPresetSlot(
                        label=str(val.get("label", "")),
                        thumbnail=val.get("thumbnail"),
                    )
            return out
        return v


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
    default_on_start: int | None = None  # preset idx to recall on startup


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
    # The chosen profile photo (one of ``thumbnails``, or a standalone crop).
    thumbnail: bytes | None = None
    # Candidate face crops captured during recognition; the user picks one as the
    # profile photo when registering, and can re-pick later.  Persisted alongside
    # the identity (see ``identity_photos`` in the store).
    thumbnails: list[bytes] = Field(default_factory=list)
    # Whether the engine should actively match/follow this identity.  Labeled
    # (named) identities default enabled; auto-harvested ones are created
    # disabled until a human names them.
    enabled: bool = True
    # False = auto-harvested/unlabeled (an in-memory "Person N" awaiting a name).
    # Retention policy: only *labeled* identities are persisted to the DB.
    labeled: bool = True
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("IdentityRecord.name must not be blank")
        return v


# ── Top-level app config ──────────────────────────────────────────────────────

CURRENT_SCHEMA_VERSION = 4


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
        return self.model_copy(update={"cameras": [c for c in self.cameras if c.id != camera_id]})
