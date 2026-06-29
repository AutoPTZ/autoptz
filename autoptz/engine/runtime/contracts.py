"""Runtime contracts for the 2.2 reliability architecture.

These types are intentionally small and source-agnostic. Capture adapters,
benchmark sources, schedulers, and the tracker/control loop should move toward
passing these contracts instead of backend-specific ad hoc tuples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RuntimeMode(str, Enum):
    """Supported runtime modes for the 2.2 reliability work."""

    PRODUCTION_SHARED_MODEL = "production_shared_model"
    LABS_MODEL_SERVER = "labs_model_server"

    @classmethod
    def from_env(cls) -> RuntimeMode:
        """Resolve the runtime mode from explicit labs/developer env flags."""
        if os.environ.get("AUTOPTZ_MODEL_SERVER", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return cls.LABS_MODEL_SERVER
        return cls.PRODUCTION_SHARED_MODEL


class TargetState(str, Enum):
    """Tracking/control states consumed by PTZ in the simplified follow path."""

    ACQUIRE = "acquire"
    TRACK = "track"
    HOLD = "hold"
    LOST = "lost"
    REACQUIRE = "reacquire"


@dataclass(frozen=True)
class SourceHealth:
    """Source-delivery counters independent of the source backend."""

    source_fps: float = 0.0
    delivered_fps: float = 0.0
    duplicate_frames: int = 0
    stale_frames: int = 0
    conversion_ms: float = 0.0
    app_induced_drops: int = 0
    backend_counters: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def app_drop_free(self) -> bool:
        return self.app_induced_drops == 0


@dataclass(frozen=True)
class FramePacket:
    """One latest-wins capture packet moving from ingest to downstream stages."""

    frame: NDArray[np.uint8]
    sequence: int
    source_ts: float | None
    capture_ts: float
    pixel_format: str
    width: int
    height: int
    health: SourceHealth = field(default_factory=SourceHealth)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("FramePacket.sequence must be non-negative")
        if self.capture_ts <= 0:
            raise ValueError("FramePacket.capture_ts must be positive")
        if self.source_ts is not None and self.source_ts < 0:
            raise ValueError("FramePacket.source_ts must be non-negative when present")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("FramePacket dimensions must be positive")
        if len(self.frame.shape) < 2:
            raise ValueError("FramePacket.frame must have height and width dimensions")
        h, w = int(self.frame.shape[0]), int(self.frame.shape[1])
        if (w, h) != (self.width, self.height):
            raise ValueError(
                f"FramePacket dimensions {self.width}x{self.height} do not match "
                f"frame shape {w}x{h}"
            )
        if not self.pixel_format.strip():
            raise ValueError("FramePacket.pixel_format must be present")

    @classmethod
    def from_frame(
        cls,
        frame: NDArray[np.uint8],
        *,
        sequence: int,
        capture_ts: float,
        source_ts: float | None = None,
        pixel_format: str = "bgr",
        health: SourceHealth | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FramePacket:
        h, w = int(frame.shape[0]), int(frame.shape[1])
        return cls(
            frame=frame,
            sequence=sequence,
            source_ts=source_ts,
            capture_ts=capture_ts,
            pixel_format=pixel_format,
            width=w,
            height=h,
            health=health or SourceHealth(),
            metadata=metadata or {},
        )
