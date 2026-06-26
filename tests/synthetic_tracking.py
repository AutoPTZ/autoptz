"""Deterministic synthetic-tracking helpers for the realtime/reliability suite.

Pure-Python scripted target generator + fake scripted detector + reusable test
fakes (MockBackend, config builder, mock tracker impl). No real model, no
boxmot, no hardware — every consumer drives the real Tracker / PTZController
public API against ground-truth boxes evaluated from a closed-form trajectory.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np

from autoptz.config.models import PTZConfig
from autoptz.engine.pipeline.detect import BBox, Detection
from autoptz.engine.ptz.base import PTZBackend, PTZCaps, PTZState

if TYPE_CHECKING:
    from numpy.typing import NDArray

# A blank BGR frame; the synthetic detector/tracker never read pixels (boxes are
# scripted), so a zero frame is sufficient and keeps tests allocation-cheap.
FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


# ── trajectory generators ──────────────────────────────────────────────────────


def sinusoid_centres(
    cx: float,
    amp: float,
    omega: float,
    frames: int,
    dt: float = 1.0,
    *,
    cy: float = 240.0,
) -> list[tuple[float, float]]:
    """Ground-truth centres x(t)=cx+amp*sin(omega*t), y(t)=cy at t=i*dt."""
    return [(cx + amp * math.sin(omega * (i * dt)), cy) for i in range(frames)]


def constant_velocity_centres(
    x0: float,
    vx: float,
    frames: int,
    dt: float = 1.0,
    *,
    cy: float = 240.0,
) -> list[tuple[float, float]]:
    """Ground-truth centres x(t)=x0+vx*t, y(t)=cy at t=i*dt."""
    return [(x0 + vx * (i * dt), cy) for i in range(frames)]


# ── box / detection builders ───────────────────────────────────────────────────


def box_at(cx: float, cy: float, w: float = 80.0, h: float = 200.0) -> BBox:
    """A fixed-size xyxy box centred on (cx, cy)."""
    return BBox(x1=cx - w / 2.0, y1=cy - h / 2.0, x2=cx + w / 2.0, y2=cy + h / 2.0)


def detections_for_centres(
    centres: list[tuple[float, float]],
    *,
    misses: frozenset[int] | set[int] = frozenset(),
    conf: float = 0.9,
    w: float = 80.0,
    h: float = 200.0,
) -> list[list[Detection]]:
    """One list[Detection] per frame; frames in *misses* yield [] (injected miss)."""
    out: list[list[Detection]] = []
    for i, (cx, cy) in enumerate(centres):
        if i in misses:
            out.append([])
        else:
            out.append([Detection(bbox=box_at(cx, cy, w, h), conf=conf, class_id=0)])
    return out


def tracker_rows_for_centres(
    centres: list[tuple[float, float]],
    *,
    track_id: int = 1,
    misses: frozenset[int] | set[int] = frozenset(),
    conf: float = 0.9,
    w: float = 80.0,
    h: float = 200.0,
) -> list[NDArray[np.float32]]:
    """Per-frame mock-tracker output rows [x1,y1,x2,y2,id,conf,cls] for side_effect."""
    rows: list[NDArray[np.float32]] = []
    for i, (cx, cy) in enumerate(centres):
        if i in misses:
            rows.append(np.empty((0, 7), dtype=np.float32))
            continue
        b = box_at(cx, cy, w, h)
        rows.append(
            np.array([[b.x1, b.y1, b.x2, b.y2, float(track_id), conf, 0.0]], dtype=np.float32)
        )
    return rows


def make_mock_impl(rows: list[NDArray[np.float32]]) -> MagicMock:
    """Mock BoxMOT impl whose update() returns each entry of *rows* in order."""
    impl = MagicMock()
    impl.update.side_effect = list(rows)
    return impl


# ── controller fakes ───────────────────────────────────────────────────────────


def make_cfg(**kw: object) -> PTZConfig:
    """Deterministic PTZConfig for unit tests (mirrors tests/test_ptz.py::_cfg)."""
    defaults: dict[str, object] = {
        "kp": 0.6,
        "kd": 0.0,
        "kv": 0.0,
        "deadzone_x": 0.0,
        "deadzone_y": 0.0,
        "max_pan_speed": 1.0,
        "max_tilt_speed": 1.0,
        "max_zoom_speed": 1.0,
        "auto_zoom": False,
        "max_accel": 0.0,
    }
    defaults.update(kw)
    return PTZConfig(**defaults)  # type: ignore[arg-type]


class MockBackend(PTZBackend):
    """Records move_velocity/stop calls; no hardware (copied from test_ptz.py)."""

    def __init__(self, has_position: bool = False) -> None:
        super().__init__()
        self.caps = PTZCaps(
            continuous_pan_tilt=True,
            continuous_zoom=True,
            native_presets=True,
        )
        self.velocity_calls: list[tuple[float, float, float]] = []
        self.stop_count: int = 0
        self._pos: PTZState | None = PTZState() if has_position else None

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        self.velocity_calls.append((pan, tilt, zoom))

    def stop(self) -> None:
        self.stop_count += 1

    def get_position(self) -> PTZState | None:
        return self._pos

    def goto_preset(self, idx: int) -> None:  # pragma: no cover - unused here
        pass

    def save_preset(self, idx: int) -> None:  # pragma: no cover - unused here
        pass

    def close(self) -> None:  # pragma: no cover - unused here
        pass
