"""Digital ("Center Stage") PTZ backend: integrate velocity into a crop window.

The controller drives this exactly like a physical backend (``move_velocity``),
but instead of moving motors we move a normalized crop rectangle inside the
sensor frame. An output stage reads :meth:`crop_rect` each frame and crops/scales
to produce the auto-framed output — so any non-PTZ camera gains auto-framing.
"""

from __future__ import annotations

from autoptz.engine.ptz.base import PTZBackend, PTZCaps

# Per-call integration step when no explicit dt is given. The controller calls
# move_velocity once per inference frame, so a fixed nominal frame period keeps
# the motion deterministic (unit-testable) and independent of wall-clock jitter
# between rapid calls. Callers that know the real frame dt may pass it explicitly.
_NOMINAL_DT = 1.0 / 30.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class DigitalPTZBackend(PTZBackend):
    def __init__(self, min_crop_frac: float = 0.34, max_step_per_s: float = 1.6) -> None:
        super().__init__()
        self.caps = PTZCaps(
            continuous_pan_tilt=True,
            continuous_zoom=True,
            absolute_pan_tilt=True,
            absolute_zoom=True,
            native_presets=False,
            query_position=False,
        )
        self._min_crop = _clamp(min_crop_frac, 0.1, 1.0)
        self._rate = max(0.01, max_step_per_s)
        # Normalized state: pan/tilt in [-1,1] (fraction of the *available* travel),
        # zoom in [0,1] (0 = full frame, 1 = tightest crop).
        self._pan = 0.0
        self._tilt = 0.0
        self._zoom = 0.0
        self._presets: dict[int, tuple[float, float, float]] = {}

    def move_velocity(
        self, pan: float, tilt: float, zoom: float = 0.0, dt: float | None = None
    ) -> None:
        # Optional dt keeps the ABC-compatible 3-arg call working (controller path)
        # while letting callers/tests integrate a known step deterministically.
        step = self._rate * (_NOMINAL_DT if dt is None else max(0.0, dt))
        self._zoom = _clamp(self._zoom + zoom * step, 0.0, 1.0)
        self._pan = _clamp(self._pan + pan * step, -1.0, 1.0)
        self._tilt = _clamp(self._tilt + tilt * step, -1.0, 1.0)

    def move_absolute(self, pan: float, tilt: float, zoom: float) -> None:
        self._pan = _clamp(pan, -1.0, 1.0)
        self._tilt = _clamp(tilt, -1.0, 1.0)
        self._zoom = _clamp(zoom, 0.0, 1.0)

    def stop(self) -> None:
        return None  # state holds; nothing to halt

    def home(self) -> None:
        self._pan = self._tilt = self._zoom = 0.0

    def save_preset(self, idx: int) -> None:
        self._presets[idx] = (self._pan, self._tilt, self._zoom)

    def goto_preset(self, idx: int) -> None:
        if idx in self._presets:
            self._pan, self._tilt, self._zoom = self._presets[idx]

    def close(self) -> None:
        return None

    def crop_rect(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Current crop as integer ``(x, y, w, h)`` inside ``frame_w × frame_h``."""
        frac = 1.0 - self._zoom * (1.0 - self._min_crop)  # 1.0 → min_crop
        cw = max(1, int(round(frame_w * frac)))
        ch = max(1, int(round(frame_h * frac)))
        # Tilt is up-positive in controller space; image y grows downward → invert.
        free_x = frame_w - cw
        free_y = frame_h - ch
        cx = (free_x / 2.0) + self._pan * (free_x / 2.0)
        cy = (free_y / 2.0) - self._tilt * (free_y / 2.0)
        x = int(round(_clamp(cx, 0.0, float(free_x))))
        y = int(round(_clamp(cy, 0.0, float(free_y))))
        return (x, y, cw, ch)
