"""Closed-loop PTZ motion controller.

Pipeline per tick:
  raw error → dead-zone → one-euro filter → PD + velocity feed-forward
            → clamp → response curve → backend.move_velocity()

  zoom: subject_height error → PD with hysteresis → backend zoom component

Coast-on-loss: when track_active goes False, hold last velocity for
coast_window_ms, then stop and enter SEARCHING state.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoptz.config.models import PTZConfig
    from autoptz.engine.ptz.base import PTZBackend

_ZOOM_FRAMING_TARGETS = {
    "tight": 0.65,
    "medium": 0.45,
    "wide": 0.25,
}
_ZOOM_HYSTERESIS = 0.05  # ±5 % of frame height before zoom moves
_POWER = 1.5             # response-curve exponent (ease-in: gentle near zero)


# ── one-euro filter ───────────────────────────────────────────────────────────


class OneEuroFilter:
    """Adaptive low-latency filter: low jitter on slow signals, low lag on fast ones.

    Reference: Casiez et al. 2012, "1€ Filter: A Simple Speed-based Low-pass
    Filter for Noisy Input in Interactive Systems".
    """

    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 0.007,
        dcutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x: float | None = None
        self._dx: float = 0.0
        self._last_t: float | None = None

    @staticmethod
    def _alpha(freq: float, cutoff: float) -> float:
        te = 1.0 / freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, t: float | None = None) -> float:
        if t is not None and self._last_t is not None and t > self._last_t:
            self.freq = 1.0 / (t - self._last_t)
        if t is not None:
            self._last_t = t

        if self._x is None:
            self._x = x
            return x

        # filtered derivative
        dx = (x - self._x) * self.freq
        alpha_d = self._alpha(self.freq, self.dcutoff)
        self._dx = alpha_d * dx + (1.0 - alpha_d) * self._dx

        # adaptive cutoff → filter value
        cutoff = self.mincutoff + self.beta * abs(self._dx)
        alpha = self._alpha(self.freq, cutoff)
        self._x = alpha * x + (1.0 - alpha) * self._x
        return self._x

    def reset(self, x: float = 0.0) -> None:
        self._x = None
        self._dx = 0.0
        self._last_t = None


# ── helpers ───────────────────────────────────────────────────────────────────


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _shape(x: float) -> float:
    """Non-linear response curve: ease-in (gentle near zero, full speed at ±1)."""
    return math.copysign(abs(x) ** _POWER, x) if x != 0.0 else 0.0


# ── controller state ──────────────────────────────────────────────────────────


class ControllerState(Enum):
    IDLE = auto()       # no target assigned
    TRACKING = auto()   # active PD loop
    COASTING = auto()   # target LOST; holding last velocity for coast_window
    SEARCHING = auto()  # coast expired; stopped and waiting for re-ID


# ── shared update payload ─────────────────────────────────────────────────────


@dataclass
class _TrackPayload:
    error: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    subject_height: float = 0.0
    track_active: bool = False
    seq: int = 0


# ── controller ────────────────────────────────────────────────────────────────


class PTZController:
    """Rate-limited closed-loop PTZ controller.

    Usage (threaded):
        ctrl = PTZController(backend, cfg)
        ctrl.start()
        # from inference thread:
        ctrl.update(error=(ex, ey), velocity=(vx, vy), subject_height=h, track_active=True)
        ...
        ctrl.stop()   # sends backend.stop() before returning

    Usage (synchronous / tests):
        pan, tilt, zoom = ctrl.step(error, velocity, subject_height, track_active, t=0.0)
    """

    def __init__(
        self,
        backend: PTZBackend,
        cfg: PTZConfig,
        *,
        coast_window_ms: int = 1500,
        rate_hz: float = 20.0,
    ) -> None:
        self._backend = backend
        self._cfg = cfg
        self._coast_window_s = coast_window_ms / 1000.0
        self._rate_hz = rate_hz

        # per-axis one-euro filters (freq seed = rate_hz)
        self._filt_ex = OneEuroFilter(freq=rate_hz, mincutoff=1.0, beta=0.01)
        self._filt_ey = OneEuroFilter(freq=rate_hz, mincutoff=1.0, beta=0.01)

        # controller state
        self._state = ControllerState.IDLE
        self._coast_start: float = 0.0
        self._coast_pan: float = 0.0
        self._coast_tilt: float = 0.0

        # PD derivative state
        self._prev_ex_f: float = 0.0
        self._prev_ey_f: float = 0.0
        self._last_t: float = -1.0

        # last command sent (for rate-limit suppression)
        self._last_pan: float = 0.0
        self._last_tilt: float = 0.0
        self._last_zoom: float = 0.0

        # thread-safe payload from inference thread
        self._payload = _TrackPayload()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> ControllerState:
        return self._state

    def update(
        self,
        error: tuple[float, float],
        velocity: tuple[float, float],
        subject_height: float,
        track_active: bool,
    ) -> None:
        """Called from the inference thread with the latest tracking data."""
        with self._lock:
            self._payload = _TrackPayload(
                error=error,
                velocity=velocity,
                subject_height=subject_height,
                track_active=track_active,
                seq=self._payload.seq + 1,
            )

    def step(
        self,
        error: tuple[float, float],
        velocity: tuple[float, float],
        subject_height: float,
        track_active: bool,
        t: float | None = None,
    ) -> tuple[float, float, float]:
        """Synchronous single tick (tests / manual drive).

        Updates internal state, calls backend, and returns (pan, tilt, zoom)."""
        with self._lock:
            self._payload = _TrackPayload(
                error=error,
                velocity=velocity,
                subject_height=subject_height,
                track_active=track_active,
                seq=self._payload.seq + 1,
            )
        return self._tick(t=t if t is not None else time.perf_counter())

    def goto_preset(self, idx: int) -> None:
        self._backend.goto_preset(idx)

    def save_preset(self, idx: int) -> None:
        self._backend.save_preset(idx)

    def start(self) -> None:
        """Start the background PTZ thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="ptz-ctrl", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and send stop to the backend."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._backend.stop()
        except Exception:
            pass

    def close(self) -> None:
        """Stop tracking and release backend resources."""
        self.stop()
        try:
            self._backend.close()
        except Exception:
            pass

    def __enter__(self) -> PTZController:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── background thread ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        interval = 1.0 / self._rate_hz
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self._tick(t=t0)
            except Exception:
                pass
            elapsed = time.perf_counter() - t0
            self._stop_event.wait(max(0.0, interval - elapsed))
        # guarantee stop on thread exit
        try:
            self._backend.stop()
        except Exception:
            pass

    # ── tick (core logic) ─────────────────────────────────────────────────────

    def _tick(self, t: float) -> tuple[float, float, float]:
        with self._lock:
            payload = self._payload

        pan_cmd, tilt_cmd, zoom_cmd = self._compute(payload, t)

        # only send if command changed meaningfully
        if (
            abs(pan_cmd - self._last_pan) > 1e-4
            or abs(tilt_cmd - self._last_tilt) > 1e-4
            or abs(zoom_cmd - self._last_zoom) > 1e-4
        ):
            self._backend.move_velocity(pan_cmd, tilt_cmd, zoom_cmd)
            self._last_pan = pan_cmd
            self._last_tilt = tilt_cmd
            self._last_zoom = zoom_cmd

        return pan_cmd, tilt_cmd, zoom_cmd

    def _compute(self, p: _TrackPayload, t: float) -> tuple[float, float, float]:
        cfg = self._cfg

        # ── state transitions ─────────────────────────────────────────────────
        if p.track_active:
            if self._state != ControllerState.TRACKING:
                # (re-)acquired target: reset filters
                self._filt_ex.reset()
                self._filt_ey.reset()
                self._prev_ex_f = 0.0
                self._prev_ey_f = 0.0
                self._last_t = -1.0
                self._state = ControllerState.TRACKING
        else:
            if self._state == ControllerState.TRACKING:
                # just lost target: enter coast
                self._state = ControllerState.COASTING
                self._coast_start = t
                self._coast_pan = self._last_pan
                self._coast_tilt = self._last_tilt
            elif self._state == ControllerState.COASTING:
                if t - self._coast_start >= self._coast_window_s:
                    self._state = ControllerState.SEARCHING
                    try:
                        self._backend.stop()
                    except Exception:
                        pass
            # IDLE / SEARCHING: nothing to do

        # ── compute commands per state ────────────────────────────────────────
        if self._state == ControllerState.TRACKING:
            pan_cmd, tilt_cmd = self._pd_step(p, t)
            zoom_cmd = self._zoom_step(p.subject_height) if cfg.auto_zoom else 0.0
        elif self._state == ControllerState.COASTING:
            pan_cmd, tilt_cmd = self._coast_pan, self._coast_tilt
            zoom_cmd = 0.0
        else:
            pan_cmd = tilt_cmd = zoom_cmd = 0.0

        return pan_cmd, tilt_cmd, zoom_cmd

    def _pd_step(self, p: _TrackPayload, t: float) -> tuple[float, float]:
        cfg = self._cfg
        ex, ey = p.error
        vx, vy = p.velocity

        # 1. dead-zone
        ex = ex if abs(ex) >= cfg.deadzone_x else 0.0
        ey = ey if abs(ey) >= cfg.deadzone_y else 0.0

        # 2. one-euro filter
        ex_f = self._filt_ex(ex, t)
        ey_f = self._filt_ey(ey, t)

        # 3. PD derivative (finite difference of filtered error)
        dt = (t - self._last_t) if self._last_t >= 0.0 else 0.0
        dt = max(dt, 1e-6)
        dex = (ex_f - self._prev_ex_f) / dt
        dey = (ey_f - self._prev_ey_f) / dt

        # 4. PD + velocity feed-forward
        pan_raw = cfg.kp * ex_f + cfg.kd * dex + cfg.kv * vx
        tilt_raw = cfg.kp * ey_f + cfg.kd * dey + cfg.kv * vy

        # 5. per-camera speed ceiling + clamp + response curve
        pan_cmd = _shape(_clamp(pan_raw * cfg.max_pan_speed, -1.0, 1.0))
        tilt_cmd = _shape(_clamp(tilt_raw * cfg.max_tilt_speed, -1.0, 1.0))

        # 6. soft limits (velocity: clamp to zero if at boundary)
        lim = cfg.soft_limits
        if lim is not None:
            if pan_cmd > 0 and self._backend.get_position() is not None:
                pos = self._backend.get_position()
                if pos is not None and pos.pan >= lim.pan_max:
                    pan_cmd = 0.0
            if pan_cmd < 0 and self._backend.get_position() is not None:
                pos = self._backend.get_position()
                if pos is not None and pos.pan <= lim.pan_min:
                    pan_cmd = 0.0

        # 7. invert
        if cfg.invert_pan:
            pan_cmd = -pan_cmd
        if cfg.invert_tilt:
            tilt_cmd = -tilt_cmd

        self._prev_ex_f = ex_f
        self._prev_ey_f = ey_f
        self._last_t = t

        return pan_cmd, tilt_cmd

    def _zoom_step(self, subject_height: float) -> float:
        cfg = self._cfg
        if subject_height <= 0.0:
            return 0.0
        target = _ZOOM_FRAMING_TARGETS.get(cfg.zoom_framing, 0.45)
        zoom_error = subject_height - target
        if zoom_error > _ZOOM_HYSTERESIS:
            # subject too tall → zoom out (negative)
            zoom_cmd = -_clamp(zoom_error * 2.0, 0.0, 1.0)
        elif zoom_error < -_ZOOM_HYSTERESIS:
            # subject too short → zoom in (positive)
            zoom_cmd = _clamp(-zoom_error * 2.0, 0.0, 1.0)
        else:
            zoom_cmd = 0.0
        return _clamp(zoom_cmd * cfg.max_zoom_speed, -1.0, 1.0)
