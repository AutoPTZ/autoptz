"""Dead-time (Smith-predictor) model for PTZ follow.

Why
---
A network PTZ shows a velocity command in its *video* only after a dead time:
encoder + transport + decode + the head's own ramp. On a live NDI PTZ this was
measured at ~0.5-0.7 s, while the control loop only knew the ~50 ms local
pipeline latency. A P/D loop that can't see its own last ~0.6 s of motion keeps
driving after the subject is centred, swings past, and rings (observed: tilt
oscillating with a ~2-2.5 s period).

What
----
The Smith predictor fixes this without slowing the loop down: the error measured
in a frame is corrected by the camera motion that has been *commanded but is not
yet visible* in that frame::

    predicted_error = measured_error + ∫ image_velocity(cmd(τ)) dτ  over  [capture_t - L, now]

``L`` (dead time) and ``G`` (error-space units/s per unit command, per axis —
changes with zoom) are identified online from what the worker already has:
the commands it sends and the camera motion it *observes* via background
optical flow (ego-motion). ``L`` is the lag maximising the fit of
measured motion against the delayed command. The velocity model is proportional
to the command, with an optional signed minimum motor speed fitted when the
camera moves at a similar speed across a range of small commands.

Until the fit is confident (enough excitation, good R²) the model returns zero
correction, i.e. today's behaviour; the correction is also clamped so a bad
estimate can't fling the aim.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Search range for the dead time (s) and its resolution.
_LAG_MIN_S = 0.0
_LAG_MAX_S = 1.2
_LAG_STEP_S = 0.05
_WINDOW_S = 30.0  # how much history the fit uses
_MIN_MOVING = 15  # observations with a non-trivial command at the lag, per axis
_MIN_CMD = 0.05  # |command| considered "moving"
_MIN_R2 = 0.35  # fit quality required to trust the model
_REFIT_EVERY_S = 1.0
_MIN_GAIN = 0.1  # reject fits to nearly stationary video
_MAX_GAIN = 8.0


@dataclass(frozen=True)
class DeadTimeEstimate:
    delay_s: float = 0.0
    gain_pan: float = 0.0
    gain_tilt: float = 0.0
    r2: float = 0.0
    confident: bool = False
    floor_pan: float = 0.0
    floor_tilt: float = 0.0


def _axis_speed(cmd: float, gain: float, floor: float) -> float:
    return gain * cmd + (floor * math.copysign(1, cmd) if abs(cmd) > 0.005 else 0.0)


class DeadTimeModel:
    """Online dead-time + gain identification and pending-motion prediction.

    Thread-safe; all timestamps are on one monotonic clock.
    """

    MAX_PENDING = 1.0  # clamp on the correction, error-space units (half-frames)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Commands are piecewise-constant (the head holds the last velocity).
        self._cmd_t: deque[float] = deque()
        self._cmd: deque[tuple[float, float]] = deque()
        self._obs: deque[tuple[float, float, float]] = deque()  # (t, vx, vy)
        self._blank_until = -1.0
        self._est = DeadTimeEstimate()
        self._last_fit_t = -1e9

    # ── inputs ──────────────────────────────────────────────────────────────

    def record_command(self, t: float, pan: float, tilt: float) -> None:
        """A pan/tilt velocity command was sent at ``t`` (as sent to the head)."""
        with self._lock:
            if self._cmd and self._cmd[-1] == (pan, tilt):
                return
            if self._cmd_t and t < self._cmd_t[-1]:
                return  # out of order — ignore
            self._cmd_t.append(float(t))
            self._cmd.append((float(pan), float(tilt)))
            self._trim(t)

    def observe_motion(self, t: float, vx: float, vy: float) -> None:
        """Camera motion measured from the video (ego-motion flow) around ``t``."""
        with self._lock:
            if t < self._blank_until:
                return
            self._obs.append((float(t), float(vx), float(vy)))
            self._trim(t)

    def note_uncommanded_motion(self, t: float, hold_s: float = 3.0) -> None:
        """The head is moving on its own (preset recall / home): don't learn from it."""
        with self._lock:
            self._blank_until = max(self._blank_until, t + hold_s)
            self._obs.clear()
            self._cmd_t.clear()
            self._cmd.clear()
            self._est = DeadTimeEstimate()

    # ── outputs ─────────────────────────────────────────────────────────────

    def estimate(self, now: float) -> DeadTimeEstimate:
        """Current fit; refits at most every ``_REFIT_EVERY_S``."""
        with self._lock:
            if now - self._last_fit_t >= _REFIT_EVERY_S:
                self._last_fit_t = now
                self._trim(now)
                self._est = self._fit()
            return self._est

    def pending_motion(self, capture_t: float, now: float) -> tuple[float, float]:
        """Commanded camera motion not yet visible in a frame captured at ``capture_t``.

        Add this to the error measured in that frame. Zero until confident.
        """
        est = self.estimate(now)
        if not est.confident:
            return (0.0, 0.0)
        with self._lock:
            ip, it = self._integrate(capture_t - est.delay_s, now, est)
        m = self.MAX_PENDING
        px = max(-m, min(m, ip))
        py = max(-m, min(m, it))
        return (px, py)

    # ── internals ───────────────────────────────────────────────────────────

    def _trim(self, now: float) -> None:
        horizon = now - _WINDOW_S - _LAG_MAX_S
        # Keep one command at/before the horizon so lookups stay defined.
        while len(self._cmd_t) > 1 and self._cmd_t[1] <= horizon:
            self._cmd_t.popleft()
            self._cmd.popleft()
        while self._obs and self._obs[0][0] < now - _WINDOW_S:
            self._obs.popleft()

    def _integrate(self, a: float, b: float, est: DeadTimeEstimate) -> tuple[float, float]:
        """Integrate predicted image velocity over the piecewise-constant command history."""
        if b <= a or not self._cmd_t:
            return (0.0, 0.0)
        times = list(self._cmd_t)
        cmds = list(self._cmd)
        sp = st = 0.0
        i = max(0, bisect.bisect_right(times, a) - 1)
        while i < len(times):
            seg_a = max(a, times[i])
            seg_b = min(b, times[i + 1] if i + 1 < len(times) else b)
            if seg_b > seg_a and times[i] <= b:
                pan, tilt = cmds[i]

                sp += _axis_speed(pan, est.gain_pan, est.floor_pan) * (seg_b - seg_a)
                st += _axis_speed(tilt, est.gain_tilt, est.floor_tilt) * (seg_b - seg_a)
            if i + 1 < len(times) and times[i + 1] >= b:
                break
            i += 1
        return (sp, st)

    def _fit(self) -> DeadTimeEstimate:
        if len(self._obs) < _MIN_MOVING or not self._cmd_t:
            return DeadTimeEstimate()
        obs = np.asarray(self._obs)
        times = np.asarray(self._cmd_t)
        commands = np.asarray(self._cmd)
        best = DeadTimeEstimate()
        for lag in np.arange(_LAG_MIN_S, _LAG_MAX_S + _LAG_STEP_S / 2, _LAG_STEP_S):
            indices = np.searchsorted(times, obs[:, 0] - lag, side="right") - 1
            valid = indices >= 0
            if valid.sum() < _MIN_MOVING:
                continue
            pan_gain, pan_floor, pan_score = _fit_axis(commands[indices[valid], 0], obs[valid, 1])
            tilt_gain, tilt_floor, tilt_score = _fit_axis(
                commands[indices[valid], 1], obs[valid, 2]
            )
            score = max(pan_score, tilt_score)
            if score > best.r2:
                best = DeadTimeEstimate(
                    float(lag), pan_gain, tilt_gain, score, True, pan_floor, tilt_floor
                )
        return best


def _fit_axis(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float, float]:
    """Fit proportional gain and minimum motor speed to the aligned command and flow."""
    if np.count_nonzero(np.abs(x) >= _MIN_CMD) < _MIN_MOVING or np.ptp(x) < _MIN_CMD:
        return 0.0, 0.0, 0.0
    variance = float(np.sum((y - y.mean()) ** 2))
    if variance < 1e-8:
        return 0.0, 0.0, 0.0
    offset = 0.0
    gain = float(np.sum(x * y) / np.sum(x * x))
    residual = y - gain * x
    # Some heads quantise even small commands to a minimum motor speed.
    # Fit that offset only with observations at distinct speeds and a
    # material reduction in residuals over the proportional model.
    moving = np.abs(x) > 0.005
    amps = np.abs(x[moving])
    if amps.size >= _MIN_MOVING and amps.std() > 0.015:
        signed = y[moving] * np.sign(x[moving])
        slope = float(np.mean((amps - amps.mean()) * (signed - signed.mean())) / amps.var())
        slope = math.copysign(max(0.0, slope * math.copysign(1, gain)), gain)
        floor = float(np.mean(signed - slope * amps))
        candidate = y - slope * x - floor * np.sign(x) * moving
        if floor * gain > 0 and np.sum(candidate**2) < 0.85 * np.sum(residual**2):
            offset = floor
            gain, residual = slope, candidate
    score = 1.0 - float(np.sum(residual * residual)) / variance
    if (
        score >= _MIN_R2
        and (_MIN_GAIN <= abs(gain) or abs(offset) > 0.02)
        and abs(gain) <= _MAX_GAIN
        and abs(offset) <= _MAX_GAIN
    ):
        return gain, offset, score
    return 0.0, 0.0, 0.0
