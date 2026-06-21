"""Ego-motion estimation — separate camera motion from subject motion.

Why
---
The PTZ control loop derives the subject's velocity from the frame-to-frame
change in its aim error.  When the camera itself pans/tilts, *every* pixel
shifts, so a stationary subject appears to move — and the loop's velocity
feed-forward chases that phantom motion, which reads as hunting/oscillation
whenever the subject and the camera move at the same time.

This estimator measures the **global image-plane motion** the camera induced
and returns it in the controller's *error space* (normalised, right-positive x,
up-positive y, units/second) so the caller can simply subtract it from the
measured aim velocity to recover the subject's **world** motion.

How
---
- **Measured flow (primary):** sparse Lucas-Kanade optical flow on a small
  greyscale copy of the frame, with the tracked person boxes masked out so the
  subject's own motion does not bias the estimate.  The robust (MAD-filtered)
  median flow of the background features is the camera's image motion.  This
  works for *any* camera motion — auto follow, manual nudges, preset recalls —
  because it observes the result, not the command.
- **Commanded model (fallback + disambiguation):** map the last PTZ command
  (normalised pan/tilt) to an expected image velocity through a per-camera gain
  that is *learned online* by regressing measured flow against the command, so
  it self-calibrates to the camera's FOV/slew without hard-coded optics.  Used
  when flow is unreliable (low-texture scenes, too few background features).

Dependencies: numpy, opencv-contrib-python (both already required).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EgoMotion:
    """Camera-induced image motion, expressed in controller error space.

    ``vx`` / ``vy`` are normalised units per second, matching ``_track_error``'s
    convention (x>0 → content moved right of centre; y>0 → content moved up).
    Subtract these from a measured aim-error velocity to get the subject's own
    (world-referenced) velocity.
    """

    vx: float = 0.0
    vy: float = 0.0
    source: str = "none"  # "flow" | "command" | "none"
    confidence: float = 0.0  # 0..1 — how much to trust this estimate

    @property
    def is_moving(self) -> bool:
        return self.source != "none" and (abs(self.vx) > 1e-4 or abs(self.vy) > 1e-4)


# Tunables kept module-level so they read like the rest of the pipeline and are
# easy to sweep in tests; the public knobs (enable, gain bounds) live on config.
_PROC_WIDTH = 320  # downscaled working width for flow (px); cheap + plenty
_MAX_FEATURES = 200
_MIN_INLIERS = 12  # below this, flow is untrustworthy → fall back to command
_BOX_PAD_FRAC = 0.08  # grow masked person boxes slightly to cover soft edges
_CONF_FULL_INLIERS = 80  # inlier count that saturates the flow confidence
_FLOW_TRUST = 0.35  # confidence at/above which flow is used over command


class EgoMotionEstimator:
    """Estimate per-tick camera ego-motion in controller error space.

    Stateful: it keeps the previous greyscale frame (for flow), the previous
    timestamp, and the online-learned command→image gain.  One instance per
    camera worker; call :meth:`reset` on a source change.
    """

    def __init__(
        self,
        *,
        proc_width: int = _PROC_WIDTH,
        max_features: int = _MAX_FEATURES,
        gain_lr: float = 0.1,
        gain_max: float = 8.0,
        smoothing: float = 0.5,
    ) -> None:
        self._proc_width = max(64, int(proc_width))
        self._max_features = max(20, int(max_features))
        self._gain_lr = float(np.clip(gain_lr, 0.0, 1.0))
        self._gain_max = abs(float(gain_max))
        self._smoothing = float(np.clip(smoothing, 0.0, 1.0))

        self._prev_gray: NDArray[np.uint8] | None = None
        self._prev_t: float | None = None
        # Online-learned command→image-velocity gains (signed; absorb hardware
        # direction and FOV).  Start at 0 so an unlearned camera contributes no
        # phantom command-based correction.
        self._gain_pan: float = 0.0
        self._gain_tilt: float = 0.0
        # EMA of the emitted ego velocity; None until the first valid estimate so
        # the first reading is exact (no warm-up bias) — matters for tests.
        self._ema: tuple[float, float] | None = None

    # ── public API ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget all state (use on source change / re-arm)."""
        self._prev_gray = None
        self._prev_t = None
        self._ema = None
        # Keep learned gains: they describe the *camera*, not the scene.

    def estimate(
        self,
        frame: NDArray[np.uint8] | None,
        now: float,
        *,
        boxes: Iterable[Sequence[float]] = (),
        ptz_cmd: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> EgoMotion:
        """Return the camera ego-motion between the previous frame and *frame*.

        Args:
            frame:   current BGR frame (full resolution).
            now:     monotonic timestamp for *frame* (seconds).
            boxes:   person boxes to exclude from the flow, full-frame xyxy
                     (e.g. ``track.bbox.as_xyxy()``) so the subject's motion
                     does not pollute the camera estimate.
            ptz_cmd: the PTZ command that produced this motion — the *last*
                     issued ``(pan, tilt, zoom)`` in normalised [-1, 1].
        """
        if frame is None or frame.ndim < 2 or frame.size == 0:
            return EgoMotion()

        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return EgoMotion()

        scale = self._proc_width / float(w)
        gray = self._to_gray(frame, scale)
        gh, gw = gray.shape[:2]

        prev_gray = self._prev_gray
        prev_t = self._prev_t
        # Roll state forward regardless of whether we can produce an estimate.
        self._prev_gray = gray
        self._prev_t = now

        if prev_gray is None or prev_t is None or prev_gray.shape != gray.shape:
            return EgoMotion()
        dt = now - prev_t
        if dt <= 1e-3:
            return EgoMotion()

        pan = float(ptz_cmd[0]) if len(ptz_cmd) > 0 else 0.0
        tilt = float(ptz_cmd[1]) if len(ptz_cmd) > 1 else 0.0

        flow = self._measure_flow(prev_gray, gray, boxes, scale, gw, gh)
        if flow is not None:
            fdx_norm, fdy_norm, conf = flow
            # error space: x grows right (content dx), y grows up (negate image dy)
            ego_vx = fdx_norm / dt
            ego_vy = -fdy_norm / dt
            if conf >= _FLOW_TRUST:
                self._learn_gain(ego_vx, ego_vy, pan, tilt)
                return self._emit(ego_vx, ego_vy, "flow", conf)

        # Flow unusable → predict from the command through the learned gain.
        cmd = self._from_command(pan, tilt)
        if cmd is not None:
            return self._emit(cmd[0], cmd[1], "command", 0.5)
        # Nothing usable: emit zero but keep the EMA warm so we don't snap later.
        return self._emit(0.0, 0.0, "none", 0.0)

    # ── internals ────────────────────────────────────────────────────────────

    def _to_gray(self, frame: NDArray[np.uint8], scale: float) -> NDArray[np.uint8]:
        if scale < 0.999:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _measure_flow(
        self,
        prev_gray: NDArray[np.uint8],
        gray: NDArray[np.uint8],
        boxes: Iterable[Sequence[float]],
        scale: float,
        gw: int,
        gh: int,
    ) -> tuple[float, float, float] | None:
        """Robust median background flow as normalised (dx, dy, confidence)."""
        mask = self._feature_mask(boxes, scale, gw, gh)
        pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self._max_features,
            qualityLevel=0.01,
            minDistance=8,
            mask=mask,
        )
        if pts is None or len(pts) < _MIN_INLIERS:
            return None

        nxt, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts, None, winSize=(15, 15), maxLevel=2
        )
        if nxt is None or status is None:
            return None
        good = status.reshape(-1) == 1
        if int(good.sum()) < _MIN_INLIERS:
            return None

        flow = (nxt[good] - pts[good]).reshape(-1, 2)
        dx, dy, n_in = self._robust_median(flow)
        if n_in < _MIN_INLIERS:
            return None

        # Normalise to error space: a half-frame is "1.0".  Because we measure in
        # the downscaled frame, dividing by its half-width yields the same
        # normalised value as the full-resolution frame would.
        fdx_norm = float(dx) / (gw * 0.5)
        fdy_norm = float(dy) / (gh * 0.5)
        conf = min(1.0, n_in / float(_CONF_FULL_INLIERS))
        return fdx_norm, fdy_norm, conf

    @staticmethod
    def _robust_median(flow: NDArray[np.float32]) -> tuple[float, float, int]:
        """MAD-filtered median flow vector; returns (dx, dy, inlier_count)."""
        med = np.median(flow, axis=0)
        dist = np.linalg.norm(flow - med, axis=1)
        mad = np.median(dist)
        thresh = max(1.0, 3.0 * float(mad))  # keep ≥1px slack for static scenes
        inliers = flow[dist <= thresh]
        if len(inliers) == 0:
            return float(med[0]), float(med[1]), 0
        m = np.median(inliers, axis=0)
        return float(m[0]), float(m[1]), int(len(inliers))

    def _feature_mask(
        self,
        boxes: Iterable[Sequence[float]],
        scale: float,
        gw: int,
        gh: int,
    ) -> NDArray[np.uint8] | None:
        boxes = list(boxes)
        if not boxes:
            return None
        mask = np.full((gh, gw), 255, dtype=np.uint8)
        for box in boxes:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = (float(v) * scale for v in box[:4])
            pad_w = (x2 - x1) * _BOX_PAD_FRAC
            pad_h = (y2 - y1) * _BOX_PAD_FRAC
            ix1 = max(0, int(x1 - pad_w))
            iy1 = max(0, int(y1 - pad_h))
            ix2 = min(gw, int(x2 + pad_w))
            iy2 = min(gh, int(y2 + pad_h))
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = 0
        return mask

    def _learn_gain(self, ego_vx: float, ego_vy: float, pan: float, tilt: float) -> None:
        """EMA-update the command→image gain from a trusted flow measurement."""
        lr = self._gain_lr
        if abs(pan) > 0.05:
            obs = float(np.clip(ego_vx / pan, -self._gain_max, self._gain_max))
            self._gain_pan = (1.0 - lr) * self._gain_pan + lr * obs
        if abs(tilt) > 0.05:
            obs = float(np.clip(ego_vy / tilt, -self._gain_max, self._gain_max))
            self._gain_tilt = (1.0 - lr) * self._gain_tilt + lr * obs

    def _from_command(self, pan: float, tilt: float) -> tuple[float, float] | None:
        if self._gain_pan == 0.0 and self._gain_tilt == 0.0:
            return None  # gains not learned yet → don't invent motion
        return self._gain_pan * pan, self._gain_tilt * tilt

    def _emit(self, vx: float, vy: float, source: str, conf: float) -> EgoMotion:
        a = self._smoothing
        if self._ema is None or a <= 0.0:
            self._ema = (vx, vy)
        else:
            self._ema = (
                a * vx + (1.0 - a) * self._ema[0],
                a * vy + (1.0 - a) * self._ema[1],
            )
        return EgoMotion(vx=self._ema[0], vy=self._ema[1], source=source, confidence=conf)
