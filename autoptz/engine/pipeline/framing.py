"""Pure aim-point / subject-height math + smoothing for pose-stable framing.

These helpers turn a set of torso keypoints (shoulders + hips) into a **stable**
aim point and subject height that ignore arm/leg motion, so an extended arm no
longer grows the person bbox and yanks the PTZ framing.  Everything here is
dependency-light (stdlib + the keypoint dataclass) and unit-testable without a
model — :mod:`autoptz.engine.pipeline.pose` produces the keypoints, and
:mod:`autoptz.engine.camera_worker` consumes the aim point.

Coordinate convention: keypoint ``(x, y)`` are pixel coordinates in the original
frame, ``y`` growing downward (image convention).  Aim points returned here are
also in that pixel space; the worker converts to the controller's
centre-relative, up-positive error.

Keypoint indexing follows COCO-17 (the layout YOLO-pose / RTMPose emit)::

    5 = left_shoulder   6 = right_shoulder
    11 = left_hip       12 = right_hip

so :data:`TORSO_KEYPOINTS` names the four points we actually use.
"""
from __future__ import annotations

from dataclasses import dataclass

# COCO-17 keypoint indices for the torso anchors we rely on.  Documented here so
# pose.py and any test can share the same constants without re-deriving them.
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

# Minimum keypoint confidence to trust a single point in the aim math.  Below
# this the point is treated as missing (the helpers fall back to whatever points
# remain, or signal "no aim" when too few survive).
DEFAULT_KP_CONF = 0.35


@dataclass(frozen=True)
class Keypoint:
    """One pose keypoint in original-frame pixel space with a confidence."""

    x: float
    y: float
    conf: float

    def usable(self, min_conf: float = DEFAULT_KP_CONF) -> bool:
        return self.conf >= min_conf


# A "pose" is just the COCO-17 keypoint list; helpers index it with the KP_*
# constants and tolerate missing/low-confidence points.
Keypoints = list[Keypoint]


def _avg_point(points: list[Keypoint]) -> tuple[float, float] | None:
    """Mean (x, y) of *points*, or ``None`` if empty."""
    if not points:
        return None
    n = float(len(points))
    return (sum(p.x for p in points) / n, sum(p.y for p in points) / n)


def _confident(
    kps: Keypoints, indices: tuple[int, ...], min_conf: float,
) -> list[Keypoint]:
    """Return the keypoints at *indices* that exist and clear *min_conf*."""
    out: list[Keypoint] = []
    for i in indices:
        if 0 <= i < len(kps):
            kp = kps[i]
            if kp.usable(min_conf):
                out.append(kp)
    return out


def shoulder_midpoint(
    kps: Keypoints, min_conf: float = DEFAULT_KP_CONF,
) -> tuple[float, float] | None:
    """Midpoint of the (confident) shoulders, or ``None`` if neither is usable."""
    return _avg_point(_confident(kps, (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER), min_conf))


def hip_midpoint(
    kps: Keypoints, min_conf: float = DEFAULT_KP_CONF,
) -> tuple[float, float] | None:
    """Midpoint of the (confident) hips, or ``None`` if neither is usable."""
    return _avg_point(_confident(kps, (KP_LEFT_HIP, KP_RIGHT_HIP), min_conf))


def torso_aim_point(
    kps: Keypoints,
    *,
    bias: str = "upper_body",
    min_conf: float = DEFAULT_KP_CONF,
) -> tuple[float, float] | None:
    """Return a **stable** aim point (x, y) in frame pixels, or ``None``.

    The point is derived from the shoulders/hips only, so raising or extending
    an arm — which would grow the YOLO person bbox and shift its centre — does
    not move the aim.  *bias* maps the configured ``tracking.aim_region`` onto a
    sensible torso anchor:

    - ``face`` / ``head_shoulders`` → just above the shoulder line (head sits a
      little above the shoulders; we lift by a fraction of the shoulder→hip span
      so the face stays framed without needing a nose keypoint).
    - ``upper_body`` → the shoulder midpoint (head + torso, robust to arms).
    - ``full_body`` → the shoulder↔hip midpoint (torso centre).

    Falls back gracefully: if hips are missing we use the shoulder midpoint; if
    shoulders are missing we use the hip midpoint; if neither is usable we return
    ``None`` so the caller keeps the bbox-based math.
    """
    shoulders = shoulder_midpoint(kps, min_conf)
    hips = hip_midpoint(kps, min_conf)

    if shoulders is None and hips is None:
        return None
    if shoulders is None:
        return hips
    if hips is None:
        # No torso span available; bias toward/above the shoulders only.
        return shoulders

    sx, sy = shoulders
    hx, hy = hips
    span = hy - sy  # shoulder→hip vertical distance (positive: hips below)

    if bias in ("face", "head_shoulders"):
        # Lift above the shoulder line toward the head.  head_shoulders sits a
        # touch lower (more shoulder) than face (more head).
        lift = 0.45 if bias == "face" else 0.30
        return (sx, sy - span * lift)
    if bias == "full_body":
        return ((sx + hx) * 0.5, (sy + hy) * 0.5)
    # upper_body (default) and any unknown bias → shoulder midpoint.
    return (sx, sy)


def subject_height_from_pose(
    kps: Keypoints, min_conf: float = DEFAULT_KP_CONF,
) -> float | None:
    """Return a **stable** subject-height span (pixels), or ``None``.

    Uses the shoulder→hip vertical distance — a torso measure that does not
    change when arms/legs move — scaled up to approximate the framing-relevant
    person height (a standing adult is roughly ~3.3× their shoulder→hip span).
    The caller divides this by the frame height for the auto-zoom fraction, so
    only the *ratio* matters; the scale just keeps the zoom target comparable to
    the legacy bbox-height behaviour.

    Returns ``None`` when shoulders or hips are not both confidently available
    (the caller then keeps the bbox-height zoom math).
    """
    shoulders = shoulder_midpoint(kps, min_conf)
    hips = hip_midpoint(kps, min_conf)
    if shoulders is None or hips is None:
        return None
    span = abs(hips[1] - shoulders[1])
    if span <= 0.0:
        return None
    # Empirical torso→full-height factor; keeps the zoom subject-height in the
    # same ballpark as the person bbox height the controller was tuned against.
    return span * 3.3


class AimSmoother:
    """Exponential-moving-average smoother for a 2-D aim point.

    Light temporal smoothing so the pose-derived aim point does not jitter
    frame-to-frame (keypoint regression is noisy).  ``alpha`` is the weight of
    the *new* sample: 1.0 = no smoothing, smaller = smoother/laggier.  Feeding
    ``None`` (no aim this tick) holds the last value and is returned unchanged so
    the caller can reuse it.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        self._alpha = max(0.0, min(1.0, alpha))
        self._value: tuple[float, float] | None = None

    @property
    def value(self) -> tuple[float, float] | None:
        return self._value

    def reset(self) -> None:
        self._value = None

    def update(self, point: tuple[float, float] | None) -> tuple[float, float] | None:
        """Blend *point* into the running estimate and return the smoothed aim.

        ``None`` holds (and returns) the previous estimate so a momentary pose
        dropout doesn't snap the aim back to a stale/zero position.
        """
        if point is None:
            return self._value
        if self._value is None:
            self._value = point
            return self._value
        a = self._alpha
        px, py = point
        vx, vy = self._value
        self._value = (vx + a * (px - vx), vy + a * (py - vy))
        return self._value
