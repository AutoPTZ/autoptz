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
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

# Head landmarks, in fallback order (nose is the best single head point).
KP_HEAD_GROUPS: tuple[tuple[int, ...], ...] = (
    (KP_NOSE,),
    (KP_LEFT_EYE, KP_RIGHT_EYE),
    (KP_LEFT_EAR, KP_RIGHT_EAR),
)

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
    kps: Keypoints,
    indices: tuple[int, ...],
    min_conf: float,
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
    kps: Keypoints,
    min_conf: float = DEFAULT_KP_CONF,
) -> tuple[float, float] | None:
    """Midpoint of the (confident) shoulders, or ``None`` if neither is usable."""
    return _avg_point(_confident(kps, (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER), min_conf))


def hip_midpoint(
    kps: Keypoints,
    min_conf: float = DEFAULT_KP_CONF,
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
    not move the aim.  *bias* maps the configured ``tracking.framing`` onto a
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


def head_point(
    kps: Keypoints,
    min_conf: float = DEFAULT_KP_CONF,
) -> tuple[float, float] | None:
    """Best head-centre estimate from real landmarks: nose → eyes → ears → None."""
    for group in KP_HEAD_GROUPS:
        pts = _confident(kps, group, min_conf)
        if pts:
            return _avg_point(pts)
    return None


_HEAD_INDICES = (KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR)
_SHOULDER_INDICES = (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
_HIP_INDICES = (KP_LEFT_HIP, KP_RIGHT_HIP)


def _mean_conf(kps: Keypoints, indices: tuple[int, ...], min_conf: float) -> float:
    """Mean confidence of the usable keypoints at *indices* (0.0 if none)."""
    pts = _confident(kps, indices, min_conf)
    if not pts:
        return 0.0
    return min(1.0, sum(p.conf for p in pts) / len(pts))


def body_aim_point(
    kps: Keypoints,
    *,
    framing: str = "upper_body",
    min_conf: float = DEFAULT_KP_CONF,
) -> tuple[tuple[float, float] | None, float]:
    """Landmark-precise aim point **and a 0–1 confidence**, in frame pixels.

    Unlike :func:`torso_aim_point` (which lifts above the shoulders by a guessed
    fraction of the torso span), this uses the *real* head keypoints
    (nose/eyes/ears) so the framing regions map to anatomy:

    - ``face``           → the head itself
    - ``head_shoulders`` → neck = midpoint(head, shoulders)
    - ``upper_body``     → chest = shoulders nudged ~20 % toward the hips
    - ``full_body``      → person centre = the hips (≈ a standing body's
      mid-height / the bbox centre)

    The horizontal anchor is the shoulder centre (steadiest), falling back to the
    head then the hips.  The returned confidence reflects whether the landmarks a
    region *actually needs* are present, so the caller can **blend** this with the
    bounding-box anchor (high conf → trust pose, low conf → lean on the box)
    without hard-switching.  Crucially each region returns **0 confidence when its
    defining landmark is missing** — so ``full_body`` without confident hips falls
    back to the stable bbox centre instead of snapping up to the shoulders (the
    "jumping near upper body" bug).  Returns ``(None, 0.0)`` when nothing usable.
    """
    shoulders = shoulder_midpoint(kps, min_conf)
    hips = hip_midpoint(kps, min_conf)
    head = head_point(kps, min_conf)
    sh_conf = _mean_conf(kps, _SHOULDER_INDICES, min_conf)
    hip_conf = _mean_conf(kps, _HIP_INDICES, min_conf)
    head_conf = _mean_conf(kps, _HEAD_INDICES, min_conf)

    # Horizontal anchor: shoulders are the most stable, then head, then hips.
    if shoulders is not None:
        ax = shoulders[0]
    elif head is not None:
        ax = head[0]
    elif hips is not None:
        ax = hips[0]
    else:
        return None, 0.0

    def _first_y(*candidates: tuple[float, float] | None) -> float | None:
        for c in candidates:
            if c is not None:
                return c[1]
        return None

    if framing == "face":
        ay = _first_y(head, shoulders, hips)
        conf = head_conf if head is not None else 0.0
    elif framing == "head_shoulders":
        if head is not None and shoulders is not None:
            ay = (head[1] + shoulders[1]) * 0.5
            conf = (head_conf + sh_conf) * 0.5
        else:
            ay = _first_y(head, shoulders, hips)
            conf = (head_conf if head is not None else sh_conf) * 0.7
    elif framing == "full_body":
        # Centre of the person ≈ the hips (mid-height of a standing body, ~the
        # bbox centre).  Gate strictly on the hips: without them, conf 0 so the
        # fused dot uses the stable bbox centre rather than jumping to the
        # shoulders.
        if hips is not None:
            ay = hips[1]
            conf = hip_conf
        else:
            ay = _first_y(shoulders, head)
            conf = 0.0
    else:  # upper_body (default) → chest, a touch below the shoulder line
        if shoulders is not None and hips is not None:
            ay = shoulders[1] + (hips[1] - shoulders[1]) * 0.20
            conf = sh_conf
        elif shoulders is not None:
            ay = shoulders[1]
            conf = sh_conf
        else:
            ay = _first_y(head, hips)
            conf = head_conf if head is not None else 0.0

    if ay is None:
        return None, 0.0
    return (ax, ay), max(0.0, min(1.0, conf))


def subject_height_from_pose(
    kps: Keypoints,
    min_conf: float = DEFAULT_KP_CONF,
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
