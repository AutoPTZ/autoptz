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

import math
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
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Every COCO-17 keypoint EXCEPT the arms (elbows 7/8, wrists 9/10): the body
# landmarks whose extent is stable when arms wave/extend.  Framing math must
# only ever measure these, so gesturing can never grow or shift the shot.
NON_ARM_KEYPOINTS: tuple[int, ...] = (
    KP_NOSE,
    KP_LEFT_EYE,
    KP_RIGHT_EYE,
    KP_LEFT_EAR,
    KP_RIGHT_EAR,
    KP_LEFT_SHOULDER,
    KP_RIGHT_SHOULDER,
    KP_LEFT_HIP,
    KP_RIGHT_HIP,
    KP_LEFT_KNEE,
    KP_RIGHT_KNEE,
    KP_LEFT_ANKLE,
    KP_RIGHT_ANKLE,
)

# Real body extent (nose→ankle) misses the crown of the head and the sole of
# the foot; pad it so the framed height covers the whole person.
_BODY_EXTENT_PAD = 1.08

# Hips-hidden (desk/webcam) stature estimate, from two ARM-INVARIANT anchors:
# the vertical head→shoulder span is ≈ 1/11.8 of standing height (nose height
# ≈ 90% of stature, acromion/shoulder height ≈ 81%, per standard standing
# anthropometric tables — a ≈8.5% span), the biacromial (shoulder) width ≈
# 1/4.1.  max() of the two is robust to both failure modes — tilting the head
# shrinks the span but not the width; turning sideways shrinks the width but
# not the span.  Composition only needs a STABLE ballpark (the framer clamps
# to min/max crop fractions), so modest anthropometric error is fine;
# following the raw bbox is not.
_HEAD_SHOULDER_SPAN_TO_HEIGHT = 11.8
_SHOULDER_WIDTH_TO_HEIGHT = 4.1
# Hips-hidden framing box: its top sits this fraction of the height above the
# head point (crown + hair margin), mirroring where the hips-based box lands.
_CROWN_PAD_FRAC = 0.10

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

    Prefers the REAL vertical extent of the confident non-arm landmarks (head →
    ankles, padded ×1.08 for crown/sole) when that is taller than the classic
    3.3× shoulder→hip heuristic; the heuristic remains the floor so a subject
    with cropped legs (extent = head→hips only) is never under-measured into an
    over-zoom.  Arms (elbows/wrists) are NEVER measured, so gesturing cannot
    change the result.  The caller divides this by the frame height for the
    auto-zoom fraction, so only the *ratio* matters.

    When the hips are hidden (desk/webcam shots — the everyday single-camera
    case) it degrades to a head+shoulders stature estimate instead of ``None``,
    because ``None`` sends the caller back to the arms-inflated bbox height:
    exactly the instability this module exists to prevent.  Returns ``None``
    only when neither anchor pair is available (no shoulders, or shoulders with
    no head landmark and no hips) — the caller then keeps the bbox-height math.
    """
    shoulders = shoulder_midpoint(kps, min_conf)
    hips = hip_midpoint(kps, min_conf)
    if shoulders is None:
        return None
    if hips is not None:
        span = abs(hips[1] - shoulders[1])
        if span <= 0.0:
            return None
        # Empirical torso→full-height factor; keeps the zoom subject-height in
        # the same ballpark as the person bbox height the controller was tuned
        # against.
        height = span * 3.3
    else:
        height = _head_shoulder_height(kps, min_conf)
        if height is None:
            return None
    extent = _body_extent(kps, min_conf)
    if extent is not None:
        height = max(height, (extent[1] - extent[0]) * _BODY_EXTENT_PAD)
    return height


def _head_shoulder_height(
    kps: Keypoints,
    min_conf: float = DEFAULT_KP_CONF,
) -> float | None:
    """Stature estimate from head + shoulders only (hips hidden), or ``None``.

    ``max(head→shoulder span × 10, shoulder width × 4.1)`` — see the constants
    above for why the max of the two anchors is stable.  The width term needs
    BOTH shoulders confident; the span term carries a lone shoulder.
    """
    shoulders = shoulder_midpoint(kps, min_conf)
    head = head_point(kps, min_conf)
    if shoulders is None or head is None:
        return None
    span = shoulders[1] - head[1]  # positive: head above the shoulder line
    if span <= 0.0:
        return None
    height = span * _HEAD_SHOULDER_SPAN_TO_HEIGHT
    both = _confident(kps, (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER), min_conf)
    if len(both) == 2:
        width = abs(both[0].x - both[1].x)
        height = max(height, width * _SHOULDER_WIDTH_TO_HEIGHT)
    return height


def _body_extent(
    kps: Keypoints,
    min_conf: float = DEFAULT_KP_CONF,
) -> tuple[float, float] | None:
    """(min_y, max_y) of the confident NON-ARM landmarks, or ``None``."""
    pts = _confident(kps, NON_ARM_KEYPOINTS, min_conf)
    if not pts:
        return None
    ys = [p.y for p in pts]
    return (min(ys), max(ys))


def torso_framing_box(
    kps: Keypoints,
    min_conf: float = DEFAULT_KP_CONF,
) -> tuple[float, float, float, float] | None:
    """A **stable** person-framing box (x1, y1, x2, y2), or ``None``.

    Derived from the torso anchors only, so raising/extending an arm — which
    inflates the YOLO person bbox and drags its centre — leaves this box
    untouched.  Center Stage crops around it when ``aim_body_mode == "torso"``
    ("Ignore arms"):

    - height = :func:`subject_height_from_pose`: the padded head→ankle body
      extent when visible, floored by the 3.3× shoulder→hip heuristic — so the
      crop zoom matches the physical auto-zoom's torso-mode subject height;
    - centre-x = the shoulder midpoint (the steadiest horizontal anchor);
    - centre-y = the body-extent midpoint when the extent drives the height
      (covers head AND feet), else the hips (≈ a standing body's mid-height).

    Hips hidden (desk/webcam shots) degrades to the head+shoulders stature
    estimate — the box top sits a crown pad above the head point — instead of
    ``None``, so the crop stays arm-invariant in the everyday single-camera
    case.  The width is nominal (the digital crop is sized height-only for a
    single person; only the centre-x matters).  ``None`` when the shoulders —
    or both the head and the hips — are not confidently present: the caller
    keeps the raw-bbox behaviour.
    """
    height = subject_height_from_pose(kps, min_conf)
    if height is None:
        return None
    shoulders = shoulder_midpoint(kps, min_conf)
    hips = hip_midpoint(kps, min_conf)
    if shoulders is None:  # pragma: no cover — height implies shoulders
        return None
    cx = shoulders[0]
    if hips is not None:
        cy = hips[1]
    else:
        head = head_point(kps, min_conf)
        if head is None:  # pragma: no cover — hips-less height implies a head
            return None
        cy = (head[1] - height * _CROWN_PAD_FRAC) + height * 0.5
    extent = _body_extent(kps, min_conf)
    if extent is not None and (extent[1] - extent[0]) * _BODY_EXTENT_PAD >= height:
        # The real body extent set the height — centre the box on it so the
        # head and the feet both stay inside (hips are NOT mid-height then).
        cy = (extent[0] + extent[1]) * 0.5
    half_w = height * 0.2  # nominal ~0.4 aspect person
    return (cx - half_w, cy - height * 0.5, cx + half_w, cy + height * 0.5)


class BoxSmoother:
    """Time-aware EMA over a framing box (x1, y1, x2, y2).

    Pose-derived framing boxes arrive as discrete ~0.2 s estimates with
    keypoint regression noise; consumed raw they make the framing target STEP
    several times a second (visible as jitter, and the PTZ velocity
    feed-forward differentiates the steps into jerks).  This smooths them into
    a continuous signal: ``alpha = 1 - exp(-dt / tau)`` so the smoothing is
    frame-rate independent — a call 1 ms after the last barely moves, a call
    seconds later lands on the target.  ``None`` holds the last box (momentary
    pose dropouts don't snap anything).
    """

    def __init__(self, tau: float = 0.35) -> None:
        self._tau = max(1e-3, tau)
        self._value: tuple[float, float, float, float] | None = None
        self._t: float | None = None

    @property
    def value(self) -> tuple[float, float, float, float] | None:
        return self._value

    def reset(self) -> None:
        self._value = None
        self._t = None

    def update(
        self,
        box: tuple[float, float, float, float] | None,
        t: float,
    ) -> tuple[float, float, float, float] | None:
        """Blend *box* (at time *t*, seconds) into the running estimate."""
        if box is None:
            return self._value
        if self._value is None or self._t is None or t < self._t:
            # First sample, or a genuine backward time jump — snap. An
            # IDENTICAL t (below) is not this case: two calls can land on the
            # same tick under a coarse clock (Windows' default
            # ``time.monotonic()`` resolution is ~15.6 ms — common there, all
            # but impossible on macOS/Linux's finer-grained clock) and must
            # hold instead of snapping onto the new box.
            self._value = box
            self._t = t
            return self._value
        dt = t - self._t
        alpha = 1.0 - math.exp(-dt / self._tau)
        self._value = tuple(v + alpha * (b - v) for v, b in zip(self._value, box, strict=True))  # type: ignore[assignment]
        self._t = t
        return self._value


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
