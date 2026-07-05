"""Shared framing-target selection for Center Stage and physical PTZ.

Both the digital (Center Stage) crop and the physical PTZ aim must frame the same
subject the same way. This module is the single source of truth for *which box to
frame this tick*:

- **Explicit lock wins.** A person locked by track id (or by configured identity)
  is followed even with group framing on. The live track is preferred; a
  ``trusted_bbox`` fallback holds the frame through brief track-id churn.
- **Group framing** (only when nothing is locked): with more than one confident,
  non-lost person, frame the UNION of their boxes; a single confident person is
  framed alone.
- Otherwise there is no target.

Center Stage crops around the returned box; physical PTZ steers its aim toward the
same box — so explicit-lock precedence and group framing apply identically to both
actuators, which is what "physical PTZ frames like Center Stage" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Box = tuple[float, float, float, float]

# Named "Framing" presets → target subject-height as a fraction of the visible
# frame.  THE single source of truth for shot composition: the physical PTZ
# auto-zoom drives the subject toward this height, and the Center Stage crop is
# sized so the subject fills this fraction of it — so both actuators produce the
# same shot for the same preset.  face/head_shoulders are deliberately moderate
# (the old 0.80/0.60 physical targets and 0.86/0.80 digital fills were
# user-reported as "intense": over-zoomed and twitchy).
SUBJECT_HEIGHT_TARGETS: dict[str, float] = {
    "face": 0.65,
    "head_shoulders": 0.52,
    "upper_body": 0.45,
    "full_body": 0.30,
}


@dataclass(frozen=True)
class FramingTarget:
    """The box to frame this tick, plus how it was chosen."""

    bbox: Box | None = None
    is_group: bool = False
    #: The single locked track id when a lone person is the target — enables the
    #: pose-fused physical aim. ``None`` for a group union or no target.
    primary_track_id: int | None = None


def confident_person_boxes(tracks: list[Any] | None) -> list[Box]:
    """Every confident, non-lost person box in *tracks* (pure, testable)."""
    out: list[Box] = []
    for t in tracks or ():
        if getattr(t, "lost", False):
            continue
        bb = getattr(t, "bbox", None)
        if bb is None:
            continue
        out.append((bb.x1, bb.y1, bb.x2, bb.y2))
    return out


def aim_error_for_box(
    box: Box,
    frame_w: int,
    frame_h: int,
    aim_fraction: float,
) -> tuple[tuple[float, float], float]:
    """Normalized center error + subject-height fraction for framing *box*.

    ``ex > 0`` → the box is right of frame center; ``ey > 0`` → above center (image
    y grows downward, so it's negated to match the PTZ tilt convention where
    positive = up).  ``aim_fraction`` is how far down the box the aim sits (the
    ``framing`` composition, e.g. ~0.38 for upper-body).  Mirrors the bbox anchor
    the single-target path uses, so a group is framed with the same composition.
    """
    if frame_w <= 0 or frame_h <= 0:
        return (0.0, 0.0), 0.0
    x1, y1, x2, y2 = box
    ax = (x1 + x2) * 0.5
    ay = y1 + (y2 - y1) * aim_fraction
    ex = (ax - frame_w * 0.5) / (frame_w * 0.5)
    ey = -((ay - frame_h * 0.5) / (frame_h * 0.5))
    return (float(ex), float(ey)), float((y2 - y1) / frame_h)


def select_framing_target(
    tracks: list[Any] | None,
    *,
    target_track_id: int | None,
    target_identity_id: Any | None,
    trusted_bbox: Box | None,
    group_framing: bool,
) -> FramingTarget:
    """Resolve the framing target for this tick (see module docstring)."""
    explicit_lock = target_track_id is not None or target_identity_id is not None

    if target_track_id is not None:
        for t in tracks or ():
            if (
                getattr(t, "track_id", None) == target_track_id
                and not getattr(t, "lost", False)
                and getattr(t, "bbox", None) is not None
            ):
                bb = t.bbox
                return FramingTarget((bb.x1, bb.y1, bb.x2, bb.y2), False, target_track_id)

    if explicit_lock:
        # Locked but the live box is momentarily absent: hold on the trusted box.
        # An explicit lock NEVER falls through to the group union.
        if trusted_bbox is not None:
            return FramingTarget(trusted_bbox, False, target_track_id)
        return FramingTarget(None, False, target_track_id)

    if group_framing:
        confident = [
            t
            for t in tracks or ()
            if not getattr(t, "lost", False) and getattr(t, "bbox", None) is not None
        ]
        if not confident:
            return FramingTarget(None, False, None)
        if len(confident) == 1:
            # A lone person is a single subject: expose their track id so the
            # pose-stable ("Ignore arms") framing applies like an explicit lock.
            t = confident[0]
            bb = t.bbox
            return FramingTarget((bb.x1, bb.y1, bb.x2, bb.y2), False, getattr(t, "track_id", None))
        from autoptz.engine.pipeline.digital_framer import union_bbox

        return FramingTarget(
            union_bbox([(t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2) for t in confident]),
            True,
            None,
        )

    return FramingTarget(None, False, None)
