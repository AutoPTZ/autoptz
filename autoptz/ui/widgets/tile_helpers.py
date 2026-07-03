"""Pure helper functions for the camera tile (no widget state).

Small, side-effect-free helpers extracted from ``camera_tile`` — context-menu
label logic, bbox geometry, and rect-jump detection — so they're easy to
unit-test and the tile widget stays focused on painting + interaction.
``camera_tile`` re-exports these.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QRectF, Qt

if TYPE_CHECKING:
    from PySide6.QtGui import QFontMetrics

log = logging.getLogger(__name__)

# Matches a trailing " 85%" token so on-video labels never elide the percentage.
_PCT_SUFFIX = re.compile(r"\s+\d+%$")

# Box-jump fraction used to detect a teleport vs. smooth motion.
_BOX_JUMP_FRAC = 0.22


def elide_keeping_pct(fm: QFontMetrics, text: str, max_px: float) -> str:
    """Elide *text* to fit *max_px*, but never drop a trailing ``" 85%"`` token.

    Plain ``ElideRight`` chops the percentage off first, so a cramped on-video
    label reads ``"Target: Alexand…"`` instead of ``"Target: Alex… 85%"``. This
    keeps the percentage and elides only the name part.
    """
    max_px = max(0.0, float(max_px))
    if fm.horizontalAdvance(text) <= max_px:
        return text
    m = _PCT_SUFFIX.search(text)
    if m is None:
        return fm.elidedText(text, Qt.TextElideMode.ElideRight, int(max_px))
    head, pct = text[: m.start()], text[m.start() :]
    avail = max(0, int(max_px - fm.horizontalAdvance(pct)))
    return fm.elidedText(head, Qt.TextElideMode.ElideRight, avail) + pct


def _tracking_enabled(rec: Any | None) -> bool:
    return bool(getattr(rec, "tracking_enabled", False)) if rec is not None else False


def _format_target_button_label(label: str) -> str:
    return f"Track: {label or 'Anyone'} ▾"


def _context_menu_action_labels(
    *,
    person: bool,
    current_target: bool,
    has_target: bool,
    tracking: bool,
) -> list[str]:
    """State matrix for the tracking/person part of the tile context menu."""
    if person:
        labels = ["Save Face / Name Person…"]
        if current_target:
            labels.extend(["Stop Tracking", "Clear"] if tracking else ["Track", "Clear"])
        else:
            labels.extend(["Set Target", "Set Target and Track"])
        return labels
    if tracking:
        return ["Stop Tracking", "Clear"]
    if has_target:
        return ["Track", "Clear"]
    return []


def _tracks(rec: Any) -> list[dict[str, Any]]:
    try:
        return rec.tracks_as_list()
    except Exception:  # noqa: BLE001
        return []


def _faces(rec: Any) -> list[dict[str, Any]]:
    try:
        return rec.faces_as_list()
    except Exception:  # noqa: BLE001
        return []


def _pose(rec: Any) -> list[dict[str, float]]:
    try:
        return rec.pose_as_list()
    except Exception:  # noqa: BLE001
        return []


def _tracking_status(rec: Any) -> dict[str, Any]:
    try:
        return rec.tracking_status_as_dict()
    except Exception:  # noqa: BLE001
        return {}


def _quality_state(rec: Any) -> dict[str, Any]:
    try:
        return rec.quality_state_as_dict()
    except Exception:  # noqa: BLE001
        return {}


# Engine ``tracking_status.state`` values → the state-chip vocabulary from the
# transparency spec (idle/searching/locked/ambiguous/coasting/lost). The engine
# emits a few extra operator-facing states ("manual", "degraded", "standby")
# that don't have a dedicated chip color; they fold onto the nearest chip
# semantics below so the chip never shows a blank/unknown color.
_STATE_CHIP_COLOR_KEY = {
    "locked": "locked",
    "coasting": "coasting",
    "searching": "searching",
    "ambiguous": "ambiguous",
    # "Standing by for reacquire" is the terminal give-up state once the coast
    # window has elapsed with no match — the operator-facing equivalent of
    # "lost".
    "standby": "lost",
    # Auto tracking is paused (manual override) or degraded (inference stall):
    # neither is "idle", but neither is a normal tracking state either — treat
    # as the same visual weight as coasting (amber, "something needs attention").
    "manual": "coasting",
    "degraded": "coasting",
}

_STATE_CHIP_TEXT = {
    "locked": "LOCKED",
    "coasting": "COASTING",
    "searching": "SEARCHING",
    "ambiguous": "AMBIGUOUS",
    "lost": "LOST",
}


def _state_chip_info(rec: Any) -> dict[str, Any]:
    """Resolve the tile's tracking-state chip: ``{visible, text, color_key}``.

    Hidden whenever tracking isn't enabled for this camera (nothing to show)
    or the engine reports "idle" (no target set yet) — matching the project
    rule that chips are unconditional truth, not a toggleable feature.
    """
    hidden = {"visible": False, "text": "", "color_key": "idle"}
    if not _tracking_enabled(rec):
        return hidden
    status = _tracking_status(rec)
    state = str(status.get("state", "") or "")
    if not state or state == "idle":
        return hidden
    color_key = _STATE_CHIP_COLOR_KEY.get(state, "searching")
    text = _STATE_CHIP_TEXT.get(color_key, color_key.upper())
    return {"visible": True, "text": text, "color_key": color_key}


def _degradation_chip_info(rec: Any) -> dict[str, Any]:
    """Resolve the tile's quality-degradation chip: ``{visible, text, tooltip}``.

    Visible only once the effective detector cadence has actually stretched
    past the configured value (the auto quality ladder engaged); hidden at the
    configured cadence so the chip never claims a degradation that isn't real.
    """
    hidden = {"visible": False, "text": "", "tooltip": ""}
    q = _quality_state(rec)
    configured = int(q.get("configured_interval", 1) or 1)
    effective = int(q.get("detect_interval", configured) or configured)
    if configured <= 0 or effective <= configured:
        return hidden
    # Round down to the nearest whole multiplier (matches the ladder's ×2/×4 steps).
    multiplier = max(2, effective // configured)
    return {
        "visible": True,
        "text": f"×{multiplier}",
        "tooltip": str(q.get("reason", "") or ""),
    }


def _ignore_arms(rec: Any) -> bool:
    """True when the camera's aim body mode ignores arms ("torso")."""
    try:
        return rec.camera_config.tracking.aim_body_mode == "torso"
    except Exception:  # noqa: BLE001
        return True


def _norm_bbox_contains(box: dict[str, float], x: float, y: float) -> bool:
    return float(box.get("x1", 0.0)) <= x <= float(box.get("x2", 0.0)) and float(
        box.get("y1", 0.0)
    ) <= y <= float(box.get("y2", 0.0))


def _upper_body_bbox(box: dict[str, float]) -> dict[str, float]:
    x1 = float(box.get("x1", 0.0))
    y1 = float(box.get("y1", 0.0))
    x2 = float(box.get("x2", 0.0))
    y2 = float(box.get("y2", 0.0))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y1 + (y2 - y1) * 0.62}


def _head_bbox(box: dict[str, float]) -> dict[str, float]:
    """Approximate the head/face region from a person box (no face detection).

    Top ~30% of the box, narrowed to the central ~70% horizontally, so the
    enroll preview frames *this person's* face rather than the whole frame when
    the subject fills the view and no face box is available."""
    x1 = float(box.get("x1", 0.0))
    y1 = float(box.get("y1", 0.0))
    x2 = float(box.get("x2", 0.0))
    y2 = float(box.get("y2", 0.0))
    cx = (x1 + x2) * 0.5
    half_w = (x2 - x1) * 0.35
    return {"x1": cx - half_w, "y1": y1, "x2": cx + half_w, "y2": y1 + (y2 - y1) * 0.30}


def _select_enrollment_face_bbox(
    faces: list[dict[str, Any]],
    track_box: dict[str, float],
    click: tuple[float, float] | None,
) -> dict[str, float] | None:
    """Pick the detected face that should be enrolled for a clicked person box."""
    candidates: list[dict[str, float]] = []
    for face in faces:
        box = face.get("bbox", {})
        if not isinstance(box, dict):
            continue
        cx = (float(box.get("x1", 0.0)) + float(box.get("x2", 0.0))) * 0.5
        cy = (float(box.get("y1", 0.0)) + float(box.get("y2", 0.0))) * 0.5
        in_track = _norm_bbox_contains(track_box, cx, cy)
        click_selects_face = (
            click is not None
            and _norm_bbox_contains(track_box, click[0], click[1])
            and _norm_bbox_contains(box, click[0], click[1])
        )
        if in_track or click_selects_face:
            candidates.append(box)
    if not candidates:
        return None
    if click is None:
        return max(
            candidates,
            key=lambda b: (float(b.get("x2", 0.0)) - float(b.get("x1", 0.0)))
            * (float(b.get("y2", 0.0)) - float(b.get("y1", 0.0))),
        )

    def area(box: dict[str, float]) -> float:
        return (float(box.get("x2", 0.0)) - float(box.get("x1", 0.0))) * (
            float(box.get("y2", 0.0)) - float(box.get("y1", 0.0))
        )

    x, y = click
    clicked = [box for box in candidates if _norm_bbox_contains(box, x, y)]
    if clicked:
        return max(clicked, key=area)

    # A body-click inside the selected person box should enroll that person's
    # head, not the nearest/largest stray face elsewhere inside a noisy person
    # bbox. Prefer faces whose center sits in the track's expected head region.
    head = _head_bbox(track_box)

    def center_in(box: dict[str, float], region: dict[str, float]) -> bool:
        cx = (float(box.get("x1", 0.0)) + float(box.get("x2", 0.0))) * 0.5
        cy = (float(box.get("y1", 0.0)) + float(box.get("y2", 0.0))) * 0.5
        return _norm_bbox_contains(region, cx, cy)

    head_candidates = [box for box in candidates if center_in(box, head)]
    if head_candidates:
        return max(head_candidates, key=area)
    return max(candidates, key=area)


def _rect_close(a: QRectF, b: QRectF) -> bool:
    return (
        abs(a.x() - b.x()) < 0.5
        and abs(a.y() - b.y()) < 0.5
        and abs(a.width() - b.width()) < 0.5
        and abs(a.height() - b.height()) < 0.5
    )


def _rect_jump(current: QRectF, target: QRectF, video: QRectF) -> bool:
    span = max(1.0, max(video.width(), video.height()))
    dc = (
        (current.center().x() - target.center().x()) ** 2
        + (current.center().y() - target.center().y()) ** 2
    ) ** 0.5
    if dc > span * _BOX_JUMP_FRAC:
        return True
    cw, ch = max(1.0, current.width()), max(1.0, current.height())
    tw, th = max(1.0, target.width()), max(1.0, target.height())
    return max(cw / tw, tw / cw, ch / th, th / ch) > 1.8


def _connect(obj: Any, name: str, slot: Any) -> None:
    try:
        getattr(obj, name).connect(slot)
    except Exception:  # noqa: BLE001
        log.debug("connect %s failed", name, exc_info=True)
