"""CameraTile — a single live camera preview with tracking HUD (QPainter).

The widget paints the latest shared-memory
frame in ``paintEvent`` (Qt double-buffers widget painting, so there is no
flicker and no crossfade needed) and draws the broadcast HUD on top: a name
pill, an fps/health chip, detection boxes, the target-lock reticle, a tracking
dead-zone, and status banners.

It reads per-frame state directly from the :class:`CameraListModel` record
(``fps``, ``health``, ``streaming``, ``tracks``) and the :class:`ShmFrameSource`.
Interactions drive ``EngineClient`` slots:
  * left-click a detection box → select the tile and set that person as target,
  * left-click the tile background → toggle this camera's selection,
  * right-click a person → save/name face, set target, or set target and track,
  * arrow keys → ``ptzNudge`` (pausing tracking while held), space → ``clearTarget``.

Manual PTZ controls and quick-recall presets now live in the Properties → PTZ
section, so the tile overlay is intentionally lean: target, track, stop, clear.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.common import HelpBadge, animate_widget_visibility, on_theme_changed
from autoptz.ui.widgets.tile_helpers import (  # re-exported for back-compat
    _connect,
    _context_menu_action_labels,  # noqa: F401  re-exported for tests
    _faces,
    _format_target_button_label,
    _head_bbox,
    _ignore_arms,
    _norm_bbox_contains,
    _pose,
    _rect_close,
    _rect_jump,
    _snap_center_axis,
    _tracking_enabled,
    _tracking_status,
    _tracks,
    _upper_body_bbox,  # noqa: F401  re-exported for tests
    elide_keeping_pct,
)

log = logging.getLogger(__name__)

_NUDGE_SPEED = 0.65

# Framing-box editing: handle hit radius (px) and clamp on the half-extents
# (fraction of the half-frame) so the box can't collapse or fill the frame.
_FB_HANDLE_HIT = 11.0
_FB_MIN = 0.03
_FB_MAX = 0.9
# The 8 resize handles, as (name, x-sign, y-sign) where sign ∈ {-1,0,1} relative
# to the box centre.  Corners drive both axes; edges drive one.
_FB_HANDLES = (
    ("nw", -1, -1),
    ("n", 0, -1),
    ("ne", 1, -1),
    ("e", 1, 0),
    ("se", 1, 1),
    ("s", 0, 1),
    ("sw", -1, 1),
    ("w", -1, 0),
)
_FB_CURSORS = {
    "nw": Qt.CursorShape.SizeFDiagCursor,
    "se": Qt.CursorShape.SizeFDiagCursor,
    "ne": Qt.CursorShape.SizeBDiagCursor,
    "sw": Qt.CursorShape.SizeBDiagCursor,
    "n": Qt.CursorShape.SizeVerCursor,
    "s": Qt.CursorShape.SizeVerCursor,
    "e": Qt.CursorShape.SizeHorCursor,
    "w": Qt.CursorShape.SizeHorCursor,
}

# COCO-17 skeleton edges (pairs of keypoint indices) for the pose overlay.
_POSE_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
)
# Arm limbs (shoulder→elbow→wrist) — hidden when "Ignore arms" is on, so the
# skeleton visibly reflects that the aim/zoom is disregarding them.
_POSE_ARM_EDGES = frozenset({(5, 7), (7, 9), (6, 8), (8, 10)})
_POSE_ARM_JOINTS = frozenset({7, 8, 9, 10})  # elbows + wrists
_POSE_MIN_CONF = 0.3

# Box interpolation: telemetry lands at ~10 Hz, while painting can happen faster.
# Interpolate only between telemetry sequences, then snap to the real box. This
# avoids the "too smooth but not true" look caused by endless per-paint EMA.
_BOX_INTERP_S = 0.09
# Motion-prediction indicator: how far ahead (seconds) to project the aim-point
# velocity for the lead arrow + ghost box, plus gates that keep it calm.  Working
# in seconds (not frames) makes the lead frame-rate independent; the persistence
# + min-speed gates stop it flicking on detection jitter.
_PREDICT_LEAD_S = 0.30  # project the aim ~0.3 s ahead
_PREDICT_SMOOTH_A = 0.20  # EMA weight on the normalized aim velocity
_PREDICT_MIN_SPEED = 0.05  # min normalized speed (fraction/s) before drawing
_PREDICT_PERSIST = 3  # consecutive moving frames required before drawing
_PREDICT_MIN_PX = 6.0  # and a floor in painted px so it never micro-jitters
_PREDICT_MAX_BOX_FACTOR = 0.85
_REORDER_DRAG_PX = 8.0


class CameraTile(QWidget):
    """Live preview + HUD for one camera, addressed by stable ``camera_id``."""

    selectedRequested = Signal(str)  # camera_id — background click (toggles selection)
    selectExclusiveRequested = Signal(str)  # camera_id — box click (selects, never toggles off)
    infoRequested = Signal(str)  # camera_id — "Camera Info" chosen
    renameRequested = Signal(str)  # camera_id — "Rename…" chosen
    reorderDragStarted = Signal(str, object)  # camera_id, global QPoint
    reorderDragMoved = Signal(str, object)  # camera_id, global QPoint
    reorderDragFinished = Signal(str, object)  # camera_id, global QPoint

    def __init__(
        self,
        camera_id: str,
        client: Any,
        frame_source: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self._client = client
        self._model = client.cameraModel
        self._frames = frame_source
        self._selected = False
        self._painted_rect = QRectF()  # where the video is drawn (overlay mapping)
        # Framing-box drag state: the handle being dragged ("nw".."e"/None) and a
        # live (half_w, half_h) override applied while dragging (committed to the
        # camera config on release).
        self._fb_drag: str | None = None
        self._fb_live: tuple[float, float] | None = None
        self._fb_center_live: tuple[float, float] | None = None
        self._fb_move_offset: tuple[float, float] | None = None
        self._empty_press_pos: QPointF | None = None
        self._empty_press_global: object | None = None
        self._reorder_dragging = False
        # Per-track interpolation state. Each entry stores the previous drawn box,
        # latest telemetry box, telemetry sequence, and transition start time.
        self._box_smooth: dict[int, dict[str, Any]] = {}
        # Motion-prediction state, all in normalized [0,1] frame space so it's
        # resolution- and zoom-independent.  Driven by the framing-aware aim
        # point (not the raw detection box) so it stays calm and on-body.
        self._pred_smooth: dict[int, tuple[float, float]] = {}  # smoothed vel /s
        self._pred_prev: dict[int, tuple[float, float, float, int]] = {}  # ax, ay, ts, seq
        self._pred_hits: dict[int, int] = {}  # consecutive frames of real motion
        self._target_choices: list[tuple[str, str]] = [("Anyone", "")]
        self._hover = False

        self.setMinimumSize(220, 124)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAutoFillBackground(True)

        # Repaint timer — paced to the stream fps (clamped) and only while shown.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        # Which on-video overlays to draw (cached; refreshed on overlaysChanged).
        self._overlays = self._read_overlays()
        _connect(self._client, "overlaysChanged", self._on_overlays_changed)

        # Pinned control bar: pick who to track + enable/disable tracking.
        self._build_overlay()

        # Always-visible "?" info badge (top-right) → per-stage fps breakdown on
        # hover/click (Capture vs +Detection vs +Face), so the operator can see
        # which subsystem costs what.  Parented to the tile (not the hover bar)
        # so it's available without hovering.
        self._info_badge = HelpBadge("Per-stage performance", self)
        self._info_badge.setObjectName("helpBadge")

    # ── overlay (per-tile tracking control) ──────────────────────────────────────

    def _build_overlay(self) -> None:
        # Compact action bar. It appears on hover, selection, or when there is a
        # pending/active target, so the video stays clean until action is useful.
        self._overlay = QWidget(self)
        self._overlay.setObjectName("tileTrackingBar")
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(self._overlay)
        row.setContentsMargins(7, 5, 7, 5)
        row.setSpacing(6)
        row.addStretch(1)

        self._target_btn = QPushButton("Track: Anyone ▾", self._overlay)
        self._target_btn.setObjectName("tileTargetBtn")
        self._target_btn.setToolTip("Choose a registered person as the target.")
        self._target_btn.setMinimumWidth(T.fs(120))
        self._target_btn.setMaximumWidth(T.fs(230))
        self._target_btn.clicked.connect(self._show_target_menu)
        row.addWidget(self._target_btn, 0)

        self._follow_btn = QPushButton("Track", self._overlay)
        self._follow_btn.setObjectName("tileFollowBtn")
        self._follow_btn.setToolTip("Start following the selected target.")
        self._follow_btn.clicked.connect(self._on_follow_clicked)
        row.addWidget(self._follow_btn)

        self._stop_btn = QPushButton("Stop", self._overlay)
        self._stop_btn.setObjectName("tileStopBtn")
        self._stop_btn.setToolTip("Pause tracking but keep the target selected.")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        row.addWidget(self._stop_btn)

        self._clear_btn = QPushButton("Clear", self._overlay)
        self._clear_btn.setObjectName("tileClearBtn")
        self._clear_btn.setToolTip("Stop tracking and clear the selected target.")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        row.addWidget(self._clear_btn)

        self._overlay.mousePressEvent = lambda event: event.accept()
        self._overlay.mouseReleaseEvent = lambda event: event.accept()

        self._restyle_overlay()
        on_theme_changed(self._client, self._restyle_overlay)
        _connect(self._client, "identitiesChanged", self._reload_overlay_targets)
        # Keep the person picker in sync when the target changes by any route
        # (clicking a recognized person on the video, the Properties picker, etc.).
        _connect(self._client, "targetChanged", self._on_target_changed)
        _connect(self._client, "trackingChanged", self._on_tracking_changed)
        self._refresh_overlay_state()

    def _restyle_overlay(self) -> None:
        """(Re)apply literal-color styling — runs at build + on theme change."""
        scrim = "rgba(15,15,18,210)"
        self._overlay.setStyleSheet(
            f"QWidget#tileTrackingBar {{ background: {scrim}; border-radius: 6px; }}"
            f"QPushButton#tileTargetBtn {{ color: {T.VIDEO_TEXT};"
            f" background: rgba(255,255,255,26); border: 1px solid rgba(255,255,255,40);"
            f" border-radius: 5px; padding: 3px 9px; min-height: {T.fs(20)}px;"
            f" font-size: {T.fs(11)}px; }}"
            f"QPushButton#tileTargetBtn:hover {{ border-color: {T.ACCENT.name()};"
            f" background: rgba(255,255,255,40); }}"
            f"QPushButton#tileFollowBtn {{ background: {T.ACCENT.name()}; color: #ffffff;"
            f" border: none; border-radius: 5px; font-weight: 700;"
            f" padding: 3px 10px; font-size: {T.fs(11)}px; }}"
            f"QPushButton#tileFollowBtn:hover {{ background: {T.ACCENT.lighter(115).name()}; }}"
            f"QPushButton#tileStopBtn {{ background: {T.ERROR}; color: #ffffff;"
            f" border: none; border-radius: 5px; font-weight: 700;"
            f" padding: 3px 10px; font-size: {T.fs(11)}px; }}"
            f"QPushButton#tileStopBtn:hover {{ background: {QColor(T.ERROR).lighter(112).name()}; }}"
            f"QPushButton#tileClearBtn {{"
            f" background: rgba(255,255,255,26); color: {T.VIDEO_TEXT};"
            f" border: 1px solid rgba(255,255,255,40); border-radius: 5px;"
            f" padding: 3px 10px; font-size: {T.fs(11)}px; font-weight: 700; }}"
            f"QPushButton#tileClearBtn:hover {{"
            f" background: rgba(255,255,255,42); border-color: {T.ACCENT.name()}; }}"
        )
        self._refresh_overlay_state()

    def _position_overlay(self) -> None:
        ov = getattr(self, "_overlay", None)
        if ov is not None:
            m = 6
            h = ov.sizeHint().height()
            ov.setGeometry(m, self.height() - h - m, max(0, self.width() - 2 * m), h)
            small = self.width() < T.fs(360)
            self._target_btn.setMaximumWidth(T.fs(150 if small else 230))
            self._follow_btn.setText(self._follow_label(short=small))
        badge = getattr(self, "_info_badge", None)
        if badge is not None:
            badge.move(self.width() - badge.width() - 8, 8)
            badge.raise_()

    def _reload_overlay_targets(self) -> None:
        people = []
        try:
            people = self._client.registeredIdentities() or []
        except Exception:  # noqa: BLE001
            people = []
        rec = self._record()
        cur = ""
        try:
            cur = rec.camera_config.target.identity_id or ""
        except Exception:  # noqa: BLE001
            cur = ""
        self._target_choices = [("Anyone", "")]
        for pdict in people:
            self._target_choices.append((pdict.get("name") or "(unnamed)", pdict.get("id") or ""))
        self._update_target_button(cur)

    def _update_target_button(self, current_id: str = "") -> None:
        self._target_btn.setText(self._target_button_label())

    def _show_target_menu(self) -> None:
        menu = QMenu(self._target_btn)
        rec = self._record()
        cur = ""
        try:
            cur = rec.camera_config.target.identity_id or ""
        except Exception:  # noqa: BLE001
            cur = ""
        for name, ident in self._target_choices:
            act = menu.addAction(name)
            act.setCheckable(True)
            act.setChecked((ident or "") == cur)
            act.triggered.connect(lambda _checked=False, i=ident: self._set_overlay_target(i))
        menu.exec(self._target_btn.mapToGlobal(self._target_btn.rect().bottomLeft()))

    def _refresh_overlay_state(self) -> None:
        self._reload_overlay_targets()
        rec = self._record()
        tracking = _tracking_enabled(rec)
        has_target = self._has_target(rec)
        interactive = self._selected or self._hover
        self._update_target_button()
        self._follow_btn.setText(self._follow_label())
        self._follow_btn.setVisible(interactive and has_target and not tracking)
        self._stop_btn.setVisible(tracking)
        self._clear_btn.setVisible((interactive or tracking) and has_target)
        animate_widget_visibility(self._overlay, self._overlay_should_show(rec), duration=105)
        self._position_overlay()
        self.update()

    def _overlay_should_show(self, rec: Any | None = None) -> bool:
        rec = rec if rec is not None else self._record()
        return self._hover or self._selected or _tracking_enabled(rec)

    def _follow_label(self, *, short: bool = False) -> str:
        return "Track"

    def _has_target(self, rec: Any | None = None) -> bool:
        rec = rec if rec is not None else self._record()
        if rec is None:
            return False
        if getattr(rec, "target_track_id", None) is not None:
            return True
        try:
            if rec.camera_config.target.identity_id:
                return True
        except Exception:  # noqa: BLE001
            pass
        return any(t.get("is_target") for t in _tracks(rec))

    def _target_button_label(self) -> str:
        """Return the compact target dropdown label for the current target."""
        rec = self._record()
        if rec is None:
            return _format_target_button_label("Anyone")
        try:
            ident = rec.camera_config.target.identity_id or ""
        except Exception:  # noqa: BLE001
            ident = ""
        if ident:
            for name, iid in self._target_choices:
                if iid == ident:
                    return _format_target_button_label(name or "Person")
            return _format_target_button_label("Person")
        track_id = getattr(rec, "target_track_id", None)
        if track_id is not None:
            return _format_target_button_label(f"ID {track_id}")
        for t in _tracks(rec):
            if t.get("is_target"):
                label = t.get("identity") or f"ID {t.get('track_id', '?')}"
                return _format_target_button_label(str(label))
        return _format_target_button_label("Anyone")

    def _set_overlay_target(self, ident: str) -> None:
        try:
            self._client.setTargetIdentity(self.camera_id, ident)
            self._refresh_overlay_state()
        except Exception:  # noqa: BLE001
            log.debug("setTargetIdentity failed", exc_info=True)

    def _on_follow_clicked(self) -> None:
        try:
            if self._has_target():
                self._client.enableTracking(self.camera_id, True)
        except Exception:  # noqa: BLE001
            log.debug("follow target failed", exc_info=True)
        self._refresh_overlay_state()

    def _apply_target_payload(self, payload: dict[str, Any]) -> None:
        """Apply a pending/right-click target payload without changing tracking."""
        identity_id = str(payload.get("identity_id") or "")
        if identity_id:
            self._client.setTargetIdentity(self.camera_id, identity_id)
        else:
            self._client.setTarget(self.camera_id, int(payload["track_id"]))

    def _on_stop_clicked(self) -> None:
        try:
            self._client.enableTracking(self.camera_id, False)
        except Exception:  # noqa: BLE001
            log.debug("enableTracking failed", exc_info=True)
        self._refresh_overlay_state()

    def _on_clear_clicked(self) -> None:
        try:
            self._client.clearTargetAndStop(self.camera_id)
        except Exception:  # noqa: BLE001
            log.debug("clearTarget failed", exc_info=True)
        self._refresh_overlay_state()

    def _on_tracking_changed(self, camera_id: str) -> None:
        if camera_id == self.camera_id:
            self._refresh_overlay_state()

    # ── state helpers ──────────────────────────────────────────────────────────

    def _record(self) -> Any | None:
        try:
            return self._model.get_record(self.camera_id)
        except Exception:  # noqa: BLE001
            return None

    def set_selected(self, selected: bool) -> None:
        if self._selected != selected:
            self._selected = selected
            self._refresh_overlay_state()
            self.update()

    def display_name(self) -> str:
        rec = self._record()
        return getattr(rec, "display_name", self.camera_id) if rec else self.camera_id

    # ── lifecycle: only paint while visible ────────────────────────────────────

    def showEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.start(40)
        self._refresh_overlay_state()
        self._position_overlay()
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        self._position_overlay()
        super().resizeEvent(event)

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = True
        self._refresh_overlay_state()
        self._position_overlay()
        self._overlay.raise_()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = False
        self._refresh_overlay_state()
        super().leaveEvent(event)

    def _tick(self) -> None:
        rec = self._record()
        fps = float(getattr(rec, "fps", 0.0) or 0.0) if rec else 0.0
        interval = int(1000 / max(8.0, min(30.0, fps if fps > 0 else 12.0)))
        if interval != self._timer.interval():
            self._timer.setInterval(interval)
        self._info_badge.set_help(self._compose_perf_tooltip(rec))
        self.update()

    def _compose_perf_tooltip(self, rec: Any) -> str:
        """Per-stage performance text for the "?" badge.

        Shows capture fps plus the *effective* fps each cumulative stage cost
        would sustain (capture → +detection → +face), mirroring the operator's
        mental model of the "30 → 23 → 19 fps" cliff, with the per-stage ms.
        """
        if rec is None:
            return "No telemetry yet."
        tel = getattr(rec, "telemetry", None)
        cap = float(getattr(rec, "fps", 0.0) or 0.0)
        ingest = float(getattr(tel, "ingest_ms", 0.0) or 0.0) if tel else 0.0
        detect = float(getattr(tel, "detect_ms", 0.0) or 0.0) if tel else 0.0
        face = float(getattr(tel, "face_ms", 0.0) or 0.0) if tel else 0.0

        def eff(total_ms: float) -> str:
            if total_ms <= 0.0:
                return "—"
            f = 1000.0 / total_ms
            if cap > 0.0:
                f = min(f, cap)
            return f"{f:.0f} fps"

        lines = [
            f"Capture      {cap:.0f} fps   (ingest {ingest:.1f} ms)"
            if cap > 0
            else "Capture      — (no signal)",
            f"+ Detection  {eff(ingest + detect)}   (detect {detect:.1f} ms)"
            if detect > 0
            else "+ Detection  — (off / warming up)",
            f"+ Face       {eff(ingest + detect + face)}   (face {face:.1f} ms)"
            if face > 0
            else "+ Face       — (off / not yet run)",
        ]
        # Source line: resolution + friendly source kind.
        src = {}
        try:
            src = (
                rec.camera_config.source.model_dump() if getattr(rec, "camera_config", None) else {}
            )
        except Exception:  # noqa: BLE001
            src = {}
        w = int(getattr(tel, "width", 0) or 0) if tel else 0
        h = int(getattr(tel, "height", 0) or 0) if tel else 0
        kind = src.get("source_label") or src.get("type") or ""
        res = f"{w}×{h}" if w and h else "—"
        lines.append(f"Source: {res}" + (f" · {kind}" if kind else ""))
        return "\n".join(lines)

    # ── painting ───────────────────────────────────────────────────────────────

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # backing: real frames cover this, while no-signal/letterbox areas still
        # read as part of the app surface instead of disappearing into black.
        p.fillRect(self.rect(), QColor(T.CURRENT.surface_alt))

        rec = self._record()
        streaming = bool(getattr(rec, "streaming", False)) if rec else False
        health = str(getattr(rec, "health", "ok")) if rec else "ok"

        # video frame
        img = self._frames.latest_qimage(self.camera_id) if self._frames else None
        if img is not None and not img.isNull():
            iw, ih = max(1, img.width()), max(1, img.height())
            scale = min(w / iw, h / ih)
            vw, vh = iw * scale, ih * scale
            x = (w - vw) / 2
            y = (h - vh) / 2
            self._painted_rect = QRectF(x, y, vw, vh)
            p.drawImage(self._painted_rect, img, QRectF(img.rect()))
        else:
            self._painted_rect = QRectF(0, 0, w, h)

        if not streaming:
            self._paint_no_signal(p, health)

        # HUD overlays (only meaningful once we have a frame/telemetry)
        if rec is not None:
            self._paint_framing_box(p, rec)
            self._paint_tracks(p, rec)
            self._paint_name_pill(p, rec)
            self._paint_fps_chip(p, rec)
            self._paint_banner(p, rec, streaming)

        # Selection glow is separate from tracking; tracking is shown on the target marker.
        self._paint_border(p, rec)
        self._paint_selection(p)
        p.end()

    def _paint_no_signal(self, p: QPainter, health: str) -> None:
        p.save()
        c = QColor(T.VIDEO_SUBTEXT)
        c.setAlphaF(0.5)
        p.setPen(c)
        f = QFont(self.font())
        f.setPixelSize(12)
        p.setFont(f)
        label = "No Signal" if health in ("ok", "stalled") else health.upper()
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "📷  " + label)
        p.restore()

    def _paint_border(self, p: QPainter, rec: Any) -> None:
        """No tracking-state edge border; the target person marker is the signal."""
        return

    def _paint_selection(self, p: QPainter) -> None:
        """Selection = a bright accent glow + corner handles."""
        if not self._selected:
            return
        accent = QColor(T.ACCENT)
        ring = QRectF(4, 4, self.width() - 8, self.height() - 8)
        p.save()
        p.setBrush(Qt.BrushStyle.NoBrush)
        glow = QColor(accent)
        glow.setAlphaF(0.28)
        p.setPen(QPen(glow, 6))
        p.drawRect(ring)
        p.setPen(QPen(accent, 2))
        p.drawRect(ring)
        # Corner handles for an unmistakable "this is the active tile" feel.
        p.setPen(QPen(accent, 3))
        length, inset = 18.0, 4.0
        w, h = float(self.width()), float(self.height())
        for cx, cy, dx, dy in (
            (inset, inset, 1, 1),
            (w - inset, inset, -1, 1),
            (inset, h - inset, 1, -1),
            (w - inset, h - inset, -1, -1),
        ):
            p.drawLine(QPointF(cx, cy), QPointF(cx + dx * length, cy))
            p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * length))
        p.restore()

    # ── framing box (adjustable PTZ dead-zone) ───────────────────────────────────

    def _framing_extents(self, rec: Any) -> tuple[float, float] | None:
        """Return the framing box's (half_w, half_h) fractions, or None if off.

        While a drag is in progress the live override is returned so the box
        tracks the pointer before the change is committed to the config.
        """
        cfg = getattr(rec, "camera_config", None)
        ptz = getattr(cfg, "ptz", None) if cfg is not None else None
        if ptz is None or not getattr(ptz, "safe_zone_enabled", False):
            return None
        if self._fb_live is not None:
            return self._fb_live
        return (float(getattr(ptz, "safe_zone_w", 0.15)), float(getattr(ptz, "safe_zone_h", 0.22)))

    def _framing_center(self, rec: Any) -> tuple[float, float]:
        """Return the framing-box centre offset in controller error coordinates."""
        if self._fb_center_live is not None:
            return self._fb_center_live
        cfg = getattr(rec, "camera_config", None)
        ptz = getattr(cfg, "ptz", None) if cfg is not None else None
        if ptz is None:
            return (0.0, 0.0)
        try:
            x = max(-0.9, min(0.9, float(getattr(ptz, "safe_zone_x", 0.0))))
            y = max(-0.9, min(0.9, float(getattr(ptz, "safe_zone_y", 0.0))))
            return (x, y)
        except Exception:  # noqa: BLE001
            return (0.0, 0.0)

    def _framing_box_rect(self, rec: Any) -> QRectF | None:
        """The framing box in widget pixels (centred on the painted video)."""
        ext = self._framing_extents(rec)
        r = self._painted_rect
        if ext is None or r.width() < 8 or r.height() < 8:
            return None
        hw, hh = ext
        half_w = hw * r.width() / 2.0
        half_h = hh * r.height() / 2.0
        off_x, off_y = self._framing_center(rec)
        cx = r.center().x() + off_x * (r.width() / 2.0)
        cy = r.center().y() - off_y * (r.height() / 2.0)
        return QRectF(cx - half_w, cy - half_h, 2 * half_w, 2 * half_h)

    def _framing_handle_points(self, box: QRectF) -> dict[str, QPointF]:
        """Map each resize-handle name to its centre point on *box*."""
        cx, cy = box.center().x(), box.center().y()
        pts: dict[str, QPointF] = {}
        for name, sx, sy in _FB_HANDLES:
            x = cx + sx * box.width() / 2.0
            y = cy + sy * box.height() / 2.0
            pts[name] = QPointF(x, y)
        return pts

    def _framing_handle_at(self, pos: QPointF, rec: Any) -> str | None:
        """Return the framing handle under *pos* (only when the tile is selected)."""
        if not self._selected:
            return None
        box = self._framing_box_rect(rec)
        if box is None:
            return None
        for name, pt in self._framing_handle_points(box).items():
            if abs(pt.x() - pos.x()) <= _FB_HANDLE_HIT and abs(pt.y() - pos.y()) <= _FB_HANDLE_HIT:
                return name
        return None

    def _framing_move_hit(self, pos: QPointF, rec: Any) -> bool:
        """Return True when *pos* can drag-move the selected framing box."""
        if not self._selected:
            return False
        box = self._framing_box_rect(rec)
        return bool(box is not None and box.contains(pos))

    def _framing_roundness(self, rec: Any) -> float:
        cfg = getattr(rec, "camera_config", None)
        ptz = getattr(cfg, "ptz", None) if cfg is not None else None
        try:
            return max(0.0, min(1.0, float(getattr(ptz, "safe_zone_roundness", 1.0))))
        except Exception:  # noqa: BLE001
            return 1.0

    def _draw_framing_shape(self, p: QPainter, box: QRectF, roundness: float) -> None:
        """Draw the framing region as a rectangle…oval per *roundness* (0…1)."""
        if roundness >= 0.99:
            p.drawEllipse(box)
        else:
            rad = roundness * min(box.width(), box.height()) / 2.0
            p.drawRoundedRect(box, rad, rad)

    def _paint_framing_box(self, p: QPainter, rec: Any) -> None:
        """Draw the adjustable framing region (PTZ dead-zone) over the video.

        Read straight from the camera config — no telemetry plumbing.  A subtle
        dashed outline (rectangle…oval by ``safe_zone_roundness``) plus a centre
        "+" crosshair marking the aim reference point; when the tile is selected,
        a brighter outline + grab handles so the operator can resize it to keep
        the subject framed.
        """
        box = self._framing_box_rect(rec)
        if box is None:
            return
        roundness = self._framing_roundness(rec)
        p.save()
        editing = self._selected
        col = QColor(T.TARGET) if editing else QColor(255, 255, 255, 130)
        pen = QPen(col, 1.6, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        self._draw_framing_shape(p, box, roundness)
        # Centre "+" crosshair — the reference point the PTZ keeps the subject on.
        cx, cy = box.center().x(), box.center().y()
        p.setPen(QPen(col, 1.4))
        p.drawLine(QPointF(cx - 7, cy), QPointF(cx + 7, cy))
        p.drawLine(QPointF(cx, cy - 7), QPointF(cx, cy + 7))
        if editing:
            # Solid square grab handles at corners + edge midpoints.
            p.setPen(QPen(QColor(T.TARGET), 1))
            p.setBrush(QColor(T.TARGET))
            hs = 3.0
            for pt in self._framing_handle_points(box).values():
                p.drawRect(QRectF(pt.x() - hs, pt.y() - hs, 2 * hs, 2 * hs))
        p.restore()

    def _resize_framing(self, handle: str, pos: QPointF) -> None:
        """Update the live box extents from a handle drag."""
        r = self._painted_rect
        if r.width() < 8 or r.height() < 8:
            return
        rec = self._record()
        box = self._framing_box_rect(rec)
        if box is None:
            return
        cx, cy = box.center().x(), box.center().y()
        cur = self._fb_live or (0.15, 0.22)
        hw, hh = cur
        if "e" in handle or "w" in handle:
            hw = abs(pos.x() - cx) / (r.width() / 2.0)
        if "n" in handle or "s" in handle:
            hh = abs(pos.y() - cy) / (r.height() / 2.0)
        hw = max(_FB_MIN, min(_FB_MAX, hw))
        hh = max(_FB_MIN, min(_FB_MAX, hh))
        self._fb_live = (hw, hh)
        self.update()

    def _move_framing(self, pos: QPointF) -> None:
        """Move the framing box centre, clamped inside the painted video."""
        r = self._painted_rect
        if r.width() < 8 or r.height() < 8:
            return
        hw, hh = self._fb_live or self._framing_extents(self._record()) or (0.15, 0.22)
        off_x, off_y = self._fb_move_offset or (0.0, 0.0)
        center = QPointF(pos.x() + off_x, pos.y() + off_y)
        x = (center.x() - r.center().x()) / (r.width() / 2.0)
        y = -((center.y() - r.center().y()) / (r.height() / 2.0))
        x = max(-1.0 + hw, min(1.0 - hw, x))
        y = max(-1.0 + hh, min(1.0 - hh, y))
        x = _snap_center_axis(x)
        y = _snap_center_axis(y)
        self._fb_center_live = (x, y)
        self.update()

    def _commit_framing(self) -> None:
        """Persist the dragged box extents to the camera config (once, on release)."""
        if self._fb_live is None and self._fb_center_live is None:
            return
        rec = self._record()
        hw, hh = self._fb_live or self._framing_extents(rec) or (0.15, 0.22)
        cx, cy = self._fb_center_live or self._framing_center(rec)
        cx, cy = _snap_center_axis(cx), _snap_center_axis(cy)
        try:
            self._client.updateCameraConfigPatch(
                self.camera_id,
                {
                    "ptz": {
                        "safe_zone_x": round(cx, 4),
                        "safe_zone_y": round(cy, 4),
                        "safe_zone_w": round(hw, 4),
                        "safe_zone_h": round(hh, 4),
                    }
                },
            )
        except Exception:  # noqa: BLE001
            log.debug("framing-box persist failed", exc_info=True)
        self._fb_drag = None
        self._fb_live = None
        self._fb_center_live = None
        self._fb_move_offset = None

    def _map_bbox(self, bbox: dict[str, float]) -> QRectF:
        """Normalized (0–1) bbox → widget pixels within the painted video rect."""
        r = self._painted_rect
        x1 = r.x() + bbox.get("x1", 0.0) * r.width()
        y1 = r.y() + bbox.get("y1", 0.0) * r.height()
        x2 = r.x() + bbox.get("x2", 0.0) * r.width()
        y2 = r.y() + bbox.get("y2", 0.0) * r.height()
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def _smoothed_rect(self, track_id: int, target: QRectF, seq: int) -> QRectF:
        """Interpolate briefly between telemetry boxes, then land on truth."""
        now = time.monotonic()
        state = self._box_smooth.get(track_id)
        if state is None:
            self._box_smooth[track_id] = {
                "seq": seq,
                "from": QRectF(target),
                "to": QRectF(target),
                "start": now,
            }
            return QRectF(target)

        to_rect = state["to"]
        changed = state.get("seq") != seq or not _rect_close(to_rect, target)
        if changed:
            drawn = self._interpolated_state_rect(state, now)
            if _rect_jump(drawn, target, self._painted_rect):
                drawn = QRectF(target)
            state = {
                "seq": seq,
                "from": QRectF(drawn),
                "to": QRectF(target),
                "start": now,
            }
            self._box_smooth[track_id] = state
        return self._interpolated_state_rect(state, now)

    @staticmethod
    def _interpolated_state_rect(state: dict[str, Any], now: float) -> QRectF:
        start = float(state.get("start", now))
        a = 1.0 if _BOX_INTERP_S <= 0 else max(0.0, min(1.0, (now - start) / _BOX_INTERP_S))
        # Smoothstep: short ease without endless lag.
        a = a * a * (3.0 - 2.0 * a)
        src: QRectF = state["from"]
        dst: QRectF = state["to"]
        return QRectF(
            src.x() + (dst.x() - src.x()) * a,
            src.y() + (dst.y() - src.y()) * a,
            src.width() + (dst.width() - src.width()) * a,
            src.height() + (dst.height() - src.height()) * a,
        )

    @staticmethod
    def _normalized_aim(t: dict[str, Any]) -> tuple[float, float]:
        """The track's framing-aware aim point in normalized [0,1] frame space.

        Falls back to the detection-box centre when no aim telemetry is present
        (pose off / non-target), so the prediction still has an anchor."""
        aim = t.get("aim")
        if isinstance(aim, dict) and aim.get("x") is not None and aim.get("y") is not None:
            return float(aim.get("x", 0.5)), float(aim.get("y", 0.5))
        bb = t.get("bbox") or {}
        cx = (float(bb.get("x1", 0.0)) + float(bb.get("x2", 0.0))) * 0.5
        cy = (float(bb.get("y1", 0.0)) + float(bb.get("y2", 0.0))) * 0.5
        return cx, cy

    def _paint_prediction(
        self,
        p: QPainter,
        t: dict[str, Any],
        rect: QRectF,
        color: QColor,
        origin: QPointF | None = None,
        seq: int = 0,
    ) -> None:
        """Draw the motion-prediction indicator: a lead arrow + a ghost box.

        Driven by the framing-aware *aim point* (engine-smoothed, normalized),
        projected ``_PREDICT_LEAD_S`` seconds ahead — not the raw detection box.
        It only appears after the subject has been moving for a few frames, so it
        glides ahead of a walking person instead of flicking on every detection
        or ID switch.  Because it follows the aim point, the Frame-on setting
        moves it just like the reticle.

        Velocity is resampled only when the telemetry ``seq`` advances, so the
        faster repaint timer can't dilute the motion estimate with zero-delta
        samples between engine frames."""
        if bool(t.get("lost")):
            return
        tid = t.get("track_id")

        ax, ay = self._normalized_aim(t)
        now = time.monotonic()
        prev = self._pred_prev.get(tid)
        if prev is None:
            self._pred_prev[tid] = (ax, ay, now, seq)
            return
        px, py, pts, pseq = prev
        if seq != pseq:
            # A genuinely new engine sample → update the velocity estimate.
            self._pred_prev[tid] = (ax, ay, now, seq)
            dt = now - pts
            if dt <= 1e-3 or dt > 0.5:
                # First frame, or a long gap (paused / occluded) — don't trust Δ.
                self._pred_hits[tid] = 0
                return
            a = _PREDICT_SMOOTH_A
            svx, svy = self._pred_smooth.get(tid, (0.0, 0.0))
            svx += ((ax - px) / dt - svx) * a  # normalized fraction/s
            svy += ((ay - py) / dt - svy) * a
            self._pred_smooth[tid] = (svx, svy)
            # Persistence gate: only draw once motion is real and sustained.
            speed = (svx * svx + svy * svy) ** 0.5
            self._pred_hits[tid] = (
                self._pred_hits.get(tid, 0) + 1 if speed >= _PREDICT_MIN_SPEED else 0
            )
        # else: a repaint between engine frames — reuse the last velocity/hits.

        if self._pred_hits.get(tid, 0) < _PREDICT_PERSIST:
            return

        svx, svy = self._pred_smooth.get(tid, (0.0, 0.0))
        r = self._painted_rect
        dx = svx * _PREDICT_LEAD_S * r.width()
        dy = svy * _PREDICT_LEAD_S * r.height()
        if (dx * dx + dy * dy) < _PREDICT_MIN_PX * _PREDICT_MIN_PX:
            return
        import math

        length = math.hypot(dx, dy)
        max_len = max(18.0, max(rect.width(), rect.height()) * _PREDICT_MAX_BOX_FACTOR)
        if length > max_len:
            s = max_len / length
            dx *= s
            dy *= s
        base = origin if origin is not None else rect.center()
        cx, cy = base.x(), base.y()
        tip = QPointF(cx + dx, cy + dy)
        p.save()
        # Ghost box where the subject is predicted to be.
        ghost = rect.translated(dx, dy)
        gc = QColor(color)
        gc.setAlphaF(0.35)
        p.setPen(QPen(gc, 1.3, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(ghost)
        # Lead arrow from the current centre toward the prediction.
        p.setPen(QPen(color, 2))
        p.drawLine(QPointF(cx, cy), tip)
        # Arrow head.
        ang = math.atan2(dy, dx)
        for off in (math.radians(150), math.radians(-150)):
            hx = tip.x() + 8 * math.cos(ang + off)
            hy = tip.y() + 8 * math.sin(ang + off)
            p.drawLine(tip, QPointF(hx, hy))
        p.restore()

    def _read_overlays(self) -> dict[str, bool]:
        try:
            return self._client.overlays()
        except Exception:  # noqa: BLE001
            return {"detection": True, "faces": False, "pose": False, "prediction": False}

    def _on_overlays_changed(self) -> None:
        self._overlays = self._read_overlays()
        self.update()

    def _paint_tracks(self, p: QPainter, rec: Any) -> None:
        tracks = _tracks(rec)
        status = _tracking_status(rec)
        seq = int(getattr(getattr(rec, "telemetry", None), "seq", 0) or 0)
        live = bool(getattr(rec, "tracking_enabled", False))
        show_boxes = self._overlays.get("detection", True)
        # Drop smoothing state for tracks that are gone (so a recycled id can't
        # inherit a stale box).
        ids = {t.get("track_id") for t in tracks}
        self._box_smooth = {k: v for k, v in self._box_smooth.items() if k in ids}
        self._pred_smooth = {k: v for k, v in self._pred_smooth.items() if k in ids}
        self._pred_prev = {k: v for k, v in self._pred_prev.items() if k in ids}
        self._pred_hits = {k: v for k, v in self._pred_hits.items() if k in ids}
        target = None
        for t in tracks:
            if t.get("is_target"):
                target = t
                continue
            # Non-target detection boxes are gated by the "Detection boxes" toggle;
            # the target box always shows (it's the lock indicator).
            if show_boxes:
                self._paint_box(p, t, seq)
        if self._overlays.get("faces", False):
            self._paint_faces(p, rec)
        if self._overlays.get("pose", False):
            self._paint_pose(p, rec)
        if target is not None:
            # Live tracking the present target → red; locked but idle → green.
            self._paint_target(p, target, live, seq, status)
        elif str(status.get("state", "idle") or "idle") != "idle":
            self._paint_target_status_label(p, status)

    def _paint_faces(self, p: QPainter, rec: Any) -> None:
        """Draw detected face boxes (amber) with the matched name, if any."""
        faces = _faces(rec)
        if not faces:
            return
        p.save()
        for f in faces:
            rect = self._map_bbox(f.get("bbox", {}))
            if rect.width() < 2 or rect.height() < 2:
                continue
            p.setPen(QPen(QColor(T.FACE_BOX), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rect)
            name = f.get("identity") or ""
            if name:
                self._draw_label(
                    p, rect.x(), rect.y() - 16, name, QColor(T.VIDEO_TEXT), bg=QColor(T.FACE_BOX)
                )
        p.restore()

    def _paint_pose(self, p: QPainter, rec: Any) -> None:
        """Draw the target's COCO-17 pose skeleton (sky-blue), if available.

        When the camera's aim body mode is "torso" (Ignore arms), the arm limbs
        and joints are hidden so the overlay matches the aim/zoom source.
        """
        kps = _pose(rec)
        if len(kps) < 17:
            return
        r = self._painted_rect
        pts = [
            (
                r.x() + k.get("x", 0.0) * r.width(),
                r.y() + k.get("y", 0.0) * r.height(),
                k.get("conf", 0.0),
            )
            for k in kps
        ]
        ignore_arms = _ignore_arms(rec)
        solid = QPen(QColor(T.POSE), 2)
        p.save()
        for a, b in _POSE_EDGES:
            if a < len(pts) and b < len(pts):
                if ignore_arms and (a, b) in _POSE_ARM_EDGES:
                    continue
                xa, ya, ca = pts[a]
                xb, yb, cb = pts[b]
                if ca >= _POSE_MIN_CONF and cb >= _POSE_MIN_CONF:
                    p.setPen(solid)
                    p.drawLine(QPointF(xa, ya), QPointF(xb, yb))
        p.setPen(Qt.PenStyle.NoPen)
        for i, (x, y, c) in enumerate(pts):
            if c >= _POSE_MIN_CONF:
                if ignore_arms and i in _POSE_ARM_JOINTS:
                    continue
                p.setBrush(QColor(T.POSE))
                p.drawEllipse(QPointF(x, y), 2.5, 2.5)
        p.restore()

    def _paint_box(self, p: QPainter, t: dict[str, Any], seq: int) -> None:
        rect = self._map_bbox(t.get("bbox", {}))
        if rect.width() < 2 or rect.height() < 2:
            return
        p.save()
        # Every other detected person reads in blue (the brand accent).
        p.setPen(QPen(T.ACCENT, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)
        ident = t.get("identity") or ""
        label = ident if ident else f"ID {t.get('track_id', '?')}"
        self._draw_label(
            p,
            rect.x(),
            rect.y() - 18,
            label,
            QColor(T.WARNING) if ident else QColor(T.VIDEO_SUBTEXT),
        )
        p.restore()

    def _paint_target(
        self,
        p: QPainter,
        t: dict[str, Any],
        live: bool,
        seq: int,
        status: dict[str, Any] | None = None,
    ) -> None:
        rect = self._map_bbox(t.get("bbox", {}))
        if rect.width() < 2 or rect.height() < 2:
            return
        lost = bool(t.get("lost"))
        p.save()
        # Lost/coasting → amber + dashed (clearly "searching", not actively
        # following).  Live-tracking the present target → red; locked-idle → green.
        accent = QColor(T.WARNING) if lost else QColor(T.ERROR) if live else QColor(T.TARGET)
        pen = QPen(accent, 2)
        if lost:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(rect)
        if not lost and live and self._overlays.get("prediction", False):
            self._paint_prediction(p, t, rect, accent, self._target_aim_point(t, rect), seq)
        # PTZ aim dot (telemetry-smoothed aim point, not raw bbox/body debug).
        aim_pt = self._target_aim_point(t, rect)
        cx, cy = aim_pt.x(), aim_pt.y()
        ring = QColor(accent)
        ring.setAlphaF(0.8)
        p.setPen(QPen(ring, 2.0))
        p.setBrush(QColor(accent))
        p.drawEllipse(QPointF(cx, cy), 4.5, 4.5)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), 9.0, 9.0)
        # Lock label — reuse this top-left target chip for tracking status too.
        conf = t.get("confidence")
        ident = t.get("identity") or f"ID {t.get('track_id', '?')}"
        status = status or {}
        status_state = str(status.get("state", "idle") or "idle")
        status_headline = str(status.get("headline", "") or "").strip()
        status_detail = str(status.get("detail", "") or "").strip()
        status_severity = str(status.get("severity", "") or "")
        label = (
            status_headline
            if status_headline and status_state != "idle"
            else (f"Searching: {ident}" if lost else f"Target: {ident}")
        )
        if status_state == "locked" and isinstance(conf, int | float) and conf > 0:
            label += f"  {round(conf * 100)}%"
        label_detail = "" if status_state in ("", "idle", "locked") else status_detail
        label_accent = (
            QColor(T.ERROR)
            if status_severity == "error"
            else QColor(T.WARNING)
            if status_severity == "warning"
            else accent
        )
        self._draw_target_label(p, label, label_accent, detail=label_detail)
        p.restore()

    def _paint_target_status_label(self, p: QPainter, status: dict[str, Any]) -> None:
        headline = str(status.get("headline", "") or "").strip()
        if not headline:
            return
        detail = str(status.get("detail", "") or "").strip()
        severity = str(status.get("severity", "") or "")
        accent = (
            QColor(T.ERROR)
            if severity == "error"
            else QColor(T.WARNING)
            if severity == "warning"
            else QColor(T.TARGET)
        )
        self._draw_target_label(p, headline, accent, detail=detail)

    def _target_aim_point(self, t: dict[str, Any], rect: QRectF) -> QPointF:
        aim = t.get("aim")
        if isinstance(aim, dict) and aim.get("x") is not None and aim.get("y") is not None:
            r = self._painted_rect
            return QPointF(
                r.x() + float(aim.get("x", 0.0)) * r.width(),
                r.y() + float(aim.get("y", 0.0)) * r.height(),
            )
        return rect.center()

    @staticmethod
    def _draw_brackets(p: QPainter, rect: QRectF, color: QColor) -> None:
        n = min(rect.width(), rect.height()) * 0.22
        p.setPen(QPen(color, 3))
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        for cx, cy, dx, dy in (
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            p.drawLine(QPointF(cx, cy), QPointF(cx + dx * n, cy))
            p.drawLine(QPointF(cx, cy), QPointF(cx, cy + dy * n))

    def _draw_label(
        self,
        p: QPainter,
        x: float,
        y: float,
        text: str,
        fg: QColor,
        bg: QColor | None = None,
    ) -> None:
        f = QFont(self.font())
        f.setPixelSize(11)
        f.setBold(True)
        p.setFont(f)
        fm = QFontMetrics(f)
        tw = fm.horizontalAdvance(text)
        pad = 6
        bounds = self._painted_rect if self._painted_rect.isValid() else QRectF(self.rect())
        width = min(float(tw + pad * 2), max(24.0, bounds.width() - 4.0))
        lx = max(bounds.left() + 2.0, min(float(x), bounds.right() - width - 2.0))
        ly = max(bounds.top() + 2.0, min(max(2.0, float(y)), bounds.bottom() - 20.0))
        rect = QRectF(lx, ly, width, 18)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg if bg is not None else T.VIDEO_SCRIM)
        p.drawRoundedRect(rect, 4, 4)
        p.setPen(fg)
        p.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter,
            elide_keeping_pct(fm, text, width - pad * 2),
        )

    def _draw_target_label(
        self,
        p: QPainter,
        text: str,
        bg: QColor,
        *,
        detail: str = "",
    ) -> None:
        """Draw the active target label as a readable tile chip, not a bbox chip."""
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setBold(True)
        detail_font = QFont(self.font())
        detail_font.setPixelSize(10)
        p.setFont(f)
        fm = QFontMetrics(f)
        detail_fm = QFontMetrics(detail_font)
        pad_x = 8
        bounds = self._painted_rect if self._painted_rect.isValid() else QRectF(self.rect())
        max_width = min(360.0, max(120.0, bounds.width() - 16.0))
        min_width = min(max_width, 180.0)
        text_width = fm.horizontalAdvance(text)
        detail_width = detail_fm.horizontalAdvance(detail) if detail else 0
        width = min(max_width, max(min_width, float(max(text_width, detail_width) + pad_x * 2)))
        x = bounds.left() + 8.0
        y = bounds.top() + 36.0
        height = 38.0 if detail else 22.0
        if y + height > bounds.bottom() - 4.0:
            y = bounds.top() + 8.0
        rect = QRectF(x, y, width, height)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(rect, 5, 5)
        p.setPen(QColor(T.VIDEO_TEXT))
        p.drawText(
            QRectF(rect.left(), rect.top(), rect.width(), 22),
            Qt.AlignmentFlag.AlignCenter,
            elide_keeping_pct(fm, text, width - pad_x * 2),
        )
        if detail:
            p.setFont(detail_font)
            p.setPen(QColor(T.VIDEO_TEXT))
            p.drawText(
                QRectF(rect.left() + pad_x, rect.top() + 20, rect.width() - pad_x * 2, 14),
                Qt.AlignmentFlag.AlignCenter,
                detail_fm.elidedText(detail, Qt.TextElideMode.ElideRight, int(width - pad_x * 2)),
            )

    def _paint_name_pill(self, p: QPainter, rec: Any) -> None:
        name = getattr(rec, "display_name", "") or self.camera_id
        health = str(getattr(rec, "health", "ok"))
        streaming = bool(getattr(rec, "streaming", False))
        dot = (
            QColor(T.TRACKING)
            if (streaming and health == "ok")
            else QColor(T.WARNING)
            if health in ("reconnecting", "stalled")
            else QColor(T.ERROR)
            if health == "error"
            else QColor(T.VIDEO_SUBTEXT)
        )
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        fm = QFontMetrics(f)
        tw = min(fm.horizontalAdvance(name), self.width() - 80)
        name = fm.elidedText(name, Qt.TextElideMode.ElideRight, int(tw))
        tw = fm.horizontalAdvance(name)
        # A solid dark scrim + hairline keeps the chip readable on *any* video —
        # so it stays legible in light mode where the frame behind it may be
        # bright. (On-video HUD is intentionally light-on-scrim, not theme-flipped.)
        rect = QRectF(8, 8, tw + 30, 23)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.VIDEO_SCRIM)
        p.drawRoundedRect(rect, 6, 6)
        p.setPen(QPen(QColor(255, 255, 255, 36), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, 6, 6)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dot)
        p.drawEllipse(QPointF(rect.x() + 14, rect.center().y()), 4, 4)
        p.setPen(QColor(T.VIDEO_TEXT))
        p.drawText(
            QRectF(rect.x() + 23, rect.y(), tw + 6, rect.height()),
            Qt.AlignmentFlag.AlignVCenter,
            name,
        )

    def _paint_fps_chip(self, p: QPainter, rec: Any) -> None:
        fps = float(getattr(rec, "fps", 0.0) or 0.0)
        health = str(getattr(rec, "health", "ok"))
        if health != "ok":
            text = health.upper()
            color = QColor(T.ERROR) if health == "error" else QColor(T.WARNING)
        else:
            text = f"{fps:.0f} fps"
            color = (
                QColor(T.TRACKING)
                if fps > 20
                else QColor(T.WARNING)
                if fps > 10
                else QColor(T.LOST)
            )
        f = QFont(self.font())
        f.setPixelSize(10)
        f.setBold(True)
        p.setFont(f)
        fm = QFontMetrics(f)
        tw = fm.horizontalAdvance(text)
        # Sit left of the top-right "?" info badge so the two never overlap.
        badge_w = self._info_badge.width() + 8 if getattr(self, "_info_badge", None) else 0
        rect = QRectF(self.width() - tw - 24 - badge_w, 8, tw + 16, 20)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.VIDEO_SCRIM)
        p.drawRoundedRect(rect, 5, 5)
        p.setPen(color)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_banner(self, p: QPainter, rec: Any, streaming: bool) -> None:
        health = str(getattr(rec, "health", "ok"))
        text = ""
        if health == "reconnecting":
            text = "RECONNECTING"
        elif health == "error":
            text = "⚠ ERROR"
        if not text:
            return
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setBold(True)
        p.setFont(f)
        fm = QFontMetrics(f)
        tw = fm.horizontalAdvance(text)
        rect = QRectF((self.width() - tw - 28) / 2, self.height() / 2 - 14, tw + 28, 28)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(T.VIDEO_SCRIM)
        p.drawRoundedRect(rect, 6, 6)
        p.setPen(QColor(T.WARNING) if health == "reconnecting" else QColor(T.ERROR))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # ── interaction ────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._empty_press_pos = None
        self._empty_press_global = None
        self._reorder_dragging = False
        self._fb_move_offset = None
        if event.button() == Qt.MouseButton.LeftButton:
            # Grabbing a framing-box handle (only when this tile is selected and
            # the press lands on a handle) starts a resize and takes priority over
            # box/background clicks so dragging never fires target selection.
            handle = self._framing_handle_at(event.position(), self._record())
            if handle is not None:
                self._fb_drag = handle
                self._fb_live = self._framing_extents(self._record())
                self._fb_center_live = self._framing_center(self._record())
                event.accept()
                return
            if self._framing_move_hit(event.position(), self._record()):
                self._fb_drag = "move"
                self._fb_live = self._framing_extents(self._record())
                self._fb_center_live = self._framing_center(self._record())
                box = self._framing_box_rect(self._record())
                if box is not None:
                    self._fb_move_offset = (
                        box.center().x() - event.position().x(),
                        box.center().y() - event.position().y(),
                    )
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            track_id, is_target = self._hit_test_track(event.position())
            if track_id is not None:
                # Person clicks always select the camera. Clicking the current
                # target clears the target only; it never toggles camera
                # selection off, which avoids the deselect/reselect bug.
                self.selectExclusiveRequested.emit(self.camera_id)
                try:
                    if is_target or self._is_current_track_target(int(track_id)):
                        self._client.clearTargetAndStop(self.camera_id)
                    else:
                        self._apply_target_payload(self._target_payload_for_track(int(track_id)))
                except Exception:  # noqa: BLE001
                    log.debug("click target failed", exc_info=True)
                self._refresh_overlay_state()
                event.accept()
                return
            self._empty_press_pos = QPointF(event.position())
            self._empty_press_global = event.globalPosition().toPoint()
            event.accept()
            return
        else:
            # Non-left (e.g. right-click for the menu): select, never toggle off.
            self.selectExclusiveRequested.emit(self.camera_id)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        if self._fb_drag is not None:
            if self._fb_drag == "move":
                self._move_framing(pos)
            else:
                self._resize_framing(self._fb_drag, pos)
            event.accept()
            return
        if self._empty_press_pos is not None:
            dx = pos.x() - self._empty_press_pos.x()
            dy = pos.y() - self._empty_press_pos.y()
            if not self._reorder_dragging and (dx * dx + dy * dy) >= (_REORDER_DRAG_PX**2):
                self._reorder_dragging = True
                self.reorderDragStarted.emit(self.camera_id, self._empty_press_global)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if self._reorder_dragging:
                self.reorderDragMoved.emit(self.camera_id, event.globalPosition().toPoint())
                event.accept()
                return
        # Hover feedback: a resize cursor over a framing handle (when editable).
        handle = self._framing_handle_at(pos, self._record())
        if handle is not None:
            self.setCursor(_FB_CURSORS.get(handle, Qt.CursorShape.SizeAllCursor))
        elif self._framing_move_hit(pos, self._record()):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._fb_drag is not None and event.button() == Qt.MouseButton.LeftButton:
            self._commit_framing()
            self.unsetCursor()
            event.accept()
            return
        if self._empty_press_pos is not None and event.button() == Qt.MouseButton.LeftButton:
            if self._reorder_dragging:
                self.reorderDragFinished.emit(self.camera_id, event.globalPosition().toPoint())
                self.unsetCursor()
            else:
                self.selectedRequested.emit(self.camera_id)
            self._empty_press_pos = None
            self._empty_press_global = None
            self._reorder_dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Double-click a person → name them (or assign to an existing person)."""
        if event.button() == Qt.MouseButton.LeftButton:
            track_id, _ = self._hit_test_track(event.position())
            if track_id is not None:
                self.selectExclusiveRequested.emit(self.camera_id)
                self._open_assign_dialog(int(track_id), event.position())
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def _open_assign_dialog(self, track_id: int, click_pos: QPointF | None = None) -> None:
        """Name the tracked person, binding their face so they're recognised later.

        Pick an existing person from the dropdown, or choose "New person…" and
        type a name.  Either way the worker captures this track's current face
        embedding and remembers it (see ``CameraWorker.enroll_track``).
        """
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QLabel,
            QLineEdit,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Name this person")
        lay = QVBoxLayout(dlg)
        preview = self._enrollment_preview_pixmap(track_id, click_pos)
        if preview is not None:
            img = QLabel(dlg)
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img.setMinimumSize(T.fs(260), T.fs(170))
            img.setPixmap(preview)
            lay.addWidget(img)
        lay.addWidget(QLabel("Assign this tracked person to:"))
        combo = QComboBox(dlg)
        combo.addItem("➕  New person…", "")
        try:
            for pdict in self._client.registeredIdentities() or []:
                combo.addItem(pdict.get("name") or "(unnamed)", pdict.get("id") or "")
        except Exception:  # noqa: BLE001
            log.debug("registeredIdentities failed", exc_info=True)
        lay.addWidget(combo)
        name = QLineEdit(dlg)
        name.setPlaceholderText("New person's name")
        lay.addWidget(name)

        def _sync() -> None:
            is_new = not combo.currentData()
            name.setVisible(is_new)
            if is_new:
                name.setFocus()

        combo.currentIndexChanged.connect(lambda *_: _sync())
        _sync()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        dlg.resize(max(T.fs(430), dlg.sizeHint().width()), max(T.fs(360), dlg.sizeHint().height()))

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        identity_id = combo.currentData() or ""
        click = self._normalized_video_point(click_pos) if click_pos is not None else None
        try:
            if identity_id:
                if click is not None:
                    self._client.assignTrackToIdentity(
                        self.camera_id,
                        identity_id,
                        track_id,
                        click[0],
                        click[1],
                    )
                else:
                    self._client.assignTrackToIdentity(self.camera_id, identity_id, track_id)
            else:
                nm = name.text().strip()
                if nm:
                    if click is not None:
                        self._client.enrollIdentity(
                            self.camera_id,
                            nm,
                            track_id,
                            click[0],
                            click[1],
                        )
                    else:
                        self._client.enrollIdentity(self.camera_id, nm, track_id)
        except Exception:  # noqa: BLE001
            log.debug("assign identity failed", exc_info=True)

    def _enrollment_preview_pixmap(
        self,
        track_id: int,
        click_pos: QPointF | None,
    ) -> QPixmap | None:
        """Return a current face crop, falling back to upper-body track crop."""
        img = self._frames.latest_qimage(self.camera_id) if self._frames else None
        if img is None or img.isNull():
            return None
        rec = self._record()
        if rec is None:
            return None
        track = next((t for t in _tracks(rec) if t.get("track_id") == track_id), None)
        if track is None:
            return None
        click = self._normalized_video_point(click_pos)
        track_box = track.get("bbox", {})
        face_box = self._face_bbox_for_preview(rec, track_box, click)
        # No detected face → frame the head region, not the whole person/frame.
        crop_box = face_box or _head_bbox(track_box)
        return self._crop_preview(img, crop_box, pad=0.28 if face_box else 0.18)

    def _face_bbox_for_preview(
        self,
        rec: Any,
        track_box: dict[str, float],
        click: tuple[float, float] | None,
    ) -> dict[str, float] | None:
        candidates = []
        for face in _faces(rec):
            box = face.get("bbox", {})
            cx = (float(box.get("x1", 0.0)) + float(box.get("x2", 0.0))) * 0.5
            cy = (float(box.get("y1", 0.0)) + float(box.get("y2", 0.0))) * 0.5
            if not _norm_bbox_contains(track_box, cx, cy):
                continue
            candidates.append(box)
        if not candidates:
            return None
        if click is None:
            return max(
                candidates,
                key=lambda b: (b.get("x2", 0.0) - b.get("x1", 0.0))
                * (b.get("y2", 0.0) - b.get("y1", 0.0)),
            )

        def score(box: dict[str, float]) -> tuple[int, float]:
            x, y = click
            inside = _norm_bbox_contains(box, x, y)
            cx = (float(box.get("x1", 0.0)) + float(box.get("x2", 0.0))) * 0.5
            cy = (float(box.get("y1", 0.0)) + float(box.get("y2", 0.0))) * 0.5
            return (0 if inside else 1, (cx - x) ** 2 + (cy - y) ** 2)

        return min(candidates, key=score)

    def _crop_preview(
        self,
        img: Any,
        bbox: dict[str, float],
        *,
        pad: float,
    ) -> QPixmap | None:
        w, h = img.width(), img.height()
        x1 = float(bbox.get("x1", 0.0)) * w
        y1 = float(bbox.get("y1", 0.0)) * h
        x2 = float(bbox.get("x2", 0.0)) * w
        y2 = float(bbox.get("y2", 0.0)) * h
        bw, bh = x2 - x1, y2 - y1
        if bw <= 2 or bh <= 2:
            return None
        x1 = max(0, int(round(x1 - bw * pad)))
        y1 = max(0, int(round(y1 - bh * pad)))
        x2 = min(w, int(round(x2 + bw * pad)))
        y2 = min(h, int(round(y2 + bh * pad)))
        if x2 <= x1 or y2 <= y1:
            return None
        pix = QPixmap.fromImage(img.copy(x1, y1, x2 - x1, y2 - y1))
        if pix.isNull():
            return None
        return pix.scaled(
            T.fs(300),
            T.fs(190),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _normalized_video_point(self, pos: QPointF | None) -> tuple[float, float] | None:
        """Map a widget point into normalized video coordinates."""
        if pos is None:
            return None
        r = self._painted_rect
        if r.width() <= 0 or r.height() <= 0:
            return None
        x = (pos.x() - r.x()) / r.width()
        y = (pos.y() - r.y()) / r.height()
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    def _on_target_changed(self, camera_id: str) -> None:
        """Refresh the person picker when this camera's target changes."""
        if camera_id == self.camera_id:
            self._reload_overlay_targets()
            self._refresh_overlay_state()

    def _identity_for_track(self, track_id: int) -> str:
        """Return the recognized identity_id for *track_id*, or '' if unrecognized."""
        rec = self._record()
        if rec is None:
            return ""
        for t in _tracks(rec):
            if t.get("track_id") == track_id:
                return str(t.get("identity_id") or "")
        return ""

    def _target_payload_for_track(self, track_id: int) -> dict[str, Any]:
        """Build the pending target descriptor for a clicked track."""
        rec = self._record()
        label = f"ID {track_id}"
        identity_id = ""
        if rec is not None:
            for t in _tracks(rec):
                if t.get("track_id") == track_id:
                    identity_id = str(t.get("identity_id") or "")
                    label = str(t.get("identity") or label)
                    break
        return {"track_id": int(track_id), "identity_id": identity_id, "label": label}

    def _is_current_track_target(self, track_id: int) -> bool:
        rec = self._record()
        if rec is None:
            return False
        if getattr(rec, "target_track_id", None) == track_id:
            return True
        try:
            identity_id = rec.camera_config.target.identity_id or ""
        except Exception:  # noqa: BLE001
            identity_id = ""
        for t in _tracks(rec):
            if t.get("track_id") != track_id:
                continue
            return bool(t.get("is_target") or (identity_id and t.get("identity_id") == identity_id))
        return False

    def _hit_test_track(self, pos: QPointF) -> tuple[int | None, bool]:
        """Return ``(track_id, is_target)`` for the box under ``pos`` (else ``(None, False)``).

        The target box is tested too so that clicking it can clear the lock; it is
        preferred over overlapping non-target boxes.
        """
        rec = self._record()
        if rec is None:
            return None, False
        hit_other: int | None = None
        for t in _tracks(rec):
            if not self._map_bbox(t.get("bbox", {})).contains(pos):
                continue
            if t.get("is_target"):
                return t.get("track_id"), True
            if hit_other is None:
                hit_other = t.get("track_id")
        return hit_other, False

    def contextMenuEvent(self, event: Any) -> None:  # noqa: N802
        self.selectExclusiveRequested.emit(self.camera_id)
        menu = QMenu(self)
        rec = self._record()
        tracking = _tracking_enabled(rec)
        has_target = self._has_target(rec)
        track_id, hit_is_target = self._hit_test_track(QPointF(event.pos()))
        if track_id is not None:
            click_pos = QPointF(event.pos())
            current_target = bool(hit_is_target or self._is_current_track_target(int(track_id)))
            menu.addAction(
                "Save Face / Name Person…",
                lambda tid=int(track_id), p=click_pos: self._open_assign_dialog(tid, p),
            )
            payload = self._target_payload_for_track(int(track_id))
            if current_target:
                if tracking:
                    menu.addAction("Stop Tracking", self._on_stop_clicked)
                    menu.addAction("Clear", self._on_clear_clicked)
                else:
                    menu.addAction("Track", self._on_follow_clicked)
                    menu.addAction("Clear", self._on_clear_clicked)
            else:
                menu.addAction(
                    "Set Target",
                    lambda p=payload: self._right_click_set_target(p, track=False),
                )
                menu.addAction(
                    "Set Target and Track",
                    lambda p=payload: self._right_click_set_target(p, track=True),
                )
            menu.addSeparator()
        elif tracking:
            menu.addAction("Stop Tracking", self._on_stop_clicked)
            menu.addAction("Clear", self._on_clear_clicked)
            menu.addSeparator()
        elif has_target:
            menu.addAction("Track", self._on_follow_clicked)
            menu.addAction("Clear", self._on_clear_clicked)
            menu.addSeparator()
        menu.addAction("Rename…", lambda: self.renameRequested.emit(self.camera_id))
        menu.addAction("Camera Info", lambda: self.infoRequested.emit(self.camera_id))
        menu.addSeparator()
        remove = menu.addAction("Remove Camera")
        remove.triggered.connect(lambda: self._client.removeCamera(self.camera_id))
        menu.exec(event.globalPos())

    def _right_click_set_target(self, payload: dict[str, Any], *, track: bool) -> None:
        try:
            self._apply_target_payload(payload)
            if track:
                self._client.enableTracking(self.camera_id, True)
        except Exception:  # noqa: BLE001
            log.debug("right-click target action failed", exc_info=True)
        self._refresh_overlay_state()

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._nudge(-1, 0, 0)
        elif key == Qt.Key.Key_Right:
            self._nudge(1, 0, 0)
        elif key == Qt.Key.Key_Up:
            self._nudge(0, 1, 0)
        elif key == Qt.Key.Key_Down:
            self._nudge(0, -1, 0)
        elif key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._client.clearTargetAndStop(self.camera_id)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            self._nudge(0, 0, 0)
        else:
            super().keyReleaseEvent(event)

    def _nudge(self, pan: int, tilt: int, zoom: int) -> None:
        try:
            self._client.ptzNudge(
                self.camera_id,
                pan * _NUDGE_SPEED,
                tilt * _NUDGE_SPEED,
                zoom * _NUDGE_SPEED,
            )
        except Exception:  # noqa: BLE001
            log.debug("ptzNudge failed", exc_info=True)
