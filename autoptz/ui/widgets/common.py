"""Shared widgets for the native UI: collapsible groups, cost chips, helpers."""
from __future__ import annotations

import base64
import logging
from typing import Any, Callable

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T

log = logging.getLogger(__name__)

_QWIDGETSIZE_MAX = (1 << 24) - 1
_ANIM_MS = 125


def animate_widget_visibility(widget: QWidget, visible: bool, *, duration: int = _ANIM_MS) -> None:
    """Fade ``widget`` in/out without repeatedly restarting an active animation."""
    target = bool(visible)
    if getattr(widget, "_autoptz_fade_target", None) == target:
        return
    setattr(widget, "_autoptz_fade_target", target)

    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)

    anim = getattr(widget, "_autoptz_fade_anim", None)
    if anim is not None:
        try:
            anim.stop()
        except Exception:  # noqa: BLE001
            pass

    if target:
        widget.setVisible(True)
    start = float(effect.opacity()) if widget.isVisible() else 0.0
    end = 1.0 if target else 0.0
    if abs(start - end) < 0.01:
        effect.setOpacity(end)
        widget.setVisible(target)
        return

    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(max(1, int(duration)))
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.setStartValue(start)
    anim.setEndValue(end)

    def _finish() -> None:
        if getattr(widget, "_autoptz_fade_target", target) == target:
            effect.setOpacity(end)
            widget.setVisible(target)

    anim.finished.connect(_finish)
    setattr(widget, "_autoptz_fade_anim", anim)
    anim.start()

# ── theme reactivity ──────────────────────────────────────────────────────────


def on_theme_changed(client: Any, slot: Callable[[], None]) -> None:
    """Call ``slot()`` whenever the user flips Light/Dark.

    Widgets that bake literal ``T.CURRENT.*`` colors into a ``setStyleSheet`` go
    stale when :class:`~autoptz.ui.theme.ThemeController` re-applies the global
    stylesheet (their per-widget literals still hold the old appearance's
    colors).  Factor that styling into a ``_restyle()`` method, call it once at
    construction, and pass it here so it re-runs on every theme change.

    Safe to call even if ``client`` lacks a ``themeChanged`` signal.
    """
    try:
        client.themeChanged.connect(lambda *_: slot())
    except Exception:  # noqa: BLE001
        log.debug("on_theme_changed: connect failed", exc_info=True)


# ── cost chip ───────────────────────────────────────────────────────────────────

_COST_COLORS = {"light": T.TRACKING, "medium": T.WARNING, "heavy": T.ERROR}


class CostChip(QLabel):
    """A small colored pill labelling a setting's relative cost (Light/Med/Heavy)."""

    def __init__(self, cost: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = _COST_COLORS.get((cost or "light").lower(), T.TRACKING)
        self.setText((cost or "light").upper())
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self._restyle()

    def _restyle(self) -> None:
        self.setStyleSheet(
            f"color: {self._color}; border: 1px solid {self._color}; border-radius: 7px;"
            f"padding: 1px 6px; font-size: {T.fs(9)}px; font-weight: 700;"
        )


# ── buttons (one reusable set, styled by the global stylesheet) ─────────────────


class AccentButton(QPushButton):
    """Primary action button — accent-filled via the global ``[accent]`` rule."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("accent", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class DangerButton(QPushButton):
    """Destructive action button — red text that fills red on hover.

    Styled entirely by the global ``QPushButton[danger="true"]`` rule, so it
    tracks light/dark for free (no per-widget literal colors that go stale).
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("danger", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class IconButton(QToolButton):
    """A compact square icon-only button — the single delete/icon affordance.

    One consistent control for small glyph actions (delete, discard, close),
    replacing the assorted hand-styled red squares/circles.  ``danger=True``
    turns the hover fill red for destructive actions.  Styled by the global
    ``QToolButton#iconButton`` rule.
    """

    def __init__(
        self,
        glyph: str,
        *,
        tip: str = "",
        danger: bool = False,
        size: int = 26,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("iconButton")
        self.setText(glyph)
        self.setProperty("danger", bool(danger))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        side = T.fs(size)
        self.setFixedSize(side, side)
        if tip:
            self.setToolTip(tip)


def section_label(text: str) -> QLabel:
    """An uppercase muted caption (palette-driven so it tracks light/dark).

    Color/metrics live in the GLOBAL stylesheet (``QLabel#sectionCaption`` in
    :func:`~autoptz.ui.theme.build_stylesheet`) so the caption stays legible when
    the appearance flips, with zero per-widget theme wiring.
    """
    lab = QLabel(text.upper())
    lab.setObjectName("sectionCaption")
    return lab


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    # Styled via the GLOBAL stylesheet (``QFrame#hline``) so it tracks the theme.
    line.setObjectName("hline")
    return line


# ── help badge ────────────────────────────────────────────────────────────────


class HelpBadge(QLabel):
    """A compact circular "?" badge that reveals help text on hover *and* click.

    Self-contained so any panel can drop one beside a section header or field:
    ``head.addWidget(HelpBadge("Explains what this does"))``.  The help text is
    the widget's tooltip; hovering shows it, and clicking shows the **exact same**
    native ``QToolTip`` at the badge (so trackpad/touch users who never "hover"
    get the identical look — one style, one code path).  Styling is palette-driven
    through the ``helpBadge`` objectName in
    :func:`~autoptz.ui.theme.build_stylesheet`, so it tracks light/dark with no
    per-widget rewiring.
    """

    def __init__(self, tip: str, parent: QWidget | None = None) -> None:
        super().__init__("?", parent)
        self.setObjectName("helpBadge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tip = tip or ""
        self.setToolTip(self._tip)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        side = T.fs(16)
        self.setFixedSize(side, side)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(180)
        self._hover_timer.timeout.connect(self._show_tip)

    def set_help(self, tip: str) -> None:
        """Update the help text (for badges whose content is live, e.g. fps stats)."""
        self._tip = tip or ""
        self.setToolTip(self._tip)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        """Show the same native tooltip on click as on hover (one consistent look)."""
        self._show_tip()
        super().mousePressEvent(event)

    def enterEvent(self, event: QEvent) -> None:  # noqa: N802
        self._hover_timer.start()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802
        self._hover_timer.stop()
        QToolTip.hideText()
        super().leaveEvent(event)

    def _show_tip(self) -> None:
        if self._tip:
            QToolTip.showText(self.mapToGlobal(self.rect().bottomLeft()), self._tip, self)


# ── collapsible group ─────────────────────────────────────────────────────────


class CollapsibleGroup(QWidget):
    """A titled section with a chevron header that expands/collapses its body.

    Add content to :pyattr:`body` (a ``QVBoxLayout``).  The macOS-inset look comes
    from the surrounding panel styling; here we keep the header lightweight.
    """

    def __init__(self, title: str, expanded: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._expanded = expanded

        # Card outline so each section is a clearly bounded block.  Styling lives
        # entirely in the GLOBAL stylesheet (``#collGroup`` + ``#collGroupHeader``
        # in theme.build_stylesheet), which ThemeController re-applies on every
        # Light↔Dark flip — so the group + its header track the appearance for
        # free, instead of baking literal colors that go stale after a flip.
        self.setObjectName("collGroup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle = QToolButton(self)
        self._toggle.setObjectName("collGroupHeader")
        self._toggle.setText(title.upper())
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.clicked.connect(self._on_toggle)
        outer.addWidget(self._toggle)

        self._content = QWidget(self)
        self.body = QVBoxLayout(self._content)
        self.body.setContentsMargins(T.fs(12), T.fs(10), T.fs(12), T.fs(12))
        self.body.setSpacing(T.fs(8))
        # Let the maximumHeight animation drive the size all the way to 0 — without
        # an explicit 0 minimum the content's minimumSizeHint floors the shrink and
        # the last frame snaps to 0, which reads as the "jumping" jitter.
        self._content.setMinimumHeight(0)
        self._content.setVisible(expanded)
        self._content.setMaximumHeight(_QWIDGETSIZE_MAX if expanded else 0)
        outer.addWidget(self._content)
        self._height_anim = QPropertyAnimation(self._content, b"maximumHeight", self)
        self._height_anim.setDuration(_ANIM_MS)
        self._height_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._height_anim.finished.connect(self._on_anim_finished)

    def _natural_height(self) -> int:
        """Accurate expanded height, including height-for-width (word-wrap) content.

        ``sizeHint().height()`` is width-independent and under-reports wrapped
        labels, so the expand animation would stop short and then *snap* to the
        real height on finish — the "jump".  Prefer the layout's
        ``heightForWidth`` at the content's actual width when available."""
        lay = self._content.layout()
        width = self._content.width() or self.width()
        hfw = -1
        if lay is not None and lay.hasHeightForWidth() and width > 0:
            hfw = lay.heightForWidth(width)
        return max(1, self._content.sizeHint().height(), hfw)

    def _on_toggle(self, checked: bool) -> None:
        self._expanded = checked
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._height_anim.stop()
        natural = self._natural_height()
        current = max(0, self._content.height() if self._content.isVisible() else 0)
        if checked:
            self._content.setVisible(True)
            self._content.setMaximumHeight(current)
            self._height_anim.setStartValue(current)
            self._height_anim.setEndValue(natural)
        else:
            start = current or min(self._content.maximumHeight(), natural)
            self._content.setMaximumHeight(start)
            self._height_anim.setStartValue(start)
            self._height_anim.setEndValue(0)
        self._height_anim.start()

    def _on_anim_finished(self) -> None:
        if self._expanded:
            self._content.setVisible(True)
            self._content.setMaximumHeight(_QWIDGETSIZE_MAX)
        else:
            self._content.setMaximumHeight(0)
            self._content.setVisible(False)

    def add_widget(self, w: QWidget) -> None:
        self.body.addWidget(w)
        if self._expanded:
            self._content.setMaximumHeight(_QWIDGETSIZE_MAX)


# ── thumbnails ──────────────────────────────────────────────────────────────────


def data_uri_to_pixmap(uri: str, size: int = 56, circular: bool = True) -> QPixmap | None:
    """Decode a ``data:image/...;base64,…`` URI to a (optionally circular) QPixmap."""
    if not uri or "," not in uri:
        return None
    try:
        b64 = uri.split(",", 1)[1]
        raw = base64.b64decode(b64)
        img = QImage.fromData(raw)
        if img.isNull():
            return None
    except Exception:  # noqa: BLE001
        log.debug("data_uri_to_pixmap failed", exc_info=True)
        return None

    pm = QPixmap.fromImage(
        img.scaled(QSize(size, size), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                   Qt.TransformationMode.SmoothTransformation)
    )
    # center-crop to square
    if pm.width() != size or pm.height() != size:
        x = max(0, (pm.width() - size) // 2)
        y = max(0, (pm.height() - size) // 2)
        pm = pm.copy(x, y, size, size)
    if not circular:
        return pm

    rounded = QPixmap(size, size)
    rounded.fill(Qt.GlobalColor.transparent)
    p = QPainter(rounded)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addEllipse(0, 0, size, size)
    p.setClipPath(path)
    p.drawPixmap(0, 0, pm)
    p.end()
    return rounded


def letter_avatar(text: str, size: int = 56) -> QPixmap:
    """A circular monogram avatar fallback for an identity with no photo."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(T.CURRENT.surface_hov))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(0, 0, size, size)
    p.setPen(QColor(T.CURRENT.text))
    f = p.font(); f.setPixelSize(int(size * 0.42)); f.setBold(True)
    p.setFont(f)
    ch = (text.strip()[:1] or "?").upper()
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, ch)
    p.end()
    return pm
