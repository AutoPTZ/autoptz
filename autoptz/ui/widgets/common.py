"""Shared widgets for the native UI: collapsible groups, cost chips, helpers."""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T

log = logging.getLogger(__name__)

_QWIDGETSIZE_MAX = (1 << 24) - 1
_ANIM_MS = 125
_COLLAPSE_MS = 180  # expand/collapse height anim — a touch longer reads as smoother


def animate_widget_visibility(widget: QWidget, visible: bool, *, duration: int = _ANIM_MS) -> None:
    """Fade ``widget`` in/out without repeatedly restarting an active animation."""
    target = bool(visible)
    if getattr(widget, "_autoptz_fade_target", None) == target:
        return
    widget._autoptz_fade_target = target

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
    widget._autoptz_fade_anim = anim
    anim.start()


# ── layout sizing ─────────────────────────────────────────────────────────────


def visible_min_width(widget: QWidget) -> int:
    """A widget's minimum width AS IF it were visible, even if currently hidden.

    Qt layouts exclude a hidden widget's size from its PARENT layout's own
    minimum-size calculation — correct for content that's gone for good, but
    wrong for a panel that toggles an "empty state" vs. a "populated state"
    (an accordion section while collapsed; an info panel's no-camera-selected
    placeholder): the real content's width requirement must still count, or a
    panel measured while collapsed/empty silently under-reports, then is too
    narrow the instant real content appears (on expand, on selecting a camera,
    or when a clip-audit test forces every section open). Querying the
    widget's OWN layout directly sidesteps the exclusion, which only applies
    where something ELSE queries this widget as a child of its parent's layout.
    """
    lay = widget.layout()
    return lay.minimumSize().width() if lay is not None else widget.minimumSizeHint().width()


def scroll_chrome_width(scroll: QScrollArea) -> int:
    """The real frame + (possible) vertical-scrollbar overhead a scroll area adds.

    Queried from the live style/frame rather than guessed, so it stays correct
    on any platform/theme/DPI — not just whatever was on hand when someone last
    hand-picked a replacement pixel constant.
    """
    chrome = scroll.frameWidth() * 2
    if scroll.verticalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff:
        chrome += scroll.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent, None, scroll)
    return chrome


def scroll_content_min_width(scroll: QScrollArea) -> int:
    """The REAL minimum width a scroll area's content needs, from live metrics.

    ``QScrollArea.minimumSizeHint()`` does **not** take its contained widget into
    account — verified directly: a plain ``QScrollArea`` wrapping a 462px-wide
    label still reports ``~90px``, a small style-driven constant that has zero
    relationship to the content. Every side panel in this app is built as
    ``QVBoxLayout(self) → QScrollArea(resizable) → body-with-real-content``, so
    floating a panel's minimum width off the scroll area's (or the panel's own
    default) ``minimumSizeHint()`` silently under-budgets — which is exactly
    what happened: the previous code used flat pixel constants "measured" once
    against macOS/Linux font metrics, with no headroom for a platform whose
    default UI font renders wider at the same point size (Windows does).

    Compute the floor from the scrolled content's OWN layout instead — that is
    a bottom-up calculation from each child widget's real ``minimumSizeHint()``
    (font-metric-driven, so it's correct on any platform/DPI/scale) — plus the
    scroll area's real chrome (:func:`scroll_chrome_width`).
    """
    body = scroll.widget()
    content_w = visible_min_width(body) if body is not None else 0
    return content_w + scroll_chrome_width(scroll)


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
        self._height_anim.setDuration(_COLLAPSE_MS)
        self._height_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._height_anim.finished.connect(self._on_anim_finished)
        # Drive minimumHeight in lockstep with the animated maximumHeight so the
        # widget is pinned to ONE exact height each frame.  Animating only the max
        # (min stayed 0) let the layout pick any in-between height per frame, which
        # is what made the expand/collapse stutter and jump.
        self._height_anim.valueChanged.connect(self._on_anim_value)

    def _natural_height(self) -> int:
        """True expanded content height at the current width — side-effect-free.

        ``sizeHint().height()`` is width-independent and under-reports wrapped
        labels, so animating to it leaves a gap that *snaps* on finish (the "end
        pop").  Lift the height clamp, activate the layout, and read the real
        (height-for-width) size, then restore the clamp — all synchronously, so
        nothing renders mid-measure.
        """
        content = self._content
        prev_min, prev_max = content.minimumHeight(), content.maximumHeight()
        content.setMinimumHeight(0)
        content.setMaximumHeight(_QWIDGETSIZE_MAX)
        lay = content.layout()
        if lay is not None:
            lay.activate()
        width = content.width() or self.width()
        h = content.sizeHint().height()
        if lay is not None and lay.hasHeightForWidth() and width > 0:
            h = max(h, lay.heightForWidth(width))
        content.setMinimumHeight(prev_min)
        content.setMaximumHeight(prev_max)
        return max(1, h)

    def _on_anim_value(self, value: int) -> None:
        # Pin the widget to exactly the animated height (min == max) so the parent
        # layout can't reflow it to an in-between size mid-frame.
        self._content.setMinimumHeight(int(value))

    def _on_toggle(self, checked: bool) -> None:
        self._expanded = checked
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._height_anim.stop()
        if checked:
            # Clip to 0 BEFORE showing so the content never flashes at full height
            # for a frame (the "start pop"); then measure the true target and grow.
            self._content.setMinimumHeight(0)
            self._content.setMaximumHeight(0)
            self._content.setVisible(True)
            start, end = 0, self._natural_height()
        else:
            start = max(0, self._content.height() if self._content.isVisible() else 0)
            start = start or self._natural_height()
            end = 0
            self._content.setMinimumHeight(start)
            self._content.setMaximumHeight(start)
        self._height_anim.setStartValue(start)
        self._height_anim.setEndValue(end)
        self._height_anim.start()

    def _on_anim_finished(self) -> None:
        # Release the min floor so static content can size naturally; the content is
        # already at its natural height, so releasing max to MAX causes no jump.
        self._content.setMinimumHeight(0)
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

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """Report the width the EXPANDED content needs, even while collapsed.

        Collapsing an accordion section hides its HEIGHT — that's the entire
        point — it must never silently shrink the reported WIDTH floor too.  A
        panel built from several of these (most starting collapsed, e.g.
        PropertiesPanel) that floors its own minimum width off "whatever's
        visible right now" would under-report until the user (or a clip-audit
        test) expands a section, at which point the panel would suddenly be
        too narrow for its own content.  :func:`visible_min_width` reads
        ``_content``'s layout directly, which is unaffected by ``_content``'s
        own hidden flag (that flag only matters to whatever layout item
        queries ``_content`` as a child of its OWN parent), so this always
        reflects the real, font-metric-driven minimum regardless of the
        current toggle state.
        """
        hdr_w = self._toggle.minimumSizeHint().width()
        content = getattr(self, "_content", None)
        content_w = visible_min_width(content) if content is not None else 0
        height = super().minimumSizeHint().height()
        return QSize(max(hdr_w, content_w), height)


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
        img.scaled(
            QSize(size, size),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
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
    f = p.font()
    f.setPixelSize(int(size * 0.42))
    f.setBold(True)
    p.setFont(f)
    ch = (text.strip()[:1] or "?").upper()
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, ch)
    p.end()
    return pm
