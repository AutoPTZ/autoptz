"""JoystickPad — a draggable virtual thumbstick for manual PTZ control.

Emits :pysig:`moved(pan, tilt)` with both axes normalized to ``[-1, 1]`` (pan:
right = +1, tilt: up = +1) continuously while the thumb is held — even when the
mouse stops moving — so the engine's manual-override window stays alive.  On
release the thumb springs back to centre and a final ``(0, 0)`` is emitted.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from autoptz.ui import theme as T


class JoystickPad(QWidget):
    """Spring-return thumbstick; reports pan/tilt velocity in ``[-1, 1]``."""

    moved = Signal(float, float)

    def __init__(self, size: int = 120, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._size = size
        self._knob_r = max(10, size // 6)
        self._max = size / 2 - self._knob_r
        self._vec = QPointF(0.0, 0.0)   # thumb offset from centre, in pixels
        self._dragging = False
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(
            "Manual PTZ joystick — drag to pan/tilt the camera; release to stop. "
            "Holding it briefly pauses auto-tracking."
        )
        # Re-emit the held vector so continuous-velocity commands keep flowing.
        self._repeat = QTimer(self)
        self._repeat.setInterval(100)
        self._repeat.timeout.connect(self._emit)

    # ── geometry helpers ─────────────────────────────────────────────────────────

    def _centre(self) -> QPointF:
        return QPointF(self._size / 2, self._size / 2)

    def _set_from_pos(self, pos: QPointF) -> None:
        d = pos - self._centre()
        # Clamp the offset to the travel radius.
        dist = (d.x() ** 2 + d.y() ** 2) ** 0.5
        if dist > self._max and dist > 0:
            d = QPointF(d.x() / dist * self._max, d.y() / dist * self._max)
        self._vec = d
        self.update()

    def _emit(self) -> None:
        pan = self._vec.x() / self._max if self._max else 0.0
        tilt = -self._vec.y() / self._max if self._max else 0.0   # screen y is down
        self.moved.emit(float(pan), float(tilt))

    # ── interaction ──────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging = True
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self._set_from_pos(QPointF(event.position()))
        self._emit()
        self._repeat.start()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self._set_from_pos(QPointF(event.position()))
            self._emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging = False
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._repeat.stop()
        self._vec = QPointF(0.0, 0.0)
        self.update()
        self.moved.emit(0.0, 0.0)

    # ── painting ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = self._centre()
        base_r = self._size / 2 - 2

        # Base well.
        p.setPen(QColor(T.CURRENT.border))
        p.setBrush(QColor(T.CURRENT.surface_alt))
        p.drawEllipse(c, base_r, base_r)
        # Crosshair guides.
        p.setPen(QColor(T.CURRENT.border))
        p.drawLine(QPointF(c.x() - base_r, c.y()), QPointF(c.x() + base_r, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - base_r), QPointF(c.x(), c.y() + base_r))

        # Thumb.
        knob = c + self._vec
        accent = T.ACCENT if self._dragging else QColor(T.CURRENT.border_hov)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(accent)
        p.drawEllipse(knob, self._knob_r, self._knob_r)
        p.end()
