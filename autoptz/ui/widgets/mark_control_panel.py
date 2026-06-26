"""MarkControlPanel — live verdict + Stop + Exit for an AutoPTZ Mark run.

A deliberately minimal control surface for the AutoPTZ Mark window.  Everything
the run needs (source, camera count, FPS target, resolution, model) is chosen
once in the pre-flight, so this panel does NOT re-ask any of it.  It shows only:

  * a live progress / verdict line ("Ramping… N cameras @ X fps" → final score);
  * a **Stop** button (enabled only while the ramp is running); and
  * an **Exit Mark…** button (always enabled → Return / Quit with optional save).

The ramp auto-starts when the window is shown, so there is no Start button.  Pure
widget logic — all wiring is via signals so the window just connects them.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T


class MarkControlPanel(QWidget):
    """Live verdict + Stop/Exit for an AutoPTZ Mark run (no source/count re-ask)."""

    stopClicked = Signal()
    exitClicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        col = QVBoxLayout(self)

        self._verdict_label = QLabel("Getting ready…")
        self._verdict_label.setWordWrap(True)
        self._verdict_label.setStyleSheet(
            f"color: {T.CURRENT.text}; font-size: {T.fs(14)}px; font-weight: 600;"
        )
        col.addWidget(self._verdict_label)

        btn_row = QHBoxLayout()
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip("Stop the simulation early.")
        self._stop_btn.clicked.connect(self.stopClicked.emit)
        # Visible exit affordance: the user can deliberately leave Mark (Return /
        # Quit, optional save) instead of only via the OS close button.
        self._exit_btn = QPushButton("Exit Mark…")
        self._exit_btn.setToolTip("Return to AutoPTZ or quit (optionally saving results).")
        self._exit_btn.clicked.connect(self.exitClicked.emit)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._exit_btn)
        col.addLayout(btn_row)

    def set_verdict(self, text: str) -> None:
        self._verdict_label.setText(text)

    def set_running(self, running: bool) -> None:
        """Stop is usable only while the ramp runs; Exit is always usable."""
        self._stop_btn.setEnabled(running)
