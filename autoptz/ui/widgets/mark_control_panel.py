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

from PySide6.QtCore import Qt, Signal
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
        # Object-named + styled-background so the global #markControlPanel rule
        # (surface_alt bg, top border) actually paints the bar.
        self.setObjectName("markControlPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        col = QVBoxLayout(self)
        col.setContentsMargins(12, 8, 12, 8)
        col.setSpacing(8)

        self._verdict_label = QLabel("Getting ready…")
        self._verdict_label.setObjectName("markVerdict")
        self._verdict_label.setWordWrap(True)
        col.addWidget(self._verdict_label)
        # Whether the verdict currently holds the FINAL score (highlighted) vs an
        # in-flight progress line (plain) — tracked so a theme flip re-applies the
        # right styling instead of reverting a finished score to the live look.
        self._verdict_final = False

        btn_row = QHBoxLayout()
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("markStopBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip("Stop the simulation early.")
        self._stop_btn.clicked.connect(self.stopClicked.emit)
        # Visible exit affordance: the user can deliberately leave Mark (Return /
        # Quit, optional save) instead of only via the OS close button.
        self._exit_btn = QPushButton("Exit Mark…")
        self._exit_btn.setObjectName("markExitBtn")
        self._exit_btn.setToolTip("Return to AutoPTZ or quit (optionally saving results).")
        self._exit_btn.clicked.connect(self.exitClicked.emit)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._exit_btn)
        col.addLayout(btn_row)
        # Per-widget restyle fallback so a Light/Dark flip repaints the verdict
        # (the global #markVerdict rule covers construction; this re-runs on theme
        # change — see MarkWindow wiring via common.on_theme_changed).
        self._restyle()

    def _restyle(self) -> None:
        """Refresh the verdict + button styling from the active palette.

        Re-run on every theme flip.  The verdict honors the current final/in-flight
        state (a finished score stays accent + bold, a live line stays plain), and the
        Stop / Exit buttons get prominent inline styling driven by theme tokens so a
        Light/Dark flip keeps Stop danger-filled and Exit a clear, obvious secondary
        (the global stylesheet rules can't be added from here, so they're inlined and
        re-applied here to survive theme flips).
        """
        if self._verdict_final:
            self._verdict_label.setStyleSheet(
                f"color: {T.TRACKING}; font-size: {T.fs(14)}px; font-weight: 700;"
            )
        else:
            self._verdict_label.setStyleSheet(
                f"color: {T.CURRENT.text}; font-size: {T.fs(14)}px; font-weight: 600;"
            )
        self._restyle_buttons()

    def _restyle_buttons(self) -> None:
        """Prominent inline styling for the Stop + Exit buttons (theme-token driven).

        Stop is a filled DANGER action (red, white text, no border) so stopping the run
        is unmistakable; it greys out cleanly when disabled (idle).  Exit Mark is a
        clear, obvious secondary: a bigger bordered button in the active palette's text
        / surface / border tokens, accent-bordered on hover.  Both are sized larger
        (bigger padding + bold) than a default button so they read as primary controls.
        """
        danger_hov = T.DANGER_HOVER
        self._stop_btn.setStyleSheet(
            f"QPushButton#markStopBtn {{ background: {T.DANGER}; color: {T.ACCENT_TEXT};"
            f" border: none; border-radius: {T.RADIUS}px;"
            f" padding: {T.fs(9)}px {T.fs(22)}px; font-size: {T.fs(14)}px; font-weight: 700; }}"
            f"QPushButton#markStopBtn:hover {{ background: {danger_hov}; }}"
            f"QPushButton#markStopBtn:disabled {{ background: {T.CURRENT.surface_alt};"
            f" color: {T.CURRENT.muted}; }}"
        )
        self._exit_btn.setStyleSheet(
            f"QPushButton#markExitBtn {{ background: {T.CURRENT.surface_alt};"
            f" color: {T.CURRENT.text}; border: 1px solid {T.CURRENT.border};"
            f" border-radius: {T.RADIUS}px; padding: {T.fs(9)}px {T.fs(22)}px;"
            f" font-size: {T.fs(14)}px; font-weight: 600; }}"
            f"QPushButton#markExitBtn:hover {{ background: {T.CURRENT.surface_hov};"
            f" border-color: {T.ACCENT.name()}; }}"
        )

    def set_verdict(self, text: str, *, final: bool = False) -> None:
        """Set the verdict line.

        ``final=True`` highlights the finished score (accent color + bold) so it
        stands out from the in-flight ``Ramping…`` progress; the default keeps the
        plain live styling.  The final state is remembered so a later theme flip
        (:meth:`_restyle`) re-applies the right look.
        """
        self._verdict_final = bool(final)
        self._verdict_label.setText(text)
        self._restyle()

    def set_running(self, running: bool) -> None:
        """Stop is usable only while the ramp runs; Exit is always usable."""
        self._stop_btn.setEnabled(running)
