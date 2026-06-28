"""MarkExitDialog — the deliberate Return / Quit choice when leaving AutoPTZ Mark.

A small modal that gives the user the explicit exit choice the OS close button
can't: **Return to AutoPTZ** (resume the suspended app) or **Quit AutoPTZ**, with
an optional *Save results* checkbox so the last run's report is written before
leaving.  The dialog is pure UI — the window reads :meth:`choice` /
:meth:`save_results` after :meth:`exec` and drives the actual return / quit /
persist.  ``Cancel`` returns :data:`None` (stay in Mark).

The *Save results* box is only enabled when a result actually exists (the window
passes ``has_result``); with nothing to save it stays unchecked and disabled so
the option never lies about what it will do.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T

# Choice constants (returned by .choice()).
RETURN = "return"
QUIT = "quit"


class MarkExitDialog(QDialog):
    """Return-to-AutoPTZ / Quit-AutoPTZ choice with an optional save-results box."""

    def __init__(self, *, has_result: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exit AutoPTZ Mark")
        self.setModal(True)
        self.setMinimumWidth(400)
        self._choice: str | None = None

        col = QVBoxLayout(self)
        col.setContentsMargins(24, 24, 24, 24)
        col.setSpacing(12)

        title = QLabel("Leaving AutoPTZ Mark")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        col.addWidget(title)

        blurb = QLabel("Return to AutoPTZ to resume the live app, or quit AutoPTZ entirely.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color: {T.CURRENT.subtext};")
        col.addWidget(blurb)

        self._save_box = QCheckBox("Save results before exiting")
        # Enable BEFORE checking so the box never lands in a checked-but-disabled
        # state (a setChecked-before-setEnabled ordering can leave that flicker).
        self._save_box.setEnabled(has_result)
        self._save_box.setChecked(has_result)
        if not has_result:
            self._save_box.setToolTip("Run a benchmark first to have results to save.")
        col.addWidget(self._save_box)

        col.addSpacing(4)
        btn_row = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        quit_btn = QPushButton("Quit AutoPTZ")
        quit_btn.clicked.connect(self._on_quit)
        return_btn = QPushButton("Return to AutoPTZ")
        return_btn.setProperty("accent", True)
        return_btn.setDefault(True)
        return_btn.clicked.connect(self._on_return)
        btn_row.addWidget(cancel)
        btn_row.addStretch(1)
        btn_row.addWidget(quit_btn)
        btn_row.addWidget(return_btn)
        col.addLayout(btn_row)

    # ── choices ──────────────────────────────────────────────────────────────────

    def _on_return(self) -> None:
        self._choice = RETURN
        self.accept()

    def _on_quit(self) -> None:
        self._choice = QUIT
        self.accept()

    def choice(self) -> str | None:
        """``"return"`` / ``"quit"`` after accept, or ``None`` if cancelled."""
        return self._choice

    def save_results(self) -> bool:
        """Whether the user asked to save the last run before exiting."""
        return bool(self._save_box.isChecked() and self._save_box.isEnabled())
