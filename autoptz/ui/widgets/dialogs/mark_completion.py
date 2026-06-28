"""MarkCompletionDialog — the unified choice shown when an AutoPTZ Mark ramp finishes.

When the ramp completes naturally the window shows ONE modal that both reports the
verdict (and, once Phase 3 lands, the rating word + the one-line score math) and
offers every next step in a single place:

  * **Save results…** — write the report (the window owns the actual Save-As / writer
    via the ``saveRequested`` signal; saving does NOT close the dialog so the user can
    save and *then* choose Return / Quit);
  * **Open results** — reveal the saved file / folder (``openRequested``); enabled only
    once something has been saved (otherwise it can't lie about what it would open);
  * **Quit AutoPTZ** — leave the app;
  * **Return to AutoPTZ** (accent, default) — resume the suspended live app.

Like :class:`MarkExitDialog` the dialog is pure UI: the window reads :meth:`choice`
after :meth:`exec` and connects to ``saveRequested`` / ``openRequested`` so all file
I/O stays in the window.  ``rating`` / ``reason`` are optional — Phase 2 renders the
score-only verdict; Phase 3 supplies the human rating word and math reason.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T

# Choice constants (returned by .choice()) — mirror MarkExitDialog.
RETURN = "return"
QUIT = "quit"


class MarkCompletionDialog(QDialog):
    """Return / Quit / Open / Save choice shown on a finished ramp."""

    #: Emitted when the user clicks *Save results…* — the window does the writing.
    saveRequested = Signal()
    #: Emitted when the user clicks *Open results* — the window reveals the file/folder.
    openRequested = Signal()

    def __init__(
        self,
        *,
        verdict: str,
        rating: str = "",
        reason: str = "",
        has_saved: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AutoPTZ Mark — Results")
        self.setModal(True)
        self.setMinimumWidth(440)
        self._choice: str | None = None

        col = QVBoxLayout(self)
        col.setContentsMargins(24, 24, 24, 24)
        col.setSpacing(8)

        title = QLabel("Benchmark complete")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        col.addWidget(title)

        # The big verdict / score line (always present).
        verdict_lbl = QLabel(verdict)
        verdict_lbl.setWordWrap(True)
        verdict_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        verdict_lbl.setStyleSheet("font-size: 15px; font-weight: 600;")
        col.addWidget(verdict_lbl)

        # The rating word (bold accent) — Phase 3 supplies it; blank in Phase 2.
        if rating:
            rating_lbl = QLabel(rating)
            rating_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            rating_lbl.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {T.TRACKING};")
            col.addWidget(rating_lbl)

        # The one-line math reason (subtext) — Phase 3 supplies it; blank in Phase 2.
        if reason:
            reason_lbl = QLabel(reason)
            reason_lbl.setWordWrap(True)
            reason_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            reason_lbl.setStyleSheet(f"color: {T.CURRENT.subtext};")
            col.addWidget(reason_lbl)

        col.addSpacing(8)
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save results…")
        save_btn.clicked.connect(self._on_save)
        self._open_btn = QPushButton("Open results")
        self._open_btn.clicked.connect(self._on_open)
        quit_btn = QPushButton("Quit AutoPTZ")
        quit_btn.clicked.connect(self._on_quit)
        return_btn = QPushButton("Return to AutoPTZ")
        return_btn.setProperty("accent", True)
        return_btn.setDefault(True)
        return_btn.clicked.connect(self._on_return)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self._open_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(quit_btn)
        btn_row.addWidget(return_btn)
        col.addLayout(btn_row)

        self.set_saved(has_saved)

    # ── live state ────────────────────────────────────────────────────────────────

    def set_saved(self, saved: bool) -> None:
        """Enable / disable *Open results* (the window calls this after a save lands)."""
        self._open_btn.setEnabled(bool(saved))
        self._open_btn.setToolTip("" if saved else "Save results first.")

    # ── choices ──────────────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        # Save is a side action — it does NOT close the dialog; the window handles
        # the Save-As + writer and may call set_saved(True) to light up Open.
        self.saveRequested.emit()

    def _on_open(self) -> None:
        self.openRequested.emit()

    def _on_return(self) -> None:
        self._choice = RETURN
        self.accept()

    def _on_quit(self) -> None:
        self._choice = QUIT
        self.accept()

    def choice(self) -> str | None:
        """``"return"`` / ``"quit"`` after accept, or ``None`` if dismissed."""
        return self._choice
