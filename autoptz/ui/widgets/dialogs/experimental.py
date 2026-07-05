"""ExperimentalFeaturesDialog — toggle curated experimental AUTOPTZ_* flags.

Selections persist via the client (ConfigStore key ``experimental_features``)
and are applied to ``os.environ`` by the supervisor at the next engine start —
this dialog never mutates the environment or restarts the engine itself.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autoptz.engine.runtime.experimental_flags import (
    EXPERIMENTAL_FLAGS,
    ExperimentalFlag,
)
from autoptz.ui import theme as T
from autoptz.ui.widgets.common import HelpBadge, scroll_content_min_width, section_label

# Section headers, in display order.
_SECTION_ORDER = ("Experiments", "Devices & tuning", "Model overrides", "Diagnostics")

log = logging.getLogger(__name__)


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _restart_badge() -> QLabel:
    pill = QLabel("Restart required")
    pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
    pill.setStyleSheet(
        f"color: {T.WARNING}; border: 1px solid {T.WARNING};"
        f" border-radius: 8px; padding: 1px 8px; font-size: {T.fs(9)}px;"
        " font-weight: 700;"
    )
    return pill


class ExperimentalFeaturesDialog(QDialog):
    """Curated experimental flags + per-camera tracking defaults."""

    def __init__(self, client: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self.setWindowTitle("Experimental Features")
        self.setModal(True)
        self.resize(720, 620)

        self._bool_boxes: dict[str, QCheckBox] = {}
        self._choice_combos: dict[str, QComboBox] = {}
        self._text_fields: dict[str, QLineEdit] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(8)

        intro = QLabel(
            "Optional engine features and overrides, grouped by area. They are read "
            "when the engine starts, so use Apply and restart when prompted for "
            "changes to take effect."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {T.CURRENT.subtext};")
        outer.addWidget(intro)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)
        body = QWidget()
        scroll.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(12)

        by_section: dict[str, list[ExperimentalFlag]] = {}
        for flag in EXPERIMENTAL_FLAGS:
            by_section.setdefault(flag.section, []).append(flag)
        # Known sections in fixed order, then any unexpected section appended.
        ordered = [s for s in _SECTION_ORDER if s in by_section]
        ordered += [s for s in by_section if s not in _SECTION_ORDER]
        for section in ordered:
            # Each section is a distinct elevated card so the four groups read as
            # separate panels rather than one flat list.
            card = QFrame()
            card.setObjectName("expCard")
            card.setStyleSheet(
                f"QFrame#expCard {{ background: {T.CURRENT.surface_alt};"
                f" border: 1px solid {T.CURRENT.border}; border-radius: 10px; }}"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 12, 14, 12)
            cl.setSpacing(2)
            cl.addWidget(section_label(section))
            for flag in by_section[section]:
                cl.addWidget(self._build_flag_row(flag))
            root.addWidget(card)
        root.addStretch(1)

        note = QLabel("Some changes need a restart to take effect.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {T.WARNING};")
        outer.addWidget(note)

        buttons = QDialogButtonBox()
        self._apply_btn = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        self._apply_btn.setProperty("accent", True)
        self._restore_btn = buttons.addButton(
            "Restore defaults", QDialogButtonBox.ButtonRole.ResetRole
        )
        close_btn = buttons.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        self._apply_btn.clicked.connect(self._on_apply)
        self._restore_btn.clicked.connect(self._restore_defaults)
        close_btn.clicked.connect(self.reject)
        outer.addWidget(buttons)

        self._load()
        # Baseline for "did the user change anything since this was last applied?"
        # — drives whether Apply offers a restart (no nag when nothing changed).
        self._applied_snapshot = self._collect()

        # Real floor: the scrolled content's live layout minimum (a row is
        # [label + editor + Browse + help + "Restart required" badge]) plus
        # this layout's own margins and the scroll area's chrome. Previously a
        # flat ``setMinimumWidth(680)`` — a guess "measured" once on
        # macOS/Linux — which silently under-budgets on any font that renders
        # wider at the same point size (Windows' default UI font measurably
        # does; this is what clipped section captions/badges on Windows CI
        # while macOS/Linux stayed green). Computed AFTER every section row is
        # built, so it reflects the real (font-metric-driven) requirement.
        m = outer.contentsMargins()
        self.setMinimumWidth(scroll_content_min_width(scroll) + m.left() + m.right())

    # ── row builders ─────────────────────────────────────────────────────────

    def _build_flag_row(self, flag: ExperimentalFlag) -> QFrame:
        row = QFrame()
        lay = QGridLayout(row)
        lay.setContentsMargins(4, 8, 4, 8)
        lay.setHorizontalSpacing(10)
        if flag.kind == "bool":
            box = QCheckBox(flag.label)
            box.setToolTip(flag.description)
            self._bool_boxes[flag.env_key] = box
            lay.addWidget(box, 0, 0)
        elif flag.kind == "choice":
            lay.addWidget(QLabel(f"<b>{flag.label}</b>"), 0, 0)
            combo = QComboBox()
            for choice in flag.choices:
                combo.addItem("(auto)" if choice == "" else choice, choice)
            combo.setToolTip(flag.description)
            self._choice_combos[flag.env_key] = combo
            lay.addWidget(combo, 0, 1, Qt.AlignmentFlag.AlignLeft)
        else:  # text / path — free-form line edit (path adds a Browse button)
            lay.addWidget(QLabel(f"<b>{flag.label}</b>"), 0, 0)
            edit = QLineEdit()
            edit.setToolTip(flag.description)
            edit.setPlaceholderText("(default)")
            self._text_fields[flag.env_key] = edit
            lay.addWidget(edit, 0, 1)
            if flag.kind == "path":
                browse = QPushButton("Browse…")
                browse.clicked.connect(lambda _=False, e=edit: self._browse_path(e))
                lay.addWidget(browse, 0, 2, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(HelpBadge(flag.description), 0, 4)
        if flag.restart_required:
            lay.addWidget(_restart_badge(), 0, 5, Qt.AlignmentFlag.AlignRight)
        desc = QLabel(
            f"<span style='color:{T.CURRENT.subtext}'>{flag.description}"
            f" Default: {'(auto)' if flag.default == '' else flag.default}.</span>"
        )
        desc.setTextFormat(Qt.TextFormat.RichText)
        desc.setWordWrap(True)
        lay.addWidget(desc, 1, 0, 1, 6)
        lay.setColumnStretch(1, 1)
        return row

    def _browse_path(self, edit: QLineEdit) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "Choose a model file")
        if path:
            edit.setText(path)

    # ── state <-> widgets ──────────────────────────────────────────────────────

    def _saved(self) -> dict[str, Any]:
        got = _safe(lambda: self._client.getSetting("experimental_features", {}), {}) or {}
        return dict(got) if isinstance(got, dict) else {}

    def _load(self) -> None:
        saved = self._saved()
        for flag in EXPERIMENTAL_FLAGS:
            value = str(saved.get(flag.env_key, flag.default))
            if flag.kind == "bool":
                self._bool_boxes[flag.env_key].setChecked(value not in ("0", "", "false"))
            elif flag.kind == "choice":
                combo = self._choice_combos[flag.env_key]
                idx = combo.findData(value)
                combo.setCurrentIndex(idx if idx >= 0 else combo.findData(flag.default))
            else:
                self._text_fields[flag.env_key].setText(value)

    def _collect(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for flag in EXPERIMENTAL_FLAGS:
            if flag.kind == "bool":
                out[flag.env_key] = "1" if self._bool_boxes[flag.env_key].isChecked() else "0"
            elif flag.kind == "choice":
                out[flag.env_key] = self._choice_combos[flag.env_key].currentData()
            else:
                out[flag.env_key] = self._text_fields[flag.env_key].text().strip()
        return out

    def _apply(self) -> None:
        """Persist the current selection.  Pure — no UI side effects (callers/tests
        rely on this), the visible feedback + restart prompt live in `_on_apply`."""
        _safe(lambda: self._client.setSetting("experimental_features", self._collect()), None)

    def _on_apply(self) -> None:
        """Apply button handler: persist, acknowledge the click, then — only if the
        user actually changed something since the last apply — offer a restart."""
        new = self._collect()
        changed = new != self._applied_snapshot
        self._apply()
        self._applied_snapshot = new
        self._flash_applied()
        if changed and self._confirm_restart():
            self._do_restart()

    def _flash_applied(self) -> None:
        """Briefly show "Applied ✓" on the button so the click reads as effective."""
        self._apply_btn.setText("Applied ✓")
        self._apply_btn.setEnabled(False)
        QTimer.singleShot(1500, self._reset_apply_button)

    def _reset_apply_button(self) -> None:
        self._apply_btn.setText("Apply")
        self._apply_btn.setEnabled(True)

    def _confirm_restart(self) -> bool:
        """Ask the operator whether to restart the app now.  Returns True for yes.

        Isolated (and overridden in tests) so the rest of the Apply flow stays
        free of a blocking modal.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Restart required")
        box.setText("Experimental settings saved.")
        box.setInformativeText(
            "AutoPTZ needs to restart for these changes to take effect. Restart now?"
        )
        restart_btn = box.addButton("Restart Now", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(restart_btn)
        box.exec()
        return box.clickedButton() is restart_btn

    def _do_restart(self) -> None:
        """Relaunch the application so every flag is re-read from a clean process.

        A full relaunch (not just an engine restart) is the honest apply: several
        flags bind library thread pools / env at *import* time, which only a fresh
        process resets.  Isolated (and overridden in tests) so it never fires there.
        """
        import sys

        from PySide6.QtCore import QCoreApplication, QProcess

        self.accept()
        if getattr(sys, "frozen", False):
            program, args = sys.executable, sys.argv[1:]
        else:
            program, args = sys.executable, ["-m", "autoptz", *sys.argv[1:]]
        try:
            QProcess.startDetached(program, args)
        except Exception:  # noqa: BLE001 — relaunch is best-effort; still quit so the
            log.warning("relaunch spawn failed; quitting anyway", exc_info=True)
        QTimer.singleShot(0, QCoreApplication.quit)

    def _restore_defaults(self) -> None:
        for flag in EXPERIMENTAL_FLAGS:
            if flag.kind == "bool":
                self._bool_boxes[flag.env_key].setChecked(flag.default not in ("0", "", "false"))
            elif flag.kind == "choice":
                combo = self._choice_combos[flag.env_key]
                combo.setCurrentIndex(combo.findData(flag.default))
            else:
                self._text_fields[flag.env_key].setText(flag.default)
        self._apply()
        self._applied_snapshot = self._collect()
