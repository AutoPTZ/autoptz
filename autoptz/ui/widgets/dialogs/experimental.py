"""ExperimentalFeaturesDialog — toggle curated experimental AUTOPTZ_* flags.

Selections persist via the client (ConfigStore key ``experimental_features``)
and are applied to ``os.environ`` by the supervisor at the next engine start —
this dialog never mutates the environment or restarts the engine itself.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autoptz.engine.runtime.experimental_flags import (
    EXPERIMENTAL_FLAGS,
    TRACKING_DEFAULT_FIELDS,
    ExperimentalFlag,
)
from autoptz.ui import theme as T
from autoptz.ui.widgets.common import HelpBadge, hline, section_label

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
        self.setMinimumWidth(560)

        self._bool_boxes: dict[str, QCheckBox] = {}
        self._choice_combos: dict[str, QComboBox] = {}
        self._tracking_boxes: dict[str, QCheckBox] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(8)

        intro = QLabel(
            "These features are experimental and may change or be removed. Most "
            "are read when the engine starts, so a restart is needed for changes "
            "to take effect."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {T.CURRENT.subtext};")
        outer.addWidget(intro)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)
        body = QWidget()
        scroll.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(4)

        root.addWidget(section_label("Engine flags"))
        for flag in EXPERIMENTAL_FLAGS:
            root.addWidget(self._build_flag_row(flag))

        root.addWidget(hline())
        th = QHBoxLayout()
        th.addWidget(section_label("New-camera tracking defaults"))
        th.addWidget(
            HelpBadge(
                "These set the defaults applied to cameras you add from now on. "
                "Existing cameras keep their current per-camera setting."
            )
        )
        th.addStretch(1)
        root.addLayout(th)
        for name, label, desc, default in TRACKING_DEFAULT_FIELDS:
            root.addWidget(self._build_tracking_row(name, label, desc, default))
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
        self._apply_btn.clicked.connect(self._apply)
        self._restore_btn.clicked.connect(self._restore_defaults)
        close_btn.clicked.connect(self.reject)
        outer.addWidget(buttons)

        self._load()

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
        else:
            lay.addWidget(QLabel(f"<b>{flag.label}</b>"), 0, 0)
            combo = QComboBox()
            for choice in flag.choices:
                combo.addItem("(auto)" if choice == "" else choice, choice)
            combo.setToolTip(flag.description)
            self._choice_combos[flag.env_key] = combo
            lay.addWidget(combo, 0, 1, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(HelpBadge(flag.description), 0, 2)
        if flag.restart_required:
            lay.addWidget(_restart_badge(), 0, 3, Qt.AlignmentFlag.AlignRight)
        desc = QLabel(
            f"<span style='color:{T.CURRENT.subtext}'>{flag.description}"
            f" Default: {'(auto)' if flag.default == '' else flag.default}.</span>"
        )
        desc.setTextFormat(Qt.TextFormat.RichText)
        desc.setWordWrap(True)
        lay.addWidget(desc, 1, 0, 1, 4)
        lay.setColumnStretch(0, 1)
        return row

    def _build_tracking_row(self, name: str, label: str, desc: str, default: bool) -> QFrame:
        row = QFrame()
        lay = QGridLayout(row)
        lay.setContentsMargins(4, 6, 4, 6)
        box = QCheckBox(label)
        box.setToolTip(desc)
        self._tracking_boxes[name] = box
        lay.addWidget(box, 0, 0)
        lay.addWidget(HelpBadge(desc), 0, 1, Qt.AlignmentFlag.AlignRight)
        detail = QLabel(
            f"<span style='color:{T.CURRENT.subtext}'>{desc} Default: "
            f"{'on' if default else 'off'}.</span>"
        )
        detail.setTextFormat(Qt.TextFormat.RichText)
        detail.setWordWrap(True)
        lay.addWidget(detail, 1, 0, 1, 2)
        lay.setColumnStretch(0, 1)
        return row

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
            else:
                combo = self._choice_combos[flag.env_key]
                idx = combo.findData(value)
                combo.setCurrentIndex(idx if idx >= 0 else combo.findData(flag.default))
        for name, _label, _desc, default in TRACKING_DEFAULT_FIELDS:
            self._tracking_boxes[name].setChecked(bool(saved.get(name, default)))

    def _collect(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for flag in EXPERIMENTAL_FLAGS:
            if flag.kind == "bool":
                out[flag.env_key] = "1" if self._bool_boxes[flag.env_key].isChecked() else "0"
            else:
                out[flag.env_key] = self._choice_combos[flag.env_key].currentData()
        for name, _label, _desc, _default in TRACKING_DEFAULT_FIELDS:
            out[name] = bool(self._tracking_boxes[name].isChecked())
        return out

    def _apply(self) -> None:
        _safe(lambda: self._client.setSetting("experimental_features", self._collect()), None)

    def _restore_defaults(self) -> None:
        for flag in EXPERIMENTAL_FLAGS:
            if flag.kind == "bool":
                self._bool_boxes[flag.env_key].setChecked(flag.default not in ("0", "", "false"))
            else:
                combo = self._choice_combos[flag.env_key]
                combo.setCurrentIndex(combo.findData(flag.default))
        for name, _label, _desc, default in TRACKING_DEFAULT_FIELDS:
            self._tracking_boxes[name].setChecked(bool(default))
        self._apply()
