"""MarkPreflightDialog — the friendly pre-flight notice for AutoPTZ Mark.

A plain, jargon-free setup screen: a one-line intro, the source + profile, and
five dropdowns (Max cameras, Target FPS, Time per step, Resolution, Model) with a
live run-time estimate.  Its Start button asks a confirm ("This suspends AutoPTZ
and runs the simulation. Continue?") before accepting, since entering Mark
suspends the live app.  On accept it yields a :class:`MarkSession`.

Pure widget logic, offscreen-testable: construct, poke control values, read
:meth:`session`.  The NDI source option is disabled (with a "requires cyndilib"
suffix) when ``ndi_sim_available()`` is False, so the window never tries to spin
up NDI senders the host can't build.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from autoptz.benchmark.ndi_sim import ndi_sim_available
from autoptz.ui import theme as T
from autoptz.ui.mark_session import MarkSession

_INTRO = (
    "AutoPTZ Mark is a quick simulation: it adds fake cameras one at a time and "
    "measures how many your computer can run smoothly. Each camera runs the real "
    "AutoPTZ pipeline on moving synthetic people."
)

# Dropdown options as (label, value).  The value is what lands in MarkSession.
_MAX_CAMERA_OPTS: list[tuple[str, int]] = [
    ("4 cameras", 4),
    ("8 cameras", 8),
    ("12 cameras", 12),
    ("16 cameras", 16),
]
_FPS_OPTS: list[tuple[str, float]] = [
    ("24 fps", 24.0),
    ("30 fps", 30.0),
    ("60 fps", 60.0),
]
_STEP_OPTS: list[tuple[str, float]] = [
    ("5 seconds", 5.0),
    ("10 seconds", 10.0),
    ("15 seconds", 15.0),
    ("20 seconds", 20.0),
]
_RES_OPTS: list[tuple[str, str]] = [
    ("720p (HD)", "720p"),
    ("1080p (Full HD)", "1080p"),
    ("4K (Ultra HD)", "4k"),
]
_MODEL_OPTS: list[tuple[str, str]] = [
    ("Auto (recommended)", "auto"),
    ("Nano (fastest)", "nano"),
    ("Small (most accurate)", "small"),
]

# Run-time estimate fudge factor: spin-up / tear-down / discovery overhead on top
# of the measured ramp.
_ETA_OVERHEAD_S = 10.0


def _add_options(combo: QComboBox, opts: list[tuple[str, Any]], current: Any) -> None:
    """Populate *combo* with (label, value) options and select *current*."""
    for label, value in opts:
        combo.addItem(label, value)
    idx = combo.findData(current)
    combo.setCurrentIndex(idx if idx >= 0 else 0)


class MarkPreflightDialog(QDialog):
    """Friendly pre-flight notice + parameter picker for an AutoPTZ Mark run."""

    def __init__(
        self,
        *,
        defaults: MarkSession | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        d = defaults or MarkSession()
        self.setWindowTitle("Run AutoPTZ Mark")
        self.setModal(True)
        self.setMinimumWidth(460)

        col = QVBoxLayout(self)
        col.setContentsMargins(18, 18, 18, 18)
        col.setSpacing(12)

        intro = QLabel(_INTRO)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {T.CURRENT.subtext};")
        col.addWidget(intro)

        # ── profile ────────────────────────────────────────────────────────────
        profile_box = QGroupBox("What to run")
        profile_col = QVBoxLayout(profile_box)
        self._profile_group = QButtonGroup(self)
        self._full_radio = QRadioButton("Full — find, track, and frame people")
        self._streams_radio = QRadioButton("Streams — show video only (no tracking)")
        self._profile_group.addButton(self._full_radio)
        self._profile_group.addButton(self._streams_radio)
        profile_col.addWidget(self._full_radio)
        profile_col.addWidget(self._streams_radio)
        (self._streams_radio if d.profile == "streams" else self._full_radio).setChecked(True)
        col.addWidget(profile_box)

        # ── source ─────────────────────────────────────────────────────────────
        source_box = QGroupBox("Camera source")
        source_col = QVBoxLayout(source_box)
        self._source_group = QButtonGroup(self)
        self._synthetic_radio = QRadioButton("Built-in (no setup needed)")
        ndi_ok = ndi_sim_available()
        ndi_text = "Real NDI cameras" if ndi_ok else "Real NDI cameras  (requires cyndilib)"
        self._ndi_radio = QRadioButton(ndi_text)
        self._ndi_radio.setEnabled(ndi_ok)
        self._source_group.addButton(self._synthetic_radio)
        self._source_group.addButton(self._ndi_radio)
        source_col.addWidget(self._synthetic_radio)
        source_col.addWidget(self._ndi_radio)
        if d.source == "ndi" and ndi_ok:
            self._ndi_radio.setChecked(True)
        else:
            self._synthetic_radio.setChecked(True)
        col.addWidget(source_box)

        # ── parameters (all dropdowns) ──────────────────────────────────────────
        params = QFormLayout()
        params.setHorizontalSpacing(14)
        params.setVerticalSpacing(8)

        self._max_combo = QComboBox()
        _add_options(self._max_combo, _MAX_CAMERA_OPTS, int(d.max_cameras))
        params.addRow("Max cameras", self._max_combo)

        self._fps_combo = QComboBox()
        _add_options(self._fps_combo, _FPS_OPTS, float(d.floor_fps))
        params.addRow("Target FPS", self._fps_combo)

        self._step_combo = QComboBox()
        _add_options(self._step_combo, _STEP_OPTS, float(d.dwell_s))
        params.addRow("Time per step", self._step_combo)
        step_hint = QLabel("How long each level is measured.")
        step_hint.setStyleSheet(f"color: {T.CURRENT.subtext}; font-size: {T.fs(11)}px;")
        params.addRow("", step_hint)

        self._res_combo = QComboBox()
        _add_options(self._res_combo, _RES_OPTS, str(d.resolution).strip().lower())
        params.addRow("Resolution", self._res_combo)

        self._model_combo = QComboBox()
        _add_options(self._model_combo, _MODEL_OPTS, str(d.model).strip().lower())
        params.addRow("Model", self._model_combo)
        col.addLayout(params)

        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet(f"color: {T.CURRENT.subtext};")
        col.addWidget(self._eta_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Start")
        # Start confirms before accepting (entering Mark suspends the live app), so
        # the box's ``accepted`` (emitted on Ok) routes through our confirm slot
        # instead of the default ``accept`` — only a Yes proceeds.
        buttons.accepted.connect(self._on_start)
        buttons.rejected.connect(self.reject)
        col.addWidget(buttons)

        # Live-update the ETA whenever a duration-affecting control changes.
        self._max_combo.currentIndexChanged.connect(self._refresh_eta)
        self._step_combo.currentIndexChanged.connect(self._refresh_eta)
        self._refresh_eta()

    def _on_start(self) -> None:
        """Confirm the run suspends AutoPTZ, then accept."""
        choice = QMessageBox.question(
            self,
            "Run AutoPTZ Mark?",
            "This suspends AutoPTZ and runs the simulation. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if choice == QMessageBox.StandardButton.Yes:
            self.accept()

    def session(self) -> MarkSession:
        """Read the current control values into a :class:`MarkSession`."""
        source = (
            "ndi" if (self._ndi_radio.isEnabled() and self._ndi_radio.isChecked()) else "synthetic"
        )
        profile = "streams" if self._streams_radio.isChecked() else "full"
        return MarkSession(
            profile=profile,
            source=source,
            floor_fps=float(self._fps_combo.currentData()),
            max_cameras=int(self._max_combo.currentData()),
            dwell_s=float(self._step_combo.currentData()),
            resolution=str(self._res_combo.currentData()),
            model=str(self._model_combo.currentData()),
        )

    @staticmethod
    def estimated_seconds(session: MarkSession) -> float:
        """Worst-case run time: every step is measured before the limit is hit."""
        return session.max_cameras * session.dwell_s + _ETA_OVERHEAD_S

    def _refresh_eta(self, *_args: Any) -> None:
        secs = self.estimated_seconds(self.session())
        mins, rem = divmod(int(round(secs)), 60)
        human = f"{mins} min {rem} s" if mins else f"{rem} s"
        self._eta_label.setText(
            f"Estimated time: up to {human} (it stops early once your computer can't keep up)."
        )
