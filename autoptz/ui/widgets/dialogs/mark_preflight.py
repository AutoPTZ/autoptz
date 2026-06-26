"""MarkPreflightDialog — the 3DMark-style pre-flight notice for AutoPTZ Mark.

Explains how the run works, shows an estimated run time, do/don't guidance, and
the controls (profile, source, floor-fps, max-cameras, dwell).  On accept it
yields a :class:`MarkSession`.  Pure widget logic, offscreen-testable: construct,
poke control values, read :meth:`session`.

The NDI source option is disabled (with a "requires cyndilib" suffix) when
``ndi_sim_available()`` is False, so the relaunched window never tries to spin up
NDI senders the host can't build.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from autoptz.benchmark.ndi_sim import ndi_sim_available
from autoptz.ui import theme as T
from autoptz.ui.mark_session import MarkSession

_INTRO = (
    "AutoPTZ Mark ramps simulated cameras 1 → N at 30 fps and stops when a "
    "camera count can't hold the fps floor. Each tile runs the real pipeline "
    "(detection → tracking → Center Stage) on moving synthetic people.\n\n"
    "Before you start: plug in power, close other heavy apps, and don't touch the "
    "machine during the run — background work skews the score. The app will "
    "relaunch into a dedicated window; “Return to AutoPTZ” brings the normal "
    "app back."
)

# Run-time estimate fudge factor: spin-up / tear-down / discovery overhead on top
# of the dwell-bounded ramp.
_ETA_OVERHEAD_S = 10.0


class MarkPreflightDialog(QDialog):
    """Pre-flight notice + parameter picker for an AutoPTZ Mark run."""

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
        profile_box = QGroupBox("Profile")
        profile_col = QVBoxLayout(profile_box)
        self._profile_group = QButtonGroup(self)
        self._full_radio = QRadioButton("Full — detection + tracking + Center Stage")
        self._streams_radio = QRadioButton("Streams — preview + manual PTZ only")
        self._profile_group.addButton(self._full_radio)
        self._profile_group.addButton(self._streams_radio)
        profile_col.addWidget(self._full_radio)
        profile_col.addWidget(self._streams_radio)
        (self._streams_radio if d.profile == "streams" else self._full_radio).setChecked(True)
        col.addWidget(profile_box)

        # ── source ─────────────────────────────────────────────────────────────
        source_box = QGroupBox("Source")
        source_col = QVBoxLayout(source_box)
        self._source_group = QButtonGroup(self)
        self._synthetic_radio = QRadioButton("Synthetic (in-process, no setup)")
        ndi_ok = ndi_sim_available()
        ndi_text = "Real NDI sources" if ndi_ok else "Real NDI sources  (requires cyndilib)"
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

        # ── parameters ───────────────────────────────────────────────────────────
        params = QFormLayout()
        params.setHorizontalSpacing(14)
        params.setVerticalSpacing(8)

        self._max_spin = QSpinBox()
        self._max_spin.setRange(1, 32)
        self._max_spin.setValue(int(d.max_cameras))
        params.addRow("Max cameras", self._max_spin)

        self._floor_spin = QDoubleSpinBox()
        self._floor_spin.setRange(5.0, 60.0)
        self._floor_spin.setSingleStep(1.0)
        self._floor_spin.setDecimals(1)
        self._floor_spin.setValue(float(d.floor_fps))
        params.addRow("FPS floor", self._floor_spin)

        self._dwell_spin = QDoubleSpinBox()
        self._dwell_spin.setRange(5.0, 60.0)
        self._dwell_spin.setSingleStep(1.0)
        self._dwell_spin.setDecimals(1)
        self._dwell_spin.setValue(float(d.dwell_s))
        params.addRow("Dwell per step (s)", self._dwell_spin)
        col.addLayout(params)

        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet(f"color: {T.CURRENT.subtext};")
        col.addWidget(self._eta_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        col.addWidget(buttons)

        # Live-update the ETA whenever a duration-affecting control changes.
        self._max_spin.valueChanged.connect(self._refresh_eta)
        self._dwell_spin.valueChanged.connect(self._refresh_eta)
        self._refresh_eta()

    def session(self) -> MarkSession:
        """Read the current control values into a :class:`MarkSession`."""
        source = (
            "ndi" if (self._ndi_radio.isEnabled() and self._ndi_radio.isChecked()) else "synthetic"
        )
        profile = "streams" if self._streams_radio.isChecked() else "full"
        return MarkSession(
            profile=profile,
            source=source,
            floor_fps=float(self._floor_spin.value()),
            max_cameras=int(self._max_spin.value()),
            dwell_s=float(self._dwell_spin.value()),
        )

    @staticmethod
    def estimated_seconds(session: MarkSession) -> float:
        """Worst-case run time: every step dwells before the floor is hit."""
        return session.max_cameras * session.dwell_s + _ETA_OVERHEAD_S

    def _refresh_eta(self, *_args: Any) -> None:
        secs = self.estimated_seconds(self.session())
        mins, rem = divmod(int(round(secs)), 60)
        human = f"{mins} min {rem} s" if mins else f"{rem} s"
        self._eta_label.setText(
            f"Estimated run time: up to {human} (the ramp stops early once a camera "
            "count can't hold the floor)."
        )
