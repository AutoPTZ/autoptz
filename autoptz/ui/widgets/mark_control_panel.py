"""MarkControlPanel — source choice, camera count, live verdict, Start/Stop.

A compact control surface for the AutoPTZ Mark window: a Synthetic/NDI source
radio pair (NDI disabled when ``cyndilib`` is absent so a host that can't build
NDI senders never offers it), a 1–16 camera spinbox, a live "sustaining N cams @
X fps" verdict label, and Start/Stop buttons.  Pure widget logic — all wiring is
via signals so the window just connects them.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from autoptz.benchmark.ndi_sim import ndi_sim_available
from autoptz.ui import theme as T


class MarkControlPanel(QWidget):
    """Source/camera-count/verdict + Start/Stop for an AutoPTZ Mark run."""

    sourceChanged = Signal(str)
    maxCamerasChanged = Signal(int)
    startClicked = Signal()
    stopClicked = Signal()
    exitClicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        col = QVBoxLayout(self)
        form = QFormLayout()

        src_row = QHBoxLayout()
        self._syn_radio = QRadioButton("Synthetic")
        self._syn_radio.setChecked(True)
        self._ndi_radio = QRadioButton("NDI")
        if not ndi_sim_available():
            self._ndi_radio.setEnabled(False)
            self._ndi_radio.setToolTip("NDI simulation needs cyndilib (not installed).")
        self._src_group = QButtonGroup(self)
        self._src_group.addButton(self._syn_radio)
        self._src_group.addButton(self._ndi_radio)
        self._syn_radio.toggled.connect(self._on_source_changed)
        self._ndi_radio.toggled.connect(self._on_source_changed)
        src_row.addWidget(self._syn_radio)
        src_row.addWidget(self._ndi_radio)
        src_row.addStretch(1)
        src_holder = QWidget(self)
        src_holder.setLayout(src_row)
        form.addRow("Source", src_holder)

        self._spin = QSpinBox()
        self._spin.setRange(1, 16)
        self._spin.setValue(8)
        self._spin.setToolTip("How many simulated cameras to ramp up to.")
        self._spin.valueChanged.connect(self.maxCamerasChanged.emit)
        form.addRow("Cameras", self._spin)

        self._verdict_label = QLabel("Idle.")
        self._verdict_label.setWordWrap(True)
        self._verdict_label.setStyleSheet(f"color: {T.CURRENT.subtext};")
        form.addRow("Verdict", self._verdict_label)
        col.addLayout(form)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.setProperty("accent", True)
        self._start_btn.clicked.connect(self.startClicked.emit)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stopClicked.emit)
        # Visible exit affordance: the user can deliberately leave Mark (Return /
        # Quit, optional save) instead of only via the OS close button.
        self._exit_btn = QPushButton("Exit Mark…")
        self._exit_btn.setToolTip("Return to AutoPTZ or quit (optionally saving results).")
        self._exit_btn.clicked.connect(self.exitClicked.emit)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._exit_btn)
        col.addLayout(btn_row)

    def _on_source_changed(self) -> None:
        self.sourceChanged.emit(self.selected_source())

    def selected_source(self) -> str:
        return "ndi" if self._ndi_radio.isChecked() else "synthetic"

    def selected_max_cameras(self) -> int:
        return int(self._spin.value())

    def set_max_cameras(self, n: int) -> None:
        """Seed the camera-count spin AND cap its maximum to *n*.

        The Mark window pre-adds exactly ``session.max_cameras`` cameras to the
        idle wall and the ramp ADOPTS that fixed set, so the ramp can never exceed
        it — capping the spin's maximum keeps the panel's count in lockstep with
        the pre-added wall (one source of truth) and stops the user from selecting
        a count with no backing camera."""
        n = max(1, int(n))
        self._spin.setRange(1, n)
        self._spin.setValue(n)

    def set_verdict(self, text: str) -> None:
        self._verdict_label.setText(text)

    def set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._spin.setEnabled(not running)
        self._syn_radio.setEnabled(not running)
        self._ndi_radio.setEnabled(not running and ndi_sim_available())
