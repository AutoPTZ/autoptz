"""AboutMarkDialog — identity, version, guide, FPS targets, and Do/Don'ts for AutoPTZ Mark."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.branding import logo_pixmap

MARK_VERSION = "1.0.0"


class AboutMarkDialog(QDialog):
    """About AutoPTZ Mark — version lives here; the window title stays versionless."""

    def __init__(self, client: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self.setWindowTitle("About AutoPTZ Mark")
        self.setModal(True)
        self.setMinimumWidth(420)

        col = QVBoxLayout(self)
        col.setContentsMargins(24, 24, 24, 24)
        col.setSpacing(8)

        mark = QLabel()
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(72, 72)
        pix = logo_pixmap(72)
        if not pix.isNull():
            mark.setPixmap(pix)
        else:
            mark.setText("AP")
            mark.setStyleSheet(
                f"background: {T.ACCENT_FALLBACK}; color: white; border-radius: 16px;"
                f" font-size: 32px; font-weight: 700;"
            )
        col.addWidget(mark, 0, Qt.AlignmentFlag.AlignHCenter)

        name = QLabel("AutoPTZ Mark")
        name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        name.setStyleSheet("font-size: 22px; font-weight: 700;")
        col.addWidget(name)
        blurb = QLabel(
            "A self-contained benchmark: AutoPTZ Mark ramps simulated cameras and "
            "runs the full detection + tracking pipeline so you can see how many "
            "streams this machine sustains."
        )
        blurb.setWordWrap(True)
        blurb.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        blurb.setStyleSheet(f"color: {T.CURRENT.subtext};")
        col.addWidget(blurb)
        col.addSpacing(8)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.addRow("Version", QLabel(MARK_VERSION))
        form.addRow(
            "Guide",
            _wrap(
                "Pick a source (synthetic or NDI) and a camera count, then Start. "
                "Mark adds fake cameras one at a time and reports the sustained "
                "frame-rate at each step."
            ),
        )
        form.addRow(
            "FPS targets",
            _wrap("≥ 30 fps: Excellent  ·  24–30 fps: Good  ·  < 24 fps: Check load"),
        )
        form.addRow(
            "Do",
            _wrap("Close heavy apps  ·  Plug in power for a steady result."),
        )
        form.addRow(
            "Don't",
            _wrap("Don't touch the machine while the ramp runs (it skews the score)."),
        )
        form.addRow(
            "Detection",
            _wrap(
                "Boxes need a model — set AUTOPTZ_MODEL_PATH or use the bundled yolo11n weights."
            ),
        )
        col.addLayout(form)

        col.addSpacing(8)
        close = QPushButton("Close")
        close.setProperty("accent", True)
        close.clicked.connect(self.accept)
        col.addWidget(close, 0, Qt.AlignmentFlag.AlignHCenter)


def _wrap(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color: {T.CURRENT.subtext};")
    return lbl
