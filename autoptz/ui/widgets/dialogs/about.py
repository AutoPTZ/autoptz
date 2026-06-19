"""AboutDialog — app identity, version, author, inference EP, and profile links."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T

GITHUB_URL = "https://github.com/AutoPTZ/autoptz"
LINKEDIN_URL = "https://www.linkedin.com/in/stevenson-chittumuri/"
VERSION = "2.0.0a0"


class AboutDialog(QDialog):
    """About AutoPTZ."""

    def __init__(self, client: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self.setWindowTitle("About AutoPTZ")
        self.setModal(True)
        self.setMinimumWidth(360)

        col = QVBoxLayout(self)
        col.setContentsMargins(24, 24, 24, 24)
        col.setSpacing(8)

        mark = QLabel("P")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(64, 64)
        mark.setStyleSheet(
            f"background: {T.ACCENT_FALLBACK}; color: white; border-radius: 16px;"
            f" font-size: 32px; font-weight: 700;"
        )
        col.addWidget(mark, 0, Qt.AlignmentFlag.AlignHCenter)

        name = QLabel("AutoPTZ"); name.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        name.setStyleSheet("font-size: 22px; font-weight: 700;")
        col.addWidget(name)
        sub = QLabel("AI-driven PTZ camera tracking")
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        sub.setStyleSheet(f"color: {T.CURRENT.subtext};")
        col.addWidget(sub)
        col.addSpacing(8)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.addRow("Version", QLabel(VERSION))
        form.addRow("Author", QLabel("Stevenson Chittumuri"))
        ep = "Engine stopped"
        try:
            if client and client.engineRunning:
                ep = (client.engineEp or "—").replace("ExecutionProvider", "")
        except Exception:  # noqa: BLE001
            pass
        form.addRow("Inference EP", QLabel(ep))
        link = QLabel(f'<a href="{GITHUB_URL}" style="color:{T.ACCENT_FALLBACK}">{GITHUB_URL}</a>')
        link.setOpenExternalLinks(False)
        link.linkActivated.connect(lambda u: QDesktopServices.openUrl(QUrl(u)))
        form.addRow("GitHub", link)
        linkedin = QLabel(
            f'<a href="{LINKEDIN_URL}" style="color:{T.ACCENT_FALLBACK}">{LINKEDIN_URL}</a>'
        )
        linkedin.setOpenExternalLinks(False)
        linkedin.linkActivated.connect(lambda u: QDesktopServices.openUrl(QUrl(u)))
        form.addRow("LinkedIn", linkedin)
        col.addLayout(form)

        col.addSpacing(8)
        close = QPushButton("Close"); close.setProperty("accent", True)
        close.clicked.connect(self.accept)
        col.addWidget(close, 0, Qt.AlignmentFlag.AlignHCenter)
