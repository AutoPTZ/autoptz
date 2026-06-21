"""UpdateDialog — offer the OS-specific release update."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.branding import logo_pixmap
from autoptz.update.checker import UpdateInfo


class UpdateDialog(QDialog):
    """Shows the new version + release notes and starts the updater."""

    def __init__(
        self,
        info: UpdateInfo,
        current_version: str,
        on_skip: Callable[[str], None] | None = None,
        on_install: Callable[[UpdateInfo], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._info = info
        self._on_skip = on_skip
        self._on_install = on_install
        self.setWindowTitle("Update Available")
        self.setModal(True)
        self.setMinimumWidth(460)

        col = QVBoxLayout(self)
        col.setContentsMargins(24, 24, 24, 24)
        col.setSpacing(10)

        header = QHBoxLayout()
        mark = QLabel()
        pm = logo_pixmap(48)
        if not pm.isNull():
            mark.setPixmap(pm)
        header.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)

        heading = QVBoxLayout()
        title = QLabel(f"AutoPTZ {info.version} is available")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        heading.addWidget(title)
        sub = QLabel(
            f"You have {current_version}." + (" (pre-release)" if info.is_prerelease else "")
        )
        sub.setStyleSheet(f"color: {T.CURRENT.subtext};")
        heading.addWidget(sub)
        header.addLayout(heading, 1)
        col.addLayout(header)

        if info.body.strip():
            notes = QTextBrowser()
            notes.setOpenExternalLinks(True)
            try:
                notes.setMarkdown(info.body)
            except Exception:  # noqa: BLE001 — fall back to plain text
                notes.setPlainText(info.body)
            notes.setMinimumHeight(180)
            col.addWidget(notes)

        buttons = QHBoxLayout()
        skip = QPushButton("Skip This Version")
        skip.clicked.connect(self._skip)
        buttons.addWidget(skip)
        buttons.addStretch(1)
        later = QPushButton("Later")
        later.clicked.connect(self.reject)
        buttons.addWidget(later)
        download = QPushButton("Download and Restart" if info.asset_for_platform() else "Download")
        download.setProperty("accent", True)
        if info.asset_for_platform():
            download.setToolTip("Download the installer for this OS, launch it, then quit AutoPTZ.")
        else:
            download.setToolTip("Open the release page because this OS has no matching asset.")
        download.clicked.connect(self._download)
        buttons.addWidget(download)
        col.addLayout(buttons)

    def _download(self) -> None:
        if self._info.asset_for_platform() and self._on_install is not None:
            self._on_install(self._info)
            self.accept()
            return
        QDesktopServices.openUrl(QUrl(self._info.html_url))
        self.accept()

    def _skip(self) -> None:
        if self._on_skip is not None:
            self._on_skip(self._info.version)
        self.reject()
