"""RegisterPersonDialog — pick a profile photo (+ name) for an identity.

Two modes:
  * **register** (default): a recognized "Person N" → choose a photo from the
    recognition shots + type a name → ``registerIdentity(id, name, index)``.
  * **change photo**: an already-registered person → just re-pick the profile
    photo → ``setProfileThumbnail(id, index)``.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.common import data_uri_to_pixmap, letter_avatar


class _PhotoStrip(QWidget):
    """A single-select row of candidate photos; exposes the chosen index."""

    def __init__(self, photo_uris: list[str], name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.selected_index = 0
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        if not photo_uris:
            lab = QLabel()
            lab.setPixmap(letter_avatar(name or "?", 72))
            row.addWidget(lab)
            self.selected_index = -1
        for i, uri in enumerate(photo_uris):
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setIconSize(QSize(72, 72))
            btn.setFixedSize(80, 80)
            pm = data_uri_to_pixmap(uri, 72, circular=False) or letter_avatar(name or "?", 72)
            btn.setIcon(QIcon(pm))
            btn.setChecked(i == 0)
            btn.setStyleSheet(
                "QToolButton { border: 2px solid transparent; border-radius: 8px; }"
                f"QToolButton:checked {{ border-color: {T.ACCENT_FALLBACK}; }}"
            )
            btn.clicked.connect(lambda _c, idx=i: self._select(idx))
            self._group.addButton(btn, i)
            row.addWidget(btn)
        row.addStretch(1)

    def _select(self, index: int) -> None:
        self.selected_index = index


class RegisterPersonDialog(QDialog):
    """Register a recognized face or change a registered person's profile photo."""

    def __init__(
        self,
        client: Any,
        identity_id: str,
        photo_uris: list[str],
        *,
        name: str = "",
        register: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._identity_id = identity_id
        self._register = register
        self.setWindowTitle("Register Person" if register else "Change Photo")
        self.setModal(True)

        col = QVBoxLayout(self)
        col.setContentsMargins(18, 18, 18, 18)
        col.setSpacing(12)

        col.addWidget(
            QLabel("Choose a profile photo" + (" and name this person:" if register else ":"))
        )
        self._strip = _PhotoStrip(photo_uris, name)
        col.addWidget(self._strip)

        self._name = QLineEdit(name)
        if register:
            self._name.setPlaceholderText("Name…")
            col.addWidget(self._name)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("Register" if register else "Set Photo")
        ok.setProperty("accent", True)
        # Reasonable, consistent button sizing (not stretched, not cramped).
        ok.setMinimumWidth(96)
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel.setMinimumWidth(96)
        # Make the dismiss affordance unmissable (the bare box button reads as a
        # faint outline against the dialog ground in some themes).  Keep it the
        # default action and Escape-dismissable so the dialog is always closable.
        cancel.setAutoDefault(False)
        cancel.setStyleSheet(
            f"QPushButton {{ border: 1px solid {T.CURRENT.border_hov};"
            f" color: {T.CURRENT.text}; padding: 4px 12px; }}"
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        col.addWidget(buttons)

    def _accept(self) -> None:
        index = self._strip.selected_index
        if self._register:
            name = self._name.text().strip()
            if not name:
                self._name.setFocus()
                return
            self._client.registerIdentity(self._identity_id, name, index)
        else:
            if index >= 0:
                self._client.setProfileThumbnail(self._identity_id, index)
        self.accept()
