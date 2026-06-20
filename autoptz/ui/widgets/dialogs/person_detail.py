"""PersonDetailDialog — review and curate ALL of a person's photos.

Where the inline card strip only shows the first few shots, this dialog shows the
*complete* photo set for one identity and lets the operator manage it end-to-end,
all through the existing EngineClient identity API:

  * **review** every stored photo in a grid,
  * **set any photo as the profile** picture (``setProfileThumbnail``),
  * **delete** an individual photo (``removeIdentityPhoto``),
  * **add a photo from a file** (``addIdentityPhoto``).

It also explains the automatic enrichment: the engine keeps gathering fresh shots
on its own while the person is on camera, so the gallery improves over time
without any manual work.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.common import (
    AccentButton,
    IconButton,
    data_uri_to_pixmap,
    on_theme_changed,
)

log = logging.getLogger(__name__)

_COLS = 3  # photos per row in the grid
_THUMB = 132  # photo cell edge (px)


class PersonDetailDialog(QDialog):
    """Manage one person's whole photo set: review / set-profile / delete / add."""

    def __init__(self, client: Any, identity_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._model = client.identityModel
        self._identity_id = identity_id
        self.setModal(True)
        self.setMinimumWidth(_COLS * (_THUMB + 16) + 48)

        col = QVBoxLayout(self)
        col.setContentsMargins(18, 18, 18, 18)
        col.setSpacing(12)

        self._title = QLabel("")
        self._title.setStyleSheet("font-size: 15px; font-weight: 700;")
        col.addWidget(self._title)

        hint = QLabel(
            "Click a photo to make it the profile picture, or use its remove button. The "
            "engine also gathers fresh shots automatically while the person is on "
            "camera, so this gallery keeps improving on its own."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {T.CURRENT.muted}; font-size: 11px;")
        col.addWidget(hint)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(12)
        scroll.setWidget(self._grid_host)
        col.addWidget(scroll, 1)

        controls = QHBoxLayout()
        add = AccentButton("Add photo...")
        add.setToolTip("Import an image file as a profile/gallery photo for this person.")
        add.clicked.connect(self._add_photo)
        controls.addWidget(add)
        controls.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        controls.addWidget(close)
        col.addLayout(controls)

        # Keep the grid live as photos are added/removed/auto-harvested.
        try:
            client.identitiesChanged.connect(self._rebuild)
        except Exception:  # noqa: BLE001
            log.debug("identitiesChanged connect failed", exc_info=True)
        on_theme_changed(client, self._rebuild)
        self._rebuild()

    # ── data ────────────────────────────────────────────────────────────────────

    def _snapshot(self) -> tuple[str, str, list[str]]:
        """Return (name, profile_uri, all_photo_uris) for this identity."""
        roles = {
            n: _role(self._model, n)
            for n in ("identityId", "identityName", "thumbnail", "thumbnails")
        }
        try:
            n = self._model.rowCount()
        except Exception:  # noqa: BLE001
            return "", "", []
        for i in range(n):
            idx = self._model.index(i, 0)
            if self._model.data(idx, roles["identityId"]) != self._identity_id:
                continue
            name = self._model.data(idx, roles["identityName"]) or ""
            profile = self._model.data(idx, roles["thumbnail"]) or ""
            thumbs = [t for t in (self._model.data(idx, roles["thumbnails"]) or []) if t]
            return name, profile, thumbs
        return "", "", []

    # ── rebuild ──────────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        name, profile, thumbs = self._snapshot()
        self._title.setText(f"Photos — {name or 'this person'}  ·  {len(thumbs)}")
        _clear(self._grid)
        if not thumbs:
            empty = QLabel(
                "No photos yet. Add one, or let the engine capture some "
                "while the person is on camera."
            )
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(f"color: {T.CURRENT.muted}; padding: 24px;")
            self._grid.addWidget(empty, 0, 0, 1, _COLS)
            return
        for i, uri in enumerate(thumbs):
            self._grid.addWidget(
                self._photo_cell(uri, i, is_profile=bool(profile) and uri == profile),
                i // _COLS,
                i % _COLS,
            )

    def _photo_cell(self, uri: str, index: int, *, is_profile: bool) -> QWidget:
        cell = QFrame()
        ring = T.ACCENT_FALLBACK if is_profile else T.CURRENT.border
        width = 2 if is_profile else 1
        cell.setStyleSheet(f"QFrame {{ border: {width}px solid {ring}; border-radius: 8px; }}")
        lay = QVBoxLayout(cell)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        img = QLabel(cell)
        pm = data_uri_to_pixmap(uri, _THUMB, circular=False)
        if pm is not None:
            img.setPixmap(pm)
        img.setFixedSize(_THUMB, _THUMB)
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(img, 0, Qt.AlignmentFlag.AlignCenter)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        if is_profile:
            badge = QLabel("● Profile")
            badge.setStyleSheet(f"color: {T.ACCENT_FALLBACK}; font-size: 11px; font-weight: 700;")
            row.addWidget(badge)
        else:
            setp = QPushButton("Set profile")
            setp.setCursor(Qt.CursorShape.PointingHandCursor)
            setp.setToolTip("Make this the profile picture")
            setp.clicked.connect(
                lambda *_a, idx=index: self._client.setProfileThumbnail(self._identity_id, idx)
            )
            row.addWidget(setp)
        row.addStretch(1)
        # The single, consistent delete affordance (small icon, red on hover).
        rm = IconButton("x", tip="Remove this photo", danger=True, size=24)
        rm.clicked.connect(
            lambda *_a, idx=index: self._client.removeIdentityPhoto(self._identity_id, idx)
        )
        row.addWidget(rm)
        lay.addLayout(row)
        return cell

    # ── add ──────────────────────────────────────────────────────────────────────

    def _add_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add Photo", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        img = QImage(path)
        if img.isNull():
            return
        # Downscale large imports so the stored thumbnail stays modest.
        if max(img.width(), img.height()) > 480:
            img = img.scaled(
                480,
                480,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        uri = _image_to_png_data_uri(img)
        if uri:
            self._client.addIdentityPhoto(self._identity_id, uri)


# ── helpers ─────────────────────────────────────────────────────────────────────


def _image_to_png_data_uri(img: QImage) -> str:
    """Encode a QImage as a ``data:image/png;base64,…`` URI (``""`` on failure)."""
    try:
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        img.save(buf, "PNG")
        buf.close()
        return "data:image/png;base64," + bytes(ba.toBase64()).decode("ascii")
    except Exception:  # noqa: BLE001
        log.debug("image encode failed", exc_info=True)
        return ""


def _role(model: Any, name: str) -> int:
    try:
        for key, val in model.roleNames().items():
            if bytes(val).decode() == name:
                return int(key)
    except Exception:  # noqa: BLE001
        pass
    return -1


def _clear(layout: Any) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
        else:
            child = item.layout()
            if child is not None:
                _clear(child)
