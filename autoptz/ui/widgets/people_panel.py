"""PeoplePanel — registered identities + recognized (unnamed) faces.

Implements the recognized → registered model: an auto-harvested face ("Person N")
is *recognized* but not **registered** until you name it.  Registered people are
named and persisted; tracking is assigned **per-camera** elsewhere (the camera's
target picker), not from this panel.

Wired to the existing identity CRUD on EngineClient: ``registerIdentity``
(register), ``renameIdentity``, ``deleteIdentity``, ``mergeIdentities``,
``setProfileThumbnail`` (change profile photo), and ``removeIdentityPhoto``
(prune a single captured shot).
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.common import (
    AccentButton,
    DangerButton,
    IconButton,
    data_uri_to_pixmap,
    letter_avatar,
    on_theme_changed,
    section_label,
)
from autoptz.ui.widgets.dialogs.person_detail import PersonDetailDialog
from autoptz.ui.widgets.dialogs.register_person import RegisterPersonDialog

# How many captured shots to render inline on a card before collapsing the rest
# into a "+N" overflow chip.
_MAX_STRIP = 6

log = logging.getLogger(__name__)


class PeoplePanel(QWidget):
    """Gallery of registered people + a tray of recognized faces to name."""

    def __init__(self, client: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._model = client.identityModel
        # Merge selection spans BOTH registered and recognized cards, so faces
        # can be combined with each other or folded into a named person.
        self._merge_selected: list[str] = []
        self._recognized_ids: list[str] = []
        # Merge only makes sense with someone to merge *into*: the per-card toggle
        # is hidden unless at least two identities exist (set in ``rebuild``).
        self._can_merge: bool = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # header
        head = QHBoxLayout()
        head.setContentsMargins(12, 10, 12, 6)
        self._title = QLabel("People")
        self._title.setStyleSheet("font-size: 15px; font-weight: 700;")
        head.addWidget(self._title)
        head.addStretch(1)
        self._merge_btn = AccentButton("Merge")
        self._merge_btn.setVisible(False)
        self._merge_btn.clicked.connect(self._merge)
        head.addWidget(self._merge_btn)
        root.addLayout(head)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        root.addWidget(scroll, 1)

        body = QWidget()
        self._col = QVBoxLayout(body)
        self._col.setContentsMargins(12, 4, 12, 12)
        self._col.setSpacing(6)
        scroll.setWidget(body)

        self._col.addWidget(section_label("Registered people"))
        self._registered_hint = _hint(
            "Named people. Tracking is assigned per camera. The engine keeps "
            "gathering fresh photos automatically while someone is on camera — "
            "open Photos to review, add, or pick the profile picture."
        )
        self._col.addWidget(self._registered_hint)
        self._registered_box = QVBoxLayout()
        self._registered_box.setSpacing(6)
        self._col.addLayout(self._registered_box)

        self._col.addSpacing(8)
        rec_head = QHBoxLayout()
        rec_head.setContentsMargins(0, 0, 0, 0)
        rec_head.addWidget(section_label("Recognized — not yet registered"))
        rec_head.addStretch(1)
        self._clear_all_btn = QPushButton("Clear all")
        self._clear_all_btn.setVisible(False)
        self._clear_all_btn.setToolTip("Discard every unregistered face")
        self._clear_all_btn.clicked.connect(self._clear_all_recognized)
        rec_head.addWidget(self._clear_all_btn)
        self._col.addLayout(rec_head)
        self._recognized_hint = _hint(
            "Faces the engine spotted. Type a name and Register to keep them."
        )
        self._col.addWidget(self._recognized_hint)
        self._recognized_box = QVBoxLayout()
        self._recognized_box.setSpacing(6)
        self._col.addLayout(self._recognized_box)
        self._col.addStretch(1)

        _connect(client, "identitiesChanged", self.rebuild)
        for sig in ("rowsInserted", "rowsRemoved", "modelReset", "layoutChanged"):
            _connect(self._model, sig, lambda *_: self.rebuild())
        # Cards bake literal T.CURRENT.* colors at build time, so a Light/Dark
        # flip leaves them stale — rebuild on every theme change.
        on_theme_changed(client, self.rebuild)

        self.rebuild()

    # ── selection routing (kept for MainWindow compatibility) ───────────────────

    def set_target_camera(self, camera_id: str, name: str = "") -> None:  # noqa: ARG002
        """No-op kept for the MainWindow caller.

        Tracking is now assigned per-camera elsewhere (the camera's target
        picker), so the People panel no longer reacts to the selected camera —
        but MainWindow still calls this on selection, so it stays as a harmless
        stub to avoid breaking that call site.
        """

    # ── rebuild ──────────────────────────────────────────────────────────────────

    def rebuild(self) -> None:
        registered, recognized = self._snapshot()
        self._recognized_ids = [r["id"] for r in recognized]
        # Prune merge selection to ids that still exist (registered OR recognized).
        alive = {r["id"] for r in registered} | set(self._recognized_ids)
        self._merge_selected = [i for i in self._merge_selected if i in alive]

        # Merging requires at least two identities to combine.
        self._can_merge = (len(registered) + len(recognized)) >= 2

        _clear(self._registered_box)
        _clear(self._recognized_box)

        if registered:
            for rec in registered:
                self._registered_box.addWidget(self._registered_card(rec))
        else:
            self._registered_box.addWidget(_empty("No one registered yet"))

        if recognized:
            for rec in recognized:
                self._recognized_box.addWidget(self._recognized_card(rec))
        else:
            self._recognized_box.addWidget(_empty("No unnamed faces — start the engine and "
                                                  "point a camera at someone"))

        n = len(registered)
        u = len(recognized)
        extra = f" · {u} awaiting a name" if u else ""
        self._title.setText(f"People  ·  {n} registered{extra}")
        self._clear_all_btn.setVisible(u > 0)
        self._refresh_merge_bar()

    def _snapshot(self) -> tuple[list[dict], list[dict]]:
        reg: list[dict] = []
        rec: list[dict] = []
        try:
            n = self._model.rowCount()
        except Exception:  # noqa: BLE001
            return reg, rec
        roles = {name: _role(self._model, name) for name in
                 ("identityId", "identityName", "thumbnail", "enabled", "labeled",
                  "thumbnails")}
        for i in range(n):
            idx = self._model.index(i, 0)
            row = {
                "id": self._model.data(idx, roles["identityId"]),
                "name": self._model.data(idx, roles["identityName"]) or "",
                "thumb": self._model.data(idx, roles["thumbnail"]) or "",
                "thumbs": list(self._model.data(idx, roles["thumbnails"]) or []),
                "enabled": bool(self._model.data(idx, roles["enabled"])),
                "labeled": bool(self._model.data(idx, roles["labeled"])),
            }
            (reg if row["labeled"] else rec).append(row)
        return reg, rec

    # ── cards ──────────────────────────────────────────────────────────────────

    def _registered_card(self, rec: dict) -> QWidget:
        card, row = _card(selected=rec["id"] in self._merge_selected)
        row.addWidget(_avatar(rec["thumb"], rec["name"]))

        mid = QVBoxLayout(); mid.setSpacing(6)
        name = QLineEdit(rec["name"])
        name.setStyleSheet("font-weight: 700;")
        name.editingFinished.connect(
            lambda i=rec["id"], w=name: self._rename(i, w.text())
        )
        mid.addWidget(name)

        # Captured shots as a deletable strip: each shot carries its own small ✕
        # so the user can prune a bad/odd-angle photo individually.
        strip = self._photo_strip(rec["id"], rec["thumbs"], rec["thumb"])
        if strip is not None:
            mid.addWidget(strip)

        # Readable, labelled action buttons (icon + text) — Photos / Merge / Delete.
        controls = QHBoxLayout(); controls.setSpacing(6)
        controls.addWidget(self._photos_button(rec["id"]))
        if self._can_merge:
            controls.addWidget(self._merge_button(rec["id"]))
        controls.addWidget(self._delete_button(
            "🗑  Delete",
            lambda *_a, i=rec["id"], nm=rec["name"]: self._delete(i, nm),
            tip="Delete this person",
        ))
        controls.addStretch(1)
        mid.addLayout(controls)
        row.addLayout(mid, 1)
        return card

    def _recognized_card(self, rec: dict) -> QWidget:
        card, row = _card(selected=rec["id"] in self._merge_selected)
        row.addWidget(_avatar(rec["thumb"], rec["name"]))
        mid = QVBoxLayout(); mid.setSpacing(6)
        label = QLabel(rec["name"] or "Unnamed face")
        label.setStyleSheet("font-weight: 700;")
        mid.addWidget(label)

        strip = self._photo_strip(rec["id"], rec["thumbs"], rec["thumb"])
        if strip is not None:
            mid.addWidget(strip)

        n = len(rec["thumbs"]) or (1 if rec["thumb"] else 0)
        hint = QLabel(f"{n} photo{'s' if n != 1 else ''} captured")
        hint.setStyleSheet(f"color: {T.CURRENT.muted}; font-size: 11px;")
        mid.addWidget(hint)

        controls = QHBoxLayout(); controls.setSpacing(6)
        register = AccentButton("Register…")
        # A normal-sized button — let it size to its text rather than stretch the
        # whole row width (the previous ``addWidget(register, 1)`` made it huge).
        register.setMaximumWidth(120)
        # ``clicked`` emits a leading ``checked`` bool — absorb it with ``*_a`` so
        # the bound identity id survives (otherwise ``i`` became ``False`` and the
        # dialog registered against a nonexistent record → silent no-op).
        register.clicked.connect(
            lambda *_a, i=rec["id"], nm=rec["name"], ph=rec["thumbs"]:
            self._open_register(i, nm, ph)
        )
        controls.addWidget(register)
        # Let recognized faces join a merge too (combine duplicates, or fold into
        # an existing named person) — only when there's something to merge with.
        if self._can_merge:
            controls.addWidget(self._merge_button(rec["id"]))
        controls.addStretch(1)
        controls.addWidget(self._discard_button(
            lambda *_a, i=rec["id"]: self._client.deleteIdentity(i),
            tip="Remove this unregistered face",
        ))
        mid.addLayout(controls)
        row.addLayout(mid, 1)
        return card

    # ── shared card affordances ────────────────────────────────────────────────

    def _photo_strip(
        self, identity_id: str, thumbs: list[str], primary: str = "",
    ) -> QWidget | None:
        """A strip of captured shots, each with its own small ``✕`` delete badge.

        Clicking a shot's ``✕`` prunes THAT photo via
        ``removeIdentityPhoto(id, index)`` (confirmed only when it is the last
        remaining shot, which effectively clears the person's photos).  Returns
        ``None`` when the identity has no captured shots.
        """
        shots = [t for t in thumbs if t]
        if not shots:
            return None

        strip = QWidget()
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        total = len(shots)
        for index, uri in enumerate(shots[:_MAX_STRIP]):
            lay.addWidget(self._photo_chip(identity_id, uri, index, total, primary))
        overflow = total - _MAX_STRIP
        if overflow > 0:
            more = QLabel(f"+{overflow}")
            more.setAlignment(Qt.AlignmentFlag.AlignCenter)
            more.setFixedSize(38, 38)
            more.setStyleSheet(
                f"color: {T.CURRENT.subtext}; border: 1px solid {T.CURRENT.border};"
                f" border-radius: 6px; font-size: 11px; font-weight: 700;"
            )
            lay.addWidget(more)
        lay.addStretch(1)
        return strip

    def _photo_chip(
        self, identity_id: str, uri: str, index: int, total: int, primary: str,
    ) -> QWidget:
        """One captured shot with a corner ``✕`` that deletes just this photo."""
        chip = QFrame()
        chip.setFixedSize(38, 38)
        # The profile photo gets an accent ring so it's distinguishable.
        ring = T.ACCENT_FALLBACK if (primary and uri == primary) else T.CURRENT.border
        chip.setStyleSheet(
            f"QFrame {{ border: 1px solid {ring}; border-radius: 6px; }}"
        )
        wrap = QVBoxLayout(chip)
        wrap.setContentsMargins(0, 0, 0, 0)

        img = QLabel(chip)
        pm = data_uri_to_pixmap(uri, 36, circular=False)
        if pm is not None:
            img.setPixmap(pm)
        img.setFixedSize(38, 38)
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wrap.addWidget(img)

        # Floating delete badge in the top-right corner of the thumbnail — a small
        # scrim-backed ✕ that turns red on hover (consistent with IconButton).
        x = QPushButton("✕", chip)
        x.setFixedSize(15, 15)
        x.setCursor(Qt.CursorShape.PointingHandCursor)
        x.setToolTip("Remove this photo")
        x.setStyleSheet(
            f"QPushButton {{ background: {T.VIDEO_SCRIM.name()}; color: {T.VIDEO_TEXT};"
            f" border: none; border-radius: 7px; font-size: 9px; font-weight: 700;"
            f" padding: 0px; }}"
            f"QPushButton:hover {{ background: {T.DANGER}; color: {T.ACCENT_TEXT}; }}"
        )
        x.move(38 - 15, 0)
        x.clicked.connect(
            lambda *_a, i=identity_id, idx=index, last=(total <= 1):
            self._remove_photo(i, idx, last)
        )
        return chip

    def _merge_button(self, identity_id: str) -> QPushButton:
        """A checkable Merge-select toggle — uses the global checked-button style."""
        selected = identity_id in self._merge_selected
        btn = QPushButton("⊕  Merge" if not selected else "✓  Merging")
        btn.setCheckable(True)
        btn.setChecked(selected)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Select two or more people, then Merge to combine them")
        btn.toggled.connect(lambda on, i=identity_id: self._toggle_merge(i, on))
        return btn

    def _delete_button(self, text: str, slot: Any, *, tip: str) -> QPushButton:
        """A readable, labelled destructive button (the shared DangerButton)."""
        btn = DangerButton(text)
        btn.setToolTip(tip)
        btn.clicked.connect(slot)
        return btn

    def _photos_button(self, identity_id: str) -> QPushButton:
        """Open the full photo manager (review all / add / set profile / delete)."""
        btn = QPushButton("🖼  Photos")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Review every photo, add one, pick the profile picture, or "
                       "remove a bad shot.")
        btn.clicked.connect(
            lambda *_a, i=identity_id:
            PersonDetailDialog(self._client, i, parent=self).exec()
        )
        return btn

    def _discard_button(self, slot: Any, *, tip: str) -> IconButton:
        """The single icon delete affordance for unregistered faces."""
        btn = IconButton("🗑", tip=tip, danger=True, size=28)
        btn.clicked.connect(slot)
        return btn

    # ── actions ────────────────────────────────────────────────────────────────

    def _open_register(self, identity_id: str, name: str, photo_uris: list[str]) -> None:
        RegisterPersonDialog(
            self._client, identity_id, photo_uris, name=name, register=True, parent=self,
        ).exec()

    def _remove_photo(self, identity_id: str, index: int, last: bool) -> None:
        """Prune a single captured photo (confirm only when it's the last one)."""
        if last and QMessageBox.question(
            self, "Remove last photo",
            "This is the only captured photo. Remove it anyway?",
        ) != QMessageBox.StandardButton.Yes:
            return
        self._client.removeIdentityPhoto(identity_id, index)

    def _rename(self, identity_id: str, name: str) -> None:
        if name.strip():
            self._client.renameIdentity(identity_id, name.strip())

    def _delete(self, identity_id: str, name: str) -> None:
        if QMessageBox.question(self, "Delete person", f"Delete “{name or 'this person'}”?") \
                == QMessageBox.StandardButton.Yes:
            self._client.deleteIdentity(identity_id)

    def _clear_all_recognized(self) -> None:
        ids = list(self._recognized_ids)
        if not ids:
            return
        n = len(ids)
        if QMessageBox.question(
            self, "Clear recognized faces",
            f"Discard {n} unregistered face{'s' if n != 1 else ''}?",
        ) == QMessageBox.StandardButton.Yes:
            for identity_id in ids:
                self._client.deleteIdentity(identity_id)

    def _toggle_merge(self, identity_id: str, on: bool) -> None:
        if on and identity_id not in self._merge_selected:
            self._merge_selected.append(identity_id)
        elif not on and identity_id in self._merge_selected:
            self._merge_selected.remove(identity_id)
        self._refresh_merge_bar()

    def _refresh_merge_bar(self) -> None:
        count = len(self._merge_selected)
        self._merge_btn.setVisible(count >= 2)
        self._merge_btn.setText(f"Merge {count} → 1")

    def _merge(self) -> None:
        if len(self._merge_selected) < 2:
            return
        # Prefer keeping a registered (named) identity so a merge into a named
        # person folds the rest in rather than the reverse.
        reg = {r["id"] for r in self._snapshot()[0]}
        keep = next((i for i in self._merge_selected if i in reg),
                    self._merge_selected[0])
        for drop in self._merge_selected:
            if drop != keep:
                self._client.mergeIdentities(keep, drop)
        self._merge_selected = []


# ── helpers ─────────────────────────────────────────────────────────────────────


def _card(selected: bool = False) -> tuple[QFrame, QHBoxLayout]:
    card = QFrame()
    card.setObjectName("personCard")
    # Selected-for-merge cards get an accent border so the pick is obvious; the
    # ``#personCard`` selector keeps the styling off child buttons/labels.
    border = T.ACCENT_FALLBACK if selected else T.CURRENT.border
    width = 2 if selected else 1
    card.setStyleSheet(
        f"#personCard {{ background: {T.CURRENT.surface};"
        f" border: {width}px solid {border}; border-radius: {10}px; }}"
    )
    row = QHBoxLayout(card)
    row.setContentsMargins(10, 10, 10, 10)
    row.setSpacing(10)
    return card, row


def _avatar(thumb: str, name: str) -> QLabel:
    lab = QLabel()
    pm = data_uri_to_pixmap(thumb, 52) if thumb else None
    lab.setPixmap(pm or letter_avatar(name or "?", 52))
    lab.setFixedSize(52, 52)
    return lab


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(f"color: {T.CURRENT.muted}; font-size: 11px;")
    return lab


def _empty(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab.setStyleSheet(f"color: {T.CURRENT.muted}; padding: 14px;")
    return lab


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


def _role(model: Any, name: str) -> int:
    try:
        for key, val in model.roleNames().items():
            if bytes(val).decode() == name:
                return int(key)
    except Exception:  # noqa: BLE001
        pass
    return -1


def _connect(obj: Any, name: str, slot: Any) -> None:
    try:
        getattr(obj, name).connect(slot)
    except Exception:  # noqa: BLE001
        log.debug("connect %s failed", name, exc_info=True)
