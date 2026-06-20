"""CameraWall — the central responsive grid of :class:`CameraTile` widgets.

Builds and tears down tiles from the :class:`CameraListModel` (reacting to
``cameraAdded``/``cameraRemoved``), reflows them into a responsive grid whose
column count is chosen so each cell stays close to 16:9 (never a tall single
column for multiple cameras), and tracks the selected camera (emitting
``cameraSelected`` so the Properties / Camera Info panels can follow it).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.camera_tile import CameraTile
from autoptz.ui.widgets.common import on_theme_changed

log = logging.getLogger(__name__)

# The aspect ratio every tile wants to read as — cameras are virtually always
# 16:9, so "Auto" chooses a column count that keeps cells as close to this as the
# wall allows (rather than collapsing into a tall single column).
_TILE_ASPECT = 16.0 / 9.0

# Named layout presets shown in the toolbar dropdown: (label, key).  "auto"
# balances the grid so every cell stays near 16:9; the rest pin an explicit
# column count whose paired number is the *minimum* row count so a 2×2 reads as a
# 2×2 even with fewer cameras (grid grows downward when there are more).  Empty
# cells just stay blank, which is what makes the wall read simply.  (No "stacked"
# single-column option — a vertical column of 16:9 tiles is never what you want.)
_LAYOUT_PRESETS: list[tuple[str, str]] = [
    ("Auto", "auto"),
    ("2×2", "2x2"),
    ("3×2", "3x2"),
    ("2×3", "2x3"),
    ("3×3", "3x3"),
]
# preset key → (columns, minimum rows)
_PRESET_DIMS: dict[str, tuple[int, int]] = {
    "2x2": (2, 2),
    "3x2": (3, 2),
    "2x3": (2, 3),
    "3x3": (3, 3),
}
_DEFAULT_LAYOUT = "auto"
_WALL_MARGIN = 6
_WALL_SPACING = 6


class _EmptyCameraSlot(QWidget):
    """Clickable placeholder for an unused fixed-grid camera slot."""

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("emptyCameraSlot")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._label = QLabel("+ Add camera", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.restyle()

    def restyle(self) -> None:
        pal = T.CURRENT
        self.setStyleSheet(
            f"QWidget#emptyCameraSlot {{ background: {pal.surface_alt};"
            f" border: 1px dashed {pal.border_hov}; border-radius: {T.RADIUS}px; }}"
            f"QWidget#emptyCameraSlot:hover {{ background: {pal.surface_hov};"
            f" border-color: {T.ACCENT.name()}; }}"
        )
        self._label.setStyleSheet(
            f"color: {pal.subtext}; font-size: {T.fs(12)}px; font-weight: 700;"
        )

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._label.setGeometry(self.rect())

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class CameraWall(QWidget):
    """Grid of live camera tiles with selection + an empty state."""

    cameraSelected = Signal(str)  # camera_id (or "" when none)
    cameraInfoRequested = Signal(str)  # camera_id — open Camera Info for it
    addCameraRequested = Signal()  # empty fixed-grid slot clicked

    def __init__(self, client: Any, frame_source: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraWall")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._client = client
        self._frames = frame_source
        self._tiles: dict[str, CameraTile] = {}
        self._empty_slots: list[_EmptyCameraSlot] = []
        self._selected: str = ""
        self._drag_camera_id = ""
        self._drag_insert_index: int | None = None
        self._drag_cursor_offset: tuple[float, float] = (0.0, 0.0)
        self._layout = str(
            _safe(lambda: client.getSetting("wall_layout", _DEFAULT_LAYOUT), _DEFAULT_LAYOUT)
            or _DEFAULT_LAYOUT
        )

        self._sub_labels: list[QLabel] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_toolbar())

        # Stacked content: [grid | empty-state]
        content = QWidget(self)
        self._stack = QStackedLayout(content)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._grid_host = QWidget(content)
        self._grid_host.setObjectName("cameraGridHost")
        self._grid_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Clicking empty space between/around tiles deselects the active camera
        # (tiles sit on top and handle their own clicks, so this only fires on the
        # bare grid background).
        self._grid_host.mousePressEvent = self._on_grid_background_click
        self._stack.addWidget(self._grid_host)
        self._insert_line = QWidget(self._grid_host)
        self._insert_line.setObjectName("cameraInsertLine")
        self._insert_line.setStyleSheet(
            f"QWidget#cameraInsertLine {{ background: {T.ACCENT.name()}; border-radius: 2px; }}"
        )
        self._insert_line.hide()

        self._empty = QLabel("No cameras\n\nAdd a source from the Cameras menu.", content)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet(f"color: {T.CURRENT.subtext};")
        self._stack.addWidget(self._empty)
        root.addWidget(content, 1)

        for cid in _camera_ids(client):
            self._add_tile(cid)
        self._reflow()

        on_theme_changed(client, self._restyle_toolbar)
        on_theme_changed(client, self._restyle_empty_state)
        on_theme_changed(client, self._restyle_empty_slots)
        _connect(client, "cameraAdded", self._on_camera_added)
        _connect(client, "cameraRemoved", self._on_camera_removed)
        _connect(client.cameraModel, "layoutChanged", self._reflow)

    # ── toolbar ──────────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> QWidget:
        bar = QWidget(self)
        bar.setObjectName("wallToolbar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._toolbar = bar
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        title = QLabel("Cameras")
        title.setStyleSheet("font-weight: 700;")
        row.addWidget(title)
        row.addStretch(1)

        cols_lab = QLabel("Layout")
        self._sub_labels.append(cols_lab)
        row.addWidget(cols_lab)
        self._cols_combo = QComboBox()
        for label, key in _LAYOUT_PRESETS:
            self._cols_combo.addItem(label, key)
        idx = self._cols_combo.findData(self._layout)
        self._cols_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._cols_combo.setToolTip(
            "Arrange the camera tiles: Auto balances the grid so every tile stays "
            "close to 16:9; 2×2 / 3×2 / 2×3 / 3×3 are fixed grids."
        )
        self._cols_combo.currentIndexChanged.connect(self._on_layout_changed)
        row.addWidget(self._cols_combo)

        self._restyle_toolbar()
        return bar

    def _restyle_toolbar(self) -> None:
        """Re-apply literal-color toolbar styling (construction + theme change)."""
        self._toolbar.setStyleSheet(
            f"#wallToolbar {{ background: {T.CURRENT.sidebar_bg};"
            f" border-bottom: 1px solid {T.CURRENT.border}; }}"
        )
        for lab in self._sub_labels:
            lab.setStyleSheet(f"color: {T.CURRENT.subtext};")

    def _restyle_empty_slots(self) -> None:
        for slot in self._empty_slots:
            slot.restyle()

    def _restyle_empty_state(self) -> None:
        self._empty.setStyleSheet(f"color: {T.CURRENT.subtext};")

    def _on_layout_changed(self, _index: int) -> None:
        self._layout = str(self._cols_combo.currentData() or _DEFAULT_LAYOUT)
        _safe(lambda: self._client.setSetting("wall_layout", self._layout), None)
        self._reflow()

    # ── public ─────────────────────────────────────────────────────────────────

    @property
    def selected_camera_id(self) -> str:
        return self._selected

    def select_camera(self, camera_id: str) -> None:
        """Toggle selection from a tile background click.

        Clicking the background of the already-active tile *deselects* it (clears
        the selection glow and lets the Properties / Camera Info panels go idle),
        so the operator can unselect a camera without having to pick another one.
        """
        self._apply_selection("" if camera_id == self._selected else camera_id)

    def select_camera_exclusive(self, camera_id: str) -> None:
        """Select ``camera_id`` without toggling off (used for box/right clicks)."""
        if camera_id == self._selected:
            return
        self._apply_selection(camera_id)

    def clear_selection(self) -> None:
        """Deselect the active camera (no-op when nothing is selected)."""
        if self._selected:
            self._apply_selection("")

    def _apply_selection(self, camera_id: str) -> None:
        self._selected = camera_id
        for cid, tile in self._tiles.items():
            tile.set_selected(cid == camera_id)
        self.cameraSelected.emit(camera_id)

    def _on_grid_background_click(self, event: Any) -> None:
        """Deselect when the bare grid background (not a tile) is clicked."""
        self.clear_selection()
        event.accept()

    # ── tile management ────────────────────────────────────────────────────────

    def _add_tile(self, camera_id: str) -> None:
        if camera_id in self._tiles:
            return
        tile = CameraTile(camera_id, self._client, self._frames, self._grid_host)
        tile.selectedRequested.connect(self.select_camera)
        tile.selectExclusiveRequested.connect(self.select_camera_exclusive)
        tile.infoRequested.connect(self.cameraInfoRequested)
        tile.renameRequested.connect(self._rename_camera)
        tile.reorderDragStarted.connect(self._on_reorder_started)
        tile.reorderDragMoved.connect(self._on_reorder_moved)
        tile.reorderDragFinished.connect(self._on_reorder_finished)
        self._tiles[camera_id] = tile

    def _on_camera_added(self, camera_id: str) -> None:
        self._add_tile(camera_id)
        self._reflow()

    def _on_camera_removed(self, camera_id: str) -> None:
        tile = self._tiles.pop(camera_id, None)
        if tile is not None:
            tile.hide()
            tile.deleteLater()
        if self._selected == camera_id:
            self._apply_selection("")
        self._reflow()

    # ── drag reorder ─────────────────────────────────────────────────────────

    def _on_reorder_started(self, camera_id: str, global_pos: Any) -> None:
        self._drag_camera_id = camera_id
        self._drag_insert_index = None
        tile = self._tiles.get(camera_id)
        if tile is not None:
            pos = self._grid_host.mapFromGlobal(global_pos)
            geo = tile.geometry()
            self._drag_cursor_offset = (
                float(pos.x() - geo.x()),
                float(pos.y() - geo.y()),
            )
            tile.show()
            tile.raise_()
        self._on_reorder_moved(camera_id, global_pos)

    def _on_reorder_moved(self, camera_id: str, global_pos: Any) -> None:
        if camera_id != self._drag_camera_id:
            return
        pos = self._move_drag_tile(camera_id, global_pos)
        if pos is None:
            return
        rects = self._tile_rects()
        order = [cid for cid in _camera_ids(self._client) if cid in self._tiles]
        idx = _drop_index_from_rects(order, camera_id, pos.x(), pos.y(), rects)
        self._drag_insert_index = idx
        self._show_insert_line(idx)

    def _on_reorder_finished(self, camera_id: str, global_pos: Any) -> None:
        if camera_id == self._drag_camera_id:
            self._on_reorder_moved(camera_id, global_pos)
            if self._drag_insert_index is not None:
                try:
                    self._client.moveCameraPersisted(camera_id, int(self._drag_insert_index))
                except Exception:  # noqa: BLE001
                    log.debug("moveCameraPersisted failed", exc_info=True)
        self._drag_camera_id = ""
        self._drag_insert_index = None
        self._drag_cursor_offset = (0.0, 0.0)
        self._insert_line.hide()
        self._reflow()

    def _move_drag_tile(self, camera_id: str, global_pos: Any) -> Any | None:
        """Place the dragged tile so the original grab point stays under cursor."""
        tile = self._tiles.get(camera_id)
        if tile is None:
            return None
        pos = self._grid_host.mapFromGlobal(global_pos)
        off_x, off_y = self._drag_cursor_offset
        tile.setGeometry(
            round(float(pos.x()) - off_x),
            round(float(pos.y()) - off_y),
            tile.width(),
            tile.height(),
        )
        tile.show()
        tile.raise_()
        return pos

    def _tile_rects(self) -> dict[str, tuple[float, float, float, float]]:
        out: dict[str, tuple[float, float, float, float]] = {}
        for cid, tile in self._tiles.items():
            g = tile.geometry()
            out[cid] = (float(g.x()), float(g.y()), float(g.width()), float(g.height()))
        return out

    def _show_insert_line(self, index: int | None) -> None:
        if index is None:
            self._insert_line.hide()
            return
        order = [
            cid
            for cid in _camera_ids(self._client)
            if cid in self._tiles and cid != self._drag_camera_id
        ]
        if not order:
            self._insert_line.hide()
            return
        if index <= 0:
            target = self._tiles[order[0]].geometry()
            x = target.left() - max(3, _WALL_SPACING // 2)
        elif index >= len(order):
            target = self._tiles[order[-1]].geometry()
            x = target.right() + max(2, _WALL_SPACING // 2)
        else:
            target = self._tiles[order[index]].geometry()
            x = target.left() - max(3, _WALL_SPACING // 2)
        self._insert_line.setGeometry(round(x), target.top(), 4, target.height())
        self._insert_line.show()
        self._insert_line.raise_()

    def _cols_rows(self) -> tuple[int, int]:
        """Resolve the active preset to (columns, rows) for the current count.

        Fixed presets pin ``columns`` (rows grow downward past the minimum so
        nothing is hidden); "Auto" balances the grid so every cell stays as close
        to 16:9 as the wall allows — never a tall single column for >1 camera.
        """
        n = len(self._tiles) or 1
        preset = self._layout
        if preset in _PRESET_DIMS:
            cols, rows_min = _PRESET_DIMS[preset]
            rows = max(rows_min, (n + cols - 1) // cols)
            return cols, rows
        return self._auto_cols_rows(n)

    def _auto_cols_rows(self, n: int) -> tuple[int, int]:
        """Pick the column count whose resulting cells read closest to 16:9.

        For each candidate column count we compute the cell aspect the current
        wall size would produce and score it by its log-distance from 16:9 (so
        "twice too wide" and "twice too tall" weigh equally), breaking ties toward
        the grid with the fewest empty cells.  A single column only ever wins when
        the wall itself is genuinely taller than it is wide.
        """
        if n <= 1:
            return 1, 1
        wall_w = max(1.0, float(self._grid_host.width() or self.width()))
        host = getattr(self, "_grid_host", None)
        wall_h = max(1.0, float(host.height() if host is not None else self.height()))
        best: tuple[float, int, int, int] | None = None
        for cols in range(1, n + 1):
            rows = (n + cols - 1) // cols
            tile_w, tile_h = _fit_16x9_tile(wall_w, wall_h, cols, rows)
            area = tile_w * tile_h
            empties = cols * rows - n
            score = (-area, empties, cols)
            if best is None or score < best[:3]:
                best = (*score, rows)
        assert best is not None
        return best[2], best[3]

    def _reflow(self) -> None:
        # Empty-state vs grid.
        if not self._tiles:
            self._stack.setCurrentWidget(self._empty)
            self._clear_grid()
            return
        self._stack.setCurrentWidget(self._grid_host)

        # The grid host is sized by the QStackedLayout, but its geometry only
        # updates on the *next* layout pass — not synchronously inside
        # setCurrentWidget().  On first show (and right after switching from the
        # empty-state page) reading its size here therefore returns a stale, tiny
        # value, which placed every tile minuscule in the top-left until a manual
        # resize/relayout.  When that happens, defer one event-loop turn so we
        # reflow against the *settled* geometry.  Guard on the wall having a real
        # size so we don't spin before the window is shown (resizeEvent covers it).
        host_w = float(self._grid_host.width())
        host_h = float(self._grid_host.height())
        if (host_w <= 1.0 or host_h <= 1.0) and self.width() > 1 and self.height() > 1:
            QTimer.singleShot(0, self._reflow)
            return

        cols, rows = self._cols_rows()
        order = [cid for cid in _camera_ids(self._client) if cid in self._tiles]
        # include any tiles not yet in the model order (shouldn't happen, but safe)
        order += [cid for cid in self._tiles if cid not in order]

        self._clear_grid()
        tile_w, tile_h = _fit_16x9_tile(
            max(1.0, host_w),
            max(1.0, host_h),
            cols,
            rows,
        )
        grid_w = cols * tile_w + max(0, cols - 1) * _WALL_SPACING
        grid_h = rows * tile_h + max(0, rows - 1) * _WALL_SPACING
        x0 = (self._grid_host.width() - grid_w) / 2.0
        y0 = (self._grid_host.height() - grid_h) / 2.0
        for i, cid in enumerate(order):
            rr, cc = i // cols, i % cols
            tile = self._tiles[cid]
            tile.setParent(self._grid_host)
            tile.setGeometry(
                round(x0 + cc * (tile_w + _WALL_SPACING)),
                round(y0 + rr * (tile_h + _WALL_SPACING)),
                max(1, round(tile_w)),
                max(1, round(tile_h)),
            )
            tile.show()
        self._place_empty_slots(len(order), cols, rows, x0, y0, tile_w, tile_h)

    def _clear_grid(self) -> None:
        """Hide all tiles before manual 16:9 placement."""
        for tile in self._tiles.values():
            tile.hide()
        for slot in self._empty_slots:
            slot.hide()

    def _place_empty_slots(
        self,
        used: int,
        cols: int,
        rows: int,
        x0: float,
        y0: float,
        tile_w: float,
        tile_h: float,
    ) -> None:
        """Show blank add-camera slots for unused cells in fixed presets only."""
        needed = _placeholder_count(self._layout, used, cols, rows)
        self._ensure_empty_slots(needed)
        for i, slot in enumerate(self._empty_slots):
            if i >= needed:
                slot.hide()
                continue
            idx = used + i
            rr, cc = idx // cols, idx % cols
            slot.setParent(self._grid_host)
            slot.setGeometry(
                round(x0 + cc * (tile_w + _WALL_SPACING)),
                round(y0 + rr * (tile_h + _WALL_SPACING)),
                max(1, round(tile_w)),
                max(1, round(tile_h)),
            )
            slot.show()

    def _ensure_empty_slots(self, count: int) -> None:
        while len(self._empty_slots) < count:
            slot = _EmptyCameraSlot(self._grid_host)
            slot.clicked.connect(self.addCameraRequested.emit)
            self._empty_slots.append(slot)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        # First show: cameras may have been added at construction (size 0), and
        # Qt won't emit a resize if the shown size equals the constructed size, so
        # the only reflow so far ran against a zero-size grid.  Reflow now, then
        # again after the layout settles, so the grid is correct on first paint
        # instead of staying tiny/top-left until a manual resize.
        super().showEvent(event)
        self._reflow()
        QTimer.singleShot(0, self._reflow)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reflow()

    # ── rename (minimal; full inline rename can come later) ────────────────────

    def _rename_camera(self, camera_id: str) -> None:
        try:
            cfg = self._client.getCameraConfig(camera_id) or {}
        except Exception:  # noqa: BLE001
            cfg = {}
        current = cfg.get("name", "")
        # Size the dialog to the current name so long names aren't cramped into a
        # tiny fixed field (clamped so it never gets absurdly wide or too narrow).
        dlg = QInputDialog(self)
        dlg.setWindowTitle("Rename Camera")
        dlg.setLabelText("Name:")
        dlg.setTextValue(current)
        fm = dlg.fontMetrics()
        width = min(680, max(320, fm.horizontalAdvance(current or "Camera") + T.fs(150)))
        dlg.resize(width, dlg.sizeHint().height())
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        name = dlg.textValue()
        if name.strip():
            cfg["name"] = name.strip()
            try:
                self._client.updateCameraConfig(camera_id, json.dumps(cfg))
            except Exception:  # noqa: BLE001
                log.debug("rename updateCameraConfig failed", exc_info=True)


def _camera_ids(client: Any) -> list[str]:
    try:
        return list(client.cameraModel.camera_ids())
    except Exception:  # noqa: BLE001
        return []


def _fit_16x9_tile(wall_w: float, wall_h: float, cols: int, rows: int) -> tuple[float, float]:
    """Largest uniform 16:9 tile size that fits the wall grid."""
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    avail_w = max(1.0, wall_w - 2 * _WALL_MARGIN - max(0, cols - 1) * _WALL_SPACING)
    avail_h = max(1.0, wall_h - 2 * _WALL_MARGIN - max(0, rows - 1) * _WALL_SPACING)
    tile_w = min(avail_w / cols, (avail_h / rows) * _TILE_ASPECT)
    tile_h = tile_w / _TILE_ASPECT
    return max(1.0, tile_w), max(1.0, tile_h)


def _placeholder_count(layout: str, used: int, cols: int, rows: int) -> int:
    """How many empty add-camera cells a layout should render."""
    if layout not in _PRESET_DIMS:
        return 0
    return max(0, int(cols) * int(rows) - max(0, int(used)))


def _drop_index_from_rects(
    order: list[str],
    dragged: str,
    x: float,
    y: float,
    rects: dict[str, tuple[float, float, float, float]],
) -> int | None:
    """Return final insertion index for a drag point against tile rects.

    The returned index is relative to the camera order *after* removing the
    dragged tile, which matches ``CameraListModel.moveCamera`` semantics.
    """
    visible = [cid for cid in order if cid != dragged and cid in rects]
    if not visible:
        return None
    for i, cid in enumerate(visible):
        rx, ry, rw, rh = rects[cid]
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return i + (1 if x > rx + rw / 2.0 else 0)

    def distance(cid: str) -> float:
        rx, ry, rw, rh = rects[cid]
        cx, cy = rx + rw / 2.0, ry + rh / 2.0
        return (x - cx) ** 2 + (y - cy) ** 2

    nearest = min(visible, key=distance)
    i = visible.index(nearest)
    rx, ry, rw, rh = rects[nearest]
    after = y > ry + rh / 2.0 or (abs(y - (ry + rh / 2.0)) <= rh / 2.0 and x > rx + rw / 2.0)
    return i + (1 if after else 0)


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _connect(obj: Any, signal_name: str, slot: Any) -> None:
    try:
        getattr(obj, signal_name).connect(slot)
    except Exception:  # noqa: BLE001
        log.debug("could not connect %s", signal_name, exc_info=True)
