"""LogsPanel — filterable log console.

A ``QTableView`` (Time / Level / Logger / Message) over the shared
``LogListModel`` through a filtering proxy that separates logs by **Source**
category (logger-name prefixes — Cameras / Detection / Identity / PTZ / Engine),
a minimum **level**, and free **text**.  Capture level (INFO/DEBUG), Copy,
Export, Clear, and auto-scroll round it out.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.log_bridge import LogListModel
from autoptz.ui.widgets.common import on_theme_changed

log = logging.getLogger(__name__)

_LEVEL_RANK = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

# Friendly source categories → logger-name prefixes.
_SOURCES: list[tuple[str, tuple[str, ...]]] = [
    ("All sources", ()),
    ("Cameras", ("autoptz.engine.camera_worker",)),
    ("Detection / Tracking", ("autoptz.engine.pipeline.detect", "autoptz.engine.pipeline.track")),
    ("Identity / Faces", ("autoptz.engine.pipeline.identify", "autoptz.engine.pipeline.reid",
                          "autoptz.engine.identity")),
    ("PTZ", ("autoptz.engine.ptz",)),
    ("Engine / System", ("autoptz.engine.supervisor", "autoptz.engine.runtime",
                         "autoptz.engine.discovery", "autoptz.ui")),
]


class LogTableModel(QAbstractTableModel):
    """4-column (Time/Level/Logger/Message) filtered mirror of a LogListModel.

    A ``QSortFilterProxyModel`` can't add columns the source lacks, so this model
    mirrors the source rows as tuples and presents them as a real table.  It
    tracks the source's leading-eviction ring buffer (remove row 0, append tail)
    incrementally so appends don't reset scroll/selection.
    """

    _HEADERS = ["Time", "Level", "Logger", "Message"]

    def __init__(self, source: LogListModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._src = source
        self._raw: list[tuple[str, str, str, str]] = []   # mirrors ALL source rows
        self._view: list[tuple[str, str, str, str]] = []  # filtered subset (same objs)
        self._prefixes: tuple[str, ...] = ()
        self._min_rank = 0
        self._text = ""
        self._seed()
        source.rowsInserted.connect(self._on_inserted)
        source.rowsRemoved.connect(self._on_removed)
        source.modelReset.connect(self._on_reset)

    # ── filters ────────────────────────────────────────────────────────────────

    def set_source_prefixes(self, prefixes: tuple[str, ...]) -> None:
        self._prefixes = prefixes
        self._rebuild_view()

    def set_min_level(self, level: str) -> None:
        self._min_rank = _LEVEL_RANK.get(level, 0)
        self._rebuild_view()

    def set_text(self, text: str) -> None:
        self._text = text.lower().strip()
        self._rebuild_view()

    # ── QAbstractTableModel ──────────────────────────────────────────────────────

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self._view)

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        return 4

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._HEADERS[section] if 0 <= section < 4 else None
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._view):
            return None
        row = self._view[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return row[index.column()]
        if role == Qt.ItemDataRole.ForegroundRole:
            return QColor(_level_color(row[1]))
        return None

    # ── source tracking ──────────────────────────────────────────────────────────

    def _read(self, r: int) -> tuple[str, str, str, str]:
        idx = self._src.index(r, 0)
        return (
            str(self._src.data(idx, LogListModel.TsRole) or ""),
            str(self._src.data(idx, LogListModel.LevelRole) or ""),
            str(self._src.data(idx, LogListModel.LoggerRole) or ""),
            str(self._src.data(idx, LogListModel.MessageRole) or ""),
        )

    def _passes(self, t: tuple[str, str, str, str]) -> bool:
        _ts, level, logger, message = t
        if self._prefixes and not any(logger.startswith(p) for p in self._prefixes):
            return False
        if _LEVEL_RANK.get(level, 0) < self._min_rank:
            return False
        if self._text and self._text not in logger.lower() and self._text not in message.lower():
            return False
        return True

    def _seed(self) -> None:
        self._raw = [self._read(r) for r in range(self._src.rowCount())]
        self._view = [t for t in self._raw if self._passes(t)]

    def _rebuild_view(self) -> None:
        self.beginResetModel()
        self._view = [t for t in self._raw if self._passes(t)]
        self.endResetModel()

    def _on_inserted(self, _parent: QModelIndex, first: int, last: int) -> None:
        for r in range(first, last + 1):
            t = self._read(r)
            self._raw.insert(r, t)
            if self._passes(t):
                row = len(self._view)
                self.beginInsertRows(QModelIndex(), row, row)
                self._view.append(t)
                self.endInsertRows()

    def _on_removed(self, _parent: QModelIndex, first: int, last: int) -> None:
        for r in range(last, first - 1, -1):
            if r >= len(self._raw):
                continue
            evicted = self._raw.pop(r)
            for vi, vt in enumerate(self._view):
                if vt is evicted:
                    self.beginRemoveRows(QModelIndex(), vi, vi)
                    self._view.pop(vi)
                    self.endRemoveRows()
                    break

    def _on_reset(self) -> None:
        self.beginResetModel()
        self._seed()
        self.endResetModel()


class LogsPanel(QWidget):
    """Filterable log console bound to the shared LogListModel."""

    def __init__(self, client: Any, log_model: LogListModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._autoscroll = True

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # toolbar
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self._source = QComboBox()
        for label, _ in _SOURCES:
            self._source.addItem(label)
        self._source.currentIndexChanged.connect(self._apply_source)
        bar.addWidget(QLabel("Source:"))
        bar.addWidget(self._source)

        self._level = QComboBox()
        self._level.addItems(["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self._level.currentTextChanged.connect(
            lambda t: self._proxy.set_min_level("" if t == "All" else t)
        )
        bar.addWidget(QLabel("Level:"))
        bar.addWidget(self._level)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search…")
        self._search.textChanged.connect(lambda t: self._proxy.set_text(t))
        bar.addWidget(self._search, 1)

        bar.addWidget(QLabel("Capture:"))
        self._capture = QComboBox(); self._capture.addItems(["INFO", "DEBUG"])
        self._capture.currentTextChanged.connect(client.setLogLevel)
        bar.addWidget(self._capture)
        root.addLayout(bar)

        # actions
        actions = QHBoxLayout()
        self._autoscroll_btn = QPushButton("Autoscroll: On")
        self._autoscroll_btn.setCheckable(True); self._autoscroll_btn.setChecked(True)
        self._autoscroll_btn.toggled.connect(self._toggle_autoscroll)
        copy_btn = QPushButton("Copy"); copy_btn.clicked.connect(client.copyLogsToClipboard)
        export_btn = QPushButton("Export…"); export_btn.clicked.connect(self._export)
        clear_btn = QPushButton("Clear"); clear_btn.clicked.connect(log_model.clear)
        actions.addWidget(self._autoscroll_btn)
        actions.addStretch(1)
        for b in (copy_btn, export_btn, clear_btn):
            actions.addWidget(b)
        root.addLayout(actions)

        # table
        self._proxy = LogTableModel(log_model, self)
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setShowGrid(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.setWordWrap(False)
        mono = self._table.font(); mono.setFamily("Menlo"); mono.setPixelSize(11)
        self._table.setFont(mono)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, 1)

        self._proxy.rowsInserted.connect(self._on_rows)
        # Level colors are palette-derived (DEBUG=muted, default=text); repaint
        # the table when the appearance flips so rows aren't stuck dark-on-light.
        on_theme_changed(client, self._restyle)

    def _restyle(self) -> None:
        """Repaint the log table so palette-driven row colors track the theme."""
        self._table.viewport().update()

    # ── filters / actions ────────────────────────────────────────────────────────

    def _apply_source(self, index: int) -> None:
        if 0 <= index < len(_SOURCES):
            self._proxy.set_source_prefixes(_SOURCES[index][1])

    def _toggle_autoscroll(self, on: bool) -> None:
        self._autoscroll = on
        self._autoscroll_btn.setText(f"Autoscroll: {'On' if on else 'Off'}")
        if on:
            self._table.scrollToBottom()

    def _on_rows(self, *_args: Any) -> None:
        if self._autoscroll:
            self._table.scrollToBottom()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Logs", "autoptz-logs.txt",
                                              "Text files (*.txt)")
        if path:
            self._client.exportLogs(path)


def _level_color(level: str) -> str:
    if level in ("ERROR", "CRITICAL"):
        return T.ERROR
    if level == "WARNING":
        return T.WARNING
    if level == "DEBUG":
        return T.CURRENT.muted
    return T.CURRENT.text
