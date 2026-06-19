"""In-app logging bridge: stdlib ``logging`` → Qt model for the QML log viewer.

Two pieces:

``QtLogHandler``
    A :class:`logging.Handler` subclass.  Each emitted record is formatted and
    marshalled onto the GUI thread via a Qt signal (queued connection), so it
    is always safe to attach to the root logger even though records may be
    emitted from worker threads.

``LogListModel``
    A :class:`~PySide6.QtCore.QAbstractListModel` ring buffer (default 2000
    rows) exposed to QML with roles ``level``, ``logger``, ``message``, ``ts``.

Wiring (done in ``app.py``)::

    log_model = LogListModel()
    handler = QtLogHandler(log_model)
    logging.getLogger().addHandler(handler)
    ctx.setContextProperty("logModel", log_model)

The QML ``LogConsole`` (built by another agent) binds to the ``logModel``
context property and reads the roles above.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

from PySide6.QtCore import (
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QThread,
    Qt,
    Signal,
    Slot,
)

# Maximum number of rows retained by the model (ring buffer).  Sized large so an
# extended diagnostics session can be exported / copied in full.
DEFAULT_CAPACITY = 5000


class LogListModel(QAbstractListModel):
    """Ring-buffered list of log records exposed to QML.

    Roles: ``level`` (str), ``logger`` (str), ``message`` (str), ``ts`` (str —
    ISO-ish ``HH:MM:SS.mmm`` formatted time).
    """

    LevelRole   = Qt.ItemDataRole.UserRole + 1
    LoggerRole  = Qt.ItemDataRole.UserRole + 2
    MessageRole = Qt.ItemDataRole.UserRole + 3
    TsRole      = Qt.ItemDataRole.UserRole + 4

    def __init__(self, capacity: int = DEFAULT_CAPACITY, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._capacity = max(1, int(capacity))
        # Each row is a dict: {level, logger, message, ts}
        self._rows: deque[dict[str, str]] = deque(maxlen=self._capacity)

    # ── QAbstractListModel API ─────────────────────────────────────────────────

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self._rows)

    def roleNames(self) -> dict[int, QByteArray]:
        return {
            self.LevelRole:   QByteArray(b"level"),
            self.LoggerRole:  QByteArray(b"logger"),
            self.MessageRole: QByteArray(b"message"),
            self.TsRole:      QByteArray(b"ts"),
        }

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or index.row() >= len(self._rows):
            return None
        row = self._rows[index.row()]
        match role:
            case self.LevelRole:   return row["level"]
            case self.LoggerRole:  return row["logger"]
            case self.MessageRole: return row["message"]
            case self.TsRole:      return row["ts"]
        return None

    # ── mutation ────────────────────────────────────────────────────────────────

    @Slot(str, str, str, str)
    def appendRow(self, level: str, logger: str, message: str, ts: str) -> None:
        """Append one log row (GUI thread only).

        When the ring buffer is full the oldest row is evicted; the model is
        kept in sync with the deque by removing row 0 first, then inserting at
        the tail.
        """
        full = len(self._rows) >= self._capacity
        if full:
            # The deque will drop row 0 on append; tell the view first.
            self.beginRemoveRows(QModelIndex(), 0, 0)
            self._rows.popleft()
            self.endRemoveRows()
        row = len(self._rows)
        self.beginInsertRows(QModelIndex(), row, row)
        self._rows.append({
            "level": level,
            "logger": logger,
            "message": message,
            "ts": ts,
        })
        self.endInsertRows()

    @Slot()
    def clear(self) -> None:
        """Drop all rows."""
        if not self._rows:
            return
        self.beginResetModel()
        self._rows.clear()
        self.endResetModel()

    def dump_text(self) -> str:
        """Return the full buffered log as one plain-text block (oldest → newest).

        Each line is ``ts  LEVEL  logger  message``.  Used by the EngineClient's
        copy/export slots so the operator can hand the console off to a bug report.
        """
        lines = [
            f"{r['ts']}  {r['level']:<8}  {r['logger']}  {r['message']}"
            for r in self._rows
        ]
        return "\n".join(lines)

    # ── test helpers ──────────────────────────────────────────────────────────

    def rows(self) -> list[dict[str, str]]:
        """Return a snapshot of the current rows (oldest → newest)."""
        return list(self._rows)


class QtLogHandler(logging.Handler):
    """A ``logging.Handler`` that pushes records into a :class:`LogListModel`.

    Records may be emitted from any thread, so the append is marshalled onto
    the GUI thread (the thread that owns *model*) via a queued Qt signal.  This
    means the handler is safe to attach to the root logger.
    """

    # Internal signal carrying (level, logger, message, ts).  Queued so the
    # connected slot runs on the owning (GUI) thread.
    class _Emitter(QObject):
        recordReady = Signal(str, str, str, str)

    _TIME_FMT = "%H:%M:%S"

    def __init__(self, model: LogListModel, level: int = logging.NOTSET) -> None:
        super().__init__(level=level)
        self._model = model
        self._emitter = self._Emitter()
        # Queued connection: emit() may run on a worker thread, but appendRow
        # must mutate the model on the GUI thread.
        self._emitter.recordReady.connect(
            model.appendRow, Qt.ConnectionType.QueuedConnection,
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                # Append a formatted traceback so errors are visible in-app.
                message = f"{message}\n{self.formatter.formatException(record.exc_info)}" \
                    if self.formatter else message
            ts = self.formatTime(record)
            if QThread.currentThread() is self._model.thread():
                self._model.appendRow(record.levelname, record.name, message, ts)
                return
            self._emitter.recordReady.emit(
                record.levelname,
                record.name,
                message,
                ts,
            )
        except Exception:  # noqa: BLE001 — logging must never raise into callers
            self.handleError(record)

    def formatTime(self, record: logging.LogRecord) -> str:
        import time as _time
        base = _time.strftime(self._TIME_FMT, _time.localtime(record.created))
        return f"{base}.{int(record.msecs):03d}"
