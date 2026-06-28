"""LogsPanel (offscreen): per-row coloring, a stylesheet that doesn't clobber it,
and user-resizable + persisted columns.

Runs in its own process (CI shards per file), so it builds a real ``QApplication``
via a local ``qtapp`` fixture rather than the headless ``QCoreApplication`` the
session-scoped ``qapp`` fixture provides — widgets need a GUI application object.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtCore import Qt

# Plain-int alias so role lookups read cleanly; importing QtCore needs no QApplication.
_FOREGROUND = int(Qt.ItemDataRole.ForegroundRole)


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _client(settings: dict[str, Any] | None = None) -> SimpleNamespace:
    """A minimal EngineClient stand-in: the few slots LogsPanel calls + a settings store."""
    store: dict[str, Any] = dict(settings or {})

    def get_setting(key: str, default: Any = None) -> Any:
        return store.get(key, default)

    def set_setting(key: str, value: Any) -> None:
        store[key] = value

    return SimpleNamespace(
        setLogLevel=lambda *_: None,
        copyLogsToClipboard=lambda *_: None,
        exportLogs=lambda *_: None,
        getSetting=get_setting,
        setSetting=set_setting,
        _store=store,
    )


def _seed_model():
    from autoptz.ui.log_bridge import LogListModel

    src = LogListModel()
    src.appendRow("ERROR", "autoptz.engine.supervisor", "boom", "00:00:01.000")
    src.appendRow("WARNING", "autoptz.engine.camera_worker", "slow frame", "00:00:02.000")
    src.appendRow("DEBUG", "autoptz.engine.pipeline.detect", "tick", "00:00:03.000")
    src.appendRow("INFO", "autoptz.engine.supervisor", "started", "00:00:04.000")
    src.appendRow(
        "INFO", "autoptz.engine.camera_worker", "opened camera_id=abcdef123456", "00:00:05.000"
    )
    return src


# ── model coloring (the data the renderer must honor) ────────────────────────


def test_model_foreground_role_maps_levels_to_theme_colors(qtapp) -> None:
    from PySide6.QtGui import QColor

    from autoptz.ui import theme as T
    from autoptz.ui.widgets.logs_panel import LogTableModel

    model = LogTableModel(_seed_model())
    fg = lambda r: model.data(model.index(r, 1), role=int(_FOREGROUND))  # noqa: E731

    assert fg(0) == QColor(T.ERROR)
    assert fg(1) == QColor(T.WARNING)
    assert fg(2) == QColor(T.CURRENT.muted)  # DEBUG → muted
    assert fg(3) == QColor(T.CURRENT.text)  # INFO → default text


def test_model_tints_message_column_per_camera_for_normal_rows(qtapp) -> None:
    from PySide6.QtGui import QColor

    from autoptz.ui import theme as T
    from autoptz.ui.widgets.logs_panel import LogTableModel

    model = LogTableModel(_seed_model())
    # Row 4 is an INFO row carrying a camera_id: its Message column (col 3) gets a
    # stable per-camera tint, distinct from the plain text color.
    msg_color = model.data(model.index(4, 3), role=int(_FOREGROUND))
    assert isinstance(msg_color, QColor)
    assert msg_color != QColor(T.CURRENT.text)
    # The non-message columns of that same row stay on the level color, not the tint.
    assert model.data(model.index(4, 1), role=int(_FOREGROUND)) == QColor(T.CURRENT.text)


def test_warning_message_keeps_level_color_not_camera_tint(qtapp) -> None:
    """A WARNING/ERROR must stay on its level color even with a camera_id present."""
    from PySide6.QtGui import QColor

    from autoptz.ui import theme as T
    from autoptz.ui.log_bridge import LogListModel
    from autoptz.ui.widgets.logs_panel import LogTableModel

    src = LogListModel()
    src.appendRow("WARNING", "autoptz.engine.camera_worker", "x camera_id=abcdef99", "00:00:01.000")
    model = LogTableModel(src)
    assert model.data(model.index(0, 3), role=int(_FOREGROUND)) == QColor(T.WARNING)


# ── the actual rendering bug: a delegate must carry the color into the option ─


def test_delegate_carries_foreground_role_into_palette(qtapp) -> None:
    """The fix for "logs aren't colored": a delegate copies ForegroundRole into the
    style option's palette, which the painter honors even under a view stylesheet
    (a hardcoded ``QTableView::item { color }`` otherwise silently wins)."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QStyleOptionViewItem

    from autoptz.ui import theme as T
    from autoptz.ui.widgets.logs_panel import LogItemDelegate, LogTableModel

    model = LogTableModel(_seed_model())
    delegate = LogItemDelegate()
    opt = QStyleOptionViewItem()
    delegate.initStyleOption(opt, model.index(0, 1))  # ERROR row
    # Both normal and selected text must carry the color so the row stays colored
    # whether or not it's selected.
    assert opt.palette.color(QPalette.ColorRole.Text) == QColor(T.ERROR)
    assert opt.palette.color(QPalette.ColorRole.HighlightedText) == QColor(T.ERROR)


def test_panel_installs_the_color_delegate(qtapp) -> None:
    from autoptz.ui.log_bridge import LogListModel
    from autoptz.ui.widgets.logs_panel import LogItemDelegate, LogsPanel

    panel = LogsPanel(_client(), LogListModel())
    assert isinstance(panel._table.itemDelegate(), LogItemDelegate)


def test_table_stylesheet_does_not_pin_item_text_color(qtapp) -> None:
    """Regression guard: a ``QTableView::item`` rule that sets ``color`` overrides
    the per-row ForegroundRole — the exact reason the logs rendered monochrome."""
    from autoptz.ui.log_bridge import LogListModel
    from autoptz.ui.widgets.logs_panel import LogsPanel

    panel = LogsPanel(_client(), LogListModel())
    ss = panel._table.styleSheet()
    # Pull out the ::item rule body (if any) and assert it pins no foreground color.
    if "::item" in ss:
        body = ss.split("::item", 1)[1]
        body = body.split("}", 1)[0]
        body = body.replace("background-color", "").replace("selection-color", "")
        assert "color:" not in body, f"::item still pins a foreground color: {body!r}"


# ── resizable + persisted columns ────────────────────────────────────────────


def test_first_columns_are_user_resizable(qtapp) -> None:
    from PySide6.QtWidgets import QHeaderView

    from autoptz.ui.log_bridge import LogListModel
    from autoptz.ui.widgets.logs_panel import LogsPanel

    panel = LogsPanel(_client(), LogListModel())
    hh = panel._table.horizontalHeader()
    for col in (0, 1, 2):
        assert hh.sectionResizeMode(col) == QHeaderView.ResizeMode.Interactive, (
            f"column {col} is not draggable"
        )


def test_resizing_a_column_persists_its_width(qtapp) -> None:
    from autoptz.ui.log_bridge import LogListModel
    from autoptz.ui.widgets.logs_panel import LogsPanel

    client = _client()
    panel = LogsPanel(client, LogListModel())
    panel._table.horizontalHeader().resizeSection(1, 123)
    saved = client.getSetting("logs_columns", {})
    assert saved.get("1") == 123


def test_persisted_column_widths_are_restored_on_construct(qtapp) -> None:
    from autoptz.ui.log_bridge import LogListModel
    from autoptz.ui.widgets.logs_panel import LogsPanel

    client = _client({"logs_columns": {"0": 140, "2": 200}})
    panel = LogsPanel(client, LogListModel())
    hh = panel._table.horizontalHeader()
    assert hh.sectionSize(0) == 140
    assert hh.sectionSize(2) == 200
