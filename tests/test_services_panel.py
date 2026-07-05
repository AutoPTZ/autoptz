"""ServicesPanel (offscreen): the panel's own surface, built with a real
EngineClient so construction exercises the real refresh path.

Runs in its own process (CI shards per file) and builds a real ``QApplication``
via a local ``qtapp`` fixture — widgets need a GUI application object, not the
session-scoped headless ``QCoreApplication``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _panel(tmp_path: Path):
    from autoptz.config.store import ConfigStore
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.widgets.services_panel import ServicesPanel

    client = EngineClient(store=ConfigStore(db_path=tmp_path / "cfg.db", debounce_s=0))
    return ServicesPanel(client)


def test_no_experimental_button_in_footer(qtapp, tmp_path) -> None:
    """Experimental Features moved to the Help menu — the Services panel must no
    longer carry its own "Experimental..." button."""
    from PySide6.QtWidgets import QPushButton

    panel = _panel(tmp_path)
    try:
        assert not hasattr(panel, "_experimental_btn")
        buttons = [b.text() for b in panel.findChildren(QPushButton)]
        assert not any("Experimental" in (t or "") for t in buttons), buttons
    finally:
        panel.deleteLater()
