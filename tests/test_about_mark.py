"""AboutMarkDialog (offscreen): constructs, shows MARK_VERSION + guide/FPS/do-dont text."""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_about_mark_has_version_and_guide(qtapp):
    from autoptz.ui.widgets.dialogs.about_mark import MARK_VERSION, AboutMarkDialog

    assert MARK_VERSION == "1.0.0"
    dlg = AboutMarkDialog()
    assert dlg.windowTitle() == "About AutoPTZ Mark"
    text = _all_label_text(dlg)
    assert MARK_VERSION in text
    assert "Excellent" in text and "Check load" in text  # FPS targets
    assert "power" in text.lower()  # Do/Don't
    dlg.deleteLater()


def _all_label_text(widget) -> str:
    from PySide6.QtWidgets import QLabel

    return " ".join(lbl.text() for lbl in widget.findChildren(QLabel))
