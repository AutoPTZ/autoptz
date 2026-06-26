"""MarkExitDialog (offscreen): Return / Quit choice + optional save-results box."""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _dlg(qtapp, **kw):
    from autoptz.ui.widgets.dialogs.mark_exit import MarkExitDialog

    return MarkExitDialog(**kw)


def test_default_choice_is_none(qtapp) -> None:
    dlg = _dlg(qtapp)
    assert dlg.choice() is None
    dlg.deleteLater()


def test_return_choice(qtapp) -> None:
    from autoptz.ui.widgets.dialogs.mark_exit import RETURN

    dlg = _dlg(qtapp)
    dlg._on_return()
    assert dlg.choice() == RETURN
    dlg.deleteLater()


def test_quit_choice(qtapp) -> None:
    from autoptz.ui.widgets.dialogs.mark_exit import QUIT

    dlg = _dlg(qtapp)
    dlg._on_quit()
    assert dlg.choice() == QUIT
    dlg.deleteLater()


def test_save_box_disabled_without_result(qtapp) -> None:
    dlg = _dlg(qtapp, has_result=False)
    # No result → the save box is unchecked AND disabled so it can't lie.
    assert dlg.save_results() is False
    assert dlg._save_box.isEnabled() is False
    dlg.deleteLater()


def test_save_box_enabled_and_checked_with_result(qtapp) -> None:
    dlg = _dlg(qtapp, has_result=True)
    assert dlg._save_box.isEnabled() is True
    assert dlg.save_results() is True
    dlg._save_box.setChecked(False)
    assert dlg.save_results() is False
    dlg.deleteLater()
