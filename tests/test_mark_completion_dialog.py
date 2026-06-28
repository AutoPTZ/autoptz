"""MarkCompletionDialog (offscreen): Return / Quit / Open / Save choice on a finished ramp.

The completion modal mirrors the MarkExitDialog seam — the window reads
:meth:`choice` after :meth:`exec` and listens to ``saveRequested`` / ``openRequested``
so the window owns the file I/O.  No ``.exec()`` is ever entered (the button slots
are driven directly).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _dlg(qtapp, **kw):
    from autoptz.ui.widgets.dialogs.mark_completion import MarkCompletionDialog

    base = {"verdict": "Done — 2 cameras at 28 fps (score 2.0).", "has_saved": False}
    base.update(kw)
    return MarkCompletionDialog(**base)


def test_default_choice_none(qtapp) -> None:
    dlg = _dlg(qtapp)
    assert dlg.choice() is None
    dlg.deleteLater()


def test_return_button_sets_choice_return(qtapp) -> None:
    from autoptz.ui.widgets.dialogs.mark_completion import RETURN

    dlg = _dlg(qtapp)
    dlg._on_return()
    assert dlg.choice() == RETURN
    dlg.deleteLater()


def test_quit_button_sets_choice_quit(qtapp) -> None:
    from autoptz.ui.widgets.dialogs.mark_completion import QUIT

    dlg = _dlg(qtapp)
    dlg._on_quit()
    assert dlg.choice() == QUIT
    dlg.deleteLater()


def test_open_disabled_without_saved_path(qtapp) -> None:
    dlg = _dlg(qtapp, has_saved=False)
    # Nothing saved yet → the Open button can't lie; it's disabled with a hint.
    assert dlg._open_btn.isEnabled() is False
    assert dlg._open_btn.toolTip()  # a "save first" hint
    dlg.deleteLater()


def test_open_enabled_with_saved_path(qtapp) -> None:
    dlg = _dlg(qtapp, has_saved=True)
    assert dlg._open_btn.isEnabled() is True
    dlg.deleteLater()


def test_save_button_emits_save_requested(qtapp) -> None:
    dlg = _dlg(qtapp)
    seen: list[int] = []
    dlg.saveRequested.connect(lambda: seen.append(1))
    dlg._on_save()
    assert seen == [1]
    # Save must NOT close the dialog — the user can still choose Return / Quit.
    assert dlg.choice() is None
    dlg.deleteLater()


def test_open_button_emits_open_requested(qtapp) -> None:
    dlg = _dlg(qtapp, has_saved=True)
    seen: list[int] = []
    dlg.openRequested.connect(lambda: seen.append(1))
    dlg._on_open()
    assert seen == [1]
    assert dlg.choice() is None  # Open does not close the dialog either
    dlg.deleteLater()


def test_set_saved_enables_open(qtapp) -> None:
    """After a save the window calls set_saved(True) so Open lights up live."""
    dlg = _dlg(qtapp, has_saved=False)
    assert dlg._open_btn.isEnabled() is False
    dlg.set_saved(True)
    assert dlg._open_btn.isEnabled() is True
    dlg.deleteLater()


def test_dialog_shows_verdict_rating_reason(qtapp) -> None:
    from PySide6.QtWidgets import QLabel

    dlg = _dlg(
        qtapp,
        verdict="Done — 2 cameras at 28 fps (score 2.0).",
        rating="Good",
        reason="score = 2 cam × 28/30 fps × 1.0 weight = 2.0",
    )
    texts = " || ".join(lbl.text() for lbl in dlg.findChildren(QLabel))
    assert "Done — 2 cameras" in texts
    assert "Good" in texts
    assert "score = 2 cam" in texts
    dlg.deleteLater()


def test_optional_rating_reason_absent(qtapp) -> None:
    """rating / reason are optional (P3 fills them) — omitting them must not crash and
    the verdict still renders."""
    from PySide6.QtWidgets import QLabel

    dlg = _dlg(qtapp, verdict="Done — score 2.0.")
    texts = " || ".join(lbl.text() for lbl in dlg.findChildren(QLabel))
    assert "Done — score 2.0." in texts
    dlg.deleteLater()
