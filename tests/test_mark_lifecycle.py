"""In-process Mark lifecycle (offscreen): confirm→suspend→swap; return resumes; quit exits; no relaunch."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _no_autostart(monkeypatch):
    # Mark window construction must not spin up the real Supervisor in these tests.
    monkeypatch.setenv("AUTOPTZ_MARK_NO_AUTOSTART", "1")


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _main(qtapp):
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.main_window import MainWindow

    client = EngineClient()
    return MainWindow(client, frame_source=ShmFrameSource())


class _FakeDlg:
    def __init__(self, *a, **k) -> None: ...

    def exec(self):  # accepted
        return 1

    def session(self):
        from autoptz.ui.mark_session import MarkSession

        return MarkSession(max_cameras=2, dwell_s=0.0)


class _RejectDlg:
    def __init__(self, *a, **k) -> None: ...

    def exec(self):  # cancelled
        return 0

    def session(self):
        from autoptz.ui.mark_session import MarkSession

        return MarkSession(max_cameras=2, dwell_s=0.0)


def test_confirm_required_before_suspend(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.main_window as mw

    win = _main(qtapp)
    monkeypatch.setattr(mw, "MarkPreflightDialog", _RejectDlg, raising=False)
    hid = {"n": 0}
    monkeypatch.setattr(win, "hide", lambda: hid.__setitem__("n", 1))
    win._start_mark()
    # Cancelled confirm: main window NOT suspended (no hide), no Mark window built.
    assert hid["n"] == 0
    assert win._mark_window is None
    win.deleteLater()


def test_enter_mark_hides_main_and_builds_isolated_window(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.main_window as mw

    win = _main(qtapp)
    monkeypatch.setattr(mw, "MarkPreflightDialog", _FakeDlg, raising=False)
    hid = {"n": 0}
    monkeypatch.setattr(win, "hide", lambda: hid.__setitem__("n", 1))
    win._start_mark()
    assert hid["n"] == 1  # main suspended (hidden) on confirm
    assert win._mark_window is not None
    assert win._mark_window.windowTitle() == "AutoPTZ Mark"
    # Isolated engine: NOT the main client.
    assert win._mark_window._client is not win._client
    win._mark_window.close()


def test_return_resumes_main(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.main_window as mw

    win = _main(qtapp)
    monkeypatch.setattr(mw, "MarkPreflightDialog", _FakeDlg, raising=False)
    win._start_mark()
    mark = win._mark_window
    mark.request_return()
    assert win._mark_window is None
    assert not win.isHidden()


def test_quit_choice_quits_app(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.main_window as mw

    win = _main(qtapp)
    monkeypatch.setattr(mw, "MarkPreflightDialog", _FakeDlg, raising=False)
    quit_called = {"n": 0}
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "quit", lambda *a: quit_called.__setitem__("n", 1))
    win._start_mark()
    win._mark_window.request_quit()
    assert quit_called["n"] == 1


def test_os_close_routes_through_return(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.main_window as mw

    win = _main(qtapp)
    monkeypatch.setattr(mw, "MarkPreflightDialog", _FakeDlg, raising=False)
    win._start_mark()
    mark = win._mark_window
    # OS window button → closedUnexpectedly → handled as Return (main reshown).
    mark.close()
    assert win._mark_window is None
    assert not win.isHidden()


def test_mark_os_close_does_not_quit_app(qtapp, monkeypatch) -> None:
    """A Mark window OS-close must route to Return, never quit the app.

    Guards the interaction with the new MainWindow quit-on-close path: the Mark
    window inherits closeEvent and chains super(), but its OS-close must NOT trip
    the primary window's QApplication.quit().
    """
    from PySide6.QtWidgets import QApplication

    import autoptz.ui.widgets.main_window as mw

    win = _main(qtapp)
    monkeypatch.setattr(mw, "MarkPreflightDialog", _FakeDlg, raising=False)
    win._start_mark()
    mark = win._mark_window
    quit_called = {"n": 0}
    monkeypatch.setattr(QApplication, "quit", lambda *a: quit_called.__setitem__("n", 1))
    mark.close()
    assert quit_called["n"] == 0
    assert not win.isHidden()  # resumed via Return


def test_engine_resumes_if_was_running(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.main_window as mw

    win = _main(qtapp)
    # Pretend the main engine was running before Mark.
    started = {"n": 0}
    stopped = {"n": 0}
    monkeypatch.setattr(type(win._client), "engineRunning", property(lambda self: True))
    monkeypatch.setattr(win._client, "stopEngine", lambda: stopped.__setitem__("n", 1))
    monkeypatch.setattr(win._client, "startEngine", lambda: started.__setitem__("n", 1))
    monkeypatch.setattr(mw, "MarkPreflightDialog", _FakeDlg, raising=False)
    win._start_mark()
    assert stopped["n"] == 1  # suspended on enter
    win._mark_window.request_return()
    assert started["n"] == 1  # resumed on return


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows-only Qt event-loop fd flake in headless MarkWindow show/close; covered on macOS/Linux + live",
)
def test_main_close_quits_app(qtapp, monkeypatch) -> None:
    """Closing the visible MainWindow (no Mark swap) terminates the app.

    With setQuitOnLastWindowClosed(False) global (for the Mark in-process swap),
    the visible main window's close must explicitly quit, or app.exec() hangs.
    """
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QApplication

    win = _main(qtapp)
    quit_called = {"n": 0}
    monkeypatch.setattr(QApplication, "quit", lambda *a: quit_called.__setitem__("n", 1))
    assert win._mark_window is None  # not in a Mark swap
    win.closeEvent(QCloseEvent())
    assert quit_called["n"] == 1


def test_main_close_during_mark_swap_does_not_quit(qtapp, monkeypatch) -> None:
    """A close that arrives while a Mark swap is active must NOT quit the app.

    The main window is hidden (not closed) during Mark; but guard defensively so
    any stray close while ``_mark_window`` is set does not tear down the process.
    """
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QApplication

    import autoptz.ui.widgets.main_window as mw

    win = _main(qtapp)
    monkeypatch.setattr(mw, "MarkPreflightDialog", _FakeDlg, raising=False)
    win._start_mark()
    assert win._mark_window is not None  # in a Mark swap
    quit_called = {"n": 0}
    monkeypatch.setattr(QApplication, "quit", lambda *a: quit_called.__setitem__("n", 1))
    win.closeEvent(QCloseEvent())
    assert quit_called["n"] == 0
    win._mark_window.close()


def test_main_close_during_swap_then_mark_exit_does_not_resume_dead_window(
    qtapp, monkeypatch
) -> None:
    """If the main window is itself closed mid-swap, a later Mark Return must NOT try
    to re-show the dead main window — it quits the app instead (nothing to return to).
    """
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QApplication

    import autoptz.ui.widgets.main_window as mw

    win = _main(qtapp)
    monkeypatch.setattr(mw, "MarkPreflightDialog", _FakeDlg, raising=False)
    win._start_mark()
    mark = win._mark_window
    assert mark is not None
    # The main window receives a close WHILE the Mark swap is active.
    win.closeEvent(QCloseEvent())
    # Now the user chooses Return from Mark.  There is no live main window to resume,
    # so it must quit rather than re-show the closed one.
    shown = {"n": 0}
    quit_called = {"n": 0}
    monkeypatch.setattr(win, "show", lambda: shown.__setitem__("n", shown["n"] + 1))
    monkeypatch.setattr(QApplication, "quit", lambda *a: quit_called.__setitem__("n", 1))
    mark.request_return()
    assert shown["n"] == 0  # the dead window is never re-shown
    assert quit_called["n"] == 1  # quit instead of orphaning


def test_app_run_sets_no_quit_on_last_window_closed(qtapp, monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    import autoptz.ui.app as app_mod

    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    monkeypatch.setattr(app_mod, "_build_main_window", lambda *a, **k: _Win())
    app_mod.run([])
    assert QApplication.instance().quitOnLastWindowClosed() is False


class _Win:
    def show(self) -> None: ...
    def isMinimized(self) -> bool:
        return False

    def showNormal(self) -> None: ...
    def raise_(self) -> None: ...
    def activateWindow(self) -> None: ...

    def statusBar(self):  # noqa: ANN201
        class _S:
            def showMessage(self, *a, **k) -> None: ...

        return _S()
