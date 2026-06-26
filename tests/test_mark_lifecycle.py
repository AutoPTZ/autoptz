"""In-process Mark lifecycle (offscreen): confirm→suspend→swap; return resumes; quit exits; no relaunch."""

from __future__ import annotations

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
