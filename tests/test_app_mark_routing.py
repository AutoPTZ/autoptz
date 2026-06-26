"""app.run() always builds MainWindow; the Mark window is reached in-process.

The subprocess relaunch model is gone: ``run()`` no longer routes ``mode="mark"``
to a ``MarkWindow``.  Instead it always builds the normal :class:`MainWindow` and
sets ``setQuitOnLastWindowClosed(False)`` so the in-process Help → Run AutoPTZ
Mark… swap (suspend MainWindow / show MarkWindow) never quits the app.

Offscreen + monkeypatched: the real event loop is stubbed (``QApplication.exec``
returns 0) and window construction is patched to a fake so no engine / Supervisor
ever runs.
"""

from __future__ import annotations


class _FakeWin:
    def __init__(self, *a, **k) -> None:
        pass

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


def _stub_event_loop(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "exec", lambda self: 0)


def test_run_always_builds_main_window_and_disables_quit_on_close(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication

    import autoptz.ui.app as app_mod

    built: dict[str, bool] = {}
    monkeypatch.setattr(
        app_mod,
        "_build_main_window",
        lambda *a, **k: built.setdefault("win", _FakeWin()) or built["win"],
    )
    _stub_event_loop(monkeypatch)

    code = app_mod.run(argv=["autoptz"])
    assert code == 0
    assert "win" in built
    # The linchpin of the in-process Mark swap.
    assert QApplication.instance().quitOnLastWindowClosed() is False


def test_mark_mode_is_a_deprecated_noop_that_builds_main_window(monkeypatch) -> None:
    # mode="mark" is accepted (so --mark doesn't crash) but builds the NORMAL
    # window — the dedicated MarkWindow is now reached in-process from the Help menu.
    import autoptz.ui.app as app_mod

    built: dict[str, bool] = {}
    monkeypatch.setattr(
        app_mod,
        "_build_main_window",
        lambda *a, **k: built.setdefault("main", _FakeWin()) or built["main"],
    )
    _stub_event_loop(monkeypatch)

    code = app_mod.run(argv=["autoptz"], mode="mark")
    assert code == 0
    assert "main" in built  # mark mode no longer builds a separate window


def test_run_has_no_build_mark_window_helper() -> None:
    # The subprocess Mark-window builder is removed entirely.
    import autoptz.ui.app as app_mod

    assert not hasattr(app_mod, "_build_mark_window")
