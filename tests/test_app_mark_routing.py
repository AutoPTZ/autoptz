"""app.run(mode=...) routing: mark mode builds MarkWindow, normal builds MainWindow.

Offscreen + monkeypatched: the real event loop is stubbed (``QApplication.exec``
returns 0), window construction is patched to a fake so no engine / Supervisor /
relaunch ever runs.
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


def test_run_mark_mode_builds_mark_window(monkeypatch) -> None:
    import autoptz.ui.app as app_mod

    built: dict[str, bool] = {}

    def _fake_mark(*a, **k):
        built["mark"] = True
        return _FakeWin()

    def _fake_main(*a, **k):
        built["main"] = True
        return _FakeWin()

    monkeypatch.setattr(app_mod, "_build_mark_window", _fake_mark, raising=True)
    monkeypatch.setattr(app_mod, "_build_main_window", _fake_main, raising=True)
    _stub_event_loop(monkeypatch)

    code = app_mod.run(argv=["autoptz"], mode="mark")
    assert code == 0
    assert built.get("mark") is True
    assert "main" not in built


def test_run_normal_mode_builds_main_window(monkeypatch) -> None:
    import autoptz.ui.app as app_mod

    built: dict[str, bool] = {}

    def _fake_mark(*a, **k):
        built["mark"] = True
        return _FakeWin()

    def _fake_main(*a, **k):
        built["main"] = True
        return _FakeWin()

    monkeypatch.setattr(app_mod, "_build_mark_window", _fake_mark, raising=True)
    monkeypatch.setattr(app_mod, "_build_main_window", _fake_main, raising=True)
    _stub_event_loop(monkeypatch)

    code = app_mod.run(argv=["autoptz"], mode="normal")
    assert code == 0
    assert built.get("main") is True
    assert "mark" not in built
