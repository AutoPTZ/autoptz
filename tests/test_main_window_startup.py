"""MainWindow startup guard (offscreen): the optional model-setup prompt must
never open a modal in headless/offscreen mode.

Regression guard for the CI re-entrancy crash: the startup prompt opened
``_open_model_manager(startup_prompt=True).exec()`` — a nested modal loop — which
let other test windows' pending ``singleShot`` startup timers fire inside it,
cascading modal dialogs (macOS 'Bus error' / Windows 60s timeout).
"""

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


def test_startup_prompt_noop_when_offscreen(qtapp, monkeypatch) -> None:
    # Sanity: the suite runs under the offscreen platform.
    assert qtapp.platformName() == "offscreen"

    import autoptz.ui.widgets.main_window as mw

    # Force the pre-flight gates open so only the offscreen guard can stop us:
    # missing models present and the reminder not suppressed.
    monkeypatch.setattr(mw, "model_setup_reminder_suppressed", lambda *a, **k: False)
    monkeypatch.setattr(mw, "startup_missing_model_keys", lambda: ["detector"])

    opened = {"n": 0}
    monkeypatch.setattr(
        mw.MainWindow,
        "_open_model_manager",
        lambda self, **k: opened.__setitem__("n", 1),
    )

    win = _main(qtapp)
    win._maybe_show_model_setup_on_startup()

    # Offscreen guard returned early — the modal manager was NOT opened.
    assert opened["n"] == 0
    win.deleteLater()
