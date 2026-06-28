"""MainWindow startup guard (offscreen): the optional model-setup prompt must
never open a modal in headless/offscreen mode.

Regression guard for the CI re-entrancy crash: the startup prompt opened
``_open_model_manager(startup_prompt=True).exec()`` — a nested modal loop — which
let other test windows' pending ``singleShot`` startup timers fire inside it,
cascading modal dialogs (macOS 'Bus error' / Windows 60s timeout).

The guard is exercised on a lightweight stand-in rather than a real
``MainWindow`` on purpose: constructing a full window pulls in heavy native state
that faults late in the single-process full-suite CI run.  The offscreen check is
the FIRST statement of the method, so it returns before touching any instance
attribute — a ``SimpleNamespace`` is a sufficient ``self``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_startup_prompt_noop_when_offscreen(qtapp, monkeypatch) -> None:
    # Sanity: the suite runs under the offscreen platform.
    assert qtapp.platformName() == "offscreen"

    import autoptz.ui.widgets.main_window as mw

    # Force the pre-flight gates open so ONLY the offscreen guard can stop us:
    # missing models present and the reminder not suppressed.
    monkeypatch.setattr(mw, "model_setup_reminder_suppressed", lambda *a, **k: False)
    monkeypatch.setattr(mw, "startup_missing_model_keys", lambda: ["detector"])

    opened = {"n": 0}
    monkeypatch.setattr(
        mw.MainWindow,
        "_open_model_manager",
        lambda self, **k: opened.__setitem__("n", 1),
    )

    # Real method, lightweight ``self`` — no heavy MainWindow construction.
    fake = SimpleNamespace(_shown_optional_setup_prompt=False, _client=object())
    mw.MainWindow._maybe_show_model_setup_on_startup(fake)

    # Offscreen guard returned early — the modal manager was NOT opened.
    assert opened["n"] == 0
