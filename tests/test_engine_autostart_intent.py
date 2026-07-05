"""Engine auto-start persists the user's INTENT, not the momentary running state.

Root cause of "services are always stopped when I open the program": the persist
logic wrote engineRunning (False whenever the engine wasn't running for ANY
reason), and a persisted-False skipped auto-start — so a stopped engine trapped
itself off across launches. The intent only clears on a deliberate user Stop.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _client():
    from autoptz.ui.engine_client import EngineClient

    return EngineClient()


def test_autostart_defaults_true(qtapp) -> None:
    assert _client().autostartDesired is True


def test_never_started_engine_keeps_autostart_true(qtapp) -> None:
    """The trap: an engine that never started must NOT record 'don't auto-start'."""
    c = _client()
    assert c.engineRunning is False  # never started (no supervisor)
    assert c.autostartDesired is True  # → next launch still auto-starts


def test_user_stop_clears_autostart(qtapp) -> None:
    c = _client()
    c.userStopEngine()
    assert c.autostartDesired is False


def test_user_start_sets_autostart(qtapp) -> None:
    c = _client()
    c.userStopEngine()
    c.userStartEngine()  # may not actually run (no supervisor), but intent is set
    assert c.autostartDesired is True


def test_system_stop_does_not_change_intent(qtapp) -> None:
    """stopEngine (shutdown / Mark suspend / restart) must not touch the intent."""
    c = _client()
    c.set_autostart_desired(True)
    c.stopEngine()
    assert c.autostartDesired is True


def test_seed_from_persisted(qtapp) -> None:
    c = _client()
    c.set_autostart_desired(False)  # a deliberate stop persisted last session
    assert c.autostartDesired is False
