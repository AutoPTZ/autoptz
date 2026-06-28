"""ExperimentalFeaturesDialog (offscreen): Apply gives visible feedback and then
offers an application restart — but only when something actually changed, and
never as a blocking modal in tests (the restart prompt + relaunch are injectable).

Runs in its own process (CI shards per file) so it builds a real ``QApplication``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _client(settings: dict[str, Any] | None = None) -> SimpleNamespace:
    store: dict[str, Any] = dict(settings or {})

    def get_setting(key: str, default: Any = None) -> Any:
        return store.get(key, default)

    def set_setting(key: str, value: Any) -> None:
        store[key] = value

    return SimpleNamespace(getSetting=get_setting, setSetting=set_setting, _store=store)


def _dialog(client: SimpleNamespace):
    """Build the dialog with restart prompt + relaunch stubbed (no modal, no exec)."""
    from autoptz.ui.widgets.dialogs.experimental import ExperimentalFeaturesDialog

    dlg = ExperimentalFeaturesDialog(client)
    calls: dict[str, int] = {"confirm": 0, "restart": 0}
    dlg._confirm_restart = lambda: (calls.__setitem__("confirm", calls["confirm"] + 1), False)[1]
    dlg._do_restart = lambda: calls.__setitem__("restart", calls["restart"] + 1)
    return dlg, calls


def test_apply_persists_selection(qtapp) -> None:
    client = _client()
    dlg, _ = _dialog(client)
    dlg._bool_boxes["AUTOPTZ_UNIFIED_POSE"].setChecked(True)
    dlg._on_apply()
    saved = client.getSetting("experimental_features", {})
    assert saved.get("AUTOPTZ_UNIFIED_POSE") == "1"
    dlg.close()


def test_apply_shows_visible_feedback(qtapp) -> None:
    """The Apply button must visibly acknowledge the click ("Applied")."""
    client = _client()
    dlg, _ = _dialog(client)
    assert dlg._apply_btn.text() == "Apply"
    dlg._bool_boxes["AUTOPTZ_PTZ_PUMP"].setChecked(True)
    dlg._on_apply()
    assert "Applied" in dlg._apply_btn.text()
    dlg.close()


def test_apply_offers_restart_when_a_flag_changed(qtapp) -> None:
    client = _client()
    dlg, calls = _dialog(client)
    # Make the (stubbed) prompt say "yes, restart".
    dlg._confirm_restart = lambda: (calls.__setitem__("confirm", calls["confirm"] + 1), True)[1]
    dlg._bool_boxes["AUTOPTZ_PTZ_PUMP"].setChecked(True)
    dlg._on_apply()
    assert calls["confirm"] == 1, "restart was not offered after a real change"
    assert calls["restart"] == 1, "confirming restart did not trigger the relaunch"
    dlg.close()


def test_apply_with_no_change_does_not_nag_for_restart(qtapp) -> None:
    """Clicking Apply without touching anything must not prompt for a restart."""
    client = _client()
    dlg, calls = _dialog(client)
    dlg._on_apply()
    assert calls["confirm"] == 0
    assert calls["restart"] == 0
    dlg.close()


def test_second_apply_without_new_change_does_not_reoffer(qtapp) -> None:
    """After applying a change, applying again with no further change is quiet."""
    client = _client()
    dlg, calls = _dialog(client)
    dlg._bool_boxes["AUTOPTZ_UNIFIED_POSE"].setChecked(True)
    dlg._on_apply()
    first = calls["confirm"]
    dlg._on_apply()  # nothing changed since the last apply
    assert calls["confirm"] == first
    dlg.close()


def test_apply_button_click_runs_the_apply_flow(qtapp) -> None:
    """The real Apply button must be wired to the new flow (feedback + persist)."""
    client = _client()
    dlg, _ = _dialog(client)
    dlg._bool_boxes["AUTOPTZ_UNIFIED_POSE"].setChecked(True)
    dlg._apply_btn.click()
    assert client.getSetting("experimental_features", {}).get("AUTOPTZ_UNIFIED_POSE") == "1"
    assert "Applied" in dlg._apply_btn.text()
    dlg.close()


def test_legacy_apply_still_just_persists(qtapp) -> None:
    """``_apply`` stays a pure persist (no modal) so existing callers/tests hold."""
    client = _client()
    dlg, calls = _dialog(client)
    dlg._bool_boxes["AUTOPTZ_PTZ_PUMP"].setChecked(True)
    dlg._apply()
    assert client.getSetting("experimental_features", {}).get("AUTOPTZ_PTZ_PUMP") == "1"
    assert calls["confirm"] == 0  # pure persist never offers a restart
    dlg.close()
