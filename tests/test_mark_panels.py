"""MarkControlPanel + MarkDetailsPanel (offscreen): signals, NDI gating, verdict, empty state."""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_control_panel_emits_start_and_reports_verdict(qtapp) -> None:
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    seen = []
    p.startClicked.connect(lambda: seen.append("start"))
    p._start_btn.click()
    assert seen == ["start"]
    p.set_verdict("sustaining 4 cams @ 28.3 fps")
    assert "sustaining 4 cams" in p._verdict_label.text()
    p.set_running(True)
    assert not p._start_btn.isEnabled() and p._stop_btn.isEnabled()
    p.deleteLater()


def test_control_panel_emits_stop(qtapp) -> None:
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    seen = []
    p.stopClicked.connect(lambda: seen.append("stop"))
    p.set_running(True)
    p._stop_btn.click()
    assert seen == ["stop"]
    p.deleteLater()


def test_control_panel_exit_button_stays_enabled_while_running(qtapp) -> None:
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    seen = []
    p.exitClicked.connect(lambda: seen.append("exit"))
    p.set_running(True)
    # The exit affordance must remain usable mid-run so the user can always leave.
    assert p._exit_btn.isEnabled()
    p._exit_btn.click()
    assert seen == ["exit"]
    p.deleteLater()


def test_control_panel_set_max_cameras_caps_spin(qtapp) -> None:
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    p.set_max_cameras(4)
    # Seeds the value AND caps the maximum so the ramp can't exceed the pre-added
    # wall (one source of truth for the camera count).
    assert p.selected_max_cameras() == 4
    assert p._spin.maximum() == 4
    p._spin.setValue(99)  # clamped to the new maximum
    assert p.selected_max_cameras() == 4
    p.deleteLater()


def test_control_panel_reports_selected_source_and_count(qtapp) -> None:
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    assert p.selected_source() == "synthetic"
    p._spin.setValue(5)
    assert p.selected_max_cameras() == 5
    p.deleteLater()


def test_control_panel_gates_ndi_when_unavailable(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.mark_control_panel as mod

    monkeypatch.setattr(mod, "ndi_sim_available", lambda: False)
    p = mod.MarkControlPanel()
    assert not p._ndi_radio.isEnabled()
    p.deleteLater()


def test_details_panel_empty_then_camera(qtapp) -> None:
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.widgets.mark_details_panel import MarkDetailsPanel

    client = EngineClient()
    d = MarkDetailsPanel(client)
    assert "Select a camera" in _text(d)
    d.set_camera("nope")  # unknown id: must not raise
    d.refresh()
    d.deleteLater()


def test_details_panel_shows_stats_for_known_camera(qtapp) -> None:
    from autoptz.benchmark.runner import _add_synthetic_camera
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.widgets.mark_details_panel import MarkDetailsPanel

    client = EngineClient()
    cid = _add_synthetic_camera(client, 0)
    d = MarkDetailsPanel(client)
    d.set_camera(cid)
    # The stat rows are shown (empty hint hidden) and source reads "synthetic".
    assert not d._empty.isVisibleTo(d)
    assert d._vals["source"].isVisibleTo(d)
    assert d._vals["source"].text() == "synthetic"
    d.deleteLater()


def _text(widget) -> str:
    from PySide6.QtWidgets import QLabel

    return " ".join(lbl.text() for lbl in widget.findChildren(QLabel))
