"""MarkControlPanel + MarkDetailsPanel (offscreen): simplified controls + details.

The control panel is trimmed to the demo essentials: a live verdict/progress
line, a Stop button, and an "Exit Mark…" button.  The redundant Source/Cameras
re-ask (set in the pre-flight) is gone — there is no Start button (the ramp
auto-starts) and no source radios / camera spinbox.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def test_control_panel_reports_verdict(qtapp) -> None:
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    p.set_verdict("sustaining 4 cams @ 28.3 fps")
    assert "sustaining 4 cams" in p._verdict_label.text()
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


def test_control_panel_stop_enabled_only_while_running(qtapp) -> None:
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    assert not p._stop_btn.isEnabled()  # idle
    p.set_running(True)
    assert p._stop_btn.isEnabled()
    p.set_running(False)
    assert not p._stop_btn.isEnabled()
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


def test_control_panel_has_no_source_or_camera_controls(qtapp) -> None:
    from PySide6.QtWidgets import QRadioButton, QSpinBox
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    # The redundant pre-flight re-ask (Synthetic/NDI radios + camera spinbox) is gone.
    assert not p.findChildren(QSpinBox)
    assert not p.findChildren(QRadioButton)
    assert not hasattr(p, "_start_btn")
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
