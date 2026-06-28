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


def test_verdict_default_is_not_final(qtapp) -> None:
    """The live verdict (default final=False) keeps the plain theme color/weight —
    no accent highlight until the run actually finishes."""
    from autoptz.ui import theme as T
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    p.set_verdict("Ramping… step 1 of 3")
    sheet = p._verdict_label.styleSheet()
    assert T.TRACKING not in sheet  # no accent while ramping
    assert "font-weight: 700" not in sheet  # not bolded to the final weight
    p.deleteLater()


def test_verdict_final_uses_accent_styling(qtapp) -> None:
    """The final verdict (final=True) is highlighted with the accent color + bold so
    the finished score stands out from the in-flight progress line."""
    from autoptz.ui import theme as T
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    p.set_verdict("Good — 2 cam × 28/30 fps × 1.0 weight = 1.87", final=True)
    sheet = p._verdict_label.styleSheet()
    assert T.TRACKING in sheet  # accent color
    assert "font-weight: 700" in sheet  # bold
    assert "Good" in p._verdict_label.text()
    p.deleteLater()


def test_verdict_final_survives_restyle(qtapp) -> None:
    """A theme flip (re-running _restyle) must re-apply the final accent — not revert
    the finished verdict back to the plain in-flight styling."""
    from autoptz.ui import theme as T
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    p.set_verdict("Great — 4 cam × 30/30 fps × 0.8 weight = 3.20", final=True)
    p._restyle()  # simulate a Light/Dark flip
    sheet = p._verdict_label.styleSheet()
    assert T.TRACKING in sheet
    assert "font-weight: 700" in sheet
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


def test_control_panel_object_names(qtapp) -> None:
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    assert p.objectName() == "markControlPanel"
    assert p._verdict_label.objectName() == "markVerdict"
    assert p._stop_btn.objectName() == "markStopBtn"
    assert p._exit_btn.objectName() == "markExitBtn"
    p.deleteLater()


def test_control_panel_verdict_has_no_inline_stylesheet(qtapp) -> None:
    # The verdict label is styled by the global #markVerdict rule + the _restyle
    # fallback — it must NOT carry a hard-coded inline color/size that ignores the
    # theme on construction.
    from autoptz.ui.widgets.mark_control_panel import MarkControlPanel

    p = MarkControlPanel()
    # _restyle sets a color so a theme flip can repaint it; that's expected. What
    # we forbid is the old verbatim T.CURRENT.text inline set baked in __init__.
    assert hasattr(p, "_restyle")
    p.deleteLater()


def test_details_panel_has_header_and_idle_hides_rows(qtapp) -> None:
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.widgets.mark_details_panel import MarkDetailsPanel

    client = EngineClient()
    d = MarkDetailsPanel(client)
    assert d.objectName() == "markDetailsPanel"
    # The "CAMERA DETAILS" header is always visible (even idle).
    assert d._header.objectName() == "detailsHeader"
    assert "CAMERA DETAILS" in d._header.text()
    # Idle: friendlier hint shown, value rows hidden.
    assert "No camera selected" in d._empty.text()
    for v in d._vals.values():
        assert not v.isVisibleTo(d)
    d.deleteLater()


def test_details_panel_empty_then_camera(qtapp) -> None:
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.widgets.mark_details_panel import MarkDetailsPanel

    client = EngineClient()
    d = MarkDetailsPanel(client)
    assert "No camera selected" in _text(d)
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
