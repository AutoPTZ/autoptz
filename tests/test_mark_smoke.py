"""End-to-end AutoPTZ Mark smoke (headless, offscreen) — the IN-PROCESS swap.

Exercises the real lifecycle without a subprocess relaunch or real inference:
construct a MainWindow, suspend it into an isolated MarkWindow via
``_enter_mark_mode`` (the same path Help → Run AutoPTZ Mark… takes), assert the
Mark window owns a FULLY isolated engine (temp-file store ≠ the main client's,
only fake cameras), then ``request_return`` to resume the main window.

``AUTOPTZ_MARK_NO_AUTOSTART`` keeps the Mark window from spinning up the real
Supervisor (model loading / staged camera open) so the smoke stays fast and
display-free.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_autostart(monkeypatch):
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


def test_in_process_swap_isolated_engine_then_return(qtapp) -> None:
    from autoptz.config.store import default_db_path
    from autoptz.ui.mark_session import MarkSession

    win = _main(qtapp)
    win._enter_mark_mode(MarkSession(max_cameras=2, dwell_s=0.0))

    mark = win._mark_window
    assert mark is not None
    assert mark.windowTitle() == "AutoPTZ Mark"

    # Fully isolated: Mark's client/store are NOT the main app's, and only fake
    # cameras are registered.
    assert mark._client is not win._client
    assert str(mark._engine.store._path) != str(default_db_path())
    # Progressive wall: starts at ONE synthetic camera and grows as the ramp runs.
    ids = mark._engine.client.cameraModel.camera_ids()
    assert len(ids) == 1
    for cid in ids:
        rec = mark._engine.client.cameraModel.get_record(cid)
        assert rec.camera_config.source.type == "synthetic"

    # Return-to-AutoPTZ resumes the main window and tears down the Mark window.
    mark.request_return()
    assert win._mark_window is None
    assert not win.isHidden()
