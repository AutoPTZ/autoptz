"""MarkWindow (offscreen): subclasses MainWindow, isolated engine, menus hidden, no relaunch."""

from __future__ import annotations

import pytest

from autoptz.benchmark.runner import BenchmarkResult, StepResult
from autoptz.ui.mark_session import MarkSession


@pytest.fixture(autouse=True)
def _no_autostart(monkeypatch):
    # Construct the window offscreen WITHOUT spinning up the real Supervisor
    # (model loading + staged camera open); these tests poke state / drive fakes.
    monkeypatch.setenv("AUTOPTZ_MARK_NO_AUTOSTART", "1")


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _win(qtapp, **kw):
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.mark_window import MarkWindow

    return MarkWindow(
        session=MarkSession(max_cameras=3, dwell_s=0.0, **kw),
        frame_source=ShmFrameSource(),
    )


def test_title_is_plain_and_versionless(qtapp) -> None:
    win = _win(qtapp)
    assert win.windowTitle() == "AutoPTZ Mark"  # version lives in About, not title
    win.deleteLater()


def test_subclasses_main_window(qtapp) -> None:
    from autoptz.ui.widgets.main_window import MainWindow

    win = _win(qtapp)
    assert isinstance(win, MainWindow)
    win.deleteLater()


def test_isolated_engine_only_fake_cameras(qtapp) -> None:
    from autoptz.config.store import default_db_path

    win = _win(qtapp)
    # Mark's client is NOT the default store; only fake cameras present.
    assert str(win._engine.store._path) != str(default_db_path())
    assert len(win._engine.client.cameraModel.camera_ids()) == 3
    win.deleteLater()


def test_wall_bound_to_isolated_client(qtapp) -> None:
    win = _win(qtapp)
    # The inherited CameraWall renders the isolated client, not a main one.
    assert win._client is win._engine.client
    win.deleteLater()


def test_menus_hidden_except_about(qtapp) -> None:
    from PySide6.QtWidgets import QMenu

    win = _win(qtapp)
    titles = [m.title().replace("&", "") for m in win.menuBar().findChildren(QMenu)]
    assert any("Help" in t for t in titles)
    assert not any(t in ("Engine", "Cameras", "View") for t in titles)
    win.deleteLater()


def test_usb_polling_disabled(qtapp) -> None:
    win = _win(qtapp)
    assert win._should_poll_usb() is False
    assert win._usb_poll_timer is None
    win.deleteLater()


def test_return_signal_and_no_relaunch_on_close(qtapp, monkeypatch) -> None:
    import autoptz.ui.mark_session as ms

    called = {"relaunch": 0}
    monkeypatch.setattr(
        ms, "relaunch", lambda **k: called.__setitem__("relaunch", called["relaunch"] + 1)
    )
    win = _win(qtapp)
    seen = []
    win.closedUnexpectedly.connect(lambda: seen.append("closed"))
    win.close()
    assert called["relaunch"] == 0  # never relaunches
    assert seen == ["closed"]


def test_request_return_emits_and_suppresses_unexpected(qtapp) -> None:
    win = _win(qtapp)
    seen = []
    win.returnToAppRequested.connect(lambda: seen.append("return"))
    win.closedUnexpectedly.connect(lambda: seen.append("closed"))
    win.request_return()
    win.close()
    assert seen == ["return"]  # a deliberate return must NOT also fire closedUnexpectedly


def test_step_updates_chart_and_verdict(qtapp) -> None:
    win = _win(qtapp)
    win._on_step(
        StepResult(
            cameras=2, min_fps=28.3, mean_fps=30.0, per_camera_fps=[28.3, 28.3], sustained=True
        )
    )
    assert len(win._chart._steps) == 1
    assert "2" in win._controls._verdict_label.text()
    win.deleteLater()


def test_finish_sets_verdict(qtapp) -> None:
    win = _win(qtapp)
    result = BenchmarkResult(
        profile="full",
        weight=1.0,
        floor_fps=24.0,
        max_cameras=3,
        sustained_cameras=2,
        min_fps_at_sustained=28.0,
        score=2.0,
        steps=[],
    )
    win._on_finished(result)
    assert "2" in win._controls._verdict_label.text()
    win.deleteLater()


def test_error_cancelled_reads_as_stopped(qtapp) -> None:
    win = _win(qtapp)
    win._on_error("cancelled")
    verdict = win._controls._verdict_label.text().lower()
    assert "stop" in verdict
    win.deleteLater()


def test_chart_kept_from_old_implementation(qtapp) -> None:
    from autoptz.ui.widgets.mark_window import _MarkRampChart

    chart = _MarkRampChart()
    chart.set_steps(
        [StepResult(cameras=1, min_fps=40.0, mean_fps=40.0, per_camera_fps=[40.0], sustained=True)],
        floor=24.0,
    )
    chart.resize(200, 120)
    chart.repaint()
    chart.deleteLater()
