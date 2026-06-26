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
    # 3DMark-style progressive wall: starts at ONE camera and grows as the ramp
    # advances (was: all N pre-added blank up front).
    ids = win._engine.client.cameraModel.camera_ids()
    assert len(ids) == 1
    rec = win._engine.client.cameraModel.get_record(ids[0])
    assert rec.camera_config.source.type == "synthetic"
    win.deleteLater()


def test_wall_bound_to_isolated_client(qtapp) -> None:
    win = _win(qtapp)
    # The inherited CameraWall renders the isolated client, not a main one.
    assert win._client is win._engine.client
    win.deleteLater()


def test_wall_binds_factory_frame_source_in_production(qtapp) -> None:
    """The blank-tile fix: with no explicit frame_source (the production path),
    the wall MUST bind to the Mark engine's own ShmFrameSource — not the main
    app's (which is attached to the MAIN engine's shm, so tiles stayed blank)."""
    from autoptz.ui.widgets.mark_window import MarkWindow

    win = MarkWindow(session=MarkSession(max_cameras=3, dwell_s=0.0))  # frame_source=None
    try:
        assert win._frames is win._engine.frame_source
        assert win._wall._frames is win._engine.frame_source
    finally:
        win.close()


def test_camera_engine_menus_hidden_view_and_help_kept(qtapp) -> None:
    from PySide6.QtWidgets import QMenu

    win = _win(qtapp)
    titles = [m.title().replace("&", "") for m in win.menuBar().findChildren(QMenu)]
    # Basic shell menus stay (View + Appearance/theme; Help with About).
    assert any("Help" in t for t in titles)
    assert any("View" in t for t in titles)
    assert any("Appearance" in t for t in titles)
    # Camera/engine management menus are gone in the demo.
    assert not any(t in ("Engine", "Cameras") for t in titles)
    win.deleteLater()


def test_help_menu_has_about_mark(qtapp) -> None:
    from PySide6.QtWidgets import QMenu

    win = _win(qtapp)
    # Collect leaf action texts across the whole menu bar.
    leaf = [a.text() for m in win.menuBar().findChildren(QMenu) for a in m.actions() if a.text()]
    assert any("About AutoPTZ Mark" in t for t in leaf)
    win.deleteLater()


def test_usb_polling_disabled(qtapp) -> None:
    win = _win(qtapp)
    assert win._should_poll_usb() is False
    assert win._usb_poll_timer is None
    win.deleteLater()


def test_tile_context_menu_disabled_in_mark(qtapp) -> None:
    # The Mark wall must NOT offer the right-click tile menu (no removing cameras /
    # retargeting the throwaway engine from the demo).
    win = _win(qtapp)
    assert win._wall._context_menu_enabled is False
    win.deleteLater()


def test_no_autostart_env_skips_ramp_on_show(qtapp) -> None:
    # With AUTOPTZ_MARK_NO_AUTOSTART set (the autouse fixture), showing the window
    # must NOT auto-start the ramp — the controller stays None.
    win = _win(qtapp)
    try:
        win.show()
        qtapp.processEvents()
        assert win._controller is None
    finally:
        win.close()


def test_show_autostarts_ramp(qtapp, monkeypatch) -> None:
    # Showing the window auto-starts the ramp (start_run is invoked) — the user no
    # longer clicks Start.  Built with the no-autostart fixture so the REAL engine
    # never spins up; we then clear the flag and drive showEvent to exercise the
    # auto-start path with start_run stubbed.
    win = _win(qtapp)  # constructed under AUTOPTZ_MARK_NO_AUTOSTART → engine idle
    started = {"n": 0}
    monkeypatch.setattr(win, "start_run", lambda: started.__setitem__("n", started["n"] + 1))
    monkeypatch.delenv("AUTOPTZ_MARK_NO_AUTOSTART", raising=False)
    try:
        win.show()
        qtapp.processEvents()  # the auto-start is deferred one event-loop turn
        assert started["n"] == 1
        # One-shot: showing again does not re-trigger the ramp.
        win.show()
        qtapp.processEvents()
        assert started["n"] == 1
    finally:
        win.close()


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


# ── Finding #2: embedded LogsPanel streams the isolated Mark logs ──────────────


def test_logs_dock_hosts_real_logs_panel(qtapp) -> None:
    """The Logs dock holds a real LogsPanel bound to the Mark log model — NOT the
    'Logs unavailable' QLabel placeholder (log_model=None used to produce that)."""
    from autoptz.ui.widgets.logs_panel import LogsPanel

    win = _win(qtapp)
    try:
        logs_dock = win._docks.get("logs")
        assert logs_dock is not None
        assert logs_dock.isVisible() or not logs_dock.isHidden()
        panel = logs_dock.widget()
        assert isinstance(panel, LogsPanel)
    finally:
        win.close()


def test_logs_stream_into_mark_model(qtapp) -> None:
    """A log record routed through the root logger lands in the Mark log model."""
    import logging as _logging

    win = _win(qtapp)
    try:
        before = len(win._log_model.rows())
        # WARNING so the record passes regardless of the root logger's level in the
        # test process (the real app raises the root to INFO; tests don't).
        _logging.getLogger("autoptz.engine.supervisor").warning("mark-test-log-line")
        qtapp.processEvents()
        rows = win._log_model.rows()
        assert len(rows) > before
        assert any("mark-test-log-line" in r["message"] for r in rows)
    finally:
        win.close()


def test_log_handler_detached_on_close(qtapp) -> None:
    """Mark's root-logger handler must be removed on exit so it never leaks into
    the resumed main app (which would write into the discarded Mark model)."""
    import logging as _logging

    win = _win(qtapp)
    handler = win._log_handler
    assert handler in _logging.getLogger().handlers
    win.close()
    assert handler not in _logging.getLogger().handlers
    assert win._log_handler is None


# ── Finding #3: single engine stack + reconciled camera count ─────────────────


def test_session_is_single_source_of_camera_count(qtapp) -> None:
    """The session's max_cameras is the single source of truth for the ramp cap.

    The control panel no longer re-asks the count (the pre-flight set it), so the
    engine cap is read straight from the session — not a panel spinbox."""
    win = _win(qtapp)  # _win builds MarkSession(max_cameras=3)
    try:
        assert win._session.max_cameras == 3
        assert win._engine.max_cameras == 3
    finally:
        win.close()


def test_engine_exposes_preadded_camera_ids(qtapp) -> None:
    win = _win(qtapp)
    try:
        # The wall starts at ONE camera (progressive ramp) and ``camera_ids``
        # mirrors the model; add_next_camera grows it up to the session max.
        ids = win._engine.camera_ids
        assert len(ids) == 1
        assert set(ids) == set(win._engine.client.cameraModel.camera_ids())
    finally:
        win.close()


def test_sample_factory_adopts_engine_no_new_cameras(qtapp) -> None:
    """The ramp's sampler ADOPTS the factory's supervisor + pre-added cameras — it
    must NOT build a second supervisor or add duplicate cameras (the double-stack
    bug: two Supervisors + doubled tiles)."""
    win = _win(qtapp)
    try:
        before = list(win._engine.client.cameraModel.camera_ids())
        factory = win._build_sample_factory()
        assert factory is not None  # always adopts now (was None for synthetic)
        sample_fn = factory()
        sampler = sample_fn._sampler
        # Adopts the SAME supervisor the window owns (no second one built).
        assert sampler._sup is win._engine.supervisor
        assert sampler._adopted is True
        # Pre-seeded with the factory's camera ids → ramping never re-adds them.
        assert sampler._cameras == before
        # Closing the sampler must NOT tear down the adopted (window-owned) engine.
        sampler.close()
        assert list(win._engine.client.cameraModel.camera_ids()) == before
    finally:
        win.close()


# ── Finding #1: visible Return / Quit exit affordance with optional save ──────


def test_control_panel_has_exit_button(qtapp) -> None:
    win = _win(qtapp)
    try:
        assert hasattr(win._controls, "_exit_btn")
        assert win._controls._exit_btn.isEnabled()
    finally:
        win.close()


def test_request_exit_return_emits_return(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.dialogs.mark_exit as me

    win = _win(qtapp)
    seen: list[str] = []
    win.returnToAppRequested.connect(lambda: seen.append("return"))
    win.closedUnexpectedly.connect(lambda: seen.append("closed"))

    class _Dlg:
        def __init__(self, *a, **k) -> None: ...
        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted

        def choice(self):
            return me.RETURN

        def save_results(self):
            return False

    monkeypatch.setattr(me, "MarkExitDialog", _Dlg)
    win._request_exit()
    assert seen == ["return"]  # deliberate return must not also fire closedUnexpectedly
    win.close()


def test_request_exit_quit_emits_quit(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.dialogs.mark_exit as me

    win = _win(qtapp)
    seen: list[str] = []
    win.quitRequested.connect(lambda: seen.append("quit"))

    class _Dlg:
        def __init__(self, *a, **k) -> None: ...
        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted

        def choice(self):
            return me.QUIT

        def save_results(self):
            return False

    monkeypatch.setattr(me, "MarkExitDialog", _Dlg)
    win._request_exit()
    assert seen == ["quit"]
    win.close()


def test_request_exit_cancel_stays(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.dialogs.mark_exit as me

    win = _win(qtapp)
    seen: list[str] = []
    win.returnToAppRequested.connect(lambda: seen.append("return"))
    win.quitRequested.connect(lambda: seen.append("quit"))

    class _Dlg:
        def __init__(self, *a, **k) -> None: ...
        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Rejected

        def choice(self):
            return None

        def save_results(self):
            return False

    monkeypatch.setattr(me, "MarkExitDialog", _Dlg)
    win._request_exit()
    assert seen == []  # cancel → stay in Mark, no exit signal
    win.close()


def test_request_exit_save_persists_result(qtapp, monkeypatch) -> None:
    import autoptz.ui.widgets.dialogs.mark_exit as me

    win = _win(qtapp)
    win._result = BenchmarkResult(
        profile="full",
        weight=1.0,
        floor_fps=24.0,
        max_cameras=3,
        sustained_cameras=2,
        min_fps_at_sustained=28.0,
        score=2.0,
        steps=[],
    )
    saved: list[object] = []
    monkeypatch.setattr(win, "_persist_result", lambda r: saved.append(r))

    class _Dlg:
        def __init__(self, *a, **k) -> None: ...
        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted

        def choice(self):
            return me.RETURN

        def save_results(self):
            return True

    monkeypatch.setattr(me, "MarkExitDialog", _Dlg)
    win._request_exit()
    assert saved == [win._result]  # save box checked → result persisted on exit
    win.close()
