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


def test_chart_object_name_and_card_ancestor(qtapp) -> None:
    from PySide6.QtWidgets import QFrame

    win = _win(qtapp)
    try:
        assert win._chart.objectName() == "markChart"
        # The chart is wrapped in a #chartCard QFrame (titled "PERFORMANCE RAMP").
        card = win._chart.parent()
        assert isinstance(card, QFrame)
        assert card.objectName() == "chartCard"
    finally:
        win.deleteLater()


def test_central_layout_spacing(qtapp) -> None:
    from PySide6.QtWidgets import QVBoxLayout

    win = _win(qtapp)
    try:
        layout = win.centralWidget().layout()
        assert isinstance(layout, QVBoxLayout)
        assert layout.spacing() == 12
    finally:
        win.deleteLater()


def test_theme_toggle_repaints_hud(qtapp) -> None:
    """Area 1: a Light/Dark flip on the isolated client repaints the HUD widgets.

    The chart/control/details bake T.CURRENT colors at paint time; without a
    listener a View→Appearance flip left them stale.  A ThemeController bound to
    the SAME isolated client flips T.CURRENT, and the wired _restyle slots refresh
    the control verdict + details idle colors.
    """
    from autoptz.ui import theme as T
    from autoptz.ui.theme import ThemeController

    win = _win(qtapp)
    # Bind a controller to the ISOLATED client so setTheme actually flips T.CURRENT.
    ThemeController(qtapp, win._engine.client)
    try:
        win._engine.client.setTheme("dark")
        qtapp.processEvents()
        dark_text = T.CURRENT.text
        dark_subtext = T.CURRENT.subtext

        win._engine.client.setTheme("light")
        qtapp.processEvents()
        assert T.CURRENT.text != dark_text  # T.CURRENT actually flipped
        light_text = T.CURRENT.text
        light_subtext = T.CURRENT.subtext
        assert light_text.lower() in win._controls._verdict_label.styleSheet().lower()
        assert light_subtext.lower() in win._details._empty.styleSheet().lower()

        # Toggle back to dark and re-assert the literals followed.
        win._engine.client.setTheme("dark")
        qtapp.processEvents()
        assert T.CURRENT.text == dark_text
        assert dark_text.lower() in win._controls._verdict_label.styleSheet().lower()
        assert dark_subtext.lower() in win._details._empty.styleSheet().lower()
    finally:
        win.deleteLater()


def test_panel_restyle_idempotent(qtapp) -> None:
    win = _win(qtapp)
    try:
        # Calling _restyle repeatedly must not raise.
        win._controls._restyle()
        win._controls._restyle()
        win._details._restyle()
        win._details._restyle()
    finally:
        win.deleteLater()


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
    win._prompt_on_completion = False  # no modal: persist directly
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
    text = win._controls._verdict_label.text()
    # P3: the finished verdict is the rating word + score math (e.g. "Good — 2 cam …").
    assert text.startswith("Good")
    assert "2 cam" in text
    assert "= 2.00" in text
    win.deleteLater()


def test_finish_verdict_includes_rating_word(qtapp) -> None:
    """P3: the finished verdict leads with the human rating word and carries the
    transparent score math whose numbers equal the displayed score."""
    win = _win(qtapp)
    win._prompt_on_completion = False  # no modal: persist directly
    result = _finish_result(score=2.0, sustained_cameras=2, min_fps_at_sustained=28.0)
    win._on_finished(result)
    text = win._controls._verdict_label.text()
    assert text.startswith("Good")  # 2.0 is in the Good band
    assert "28/30 fps" in text
    assert "1.0 weight" in text
    assert "= 2.00" in text
    win.deleteLater()


def test_finish_verdict_is_highlighted_final(qtapp) -> None:
    """P3: the finished verdict is shown as the FINAL (highlighted) verdict — accent +
    bold — not the plain in-flight progress styling."""
    from autoptz.ui import theme as T

    win = _win(qtapp)
    win._prompt_on_completion = False
    win._on_finished(_finish_result())
    sheet = win._controls._verdict_label.styleSheet()
    assert T.TRACKING in sheet
    assert "font-weight: 700" in sheet
    win.deleteLater()


def test_finish_verdict_shows_min_fps_over_target(qtapp) -> None:
    """The rating math shows the sustained min fps over the 30 fps target divisor, not
    the runner's internal DISCOUNTED pass floor (result.floor_fps) — that threshold
    stays out of the user-facing copy."""
    win = _win(qtapp, floor_fps=30.0)
    win._prompt_on_completion = False  # no modal: persist directly
    # result.floor_fps is the discounted pass floor (25.5) the runner ran with — the
    # verdict must NOT print that internal threshold.
    result = BenchmarkResult(
        profile="full",
        weight=1.0,
        floor_fps=25.5,
        max_cameras=3,
        sustained_cameras=2,
        min_fps_at_sustained=28.0,
        score=2.0,
        steps=[],
    )
    win._on_finished(result)
    text = win._controls._verdict_label.text()
    assert "28/30 fps" in text  # sustained min fps over the 30 fps target
    assert "25" not in text  # NOT the discounted pass floor
    win.deleteLater()


def test_chart_floor_matches_discounted_pass_threshold(qtapp) -> None:
    """Change C: the chart's pass line is the SAME discounted floor used to color the
    steps (session target × ratio), so green dots sit above the line and red below —
    not the raw target (which would put passing-but-capped steps below the line)."""
    from autoptz.ui.mark_runner import MarkRampController

    win = _win(qtapp, floor_fps=30.0)
    win._prompt_on_completion = False  # no modal: persist directly
    expected = 30.0 * MarkRampController._MARK_SUSTAIN_RATIO
    # Idle floor (set at construction / _refresh_idle_status).
    assert win._chart._floor == expected
    # After a step.
    win._on_step(
        StepResult(cameras=1, min_fps=26.0, mean_fps=26.0, per_camera_fps=[26.0], sustained=True)
    )
    assert win._chart._floor == expected
    # After finish.
    result = BenchmarkResult(
        profile="full",
        weight=1.0,
        floor_fps=expected,
        max_cameras=3,
        sustained_cameras=1,
        min_fps_at_sustained=26.0,
        score=1.0,
        steps=[
            StepResult(
                cameras=1, min_fps=26.0, mean_fps=26.0, per_camera_fps=[26.0], sustained=True
            )
        ],
    )
    win._on_finished(result)
    assert win._chart._floor == expected
    win.deleteLater()


def _finish_result(**kw) -> BenchmarkResult:
    base = {
        "profile": "full",
        "weight": 1.0,
        "floor_fps": 24.0,
        "max_cameras": 3,
        "sustained_cameras": 2,
        "min_fps_at_sustained": 28.0,
        "score": 2.0,
        "steps": [],
    }
    base.update(kw)
    return BenchmarkResult(**base)


class _FakeCompletionDialog:
    """A non-modal stand-in for MarkCompletionDialog.

    Records constructor kwargs, exposes the real ``saveRequested`` / ``openRequested``
    signals (so the window's connections fire), and lets a test script the post-``exec``
    ``choice()`` plus a callback run *inside* ``exec`` (to emit save/open).  No real
    modal is ever entered.
    """

    instances: list[_FakeCompletionDialog] = []

    def __init__(self, *, verdict, rating="", reason="", has_saved=False, parent=None) -> None:
        from PySide6.QtCore import QObject, Signal

        self.kwargs = {
            "verdict": verdict,
            "rating": rating,
            "reason": reason,
            "has_saved": has_saved,
        }
        self._choice: object | None = None
        self.saved_state = has_saved
        self.on_exec = None  # optional callable invoked inside exec()

        class _Signals(QObject):
            saveRequested = Signal()
            openRequested = Signal()

        self._sig = _Signals()
        self.saveRequested = self._sig.saveRequested
        self.openRequested = self._sig.openRequested
        _FakeCompletionDialog.instances.append(self)

    def set_saved(self, saved: bool) -> None:
        self.saved_state = bool(saved)

    def exec(self):
        from PySide6.QtWidgets import QDialog

        if self.on_exec is not None:
            self.on_exec(self)
        return QDialog.DialogCode.Accepted

    def choice(self):
        return self._choice

    def deleteLater(self) -> None:  # noqa: N802 — Qt parity
        pass


@pytest.fixture
def _patch_completion(monkeypatch):
    """Patch MarkCompletionDialog (imported lazily in the window) with the fake."""
    import autoptz.ui.widgets.dialogs.mark_completion as mc

    _FakeCompletionDialog.instances = []
    monkeypatch.setattr(mc, "MarkCompletionDialog", _FakeCompletionDialog)
    return _FakeCompletionDialog


def test_completion_modal_shows_verdict_and_score(qtapp, _patch_completion) -> None:
    """On completion the window builds the modal with the verdict/score string (NEVER
    a real modal — MarkCompletionDialog is patched to a fake recording its kwargs)."""
    win = _win(qtapp)
    try:
        win._on_finished(_finish_result())
        assert len(_patch_completion.instances) == 1
        verdict = _patch_completion.instances[0].kwargs["verdict"]
        # P3 verdict is the rating reason: "Good — 2 cam × 28/30 fps × 1.0 weight = 2.00".
        assert "2 cam" in verdict
        assert "= 2.00" in verdict
    finally:
        win.deleteLater()


def test_completion_modal_shows_rating_word(qtapp, _patch_completion) -> None:
    """P3: the modal is built with the human rating word and the EXPANDED score math
    (the two-line reason carrying the numeric substitution that yields the score)."""
    win = _win(qtapp)
    try:
        win._on_finished(_finish_result(score=2.0))
        dlg = _patch_completion.instances[0]
        assert dlg.kwargs["rating"] in {"Needs work", "Fair", "Good", "Great", "Excellent"}
        assert dlg.kwargs["rating"] == "Good"  # 2.0 → Good
        reason = dlg.kwargs["reason"]
        assert "score = cameras" in reason  # the named formula line
        assert "= 2.00" in reason  # the numeric substitution result
    finally:
        win.deleteLater()


def test_completion_modal_return_button(qtapp, _patch_completion) -> None:
    """RETURN from the modal emits returnToAppRequested and sets _returning."""
    from autoptz.ui.widgets.dialogs.mark_completion import RETURN

    win = _win(qtapp)
    seen: list[int] = []
    win.returnToAppRequested.connect(lambda: seen.append(1))

    import autoptz.ui.widgets.dialogs.mark_completion as mc

    def _build(*a, **k):
        dlg = _FakeCompletionDialog(*a, **k)
        dlg.on_exec = lambda d: setattr(d, "_choice", RETURN)
        return dlg

    orig = mc.MarkCompletionDialog
    mc.MarkCompletionDialog = _build  # type: ignore[attr-defined]
    try:
        win._on_finished(_finish_result())
        assert seen == [1]
        assert win._returning is True
    finally:
        mc.MarkCompletionDialog = orig  # type: ignore[attr-defined]
        win.deleteLater()


def test_completion_modal_quit_button(qtapp, _patch_completion) -> None:
    """QUIT from the modal emits quitRequested."""
    from autoptz.ui.widgets.dialogs.mark_completion import QUIT

    win = _win(qtapp)
    seen: list[int] = []
    win.quitRequested.connect(lambda: seen.append(1))

    import autoptz.ui.widgets.dialogs.mark_completion as mc

    def _build(*a, **k):
        dlg = _FakeCompletionDialog(*a, **k)
        dlg.on_exec = lambda d: setattr(d, "_choice", QUIT)
        return dlg

    orig = mc.MarkCompletionDialog
    mc.MarkCompletionDialog = _build  # type: ignore[attr-defined]
    try:
        win._on_finished(_finish_result())
        assert seen == [1]
    finally:
        mc.MarkCompletionDialog = orig  # type: ignore[attr-defined]
        win.deleteLater()


def test_completion_modal_dismissed_keeps_window_open(qtapp, _patch_completion) -> None:
    """Dismissing the modal with no choice keeps the result on the window (for the
    Exit button's save-on-exit) and does NOT auto-persist or quit."""
    win = _win(qtapp)
    persisted: list[object] = []
    monkeypatch_persist = win._persist_result
    win._persist_result = lambda r: persisted.append(r)  # type: ignore[assignment]
    seen: list[int] = []
    win.returnToAppRequested.connect(lambda: seen.append(1))
    win.quitRequested.connect(lambda: seen.append(1))
    try:
        result = _finish_result()
        win._on_finished(result)  # fake exec() returns Accepted with choice None
        assert win._result is not None
        assert win._result.score == result.score
        assert persisted == []  # dismiss must NOT auto-persist
        assert seen == []  # no return / quit emitted
    finally:
        win._persist_result = monkeypatch_persist  # type: ignore[assignment]
        win.deleteLater()


def test_completion_modal_save_then_open(qtapp, monkeypatch, tmp_path, _patch_completion) -> None:
    """The modal's Save then Open round-trip: saveRequested writes the chosen file (via
    the patched getSaveFileName) and sets _benchmarks_dir; openRequested reveals it via
    QDesktopServices.openUrl (patched)."""
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QFileDialog

    win = _win(qtapp)
    target = tmp_path / "from-modal.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), ""))
    )
    opened: list[object] = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url) or True)
    )

    def _exec_save_then_open(dlg):
        dlg.saveRequested.emit()
        dlg.openRequested.emit()

    import autoptz.ui.widgets.dialogs.mark_completion as mc

    def _build(*a, **k):
        dlg = _FakeCompletionDialog(*a, **k)
        dlg.on_exec = _exec_save_then_open
        return dlg

    orig = mc.MarkCompletionDialog
    mc.MarkCompletionDialog = _build  # type: ignore[attr-defined]
    try:
        win._on_finished(_finish_result())
        assert target.exists()  # save wrote the chosen path
        assert win._benchmarks_dir == target.parent
        assert opened  # open revealed the saved dir
        assert str(target.parent) in opened[0].toLocalFile()
    finally:
        mc.MarkCompletionDialog = orig  # type: ignore[attr-defined]
        win.deleteLater()


def test_completion_modal_save_cancelled_keeps_result(
    qtapp, monkeypatch, _patch_completion
) -> None:
    """Cancelling the Save-As ("","") writes no file but the result stays on the window
    for the Exit button's save-on-exit (no auto-persist)."""
    from PySide6.QtWidgets import QFileDialog

    win = _win(qtapp)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    persisted: list[object] = []
    monkeypatch.setattr(win, "_persist_result", lambda r: persisted.append(r))

    import autoptz.ui.widgets.dialogs.mark_completion as mc

    def _build(*a, **k):
        dlg = _FakeCompletionDialog(*a, **k)
        dlg.on_exec = lambda d: d.saveRequested.emit()
        return dlg

    orig = mc.MarkCompletionDialog
    mc.MarkCompletionDialog = _build  # type: ignore[attr-defined]
    try:
        result = _finish_result()
        win._on_finished(result)
        assert win._result is not None
        assert win._result.score == result.score
        assert persisted == []  # save cancelled → no auto-persist
    finally:
        mc.MarkCompletionDialog = orig  # type: ignore[attr-defined]
        win.deleteLater()


def test_completion_modal_prompt_disabled_no_dialog(qtapp, _patch_completion) -> None:
    """With _prompt_on_completion False the window does NOT open the modal — it persists
    the result directly (the headless / test path)."""
    win = _win(qtapp)
    win._prompt_on_completion = False
    persisted: list[object] = []
    win._persist_result = lambda r: persisted.append(r)  # type: ignore[assignment]
    try:
        result = _finish_result()
        win._on_finished(result)
        assert _patch_completion.instances == []  # no modal built
        assert len(persisted) == 1
        assert persisted[0].score == result.score
        assert persisted[0].scene_clip_id  # the enriched copy carries the scene id
    finally:
        win.deleteLater()


def test_completion_modal_reentrant_guard(qtapp, _patch_completion) -> None:
    """A re-entrant _show_completion_modal (a second finish while the modal is up) must
    build the dialog only once."""
    win = _win(qtapp)

    def _reenter(dlg):
        # Re-enter while "inside" the modal — the guard must swallow it.
        win._show_completion_modal(_finish_result())

    import autoptz.ui.widgets.dialogs.mark_completion as mc

    def _build(*a, **k):
        d = _FakeCompletionDialog(*a, **k)
        d.on_exec = _reenter
        return d

    orig = mc.MarkCompletionDialog
    mc.MarkCompletionDialog = _build  # type: ignore[attr-defined]
    try:
        win._on_finished(_finish_result())
        assert len(_FakeCompletionDialog.instances) == 1  # nested call guarded out
    finally:
        mc.MarkCompletionDialog = orig  # type: ignore[attr-defined]
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


# ── Slice 5: feed quality metrics from live telemetry ─────────────────────────


def _telemetry(camera_id: str, *, fps: float = 30.0, target: bool = True):
    """A minimal TelemetryMsg carrying one (optionally target) track for a camera."""
    from autoptz.engine.runtime.messages import BBox, TelemetryMsg, TrackInfo

    return TelemetryMsg(
        camera_id=camera_id,
        seq=1,
        fps=fps,
        tracks=[
            TrackInfo(
                track_id=7,
                bbox=BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
                is_target=target,
                confidence=0.9,
            )
        ],
    )


def test_registers_a_telemetry_observer_on_the_client(qtapp) -> None:
    """The Mark window registers exactly one telemetry observer on its isolated
    client so live telemetry feeds the quality accumulators."""
    win = _win(qtapp)
    try:
        assert callable(win._engine.client.add_telemetry_observer)
        # An observer is registered (the list is non-empty after construction).
        assert len(win._engine.client._telemetry_observers) >= 1
    finally:
        win.deleteLater()


def test_telemetry_then_step_carries_per_camera_quality(qtapp) -> None:
    """Feeding telemetry through the client, then completing a ramp step, enriches
    the StepResult with per_camera_quality before it reaches the chart."""
    win = _win(qtapp)
    try:
        cid = win._engine.client.cameraModel.camera_ids()[0]
        for _ in range(5):
            win._engine.client.push_telemetry(_telemetry(cid))
        step = StepResult(
            cameras=1, min_fps=30.0, mean_fps=30.0, per_camera_fps=[30.0], sustained=True
        )
        win._on_step(step)
        charted = win._chart._steps[-1]
        assert charted.per_camera_quality  # non-empty
        assert cid in charted.per_camera_quality
        assert charted.per_camera_quality[cid]["target_hold_pct"] == 100.0
    finally:
        win.deleteLater()


def test_accumulators_reset_between_steps_no_carryover(qtapp) -> None:
    """Each step's quality reflects only that step's telemetry — the accumulators
    are reset after every _on_step so frames don't carry over."""
    win = _win(qtapp)
    try:
        cid = win._engine.client.cameraModel.camera_ids()[0]
        # Step 1: two held frames.
        win._engine.client.push_telemetry(_telemetry(cid))
        win._engine.client.push_telemetry(_telemetry(cid))
        win._on_step(
            StepResult(
                cameras=1, min_fps=30.0, mean_fps=30.0, per_camera_fps=[30.0], sustained=True
            )
        )
        q1 = win._chart._steps[-1].per_camera_quality[cid]
        # Step 2: a single NON-target frame → hold pct must be 0 (no carryover from
        # step 1's 100%).
        win._engine.client.push_telemetry(_telemetry(cid, target=False))
        win._on_step(
            StepResult(
                cameras=1, min_fps=29.0, mean_fps=29.0, per_camera_fps=[29.0], sustained=True
            )
        )
        q2 = win._chart._steps[-1].per_camera_quality[cid]
        assert q1["target_hold_pct"] == 100.0
        assert q2["target_hold_pct"] == 0.0  # fresh accumulator, no carryover
    finally:
        win.deleteLater()


def test_bad_observer_never_breaks_telemetry(qtapp) -> None:
    """A raising observer is swallowed so telemetry delivery never breaks."""
    win = _win(qtapp)
    try:
        boom = {"n": 0}

        def _bad(_msg):
            boom["n"] += 1
            raise RuntimeError("observer boom")

        win._engine.client.add_telemetry_observer(_bad)
        cid = win._engine.client.cameraModel.camera_ids()[0]
        # Must not raise despite the bad observer.
        win._engine.client.push_telemetry(_telemetry(cid))
        assert boom["n"] == 1
    finally:
        win.deleteLater()


# ── Slice 7b: CSV option in the Save-As (now invoked from the completion modal) ──
#
# The completion modal owns the Return/Quit/Open/Save choice; the actual Save-As +
# writer selection lives in the window's extracted ``_save_via_dialog`` helper.  These
# tests pin the writer/filter routing by driving that helper directly (the modal's
# ``saveRequested`` calls it; see test_completion_modal_save_then_open for the wiring).


def test_save_via_dialog_csv_filter_writes_csv(qtapp, monkeypatch, tmp_path) -> None:
    """Choosing the CSV filter routes the save through save_mark_result_csv and
    writes a .csv file."""
    from PySide6.QtWidgets import QFileDialog

    win = _win(qtapp)
    target = tmp_path / "chosen-mark.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "CSV (*.csv)")),
    )
    import autoptz.benchmark.results as results_mod

    calls: list[object] = []
    real_csv = results_mod.save_mark_result_csv

    def _spy_csv(results, path, **kw):
        calls.append(results)
        return real_csv(results, path, **kw)

    monkeypatch.setattr(results_mod, "save_mark_result_csv", _spy_csv)
    try:
        win._save_via_dialog(
            _finish_result(
                steps=[
                    StepResult(
                        cameras=1,
                        min_fps=30.0,
                        mean_fps=30.0,
                        per_camera_fps=[30.0],
                        sustained=True,
                    )
                ]
            )
        )
        assert len(calls) == 1  # routed through the CSV writer
        assert target.exists()
        text = target.read_text()
        assert "profile" in text  # the CSV header
    finally:
        win.deleteLater()


def test_save_via_dialog_json_filter_writes_json(qtapp, monkeypatch, tmp_path) -> None:
    """Choosing the JSON filter routes the save through the JSON writer."""
    from PySide6.QtWidgets import QFileDialog

    win = _win(qtapp)
    target = tmp_path / "chosen-mark.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "JSON (*.json)")),
    )
    try:
        win._save_via_dialog(_finish_result())
        assert target.exists()
        import json as _json

        data = _json.loads(target.read_text())
        assert data["results"][0]["profile"] == "full"
    finally:
        win.deleteLater()


def test_save_via_dialog_csv_by_suffix(qtapp, monkeypatch, tmp_path) -> None:
    """A .csv path suffix selects the CSV writer even under the All-Files filter."""
    from PySide6.QtWidgets import QFileDialog

    win = _win(qtapp)
    target = tmp_path / "by-suffix.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "All Files (*)")),
    )
    try:
        win._save_via_dialog(
            _finish_result(
                steps=[
                    StepResult(
                        cameras=1,
                        min_fps=30.0,
                        mean_fps=30.0,
                        per_camera_fps=[30.0],
                        sustained=True,
                    )
                ]
            )
        )
        assert target.exists()
        text = target.read_text()
        assert text.splitlines()[0].startswith("created_at")  # CSV header
    finally:
        win.deleteLater()


# ── Slice 5: ground-truth comparison (env-gated + drawn-scene only) ───────────


def _win_source(qtapp, source: str, **kw):
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.mark_window import MarkWindow

    return MarkWindow(
        session=MarkSession(source=source, max_cameras=3, dwell_s=0.0, **kw),
        frame_source=ShmFrameSource(),
    )


def test_gt_inactive_without_env(qtapp, monkeypatch) -> None:
    """Without AUTOPTZ_MARK_GT the comparator map is None even on the drawn scene."""
    monkeypatch.delenv("AUTOPTZ_MARK_GT", raising=False)
    win = _win_source(qtapp, "synthetic")
    try:
        assert win._gt is None
    finally:
        win.deleteLater()


def test_gt_active_on_drawn_scene_with_env(qtapp, monkeypatch) -> None:
    """AUTOPTZ_MARK_GT + a synthetic (drawn) scene activates the comparator map."""
    monkeypatch.setenv("AUTOPTZ_MARK_GT", "1")
    win = _win_source(qtapp, "synthetic")
    try:
        assert win._gt is not None
    finally:
        win.deleteLater()


def test_gt_finalized_on_finish(qtapp, monkeypatch) -> None:
    """With GT active, telemetry feeds the comparator and _on_finished aggregates a
    per-camera CLEAR-MOT summary onto the window."""
    from autoptz.engine.runtime.messages import BBox, GroundTruthPerson, TelemetryMsg, TrackInfo

    monkeypatch.setenv("AUTOPTZ_MARK_GT", "1")
    win = _win_source(qtapp, "synthetic")
    # No modal save dialog on finish: persist directly (stubbed) so the test never
    # blocks on the real QFileDialog and never writes to the config dir.
    win._prompt_on_completion = False
    monkeypatch.setattr(win, "_persist_result", lambda r: None)
    try:
        cid = win._engine.client.cameraModel.camera_ids()[0]
        # A track that overlaps the ground-truth person exactly → a clean match.
        box = BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
        msg = TelemetryMsg(
            camera_id=cid,
            seq=1,
            tracks=[TrackInfo(track_id=1, bbox=box, is_target=True)],
            ground_truth=[GroundTruthPerson(person_id=1, bbox=box)],
        )
        win._engine.client.push_telemetry(msg)
        win._on_finished(_finish_result())
        assert cid in win._gt_summary
        summary = win._gt_summary[cid]
        assert summary["miss_rate"] == 0.0  # the gt person was matched
        assert summary["mota"] == 1.0
    finally:
        win.deleteLater()


# ── PHASE 1: idempotent, single-source teardown + strict close ordering ───────


class _FakeController:
    """A fake MarkRampController recording stop/wait into a shared order list."""

    def __init__(self, order: list[str], *, wait_ok: bool = True) -> None:
        self._order = order
        self._wait_ok = wait_ok
        self.stop_calls = 0
        self.wait_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1
        self._order.append("controller.stop")

    def wait(self, _timeout_ms: int = 5000) -> bool:
        self.wait_calls += 1
        self._order.append("controller.wait")
        return self._wait_ok


def test_teardown_controller_idempotent(qtapp) -> None:
    """Calling _teardown_controller twice must stop/wait the live controller at most
    once and never wait after the ref is cleared (no double-teardown race)."""
    win = _win(qtapp)
    try:
        order: list[str] = []
        ctrl = _FakeController(order)
        win._controller = ctrl
        win._teardown_controller()
        assert win._controller is None
        # A second call is a clean no-op (the ref is already gone).
        win._teardown_controller()
        assert ctrl.stop_calls == 1
        assert ctrl.wait_calls == 1
    finally:
        win.deleteLater()


def test_close_waits_controller_before_engine_stop(qtapp, monkeypatch) -> None:
    """closeEvent must stop the pump, wait the controller, THEN stop the engine —
    so engine.tick() can't fire mid-wait and the engine is down after the join."""
    win = _win(qtapp)
    order: list[str] = []
    win._controller = _FakeController(order)
    monkeypatch.setattr(win._pump, "stop", lambda: order.append("pump.stop"))
    monkeypatch.setattr(win._engine, "stop", lambda: order.append("engine.stop"))
    win.close()
    # pump → controller(stop+wait) → engine, strictly ordered.
    assert order.index("pump.stop") < order.index("controller.wait")
    assert order.index("controller.wait") < order.index("engine.stop")


def test_close_logs_warning_on_controller_wait_timeout(qtapp, monkeypatch, caplog) -> None:
    """A controller whose wait() returns False (didn't finish in time) must log a
    WARNING on teardown rather than failing silently."""
    import logging as _logging

    win = _win(qtapp)
    order: list[str] = []
    win._controller = _FakeController(order, wait_ok=False)
    with caplog.at_level(_logging.WARNING, logger="autoptz.ui.widgets.mark_window"):
        win.close()
    assert any(
        record.levelno == _logging.WARNING and "controller" in record.message.lower()
        for record in caplog.records
    )


def test_close_event_teardown_order(qtapp, monkeypatch) -> None:
    """The full close sequence records exactly: pump → controller → engine → log."""
    win = _win(qtapp)
    order: list[str] = []
    win._controller = _FakeController(order)
    monkeypatch.setattr(win._pump, "stop", lambda: order.append("pump"))
    monkeypatch.setattr(win._engine, "stop", lambda: order.append("engine"))
    monkeypatch.setattr(win, "_detach_log_handler", lambda: order.append("log"))
    win.close()
    # Drop the controller.stop entry (it precedes controller.wait); reduce to phases.
    phases = [p for p in order if p in ("pump", "controller.wait", "engine", "log")]
    assert phases == ["pump", "controller.wait", "engine", "log"]


def test_close_event_is_reentrant_safe(qtapp, monkeypatch) -> None:
    """Calling close() twice must stop the engine exactly once and emit
    closedUnexpectedly exactly once (the re-entrancy guard)."""
    win = _win(qtapp)
    stops = {"n": 0}
    monkeypatch.setattr(win._engine, "stop", lambda: stops.__setitem__("n", stops["n"] + 1))
    seen: list[str] = []
    win.closedUnexpectedly.connect(lambda: seen.append("closed"))
    win.close()
    win.close()
    assert stops["n"] == 1
    assert seen == ["closed"]


def test_closed_unexpectedly_emitted_exactly_once(qtapp) -> None:
    """An OS-close (no deliberate return/quit) emits closedUnexpectedly exactly once
    even across repeated close() calls."""
    win = _win(qtapp)
    count = {"n": 0}
    win.closedUnexpectedly.connect(lambda: count.__setitem__("n", count["n"] + 1))
    win.close()
    win.close()
    assert count["n"] == 1


def test_request_exit_cancel_does_not_set_returning(qtapp, monkeypatch) -> None:
    """Cancelling the exit dialog must leave _returning False so a later OS-close
    still routes through Return (emits closedUnexpectedly)."""
    import autoptz.ui.widgets.dialogs.mark_exit as me

    win = _win(qtapp)

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
    assert win._returning is False
    win.close()


# ── PHASE 1 (Step 1.7): Cmd+Q bypass — aboutToQuit cleanup hook ───────────────


def test_cmd_q_while_mark_visible_tears_down_engine(qtapp, monkeypatch) -> None:
    """Cmd+Q routes through QApplication.aboutToQuit, bypassing closeEvent.  The
    Mark window's aboutToQuit slot must still tear down the engine + detach the log
    handler, but must NOT emit closedUnexpectedly (the app is already quitting)."""
    win = _win(qtapp)
    stops = {"n": 0}
    detached = {"n": 0}
    monkeypatch.setattr(win._engine, "stop", lambda: stops.__setitem__("n", stops["n"] + 1))
    monkeypatch.setattr(
        win, "_detach_log_handler", lambda: detached.__setitem__("n", detached["n"] + 1)
    )
    seen: list[str] = []
    win.closedUnexpectedly.connect(lambda: seen.append("closed"))
    win._on_app_about_to_quit()
    assert stops["n"] == 1
    assert detached["n"] == 1
    assert seen == []  # the app is quitting; do NOT emit closedUnexpectedly


def test_app_about_to_quit_is_idempotent_with_close(qtapp, monkeypatch) -> None:
    """close() then the aboutToQuit slot (or vice versa) must stop the engine only
    once — they share the _torn_down guard."""
    win = _win(qtapp)
    stops = {"n": 0}
    monkeypatch.setattr(win._engine, "stop", lambda: stops.__setitem__("n", stops["n"] + 1))
    win.close()
    win._on_app_about_to_quit()
    assert stops["n"] == 1


def test_cmd_q_does_not_leak_log_handler(qtapp) -> None:
    """After aboutToQuit, the Mark session's root-logger handler must be removed so
    it never leaks into the resumed/next app."""
    import logging as _logging

    win = _win(qtapp)
    handler = win._log_handler
    assert handler in _logging.getLogger().handlers
    win._on_app_about_to_quit()
    assert handler not in _logging.getLogger().handlers
    assert win._log_handler is None
