"""MarkWindow — the polished, self-contained AutoPTZ Mark benchmark window.

``MarkWindow`` **subclasses** :class:`MainWindow` so it inherits the camera wall,
tiles, theme, dock layout, and status bar — but it owns a fully **isolated**
engine stack via :class:`MarkEngineFactory` (its own temp-file ``ConfigStore`` +
``EngineClient`` + ``Supervisor``, populated with ONLY fake synthetic / NDI
cameras).  Sharing the main app's client/store was the original bug: real
cameras appeared in the Mark wall and closing Mark killed the whole app.

The window trims the inherited shell down to the benchmark essentials:
  * title is JUST ``"AutoPTZ Mark"`` (version lives in :class:`AboutMarkDialog`);
  * ``_build_menus`` is overridden to a single Help → About entry;
  * the Properties / People / Services right docks are hidden (details + logs stay);
  * a HUD ramp chart + :class:`MarkControlPanel` + :class:`MarkDetailsPanel` are
    injected around the inherited wall;
  * ``_should_poll_usb`` returns ``False`` so the inherited USB poll never probes
    the fake client.

It runs a 33 ms GUI ``QTimer`` pump calling ``engine.tick()`` (mirroring
``app.py``) and drives the ramp via :class:`MarkRampController` on the isolated
client.  On close it stops the pump FIRST, then the controller, then the engine,
and emits a return/exit signal — it NEVER relaunches a subprocess.

Test seams (offscreen): ``_wall_type`` (the embedded wall class), the controller
slots ``_on_progress`` / ``_on_step`` / ``_on_finished`` / ``_on_error`` can be
driven with fake ``StepResult`` / ``BenchmarkResult`` values, and the verdict is
read off ``self._controls``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QMetaObject, Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QColor, QDesktopServices, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from autoptz.benchmark.profiles import get_profile
from autoptz.benchmark.runner import BenchmarkResult, StepResult
from autoptz.ui import theme as T
from autoptz.ui.mark_session import MarkSession
from autoptz.ui.widgets.camera_wall import CameraWall
from autoptz.ui.widgets.common import on_theme_changed
from autoptz.ui.widgets.main_window import MainWindow, _action, _safe
from autoptz.ui.widgets.mark_control_panel import MarkControlPanel
from autoptz.ui.widgets.mark_details_panel import MarkDetailsPanel

log = logging.getLogger(__name__)


def _no_autostart() -> bool:
    """True when ``AUTOPTZ_MARK_NO_AUTOSTART`` is set (offscreen tests).

    Tests construct the window to poke private state / drive fake signals; this
    flag keeps both the real Supervisor (model loading + staged camera open) AND
    the auto-started ramp from spinning up.
    """
    import os

    return os.environ.get("AUTOPTZ_MARK_NO_AUTOSTART", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class _MarkRampChart(QWidget):
    """A small QPainter line chart: x = camera count, y = min fps per step.

    Each step's marker is green when it held the floor and red when it failed; a
    dashed horizontal line marks the floor.  Painting tolerates an empty / single
    step list (draws just the axes + floor).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("markChart")
        self._steps: list[StepResult] = []
        self._floor: float = 24.0
        self.setMinimumHeight(140)

    def set_steps(self, steps: list[StepResult], floor: float) -> None:
        self._steps = list(steps)
        self._floor = float(floor)
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 — Qt override
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            w = self.width()
            h = self.height()
            pad = 10
            x0, y0 = pad, pad
            x1, y1 = max(pad + 1, w - pad), max(pad + 1, h - pad)

            painter.fillRect(self.rect(), QColor(T.CURRENT.surface))

            # Vertical scale: 0 .. max(observed fps, floor) * 1.15 headroom.
            observed = [s.min_fps for s in self._steps] + [self._floor]
            top_fps = max(observed) * 1.15 if observed else self._floor * 1.15
            top_fps = max(top_fps, 1.0)

            def fps_y(fps: float) -> float:
                frac = min(1.0, max(0.0, fps / top_fps))
                return y1 - frac * (y1 - y0)

            # axes
            painter.setPen(QPen(QColor(T.CURRENT.border), 1))
            painter.drawLine(x0, y0, x0, y1)
            painter.drawLine(x0, y1, x1, y1)

            # floor line (dashed amber)
            floor_pen = QPen(QColor(T.WARNING), 1, Qt.PenStyle.DashLine)
            painter.setPen(floor_pen)
            fy = fps_y(self._floor)
            painter.drawLine(x0, int(fy), x1, int(fy))

            n = len(self._steps)
            if n == 0:
                return
            span = max(1, n)

            def step_x(i: int) -> float:
                if n == 1:
                    return (x0 + x1) / 2.0
                return x0 + (i / (span - 1)) * (x1 - x0)

            # connecting line (neutral)
            painter.setPen(QPen(QColor(T.CURRENT.subtext), 1))
            prev: tuple[float, float] | None = None
            for i, s in enumerate(self._steps):
                pt = (step_x(i), fps_y(s.min_fps))
                if prev is not None:
                    painter.drawLine(int(prev[0]), int(prev[1]), int(pt[0]), int(pt[1]))
                prev = pt

            # markers (green sustained / red fail)
            for i, s in enumerate(self._steps):
                color = QColor(T.TRACKING) if s.sustained else QColor(T.ERROR)
                painter.setBrush(color)
                painter.setPen(QPen(color, 1))
                px, py = step_x(i), fps_y(s.min_fps)
                painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)
        finally:
            painter.end()


class MarkWindow(MainWindow):
    """AutoPTZ Mark: a MainWindow subclass over a fully isolated engine stack."""

    returnToAppRequested = Signal()
    quitRequested = Signal()
    closedUnexpectedly = Signal()

    def __init__(
        self,
        *,
        session: MarkSession | None = None,
        theme: Any | None = None,
        frame_source: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        from autoptz.ui.log_bridge import LogListModel, QtLogHandler
        from autoptz.ui.mark_engine import MarkEngineFactory

        self._session = session or MarkSession()
        # The isolated stack (temp-file store + own client/supervisor, fake-only
        # cameras) MUST exist before super().__init__ binds the wall to a client.
        self._engine = MarkEngineFactory(self._session)
        self._controller: Any | None = None
        self._result: BenchmarkResult | None = None
        self._benchmarks_dir: Any | None = None
        self._returning = False
        # Slice 5: fold the live telemetry stream into per-camera tracking quality.
        # ``_quality`` maps camera id -> PerCameraQualityAccumulator for the CURRENT
        # ramp step; ``_on_step`` finalizes + resets it.  When AUTOPTZ_MARK_GT is set
        # AND the scene is a synthetic/drawn capability scene, ``_gt`` additionally
        # maps camera id -> GroundTruthComparator (accumulated across the whole run
        # and finalized on _on_finished).  ``_gt`` is None when GT is inactive.
        self._quality: dict[str, Any] = {}
        self._gt: dict[str, Any] | None = {} if self._gt_active() else None
        self._gt_summary: dict[str, dict] = {}
        # Completion (Area 2): on a finished ramp, prompt a Save-As dialog.  Tests
        # flip ``_prompt_on_completion`` off to persist directly (no modal); the
        # re-entrancy guard keeps a second finish from stacking a second dialog.
        self._prompt_on_completion = True
        self._showing_completion_dialog = False
        # A dedicated log model + handler so the inherited LogsPanel streams the
        # ISOLATED Mark session's logs (was: log_model=None → the dock showed a
        # static "Logs unavailable" label).  The handler is attached to the root
        # logger here and removed in closeEvent so Mark sessions never leak handlers
        # back into the resumed main app.
        self._log_model = LogListModel()
        # Annotated optional: closeEvent detaches + clears it (idempotent).
        self._log_handler: QtLogHandler | None = QtLogHandler(self._log_model)
        self._log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._log_handler)
        # Route the isolated client's log slots (capture-level / copy / export) at
        # the Mark session's own bridge, not the main app's.
        self._engine.client.set_log_bridge(self._log_model, self._log_handler)
        # CRITICAL: bind the wall to the ISOLATED engine's frame source, never the
        # main app's.  The main app's ShmFrameSource is attached to the MAIN engine's
        # shm; the Mark synthetic workers write to the Mark engine's shm, so binding
        # to anything but ``self._engine.frame_source`` leaves every tile blank.  An
        # explicit ``frame_source`` arg (offscreen tests) overrides, but production
        # passes None → the isolated source is used.
        super().__init__(
            self._engine.client,
            log_model=self._log_model,
            frame_source=frame_source or self._engine.frame_source,
            theme=theme,
            parent=parent,
            # No right-click tile context menu in the demo: the viewer must not be
            # able to remove cameras / retarget the throwaway Mark engine.
            context_menu_enabled=False,
        )
        self.setWindowTitle("AutoPTZ Mark")  # NO version (lives in About)
        # Auto-size so the whole wall + panels fit (was a fixed 1200×780 that
        # cramped/scrolled the content); the showEvent maximizes once on first show.
        self.resize(1280, 820)
        self._autostarted_ramp = False
        # Cross-thread camera-grow handoff: the ramp runs on a worker thread but
        # adding a camera mutates the Qt model (builds the tile), which MUST happen
        # on the GUI thread.  The worker posts _grow_one_slot and waits on the event.
        self._grow_event = threading.Event()
        self._grow_result: str | None = None
        self._inject_mark_ui()
        self._hide_nonessential_docks()
        self._start_pump()
        self._refresh_idle_status()

    # ── overrides ────────────────────────────────────────────────────────────────

    def _should_poll_usb(self) -> bool:
        return False

    def _should_persist_geometry(self) -> bool:
        # Throwaway maximized window backed by a temp store that engine.stop()
        # closes on exit — don't persist its geometry (would hit the closed store).
        return False

    def _build_menus(self) -> None:  # override: View (basics) + Help (About/Exit)
        """A trimmed menu bar: basic View shell + Help.

        Mark keeps the genuinely useful shell menus — View → Appearance (theme),
        UI Scale, and the panel toggles — but HIDES the camera/engine-management
        menus (Cameras, Engine, Overlays, Layouts) the inherited
        :meth:`MainWindow._build_menus` builds, since the ramp owns the engine and
        the demo viewer must not mutate it.  Help adds About AutoPTZ Mark + a
        second path to the deliberate Return / Quit exit.
        """
        bar = self.menuBar()

        view = bar.addMenu("&View")
        self._build_appearance_menu(view)
        self._build_scale_menu(view)
        view.addSeparator()
        # Panel toggles (Properties is hidden in Mark, but Details + Logs are real).
        panels = view.addMenu("Panels")
        for key in ("camera_info", "logs"):
            dock = self._docks.get(key)
            if dock is not None:
                panels.addAction(dock.toggleViewAction())

        helpm = bar.addMenu("&Help")
        helpm.addAction(_action(self, "About AutoPTZ Mark", self._show_about_mark))
        helpm.addSeparator()
        # A second, always-visible path to the deliberate Return / Quit choice
        # (the primary one is the control panel's "Exit Mark…" button).
        helpm.addAction(_action(self, "Exit AutoPTZ Mark…", self._request_exit))

    def _build_appearance_menu(self, view: Any) -> None:
        """View → Appearance: the light/dark/system theme picker (inherited concept)."""
        appearance = view.addMenu("Appearance")
        appearance.setToolTipsVisible(True)
        group = QActionGroup(self)
        group.setExclusive(True)
        current = _safe(lambda: str(self._client.themeMode), "dark")
        tips = {
            "system": "Follow the OS light/dark setting.",
            "dark": "Dark broadcast palette (easiest on the eyes).",
            "light": "Light palette for bright rooms.",
        }
        for mode, label in (("system", "System"), ("dark", "Dark"), ("light", "Light")):
            act = QAction(label, self, checkable=True)
            act.setChecked(mode == current)
            act.setToolTip(tips[mode])
            act.triggered.connect(lambda _c, m=mode: self._client.setTheme(m))
            group.addAction(act)
            appearance.addAction(act)

    def _refresh_engine_state(self) -> None:
        # The Mark window has no Engine menu (the ramp owns the engine), so the
        # inherited handler's references to ``_act_start`` etc. don't apply.  Keep
        # the inherited startup-progress banner refresh, which is still wired.
        self._refresh_startup_progress()

    def _show_about_mark(self) -> None:
        from autoptz.ui.widgets.dialogs.about_mark import AboutMarkDialog

        AboutMarkDialog(self._engine.client, self).exec()

    # ── construction helpers ─────────────────────────────────────────────────────

    def _hide_nonessential_docks(self) -> None:
        for key in ("properties", "people", "services"):
            dock = self._docks.get(key)
            if dock is not None:
                dock.setVisible(False)

    def _inject_mark_ui(self) -> None:
        """Inject the HUD chart + controls + details around the inherited wall.

        The chart and controls go above the central CameraWall; the per-stream
        details panel rides in a right dock (replacing the inherited camera-info
        tab content for the Mark context).  Tile selection (inherited) drives the
        details panel via :meth:`_on_camera_selected`.
        """
        self._chart = _MarkRampChart()
        self._controls = MarkControlPanel()
        # The control panel no longer re-asks source/count (the pre-flight set them)
        # and has no Start button — the ramp auto-starts.  Only Stop + Exit are wired.
        self._controls.stopClicked.connect(self.stop_run)
        self._controls.exitClicked.connect(self._request_exit)

        # Wrap the chart in a titled card so it reads as a panel (the global
        # #chartCard rule paints the surface + border; #chartTitle the caption).
        chart_card = QFrame()
        chart_card.setObjectName("chartCard")
        card_col = QVBoxLayout(chart_card)
        card_col.setContentsMargins(0, 0, 0, 0)
        card_col.setSpacing(0)
        chart_title = QLabel("PERFORMANCE RAMP")
        chart_title.setObjectName("chartTitle")
        card_col.addWidget(chart_title)
        card_col.addWidget(self._chart)

        # Insert the chart card + controls at the TOP of the inherited central
        # column, above the CameraWall (index 0/1 keep the startup banner first).
        central = self.centralWidget()
        layout = central.layout() if central is not None else None
        if isinstance(layout, QVBoxLayout):
            layout.setSpacing(12)
            layout.insertWidget(1, chart_card)
            layout.insertWidget(2, self._controls)

        # Replace the right "Camera Info" dock content with the Mark details panel.
        self._details = MarkDetailsPanel(self._engine.client)
        info_dock = self._docks.get("camera_info")
        if isinstance(info_dock, QDockWidget):
            info_dock.setWidget(self._details)
            info_dock.setWindowTitle("Details")
            info_dock.setVisible(True)
            info_dock.raise_()

        # Theme reactivity (Area 1): the chart/control/details bake T.CURRENT
        # colors at paint time but have no listener, so a View→Appearance flip
        # never repaints them.  Wire each to the ISOLATED client's themeChanged so
        # toggling the theme re-runs their restyle/repaint.  Each restyle is
        # DEFERRED one event-loop turn so it reads T.CURRENT *after* the
        # ThemeController's apply() has flipped it — the themeChanged slots fire in
        # connection order, which can put the panels before the controller and
        # otherwise leave them one appearance behind.
        client = self._engine.client
        on_theme_changed(client, self._chart.update)
        on_theme_changed(client, lambda: self._defer(self._controls._restyle))
        on_theme_changed(client, lambda: self._defer(self._details._restyle))

        # Slice 5: register ONE telemetry observer so the live stream feeds the
        # per-camera quality accumulators (and, when GT is active, the comparators).
        client.add_telemetry_observer(self._on_telemetry)

    def _gt_active(self) -> bool:
        """True when ground-truth comparison should run for this Mark session.

        Requires both the ``AUTOPTZ_MARK_GT`` env flag AND a synthetic/drawn
        capability scene — the engine only populates ``TelemetryMsg.ground_truth``
        for the drawn (``"anim"``) scene, which is what a synthetic session (or a
        clip session falling back to drawn people) renders.
        """
        if os.environ.get("AUTOPTZ_MARK_GT", "").strip().lower() not in ("1", "true", "yes", "on"):
            return False
        session = self._session
        # The drawn scene is rendered for any non-clip source, or a clip session
        # whose bundled asset is missing (falls back to drawn people).
        try:
            return (not session.is_clip()) or (not session.clip_available())
        except Exception:  # noqa: BLE001 — a probe must never block Mark startup
            return False

    def _on_telemetry(self, msg: Any) -> None:
        """Fold one live ``TelemetryMsg`` into the current step's accumulators.

        Looks up (or creates) the per-camera quality accumulator and feeds it; when
        GT is active, also feeds the per-camera ground-truth comparator.  Runs on
        the GUI thread (the EngineClient marshals telemetry there before fanning out
        to observers), so no extra locking is needed.
        """
        from autoptz.benchmark.runner import PerCameraQualityAccumulator

        cid = getattr(msg, "camera_id", None)
        if cid is None:
            return
        acc = self._quality.get(cid)
        if acc is None:
            acc = PerCameraQualityAccumulator(self._session.target_fps())
            self._quality[cid] = acc
        acc.on_telemetry(msg)

        if self._gt is not None:
            from autoptz.benchmark.ground_truth import GroundTruthComparator

            comparator = self._gt.get(cid)
            if comparator is None:
                comparator = GroundTruthComparator()
                self._gt[cid] = comparator
            comparator.on_frame(
                getattr(msg, "tracks", None) or [], getattr(msg, "ground_truth", None) or []
            )

    @staticmethod
    def _defer(fn: Any) -> None:
        """Run ``fn`` next event-loop turn (after the theme apply flips T.CURRENT)."""
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, fn)

    def _on_camera_selected(self, camera_id: str) -> None:
        # Drive the Mark details panel from the inherited tile-selection slot.
        self._selected_camera = camera_id
        details = getattr(self, "_details", None)
        if details is not None:
            details.set_camera(camera_id)

    def _start_pump(self) -> None:
        from PySide6.QtCore import QTimer

        self._pump = QTimer(self)
        self._pump.setInterval(33)  # ~30 Hz
        self._pump.timeout.connect(self._engine.tick)
        self._pump.start()
        if not _no_autostart():
            self._engine.start()
            # Auto-track a (seeded) target per camera so Center Stage visibly engages
            # on the full profile; a no-op for the streams (no-inference) profile.
            self._engine.auto_track_targets(seed=0xA17)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        """Maximize the window and AUTO-START the ramp on first show.

        The Mark window opens full-screen-sized so the whole wall + panels fit
        (no cramped/scrolled content), and the benchmark ramp starts on its own —
        the user no longer has to click Start.  The auto-start is one-shot
        (``_autostarted_ramp``) and skipped under ``AUTOPTZ_MARK_NO_AUTOSTART`` so
        offscreen tests can drive the run by hand.  ``MainWindow.showEvent`` (theme
        tab restyle, geometry) still runs via ``super()``.
        """
        super().showEvent(event)
        self.showMaximized()
        if self._autostarted_ramp or _no_autostart():
            return
        self._autostarted_ramp = True
        from PySide6.QtCore import QTimer

        # Defer one turn so first paint (and the engine's staged camera open)
        # isn't blocked by spinning up the ramp controller synchronously.
        QTimer.singleShot(0, self.start_run)

    # ── run lifecycle ────────────────────────────────────────────────────────────

    def start_run(self) -> None:
        if self._controller is not None:
            return
        from autoptz.ui.mark_runner import MarkRampController

        self._result = None
        self._chart.set_steps([], self._effective_floor())
        self._controls.set_running(True)
        self._controls.set_verdict("Starting…")
        controller = MarkRampController(
            profile=self._session.profile,
            floor_fps=self._session.target_fps(),
            max_cameras=self._session.max_cameras,
            dwell_s=self._session.dwell_s,
            sample_factory=self._build_sample_factory(),
            client=self._engine.client,  # ISOLATED client the wall is bound to
        )
        controller.progress.connect(self._on_progress)
        controller.step_completed.connect(self._on_step)
        controller.finished.connect(self._on_finished)
        controller.error.connect(self._on_error)
        self._controller = controller
        controller.start()

    def stop_run(self) -> None:
        if self._controller is not None:
            self._controller.stop()
            self._controls.set_verdict("Stopping…")

    @Slot()
    def _grow_one_slot(self) -> None:
        """Add the next fake camera ON THE GUI THREAD.

        Qt model mutation (which builds the new wall tile) must run on the GUI
        thread; the ramp's worker thread posts this slot and waits on the event.
        """
        try:
            self._grow_result = self._engine.add_next_camera()
        except Exception:  # noqa: BLE001
            log.exception("Mark add_next_camera failed")
            self._grow_result = None
        finally:
            self._grow_event.set()

    def _grow_one_threadsafe(self) -> str | None:
        """``on_grow`` for the ramp sampler — marshals the camera add to the GUI thread.

        The sampler calls this from its worker thread when the ramp steps up; the
        actual ``add_next_camera`` (model insert + worker spawn + tile creation) is
        run on the GUI thread so the new tile actually appears.  Returns the new
        camera id (None at the cap or on timeout).  Direct call when already on the
        GUI thread (e.g. tests).
        """
        from PySide6.QtCore import QThread

        if QThread.currentThread() is self.thread():
            return self._engine.add_next_camera()
        self._grow_event.clear()
        self._grow_result = None
        QMetaObject.invokeMethod(self, "_grow_one_slot", Qt.ConnectionType.QueuedConnection)
        if not self._grow_event.wait(10.0):
            log.warning("Mark camera-grow timed out waiting for the GUI thread")
            return None
        return self._grow_result

    def _build_sample_factory(self) -> Any:
        """Return the controller's ``sample_factory`` — always ADOPTS the engine.

        The Mark window already owns ONE isolated engine stack (the factory's
        supervisor + the cameras pre-added to the idle wall, plus any NDI fleet).
        The ramp must drive THAT stack, not build a second supervisor over a
        disjoint camera set (which doubled tiles + CPU and broadcast duplicate NDI
        sources).  So the returned factory builds a sampler that adopts the
        existing supervisor + pre-added cameras (and, for NDI, the existing fleet);
        ``_SupervisorSampler``/``MarkNDIFleetSampler`` then sample fps without
        adding cameras or starting a second supervisor.
        """
        profile = get_profile(self._session.profile)
        dwell = self._session.dwell_s
        engine = self._engine

        if self._session.source == "ndi" and engine.ndi_fleet is not None:
            from autoptz.benchmark.ndi_sim import MarkNDIFleetSampler

            def ndi_factory() -> Any:
                sampler = MarkNDIFleetSampler(
                    profile,
                    client=engine.client,
                    supervisor=engine.supervisor,
                    fleet=engine.ndi_fleet,
                    cameras=engine.camera_ids,
                    adopted_started=engine.is_started,
                    on_grow=self._grow_one_threadsafe,
                )

                def sample_fn(n: int) -> list[float]:
                    return sampler.sample(n, dwell_s=dwell, max_ticks=2000, tick_sleep_s=0.005)

                sample_fn._sampler = sampler  # type: ignore[attr-defined]
                return sample_fn

            return ndi_factory

        from autoptz.benchmark.runner import _SupervisorSampler

        def syn_factory() -> Any:
            sampler = _SupervisorSampler(
                profile,
                client=engine.client,
                supervisor=engine.supervisor,
                cameras=engine.camera_ids,
                adopted_started=engine.is_started,
                on_grow=self._grow_one_threadsafe,
            )

            def sample_fn(n: int) -> list[float]:
                return sampler.sample(n, dwell_s=dwell, max_ticks=2000, tick_sleep_s=0.005)

            sample_fn._sampler = sampler  # type: ignore[attr-defined]
            return sample_fn

        return syn_factory

    # ── controller slots ─────────────────────────────────────────────────────────

    def _on_progress(self, step_index: int, total: int, eta_s: float) -> None:
        self._controls.set_verdict(
            f"Ramping… step {step_index} of {total} (ETA ~{int(round(eta_s))} s)"
        )

    def _on_step(self, step: StepResult) -> None:
        # Slice 5: snapshot the live quality accumulators for THIS step, enrich the
        # StepResult with {cid: QualityMetrics dict} BEFORE forwarding to the chart /
        # persistence, then RESET the accumulators so the next step starts clean (no
        # double-feeding / carryover between steps).
        quality = {cid: acc.finalize().to_dict() for cid, acc in self._quality.items()}
        if quality:
            step = dataclasses.replace(step, per_camera_quality=quality)
        self._quality = {}
        self._chart._steps.append(step)
        self._chart.set_steps(self._chart._steps, self._effective_floor())
        # The ramp grew the wall this step; re-target so any newly added camera also
        # locks on (idempotent — same seed re-commits the same per-position targets).
        self._engine.auto_track_targets(seed=0xA17)
        if step.per_camera_fps:
            verb = "sustaining" if step.sustained else "dropped at"
            self._controls.set_verdict(f"{verb} {step.cameras} cam(s) @ {step.min_fps:.1f} fps")

    def _on_finished(self, result: BenchmarkResult) -> None:
        self._result = result
        # Slice 5: when GT is active, finalize each camera's comparator into an
        # aggregate {cid: gt.finalize()} stashed for persistence (the Save path
        # passes it to the writer so the CSV/JSON carry the CLEAR-MOT accuracy).
        if self._gt is not None:
            self._gt_summary = {cid: cmp.finalize() for cid, cmp in self._gt.items()}
        self._teardown_controller()
        self._controls.set_running(False)
        # Color steps against the SAME discounted floor the ramp graded with (the
        # chart's pass line), not result.floor_fps mixed with the raw target — so
        # green dots sit above the line and red below.
        self._chart.set_steps(list(result.steps) or self._chart._steps, self._effective_floor())
        # Show the user's TARGET fps (what they chose), not the discounted pass floor
        # the runner ran with (result.floor_fps) — that internal threshold is confusing.
        self._controls.set_verdict(
            f"Done — {result.sustained_cameras} camera(s) kept up with "
            f"{self._session.target_fps():.0f} fps (score {result.score})."
        )
        # Offer to save the result (Save-As).  When prompting is off (tests /
        # headless) this persists directly under the benchmarks dir instead.
        self._show_completion_dialog(result)

    def _show_completion_dialog(self, result: BenchmarkResult) -> None:
        """Prompt a Save-As for the finished result (or persist directly when off).

        With ``_prompt_on_completion`` off this just calls :meth:`_persist_result`
        (the headless path).  Otherwise it opens a single ``QFileDialog`` (guarded
        against re-entry) offering JSON or CSV: the chosen filter / path suffix
        selects the writer (:func:`save_mark_result_csv` for ``.csv``, else
        :func:`save_mark_result_to_path` for JSON) and the directory is remembered;
        Cancel leaves the result in :attr:`_result` for save-on-exit and does NOT
        auto-persist.
        """
        if not self._prompt_on_completion:
            self._persist_result(result)
            return
        if self._showing_completion_dialog:
            return
        self._showing_completion_dialog = True
        try:
            from datetime import UTC, datetime

            from autoptz.config.store import default_config_dir

            base = self._benchmarks_dir or (default_config_dir() / "benchmarks")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            default_path = str(Path(base) / f"autoptz-mark-{stamp}.json")
            path, chosen_filter = QFileDialog.getSaveFileName(
                self,
                "Save AutoPTZ Mark results",
                default_path,
                "JSON (*.json);;CSV (*.csv);;All Files (*)",
            )
            if not path:
                return  # Cancel → keep self._result for save-on-exit; no auto-persist
            try:
                saved = self._write_result(result, Path(path), chosen_filter)
                self._benchmarks_dir = saved.parent
            except Exception:  # noqa: BLE001
                log.exception("Failed to save AutoPTZ Mark result")
        finally:
            self._showing_completion_dialog = False

    def _write_result(self, result: BenchmarkResult, path: Path, chosen_filter: str) -> Path:
        """Write *result* to *path* as CSV or JSON per the chosen filter / suffix.

        CSV is selected when the path suffix is ``.csv`` OR the chosen filter names
        CSV; everything else writes the JSON bundle.  Both remember the directory
        via the returned path.
        """
        from autoptz.benchmark.results import save_mark_result_csv, save_mark_result_to_path

        wants_csv = path.suffix.lower() == ".csv" or "csv" in (chosen_filter or "").lower()
        if wants_csv:
            return save_mark_result_csv([result], path, store=self._engine.store)
        saved, _bundle = save_mark_result_to_path([result], path, store=self._engine.store)
        return saved

    def _on_error(self, message: str) -> None:
        self._teardown_controller()
        self._controls.set_running(False)
        if message.lower() == "cancelled":
            self._controls.set_verdict("Stopped.")
        else:
            self._controls.set_verdict(f"Error: {message}")

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _teardown_controller(self) -> None:
        """Drop our ref to the controller, but only after its thread has finished.

        ``finished``/``error`` are emitted from inside the worker's ``_run`` (it
        calls ``thread.quit()`` in its ``finally`` *after* emitting), so when this
        slot fires the QThread may still be unwinding its event loop.  We wait for
        it to truly finish before releasing the last strong ref.
        """
        controller = self._controller
        self._controller = None
        if controller is not None:
            try:
                controller.wait(5000)
            except Exception:  # noqa: BLE001
                log.debug("controller wait failed", exc_info=True)

    def _persist_result(self, result: BenchmarkResult) -> None:
        from autoptz.benchmark.results import save_mark_result

        try:
            path, _bundle = save_mark_result([result], store=self._engine.store)
            self._benchmarks_dir = path.parent
        except Exception:  # noqa: BLE001
            log.exception("Failed to persist AutoPTZ Mark result")

    def _open_results_folder(self) -> None:
        from PySide6.QtCore import QUrl

        target = self._benchmarks_dir
        if target is None:
            from autoptz.config.store import default_config_dir

            target = default_config_dir() / "benchmarks"
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        except Exception:  # noqa: BLE001
            log.debug("could not open results folder", exc_info=True)

    def _effective_floor(self) -> float:
        """The DISCOUNTED pass floor the ramp actually grades against.

        Mark grades a camera as sustaining at ``target × _MARK_SUSTAIN_RATIO`` (the
        capped sources can't hit the raw target exactly — see
        :class:`MarkRampController`).  The chart's pass line must use THIS threshold
        so green (sustained) dots sit above the line and red (fail) below; using the
        raw target would put passing-but-capped steps below the line.
        """
        from autoptz.ui.mark_runner import MarkRampController

        return float(self._session.target_fps()) * MarkRampController._MARK_SUSTAIN_RATIO

    def _refresh_idle_status(self) -> None:
        self._controls.set_verdict(
            f"Ready — profile {self._session.profile}, up to {self._session.max_cameras} cameras."
        )
        self._chart.set_steps([], self._effective_floor())

    # ── exit / lifecycle ─────────────────────────────────────────────────────────

    def request_return(self) -> None:
        """Deliberate Return-to-AutoPTZ: emit the resume signal (not a quit)."""
        self._returning = True
        self.returnToAppRequested.emit()

    def request_quit(self) -> None:
        """Deliberate Quit-AutoPTZ from the Mark window."""
        self._returning = True
        self.quitRequested.emit()

    def _request_exit(self) -> None:
        """Show the deliberate Return / Quit choice with an optional save.

        This is the visible exit affordance (control-panel button + Help menu).
        ``Cancel`` stays in Mark; ``Return`` / ``Quit`` optionally persist the last
        result (if any) and then route through :meth:`request_return` /
        :meth:`request_quit` — which set ``_returning`` BEFORE emitting so the
        deliberate path never also fires ``closedUnexpectedly``.
        """
        from autoptz.ui.widgets.dialogs.mark_exit import QUIT, RETURN, MarkExitDialog

        dlg = MarkExitDialog(has_result=self._result is not None, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return  # Cancel → stay in Mark
        if dlg.save_results() and self._result is not None:
            self._persist_result(self._result)
        choice = dlg.choice()
        if choice == QUIT:
            self.request_quit()
        elif choice == RETURN:
            self.request_return()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 — Qt override
        """Stop the pump FIRST, then the controller, then the isolated engine.

        Closing NEVER relaunches.  An OS-window close (no deliberate return/quit)
        emits :attr:`closedUnexpectedly` so the owner can route it through the same
        exit flow (default: Return) instead of silently killing the app.
        """
        try:
            pump = getattr(self, "_pump", None)
            if pump is not None:
                pump.stop()
        except Exception:  # noqa: BLE001
            log.debug("mark pump stop failed", exc_info=True)
        controller = self._controller
        if controller is not None:
            try:
                controller.stop()
                controller.wait(5000)
            except Exception:  # noqa: BLE001
                log.debug("controller stop/wait on close failed", exc_info=True)
            self._controller = None
        try:
            self._engine.stop()
        except Exception:  # noqa: BLE001
            log.debug("mark engine stop failed", exc_info=True)
        self._detach_log_handler()
        if not self._returning:
            self.closedUnexpectedly.emit()
        super().closeEvent(event)

    def _detach_log_handler(self) -> None:
        """Remove the Mark session's root-logger handler (idempotent).

        Mark attaches its own ``QtLogHandler`` to the root logger; on exit it must
        be detached so the resumed main app never accumulates a stale handler
        writing into the (now-discarded) Mark log model.
        """
        handler = getattr(self, "_log_handler", None)
        if handler is None:
            return
        self._log_handler = None
        try:
            logging.getLogger().removeHandler(handler)
        except Exception:  # noqa: BLE001
            log.debug("mark log handler detach failed", exc_info=True)

    def _close_should_quit_app(self, event: Any) -> bool:  # noqa: ARG002
        """Closing the Mark window NEVER quits the app — it routes via signals.

        Return/Quit are owned by the suspended :class:`MainWindow` (through
        ``returnToAppRequested`` / ``quitRequested`` / ``closedUnexpectedly``); an
        OS-close here must not trip the primary window's quit-on-close path that
        ``MainWindow.closeEvent`` runs via ``super().closeEvent``.
        """
        return False

    # ── test seams ───────────────────────────────────────────────────────────────

    @classmethod
    def _wall_type(cls) -> type:
        return CameraWall

    def _score_value(self) -> float | None:
        return None if self._result is None else self._result.score
