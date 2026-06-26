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

import logging
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PySide6.QtWidgets import (
    QDockWidget,
    QVBoxLayout,
    QWidget,
)

from autoptz.benchmark.runner import BenchmarkResult, StepResult
from autoptz.ui import theme as T
from autoptz.ui.mark_session import MarkSession
from autoptz.ui.widgets.camera_wall import CameraWall
from autoptz.ui.widgets.main_window import MainWindow, _action
from autoptz.ui.widgets.mark_control_panel import MarkControlPanel
from autoptz.ui.widgets.mark_details_panel import MarkDetailsPanel

log = logging.getLogger(__name__)


class _MarkRampChart(QWidget):
    """A small QPainter line chart: x = camera count, y = min fps per step.

    Each step's marker is green when it held the floor and red when it failed; a
    dashed horizontal line marks the floor.  Painting tolerates an empty / single
    step list (draws just the axes + floor).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._steps: list[StepResult] = []
        self._floor: float = 24.0
        self.setMinimumHeight(120)

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
        from autoptz.ui.mark_engine import MarkEngineFactory

        self._session = session or MarkSession()
        # The isolated stack (temp-file store + own client/supervisor, fake-only
        # cameras) MUST exist before super().__init__ binds the wall to a client.
        self._engine = MarkEngineFactory(self._session)
        self._controller: Any | None = None
        self._result: BenchmarkResult | None = None
        self._benchmarks_dir: Any | None = None
        self._returning = False
        super().__init__(
            self._engine.client,
            log_model=None,
            frame_source=frame_source,
            theme=theme,
            parent=parent,
        )
        self.setWindowTitle("AutoPTZ Mark")  # NO version (lives in About)
        self.resize(1200, 780)
        self._inject_mark_ui()
        self._hide_nonessential_docks()
        self._start_pump()
        self._refresh_idle_status()

    # ── overrides ────────────────────────────────────────────────────────────────

    def _should_poll_usb(self) -> bool:
        return False

    def _build_menus(self) -> None:  # override: Help / About only
        bar = self.menuBar()
        helpm = bar.addMenu("&Help")
        helpm.addAction(_action(self, "About AutoPTZ Mark", self._show_about_mark))

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
        self._controls.startClicked.connect(self.start_run)
        self._controls.stopClicked.connect(self.stop_run)

        # Insert chart + controls at the TOP of the inherited central column,
        # above the CameraWall (index 0/1 keep the startup banner first).
        central = self.centralWidget()
        layout = central.layout() if central is not None else None
        if isinstance(layout, QVBoxLayout):
            layout.insertWidget(1, self._chart)
            layout.insertWidget(2, self._controls)

        # Replace the right "Camera Info" dock content with the Mark details panel.
        self._details = MarkDetailsPanel(self._engine.client)
        info_dock = self._docks.get("camera_info")
        if isinstance(info_dock, QDockWidget):
            info_dock.setWidget(self._details)
            info_dock.setWindowTitle("Details")
            info_dock.setVisible(True)
            info_dock.raise_()

    def _on_camera_selected(self, camera_id: str) -> None:
        # Drive the Mark details panel from the inherited tile-selection slot.
        self._selected_camera = camera_id
        details = getattr(self, "_details", None)
        if details is not None:
            details.set_camera(camera_id)

    def _start_pump(self) -> None:
        import os

        from PySide6.QtCore import QTimer

        self._pump = QTimer(self)
        self._pump.setInterval(33)  # ~30 Hz
        self._pump.timeout.connect(self._engine.tick)
        self._pump.start()
        # Offscreen lifecycle tests construct the window to poke private state /
        # drive fake signals; they set this flag so the real Supervisor (model
        # loading + staged camera open) never spins up.
        if os.environ.get("AUTOPTZ_MARK_NO_AUTOSTART", "").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            self._engine.start()

    # ── run lifecycle ────────────────────────────────────────────────────────────

    def start_run(self) -> None:
        if self._controller is not None:
            return
        from autoptz.ui.mark_runner import MarkRampController

        self._result = None
        self._chart.set_steps([], self._session.floor_fps)
        self._controls.set_running(True)
        self._controls.set_verdict("Starting…")
        controller = MarkRampController(
            profile=self._session.profile,
            floor_fps=self._session.floor_fps,
            max_cameras=self._controls.selected_max_cameras(),
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

    def _build_sample_factory(self) -> Any:
        """Return the controller's ``sample_factory`` (or None for the default).

        ``None`` → :class:`MarkRampController` builds the real ``_SupervisorSampler``
        on the isolated client (synthetic in-process cameras).  For ``source ==
        "ndi"`` (and cyndilib present), return a zero-arg factory that builds a
        :class:`MarkNDIFleetSampler` so a real ``MarkNDIFleet`` is broadcast and
        ingested through the live NDIAdapter on the isolated client.  If cyndilib
        is unavailable we fall back to the synthetic default (the control panel
        already disables the NDI radio in that case).
        """
        if self._session.source != "ndi":
            return None
        from autoptz.benchmark.ndi_sim import ndi_sample_factory, ndi_sim_available

        if not ndi_sim_available():
            log.warning("NDI source requested but cyndilib is unavailable; using synthetic.")
            return None

        def factory() -> Any:
            return ndi_sample_factory(
                self._session.profile,
                self._session.dwell_s,
                client=self._engine.client,
                max_cameras=self._controls.selected_max_cameras(),
            )

        return factory

    # ── controller slots ─────────────────────────────────────────────────────────

    def _on_progress(self, step_index: int, total: int, eta_s: float) -> None:
        self._controls.set_verdict(
            f"Ramping… step {step_index} of {total} (ETA ~{int(round(eta_s))} s)"
        )

    def _on_step(self, step: StepResult) -> None:
        self._chart._steps.append(step)
        self._chart.set_steps(self._chart._steps, self._session.floor_fps)
        if step.per_camera_fps:
            verb = "sustaining" if step.sustained else "dropped at"
            self._controls.set_verdict(f"{verb} {step.cameras} cam(s) @ {step.min_fps:.1f} fps")

    def _on_finished(self, result: BenchmarkResult) -> None:
        self._result = result
        self._teardown_controller()
        self._controls.set_running(False)
        self._chart.set_steps(list(result.steps) or self._chart._steps, result.floor_fps)
        self._controls.set_verdict(
            f"Done — sustained {result.sustained_cameras} cam(s) "
            f"@ ≥{result.floor_fps:.0f} fps (score {result.score})."
        )
        self._persist_result(result)

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

    def _refresh_idle_status(self) -> None:
        self._controls.set_verdict(
            f"Ready — profile {self._session.profile}, up to "
            f"{self._controls.selected_max_cameras()} cameras."
        )
        self._chart.set_steps([], self._session.floor_fps)

    # ── exit / lifecycle ─────────────────────────────────────────────────────────

    def request_return(self) -> None:
        """Deliberate Return-to-AutoPTZ: emit the resume signal (not a quit)."""
        self._returning = True
        self.returnToAppRequested.emit()

    def request_quit(self) -> None:
        """Deliberate Quit-AutoPTZ from the Mark window."""
        self._returning = True
        self.quitRequested.emit()

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
        if not self._returning:
            self.closedUnexpectedly.emit()
        super().closeEvent(event)

    # ── test seams ───────────────────────────────────────────────────────────────

    @classmethod
    def _wall_type(cls) -> type:
        return CameraWall

    def _score_value(self) -> float | None:
        return None if self._result is None else self._result.score
