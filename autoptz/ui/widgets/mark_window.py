"""MarkWindow — the headful AutoPTZ Mark benchmark window.

A dedicated window the app relaunches into under ``--mark``.  It embeds the real
:class:`CameraWall` (so Center-Stage / framing overlays render on each tile while
the ramp runs) and a top HUD: step / N-cameras / per-camera fps / ETA + a
``QPainter`` ramp chart (green sustained → red fail), Start / Stop / Return
controls, and a results panel that appears when the run finishes.

The ramp itself runs on a worker thread via :class:`MarkRampController`; this
window only marshals its queued signals into HUD updates.  On finish it persists
the result via :func:`save_mark_result`.  "Return to AutoPTZ" relaunches the
normal app and closes this window.

Test seams (offscreen):
  * :meth:`_wall_type` — the embedded wall class (so a test can ``findChild`` it).
  * :meth:`_status_text` / :meth:`_results_text` / :meth:`_score_value` — read the
    HUD/results without painting.
  * The slots ``_on_progress`` / ``_on_step`` / ``_on_finished`` / ``_on_error`` can
    be driven directly with fake ``StepResult`` / ``BenchmarkResult`` values.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autoptz.benchmark.runner import BenchmarkResult, StepResult
from autoptz.ui import theme as T
from autoptz.ui.mark_session import MarkSession
from autoptz.ui.widgets.camera_wall import CameraWall

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


class MarkWindow(QMainWindow):
    """Headful AutoPTZ Mark window: embedded camera wall + ramp HUD + results."""

    def __init__(
        self,
        client: Any,
        frame_source: Any,
        *,
        session: MarkSession | None = None,
        store: Any | None = None,
        theme: Any | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._frames = frame_source
        self._session = session or MarkSession()
        # Persist target for results: prefer an explicit store, else the client's.
        self._store = store if store is not None else getattr(client, "_store", None)
        self._theme = theme
        self._controller: Any | None = None
        self._result: BenchmarkResult | None = None
        self._benchmarks_dir: Any | None = None

        self.setWindowTitle("AutoPTZ Mark")
        self.resize(1100, 740)

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addWidget(self._build_hud())
        self._wall = CameraWall(client, frame_source)
        root.addWidget(self._wall, 1)
        root.addWidget(self._build_controls())
        root.addWidget(self._build_results_panel())

        self.setCentralWidget(central)
        self._refresh_idle_status()

    # ── construction helpers ────────────────────────────────────────────────────

    def _build_hud(self) -> QWidget:
        hud = QWidget(self)
        row = QHBoxLayout(hud)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-weight: 600;")
        self._fps_label = QLabel("")
        self._fps_label.setStyleSheet(f"color: {T.CURRENT.subtext};")
        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet(f"color: {T.CURRENT.subtext};")

        row.addWidget(self._status_label)
        row.addStretch(1)
        row.addWidget(self._fps_label)
        row.addWidget(self._eta_label)

        outer = QWidget(self)
        col = QVBoxLayout(outer)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        col.addWidget(hud)
        self._chart = _MarkRampChart(outer)
        col.addWidget(self._chart)
        return outer

    def _build_controls(self) -> QWidget:
        bar = QWidget(self)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self.start_run)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self.stop_run)
        self._stop_btn.setEnabled(False)
        self._return_btn = QPushButton("Return to AutoPTZ")
        self._return_btn.clicked.connect(self.return_to_app)

        row.addWidget(self._start_btn)
        row.addWidget(self._stop_btn)
        row.addStretch(1)
        row.addWidget(self._return_btn)
        return bar

    def _build_results_panel(self) -> QWidget:
        self._results_panel = QWidget(self)
        col = QVBoxLayout(self._results_panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self._score_label = QLabel("")
        self._score_label.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {T.TRACKING};")
        self._results_label = QLabel("")
        self._results_label.setWordWrap(True)
        self._results_label.setStyleSheet(f"color: {T.CURRENT.subtext};")

        open_row = QHBoxLayout()
        self._open_folder_btn = QPushButton("Open results folder")
        self._open_folder_btn.clicked.connect(self._open_results_folder)
        open_row.addWidget(self._open_folder_btn)
        open_row.addStretch(1)

        col.addWidget(self._score_label)
        col.addWidget(self._results_label)
        col.addLayout(open_row)
        self._results_panel.setVisible(False)
        return self._results_panel

    # ── run lifecycle ───────────────────────────────────────────────────────────

    def start_run(self) -> None:
        if self._controller is not None:
            return
        from autoptz.ui.mark_runner import MarkRampController

        self._result = None
        self._chart.set_steps([], self._session.floor_fps)
        self._results_panel.setVisible(False)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_label.setText("Starting AutoPTZ Mark…")

        controller = MarkRampController(
            profile=self._session.profile,
            floor_fps=self._session.floor_fps,
            max_cameras=self._session.max_cameras,
            dwell_s=self._session.dwell_s,
            sample_factory=self._build_sample_factory(),
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
            self._status_label.setText("Stopping…")
        self._stop_btn.setEnabled(False)

    def return_to_app(self) -> None:
        from autoptz.ui.mark_session import clear_mark_session, relaunch

        if self._store is not None:
            try:
                clear_mark_session(self._store)
            except Exception:  # noqa: BLE001
                log.debug("clear_mark_session failed", exc_info=True)
        try:
            relaunch(mark=False)
        except Exception:  # noqa: BLE001
            log.exception("relaunch to normal app failed")
        self.close()

    def _build_sample_factory(self) -> Any:
        """Return the controller's ``sample_factory`` (or None for the default).

        ``None`` → :class:`MarkRampController` builds the real ``_SupervisorSampler``
        (synthetic in-process cameras).  NDI mode is the user-validated path; if
        cyndilib is unavailable we silently fall back to the synthetic default.
        """
        if self._session.source != "ndi":
            return None
        from autoptz.benchmark.ndi_sim import ndi_sim_available

        if not ndi_sim_available():
            log.warning("NDI source requested but cyndilib is unavailable; using synthetic.")
            return None
        # NDI fleets are validated live by the user; the synthetic sampler still
        # measures throughput, so default to it until the NDI ingest path is wired.
        return None

    # ── controller slots ────────────────────────────────────────────────────────

    def _on_progress(self, step_index: int, total: int, eta_s: float) -> None:
        self._status_label.setText(f"Ramping… step {step_index} of {total}")
        self._eta_label.setText(f"ETA: ~{int(round(eta_s))} s")

    def _on_step(self, step: StepResult) -> None:
        self._chart._steps.append(step)
        self._chart.set_steps(self._chart._steps, self._session.floor_fps)
        if step.per_camera_fps:
            self._fps_label.setText(
                f"{step.cameras} cam(s): min {step.min_fps:.1f} fps / mean {step.mean_fps:.1f} fps"
            )

    def _on_finished(self, result: BenchmarkResult) -> None:
        self._result = result
        self._teardown_controller()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._chart.set_steps(list(result.steps) or self._chart._steps, result.floor_fps)
        self._status_label.setText("AutoPTZ Mark complete.")
        self._eta_label.setText("")
        self._persist_result(result)
        self._populate_results(result)

    def _on_error(self, message: str) -> None:
        self._teardown_controller()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if message.lower() == "cancelled":
            self._status_label.setText("AutoPTZ Mark stopped.")
        else:
            self._status_label.setText(f"AutoPTZ Mark error: {message}")

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _teardown_controller(self) -> None:
        self._controller = None

    def _persist_result(self, result: BenchmarkResult) -> None:
        from autoptz.benchmark.results import save_mark_result

        try:
            path, _bundle = save_mark_result([result], store=self._store)
            self._benchmarks_dir = path.parent
        except Exception:  # noqa: BLE001
            log.exception("Failed to persist AutoPTZ Mark result")

    def _populate_results(self, result: BenchmarkResult) -> None:
        self._score_label.setText(f"Score: {result.score}")
        self._results_label.setText(
            f"Profile: {result.profile} · sustained {result.sustained_cameras} camera(s) "
            f"@ ≥{result.floor_fps:.0f} fps (min {result.min_fps_at_sustained:.1f} fps "
            f"at that count) · est. max {result.sustained_cameras} cameras."
        )
        self._results_panel.setVisible(True)

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
        self._status_label.setText(
            f"Ready — profile {self._session.profile}, up to {self._session.max_cameras} cameras."
        )
        self._chart.set_steps([], self._session.floor_fps)

    # ── test seams ───────────────────────────────────────────────────────────────

    @classmethod
    def _wall_type(cls) -> type:
        return CameraWall

    def _status_text(self) -> str:
        return self._status_label.text()

    def _results_text(self) -> str:
        return f"{self._score_label.text()} {self._results_label.text()}"

    def _score_value(self) -> float | None:
        return None if self._result is None else self._result.score
