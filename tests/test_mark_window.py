"""MarkWindow (offscreen): construct, drive HUD/result/error slots with fakes."""

from __future__ import annotations

import pytest

from autoptz.benchmark.runner import BenchmarkResult, StepResult
from autoptz.ui.mark_session import MarkSession


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _make_window(qtapp, *, session: MarkSession | None = None):
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.mark_window import MarkWindow

    client = EngineClient()
    frames = ShmFrameSource()
    win = MarkWindow(
        client,
        frames,
        session=session or MarkSession(max_cameras=3, dwell_s=0.0),
    )
    return win


class TestMarkWindow:
    def test_constructs_with_camera_wall_and_hud(self, qtapp) -> None:
        win = _make_window(qtapp)
        assert win.findChild(type(win)._wall_type()) is not None  # CameraWall embedded
        win.deleteLater()

    def test_progress_updates_hud(self, qtapp) -> None:
        win = _make_window(qtapp)
        win._on_progress(2, 3, 4.0)
        status = win._status_text().lower()
        assert "2" in status and "3" in status  # step 2 of 3
        win.deleteLater()

    def test_step_updates_chart_and_fps(self, qtapp) -> None:
        win = _make_window(qtapp)
        win._on_step(
            StepResult(
                cameras=1,
                min_fps=40.0,
                mean_fps=40.0,
                per_camera_fps=[40.0],
                sustained=True,
            )
        )
        # The chart received at least one step.
        assert len(win._chart._steps) == 1
        win.deleteLater()

    def test_finish_populates_results_panel(self, qtapp) -> None:
        win = _make_window(qtapp)
        win._on_step(
            StepResult(
                cameras=1, min_fps=40.0, mean_fps=40.0, per_camera_fps=[40.0], sustained=True
            )
        )
        result = BenchmarkResult(
            profile="full",
            weight=1.0,
            floor_fps=24.0,
            max_cameras=3,
            sustained_cameras=2,
            min_fps_at_sustained=40.0,
            score=2.0,
            steps=[],
        )
        win._on_finished(result)
        assert win._score_value() == 2.0
        assert "2.0" in win._results_text()
        win.deleteLater()

    def test_error_is_surfaced(self, qtapp) -> None:
        win = _make_window(qtapp)
        win._on_error("kaboom")
        status = win._status_text().lower()
        assert "kaboom" in status or "error" in status
        win.deleteLater()

    def test_cancelled_error_reads_as_stopped_not_failed(self, qtapp) -> None:
        win = _make_window(qtapp)
        win._on_error("cancelled")
        status = win._status_text().lower()
        assert "stop" in status
        assert "fail" not in status and "error" not in status
        win.deleteLater()

    def test_chart_set_steps_does_not_raise(self, qtapp) -> None:
        from autoptz.ui.widgets.mark_window import _MarkRampChart

        chart = _MarkRampChart()
        chart.set_steps(
            [
                StepResult(
                    cameras=1, min_fps=40.0, mean_fps=40.0, per_camera_fps=[40.0], sustained=True
                ),
                StepResult(
                    cameras=2, min_fps=10.0, mean_fps=10.0, per_camera_fps=[10.0], sustained=False
                ),
            ],
            floor=24.0,
        )
        chart.resize(200, 120)
        chart.repaint()  # exercise paintEvent offscreen
        chart.deleteLater()
