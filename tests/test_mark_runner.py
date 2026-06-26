from __future__ import annotations

from autoptz.benchmark.runner import BenchmarkResult, StepResult
from autoptz.ui.mark_runner import MarkRampController


def _drive(controller, qapp):
    """Run the controller's worker to completion, pumping the event loop."""
    from PySide6.QtCore import QCoreApplication

    done: dict[str, object] = {}
    steps: list[StepResult] = []
    progress: list[tuple[int, int, float]] = []
    controller.step_completed.connect(steps.append)
    controller.progress.connect(lambda s, t, e: progress.append((s, t, e)))
    controller.finished.connect(lambda r: done.setdefault("result", r))
    controller.error.connect(lambda m: done.setdefault("error", m))
    controller.start()
    # Pump until finished/error or a generous bound.
    import time

    deadline = time.monotonic() + 10.0
    while "result" not in done and "error" not in done and time.monotonic() < deadline:
        QCoreApplication.instance().processEvents()
        time.sleep(0.005)
    return done, steps, progress


class TestRampController:
    def test_runs_to_completion_and_emits_result(self, qapp) -> None:
        # 1@40, 2@40, 3@10 (below floor) -> sustained 2.
        script = {1: 40.0, 2: 40.0, 3: 10.0}

        def factory():
            return lambda n: [script[n]] * n

        c = MarkRampController(
            profile="full",
            floor_fps=24.0,
            max_cameras=16,
            dwell_s=0.0,
            sample_factory=factory,
        )
        done, steps, progress = _drive(c, qapp)
        assert "error" not in done
        res = done["result"]
        assert isinstance(res, BenchmarkResult)
        assert res.sustained_cameras == 2
        assert [s.cameras for s in steps] == [1, 2, 3]
        assert progress and progress[0][1] == 16  # total == max_cameras

    def test_error_signal_on_sample_exception(self, qapp) -> None:
        def factory():
            def boom(n):
                raise RuntimeError("kaboom")

            return boom

        c = MarkRampController(profile="full", dwell_s=0.0, sample_factory=factory)
        done, steps, progress = _drive(c, qapp)
        assert "error" in done and "kaboom" in done["error"]
