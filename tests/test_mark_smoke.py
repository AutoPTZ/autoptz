"""End-to-end AutoPTZ Mark smoke (headless, offscreen).

Drives a real :class:`MarkRampController` (the default ``_SupervisorSampler``)
capped tiny — ``max_cameras=1``, ``dwell_s=0.0`` — to completion through the Qt
event loop, then persists the result via :func:`save_mark_result`.  Tolerant of
``fps == 0`` on CI (no bundled model): asserts a result object + at least one
step + a written JSON file, NOT a specific fps.
"""

from __future__ import annotations

import time

import pytest

from autoptz.benchmark.runner import BenchmarkResult, StepResult


@pytest.fixture
def qapp():
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


def _drive(controller, *, timeout_s: float = 30.0):
    from PySide6.QtCore import QCoreApplication

    done: dict[str, object] = {}
    steps: list[StepResult] = []
    controller.step_completed.connect(steps.append)
    controller.finished.connect(lambda r: done.setdefault("result", r))
    controller.error.connect(lambda m: done.setdefault("error", m))
    controller.start()
    deadline = time.monotonic() + timeout_s
    while "result" not in done and "error" not in done and time.monotonic() < deadline:
        QCoreApplication.instance().processEvents()
        time.sleep(0.005)
    return done, steps


def test_mark_end_to_end_smoke(qapp, tmp_path) -> None:
    from autoptz.benchmark.results import save_mark_result
    from autoptz.ui.mark_runner import MarkRampController

    controller = MarkRampController(
        profile="streams",  # lightest profile (capture only)
        floor_fps=0.0,  # accept any fps so N=1 always "sustains"
        max_cameras=1,
        dwell_s=0.0,
    )
    done, steps = _drive(controller)

    assert "error" not in done, f"ramp errored: {done.get('error')}"
    result = done.get("result")
    assert isinstance(result, BenchmarkResult)
    assert len(steps) == 1
    assert steps[0].cameras == 1

    path, bundle = save_mark_result([result], config_dir=tmp_path)
    assert path.exists()
    assert path.parent.name == "benchmarks"
    assert bundle.results and bundle.results[0]["profile"] == "streams"
