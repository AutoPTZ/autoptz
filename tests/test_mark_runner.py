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

    def test_default_factory_threads_injected_client(self, qapp) -> None:
        """The default sampler registers its synthetic cameras on the injected client.

        Drives the controller WITHOUT a sample_factory (so the real default
        ``_SupervisorSampler`` path runs) but with a fake supervisor factory so no
        real inference happens, and asserts the synthetic cameras land on the
        client we handed the controller (the window's CameraWall client).
        """
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient
        from tests.test_benchmark_runner import _FakeSamplerWorker

        injected = EngineClient()

        def sup_factory(client, store):
            return Supervisor(client, store=store, worker_factory=_FakeSamplerWorker)

        c = MarkRampController(
            profile="full",
            floor_fps=0.0,  # accept any fps so the ramp reaches max_cameras
            max_cameras=2,
            dwell_s=0.0,
            client=injected,
            supervisor_factory=sup_factory,
        )
        done, steps, progress = _drive(c, qapp)
        assert "error" not in done, done.get("error")
        # Cameras registered on the SAME client the window's wall is bound to.
        assert len(injected.cameraModel.camera_ids()) == 2

    def test_discounts_floor_for_sustain_tolerance(self, qapp) -> None:
        """Change B: the MARK path builds its BenchmarkRunner with a DISCOUNTED floor
        (target × 0.85) so a camera keeping up with a fps-capped 30fps source still
        "sustains" despite real-world per-frame overhead (it lands just under 30).

        Asserted via the result's ``floor_fps`` (BenchmarkRunner records the floor it
        ran with) and the sustain decision: at a Target of 30, a camera holding 26fps
        must count as sustained (26 ≥ 30×0.85 = 25.5) even though 26 < 30.
        """
        from autoptz.ui.mark_runner import MarkRampController

        # Every camera holds 26 fps: above the discounted floor (25.5), below 30.
        def factory():
            return lambda n: [26.0] * n

        c = MarkRampController(
            profile="full",
            floor_fps=30.0,  # the user's Target FPS
            max_cameras=2,
            dwell_s=0.0,
            sample_factory=factory,
        )
        done, steps, progress = _drive(c, qapp)
        assert "error" not in done, done.get("error")
        res = done["result"]
        # The runner ran with the DISCOUNTED floor (30 × 0.85 = 25.5), not 30.
        assert res.floor_fps == 30.0 * MarkRampController._MARK_SUSTAIN_RATIO
        # 26 fps clears the discounted floor → both cameras sustain (would FAIL at 30).
        assert res.sustained_cameras == 2
        assert all(s.sustained for s in steps)

    def test_user_target_preserved_for_display(self, qapp) -> None:
        """The discount applies only to the runner's pass floor — the user's Target
        stays available unchanged for display."""
        from autoptz.ui.mark_runner import MarkRampController

        c = MarkRampController(profile="full", floor_fps=30.0, dwell_s=0.0)
        assert c._floor == 30.0  # the user's target, undiscounted

    def test_thread_is_joinable_after_run(self, qapp) -> None:
        """The worker QThread is joinable via wait() so the controller drops its
        ref only after the thread has truly finished.

        Guards the 'QThread: Destroyed while thread is still running' abort: after
        the run completes the thread must be truly finished before any ref drops.
        The controller is moved INTO the thread via moveToThread, so the QThread
        is intentionally unparented — joinability, not parenting, is the contract.
        """
        c = MarkRampController(
            profile="full",
            floor_fps=24.0,
            max_cameras=1,
            dwell_s=0.0,
            sample_factory=lambda: (lambda n: [40.0] * n),
        )
        done, steps, progress = _drive(c, qapp)
        assert "error" not in done
        # Held by the controller and joinable (moveToThread → unparented thread).
        assert c._thread is not None
        assert c.wait(2000) is True
        assert c._thread.isFinished()
