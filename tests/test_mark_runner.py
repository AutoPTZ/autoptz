from __future__ import annotations

from autoptz.benchmark.runner import BenchmarkResult, StepResult
from autoptz.ui.mark_runner import MarkRampController


def _drive(controller, qapp):  # noqa: ARG001
    """Run the controller body synchronously and collect emitted signals."""
    done: dict[str, object] = {}
    steps: list[StepResult] = []
    progress: list[tuple[int, int, float]] = []
    controller.step_completed.connect(steps.append)
    controller.progress.connect(lambda s, t, e: progress.append((s, t, e)))
    controller.finished.connect(lambda r: done.setdefault("result", r))
    controller.error.connect(lambda m: done.setdefault("error", m))
    controller._run()
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

    def test_default_factory_threads_injected_client(self, qapp, monkeypatch) -> None:
        """The default sampler registers its synthetic cameras on the injected client.

        Drives the controller WITHOUT a sample_factory (so the real default
        ``_SupervisorSampler`` path runs) but with a fake supervisor factory so no
        real inference happens, and asserts the synthetic cameras land on the
        client we handed the controller (the window's CameraWall client).
        """
        from autoptz.benchmark.runner import _SupervisorSampler
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient
        from tests.test_benchmark_runner import _FakeSamplerWorker

        monkeypatch.setattr(_SupervisorSampler, "_drain_events", staticmethod(lambda: None))
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

    def test_uses_exact_floor_for_release_scoring(self, qapp) -> None:
        """Mark grades against the selected target FPS, without a hidden 85% floor."""
        from autoptz.ui.mark_runner import MarkRampController

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
        assert res.floor_fps == 30.0
        assert MarkRampController._MARK_SUSTAIN_RATIO == 1.0
        assert res.sustained_cameras == 0
        assert len(steps) == 1
        assert steps[0].sustained is False

    def test_user_target_preserved_for_display(self, qapp) -> None:
        """The user's Target stays available unchanged for display."""
        from autoptz.ui.mark_runner import MarkRampController

        c = MarkRampController(profile="full", floor_fps=30.0, dwell_s=0.0)
        assert c._floor == 30.0  # the user's target, undiscounted

    def test_stop_before_run_sets_sampler_cancel_event(self, qapp) -> None:
        """A mid-run Mark exit must propagate cancellation into the active sampler."""
        seen: dict[str, object] = {}

        class _Sampler:
            def set_cancel_event(self, event) -> None:  # noqa: ANN001
                seen["event"] = event

            def close(self) -> None:
                seen["closed"] = True

        sampler = _Sampler()

        def factory():
            def sample(_n: int) -> list[float]:
                seen["sampled"] = True
                return [40.0]

            sample._sampler = sampler  # type: ignore[attr-defined]
            return sample

        c = MarkRampController(profile="full", max_cameras=1, dwell_s=10.0, sample_factory=factory)
        c.stop()
        done, _steps, _progress = _drive(c, qapp)

        assert done["error"] == "cancelled"
        assert "sampled" not in seen
        assert seen["event"].is_set()
        assert seen["closed"] is True

    def test_run_emits_finished_before_thread_quit(self, qapp) -> None:
        """The worker must emit ``finished`` BEFORE it quits the thread, so the
        queued result is enqueued ahead of the thread tearing down its event loop."""
        order: list[str] = []

        c = MarkRampController(
            profile="full",
            floor_fps=24.0,
            max_cameras=1,
            dwell_s=0.0,
            sample_factory=lambda: (lambda n: [40.0] * n),
        )
        # Record finished-emission order via a DIRECT connection (runs inline in the
        # worker's _run, before the finally's thread.quit()).
        from PySide6.QtCore import Qt

        c.finished.connect(lambda _r: order.append("finished"), Qt.ConnectionType.DirectConnection)

        class _FakeThread:
            def quit(self_inner) -> None:
                order.append("quit")

            def wait(self_inner, _ms: int = 5000) -> bool:
                return True

        c._thread = _FakeThread()  # type: ignore[assignment]
        c._run()  # drive the worker body synchronously
        assert order == ["finished", "quit"]

    def test_run_error_emitted_before_thread_quit(self, qapp) -> None:
        """On a sample exception the worker emits ``error`` BEFORE thread.quit()."""
        order: list[str] = []

        def factory():
            def boom(n):
                raise RuntimeError("kaboom")

            return boom

        c = MarkRampController(profile="full", dwell_s=0.0, sample_factory=factory)
        from PySide6.QtCore import Qt

        c.error.connect(lambda _m: order.append("error"), Qt.ConnectionType.DirectConnection)

        class _FakeThread:
            def quit(self_inner) -> None:
                order.append("quit")

            def wait(self_inner, _ms: int = 5000) -> bool:
                return True

        c._thread = _FakeThread()  # type: ignore[assignment]
        c._run()
        assert order == ["error", "quit"]

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
        c.start()
        assert c.wait(2000) is True
        # Held by the controller and joinable (moveToThread → unparented thread).
        assert c._thread is not None
        assert c._thread.isFinished()
