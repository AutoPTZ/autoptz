"""Unit tests for autoptz.benchmark.runner (ramp + score math, headless)."""

from __future__ import annotations

from autoptz.benchmark.profiles import get_profile
from autoptz.benchmark.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    StepResult,
    _add_synthetic_camera,
    _SupervisorSampler,
    run_benchmark,
)


def _runner_with_fps(fps_by_count, **kw):
    """Build a runner whose sample_fn returns scripted per-camera fps lists.

    ``fps_by_count`` maps a camera count -> a single fps value applied to every
    camera at that count (so min == mean == that value).
    """
    prof = get_profile("full")

    def sample_fn(n: int) -> list[float]:
        return [float(fps_by_count[n])] * n

    return BenchmarkRunner(prof, sample_fn=sample_fn, **kw)


class TestRampStop:
    def test_stops_when_min_fps_drops_below_floor(self) -> None:
        # 1 cam @ 60, 2 @ 50, 3 @ 20 (below floor 24) -> sustained at 2 cameras.
        runner = _runner_with_fps(
            {1: 60.0, 2: 50.0, 3: 20.0},
            floor_fps=24.0,
            max_cameras=16,
            dwell_s=0.0,
        )
        result = runner.run()
        assert isinstance(result, BenchmarkResult)
        assert result.sustained_cameras == 2
        assert result.min_fps_at_sustained == 50.0
        # Three steps were measured (the failing 3rd is recorded, sustained=False).
        assert [s.cameras for s in result.steps] == [1, 2, 3]
        assert result.steps[-1].sustained is False

    def test_stops_at_max_cameras_when_always_sustained(self) -> None:
        runner = _runner_with_fps(
            {1: 40.0, 2: 40.0, 3: 40.0},
            floor_fps=24.0,
            max_cameras=3,
            dwell_s=0.0,
        )
        result = runner.run()
        assert result.sustained_cameras == 3
        assert result.min_fps_at_sustained == 40.0
        assert len(result.steps) == 3  # never exceeded the cap

    def test_zero_sustained_when_one_camera_fails_floor(self) -> None:
        runner = _runner_with_fps({1: 10.0}, floor_fps=24.0, max_cameras=4, dwell_s=0.0)
        result = runner.run()
        assert result.sustained_cameras == 0
        assert result.score == 0.0
        assert len(result.steps) == 1


class TestScore:
    def test_score_formula_full_weight(self) -> None:
        # sustained 2 cams @ 30 fps, weight 1.0 -> 2 * (30/30) * 1.0 = 2.0
        runner = _runner_with_fps(
            {1: 30.0, 2: 30.0, 3: 10.0},
            floor_fps=24.0,
            max_cameras=16,
            dwell_s=0.0,
        )
        result = runner.run()
        assert result.sustained_cameras == 2
        assert result.score == 2.0

    def test_score_uses_profile_weight(self) -> None:
        from autoptz.benchmark.profiles import get_profile as gp

        prof = gp("streams")  # weight 0.8

        def sample_fn(n: int) -> list[float]:
            return [30.0] * n if n <= 4 else [10.0] * n

        runner = BenchmarkRunner(
            prof, sample_fn=sample_fn, floor_fps=24.0, max_cameras=16, dwell_s=0.0
        )
        result = runner.run()
        # sustained 4 @ 30 -> 4 * (30/30) * 0.8 = 3.2
        assert result.sustained_cameras == 4
        assert result.score == 3.2


class TestStepResult:
    def test_step_min_and_mean_and_to_dict(self) -> None:
        prof = get_profile("full")

        def sample_fn(n: int) -> list[float]:
            return [30.0, 18.0]  # min 18 < floor -> not sustained

        runner = BenchmarkRunner(
            prof, sample_fn=sample_fn, floor_fps=24.0, max_cameras=2, dwell_s=0.0
        )
        result = runner.run()
        step = result.steps[0]
        assert isinstance(step, StepResult)
        assert step.min_fps == 18.0
        assert step.mean_fps == 24.0
        assert step.sustained is False
        d = step.to_dict()
        assert d["cameras"] == 1
        assert d["min_fps"] == 18.0


class TestOnStepCallback:
    def test_on_step_called_per_step(self) -> None:
        prof = get_profile("full")
        seen: list[int] = []

        def sample_fn(n: int) -> list[float]:
            return [40.0] * n if n < 3 else [10.0] * n

        runner = BenchmarkRunner(
            prof,
            sample_fn=sample_fn,
            floor_fps=24.0,
            max_cameras=16,
            dwell_s=0.0,
            on_step=lambda s: seen.append(s.cameras),
        )
        runner.run()
        assert seen == [1, 2, 3]


# ── real sampler wiring (injected supervisor + fake worker) ───────────────────


class _FakeSamplerWorker:
    """Fake worker the sampler's injected Supervisor factory builds."""

    def __init__(self, camera_id, config, on_telemetry) -> None:
        self.camera_id = camera_id
        self.config = config
        self.on_telemetry = on_telemetry
        self.shm_name = f"cam_{camera_id[:8]}_preview"
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_running(self) -> bool:
        return self._alive

    def set_features(self, features) -> None:
        self.features = dict(features)

    def start(self) -> None:
        # Emit one telemetry frame so the fps reader has data.
        from autoptz.engine.runtime.messages import TelemetryMsg

        self.on_telemetry(TelemetryMsg(camera_id=self.camera_id, seq=1, fps=30.0, target_fps=30.0))

    def stop(self, timeout: float = 5.0) -> None:
        self._alive = False


class TestAddSyntheticCamera:
    def test_registers_a_synthetic_source(self, qapp) -> None:
        from autoptz.ui.engine_client import EngineClient

        client = EngineClient()
        cid = _add_synthetic_camera(client, 0)
        rec = client.cameraModel.get_record(cid)
        assert rec is not None
        assert rec.camera_config is not None
        assert rec.camera_config.source.type == "synthetic"
        assert rec.camera_config.source.address == "anim"
        assert rec.camera_config.source.fps == 30.0

    def test_add_synthetic_camera_native_fps(self, qapp) -> None:
        # A clip's native fps flows to the synthetic source; sane values pass through,
        # None falls back to 30, and out-of-range values clamp to (0, 240].
        from autoptz.ui.engine_client import EngineClient

        client = EngineClient()

        def fps_for(native):
            cid = _add_synthetic_camera(client, 0, native_fps=native)
            return client.cameraModel.get_record(cid).camera_config.source.fps

        assert fps_for(60.0) == 60.0
        assert fps_for(None) == 30.0
        assert fps_for(999.0) == 240.0
        assert fps_for(0) == 30.0

    def test_address_override_flows_to_source(self, qapp) -> None:
        # A non-default address (e.g. the bundled clip path) lands on the source so
        # the SyntheticAdapter loops the real clip instead of drawing synthetic people.
        from autoptz.ui.engine_client import EngineClient

        client = EngineClient()
        cid = _add_synthetic_camera(client, 0, address="/some/clip.mp4")
        rec = client.cameraModel.get_record(cid)
        assert rec is not None and rec.camera_config is not None
        assert rec.camera_config.source.type == "synthetic"
        assert rec.camera_config.source.address == "/some/clip.mp4"


class TestSupervisorSampler:
    def test_sampler_reports_fps_per_camera(self, qapp) -> None:
        from autoptz.engine.supervisor import Supervisor

        def factory(client, store):
            return Supervisor(client, store=store, worker_factory=_FakeSamplerWorker)

        sampler = _SupervisorSampler(get_profile("full"), supervisor_factory=factory)
        try:
            # fps_reader reads the model record's last telemetry fps.
            def reader(client, cid) -> float:
                rec = client.cameraModel.get_record(cid)
                return float(getattr(rec, "fps", 0.0) or 0.0)

            fps = sampler.sample(2, dwell_s=0.0, max_ticks=5, tick_sleep_s=0.0, fps_reader=reader)
            assert len(fps) == 2
            assert all(f == 30.0 for f in fps)
        finally:
            sampler.close()

    def test_sampler_registers_cameras_on_injected_client(self, qapp) -> None:
        """An injected client (the Mark window's) receives the synthetic cameras.

        Without this the window's CameraWall — bound to that client — stays empty
        during a run (the sampler would build a private client instead).
        """
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient

        injected = EngineClient()

        def factory(client, store):
            return Supervisor(client, store=store, worker_factory=_FakeSamplerWorker)

        sampler = _SupervisorSampler(
            get_profile("full"), supervisor_factory=factory, client=injected
        )
        try:
            assert sampler._client is injected
            sampler.sample(2, dwell_s=0.0, max_ticks=3, tick_sleep_s=0.0)
            # The synthetic cameras landed on the SAME client the wall observes.
            assert len(injected.cameraModel.camera_ids()) == 2
        finally:
            sampler.close()

    def test_adopted_sampler_grows_cameras_progressively(self, qapp) -> None:
        """An ADOPTED sampler with an ``on_grow`` hook adds cameras ONE AT A TIME.

        3DMark-style: the wall starts at 1 camera and grows as the ramp steps up.
        The sampler calls ``on_grow`` (the Mark factory's add-next) to register the
        next synthetic camera on the SAME client + supervisor, never building a
        second supervisor.
        """
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient

        injected = EngineClient()
        sup = Supervisor(injected, store=None, worker_factory=_FakeSamplerWorker)
        sup.prime_features(dict(get_profile("full").features))
        # Pre-add only ONE camera (the idle wall starts at 1), start the supervisor.
        cams = [_add_synthetic_camera(injected, 0)]
        sup.start(run_pump=False)

        def on_grow() -> str:
            cid = _add_synthetic_camera(injected, len(cams))
            cams.append(cid)
            sup._spawn_worker(cid)  # bring up just the new worker (no full restart)
            return cid

        try:
            sampler = _SupervisorSampler(
                get_profile("full"),
                client=injected,
                supervisor=sup,
                cameras=cams,
                adopted_started=True,
                on_grow=on_grow,
            )
            # Step 1: one camera, no growth.
            fps1 = sampler.sample(1, dwell_s=0.0, max_ticks=3, tick_sleep_s=0.0)
            assert len(fps1) == 1
            assert len(injected.cameraModel.camera_ids()) == 1
            # Step 2: the sampler grows the wall to 2.
            fps2 = sampler.sample(2, dwell_s=0.0, max_ticks=3, tick_sleep_s=0.0)
            assert len(fps2) == 2
            assert len(injected.cameraModel.camera_ids()) == 2
            # Step 3: grows to 3.
            sampler.sample(3, dwell_s=0.0, max_ticks=3, tick_sleep_s=0.0)
            assert len(injected.cameraModel.camera_ids()) == 3
            sampler.close()
            assert sup.is_running is True  # adopted supervisor left running
        finally:
            sup.stop()

    def test_adopted_sampler_reuses_supervisor_and_cameras(self, qapp) -> None:
        """An ADOPTED sampler reuses an external supervisor + pre-added cameras and
        adds no new cameras (the Mark double-stack fix): it must not build a second
        supervisor, must not register extra cameras, and close() must not stop the
        adopted supervisor (the owner does)."""
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient

        injected = EngineClient()
        sup = Supervisor(injected, store=None, worker_factory=_FakeSamplerWorker)
        sup.prime_features(dict(get_profile("full").features))
        # Pre-add 2 cameras and start the supervisor (mirrors the Mark factory).
        cams = [_add_synthetic_camera(injected, i) for i in range(2)]
        sup.start(run_pump=False)
        try:
            sampler = _SupervisorSampler(
                get_profile("full"),
                client=injected,
                supervisor=sup,
                cameras=cams,
                adopted_started=True,
            )
            assert sampler._adopted is True
            assert sampler._sup is sup  # no second supervisor built
            # Sampling does NOT add cameras (the adopted set is reused).
            sampler.sample(2, dwell_s=0.0, max_ticks=5, tick_sleep_s=0.0)
            assert injected.cameraModel.camera_ids() == cams
            # close() leaves the adopted supervisor running (owner stops it).
            sampler.close()
            assert sup.is_running is True
        finally:
            sup.stop()


class TestSamplerWarmup:
    """Change A: the adopted sampler waits out the one-time engine warmup before it
    measures, so step 1 reflects steady state (model loaded + frames flowing) rather
    than the ~8s model-load window that reads ~0 fps and stops the ramp instantly."""

    def _adopted_sampler(self, qapp):
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient

        injected = EngineClient()
        sup = Supervisor(injected, store=None, worker_factory=_FakeSamplerWorker)
        sup.prime_features(dict(get_profile("full").features))
        cams = [_add_synthetic_camera(injected, 0)]
        sup.start(run_pump=False)
        sampler = _SupervisorSampler(
            get_profile("full"),
            client=injected,
            supervisor=sup,
            cameras=cams,
            adopted_started=True,
        )
        return sampler, sup

    def test_warmup_waits_until_fps_rises(self, qapp) -> None:
        sampler, sup = self._adopted_sampler(qapp)
        try:
            # fps reads 0 for the first few polls (model still loading), then rises
            # above the warmup floor (10) — the sampler must keep polling until then.
            polls = {"n": 0}

            def rising_reader(_client, _cid) -> float:
                polls["n"] += 1
                return 0.0 if polls["n"] < 4 else 30.0

            # Tight warmup knobs so the test is fast and offscreen-safe.
            sampler._warmup(rising_reader, min_fps=10.0, timeout_s=5.0, settle_s=0.0, poll_s=0.0)
            # It polled past the cold reads until fps cleared the floor, then warmed.
            assert polls["n"] >= 4
            assert sampler._warmed is True
        finally:
            sup.stop()

    def test_warmup_is_one_shot(self, qapp) -> None:
        sampler, sup = self._adopted_sampler(qapp)
        try:
            calls = {"n": 0}

            def reader(_client, _cid) -> float:
                calls["n"] += 1
                return 30.0

            sampler._warmup(reader, min_fps=10.0, timeout_s=1.0, settle_s=0.0, poll_s=0.0)
            first = calls["n"]
            assert first >= 1
            # Second warmup is a no-op (already warmed) — no further reads.
            sampler._warmup(reader, min_fps=10.0, timeout_s=1.0, settle_s=0.0, poll_s=0.0)
            assert calls["n"] == first
        finally:
            sup.stop()

    def test_warmup_times_out_when_fps_never_rises(self, qapp) -> None:
        sampler, sup = self._adopted_sampler(qapp)
        try:
            import time as _time

            t0 = _time.monotonic()
            # fps stuck at 0 → warmup must give up at the (small) timeout, not hang.
            sampler._warmup(
                lambda _c, _i: 0.0, min_fps=10.0, timeout_s=0.2, settle_s=0.0, poll_s=0.02
            )
            elapsed = _time.monotonic() - t0
            assert sampler._warmed is True
            assert elapsed < 2.0  # bounded by the timeout, didn't hang
        finally:
            sup.stop()


class TestRunBenchmarkWiring:
    def test_run_benchmark_with_injected_supervisor(self, qapp, capsys) -> None:
        from autoptz.engine.supervisor import Supervisor

        def factory(client, store):
            return Supervisor(client, store=store, worker_factory=_FakeSamplerWorker)

        def reader(client, cid) -> float:
            rec = client.cameraModel.get_record(cid)
            return float(getattr(rec, "fps", 0.0) or 0.0)

        code = run_benchmark(
            profile="full",
            floor_fps=24.0,
            max_cameras=3,
            dwell_s=0.0,
            supervisor_factory=factory,
            fps_reader=reader,
            max_ticks=5,
            tick_sleep_s=0.0,
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "AutoPTZ Mark" in out
        # 30 fps everywhere, floor 24 -> sustained at the cap (3).
        assert "sustained 3" in out

    def test_run_benchmark_unknown_profile(self, qapp, capsys) -> None:
        code = run_benchmark(profile="bogus")
        assert code == 2
        assert "unknown benchmark profile" in capsys.readouterr().out.lower()


class TestJsonOutput:
    def test_run_benchmark_writes_json(self, qapp, tmp_path, capsys) -> None:
        from autoptz.engine.supervisor import Supervisor

        def factory(client, store):
            return Supervisor(client, store=store, worker_factory=_FakeSamplerWorker)

        def reader(client, cid) -> float:
            rec = client.cameraModel.get_record(cid)
            return float(getattr(rec, "fps", 0.0) or 0.0)

        out = tmp_path / "mark.json"
        code = run_benchmark(
            profile="full",
            floor_fps=24.0,
            max_cameras=2,
            dwell_s=0.0,
            json_path=str(out),
            supervisor_factory=factory,
            fps_reader=reader,
            max_ticks=5,
            tick_sleep_s=0.0,
        )
        assert code == 0
        assert f"wrote: {out}" in capsys.readouterr().out

        import json

        data = json.loads(out.read_text())
        assert data["profile"] == "full"
        assert data["weight"] == 1.0
        assert data["sustained_cameras"] == 2
        assert data["score"] == 2.0  # 2 * (30/30) * 1.0
        assert isinstance(data["steps"], list) and len(data["steps"]) == 2
        assert data["steps"][0]["cameras"] == 1
        assert "min_fps" in data["steps"][0]
