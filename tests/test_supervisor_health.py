"""Tests for R3' — worker liveness monitoring + auto-restart.

All tests are headless and use no real threads: they drive
``_scan_worker_health(now)`` directly with a synthetic monotonic clock so
backoff / cap behaviour is deterministic.  The supervisor's ``worker_factory``
injection keeps the existing fake-worker pattern from ``test_orchestration.py``.
"""

from __future__ import annotations

from autoptz.engine.supervisor import (
    _BASE_BACKOFF_S,
    _INFER_RESTART_S,
    _MAX_BACKOFF_S,
    _MAX_RESTART_ATTEMPTS,
    _MS_SPAWN_TIMEOUT_S,
    _WORKER_HANG_S,
    _WORKER_WARMUP_GRACE_S,
)

# ── helpers / fakes ──────────────────────────────────────────────────────────


def _camera_config(camera_id: str = "cam-1234abcd5678", name: str = "Cam"):
    from autoptz.config.models import CameraConfig, SourceConfig

    return CameraConfig(
        id=camera_id,
        name=name,
        source=SourceConfig(type="usb", address="usb://0"),
    )


class _HealthFakeWorker:
    """Minimal fake worker with a settable is_alive() result.

    Also models the in-process inference-thread health surface
    (``inference_stalled_for`` / ``inference_thread_alive``) so hung-inference
    tests can drive it the same way real ``CameraWorker`` health is driven —
    both default to "healthy" (no stall, thread alive) so existing capture-death
    tests are unaffected.
    """

    def __init__(self, camera_id: str, config, on_telemetry) -> None:
        self.camera_id = camera_id
        self.config = config
        self.on_telemetry = on_telemetry
        self.shm_name = f"cam_{camera_id[:8]}_preview"
        self._alive = True
        self.start_calls = 0
        self.stop_calls = 0
        self._infer_stall_s = 0.0
        self._infer_thread_alive = True

    def is_alive(self) -> bool:
        return self._alive

    @property
    def is_running(self) -> bool:  # compat with _spawn_worker guards
        return self._alive

    def start(self) -> None:
        self.start_calls += 1

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_calls += 1
        self._alive = False

    def inference_stalled_for(self, now: float) -> float:
        return self._infer_stall_s

    def inference_thread_alive(self) -> bool:
        return self._infer_thread_alive


def _make_client(qapp):
    from autoptz.ui.engine_client import EngineClient

    return EngineClient()


def _make_sup_with_factory(client, factory):
    from autoptz.engine.supervisor import Supervisor

    return Supervisor(client, store=None, worker_factory=factory)


def _build(qapp):
    """Return (supervisor, client, factory_log, camera_id).

    ``factory_log`` is a list of every worker the factory ever created
    (one entry per factory call), so ``len(factory_log)`` counts total spawns
    and ``factory_log[0]`` is the first worker, ``factory_log[-1]`` the latest.
    """
    client = _make_client(qapp)
    cid = client.addCamera("usb://0", "X")
    client.drain_commands()  # clear the add cmd

    factory_log: list[_HealthFakeWorker] = []

    def factory(camera_id, config, on_tel):
        w = _HealthFakeWorker(camera_id, config, on_tel)
        factory_log.append(w)
        return w

    sup = _make_sup_with_factory(client, factory)
    # Stub out heavyweight helpers so the test stays headless.
    sup._ensure_identity_service = lambda: None  # type: ignore[method-assign]
    sup._ensure_inference_pool = lambda: None  # type: ignore[method-assign]
    sup.start()
    return sup, client, factory_log, cid


# ── CameraWorker.is_alive() unit test ────────────────────────────────────────


class TestCameraWorkerIsAlive:
    def test_false_before_start(self, qapp) -> None:
        """is_alive() is False before start(): _thread is None, no thread running."""
        from autoptz.engine.camera_worker import CameraWorker

        worker = CameraWorker(
            "hltcam01abcd",
            _camera_config("hltcam01abcd"),
            lambda m: None,
        )
        assert worker._thread is None
        assert worker.is_alive() is False

    def test_inference_stalled_for_zero_before_first_tick(self, qapp) -> None:
        """No inference tick has completed yet → stall age is 0.0 (not a huge
        elapsed-since-epoch number that would falsely look stalled)."""
        from autoptz.engine.camera_worker import CameraWorker

        worker = CameraWorker(
            "hltcam01abcd",
            _camera_config("hltcam01abcd"),
            lambda m: None,
        )
        assert worker._frames_inferred == 0
        assert worker.inference_stalled_for(1_000_000.0) == 0.0

    def test_inference_stalled_for_measures_age_after_first_tick(self, qapp) -> None:
        from autoptz.engine.camera_worker import CameraWorker

        worker = CameraWorker(
            "hltcam01abcd",
            _camera_config("hltcam01abcd"),
            lambda m: None,
        )
        worker._frames_inferred = 1
        worker._last_infer_t = 1000.0
        # A new frame is pending (latest != current-inference) so the stall is real.
        worker._latest_frame_id = 2
        worker._current_inference_frame_id = 1
        assert worker.inference_stalled_for(1000.0) == 0.0
        assert worker.inference_stalled_for(1020.0) == 20.0

    def test_inference_stalled_for_zero_during_source_outage(self, qapp) -> None:
        """Critical (R2 review): a source outage (no new frames) must NOT read as
        an inference stall — the capture loop idles with no pending frame, so the
        inference thread has nothing to consume. Only the capture thread's own
        reconnect backoff should handle this, not a worker restart.

        This is the RED test for the churn bug: before the pending-frame gate,
        ``inference_stalled_for`` grew unboundedly from ``_last_infer_t`` alone
        and would report a large stall here even though nothing is wrong.
        """
        from autoptz.engine.camera_worker import CameraWorker

        worker = CameraWorker(
            "hltcam01abcd",
            _camera_config("hltcam01abcd"),
            lambda m: None,
        )
        worker._frames_inferred = 1
        worker._last_infer_t = 1000.0
        # No new frame since inference last ran — pending id equals last-inferred id.
        worker._latest_frame_id = 1
        worker._current_inference_frame_id = 1
        # Advance the clock far past the 15 s restart threshold: still must be 0.0.
        assert worker.inference_stalled_for(1000.0) == 0.0
        assert worker.inference_stalled_for(1020.0) == 0.0
        assert worker.inference_stalled_for(1_000_000.0) == 0.0

    def test_inference_thread_alive_false_before_start(self, qapp) -> None:
        from autoptz.engine.camera_worker import CameraWorker

        worker = CameraWorker(
            "hltcam01abcd",
            _camera_config("hltcam01abcd"),
            lambda m: None,
        )
        assert worker._inference_thread is None
        assert worker.inference_thread_alive() is False


# ── _scan_worker_health ───────────────────────────────────────────────────────


class TestScanWorkerHealth:
    def test_healthy_worker_is_not_touched(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            assert original._alive is True
            # A healthy worker emits telemetry; record a fresh timestamp on the
            # SAME (injected) clock as the scan.  Without this the hang check
            # compares the injected ``now`` against the worker's real-monotonic
            # spawn_t — fine on a high-uptime dev box, but on a low-uptime CI
            # runner (monotonic < 1000) the worker looks "stale" and is wrongly
            # respawned (production uses one real clock, so it's unaffected).
            sup._last_telemetry_t[cid] = 1000.0
            sup._scan_worker_health(1000.0)
            assert len(factory_log) == 1  # no new worker built
            assert original.stop_calls == 0
        finally:
            sup.stop()

    def test_dead_worker_is_respawned(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            factory_log[0]._alive = False
            sup._scan_worker_health(1000.0)
            # Factory called again → a second worker in the log.
            assert len(factory_log) == 2
            assert sup.has_worker(cid)
            assert sup._workers[cid] is factory_log[1]
        finally:
            sup.stop()

    def test_backoff_prevents_immediate_second_respawn(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            factory_log[0]._alive = False
            now = 1000.0
            sup._scan_worker_health(now)  # first respawn at t=1000
            assert len(factory_log) == 2
            # Simulate the new worker dying too.
            factory_log[1]._alive = False
            # Second scan immediately (before backoff expires) → no extra spawn.
            sup._scan_worker_health(now + 0.1)
            assert len(factory_log) == 2  # still just 2 workers
        finally:
            sup.stop()

    def test_backoff_expires_and_allows_respawn(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            factory_log[0]._alive = False
            now = 1000.0
            sup._scan_worker_health(now)  # attempt 1 → factory_log[1] spawned
            assert len(factory_log) == 2
            # Kill the second worker.
            factory_log[1]._alive = False
            # Advance past the base backoff window (1 s).
            sup._scan_worker_health(now + _BASE_BACKOFF_S + 0.1)  # attempt 2
            assert len(factory_log) == 3  # original + 2 respawns
        finally:
            sup.stop()

    def test_cap_stops_respawning(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            now = 1000.0
            # Drive through all MAX attempts.
            for _attempt in range(_MAX_RESTART_ATTEMPTS):
                sup._workers[cid]._alive = False
                # Advance well past any back-off.
                now += _MAX_BACKOFF_S + 1.0
                sup._scan_worker_health(now)

            # After the cap is reached the last attempt is logged but no new
            # worker is spawned; the restart_state stays at MAX.
            count_at_cap = len(factory_log)
            # Kill any remaining worker (may or may not still be registered).
            if cid in sup._workers:
                sup._workers[cid]._alive = False
            now += _MAX_BACKOFF_S + 1.0
            sup._scan_worker_health(now)
            assert len(factory_log) == count_at_cap
        finally:
            sup.stop()

    def test_remove_camera_clears_restart_state(self, qapp) -> None:
        from autoptz.engine.runtime.messages import RemoveCameraCmd

        sup, client, factory_log, cid = _build(qapp)
        try:
            # Seed some restart state.
            sup._restart_state[cid] = (2, 9999.0, False)
            # Route a RemoveCamera command.
            sup._on_remove_camera(RemoveCameraCmd(camera_id=cid))
            assert cid not in sup._restart_state
            assert not sup.has_worker(cid)
        finally:
            sup.stop()

    def test_exponential_backoff_values(self, qapp) -> None:
        """Back-off doubles each attempt, capped at _MAX_BACKOFF_S."""
        sup, client, factory_log, cid = _build(qapp)
        try:
            now = 1000.0
            for i in range(_MAX_RESTART_ATTEMPTS - 1):
                sup._workers[cid]._alive = False
                # Advance well past previous back-off so this attempt always fires.
                now += _MAX_BACKOFF_S + 1.0
                sup._scan_worker_health(now)
                attempts, next_t, _f = sup._restart_state.get(cid, (0, now, False))
                expected_backoff = min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * (2**i))
                assert abs((next_t - now) - expected_backoff) < 0.01, (
                    f"attempt {i + 1}: expected backoff {expected_backoff}, got {next_t - now}"
                )
        finally:
            sup.stop()

    def test_reset_on_recovery(self, qapp) -> None:
        """A recovered worker resets backoff state so re-crash starts at attempt 1.

        Scenario:
        1. Worker crashes twice → _restart_state[cid] = (2, <future>).
        2. Re-spawned worker is healthy on the next scan → state cleared.
        3. Worker crashes again → next_allowed_t reflects the 1 s base delay
           (attempt 1), NOT a continued-from-2 delay.
        """
        sup, client, factory_log, cid = _build(qapp)
        try:
            now = 1000.0

            # --- Phase 1: crash twice so attempts accumulate to 2 ---
            # First crash + respawn.
            sup._workers[cid]._alive = False
            now += _MAX_BACKOFF_S + 1.0
            sup._scan_worker_health(now)  # attempt 1 → factory_log[1] spawned
            assert len(factory_log) == 2

            # Kill the second worker immediately.
            factory_log[1]._alive = False
            now += _MAX_BACKOFF_S + 1.0
            sup._scan_worker_health(now)  # attempt 2 → factory_log[2] spawned
            assert len(factory_log) == 3
            assert sup._restart_state[cid][0] == 2  # two attempts recorded

            # --- Phase 2: new worker is healthy → state cleared ---
            # factory_log[2] starts alive (default); record fresh telemetry on the
            # injected clock so the hang check doesn't false-positive on a
            # low-uptime CI runner (see test_healthy_worker_is_not_touched).
            sup._last_telemetry_t[cid] = now + 1.0
            sup._scan_worker_health(now + 1.0)
            assert cid not in sup._restart_state  # recovery cleared the slate

            # --- Phase 3: crash again → backoff starts over from attempt 1 ---
            sup._workers[cid]._alive = False
            t_crash = now + 2.0
            sup._scan_worker_health(t_crash)  # attempt 1 fresh start
            attempts, next_allowed_t, _failed = sup._restart_state[cid]
            assert attempts == 1
            # next_allowed_t should reflect _BASE_BACKOFF_S (1 s), not a 2^2 delay.
            assert abs((next_allowed_t - t_crash) - _BASE_BACKOFF_S) < 0.01, (
                f"expected base backoff {_BASE_BACKOFF_S} s after recovery, "
                f"got {next_allowed_t - t_crash:.3f} s"
            )
        finally:
            sup.stop()


class TestWorkerTelemetryTracking:
    def test_telemetry_callback_stamps_last_seen_and_forwards(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            # The wrapped callback the factory received is the supervisor's wrapper,
            # not push_telemetry directly.
            wrapped = factory_log[0].on_telemetry
            before = sup._last_telemetry_t.get(cid)
            # push_telemetry needs a real TelemetryMsg; build a minimal one.
            from autoptz.engine.runtime.messages import TelemetryMsg

            wrapped(TelemetryMsg(camera_id=cid, seq=0))
            after = sup._last_telemetry_t.get(cid)
            assert before is None
            assert after is not None and after > 0.0
        finally:
            sup.stop()

    def test_spawn_records_spawn_time(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            assert cid in sup._spawn_t
            assert sup._spawn_t[cid] > 0.0
        finally:
            sup.stop()

    def test_remove_camera_clears_telemetry_and_spawn_state(self, qapp) -> None:
        from autoptz.engine.runtime.messages import RemoveCameraCmd

        sup, client, factory_log, cid = _build(qapp)
        try:
            sup._last_telemetry_t[cid] = 123.0
            sup._on_remove_camera(RemoveCameraCmd(camera_id=cid))
            assert cid not in sup._last_telemetry_t
            assert cid not in sup._spawn_t
        finally:
            sup.stop()


class TestHangDetection:
    def test_no_hang_within_warmup_grace(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            # Spawn just happened; telemetry never arrived. Within warmup → not hung.
            spawn_t = sup._spawn_t[cid]
            assert sup._worker_hung(cid, spawn_t + _WORKER_WARMUP_GRACE_S - 0.1) is False
        finally:
            sup.stop()

    def test_hung_when_no_telemetry_past_warmup(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            spawn_t = sup._spawn_t[cid]
            # Past warmup, still no telemetry → hung.
            now = spawn_t + _WORKER_WARMUP_GRACE_S + _WORKER_HANG_S + 0.1
            assert sup._worker_hung(cid, now) is True
        finally:
            sup.stop()

    def test_hung_when_telemetry_goes_stale(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            spawn_t = sup._spawn_t[cid]
            past_warmup = spawn_t + _WORKER_WARMUP_GRACE_S + 1.0
            sup._last_telemetry_t[cid] = past_warmup  # fresh telemetry arrives
            assert sup._worker_hung(cid, past_warmup + _WORKER_HANG_S - 0.1) is False
            assert sup._worker_hung(cid, past_warmup + _WORKER_HANG_S + 0.1) is True
        finally:
            sup.stop()

    def test_alive_but_hung_worker_is_respawned(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            assert original._alive is True  # alive, not dead
            # Force "past warmup, telemetry stale".
            sup._spawn_t[cid] = 0.0
            sup._last_telemetry_t[cid] = 0.0
            now = _WORKER_WARMUP_GRACE_S + _WORKER_HANG_S + 100.0
            sup._scan_worker_health(now)
            assert len(factory_log) == 2  # respawned despite being alive
            assert original.stop_calls == 1  # old (hung) worker was stopped
            assert sup._workers[cid] is factory_log[1]

            # Guard: a freshly-respawned worker must NOT be immediately re-hung.
            # Simulate the new worker emitting telemetry so it looks healthy, then
            # run one more scan a small delta later (still within the warmup grace
            # window) — no third spawn should occur.
            sup._last_telemetry_t[cid] = now  # fresh telemetry from the new worker
            sup._scan_worker_health(now + 0.5)
            assert len(factory_log) == 2, (
                "New worker was immediately re-respawned — hang-detection thrash detected"
            )
        finally:
            sup.stop()

    def test_healthy_streaming_worker_not_respawned(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            sup._spawn_t[cid] = 0.0
            now = _WORKER_WARMUP_GRACE_S + 100.0
            sup._last_telemetry_t[cid] = now - 0.05  # fresh telemetry
            sup._scan_worker_health(now)
            assert len(factory_log) == 1  # untouched
        finally:
            sup.stop()


class TestInferenceDeathDetection:
    """R-2: capture thread alive but inference thread dead/stalled → restart.

    Complementary to ``TestHangDetection`` (capture-thread staleness). These
    drive ``_worker_inference_dead`` / ``_scan_worker_health`` with a worker
    that is ``is_alive() == True`` (capture healthy) but whose inference-thread
    health surface reports a stall or a dead thread object.
    """

    def test_stall_past_threshold_triggers_restart(self, qapp) -> None:
        """(a) inference stall age past 15 s while capture alive → restarted."""
        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            assert original._alive is True
            # Anchor spawn_t to the injected clock (past warmup grace) — see
            # test_healthy_worker_is_not_touched for why: real spawn_t is a
            # real-monotonic timestamp, which desyncs from the injected `now`
            # used here on a low-uptime CI runner.
            sup._spawn_t[cid] = 1000.0 - _WORKER_WARMUP_GRACE_S - 1.0
            # Fresh capture-side telemetry so _worker_hung stays False — isolates
            # the inference-death predicate.
            sup._last_telemetry_t[cid] = 1000.0
            original._infer_stall_s = _INFER_RESTART_S + 0.1
            sup._scan_worker_health(1000.0)
            assert len(factory_log) == 2  # respawned
            assert original.stop_calls == 1
            assert sup._workers[cid] is factory_log[1]
        finally:
            sup.stop()

    def test_stall_below_threshold_untouched(self, qapp) -> None:
        """(b) stall below threshold → untouched."""
        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            sup._spawn_t[cid] = 1000.0 - _WORKER_WARMUP_GRACE_S - 1.0
            sup._last_telemetry_t[cid] = 1000.0
            original._infer_stall_s = _INFER_RESTART_S - 0.1
            sup._scan_worker_health(1000.0)
            assert len(factory_log) == 1  # untouched
            assert original.stop_calls == 0
        finally:
            sup.stop()

    def test_inference_thread_dead_triggers_restart(self, qapp) -> None:
        """(c) inference thread dead + capture alive → restarted."""
        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            sup._spawn_t[cid] = 1000.0 - _WORKER_WARMUP_GRACE_S - 1.0
            sup._last_telemetry_t[cid] = 1000.0
            original._infer_thread_alive = False
            sup._scan_worker_health(1000.0)
            assert len(factory_log) == 2  # respawned
            assert original.stop_calls == 1
            assert sup._workers[cid] is factory_log[1]
        finally:
            sup.stop()

    def test_restart_budget_still_capped_at_max(self, qapp) -> None:
        """(d) restart budget still capped at 5 — no change to budget semantics."""
        sup, client, factory_log, cid = _build(qapp)
        try:
            now = 1000.0
            for _attempt in range(_MAX_RESTART_ATTEMPTS):
                sup._workers[cid]._infer_thread_alive = False
                # Re-anchor spawn_t on the injected clock every iteration: each
                # respawn resets it to real time.monotonic() (see _spawn_worker),
                # which would otherwise desync from the injected `now` again and
                # spuriously suppress the predicate under the warmup grace.
                sup._spawn_t[cid] = now - _WORKER_WARMUP_GRACE_S - 1.0
                sup._last_telemetry_t[cid] = now
                now += _MAX_BACKOFF_S + 1.0
                sup._scan_worker_health(now)

            count_at_cap = len(factory_log)
            if cid in sup._workers:
                sup._workers[cid]._infer_thread_alive = False
                sup._spawn_t[cid] = now - _WORKER_WARMUP_GRACE_S - 1.0
                sup._last_telemetry_t[cid] = now
            now += _MAX_BACKOFF_S + 1.0
            sup._scan_worker_health(now)
            assert len(factory_log) == count_at_cap
            assert sup.is_camera_failed(cid) is True
        finally:
            sup.stop()

    def test_healthy_worker_untouched(self, qapp) -> None:
        """(e) healthy worker (capture alive, inference alive, no stall) → untouched."""
        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            sup._last_telemetry_t[cid] = 1000.0
            original._infer_stall_s = 0.0
            original._infer_thread_alive = True
            sup._scan_worker_health(1000.0)
            assert len(factory_log) == 1  # untouched
            assert original.stop_calls == 0
        finally:
            sup.stop()

    def test_warmup_grace_suppresses_inference_dead_predicate(self, qapp) -> None:
        """Important-1: a freshly (re)spawned worker must not be flagged as
        inference-dead during the spawn-time warmup grace, mirroring
        ``_worker_hung``'s existing grace handling.
        """
        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            # Spawn "just happened" on the injected clock; inference already
            # looks stalled/dead, but we are still within warmup grace.
            sup._spawn_t[cid] = 1000.0
            sup._last_telemetry_t[cid] = 1000.0
            original._infer_stall_s = _INFER_RESTART_S + 100.0
            original._infer_thread_alive = False
            now = 1000.0 + _WORKER_WARMUP_GRACE_S - 0.1
            assert sup._worker_inference_dead(cid, original, now) is False
            sup._scan_worker_health(now)
            assert len(factory_log) == 1  # untouched — still within warmup grace
        finally:
            sup.stop()

    def test_inference_dead_predicate_fires_once_warmup_grace_elapses(self, qapp) -> None:
        """Sanity check for the warmup-grace test above: once grace elapses the
        same stalled worker IS flagged (grace only delays, never masks forever).
        """
        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            sup._spawn_t[cid] = 1000.0
            sup._last_telemetry_t[cid] = 1000.0
            original._infer_stall_s = _INFER_RESTART_S + 100.0
            now = 1000.0 + _WORKER_WARMUP_GRACE_S + 0.1
            assert sup._worker_inference_dead(cid, original, now) is True
        finally:
            sup.stop()

    def test_inference_dead_check_raising_is_treated_as_not_dead(self, qapp, caplog) -> None:
        """Important-2: the try/except around the predicate call site
        (``_scan_worker_health``) — if ``inference_stalled_for`` raises, the
        worker must be treated as NOT inference-dead (debug-logged), not
        restarted.
        """
        import logging

        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            sup._spawn_t[cid] = 1000.0 - _WORKER_WARMUP_GRACE_S - 1.0
            sup._last_telemetry_t[cid] = 1000.0

            def _raise(now: float) -> float:
                raise RuntimeError("boom")

            original.inference_stalled_for = _raise  # type: ignore[method-assign]
            with caplog.at_level(logging.DEBUG):
                sup._scan_worker_health(1000.0)
            assert len(factory_log) == 1  # not restarted
            assert original.stop_calls == 0
            assert any("inference-death check raised" in r.getMessage() for r in caplog.records)
        finally:
            sup.stop()

    def test_process_worker_not_subject_to_inference_predicate(self, qapp) -> None:
        """Process-per-camera handles have no inference_stalled_for/thread_alive
        surface — the predicate must not blow up or misfire on them (scoped to
        in-process/threaded workers only, per the process-worker liveness path).
        """
        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()

        class _FakeProcessWorker:
            _is_process_worker = True

            def __init__(self, camera_id, config, on_telemetry) -> None:
                self.camera_id = camera_id
                self._alive = True
                self.stop_calls = 0

            def is_alive(self) -> bool:
                return self._alive

            @property
            def is_running(self) -> bool:
                return self._alive

            def start(self) -> None:
                pass

            def stop(self, timeout: float = 5.0) -> None:
                self.stop_calls += 1
                self._alive = False

        factory_log: list[_FakeProcessWorker] = []

        def factory(camera_id, config, on_tel):
            w = _FakeProcessWorker(camera_id, config, on_tel)
            factory_log.append(w)
            return w

        sup = _make_sup_with_factory(client, factory)
        sup._ensure_identity_service = lambda: None  # type: ignore[method-assign]
        sup._ensure_inference_pool = lambda: None  # type: ignore[method-assign]
        sup.start()
        try:
            sup._last_telemetry_t[cid] = 1000.0
            sup._scan_worker_health(1000.0)  # must not raise
            assert len(factory_log) == 1  # untouched — no inference surface, alive
            assert factory_log[0].stop_calls == 0
        finally:
            sup.stop()


class _FakeModelServerProc:
    """Minimal fake for supervisor._model_server_proc with a settable is_alive()."""

    def __init__(self, *_a, **_k) -> None:  # noqa: ANN002, ANN003
        self._alive = True
        self.start_calls = 0
        self.terminate_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False

    def join(self, timeout: float = 0.0) -> None:  # noqa: ARG002
        pass


class _FakeEvent:
    """Stand-in for ctx.Event() used by both the ready-handshake and the
    stop/down/failed gates. The fake process never runs run_inference_server to
    set "ready" for real, so wait() always reports ready instantly instead of
    blocking the full 30s production timeout; set()/clear()/is_set() behave
    normally so down-gate assertions still work."""

    def __init__(self) -> None:
        self._flag = False

    def set(self) -> None:
        self._flag = True

    def clear(self) -> None:
        self._flag = False

    def is_set(self) -> bool:
        return self._flag

    def wait(self, timeout: float = 0.0) -> bool:  # noqa: ARG002
        return True


class _MsFakeProcessWorker(_HealthFakeWorker):
    """Fake model-server-mode camera worker: flags itself as a process worker (like
    ProcessWorkerHandle) and records refresh_detector_from_pool() calls so tests can
    assert the supervisor invoked the local-fallback seam."""

    _is_process_worker = True

    def __init__(self, camera_id, config, on_telemetry) -> None:  # noqa: ANN001
        super().__init__(camera_id, config, on_telemetry)
        self.refresh_calls = 0

    def refresh_detector_from_pool(self) -> None:
        self.refresh_calls += 1


class TestModelServerHealthScan:
    """R-3: the supervisor's model-server respawn path mirrors the worker
    auto-restart machinery (same backoff constants, same budget/cap shape) instead
    of inventing a new mechanism."""

    def _build_ms(self, qapp, monkeypatch):  # noqa: ANN001
        """Supervisor in model-server mode with a fake process + one fake
        process-worker camera, bypassing the real mp.Process spawn."""
        import multiprocessing as mp

        from autoptz.ui.engine_client import EngineClient

        monkeypatch.setenv("AUTOPTZ_MODEL_SERVER", "1")
        client = EngineClient()
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()

        factory_log: list[_MsFakeProcessWorker] = []

        def factory(camera_id, config, on_tel):  # noqa: ANN001
            w = _MsFakeProcessWorker(camera_id, config, on_tel)
            factory_log.append(w)
            return w

        sup = _make_sup_with_factory(client, factory)
        sup._ensure_identity_service = lambda: None  # type: ignore[method-assign]
        sup._ensure_inference_pool = lambda: None  # type: ignore[method-assign]

        real_ctx = mp.get_context("spawn")
        proc_log: list[_FakeModelServerProc] = []

        class _FakeCtx:
            Queue = staticmethod(real_ctx.Queue)
            Event = staticmethod(_FakeEvent)

            def Process(self, *_a, **_k):  # noqa: ANN002, ANN003, N802
                p = _FakeModelServerProc()
                proc_log.append(p)
                return p

        monkeypatch.setattr(mp, "get_context", lambda *_a, **_k: _FakeCtx())
        # The fake process never actually serves, so don't block start() waiting
        # for the real ready-event handshake.
        monkeypatch.setattr(
            "autoptz.engine.supervisor.Supervisor._ensure_model_server",
            lambda self, camera_ids: _fake_ensure_model_server(self, camera_ids, proc_log),
        )
        sup.start()
        return sup, client, factory_log, proc_log, cid

    def test_dead_model_server_is_respawned(self, qapp, monkeypatch) -> None:
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            assert len(proc_log) == 1
            proc_log[0]._alive = False
            sup._scan_model_server_health(1000.0)
            assert len(proc_log) == 2  # respawned
            assert sup._model_server_proc is proc_log[1]
        finally:
            sup.stop()

    def test_respawn_reuses_same_queues_no_reconstruction(self, qapp, monkeypatch) -> None:
        """(d) Across a respawn the SAME request queue / response queues stay in
        place — clients need no reconstruction, only the server process changes."""
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            req_q_before = sup._infer_req_q
            resp_qs_before = sup._infer_resp_qs
            proc_log[0]._alive = False
            sup._scan_model_server_health(1000.0)
            assert sup._infer_req_q is req_q_before
            assert sup._infer_resp_qs is resp_qs_before
        finally:
            sup.stop()

    def test_backoff_prevents_immediate_second_respawn(self, qapp, monkeypatch) -> None:
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            proc_log[0]._alive = False
            now = 1000.0
            sup._scan_model_server_health(now)
            assert len(proc_log) == 2
            proc_log[1]._alive = False
            sup._scan_model_server_health(now + 0.1)  # still inside backoff window
            assert len(proc_log) == 2
        finally:
            sup.stop()

    def test_backoff_expires_and_allows_respawn(self, qapp, monkeypatch) -> None:
        """Updated for non-blocking respawn (R-3 review fix): a single
        _scan_model_server_health call now only STARTS a respawn attempt (spawn
        deadline pending) instead of synchronously spawning-and-waiting-for-ready
        in one call. ``_FakeEvent`` here never actually signals ready (nothing
        drives a real server loop that would call ``.set()``), so — like a
        genuinely stuck child — the in-flight attempt only resolves once its
        spawn deadline passes; see TestModelServerRespawnNonBlocking for
        dedicated polling-across-ticks coverage including the ready-succeeds
        case. This test now walks a full spawn-timeout -> backoff -> respawn
        cycle instead of asserting a same-tick third spawn."""
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            proc_log[0]._alive = False
            now = 1000.0
            sup._scan_model_server_health(now)  # attempt 1: starts spawning
            assert len(proc_log) == 2

            # Spawn deadline passes without a ready signal -> attempt 1 fails and
            # enters its backoff window.
            now += _MS_SPAWN_TIMEOUT_S + 0.1
            sup._scan_model_server_health(now)
            assert len(proc_log) == 2  # no new process yet — still in backoff
            assert sup._ms_spawn_deadline is None

            sup._scan_model_server_health(now + 0.01)  # still inside backoff window
            assert len(proc_log) == 2  # untouched

            sup._scan_model_server_health(now + _BASE_BACKOFF_S + 0.1)
            assert len(proc_log) == 3  # backoff expired — attempt 2 starts
        finally:
            sup.stop()

    def test_budget_exhaustion_sets_failed_flag_and_triggers_worker_fallback(
        self, qapp, monkeypatch
    ) -> None:
        """(c) After restart attempts are exhausted, model_server_failed() is True
        and every model-server-mode worker's refresh_detector_from_pool() fires
        (the seam that lets RemotePool swap in a local detector).

        Updated for non-blocking respawn (R-3 review fix): starting a respawn and
        observing its outcome are now two separate ticks, and ``_FakeEvent``
        never actually signals ready, so each cycle below ticks twice: once to
        start the respawn, once more past the spawn deadline to resolve it as a
        failed attempt (mirroring a server that never comes up) before advancing
        past the backoff window for the next cycle."""
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            now = 1000.0
            for _attempt in range(_MAX_RESTART_ATTEMPTS):
                sup._model_server_proc._alive = False
                now += _MAX_BACKOFF_S + 1.0
                sup._scan_model_server_health(now)  # starts the respawn attempt
                now += _MS_SPAWN_TIMEOUT_S + 0.1
                sup._scan_model_server_health(now)  # deadline exceeded -> failed

            assert sup.model_server_failed() is True
            assert factory_log[0].refresh_calls == 1  # fired exactly once, at exhaustion
        finally:
            sup.stop()

    def test_healthy_server_is_not_touched(self, qapp, monkeypatch) -> None:
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            sup._scan_model_server_health(1000.0)
            assert len(proc_log) == 1  # no respawn
            assert sup.model_server_failed() is False
        finally:
            sup.stop()

    def test_not_enabled_is_a_noop(self, qapp, monkeypatch) -> None:
        """With AUTOPTZ_MODEL_SERVER off, scanning must never try to touch a
        model-server process that was never started."""
        from autoptz.ui.engine_client import EngineClient

        monkeypatch.delenv("AUTOPTZ_MODEL_SERVER", raising=False)
        client = EngineClient()
        sup = _make_sup_with_factory(client, lambda cid, cfg, tel: _HealthFakeWorker(cid, cfg, tel))
        sup._ensure_identity_service = lambda: None  # type: ignore[method-assign]
        sup._ensure_inference_pool = lambda: None  # type: ignore[method-assign]
        sup._running = True
        sup._scan_model_server_health(1000.0)  # must not raise
        assert sup._model_server_proc is None
        assert sup.model_server_failed() is False


class _AssertNoBlockingWaitEvent:
    """Stand-in for ctx.Event() that records any ``.wait(timeout=...)`` call made
    with a nonzero timeout, so a test can assert the tick path never blocked —
    the health-scan/tick thread must only ever poll ``is_set()``.

    Deliberately does NOT raise from inside ``wait()``: production code wraps the
    old blocking respawn in a broad ``except Exception``, which would silently
    swallow an AssertionError raised here and give a false-negative (test passes
    even though the code blocked in real life, where wait() actually sleeps
    instead of raising). Recording to a class-level list and asserting on it
    from the test body survives that broad except clause. ``wait(timeout=0)`` (a
    non-blocking check) is tolerated and not recorded."""

    blocking_wait_calls: list[float] = []

    def __init__(self) -> None:
        self._flag = False

    def set(self) -> None:
        self._flag = True

    def clear(self) -> None:
        self._flag = False

    def is_set(self) -> bool:
        return self._flag

    def wait(self, timeout: float = 0.0) -> bool:
        if timeout:
            _AssertNoBlockingWaitEvent.blocking_wait_calls.append(timeout)
        return self._flag


class _ControllableModelServerProc(_FakeModelServerProc):
    """Fake model-server process whose is_alive() can be flipped after start()."""


class TestModelServerRespawnNonBlocking:
    """IMPORTANT-1: _respawn_model_server (called from _scan_model_server_health,
    which tick() drives on the GUI thread in the shipped app) must never block —
    `proc.start()` + record a spawn deadline, then poll `ready.is_set()` on later
    ticks instead of `ready.wait(timeout=...)`. All tests here use a synthetic
    monotonic clock (`now`) — no real sleeps."""

    def _build_ms(self, qapp, monkeypatch, event_cls=_AssertNoBlockingWaitEvent):  # noqa: ANN001
        """Same shape as TestModelServerHealthScan._build_ms but with a pluggable
        Event class so this suite can prove the tick path never awaits it."""
        import multiprocessing as mp

        from autoptz.ui.engine_client import EngineClient

        _AssertNoBlockingWaitEvent.blocking_wait_calls.clear()
        monkeypatch.setenv("AUTOPTZ_MODEL_SERVER", "1")
        client = EngineClient()
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()

        factory_log: list[_MsFakeProcessWorker] = []

        def factory(camera_id, config, on_tel):  # noqa: ANN001
            w = _MsFakeProcessWorker(camera_id, config, on_tel)
            factory_log.append(w)
            return w

        sup = _make_sup_with_factory(client, factory)
        sup._ensure_identity_service = lambda: None  # type: ignore[method-assign]
        sup._ensure_inference_pool = lambda: None  # type: ignore[method-assign]

        real_ctx = mp.get_context("spawn")
        proc_log: list[_ControllableModelServerProc] = []

        class _FakeCtx:
            Queue = staticmethod(real_ctx.Queue)
            Event = staticmethod(event_cls)

            def Process(self, *_a, **_k):  # noqa: ANN002, ANN003, N802
                p = _ControllableModelServerProc()
                proc_log.append(p)
                return p

        monkeypatch.setattr(mp, "get_context", lambda *_a, **_k: _FakeCtx())
        monkeypatch.setattr(
            "autoptz.engine.supervisor.Supervisor._ensure_model_server",
            lambda self, camera_ids: _fake_ensure_model_server(self, camera_ids, proc_log),
        )
        sup.start()
        return sup, client, factory_log, proc_log, cid

    def test_tick_path_never_calls_blocking_wait(self, qapp, monkeypatch) -> None:
        """RED against fc6eee1: the old _respawn_model_server called
        ready.wait(timeout=30.0) synchronously from the scan. With the fix, the
        scan only starts the child and polls is_set() on later ticks, so no
        nonzero-timeout wait() call is ever recorded.

        Asserts on the event's call recorder rather than letting wait() raise,
        because the old code wraps the whole respawn in a broad
        ``except Exception`` that would otherwise swallow an in-wait assertion
        and produce a false-negative RED."""
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            proc_log[0]._alive = False
            sup._scan_model_server_health(1000.0)
            assert len(proc_log) == 2  # respawn attempt was started
            assert _AssertNoBlockingWaitEvent.blocking_wait_calls == []
        finally:
            sup.stop()

    def test_spawn_pending_tick_is_a_noop(self, qapp, monkeypatch) -> None:
        """While a respawn is in flight and not yet at its deadline, subsequent
        scan ticks must not start another child or touch restart state."""
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            proc_log[0]._alive = False
            now = 1000.0
            sup._scan_model_server_health(now)
            assert len(proc_log) == 2
            assert sup._ms_spawn_deadline == now + _MS_SPAWN_TIMEOUT_S

            # Well before the deadline — should be a pure no-op.
            sup._scan_model_server_health(now + 1.0)
            assert len(proc_log) == 2  # no new spawn
            assert sup._ms_spawn_deadline == now + _MS_SPAWN_TIMEOUT_S  # unchanged
            assert sup._ms_restart_state == (1, now + _BASE_BACKOFF_S, False)
        finally:
            sup.stop()

    def test_ready_on_later_tick_succeeds(self, qapp, monkeypatch) -> None:
        """Ready-signal arriving on a LATER tick (not the same one that started
        the spawn) must clear the gate and reset attempts, mirroring the old
        synchronous-success semantics — just detected asynchronously."""
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            proc_log[0]._alive = False
            now = 1000.0
            sup._scan_model_server_health(now)
            assert sup._ms_spawn_deadline is not None
            assert sup._model_server_down.is_set() is True  # fast-fail gate held

            # Simulate the child signalling ready sometime later, well before the
            # spawn deadline.
            sup._model_server_proc._alive = True
            ready_ev = sup._ms_ready_ev
            ready_ev.set()

            sup._scan_model_server_health(now + 2.0)

            assert sup._ms_spawn_deadline is None
            assert sup._ms_restart_state == (0, 0.0, False)
            assert sup._model_server_down.is_set() is False  # gate cleared
        finally:
            sup.stop()

    def test_deadline_exceeded_counts_as_failed_attempt_with_backoff(
        self, qapp, monkeypatch
    ) -> None:
        """If the child never signals ready before the spawn deadline, that tick
        must count as a failed attempt (existing backoff/budget accounting) and
        kill the stuck child — not hang waiting for it."""
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            proc_log[0]._alive = False
            now = 1000.0
            sup._scan_model_server_health(now)
            assert sup._ms_spawn_deadline == now + _MS_SPAWN_TIMEOUT_S
            stuck_proc = sup._model_server_proc

            # Never signals ready; advance past the spawn deadline.
            sup._scan_model_server_health(now + _MS_SPAWN_TIMEOUT_S + 0.1)

            assert sup._ms_spawn_deadline is None
            assert stuck_proc.terminate_calls == 1  # stuck child was killed
            attempts, next_allowed_t, failed = sup._ms_restart_state
            # The attempt was already counted (attempts=1) when the respawn was
            # initiated — mirrors the old synchronous code, which incremented
            # before the (blocking) wait. A timed-out spawn does not double-count.
            assert attempts == 1
            assert next_allowed_t > now + _MS_SPAWN_TIMEOUT_S  # backoff window set
            assert failed is False
            assert sup._model_server_down.is_set() is True  # gate still held
        finally:
            sup.stop()

    def test_gate_stays_set_across_spawning_and_backoff_window(self, qapp, monkeypatch) -> None:
        """The down-gate must remain set continuously from the moment the server
        is first found dead through spawning AND the subsequent backoff wait —
        clients must keep fast-failing the whole time, never just during one
        phase."""
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            proc_log[0]._alive = False
            now = 1000.0
            sup._scan_model_server_health(now)
            assert sup._model_server_down.is_set() is True  # spawning phase

            # Deadline exceeded -> failed attempt -> now in backoff phase.
            now += _MS_SPAWN_TIMEOUT_S + 0.1
            sup._scan_model_server_health(now)
            assert sup._ms_spawn_deadline is None
            assert sup._model_server_down.is_set() is True  # backoff phase

            # Still inside the backoff window — must stay a no-op, gate still set.
            now += 0.1
            sup._scan_model_server_health(now)
            assert sup._model_server_down.is_set() is True
        finally:
            sup.stop()

    def test_deadline_exceeded_at_budget_exhaustion_triggers_fallback(
        self, qapp, monkeypatch
    ) -> None:
        """A spawn that times out on the FINAL allowed attempt must exhaust the
        budget exactly like the old synchronous failure path did: latch
        model_server_failed() and fire the worker fallback."""
        sup, client, factory_log, proc_log, cid = self._build_ms(qapp, monkeypatch)
        try:
            now = 1000.0
            proc_log[0]._alive = False
            for _attempt in range(_MAX_RESTART_ATTEMPTS - 1):
                sup._scan_model_server_health(now)
                now += _MS_SPAWN_TIMEOUT_S + 0.1  # spawn never signals ready
                sup._scan_model_server_health(now)  # deadline exceeded -> failed
                now += _MAX_BACKOFF_S + 1.0  # clear backoff before next attempt
                if sup._model_server_proc is not None:
                    sup._model_server_proc._alive = False

            assert sup.model_server_failed() is False  # not yet — one more to go

            sup._scan_model_server_health(now)  # final attempt starts spawning
            now += _MS_SPAWN_TIMEOUT_S + 0.1
            sup._scan_model_server_health(now)  # final deadline exceeded -> exhausted

            assert sup.model_server_failed() is True
            assert factory_log[0].refresh_calls == 1
            assert sup._model_server_down.is_set() is True
        finally:
            sup.stop()


def _fake_ensure_model_server(sup, camera_ids, proc_log) -> None:  # noqa: ANN001
    """Deterministic stand-in for Supervisor._ensure_model_server: builds the same
    queue/event handles but skips the real ready-event wait (the fake process never
    serves), so tests can drive _scan_model_server_health directly."""
    import multiprocessing as mp

    from autoptz.engine.runtime.flags import env_model_server

    if not env_model_server() or sup._model_server_proc is not None:
        return
    ctx = mp.get_context("spawn")
    sup._infer_req_q = ctx.Queue()
    sup._infer_resp_qs = {cid: ctx.Queue() for cid in camera_ids}
    sup._model_server_stop = ctx.Event()
    sup._model_server_down = ctx.Event()
    sup._model_server_failed_ev = ctx.Event()
    sup._model_server_camera_ids = list(camera_ids)
    sup._ms_restart_state = (0, 0.0, False)
    sup._model_server_proc = ctx.Process()
    sup._model_server_proc.start()


class TestPermanentFailed:
    def test_failed_flag_and_accessor_set_at_cap(self, qapp, caplog) -> None:
        import logging

        sup, client, factory_log, cid = _build(qapp)
        try:
            now = 1000.0
            with caplog.at_level(logging.ERROR):
                for _ in range(_MAX_RESTART_ATTEMPTS):
                    sup._workers[cid]._alive = False
                    now += _MAX_BACKOFF_S + 1.0
                    sup._scan_worker_health(now)
            assert sup.is_camera_failed(cid) is True
            assert cid in sup.failed_cameras()
            # Exactly one clear permanent-failure ERROR log (not per-scan spam).
            errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
            assert any("permanently failed" in r.getMessage().lower() for r in errors)
            # Run additional scans (still dead, past backoff) — the ERROR must NOT
            # fire again; exactly one "permanently failed" record across all scans.
            now += _MAX_BACKOFF_S + 1.0
            sup._scan_worker_health(now)
            now += _MAX_BACKOFF_S + 1.0
            sup._scan_worker_health(now)
            perm_logs = [
                r for r in caplog.records if "permanently failed" in r.getMessage().lower()
            ]
            assert len(perm_logs) == 1, (
                f"Expected exactly 1 'permanently failed' log, got {len(perm_logs)}"
            )
        finally:
            sup.stop()

    def test_not_failed_before_cap(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            sup._workers[cid]._alive = False
            sup._scan_worker_health(1000.0)  # attempt 1 only
            assert sup.is_camera_failed(cid) is False
            assert sup.failed_cameras() == []
        finally:
            sup.stop()

    def test_remove_clears_failed_state(self, qapp) -> None:
        from autoptz.engine.runtime.messages import RemoveCameraCmd

        sup, client, factory_log, cid = _build(qapp)
        try:
            sup._restart_state[cid] = (_MAX_RESTART_ATTEMPTS, 9999.0, True)
            sup._on_remove_camera(RemoveCameraCmd(camera_id=cid))
            assert sup.is_camera_failed(cid) is False
        finally:
            sup.stop()
