"""Tests for R3' — worker liveness monitoring + auto-restart.

All tests are headless and use no real threads: they drive
``_scan_worker_health(now)`` directly with a synthetic monotonic clock so
backoff / cap behaviour is deterministic.  The supervisor's ``worker_factory``
injection keeps the existing fake-worker pattern from ``test_orchestration.py``.
"""

from __future__ import annotations

from autoptz.engine.supervisor import (
    _BASE_BACKOFF_S,
    _MAX_BACKOFF_S,
    _MAX_RESTART_ATTEMPTS,
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
    """Minimal fake worker with a settable is_alive() result."""

    def __init__(self, camera_id: str, config, on_telemetry) -> None:
        self.camera_id = camera_id
        self.config = config
        self.on_telemetry = on_telemetry
        self.shm_name = f"cam_{camera_id[:8]}_preview"
        self._alive = True
        self.start_calls = 0
        self.stop_calls = 0

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


# ── _scan_worker_health ───────────────────────────────────────────────────────


class TestScanWorkerHealth:
    def test_healthy_worker_is_not_touched(self, qapp) -> None:
        sup, client, factory_log, cid = _build(qapp)
        try:
            original = factory_log[0]
            assert original._alive is True
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
            sup._restart_state[cid] = (2, 9999.0)
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
                attempts, next_t = sup._restart_state.get(cid, (0, now))
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
            # factory_log[2] starts alive (default); a scan sees it healthy.
            sup._scan_worker_health(now + 1.0)
            assert cid not in sup._restart_state  # recovery cleared the slate

            # --- Phase 3: crash again → backoff starts over from attempt 1 ---
            sup._workers[cid]._alive = False
            t_crash = now + 2.0
            sup._scan_worker_health(t_crash)  # attempt 1 fresh start
            attempts, next_allowed_t = sup._restart_state[cid]
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
