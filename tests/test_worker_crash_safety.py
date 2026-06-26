"""Crash-safety tests for the per-camera worker threads.

These drive a real :class:`CameraWorker` with fake frame sources / fake stages
that raise, and assert the loop guards added for crash-safety hold:

  - ``_run`` ALWAYS runs ``_close_resources`` in a ``finally`` even when
    ``_open_resources`` or the loop body raises (no leaked cv2 capture / shm),
    the failure is reported as an ERROR/STOPPED health telemetry, and the
    capture thread exits cleanly instead of dying silently.
  - A single raising ``source.read()`` (or per-iteration body fault) logs a
    throttled warning and the loop CONTINUES rather than killing the thread.
  - ``_inference_loop`` runs ``_stop_appearance_thread`` in a ``finally`` and a
    single raising stage does not kill inference — later frames still produce
    output.

All tests are headless and inject fakes, so no camera, model, or GUI is
required.  They follow the construction patterns in ``test_async_appearance``
and ``test_inference_watchdog``.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from autoptz.config.models import CameraConfig, SourceConfig
from autoptz.engine.camera_worker import CameraWorker
from autoptz.engine.runtime.messages import HealthState, TelemetryMsg


def _cfg(camera_id: str = "cam-crash-1234abcd") -> CameraConfig:
    return CameraConfig(
        id=camera_id,
        name="Crash",
        source=SourceConfig(type="usb", address="usb://0"),
    )


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


class _OkSource:
    """Minimal frame source that opens and yields solid frames."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.reads = 0

    def open(self) -> bool:
        self.opened = True
        return True

    def read(self):  # noqa: ANN202
        self.reads += 1
        return _frame()

    def close(self) -> None:
        self.closed = True


class _ReadRaisesOnceSource(_OkSource):
    """Source whose ``read`` raises on the FIRST call, then succeeds."""

    def read(self):  # noqa: ANN202
        self.reads += 1
        if self.reads == 1:
            raise RuntimeError("decode boom")
        return _frame()


def _worker(source) -> CameraWorker:  # noqa: ANN001
    w = CameraWorker(
        "cam-crash-1234abcd",
        _cfg(),
        on_telemetry=lambda _m: None,
        frame_source=source,
    )
    # Keep the capture-loop tests off the heavy ML path: the inference thread
    # otherwise loads real models (insightface) and starves/serialises the
    # capture loop, making read-count assertions flaky under suite load.
    w._build_inference_stacks = lambda: None  # type: ignore[assignment]
    return w


def _run_until(w: CameraWorker, predicate, *, timeout: float = 5.0) -> threading.Thread:  # noqa: ANN001
    """Run ``_run`` on a thread until ``predicate()`` holds, then stop + join."""
    t = threading.Thread(target=w._run, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    w._stop_event.set()
    w._capture_wake.set()
    w._inference_start.set()
    t.join(timeout=5.0)
    return t


# ── _run: cleanup always runs ────────────────────────────────────────────────


def test_run_closes_resources_when_open_resources_raises() -> None:
    """An exception in ``_open_resources`` must NOT leak: ``_close_resources``
    runs in the ``finally`` and the thread exits cleanly (no silent death)."""
    src = _OkSource()
    w = _worker(src)

    closed = threading.Event()
    orig_close = w._close_resources

    def tracking_close() -> None:
        orig_close()
        closed.set()

    w._close_resources = tracking_close  # type: ignore[assignment]
    w._open_resources = lambda: (_ for _ in ()).throw(RuntimeError("open boom"))  # type: ignore[assignment]

    t = threading.Thread(target=w._run, daemon=True)
    t.start()
    t.join(timeout=5.0)

    assert not t.is_alive(), "capture thread should exit cleanly, not hang"
    assert closed.is_set(), "_close_resources must run even when _open_resources raises"


def test_run_emits_error_health_when_loop_dies() -> None:
    """When the loop body raises fatally, the worker must emit an ERROR/STOPPED
    health telemetry so the UI reflects the failure rather than going dark."""
    src = _OkSource()
    states: list[HealthState] = []

    def on_tel(msg: TelemetryMsg) -> None:
        states.append(msg.health.state)

    w = CameraWorker("cam-crash-1234abcd", _cfg(), on_telemetry=on_tel, frame_source=src)
    # Make the loop body raise fatally on the first iteration.
    w._open_resources = lambda: (_ for _ in ()).throw(RuntimeError("fatal boom"))  # type: ignore[assignment]

    t = threading.Thread(target=w._run, daemon=True)
    t.start()
    t.join(timeout=5.0)

    assert not t.is_alive()
    assert states, "expected at least one telemetry emit on failure"
    assert any(s in (HealthState.ERROR, HealthState.STOPPED) for s in states), (
        f"expected ERROR/STOPPED health, got {states}"
    )


def test_run_per_iteration_guard_survives_single_bad_read() -> None:
    """A single raising ``source.read()`` is handled by the read guard (already
    present) — the loop keeps going and the thread stays alive until stopped."""
    src = _ReadRaisesOnceSource()
    w = _worker(src)

    # The first read raises; the loop must continue and read again.
    t = _run_until(w, lambda: src.reads >= 3)

    assert not t.is_alive()
    # It read more than once → it did not die on the first raising read.
    assert src.reads > 1, "loop should have continued past the raising read"
    assert src.closed, "_close_resources should release the source on shutdown"


def test_run_normal_shutdown_closes_resources() -> None:
    """Happy path: a clean stop still releases resources (finally not regressed)."""
    src = _OkSource()
    w = _worker(src)

    t = _run_until(w, lambda: src.reads >= 2)

    assert not t.is_alive()
    assert src.opened
    assert src.closed
    assert src.reads >= 1


# ── _inference_loop: survives a raising stage + cleans up ────────────────────


def test_inference_loop_survives_single_raising_stage() -> None:
    """A stage that raises once then succeeds must NOT kill the inference loop —
    later frames still produce output (per-iteration guard)."""
    w = _worker(_OkSource())

    outputs: list[int] = []
    calls = {"n": 0}
    gate = threading.Event()

    def flaky_track(frame):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("stage boom")
        outputs.append(calls["n"])
        gate.set()
        return []

    # Stub the heavy build + the post-track work so we isolate the stage guard.
    w._maybe_track = flaky_track  # type: ignore[assignment]
    w._build_inference_stacks = lambda: None  # type: ignore[assignment]
    w._apply_target_lock = lambda *a, **k: None  # type: ignore[assignment]
    w._drive_ptz_auto = lambda *a, **k: None  # type: ignore[assignment]
    w._annotate_target_aim = lambda *a, **k: None  # type: ignore[assignment]
    w._update_ego_motion = lambda *a, **k: None  # type: ignore[assignment]
    w._async_appearance = False
    w._maybe_reid_recover = lambda *a, **k: None  # type: ignore[assignment]
    w._maybe_identify = lambda *a, **k: None  # type: ignore[assignment]
    w._maybe_estimate_pose_overlay = lambda *a, **k: None  # type: ignore[assignment]

    w._inference_start.set()  # skip the wait-for-start gate
    t = threading.Thread(target=w._inference_loop, daemon=True)
    t.start()
    try:
        # Feed frame 1 → raises; feed frame 2 → succeeds.
        for _ in range(1, 6):
            with w._frame_lock:
                w._latest_frame = _frame()
                w._latest_frame_id += 1
            w._frame_ready.set()
            time.sleep(0.05)
        assert gate.wait(2.0), "inference loop died after a raising stage"
    finally:
        w._stop_event.set()
        w._frame_ready.set()
        t.join(timeout=5.0)

    assert not t.is_alive()
    assert outputs, "later frames should still produce output after the raising stage"


def test_inference_loop_stops_appearance_thread_on_fatal_exit() -> None:
    """If the inference loop dies fatally, ``_stop_appearance_thread`` must run in
    the ``finally`` so the appearance thread is never orphaned."""
    w = _worker(_OkSource())

    stopped = threading.Event()
    orig_stop = w._stop_appearance_thread

    def tracking_stop() -> None:
        orig_stop()
        stopped.set()

    w._stop_appearance_thread = tracking_stop  # type: ignore[assignment]
    # Make the loop setup itself raise fatally (outside the per-iteration guard).
    w._build_inference_stacks = lambda: (_ for _ in ()).throw(RuntimeError("build boom"))  # type: ignore[assignment]

    w._inference_start.set()
    t = threading.Thread(target=w._inference_loop, daemon=True)
    t.start()
    t.join(timeout=5.0)

    assert not t.is_alive(), "inference thread should exit cleanly on a fatal error"
    assert stopped.is_set(), "_stop_appearance_thread must run in the finally"


# ── _infer_last_error: stale-error clearing + dead-thread visibility ──────────


class _FakeDetector:
    """Fake YOLO detector whose ``detect`` raises on the first N calls."""

    def __init__(self, raise_calls: int = 0) -> None:
        self.calls = 0
        self.raise_calls = raise_calls

    def detect(self, frame):  # noqa: ANN001, ANN202
        self.calls += 1
        if self.calls <= self.raise_calls:
            raise RuntimeError("detect boom")
        return []  # no detections → empty, healthy tick


class _FakeTracker:
    def update(self, detections, frame, fps=30.0):  # noqa: ANN001, ANN202, ANN001
        return []


class _FakeDetectStack:
    def __init__(self, raise_calls: int = 0) -> None:
        self.detector = _FakeDetector(raise_calls)
        self.tracker = _FakeTracker()


def test_transient_detect_failure_does_not_pin_stale_error() -> None:
    """Finding #1: a transient detect/track failure sets ``_infer_last_error``,
    but the NEXT successful tick must CLEAR it — otherwise the stale error gets
    attached to every later healthy telemetry and the camera looks permanently
    faulted long after it recovered."""
    w = _worker(_OkSource())
    # First call raises; subsequent calls succeed.
    w._detect = _FakeDetectStack(raise_calls=1)  # type: ignore[assignment]

    # Tick 1: detect raises → error is stashed for the Camera Info panel.
    out1 = w._maybe_track(_frame())
    assert out1 == []
    assert w._infer_last_error is not None, "transient failure should stash an error"

    # Tick 2: detect succeeds → the stale error MUST be cleared.
    out2 = w._maybe_track(_frame())
    assert out2 == []
    assert w._infer_last_error is None, (
        "a successful tick must clear the stale error so later healthy telemetry "
        "does not report the camera as permanently faulted"
    )


def test_emit_telemetry_does_not_carry_cleared_error() -> None:
    """End-to-end of finding #1: after a transient failure followed by a healthy
    tick, ``_emit_telemetry`` must NOT attach the (now cleared) stale error."""
    captured: list[TelemetryMsg] = []
    w = CameraWorker(
        "cam-crash-1234abcd",
        _cfg(),
        on_telemetry=captured.append,
        frame_source=_OkSource(),
    )
    w._detect = _FakeDetectStack(raise_calls=1)  # type: ignore[assignment]

    # Transient failure then recovery.
    w._maybe_track(_frame())
    assert w._infer_last_error is not None
    w._maybe_track(_frame())
    assert w._infer_last_error is None

    w._emit_telemetry(tracks=[], health=HealthState.OK, last_error=None)

    assert captured, "expected a telemetry emit"
    assert captured[-1].health.last_error is None, (
        "healthy telemetry must not carry the cleared transient error"
    )


def test_inference_loop_fatal_exit_surfaces_error_for_capture_telemetry() -> None:
    """Finding #2: when the inference thread dies fatally, the fatal handler must
    set ``_infer_last_error`` so the still-running capture telemetry surfaces a
    dead, box-blind camera instead of showing it as healthy."""
    w = _worker(_OkSource())
    assert w._infer_last_error is None
    # Make the loop setup raise fatally (outside the per-iteration guard).
    w._build_inference_stacks = lambda: (_ for _ in ()).throw(RuntimeError("build boom"))  # type: ignore[assignment]

    w._inference_start.set()
    t = threading.Thread(target=w._inference_loop, daemon=True)
    t.start()
    t.join(timeout=5.0)

    assert not t.is_alive()
    assert w._infer_last_error is not None, (
        "the inference fatal handler must pin an error so capture telemetry shows "
        "the camera went box-blind"
    )
    assert "inference thread stopped" in w._infer_last_error


# ── process handle composes with the supervisor's auto-restart ────────────────


def test_supervisor_restarts_dead_process_handle(qapp) -> None:  # noqa: ANN001
    """A handle whose is_alive() goes False is torn down and respawned via the
    backoff path — the same path a dead child process travels in opt-in mode.

    The process-per-camera handle (ProcessWorkerHandle) only adds a different
    factory; the supervisor's health scan calls ``worker.is_alive()`` for BOTH
    worker types and respawns through the shared backoff path. This proves a
    process-style handle (is_alive() flips False) composes with that path.
    """
    from autoptz.engine.supervisor import Supervisor
    from autoptz.ui.engine_client import EngineClient

    built: list[str] = []

    class _FakeHandle:
        def __init__(self, camera_id, config, on_telemetry) -> None:  # noqa: ANN001
            self.camera_id = camera_id
            self.shm_name = f"cam_{camera_id[:8]}_preview"
            self._alive = True
            built.append(camera_id)

        def is_alive(self) -> bool:
            return self._alive

        def set_identity_service(self, _s) -> None: ...  # noqa: ANN001
        def set_identity_callback(self, _c) -> None: ...  # noqa: ANN001
        def set_inference_pool(self, _p) -> None: ...  # noqa: ANN001
        def set_features(self, _f) -> None: ...  # noqa: ANN001
        def start(self) -> None: ...
        def stop(self) -> None:
            self._alive = False

    client = EngineClient()
    cid = client.addCamera("usb://0", "ProcRestart")
    client.drain_commands()  # clear the AddCameraCmd so it isn't double-spawned
    sup = Supervisor(client, store=None, worker_factory=_FakeHandle)
    sup.start()
    try:
        handle = sup._workers[cid]
        handle._alive = False  # simulate the child process dying
        # Drive the health scan past its throttle window.
        sup._last_health_scan_t = 0.0
        sup._scan_worker_health(now=1000.0)
        assert built.count(cid) == 2, "supervisor should have respawned the dead handle"
        assert sup._restart_state.get(cid, (0, 0.0, False))[0] == 1
    finally:
        sup.stop()
