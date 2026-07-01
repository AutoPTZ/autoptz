"""Async appearance-thread (face + ReID off the hot loop) behaviour tests.

These exercise the M2d thread split's plumbing + concurrency contract without
real models: that published frames drive the appearance passes on their own
thread, the off-switch works, and the shared target-state lock is re-entrant.
"""

from __future__ import annotations

import threading

import numpy as np

from autoptz.config.models import CameraConfig, SourceConfig
from autoptz.engine.camera_worker import CameraWorker


def _cfg() -> CameraConfig:
    return CameraConfig(
        id="cam-async-1234abcd",
        name="Async",
        source=SourceConfig(type="usb", address="usb://0"),
    )


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _worker() -> CameraWorker:
    return CameraWorker("cam-async-1234abcd", _cfg(), on_telemetry=lambda _m: None)


def test_async_on_by_default() -> None:
    assert _worker()._async_appearance is True


def test_appearance_thread_runs_published_passes() -> None:
    w = _worker()
    calls: list[tuple[str, float]] = []
    done = threading.Event()

    w._maybe_reid_recover = lambda tracks, frame, now: calls.append(("reid", now))  # type: ignore[assignment]

    def fake_identify(frame, tracks, now):  # noqa: ANN001, ANN202
        calls.append(("face", now))
        done.set()

    w._maybe_identify = fake_identify  # type: ignore[assignment]

    w._start_appearance_thread()
    try:
        w._publish_appearance_input(_frame(), [], 1.0, 1)
        assert done.wait(2.0), "appearance thread did not run the published passes"
    finally:
        w._stop_event.set()
        w._stop_appearance_thread()

    assert ("reid", 1.0) in calls
    assert ("face", 1.0) in calls


def test_appearance_thread_drops_stale_duplicate_frames() -> None:
    w = _worker()
    seen: list[int] = []
    gate = threading.Event()

    def fake_identify(frame, tracks, now):  # noqa: ANN001, ANN202
        seen.append(int(now))
        gate.set()

    w._maybe_reid_recover = lambda *a, **k: None  # type: ignore[assignment]
    w._maybe_identify = fake_identify  # type: ignore[assignment]

    w._start_appearance_thread()
    try:
        # Same frame id published twice → processed once.
        w._publish_appearance_input(_frame(), [], 5.0, 7)
        assert gate.wait(2.0)
        gate.clear()
        w._publish_appearance_input(_frame(), [], 6.0, 7)  # same fid=7
        assert not gate.wait(0.4), "duplicate frame id should be skipped"
    finally:
        w._stop_event.set()
        w._stop_appearance_thread()

    assert seen == [5]


def test_off_switch_skips_thread() -> None:
    w = _worker()
    w._async_appearance = False
    w._start_appearance_thread()
    assert w._appearance_thread is None
    w._stop_appearance_thread()  # idempotent / safe


def test_target_lock_is_reentrant() -> None:
    """_commit_target_track is called from within _appearance-guarded methods, so
    the lock must be re-entrant (no self-deadlock)."""
    w = _worker()
    with w._appearance_lock:
        # nested acquire via a guarded method must not deadlock
        w._commit_target_track(42, reason="test")
    assert w._target_track_id == 42


def test_stop_is_idempotent(wait_until) -> None:
    w = _worker()
    w._start_appearance_thread()
    wait_until(
        lambda: w._appearance_thread is not None,
        timeout=1.0,
        message="appearance thread was not created",
    )
    w._stop_event.set()
    w._stop_appearance_thread()
    w._stop_appearance_thread()  # second call is a no-op
    assert w._appearance_thread is None
