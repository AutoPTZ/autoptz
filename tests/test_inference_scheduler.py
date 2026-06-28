"""InferenceScheduler — the single accelerator owner for the scalable redesign.

The Neural Engine is a fixed ~20 detections/sec resource shared by every camera
(measured). Having N camera threads each call the detector inline both (a) thrashes
that one accelerator and (b) runs the NMS/letterbox/track glue under the GIL N times.
The scheduler centralizes ALL inference behind ONE worker that pulls work *fairly*
across cameras (round-robin) and *latest-wins* per (camera, kind) — so a slow
accelerator never makes a camera act on a stale frame, and no camera is starved.
Camera threads keep only capture + track + control, which run in parallel because
cv2/NDI release the GIL.

These tests pin the scheduling contract with a fake ``run_fn`` (no real model).
"""

from __future__ import annotations

import threading
import time

from autoptz.engine.pipeline.inference_scheduler import InferenceScheduler


def _wait(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_runs_submitted_job_and_delivers_result() -> None:
    sched = InferenceScheduler(run_fn=lambda cam, kind, frame: f"{cam}:{kind}:{frame}")
    got: list[str] = []
    sched.start()
    try:
        sched.submit("camA", "detect", 1, on_result=got.append)
        assert _wait(lambda: got == ["camA:detect:1"]), got
    finally:
        sched.stop()


def test_latest_wins_drops_stale_frame_per_camera_kind() -> None:
    # A blocked worker accumulates 3 submits for the SAME (cam, kind); only the
    # newest frame must run — the two stale ones are dropped (never act on old data).
    release = threading.Event()
    seen: list[int] = []

    def run_fn(cam, kind, frame):  # noqa: ANN001
        release.wait(2.0)  # hold the worker on the first job so the rest queue
        seen.append(frame)
        return frame

    sched = InferenceScheduler(run_fn=run_fn)
    done: list[int] = []
    sched.start()
    try:
        sched.submit("camA", "detect", 1, on_result=done.append)  # starts running, blocks
        assert _wait(lambda: sched._running_started, 1.0)
        sched.submit("camA", "detect", 2, on_result=done.append)  # queued
        sched.submit("camA", "detect", 3, on_result=done.append)  # replaces 2 (latest-wins)
        release.set()
        assert _wait(lambda: len(seen) == 2, 2.0), seen
        # Frame 1 (already running) completes, then only the latest queued (3) runs; 2 dropped.
        assert seen == [1, 3], seen
    finally:
        sched.stop()


def test_fair_round_robin_across_cameras() -> None:
    # Three cameras each flood the scheduler; the worker must alternate fairly,
    # not drain one camera before serving the others.
    release = threading.Event()
    order: list[str] = []

    def run_fn(cam, kind, frame):  # noqa: ANN001
        release.wait(2.0)
        order.append(cam)
        return None

    sched = InferenceScheduler(run_fn=run_fn)
    sched.start()
    try:
        # Prime one job so the worker is busy while we enqueue the rest.
        sched.submit("A", "detect", 0, on_result=lambda _r: None)
        assert _wait(lambda: sched._running_started, 1.0)
        for f in range(3):
            for cam in ("A", "B", "C"):
                sched.submit(cam, "detect", f, on_result=lambda _r: None)
        release.set()
        # Each (cam,detect) keeps only its latest → 1 pending per camera + the running A.
        assert _wait(lambda: len(order) >= 4, 2.0), order
        # After the initial running A, the next three must be one each of A/B/C (fair).
        assert set(order[1:4]) == {"A", "B", "C"}, order
    finally:
        sched.stop()


def test_stop_is_clean_and_idempotent() -> None:
    sched = InferenceScheduler(run_fn=lambda c, k, f: None)
    sched.start()
    sched.stop()
    sched.stop()  # second stop must not raise
    assert not sched.is_running


def test_run_fn_exception_does_not_kill_the_worker() -> None:
    calls: list[int] = []

    def run_fn(cam, kind, frame):  # noqa: ANN001
        calls.append(frame)
        if frame == 1:
            raise RuntimeError("boom")
        return frame

    sched = InferenceScheduler(run_fn=run_fn)
    results: list[int] = []
    sched.start()
    try:
        sched.submit("A", "detect", 1, on_result=results.append)  # raises inside worker
        sched.submit("A", "detect", 2, on_result=results.append)  # must still run
        assert _wait(lambda: 2 in calls, 2.0), calls
        assert results == [2], results  # the failed job delivered no result, worker survived
    finally:
        sched.stop()
