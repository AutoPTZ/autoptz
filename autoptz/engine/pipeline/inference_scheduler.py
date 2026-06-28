"""Single-owner inference scheduler — the heart of the scalable redesign.

**Why this exists (measured):** the Apple Neural Engine is a *fixed* ~20
detections/sec resource shared by every camera, and it does **not** batch (batch-8
is only ~1.19x the single-frame rate). So the throughput ceiling is the accelerator,
not the model or the language. Having N camera worker threads each call the detector
inline does two bad things: it thrashes that one accelerator with uncoordinated
requests, and it runs the NMS/letterbox/track glue under the GIL N times in parallel
(the dominant contention — turning off the *second* ANE consumer, face recognition,
doubled measured throughput).

**What this does:** centralizes ALL accelerator work behind ONE worker thread that
pulls jobs **fairly** across cameras (round-robin, so no camera is starved) and
**latest-wins** per ``(camera_id, kind)`` (a newer frame replaces an unstarted older
one, so a camera never acts on a stale frame when the accelerator is the bottleneck).
Camera threads keep only capture + tracking + control, which run in parallel because
cv2/NDI/ORT release the GIL during their native calls.

This module is the *scheduling core* — pure, dependency-light, and unit-tested with a
fake ``run_fn``. Wiring it into ``CameraWorker`` (workers submit frames instead of
calling the detector inline) is a separate, later step.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

#: ``run_fn(camera_id, kind, frame) -> result`` — runs one inference on the accelerator.
RunFn = Callable[[str, str, Any], Any]
#: ``on_result(result) -> None`` — delivers a finished job's result to the submitter.
ResultFn = Callable[[Any], None]


class InferenceScheduler:
    """One worker that owns the accelerator and serves all cameras fairly.

    Usage::

        sched = InferenceScheduler(run_fn=pool.detect_for)
        sched.start()
        # from a camera thread, each new frame:
        sched.submit(camera_id, "detect", frame, on_result=worker.on_detections)
        ...
        sched.stop()
    """

    def __init__(self, run_fn: RunFn) -> None:
        self._run_fn = run_fn
        # camera_id -> {kind -> (frame, on_result)}; one slot per (camera, kind) so a
        # newer submit overwrites an unstarted older frame (latest-wins).
        self._pending: dict[str, dict[str, tuple[Any, ResultFn | None]]] = {}
        self._cameras: list[str] = []  # insertion-ordered rotation of known cameras
        self._cursor = 0  # round-robin position into _cameras
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # True once the worker has dequeued and begun running at least one job — lets
        # callers/tests observe that work is in flight without polling run_fn itself.
        self._running_started = False

    # ── public API ───────────────────────────────────────────────────────────

    def submit(
        self, camera_id: str, kind: str, frame: Any, on_result: ResultFn | None = None
    ) -> None:
        """Queue a frame for inference, replacing any unstarted frame for this
        ``(camera_id, kind)`` (latest-wins). Cheap + non-blocking — safe on the hot path."""
        with self._cv:
            slot = self._pending.get(camera_id)
            if slot is None:
                slot = self._pending[camera_id] = {}
                self._cameras.append(camera_id)
            slot[kind] = (frame, on_result)
            self._cv.notify()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="infer-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker and wait for it to exit. Idempotent."""
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    # ── worker ───────────────────────────────────────────────────────────────

    def _pick(self) -> tuple[str, str, Any, ResultFn | None] | None:
        """Return the next job (caller holds ``_cv``), advancing the round-robin
        cursor past the chosen camera so no camera is served twice before its peers."""
        n = len(self._cameras)
        for offset in range(n):
            idx = (self._cursor + offset) % n
            cam = self._cameras[idx]
            slot = self._pending.get(cam)
            if slot:
                kind = next(iter(slot))  # one job per camera-selection (kinds rotate over picks)
                frame, on_result = slot.pop(kind)
                self._cursor = (idx + 1) % n  # next pick starts at the following camera
                return cam, kind, frame, on_result
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                job = self._pick()
                while job is None and not self._stop.is_set():
                    self._cv.wait(timeout=0.1)
                    job = self._pick()
            if self._stop.is_set():
                return
            assert job is not None
            cam, kind, frame, on_result = job
            self._running_started = True
            try:
                result = self._run_fn(cam, kind, frame)
            except Exception:  # noqa: BLE001 — a bad inference must not kill the scheduler
                log.warning("inference job %s/%s failed", cam, kind, exc_info=True)
                continue
            if on_result is not None:
                try:
                    on_result(result)
                except Exception:  # noqa: BLE001 — a bad consumer must not kill the scheduler
                    log.debug("inference result callback for %s/%s failed", cam, kind, exc_info=True)


class SchedulerDetector:
    """Drop-in detector that routes ``detect()`` through a shared scheduler.

    Has the same surface a worker uses (``detect(frame)`` + ``ep`` + anything else it
    delegates to the real detector), so swapping it in needs no change to the camera
    worker's inference loop. ``detect`` submits the frame and **blocks** until the one
    scheduler thread runs it — while blocked, this camera's thread holds no GIL, so
    other cameras' capture/track and the scheduler run freely. Net effect: the heavy
    detect glue (letterbox/NMS) happens once, serially, on the scheduler instead of N
    camera threads thrashing the GIL in parallel.

    Each ``detect`` call is synchronous, so there is at most one in-flight job per
    camera and the scheduler's latest-wins drop never discards a frame here.
    """

    def __init__(
        self,
        camera_id: str,
        real_detector: Any,
        scheduler: InferenceScheduler,
        timeout_s: float = 5.0,
    ) -> None:
        self._camera_id = camera_id
        self._real = real_detector
        self._scheduler = scheduler
        self._timeout_s = timeout_s

    @property
    def ep(self) -> str:
        return str(getattr(self._real, "ep", "") or "")

    def detect(self, frame: Any) -> Any:
        box: list[Any] = []
        done = threading.Event()

        def _cb(result: Any) -> None:
            box.append(result)
            done.set()

        self._scheduler.submit(self._camera_id, "detect", frame, on_result=_cb)
        if not done.wait(self._timeout_s):
            log.warning("scheduler detect timed out for %s", self._camera_id)
            return []
        return box[0] if box else []

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (e.g. input_size, model metadata) to the real detector.
        return getattr(self._real, name)
