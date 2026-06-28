"""InferenceServer/InferenceClient — the IPC detection mechanism for the
multi-process model-server architecture (validated to scale: 16 NDI cams, one model
set, no RAM cliff).

A camera *process* delegates detection to ONE shared model-server *process*: it writes
its frame into a per-camera shared-memory slot, enqueues a tiny request, and blocks for
the detections. The server holds the single detector, reads the frame, detects, replies.
These tests pin the contract in-process (real queues + real shm) with a fake detector —
no spawn, no real model.
"""

from __future__ import annotations

import threading
import time
import uuid

import numpy as np

from autoptz.engine.pipeline.inference_server import InferenceClient, RemotePool, serve
from autoptz.engine.runtime.shm import ShmReader, ShmWriter


def _frame(val: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_client_roundtrips_detection_through_server() -> None:
    import queue

    cam = "camA"
    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 64, 64)
    reader = ShmReader(name, 64, 64)
    req_q: queue.Queue = queue.Queue()
    resp_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    # The detector "result" encodes the frame content so we can prove the SERVER read
    # the exact frame the CLIENT wrote (not a stale/blank slot).
    def detect_fn(frame):  # noqa: ANN001
        return [("det", int(frame[0, 0, 0]))]

    t = threading.Thread(
        target=serve, args=(req_q, {cam: resp_q}, {cam: reader}, detect_fn, stop), daemon=True
    )
    t.start()
    try:
        client = InferenceClient(cam, req_q, resp_q, writer, timeout_s=2.0)
        assert client.detect(_frame(7)) == [("det", 7)]
        assert client.detect(_frame(200)) == [("det", 200)]  # fresh frame each call
        assert client.ep  # exposes an EP string for the worker's diagnostics
    finally:
        stop.set()
        t.join(timeout=1.0)
        writer.close()
        reader.close()


def test_client_returns_empty_on_timeout() -> None:
    import queue

    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 32, 32)
    # No server running → detect must return [] within the timeout, not hang.
    client = InferenceClient("camA", queue.Queue(), queue.Queue(), writer, timeout_s=0.2)
    t0 = time.monotonic()
    assert client.detect(_frame(1, 32, 32)) == []
    assert time.monotonic() - t0 < 1.5
    writer.close()


def test_remote_pool_exposes_client_as_detector() -> None:
    import queue

    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 16, 16)
    client = InferenceClient("camA", queue.Queue(), queue.Queue(), writer)
    pool = RemotePool(client)
    assert pool.detector() is client
    assert hasattr(pool, "detector_ep")
    writer.close()


def test_server_survives_a_detector_exception() -> None:
    import queue

    cam = "camA"
    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 32, 32)
    reader = ShmReader(name, 32, 32)
    req_q: queue.Queue = queue.Queue()
    resp_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def detect_fn(frame):  # noqa: ANN001
        if int(frame[0, 0, 0]) == 1:
            raise RuntimeError("boom")
        return [("ok", int(frame[0, 0, 0]))]

    t = threading.Thread(
        target=serve, args=(req_q, {cam: resp_q}, {cam: reader}, detect_fn, stop), daemon=True
    )
    t.start()
    try:
        client = InferenceClient(cam, req_q, resp_q, writer, timeout_s=2.0)
        assert client.detect(_frame(1, 32, 32)) == []  # detector raised → empty, no crash
        assert client.detect(_frame(5, 32, 32)) == [("ok", 5)]  # server still serving
    finally:
        stop.set()
        t.join(timeout=1.0)
        writer.close()
        reader.close()
