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


def test_serve_attaches_reader_lazily_when_writer_appears_after_server() -> None:
    """Production ordering: the server starts BEFORE the camera child creates its
    writer (the supervisor spawns the server, blocks on ready, THEN spawns cameras).
    serve() must attach each camera's reader LAZILY — on the first request after the
    writer exists — and then serve real detections, instead of skipping forever.

    This pins the fix for the dead-on-arrival bug where the server eagerly attached
    all readers at startup (when no writer existed yet) and every detect() returned [].
    """
    import queue

    cam = "camLazy"
    name = f"itest_{uuid.uuid4().hex[:8]}"
    req_q: queue.Queue = queue.Queue()
    resp_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def detect_fn(frame):  # noqa: ANN001
        return [("det", int(frame[0, 0, 0]))]

    def attach(c: str):  # noqa: ANN202 — returns ShmReader | None
        try:
            return ShmReader(name, 48, 48)
        except FileNotFoundError:
            return None

    # Server starts with NO readers attached (writer does not exist yet).
    readers: dict = {}
    t = threading.Thread(
        target=serve,
        args=(req_q, {cam: resp_q}, readers, detect_fn, stop),
        kwargs={"attach": attach},
        daemon=True,
    )
    t.start()
    # NOW the camera child comes up and creates its writer (after the server is serving).
    writer = ShmWriter(name, 48, 48)
    try:
        client = InferenceClient(cam, req_q, resp_q, writer, timeout_s=2.0)
        assert client.detect(_frame(7, 48, 48)) == [("det", 7)]  # lazily attached + served
        assert client.detect(_frame(9, 48, 48)) == [("det", 9)]  # reader cached, still fresh
    finally:
        stop.set()
        t.join(timeout=1.0)
        writer.close()


def test_one_timeout_does_not_desync_subsequent_detections() -> None:
    """A single slow/timed-out detect() must NOT permanently lag the camera. After a
    timeout the in-flight reply may still land on the response queue; the NEXT detect()
    must discard that stale reply (via the per-request sequence id) and return the
    detections for the frame it actually submitted — not the previous frame's boxes.
    """
    import queue

    cam = "camD"
    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 32, 32)
    reader = ShmReader(name, 32, 32)
    req_q: queue.Queue = queue.Queue()
    resp_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def detect_fn(frame):  # noqa: ANN001
        return [("det", int(frame[0, 0, 0]))]

    t = threading.Thread(
        target=serve, args=(req_q, {cam: resp_q}, {cam: reader}, detect_fn, stop), daemon=True
    )
    t.start()
    try:
        client = InferenceClient(cam, req_q, resp_q, writer, timeout_s=2.0)
        # Simulate a leftover reply from a PRIOR request that the client already gave up
        # on (timed out): a response tagged with a sequence id the client will never use
        # again. A correct client discards it and waits for its own request's reply.
        resp_q.put((-999, [("STALE", 123)]))
        assert client.detect(_frame(42, 32, 32)) == [("det", 42)]  # fresh, not STALE
    finally:
        stop.set()
        t.join(timeout=1.0)
        writer.close()
        reader.close()


def test_client_drains_stale_replies_on_construction() -> None:
    """A response queue is reused across a worker restart. If the crashed run left
    replies on it, the new client must drain them on construction so a leftover can't be
    matched against the restarted worker's first request.
    """
    import queue

    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 16, 16)
    resp_q: queue.Queue = queue.Queue()
    resp_q.put((1, [("OLD", 1)]))
    resp_q.put((2, [("OLD", 2)]))
    InferenceClient("camX", queue.Queue(), resp_q, writer)
    assert resp_q.empty()  # constructor drained the leftovers
    writer.close()


def test_detections_scaled_back_to_native_frame_coords() -> None:
    """When the camera frame is not the slot size, the client resizes it to the slot
    before sending, so the detector returns boxes in SLOT coordinates. The client must
    map them BACK to the camera's native frame — otherwise the worker draws overlays and
    aims the PTZ at the wrong place on any non-1080p source.
    """
    import queue

    from autoptz.engine.pipeline.detect import BBox, Detection

    cam = "camScale"
    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 100, 200)  # slot is H=100 x W=200
    reader = ShmReader(name, 100, 200)
    req_q: queue.Queue = queue.Queue()
    resp_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def detect_fn(frame):  # noqa: ANN001 — returns a box in SLOT (200x100) coords
        return [Detection(bbox=BBox(20.0, 10.0, 100.0, 50.0), conf=0.9)]

    t = threading.Thread(
        target=serve, args=(req_q, {cam: resp_q}, {cam: reader}, detect_fn, stop), daemon=True
    )
    t.start()
    try:
        client = InferenceClient(cam, req_q, resp_q, writer, timeout_s=2.0)
        native = np.full((50, 100, 3), 7, dtype=np.uint8)  # native H=50 x W=100 (half the slot)
        dets = client.detect(native)
        bb = dets[0].bbox
        # slot W,H = 200,100; native W,H = 100,50 → scale x by 0.5, y by 0.5
        assert (bb.x1, bb.y1, bb.x2, bb.y2) == (10.0, 5.0, 50.0, 25.0)
    finally:
        stop.set()
        t.join(timeout=1.0)
        writer.close()
        reader.close()


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
