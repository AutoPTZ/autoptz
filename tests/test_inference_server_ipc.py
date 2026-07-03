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


# ── R-3: model-server crash recovery ─────────────────────────────────────────


def test_default_timeout_is_two_seconds() -> None:
    """The client must fast-fail at 2.0s (not the old 5.0s) so a dead server does
    not stall a camera's inference loop for 5 seconds on every single frame."""
    import inspect

    sig = inspect.signature(InferenceClient.__init__)
    assert sig.parameters["timeout_s"].default == 2.0


def test_default_timeout_actually_bounds_detect_when_server_never_replies() -> None:
    """Behavioral companion to the signature check above: construct a client with
    the DEFAULT timeout_s (no override) against a request queue nobody is
    servicing, and prove detect() actually gives up around 2s — not the old 5s —
    instead of only trusting the __init__ default value in isolation."""
    import queue

    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 32, 32)
    # No server thread at all — the response queue is never populated, so detect()
    # must fall through its own timeout_s deadline rather than hang.
    client = InferenceClient("camA", queue.Queue(), queue.Queue(), writer)
    t0 = time.monotonic()
    assert client.detect(_frame(1, 32, 32)) == []
    elapsed = time.monotonic() - t0
    assert 1.5 <= elapsed < 4.0, f"detect() took {elapsed:.2f}s — expected ~2.0s default timeout"
    writer.close()


def test_server_down_gate_short_circuits_detect_without_waiting_on_queue() -> None:
    """(b) While the supervisor has the server marked down, detect() must return []
    IMMEDIATELY (well under timeout_s) instead of blocking on the (dead) response
    queue — otherwise every camera crawls at timeout_s per frame during an outage.
    """
    import queue

    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 32, 32)
    server_down = threading.Event()
    server_down.set()  # supervisor has flagged the server as down/being restarted
    # No server thread running at all — if the gate didn't short-circuit, this would
    # block for the full timeout waiting on an empty resp_q.
    client = InferenceClient(
        "camA", queue.Queue(), queue.Queue(), writer, timeout_s=5.0, server_down=server_down
    )
    t0 = time.monotonic()
    assert client.detect(_frame(1, 32, 32)) == []
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"detect() blocked {elapsed:.2f}s despite the down-gate being set"
    writer.close()


def test_server_down_gate_clears_and_detect_resumes() -> None:
    """Once the supervisor clears the down-gate (respawned server signalled ready),
    detect() must go back to actually talking to the server instead of staying
    latched in the fast-fail path."""
    import queue

    cam = "camA"
    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 32, 32)
    reader = ShmReader(name, 32, 32)
    req_q: queue.Queue = queue.Queue()
    resp_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    server_down = threading.Event()
    server_down.set()

    def detect_fn(frame):  # noqa: ANN001
        return [("ok", int(frame[0, 0, 0]))]

    t = threading.Thread(
        target=serve, args=(req_q, {cam: resp_q}, {cam: reader}, detect_fn, stop), daemon=True
    )
    t.start()
    try:
        client = InferenceClient(cam, req_q, resp_q, writer, timeout_s=2.0, server_down=server_down)
        assert client.detect(_frame(3, 32, 32)) == []  # gate set → fast-fail, no real call
        server_down.clear()  # supervisor: respawned server is ready again
        assert client.detect(_frame(3, 32, 32)) == [("ok", 3)]  # resumes real detection
    finally:
        stop.set()
        t.join(timeout=1.0)
        writer.close()
        reader.close()


def test_remote_pool_uses_client_while_server_is_healthy() -> None:
    """RemotePool must not fall back to a local detector while the model-server
    has not been declared permanently failed — the IPC client stays authoritative."""
    import queue

    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 16, 16)
    client = InferenceClient("camA", queue.Queue(), queue.Queue(), writer)
    built_local = []

    def _build_local():
        built_local.append(True)
        return object()

    pool = RemotePool(client, failed=threading.Event(), build_local_fn=_build_local)
    assert pool.detector() is client
    assert built_local == []
    writer.close()


def test_remote_pool_falls_back_to_local_detector_once_failed() -> None:
    """(c) After the supervisor exhausts the model-server restart budget, it sets
    the pool's failed flag. RemotePool.detector() must then hand back a LOCAL
    detector (built via the injected factory) instead of the dead IPC client, so
    ``worker.refresh_detector_from_pool()`` — which just re-calls ``pool.detector()``
    — is enough to make detection resume in degraded (per-worker) mode.
    """
    import queue

    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 16, 16)
    client = InferenceClient("camA", queue.Queue(), queue.Queue(), writer)
    local_detector = object()
    calls = []

    def _build_local():
        calls.append(True)
        return local_detector

    failed = threading.Event()
    pool = RemotePool(client, failed=failed, build_local_fn=_build_local)
    assert pool.detector() is client  # healthy → still the IPC client

    failed.set()  # supervisor: restart budget exhausted
    assert pool.detector() is local_detector
    assert len(calls) == 1

    # Subsequent calls reuse the cached local detector, not rebuild it every time.
    assert pool.detector() is local_detector
    assert len(calls) == 1
    writer.close()


def test_respawned_server_reuses_same_queues_no_client_reconstruction() -> None:
    """(a)+(d) Kill the server-side thread mid-run, "respawn" it (a fresh serve()
    loop reusing the SAME req/resp queues and shm reader dict — exactly what the
    supervisor does across a real process respawn), and prove the ORIGINAL client
    resumes getting real detections with NO reconstruction. This pins that shm
    re-attach is lazy per-request (serve()'s ``attach`` callback), so reusing the
    same queues + a fresh reader dict is sufficient for recovery.
    """
    import queue

    cam = "camA"
    name = f"itest_{uuid.uuid4().hex[:8]}"
    writer = ShmWriter(name, 32, 32)
    req_q: queue.Queue = queue.Queue()
    resp_q: queue.Queue = queue.Queue()

    def detect_fn(frame):  # noqa: ANN001
        return [("ok", int(frame[0, 0, 0]))]

    def attach(_c: str):  # noqa: ANN202
        try:
            return ShmReader(name, 32, 32)
        except FileNotFoundError:
            return None

    stop1 = threading.Event()
    t1 = threading.Thread(
        target=serve,
        args=(req_q, {cam: resp_q}, {}, detect_fn, stop1),
        kwargs={"attach": attach},
        daemon=True,
    )
    t1.start()
    client = InferenceClient(cam, req_q, resp_q, writer, timeout_s=2.0)
    try:
        assert client.detect(_frame(4, 32, 32)) == [("ok", 4)]  # server #1 alive

        # Kill server #1 ("the child process died").
        stop1.set()
        t1.join(timeout=1.0)

        # Respawn: a brand-new serve() loop, SAME req_q/resp_q, fresh reader dict —
        # mirrors the supervisor building a new Process with the queues it already
        # held. No new InferenceClient is constructed.
        stop2 = threading.Event()
        t2 = threading.Thread(
            target=serve,
            args=(req_q, {cam: resp_q}, {}, detect_fn, stop2),
            kwargs={"attach": attach},
            daemon=True,
        )
        t2.start()
        try:
            assert client.detect(_frame(9, 32, 32)) == [("ok", 9)]  # resumed, same client
        finally:
            stop2.set()
            t2.join(timeout=1.0)
    finally:
        writer.close()
