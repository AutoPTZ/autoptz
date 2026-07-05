"""Multi-process model-server: ONE detector serves many camera processes over IPC.

Validated to scale where threaded (GIL cliff) and per-process (RAM cliff) do not:
16 NDI cameras, ONE model set, ~6 GB RAM, all cameras alive at 30 fps capture. Each
camera runs in its own process (escaping the GIL for capture/track/control) and
*delegates* detection to a single shared model-server process — so there is exactly
one model set (no per-process duplication) and the scarce accelerator is used by one
owner.

Transport: frames cross via the existing torn-read-safe shared-memory ring
(:class:`ShmWriter`/:class:`ShmReader`) — one slot per camera, latest-wins; requests
and the small detection lists cross via :class:`multiprocessing.Queue`.

This module is the mechanism. Wiring it behind ``AUTOPTZ_MODEL_SERVER`` (supervisor
spawns the server; camera children build an :class:`InferenceClient` instead of their
own pool) is done in the supervisor / process_worker.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

#: Detection shm is sized to this fixed resolution for v1 (NDI is ~1080p); a camera
#: resizes its frame to fit before pushing. Variable per-source resolution is a v2.
SERVER_FRAME_H = 1080
SERVER_FRAME_W = 1920


def shm_name_for(camera_id: str) -> str:
    """Stable shm name for a camera's detection-frame slot (writer + server reader)."""
    return f"infer_{camera_id[:8]}"


def _rescale_detections(dets: Any, sx: float, sy: float) -> Any:
    """Scale each detection's bbox (and keypoints) by ``(sx, sy)`` — used to map slot-
    space boxes back to a camera's native frame. Type-agnostic (``dataclasses.replace``)
    so it doesn't couple the IPC layer to the detection types; leaves anything it can't
    rescale untouched.
    """
    import dataclasses  # noqa: PLC0415

    out = []
    for d in dets:
        try:
            bb = d.bbox
            nb = dataclasses.replace(bb, x1=bb.x1 * sx, y1=bb.y1 * sy, x2=bb.x2 * sx, y2=bb.y2 * sy)
            kps = getattr(d, "keypoints", None)
            if kps:
                kps = tuple((float(x) * sx, float(y) * sy, c) for (x, y, c) in kps)
            out.append(dataclasses.replace(d, bbox=nb, keypoints=kps))
        except Exception:  # noqa: BLE001, PERF203 — unknown shape: pass through unchanged
            out.append(d)
    return out


class InferenceClient:
    """Drop-in detector that delegates to the shared model-server over IPC.

    ``detect(frame)`` pushes the frame into this camera's shm slot, enqueues a tiny
    request, and blocks for the detections — so it slots in wherever the worker calls
    ``detector.detect(frame)`` with no change to the inference loop. Returns ``[]`` on
    timeout rather than hanging the camera process.
    """

    def __init__(
        self,
        camera_id: str,
        req_q: Any,
        resp_q: Any,
        shm_writer: Any,
        timeout_s: float = 2.0,
        server_down: Any = None,
    ) -> None:
        self._cam = camera_id
        self._req_q = req_q
        self._resp_q = resp_q
        self._shm = shm_writer
        self._timeout_s = timeout_s
        # Supervisor-owned gate (multiprocessing.Event or equivalent — anything with
        # ``is_set()``): set while the shared server is down/being restarted, so
        # detect() fast-fails instead of blocking a whole timeout per frame during
        # an outage. None (tests/no-recovery callers) → gate is always "up".
        self._server_down = server_down
        self._seq = 0  # per-request id so a timed-out reply can't desync the next call
        # Real execution provider of the SERVER's detector (e.g. "CoreMLExecutionProvider").
        # Arrives tagged on replies once the server's background model load finishes;
        # until then the label is the plain "model-server".
        self._server_ep = ""
        # The response queue may be reused across a worker restart; drop any replies
        # left by a previous (crashed) run so they can't be matched to a fresh request.
        try:
            while True:
                self._resp_q.get_nowait()
        except Exception:  # noqa: BLE001 — queue empty (or unsupported) → nothing to drain
            pass

    @property
    def ep(self) -> str:
        """Diagnostics label: enriched with the server's REAL provider once known,
        so the UI can say "model-server (CoreML)" instead of hiding the engine."""
        if self._server_ep:
            short = self._server_ep.replace("ExecutionProvider", "")
            return f"model-server ({short})"
        return "model-server"

    def detect(self, frame: Any) -> Any:
        if self._server_down is not None and self._server_down.is_set():
            # Supervisor has flagged the server as down/being restarted — fail fast
            # instead of waiting on a queue nothing is going to answer.
            return []
        try:
            h, w = int(self._shm.height), int(self._shm.width)
            oh, ow = int(frame.shape[0]), int(frame.shape[1])
            if (oh, ow) != (h, w):
                import cv2  # noqa: PLC0415

                # The shm slot is a fixed size; resize to fit it. Detections come back
                # in slot coords and are mapped back to the native frame below.
                frame = cv2.resize(frame, (w, h))
            self._seq += 1
            seq = self._seq
            self._shm.push(frame)
            self._req_q.put((self._cam, seq))
        except Exception:  # noqa: BLE001 — IPC hiccup must not kill the camera loop
            log.debug("inference client %s submit failed", self._cam, exc_info=True)
            return []
        # Read replies until we get the one tagged with OUR seq. A reply left over
        # from a previously timed-out request carries an old seq and is discarded —
        # without this, one timeout permanently lags the camera one frame behind.
        deadline = time.monotonic() + self._timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            try:
                msg = self._resp_q.get(timeout=remaining)
            except Exception:  # noqa: BLE001 — server gone / timed out
                return []
            if isinstance(msg, tuple) and len(msg) == 3:
                # New replies carry the server detector's real EP; remember it for
                # the ``ep`` label. Empty while the server is still loading its model.
                rseq, dets, srv_ep = msg
                if srv_ep:
                    self._server_ep = str(srv_ep)
            elif isinstance(msg, tuple) and len(msg) == 2:
                rseq, dets = msg
            else:
                rseq, dets = seq, msg
            if rseq != seq:
                continue  # stale reply from a prior (timed-out) request — discard
            if not dets:
                return []
            # Detections are in slot coords; map back to the native frame so overlays
            # and PTZ aim land correctly on non-slot-sized sources (no-op at 1080p).
            if (oh, ow) != (h, w):
                dets = _rescale_detections(dets, ow / float(w), oh / float(h))
            return dets


class RemotePool:
    """Pool-shaped wrapper so the worker's pooled detect-stack build uses the IPC
    client as its detector (the tracker stays local per-camera).

    Degraded fallback (R-3): once the supervisor gives up trying to respawn a dead
    model-server (``failed`` set), ``detector()`` stops handing back the (now
    permanently useless) IPC client and instead lazily builds + caches a LOCAL
    detector via ``build_local_fn``. This is the seam
    ``CameraWorker.refresh_detector_from_pool()`` already re-invokes on command, so
    the supervisor's only job on budget exhaustion is: set ``failed`` and ask every
    model-server worker to refresh — no new IPC, no protocol change.
    """

    def __init__(
        self,
        client: InferenceClient,
        *,
        failed: Any = None,
        build_local_fn: Any = None,
    ) -> None:
        self._client = client
        self._failed = failed
        self._build_local_fn = build_local_fn
        self._local: Any | None = None
        self._pose: Any | None = None
        self._pose_built = False

    @property
    def detector_ep(self) -> str:
        """ "model-server", enriched to "model-server (CoreML)" once the server
        has reported its detector's real execution provider."""
        return str(getattr(self._client, "ep", "") or "model-server")

    def detector(self) -> Any:
        if self._failed is not None and self._failed.is_set():
            if self._local is None and self._build_local_fn is not None:
                try:
                    self._local = self._build_local_fn()
                except Exception:  # noqa: BLE001 — local fallback must never crash the worker
                    log.warning("model-server fallback: local detector build failed", exc_info=True)
            if self._local is not None:
                return self._local
        return self._client

    def pose(self) -> Any | None:
        """Local (per-child) pose estimator — only DETECTION is delegated to the
        server.  Pose runs on the target's crop a few times a second, so a small
        per-camera session is cheap; without this the worker's
        ``_ensure_pose`` → ``pool.pose()`` raised AttributeError and permanently
        disabled pose — every pose-derived behaviour (arm-invariant aim, torso
        framing box, skeleton overlay) silently degraded to the raw
        arms-inflated detection bbox.  Built lazily, cached (including a
        ``None`` failure).  Never downloads models (children mirror the
        local-detector fallback; provisioning belongs to the parent/UI)."""
        if self._pose_built:
            return self._pose
        self._pose_built = True
        try:
            from autoptz.engine.pipeline.pose import PoseEstimator

            self._pose = PoseEstimator(allow_download=False)
        except Exception:  # noqa: BLE001 — pose must never break the camera child
            log.warning("camera-child pose estimator init failed; bbox aim only.", exc_info=True)
            self._pose = None
        return self._pose

    # ── release (mirrors InferencePool's release_detector/release_pose) ─────────
    #
    # Supervisor.release_model_sessions/rebuild_model_sessions call these BY NAME
    # (``getattr(pool, method, None)``) so a Manage Models mutation (detector tier
    # switch, pose model swap) invalidates any cached session before the on-disk
    # cache is mutated/rebuilt.  For a model-server camera child this pool is a
    # per-PROCESS RemotePool the supervisor can never reach directly — the only
    # thing that ever calls these is this same child's own
    # ``CameraWorker._release_inference_models``/``_reload_inference_models``
    # (see camera_worker.py), which mirrors the supervisor's generic dispatch for
    # whatever pool it was given.  Matching InferencePool's method names/semantics
    # here is what lets that one generic call site work for both pool types.

    def release_detector(self) -> None:
        """Drop the cached LOCAL fallback detector so it rebuilds on next use.

        Detection itself is delegated to the shared model-server
        (``self._client``, an IPC handle owned by the supervisor's server
        process) — there is no local ORT session here to free in the normal
        case.  The one piece of local, per-child state this pool DOES cache is
        the R-3 degraded-mode fallback built by ``build_local_fn`` once the
        supervisor marks the server ``failed`` (see :meth:`detector`).  Dropping
        it here means a Manage Models mutation while a camera is running in
        degraded/local-fallback mode doesn't leave it stuck on a stale local
        detector session until the app restarts.
        """
        self._local = None

    def release_pose(self) -> None:
        """Drop the cached local pose estimator so :meth:`pose` rebuilds it.

        Without this, swapping the pose model in Manage Models never invalidated
        a model-server camera child's own cached ``self._pose``/
        ``self._pose_built`` — every camera process kept the stale pose session
        until a full app restart.
        """
        self._pose = None
        self._pose_built = False


def serve(
    req_q: Any,
    resp_qs: dict[str, Any],
    readers: dict[str, Any],
    detect_fn: Any,
    stop_ev: Any,
    attach: Any = None,
    ep_fn: Any = None,
) -> None:
    """Server loop: drain detection requests, read each camera's latest frame from
    shm, run ``detect_fn`` once, and reply ``(seq, dets)`` on that camera's response
    queue (the ``seq`` echoes the request so the client can drop stale replies).

    Readers are attached LAZILY via ``attach(cam) -> reader | None``: the camera
    children create their shm writers AFTER the server is already serving, so a reader
    cannot exist at server startup. On the first request for a camera whose writer is
    up, ``attach`` succeeds and the reader is cached; until then (or for an unknown
    camera) the server replies ``[]`` immediately so the client never eats its full
    timeout, and retries the attach on the camera's next request.

    Single-owner of the accelerator → naturally serializes (the accelerator is serial
    anyway) and shares it fairly FIFO across cameras (each camera keeps one request
    outstanding). A detector exception yields an empty result, never a crash.
    """

    def _reply(rq: Any, seq: int, dets: Any) -> None:
        try:
            if ep_fn is not None:
                # Tag the reply with the detector's real EP ("" while still loading)
                # so clients can label themselves "model-server (CoreML)".
                rq.put((seq, dets, str(ep_fn() or "")))
            else:
                rq.put((seq, dets))
        except Exception:  # noqa: BLE001 — client gone
            pass

    # Health counters. A model-server that silently serves zero detections (e.g. the
    # writers never came up) is a hard-to-spot failure, so when AUTOPTZ_MS_DIAG=1 the
    # server logs throughput periodically. Default: off, near-zero overhead.
    import os  # noqa: PLC0415

    diag = os.environ.get("AUTOPTZ_MS_DIAG") == "1"
    served = waited = 0
    last_emit = time.monotonic()

    while not stop_ev.is_set():
        if diag and time.monotonic() - last_emit >= 5.0:
            log.info(
                "model-server health: served=%d/s waiting-on-writer=%d/s attached=%d",
                round(served / 5.0),
                round(waited / 5.0),
                len(readers),
            )
            served = waited = 0
            last_emit = time.monotonic()
        try:
            msg = req_q.get(timeout=0.2)
        except Exception:  # noqa: BLE001 — empty/closed queue
            continue
        if isinstance(msg, tuple):
            cam = msg[0]
            seq = msg[1] if len(msg) > 1 else 0
        else:
            cam, seq = msg, 0
        rq = resp_qs.get(cam)
        if rq is None:
            continue  # unknown camera — no queue to reply on
        reader = readers.get(cam)
        if reader is None and attach is not None:
            reader = attach(cam)
            if reader is not None:
                readers[cam] = reader
        if reader is None:
            waited += 1
            _reply(rq, seq, [])  # writer not up yet — reply empty, retry attach next time
            continue
        frame = None
        for _ in range(5):  # the push may not be visible the instant the request is
            got = reader.latest()
            if got is not None:
                frame = got[1]
                break
            time.sleep(0.001)
        try:
            dets = detect_fn(frame) if frame is not None else []
        except Exception:  # noqa: BLE001 — a bad detect must not kill the server
            log.debug("model-server detect for %s failed", cam, exc_info=True)
            dets = []
        served += 1
        _reply(rq, seq, dets)


def run_inference_server(
    req_q: Any,
    resp_qs: dict[str, Any],
    cam_ids: list[str],
    detector_tier: str,
    unified_pose: bool,
    ready_ev: Any,
    stop_ev: Any,
) -> None:
    """Process entrypoint: signal ready, then serve while the ONE shared detector loads
    in the background. Spawn-safe (top-level, picklable args). Best-effort — the detector
    loads off the serve path so the server accepts requests immediately (replying [] until
    the model is in), and a build failure still serves empty so cameras never hang.
    """
    import logging as _logging
    import os as _os
    import threading as _threading

    from autoptz.engine.process_worker import (
        _configure_child_logging,
        _install_parent_death_watchdog,
    )
    from autoptz.engine.runtime.shm import ShmReader

    # A spawned child inherits no log handlers, so the detector-build error and the
    # optional health log would go nowhere without this.
    _configure_child_logging()
    # Never outlive the app: exit if the parent is killed by signal/crash so the shared
    # detector process (RAM + accelerator) is never orphaned.
    _install_parent_death_watchdog()
    if _os.environ.get("AUTOPTZ_MS_DIAG") == "1":
        _logging.getLogger(__name__).setLevel(_logging.INFO)
        _logging.getLogger().setLevel(_logging.INFO)

    # Load the detector OFF the serve path: building it can take seconds, and blocking
    # the supervisor's start() that long would freeze the UI. The holder is swapped in
    # atomically (single assignment under the GIL) once the model is ready.
    holder: dict[str, Any] = {"detector": None}

    def _load_detector() -> None:
        built = None
        raised = False
        try:
            from autoptz.engine.pipeline.pool import build_inference_pool

            pool = build_inference_pool(
                detector_tier=detector_tier, unified_pose=unified_pose, allow_model_download=False
            )
            built = pool.detector() if pool is not None else None
        except Exception:  # noqa: BLE001
            raised = True
            log.warning("model-server: detector build raised; serving empty.", exc_info=True)
        holder["detector"] = built
        if built is None:
            # Surface loudly: otherwise EVERY camera silently never detects and the UI
            # just shows live video with no tracking, with no hint why.
            log.error(
                "model-server: no detector built (tier=%s, build_raised=%s) — all %d "
                "camera(s) will receive EMPTY detections until the app is restarted.",
                detector_tier,
                raised,
                len(cam_ids),
            )

    loader = _threading.Thread(target=_load_detector, name="model-server-load", daemon=True)
    loader.start()

    # Readers attach LAZILY (the camera writers don't exist yet at this point — the
    # supervisor spawns the server before the camera children). serve() attaches each
    # on the camera's first request once its writer is up.
    readers: dict[str, Any] = {}

    def _attach(cam: str) -> Any:
        try:
            return ShmReader(shm_name_for(cam), SERVER_FRAME_H, SERVER_FRAME_W)
        except Exception:  # noqa: BLE001 — writer not up yet; retried next request
            return None

    def _detect(frame: Any) -> Any:
        det = holder["detector"]
        return det.detect(frame) if det is not None else []

    def _detector_ep() -> str:
        det = holder["detector"]
        return str(getattr(det, "ep", "") or "") if det is not None else ""

    ready_ev.set()  # "accepting requests" — detector may still be loading in the background
    try:
        serve(req_q, resp_qs, readers, _detect, stop_ev, attach=_attach, ep_fn=_detector_ep)
    finally:
        for r in readers.values():  # release attached shm views on shutdown
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
