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


class InferenceClient:
    """Drop-in detector that delegates to the shared model-server over IPC.

    ``detect(frame)`` pushes the frame into this camera's shm slot, enqueues a tiny
    request, and blocks for the detections — so it slots in wherever the worker calls
    ``detector.detect(frame)`` with no change to the inference loop. Returns ``[]`` on
    timeout rather than hanging the camera process.
    """

    def __init__(
        self, camera_id: str, req_q: Any, resp_q: Any, shm_writer: Any, timeout_s: float = 5.0
    ) -> None:
        self._cam = camera_id
        self._req_q = req_q
        self._resp_q = resp_q
        self._shm = shm_writer
        self._timeout_s = timeout_s

    @property
    def ep(self) -> str:
        return "model-server"

    def detect(self, frame: Any) -> Any:
        try:
            h, w = int(self._shm.height), int(self._shm.width)
            if frame.shape[:2] != (h, w):
                import cv2  # noqa: PLC0415

                # The shm slot is a fixed size; resize to fit it. (Detections then
                # come back in slot coords — fine when source==slot size; full
                # per-source resolution is a v2 refinement.)
                frame = cv2.resize(frame, (w, h))
            self._shm.push(frame)
            self._req_q.put((self._cam,))
        except Exception:  # noqa: BLE001 — IPC hiccup must not kill the camera loop
            log.debug("inference client %s submit failed", self._cam, exc_info=True)
            return []
        try:
            dets = self._resp_q.get(timeout=self._timeout_s)
        except Exception:  # noqa: BLE001 — server gone / timed out
            return []
        return dets if dets is not None else []


class RemotePool:
    """Pool-shaped wrapper so the worker's pooled detect-stack build uses the IPC
    client as its detector (the tracker stays local per-camera)."""

    detector_ep = "model-server"

    def __init__(self, client: InferenceClient) -> None:
        self._client = client

    def detector(self) -> InferenceClient:
        return self._client


def serve(
    req_q: Any, resp_qs: dict[str, Any], readers: dict[str, Any], detect_fn: Any, stop_ev: Any
) -> None:
    """Server loop: drain detection requests, read each camera's latest frame from
    shm, run ``detect_fn`` once, and reply on that camera's response queue.

    Single-owner of the accelerator → naturally serializes (the accelerator is serial
    anyway) and shares it fairly FIFO across cameras (each camera keeps one request
    outstanding). A detector exception yields an empty result, never a crash.
    """
    while not stop_ev.is_set():
        try:
            msg = req_q.get(timeout=0.2)
        except Exception:  # noqa: BLE001 — empty/closed queue
            continue
        cam = msg[0] if isinstance(msg, tuple) else msg
        reader = readers.get(cam)
        rq = resp_qs.get(cam)
        if reader is None or rq is None:
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
        try:
            rq.put(dets)
        except Exception:  # noqa: BLE001 — client gone
            pass


def run_inference_server(
    req_q: Any,
    resp_qs: dict[str, Any],
    cam_ids: list[str],
    detector_tier: str,
    unified_pose: bool,
    ready_ev: Any,
    stop_ev: Any,
) -> None:
    """Process entrypoint: build the ONE shared detector + per-camera shm readers,
    signal ready, then serve. Spawn-safe (top-level, picklable args). Best-effort —
    a failure to build the detector still serves (returning empty) so cameras don't hang.
    """
    from autoptz.engine.runtime.shm import ShmReader

    detector = None
    try:
        from autoptz.engine.pipeline.pool import build_inference_pool

        pool = build_inference_pool(
            detector_tier=detector_tier, unified_pose=unified_pose, allow_model_download=False
        )
        detector = pool.detector() if pool is not None else None
    except Exception:  # noqa: BLE001
        log.warning("model-server: detector build failed; serving empty.", exc_info=True)

    readers: dict[str, Any] = {}
    for cam in cam_ids:
        try:
            readers[cam] = ShmReader(shm_name_for(cam), SERVER_FRAME_H, SERVER_FRAME_W)
        except Exception:  # noqa: BLE001 — a camera whose writer isn't up yet is skipped
            log.debug("model-server: reader attach failed for %s", cam, exc_info=True)

    def _detect(frame: Any) -> Any:
        return detector.detect(frame) if detector is not None else []

    ready_ev.set()
    serve(req_q, resp_qs, readers, _detect, stop_ev)
