"""Process-wide shared inference pool: build the heavy models once, share them.

Why
---
Every camera previously built its **own** YOLO :class:`PersonDetector`, its own
insightface stack, and its own :class:`PoseEstimator`.  With several cameras that
multiplied the ONNX session count (and the model download / EP-init cost) by the
camera count for no benefit — the detector / face recogniser / pose estimator
are all *stateless* across cameras (the only per-camera state is the boxmot
**tracker**, which stays per-worker).  This module builds each heavy model
**once** for the whole app and hands the same instance to every worker.

Thread-safety
-------------
ONNX Runtime's ``InferenceSession.run`` is thread-safe, so the shared
:class:`PersonDetector` can be called concurrently from every camera thread
without a lock.  The insightface ``FaceAnalysis.get`` wrapper and the pose
``estimate`` path mutate a little internal scratch state, so the pool wraps those
two calls behind a :class:`threading.Lock` via thin **proxies** (see
:class:`_LockedFace` / :class:`_LockedPose`) — the worker calls them with the
exact same API as the bare objects.

Laziness & graceful degradation
-------------------------------
Each accessor (:meth:`InferencePool.detector` / :meth:`~.face` / :meth:`~.pose`)
builds its model on first use, caches the result (including a ``None`` failure so
it is not retried every tick), and returns ``None`` on any failure.  A missing
model / dependency therefore degrades exactly as the per-worker path did:
live-preview-only, no crash, one debug/info line.

The build logic mirrors what :mod:`autoptz.engine.camera_worker` did per-worker
(EP selection via :mod:`autoptz.engine.runtime.inference`, model-path resolution
via :class:`~autoptz.engine.runtime.models.ModelManager`); the worker keeps that
per-worker fallback for tests/fakes that do not inject a pool.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from autoptz.engine.pipeline.framing import Keypoint

log = logging.getLogger(__name__)


# ── locked proxies (serialise non-reentrant wrappers) ────────────────────────────


class _LockedFace:
    """Thin proxy serialising the shared :class:`FaceRecognizer` ``detect`` call.

    insightface's ``FaceAnalysis.get`` keeps a little internal scratch state that
    is not safe to call concurrently from several camera threads, so the single
    shared recogniser is funnelled through one lock here.  ``match`` is delegated
    untouched (it only reads the gallery, which has its own re-entrant lock) and
    ``available`` / ``threshold`` pass through so the worker's existing checks
    work verbatim.
    """

    def __init__(self, recognizer: Any, lock: threading.Lock) -> None:
        self._recognizer = recognizer
        self._lock = lock

    @property
    def available(self) -> bool:
        return bool(getattr(self._recognizer, "available", False))

    @property
    def threshold(self) -> float:
        return float(getattr(self._recognizer, "threshold", 0.0))

    def detect(self, frame: NDArray[np.uint8]) -> list[Any]:
        with self._lock:
            return self._recognizer.detect(frame)

    def match(self, *args: Any, **kwargs: Any) -> Any:
        # Gallery match reads the IdentityService (its own RLock); no extra
        # serialisation needed here.
        return self._recognizer.match(*args, **kwargs)


class _LockedPose:
    """Thin proxy serialising the shared :class:`PoseEstimator` ``estimate`` call.

    The pose estimator's letterbox + ORT path reuses small scratch buffers; one
    lock keeps concurrent target crops from several cameras from racing.  ORT
    ``run`` itself is thread-safe, so this lock is conservative rather than
    strictly required, but pose runs only a few Hz on one target per camera so
    the contention is negligible.
    """

    def __init__(self, estimator: Any, lock: threading.Lock) -> None:
        self._estimator = estimator
        self._lock = lock

    @property
    def available(self) -> bool:
        return bool(getattr(self._estimator, "available", False))

    @property
    def ep(self) -> str:
        return str(getattr(self._estimator, "ep", ""))

    def estimate(
        self,
        frame: NDArray[np.uint8],
        bbox: tuple[float, float, float, float],
    ) -> list[Keypoint] | None:
        with self._lock:
            return self._estimator.estimate(frame, bbox)


# ── the pool ─────────────────────────────────────────────────────────────────────


class InferencePool:
    """Lazily builds + caches the shared detector / face / pose models.

    One instance lives per process (built by :func:`build_inference_pool` and
    injected into every worker via ``worker.set_inference_pool``).  Each accessor
    builds its model on first call under a build lock, caches the result (even a
    ``None`` failure, so a missing model is not retried every tick), and returns
    the shared instance — or ``None`` on any failure (graceful degradation).

    The boxmot tracker is intentionally **not** here: it holds per-camera state
    and must stay per-worker.
    """

    def __init__(self) -> None:
        # Guards the lazy *build* of each model (not their per-call use).
        self._build_lock = threading.Lock()

        self._detector: Any | None = None
        self._detector_built = False
        self._detector_ep = ""

        self._face: _LockedFace | None = None
        self._face_built = False
        self._face_call_lock = threading.Lock()

        self._pose: _LockedPose | None = None
        self._pose_built = False
        self._pose_call_lock = threading.Lock()

    # ── detector ────────────────────────────────────────────────────────────────

    def detector(self) -> Any | None:
        """Return the shared :class:`PersonDetector`, or ``None`` if unavailable.

        ORT ``run`` is thread-safe, so the returned detector is shared directly
        (no proxy/lock) across every camera thread.
        """
        if self._detector_built:
            return self._detector
        with self._build_lock:
            if self._detector_built:
                return self._detector
            self._detector = self._build_detector()
            self._detector_built = True
        return self._detector

    @property
    def detector_ep(self) -> str:
        """Active ORT EP of the shared detector ("" until built / on failure)."""
        return self._detector_ep

    def _build_detector(self) -> Any | None:
        """Resolve the model + build the shared PersonDetector; ``None`` on failure.

        Mirrors the per-worker build that lived in ``camera_worker`` — gated only
        on the detection runtime (onnxruntime + cv2); boxmot is NOT required.
        The ``detect_interval`` is left at the default (every frame): the shared
        detector serves all cameras, so a per-camera interval can't be baked in
        here (the previous per-worker interval was a minor pacing knob).
        """
        try:
            from autoptz.engine.camera_worker import detection_runtime_available
        except Exception:  # noqa: BLE001
            return None
        if not detection_runtime_available():
            log.debug("inference pool: detection runtime unavailable; no detector.")
            return None

        try:
            from autoptz.engine.runtime.models import default_manager

            model_path = default_manager().ensure_detector()
        except Exception:  # noqa: BLE001 — model bootstrap must never break startup
            log.warning("inference pool: detector model resolution failed.",
                        exc_info=True)
            return None
        if model_path is None:
            log.debug("inference pool: no detector model; live-preview-only.")
            return None

        try:
            from autoptz.engine.pipeline.detect import PersonDetector

            detector = PersonDetector(model_path=model_path)
            self._detector_ep = detector.ep
            log.info("inference pool: shared detector ready (model=%s, ep=%s)",
                     model_path, detector.ep)
            return detector
        except Exception:  # noqa: BLE001
            log.warning("inference pool: detector init failed; live-preview-only.",
                        exc_info=True)
            return None

    # ── face recogniser ───────────────────────────────────────────────────────────

    def face(self) -> _LockedFace | None:
        """Return the shared (lock-wrapped) face recogniser, or ``None``.

        The wrapper exposes ``available`` / ``threshold`` / ``detect`` / ``match``
        identically to a bare :class:`FaceRecognizer`, so the worker uses it the
        same way.  ``detect`` is serialised by an internal lock (insightface's
        ``get`` is not reentrant); ``match`` reads the gallery directly.
        """
        if self._face_built:
            return self._face
        with self._build_lock:
            if self._face_built:
                return self._face
            self._face = self._build_face()
            self._face_built = True
        return self._face

    def _build_face(self) -> _LockedFace | None:
        """Build the shared FaceRecognizer wrapped in a serialising proxy."""
        try:
            from autoptz.engine.pipeline.identify import FaceRecognizer

            recognizer = FaceRecognizer()
            return _LockedFace(recognizer, self._face_call_lock)
        except Exception:  # noqa: BLE001 — face stack must never break startup
            log.warning("inference pool: face recogniser init failed; "
                        "identity features off.", exc_info=True)
            return None

    # ── pose estimator ─────────────────────────────────────────────────────────────

    def pose(self) -> _LockedPose | None:
        """Return the shared (lock-wrapped) pose estimator, or ``None``.

        Returns a wrapper whose ``available`` / ``estimate`` mirror a bare
        :class:`PoseEstimator`.  ``estimate`` is serialised by an internal lock.
        ``None`` when the detection runtime or the pose model is unavailable
        (pose-stable framing then degrades to the bbox aim).
        """
        if self._pose_built:
            return self._pose
        with self._build_lock:
            if self._pose_built:
                return self._pose
            self._pose = self._build_pose()
            self._pose_built = True
        return self._pose

    def _build_pose(self) -> _LockedPose | None:
        """Build the shared PoseEstimator wrapped in a serialising proxy."""
        try:
            from autoptz.engine.camera_worker import detection_runtime_available
        except Exception:  # noqa: BLE001
            return None
        if not detection_runtime_available():
            return None
        try:
            from autoptz.engine.pipeline.pose import PoseEstimator

            estimator = PoseEstimator()
            return _LockedPose(estimator, self._pose_call_lock)
        except Exception:  # noqa: BLE001 — pose must never break startup
            log.debug("inference pool: pose estimator init failed; bbox aim only.",
                      exc_info=True)
            return None


# ── factory ──────────────────────────────────────────────────────────────────────


def build_inference_pool() -> InferencePool:
    """Return a fresh :class:`InferencePool` (models built lazily on first use).

    The supervisor calls this once and injects the result into every worker via
    ``worker.set_inference_pool``.  Construction is cheap — no model is loaded
    until the first ``detector()`` / ``face()`` / ``pose()`` call — so this never
    blocks startup and never raises.
    """
    return InferencePool()
