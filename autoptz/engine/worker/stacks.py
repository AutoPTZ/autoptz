"""Detection / face stack construction + ML capability probes for the worker.

Everything about *whether* the ML runtime is importable and *how* to build the
per-camera detector+tracker and face-recognition+identity stacks lives here, so
the worker body deals only with running them. All builders degrade gracefully:
missing models / deps return ``None`` (live-preview-only), never raise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from autoptz.config.models import CameraConfig

log = logging.getLogger(__name__)

# ── ML-stack capability probe (cached) ──────────────────────────────────────────

_ML_AVAILABLE: bool | None = None
_DETECT_RUNTIME_AVAILABLE: bool | None = None


def _probe_modules(mods: tuple[str, ...]) -> bool:
    for mod in mods:
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001 - any import failure disables the stack
            return False
    return True


def detection_runtime_available() -> bool:
    """Return True iff the *detection* runtime (onnxruntime + cv2) is importable.

    This is intentionally decoupled from ``boxmot``: person detection + boxes
    only need ONNX Runtime and OpenCV.  The tracker degrades to a built-in
    lightweight IoU fallback when ``boxmot`` is absent (see
    :mod:`autoptz.engine.pipeline.track`), so detection must NOT be gated on it.
    Cached so the per-frame hot path never re-pays the import-probe cost.
    """
    global _DETECT_RUNTIME_AVAILABLE
    if _DETECT_RUNTIME_AVAILABLE is None:
        _DETECT_RUNTIME_AVAILABLE = _probe_modules(("onnxruntime", "cv2"))
    return bool(_DETECT_RUNTIME_AVAILABLE)


def ml_stack_available() -> bool:
    """Return True iff onnxruntime + boxmot + cv2 can be imported.

    Retained for callers that want the *full* stack (advanced tracker).  New
    code that only needs detection should use :func:`detection_runtime_available`.
    Cached so the per-frame hot path never pays the import-probe cost twice.
    """
    global _ML_AVAILABLE
    if _ML_AVAILABLE is None:
        _ML_AVAILABLE = _probe_modules(("onnxruntime", "boxmot", "cv2"))
    return bool(_ML_AVAILABLE)


# ── detection backend (lazy, graceful) ──────────────────────────────────────────


@dataclass
class _DetectStack:
    detector: Any
    tracker: Any
    ep: str


@dataclass
class _FaceStack:
    """Face recognition + identity gallery for one worker.

    ``recognizer`` is a :class:`~autoptz.engine.pipeline.identify.FaceRecognizer`
    (graceful no-op if insightface is missing); ``service`` is the shared
    :class:`~autoptz.engine.identity.service.IdentityService` gallery.
    """

    recognizer: Any
    service: Any


def _build_face_stack(
    config: CameraConfig,
    identity_service: Any | None,
) -> _FaceStack | None:
    """Try to build the face recognizer + identity gallery; None on any failure.

    Never raises.  When ``insightface`` (or its model/network) is unavailable
    the :class:`FaceRecognizer` reports ``available == False`` and we still
    return a stack so the gallery/CRUD path works — the worker just won't detect
    faces (auto-harvest + identity binding are skipped, manual click-to-track
    keeps working).  Returns ``None`` only if even the gallery can't be built.
    """
    try:
        from autoptz.engine.identity.service import IdentityService
        from autoptz.engine.pipeline.identify import FaceRecognizer

        service = identity_service or IdentityService()
        recognizer = FaceRecognizer()
        return _FaceStack(recognizer=recognizer, service=service)
    except Exception:  # noqa: BLE001 — face stack must never break the worker
        log.warning(
            "camera_id=%s face stack init failed; identity features off.",
            config.id,
            exc_info=True,
        )
        return None


def _xyxy(bbox: Any) -> tuple[float, float, float, float]:
    """A ``BBox`` model → an ``(x1, y1, x2, y2)`` tuple for the ReID embedder."""
    return (float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2))


def _face_crop_png(
    frame: NDArray[np.uint8],
    bbox: tuple[float, float, float, float],
    *,
    pad: float = 0.25,
    max_side: int = 160,
) -> bytes | None:
    """Crop the face *bbox* (with padding) from *frame* and PNG-encode it.

    Returns ``None`` if cv2 is unavailable or encoding fails — the caller then
    harvests the identity without a thumbnail.
    """
    try:
        import cv2  # noqa: PLC0415

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = (x2 - x1), (y2 - y1)
        px, py = bw * pad, bh * pad
        cx1 = max(0, int(x1 - px))
        cy1 = max(0, int(y1 - py))
        cx2 = min(w, int(x2 + px))
        cy2 = min(h, int(y2 + py))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        if max(ch, cw) > max_side:
            scale = max_side / float(max(ch, cw))
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            return None
        return bytes(buf.tobytes())
    except Exception:  # noqa: BLE001
        log.debug("face crop PNG encode failed", exc_info=True)
        return None


def _maybe_quantize_int8(model_path: str) -> str:
    """Return an INT8-quantized detector when ``AUTOPTZ_PRECISION=int8``, else as-is.

    Falls back to the FP32 path on any failure so the detector always builds.
    """
    import os

    if os.environ.get("AUTOPTZ_PRECISION", "auto") != "int8":
        return model_path
    try:
        from autoptz.engine.runtime.models import default_manager

        int8 = default_manager().ensure_detector_int8(model_path)
        if int8:
            log.info("using INT8 detector model %s", int8)
            return int8
    except Exception:  # noqa: BLE001 — never block detector build on quantization
        log.warning("INT8 detector resolution failed; using FP32.", exc_info=True)
    return model_path


def _resolve_model_path(config: CameraConfig) -> str | None:
    """Best-effort lookup of a usable detector model.

    Delegates to :class:`~autoptz.engine.runtime.models.ModelManager`, which
    honours the ``AUTOPTZ_MODEL_PATH`` override, reuses a cached ONNX, or
    downloads + exports a YOLO11 ONNX on first run.  Never raises — returns
    ``None`` (live-preview-only) if the model can't be obtained.
    """
    try:
        from autoptz.engine.runtime.models import default_manager

        return default_manager().ensure_detector()
    except Exception:  # noqa: BLE001 — model bootstrap must never break the worker
        log.warning(
            "camera_id=%s detector model resolution failed; live-preview-only.",
            config.id,
            exc_info=True,
        )
        return None


# One-time log guards so the "detector ready / no model" lines are emitted once
# per process rather than once per camera worker.
_LOGGED_DETECTOR_READY = False
_LOGGED_NO_DETECTOR = False


def _log_no_detector_once() -> None:
    """Emit the actionable 'no detector' INFO line a single time per process."""
    global _LOGGED_NO_DETECTOR
    if _LOGGED_NO_DETECTOR:
        return
    _LOGGED_NO_DETECTOR = True
    log.info(
        "no detector model found — boxes disabled; run tools/fetch_models.py "
        "or install ultralytics (live-preview-only).",
    )


def _log_detector_ready_once(model_path: str, ep: str) -> None:
    """Emit the 'detector ready' INFO line a single time per process."""
    global _LOGGED_DETECTOR_READY
    if _LOGGED_DETECTOR_READY:
        return
    _LOGGED_DETECTOR_READY = True
    log.info("detector ready (model=%s, ep=%s)", model_path, ep)


def _want_unified_pose(config: CameraConfig) -> bool:
    """True iff this camera wants the unified pose detector (config or env)."""
    import os

    if getattr(config.tracking, "unified_pose", False):
        return True
    return os.environ.get("AUTOPTZ_UNIFIED_POSE", "").strip().lower() in ("1", "true", "yes", "on")


def _build_detect_stack(config: CameraConfig) -> _DetectStack | None:
    """Try to build a PersonDetector + Tracker; return None on any failure.

    Never raises — a missing model file or ML dependency degrades to
    live-preview-only.  Emits a clear one-time INFO line either way so the
    operator can tell whether detection/overlays will appear.

    Gated only on the detection runtime (onnxruntime + cv2); ``boxmot`` is NOT
    required — :class:`~autoptz.engine.pipeline.track.Tracker` falls back to a
    built-in lightweight IoU tracker when it is absent.
    """
    if not detection_runtime_available():
        _log_no_detector_once()
        return None

    model_path = _resolve_model_path(config)
    if model_path is None:
        _log_no_detector_once()
        return None
    model_path = _maybe_quantize_int8(model_path)

    try:
        from autoptz.engine.pipeline.detect import PersonDetector
        from autoptz.engine.pipeline.track import Tracker

        detector: Any
        if _want_unified_pose(config):
            try:
                from autoptz.engine.pipeline.pose_detect import PoseDetector

                detector = PoseDetector(detect_interval=config.tracking.detect_interval)
            except Exception:  # noqa: BLE001 — unified must never disable detection
                log.warning(
                    "camera_id=%s unified pose detector unavailable; plain detector.",
                    config.id,
                    exc_info=True,
                )
                detector = PersonDetector(
                    model_path=model_path,
                    detect_interval=config.tracking.detect_interval,
                )
        else:
            detector = PersonDetector(
                model_path=model_path,
                detect_interval=config.tracking.detect_interval,
            )
        # NOTE: the *tracker's* internal BoT-SORT ``reid_model`` stays unset —
        # BoxMOT 19 wants a built ReID object (not a weights path) and only fails
        # at update() time, so the tracker runs motion-only (robust).  Appearance
        # ReID instead lives in a separate, gated recovery layer
        # (``_maybe_reid_recover`` + ``pipeline.reid.BodyReID``): when
        # ``_reid_active`` (global "reid" feature + stable mode) it re-binds the
        # target onto the right track after an occlusion.  Identity stability
        # also comes from
        # face-recognition de-duplication in ``_maybe_identify`` (one person → one
        # identity regardless of track-ID churn).
        tracker = Tracker(
            tracker_type=config.tracking.tracker,
            coast_window=config.tracking.coast_window_ms / 1000.0,
        )
        _log_detector_ready_once(model_path, f"{detector.ep} ({detector.precision})")
        return _DetectStack(detector=detector, tracker=tracker, ep=detector.ep)
    except Exception:  # noqa: BLE001
        log.warning("Detector/tracker init failed; live-preview-only.", exc_info=True)
        _log_no_detector_once()
        return None
