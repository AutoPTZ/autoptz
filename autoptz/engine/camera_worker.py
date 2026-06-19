"""Per-camera pipeline worker: ingest → (detect → track) → telemetry + live preview.

P0 implementation
-----------------
A :class:`CameraWorker` runs one camera on a background thread.  Each tick it:

1. Reads one BGR frame from a :class:`FrameSource` (USB / RTSP / NDI, or a
   fake injected source in tests).
2. Pushes the frame into a :class:`~autoptz.engine.runtime.shm.ShmWriter` named
   **exactly** ``cam_{camera_id[:8]}_preview`` so the UI's ``ShmFrameSource``
   can read it for the live tile.
3. If ``onnxruntime`` + ``boxmot`` + a usable model are importable **and**
   tracking is enabled, runs :class:`PersonDetector` + :class:`Tracker` to
   produce tracks.  Otherwise it skips detection gracefully and still delivers
   live preview + fps + empty tracks.  **Missing ML deps / model never hard-fail.**
4. Emits a :class:`TelemetryMsg` (~``telemetry_hz``, default 10 Hz) via a callback.

Per-camera commands are honoured between ticks:
``enableTracking(bool)``, ``setTarget(track_id|None)``,
``ptzNudge(pan, tilt, zoom)`` (drives the PTZ controller/backend if one is
configured; otherwise a safe no-op), and ``updateConfig(CameraConfig)``.

Threading model is per-thread for P0; process-per-camera is the future
hardening step (see ``supervisor.py``).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from autoptz.config.models import AIM_REGION_FRACTION, TrackingConfig
from autoptz.engine.runtime.messages import (
    BBox,
    FaceBox,
    HealthInfo,
    HealthState,
    PoseKeypoint,
    PTZState,
    TelemetryMsg,
    TrackInfo,
)

if TYPE_CHECKING:
    from autoptz.config.models import CameraConfig, IdentityRecord
    from autoptz.engine.runtime.shm import ShmWriter

log = logging.getLogger(__name__)

# Default preview dimensions — MUST match CameraRecord.shm_width / shm_height
# defaults in autoptz.ui.engine_client so the provider attaches with the same
# shape it reads.
_PREVIEW_W = 1280
_PREVIEW_H = 720

_DEFAULT_TELEMETRY_HZ = 10.0

# How long a manual PTZ nudge suspends auto control before auto resumes.
_MANUAL_OVERRIDE_WINDOW_S = 1.5

# Face stack run-rate: detect/embed faces a few times a second on the target
# region (per docs/03 §3.8), not every frame.  Period in seconds.
_FACE_INTERVAL_S = 0.25
_FACE_TTL_S = 0.45

# ── auto-harvest quality gates ───────────────────────────────────────────────────
# Auto-harvesting a NEW "Person N" is deliberately strict so the user rarely has
# to merge junk identities.  A face must clear ALL of these before it earns a new
# in-memory identity: a comfortable crop size, a confident SCRFD detection, a
# roughly frontal pose, a non-blurry crop, AND a low best-match similarity
# against the existing gallery (so a known person is re-bound, never re-harvested).

# Minimum face crop size (px, longer edge) to auto-harvest.  Raised from the old
# 60 px floor so distant/tiny faces (which embed poorly) are skipped.
_MIN_HARVEST_FACE_PX = 90.0

# Minimum SCRFD detection confidence for a harvest-worthy face.
_MIN_HARVEST_DET_SCORE = 0.60

# Maximum allowed face yaw (frontal-ness) for a harvest, expressed as the
# fractional horizontal offset of the nose from the eye-midpoint relative to the
# inter-ocular distance.  ~0.0 = perfectly frontal; we reject clear profiles.
_MAX_HARVEST_YAW_RATIO = 0.55

# Laplacian-variance floor below which a face crop is considered too blurry to
# auto-harvest (motion blur / soft focus embed poorly).
_MIN_HARVEST_SHARPNESS = 40.0

# A face is only "new" if its best similarity to the *whole* gallery (labeled +
# unlabeled) is below this — keeps the same person from spawning duplicates even
# before the recogniser's match threshold is reached.
_HARVEST_NOVELTY_MAX_SIM = 0.30

# Cooldown between auto-harvesting two distinct unlabeled identities so a single
# busy scene doesn't spray dozens of "Person N" rows at the UI.
_HARVEST_COOLDOWN_S = 2.0

# How often the worker emits a periodic frame-drop summary (INFO) while drops
# continue to accrue — keeps the console honest about a flaky source without
# spamming a line per missed frame.
_DROP_LOG_INTERVAL_S = 10.0

# How often pose keypoints are (re)estimated for the active target.  Pose runs
# only on the single target crop, not every frame; between runs the worker reuses
# the last keypoints (the bbox still tracks the person each frame, so a slightly
# stale torso point is fine and far cheaper than per-frame pose inference).
_POSE_INTERVAL_S = 0.2

# How often (seconds) to run OSNet appearance ReID: refresh the target's template
# while it's visible, or attempt recovery while it's lost.  Throttled because the
# embedder is moderately heavy and per-frame recovery is unnecessary.
_REID_INTERVAL_S = 0.25

# EMA weight for the pose-derived aim point: lower = smoother/laggier.  Light
# smoothing so noisy keypoint regression doesn't jitter the framing.
_POSE_AIM_ALPHA = 0.4

# Map the unified ``tracking.framing`` preset onto a pose torso bias (see
# framing.torso_aim_point).  face→head/shoulders, upper_body→shoulders,
# full_body→shoulder↔hip midpoint.  The four names line up 1:1 with the
# AIM_REGION_FRACTION keys, so the same value drives both the bbox-fallback aim
# fraction and the pose torso anchor.
_AIM_REGION_POSE_BIAS: dict[str, str] = {
    "face": "face",
    "head_shoulders": "head_shoulders",
    "upper_body": "upper_body",
    "full_body": "full_body",
}

# Default feature switches — every subsystem on until the supervisor pushes a
# narrower set via ``set_features``.
_DEFAULT_FEATURES: dict[str, bool] = {
    "detection": True,
    "tracking": True,
    "face_recognition": True,
    "pose": True,
}


def _resolve_framing(tracking: TrackingConfig) -> str:
    """Return the effective framing preset for aim/pose.

    ``tracking.framing`` is the unified user-facing control; when it has been
    moved off its default it wins.  The legacy ``tracking.aim_region`` is honoured
    as a fallback when ``framing`` is still at its default (older configs / the
    current Properties panel still write ``aim_region``), so both controls keep
    working without the two ever disagreeing in practice.
    """
    default = TrackingConfig.model_fields["framing"].default
    framing = getattr(tracking, "framing", default)
    if framing != default:
        return framing
    return getattr(tracking, "aim_region", default)


# ── frame source abstraction ───────────────────────────────────────────────────


@runtime_checkable
class FrameSource(Protocol):
    """Minimal synchronous frame source the worker drives itself.

    Implementations open a device/stream, hand back one BGR frame per
    ``read()`` (or ``None`` on a transient miss), and release on ``close()``.
    Tests inject a fake implementation; production wraps the ingest adapters.
    """

    def open(self) -> bool:
        """Open the source.  Return ``True`` on success."""
        ...

    def read(self) -> NDArray[np.uint8] | None:
        """Return one BGR (H, W, 3) frame, or ``None`` on a transient miss."""
        ...

    def close(self) -> None:
        """Release the source."""
        ...


class _AdapterFrameSource:
    """Adapt an ingest ``SourceAdapter`` subclass to the synchronous FrameSource.

    The ingest adapters expose synchronous ``_open`` / ``_read_frame`` /
    ``_close`` primitives (their own capture thread is *not* started here — the
    worker owns the loop so it can also feed detection).
    """

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._last_read_t = 0.0

    def open(self) -> bool:
        return bool(self._adapter._open())

    def read(self) -> NDArray[np.uint8] | None:
        self._pace_read()
        frame = self._adapter._read_frame()
        if frame is not None:
            self._last_read_t = time.monotonic()
        return frame

    def close(self) -> None:
        try:
            self._adapter._close()
        except Exception:  # noqa: BLE001
            log.debug("frame source _close raised", exc_info=True)

    def set_target_fps(self, fps: float) -> None:
        """Forward a live fps change to the wrapped ingest adapter (best-effort)."""
        fn = getattr(self._adapter, "set_target_fps", None)
        if callable(fn):
            fn(fps)

    def _pace_read(self) -> None:
        """Throttle worker-owned direct reads to the adapter's target fps."""
        target = float(getattr(self._adapter, "_target_fps", 0.0) or 0.0)
        if target <= 0.0 or self._last_read_t <= 0.0:
            return
        wait = (1.0 / max(1.0, target)) - (time.monotonic() - self._last_read_t)
        if wait > 0.001:
            time.sleep(wait)

    def source_fps_cap(self) -> float | None:
        """Return the adapter's detected hardware fps ceiling, or ``None``."""
        try:
            return self._adapter.status.source_fps_cap
        except Exception:  # noqa: BLE001
            return None


def build_frame_source(camera_id: str, config: CameraConfig) -> FrameSource:
    """Construct the right ingest-adapter-backed FrameSource for *config*.

    Importing ``ingest`` pulls in ``cv2``; if that is unavailable we raise so
    the worker can fall back to a no-signal state without crashing the engine.
    """
    from autoptz.engine.pipeline.ingest import NDIAdapter, RTSPAdapter, USBAdapter

    source = config.source
    target_fps = source.fps
    stall = config.reconnect.stall_timeout_s

    if source.type == "usb":
        dev = _resolve_usb_device(source)
        adapter: Any = USBAdapter(camera_id, source=dev, target_fps=target_fps,
                                  stall_timeout=stall,
                                  unique_id=getattr(source, "unique_id", None))
    elif source.type in ("rtsp", "onvif"):
        adapter = RTSPAdapter(camera_id, url=source.address, target_fps=target_fps,
                              stall_timeout=stall)
    elif source.type == "ndi":
        adapter = NDIAdapter(camera_id, ndi_name=_strip_scheme(source.address, "ndi://"),
                             target_fps=target_fps, stall_timeout=stall)
    else:  # pragma: no cover - validated by pydantic Literal
        adapter = USBAdapter(camera_id, source=0, target_fps=target_fps,
                             stall_timeout=stall)

    return _AdapterFrameSource(adapter)


def _resolve_usb_device(source: Any) -> int | str:
    """Resolve the *fallback* cv2 source (index/path) for a USB ``SourceConfig``.

    This is only the fallback the ``USBAdapter`` uses when it cannot resolve a
    stable ``unique_id`` to a verified capture index at open time (off macOS, or
    when the device is gone).  When a ``unique_id`` is present we still pre-seed
    a best-effort current index from discovery so a non-macOS adapter — which
    has no uniqueID resolution — opens the right enumeration slot; macOS gets the
    authoritative verified resolution inside :class:`USBAdapter`.
    """
    unique_id = getattr(source, "unique_id", None)
    if unique_id:
        idx = _index_for_unique_id(unique_id)
        if idx is not None:
            return idx
        log.debug("USB unique_id=%s not in current enumeration; "
                  "falling back to address index.", unique_id)
    return _parse_usb_index(getattr(source, "address", ""))


def _index_for_unique_id(unique_id: str) -> int | None:
    """Look up the enumerated camera index whose ``unique_id`` matches.

    Returns ``None`` if enumeration is unavailable or no device matches.
    """
    try:
        from autoptz.engine.discovery.usb import enumerate_cameras

        for cam in enumerate_cameras():
            if cam.get("unique_id") == unique_id:
                return int(cam["index"])
    except Exception:  # noqa: BLE001 — enumeration must never break source build
        log.debug("USB enumeration lookup failed for unique_id=%s", unique_id,
                  exc_info=True)
    return None


def _parse_usb_index(address: str) -> int | str:
    """Map a ``usb://N`` address (or bare index/path) to a cv2 source."""
    raw = _strip_scheme(address, "usb://")
    if raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        return raw  # device path


def _strip_scheme(address: str, scheme: str) -> str:
    return address[len(scheme):] if address.startswith(scheme) else address


def _sanitize_address(address: str | None) -> str:
    """Strip any ``user:pass@`` credentials from a URL for safe logging.

    ``rtsp://user:pass@host/stream`` → ``rtsp://host/stream``.  Non-URL
    addresses (bare USB indices/paths) pass through unchanged.  Never raises.
    """
    if not address:
        return ""
    try:
        import re

        return re.sub(r"(\w+://)[^@/]*@", r"\1", str(address))
    except Exception:  # noqa: BLE001
        return str(address)


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
    config: CameraConfig, identity_service: Any | None,
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
        log.warning("camera_id=%s face stack init failed; identity features off.",
                    config.id, exc_info=True)
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
        log.warning("camera_id=%s detector model resolution failed; "
                    "live-preview-only.", config.id, exc_info=True)
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

    try:
        from autoptz.engine.pipeline.detect import PersonDetector
        from autoptz.engine.pipeline.track import Tracker

        detector = PersonDetector(
            model_path=model_path,
            detect_interval=config.tracking.detect_interval,
        )
        # NOTE: the *tracker's* internal BoT-SORT ``reid_model`` stays unset —
        # BoxMOT 19 wants a built ReID object (not a weights path) and only fails
        # at update() time, so the tracker runs motion-only (robust).  Appearance
        # ReID instead lives in a separate, gated recovery layer
        # (``_maybe_reid_recover`` + ``pipeline.reid.BodyReID``): when
        # ``config.tracking.reid_enabled`` is on it re-binds the target onto the
        # right track after an occlusion.  Identity stability also comes from
        # face-recognition de-duplication in ``_maybe_identify`` (one person → one
        # identity regardless of track-ID churn).
        tracker = Tracker(
            tracker_type=config.tracking.tracker,
            coast_window=config.tracking.coast_window_ms / 1000.0,
        )
        _log_detector_ready_once(model_path, detector.ep)
        return _DetectStack(detector=detector, tracker=tracker, ep=detector.ep)
    except Exception:  # noqa: BLE001
        log.warning("Detector/tracker init failed; live-preview-only.", exc_info=True)
        _log_no_detector_once()
        return None


# ── camera worker ───────────────────────────────────────────────────────────────


class CameraWorker:
    """Runs ingest + optional detection/tracking for a single camera on a thread.

    Args:
        camera_id:    Stable camera UUID.
        config:       The camera's :class:`CameraConfig`.
        on_telemetry: Callback invoked (from the worker thread) with each
                      :class:`TelemetryMsg`.  Must be thread-safe / non-blocking.
        frame_source: Injected FrameSource (tests).  If ``None`` one is built
                      from ``config`` lazily inside the thread.
        shm_writer:   Injected ShmWriter (tests).  If ``None`` one is created
                      lazily inside the thread, named ``cam_{id[:8]}_preview``.
        ptz_controller: Optional pre-built PTZController to drive on PTZ commands.
        telemetry_hz: Telemetry emission rate (default 10 Hz).
    """

    def __init__(
        self,
        camera_id: str,
        config: CameraConfig,
        on_telemetry: Callable[[TelemetryMsg], None],
        *,
        frame_source: FrameSource | None = None,
        shm_writer: ShmWriter | None = None,
        ptz_controller: Any | None = None,
        ptz_backend: Any | None = None,
        on_identity: Callable[[IdentityRecord], None] | None = None,
        identity_service: Any | None = None,
        face_stack: Any | None = None,
        telemetry_hz: float = _DEFAULT_TELEMETRY_HZ,
    ) -> None:
        self.camera_id = camera_id
        self.config = config
        self._on_telemetry = on_telemetry
        self._on_identity = on_identity
        self._injected_source = frame_source
        self._injected_shm = shm_writer
        self._telemetry_period = 1.0 / max(1.0, telemetry_hz)

        # ── face / identity wiring ──────────────────────────────────────────────
        # The face stack (insightface + the gallery service) is built lazily in
        # the worker thread unless injected (tests).  When a worker-thread face
        # match annotates a track we publish identity+confidence in telemetry;
        # unmatched "good" faces are auto-harvested into memory-only unlabeled
        # identities and pushed to the UI via ``on_identity``.
        self._injected_identity_service = identity_service
        self._injected_face_stack = face_stack
        self._face: Any | None = None
        self._last_face_t = 0.0
        self._last_harvest_t = 0.0
        self._last_crop_t = 0.0
        # Most recent face→identity bindings seen this tick: track_id → (id, conf).
        # track_id → (identity_id, display_name, score)
        self._track_identity: dict[int, tuple[str, str, float]] = {}
        # Click-to-assign: tracks the operator named, awaiting a detected face to
        # bind the embedding to.  track_id → (identity_id, display_name,
        # click_xy_norm).  The click point keeps enrollment tied to the exact
        # face the operator clicked when several faces overlap one person box.
        self._pending_enroll: dict[int, tuple[str, str, tuple[float, float] | None]] = {}
        # Last detected faces (pixel-space), for the optional face overlay; rebuilt
        # each face tick and published in telemetry so the UI can draw them.
        self._last_faces: list[FaceBox] = []
        self._last_faces_t = 0.0
        self._last_face_track_ids: set[int] = set()
        # Identity the operator asked us to follow ("track when found"); single
        # target per camera — supersedes an explicit track id when its identity
        # is detected on a live track.
        self._target_identity_id: str | None = config.target.identity_id

        self.shm_name = f"cam_{camera_id[:8]}_preview"

        # ── PTZ wiring ──────────────────────────────────────────────────────────
        # `_ptz` is the closed-loop PTZController (built lazily from config if not
        # injected); `_ptz_backend` is the raw backend used for direct manual
        # nudges and the source of the real PTZState in telemetry.
        #
        # Back-compat: a caller/test may inject either a PTZController (has step())
        # or a bare backend (has move_velocity()) as `ptz_controller`.
        self._ptz: Any | None = None
        self._ptz_backend: Any | None = ptz_backend
        self._ptz_owned = False  # True when we built the controller ourselves
        if ptz_controller is not None:
            if hasattr(ptz_controller, "step"):
                self._ptz = ptz_controller
                if self._ptz_backend is None:
                    self._ptz_backend = getattr(ptz_controller, "_backend", None)
            else:
                # bare backend injected — drive it directly on nudge
                self._ptz_backend = ptz_controller

        # manual-override window: while active, the auto loop is suspended and
        # nudges drive the backend directly; auto resumes after it expires.
        self._manual_override_until: float = 0.0

        # last published PTZ snapshot (pan/tilt/zoom/moving/state)
        self._ptz_last_cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)

        # tracking state
        self._tracking_enabled = config.target.mode != "off"
        self._target_track_id: int | None = None

        # ── pose-stable aim (lazy) ───────────────────────────────────────────────
        # Optional keypoint estimator for the active target so an extended arm
        # (which grows the person bbox) doesn't yank the framing.  Built lazily
        # the first time auto tracking actually needs an aim point, mirroring the
        # detector/face lazy build, so idle cameras pay nothing.  ``_pose_probed``
        # guards against re-attempting the build every tick once it has failed.
        self._pose: Any | None = None
        self._pose_probed = False
        self._pose_keypoints: list[Any] | None = None  # last keypoints (reused)
        self._pose_kp_track_id: int | None = None       # track they belong to
        self._last_pose_t = 0.0
        self._aim_smoother: Any | None = None  # framing.AimSmoother (lazy)

        # ── appearance ReID recovery (lazy, gated on tracking.reid_enabled) ───────
        # OSNet body-appearance matcher used to re-bind the target onto the right
        # track after an occlusion (built lazily; ``_reid_probed`` stops us from
        # re-attempting a failed/missing-boxmot build every tick).  The template is
        # seeded/refreshed while the target is visible and queried when it's lost.
        self._reid: Any | None = None
        self._reid_probed = False
        self._last_reid_t = 0.0

        # Aim-error velocity estimate (normalized error units / second), fed to the
        # PTZ controller so its feed-forward + motion prediction actually engage —
        # previously the worker passed a hardcoded (0, 0), so the camera only ever
        # chased the subject's *past* position (the "laggy following" complaint).
        self._prev_aim_err: tuple[float, float] | None = None
        self._prev_aim_t: float = 0.0
        self._aim_vel: tuple[float, float] = (0.0, 0.0)

        # runtime state
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._cmd_lock = threading.Lock()
        self._cmd_queue: deque[tuple[str, Any]] = deque()

        # ── async pipeline handoff ───────────────────────────────────────────────
        # The capture thread reads frames, pushes the live preview, and emits
        # telemetry at the full source rate; it stashes the newest frame here and
        # a SINGLE inference thread consumes the latest one (latest-frame-wins) to
        # run detect → track → face → pose → PTZ.  Keeping every inference stage on
        # that one thread preserves their existing sequential relationship (no new
        # concurrency among them) while decoupling capture/preview fps from the
        # heavy ML cost — that's what stops the "30 → 23 → 19 fps" cliff.
        self._frame_lock = threading.Lock()
        self._latest_frame: NDArray[np.uint8] | None = None
        self._latest_frame_id = 0
        self._frame_ready = threading.Event()
        # Most recent inference output, read by the capture thread for telemetry.
        self._last_tracks: list[TrackInfo] = []
        # Serializes PTZ-backend access between the inference thread (which drives
        # motion) and the capture thread (which reads position for telemetry).
        self._ptz_lock = threading.Lock()

        # ── shared inference pool + global feature switches ──────────────────────
        # The supervisor injects a process-wide :class:`InferencePool` (heavy
        # models built once for every camera) via ``set_inference_pool`` before
        # ``start()``.  When present the worker uses the pool's shared detector /
        # face / pose; when absent (tests/fakes never inject one) it builds its
        # own per-worker models exactly as before — that fallback is what keeps
        # the orchestration/tracking tests green.  The boxmot tracker stays
        # per-worker regardless (it holds per-camera state).
        self._pool: Any | None = None
        # Thread-safe global feature flags (detection / tracking /
        # face_recognition / pose); default all-on until the supervisor narrows
        # them.  Read under ``_cmd_lock`` snapshots in the tick.
        self._features: dict[str, bool] = dict(_DEFAULT_FEATURES)

        # owned resources (created in thread)
        self._source: FrameSource | None = None
        self._shm: ShmWriter | None = None
        self._detect: _DetectStack | None = None

        self._seq = 0
        self._fps = 0.0
        self._ep = ""

        # Camera-info telemetry: last-seen source frame size and a cumulative
        # count of read() misses / decode failures.
        self._frame_w = 0
        self._frame_h = 0
        self._dropped_frames = 0

        # Per-frame processing latency (ingest read + detect + track wall time)
        # in milliseconds, published in telemetry for the live stats overlay.
        self._latency_ms = 0.0
        self._ingest_ms = 0.0
        self._detect_ms = 0.0
        self._track_ms = 0.0
        # Face / pose stage cost (ms) from the most recent run that actually
        # executed (both are throttled, so they hold between runs).
        self._face_ms = 0.0
        self._pose_ms = 0.0

        # Periodic diagnostics bookkeeping: when the next dropped-frame summary
        # is due (monotonic seconds) and the count last reported, so we only log
        # when drops actually accrued.
        self._next_drop_log_t = 0.0
        self._last_logged_drops = 0
        # PTZ command-rate-limit so DEBUG nudge/auto lines don't spam at
        # per-frame rate — only log when the command meaningfully changes.
        self._last_logged_ptz: tuple[float, float, float] | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the worker thread (idempotent).

        The preview :class:`ShmWriter` is created **here, synchronously, before**
        the thread starts, so the shared-memory segment exists as soon as
        ``start()`` returns.  This lets the supervisor emit the provider attach
        request right after ``start()`` and have the UI's lazy ShmReader open
        immediately — removing the old attach/writer ordering race that produced
        the blank-navy preview.  The thread still owns frame reads + detection.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._create_shm_writer_eager()
        self._thread = threading.Thread(
            target=self._run, name=f"camworker-{self.camera_id[:8]}", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal stop and block until the thread exits; releases all resources."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── command intake (thread-safe; called from supervisor/command pump) ────────

    def enable_tracking(self, enabled: bool) -> None:
        with self._cmd_lock:
            self._cmd_queue.append(("enable_tracking", bool(enabled)))

    def set_target(self, track_id: int | None) -> None:
        with self._cmd_lock:
            self._cmd_queue.append(("set_target", track_id))

    def set_target_identity(self, identity_id: str | None) -> None:
        """Follow a *named identity* ("track when found").  ``None`` clears it."""
        with self._cmd_lock:
            self._cmd_queue.append(("set_target_identity", identity_id))

    def enroll_track(
        self, track_id: int, identity_id: str, name: str,
        click_x: float | None = None, click_y: float | None = None,
    ) -> None:
        """Bind a clicked track's face to ``identity_id`` (click-to-assign).

        The embedding is captured on the next face tick where a face is detected
        on that track; until then the name shows immediately on the box.
        """
        click = None
        if click_x is not None and click_y is not None:
            click = (
                max(0.0, min(1.0, float(click_x))),
                max(0.0, min(1.0, float(click_y))),
            )
        with self._cmd_lock:
            self._cmd_queue.append(("enroll_track", (track_id, identity_id, name, click)))

    def set_identity_callback(
        self, callback: Callable[[IdentityRecord], None] | None,
    ) -> None:
        """Wire the worker→client identity push (mirrors the telemetry callback).

        The supervisor calls this right after constructing the worker so the
        3-arg ``worker_factory`` contract used by tests stays unchanged.
        """
        self._on_identity = callback

    def set_identity_service(self, service: Any | None) -> None:
        """Inject the shared identity gallery (used when the face stack builds).

        Set by the supervisor before ``start()`` so every worker shares one
        gallery.  Ignored once the face stack has already been built.
        """
        self._injected_identity_service = service

    def set_inference_pool(self, pool: Any | None) -> None:
        """Inject the process-wide shared inference pool (heavy models once).

        Called by the supervisor before ``start()``.  When set, the worker pulls
        its detector / face / pose from the pool instead of building its own; when
        ``None`` (tests/fakes) the worker keeps its per-worker build path.
        """
        self._pool = pool

    def set_features(self, features: dict[str, bool] | None) -> None:
        """Update the global feature switches (thread-safe; merges over defaults).

        Keys: ``detection``, ``tracking``, ``face_recognition``, ``pose`` — each
        defaulting True when absent.  Mirrors the other command setters by
        snapshotting under ``_cmd_lock`` so a live update from the supervisor's
        command pump is seen atomically by the worker tick.
        """
        merged = dict(_DEFAULT_FEATURES)
        if features:
            for key in _DEFAULT_FEATURES:
                if key in features:
                    merged[key] = bool(features[key])
        with self._cmd_lock:
            self._features = merged

    def _feature(self, name: str) -> bool:
        """Thread-safe read of one feature flag (default True when unset)."""
        with self._cmd_lock:
            return bool(self._features.get(name, True))

    def ptz_home(self) -> None:
        """Drive the PTZ backend to its optical home position (queued command)."""
        with self._cmd_lock:
            self._cmd_queue.append(("ptz_home", None))

    def ptz_menu(self) -> None:
        """Toggle the camera's on-screen-display (OSD) menu (queued command)."""
        with self._cmd_lock:
            self._cmd_queue.append(("ptz_menu", None))

    def ptz_nudge(self, pan: float, tilt: float, zoom: float) -> None:
        with self._cmd_lock:
            self._cmd_queue.append(("ptz_nudge", (float(pan), float(tilt), float(zoom))))

    def set_target_fps(self, fps: float) -> None:
        """Change capture/detection pacing **live** (no engine restart)."""
        with self._cmd_lock:
            self._cmd_queue.append(("set_target_fps", float(fps)))

    def save_ptz_preset(self, slot: int) -> None:
        """Store the current PTZ position into hardware preset *slot*."""
        with self._cmd_lock:
            self._cmd_queue.append(("save_ptz_preset", int(slot)))

    def recall_ptz_preset(self, slot: int) -> None:
        """Recall hardware PTZ preset *slot*."""
        with self._cmd_lock:
            self._cmd_queue.append(("recall_ptz_preset", int(slot)))

    def update_config(self, config: CameraConfig) -> None:
        with self._cmd_lock:
            self._cmd_queue.append(("update_config", config))

    def _drain_commands(self) -> None:
        with self._cmd_lock:
            pending = list(self._cmd_queue)
            self._cmd_queue.clear()
        for kind, payload in pending:
            try:
                self._apply_command(kind, payload)
            except Exception:  # noqa: BLE001
                log.warning("camera_id=%s command %s failed", self.camera_id, kind,
                            exc_info=True)

    def _apply_command(self, kind: str, payload: Any) -> None:
        if kind == "enable_tracking":
            self._tracking_enabled = bool(payload)
        elif kind == "set_target":
            # An explicit track id supersedes identity targeting for this camera.
            if payload != self._target_track_id:
                self._reset_pose_aim()
                self._reset_reid()  # drop the old subject's appearance template
            self._target_track_id = payload
            self._target_identity_id = None
        elif kind == "set_target_identity":
            self._target_identity_id = payload
            # Clear the explicit track lock so the identity match takes over once
            # the named person is detected ("track when found").
            self._target_track_id = None
            self._reset_pose_aim()
            self._reset_reid()
        elif kind == "enroll_track":
            if len(payload) == 3:
                track_id, identity_id, name = payload
                click = None
            else:
                track_id, identity_id, name, click = payload
            if track_id is not None and identity_id:
                self._pending_enroll[int(track_id)] = (identity_id, name, click)
                # Show the assigned name on the box immediately; the embedding
                # binds on the next face tick (see _maybe_identify).
                self._track_identity[int(track_id)] = (identity_id, name, 1.0)
        elif kind == "update_config":
            prev_fps = getattr(self.config.source, "fps", None)
            self.config = payload
            # Apply an fps change from a full-config push live too, so the UI's
            # fps slider takes effect whether it routes through updateCameraConfig
            # or the dedicated setTargetFps slot.
            new_fps = getattr(payload.source, "fps", None)
            if new_fps is not None and new_fps != prev_fps:
                self._apply_target_fps(float(new_fps))
            # Push the new PTZ tuning to the live controller (gains, lead time,
            # smoothing, safe zone, loss-recovery) without an engine restart.  The
            # controller was built with the *old* cfg, so this keeps it current.
            ctrl = self._ptz
            if ctrl is not None and hasattr(ctrl, "update_config"):
                try:
                    ctrl.update_config(payload.ptz)
                except Exception:  # noqa: BLE001
                    log.debug("camera_id=%s ptz update_config failed",
                              self.camera_id, exc_info=True)
        elif kind == "set_target_fps":
            self._apply_target_fps(float(payload))
        elif kind == "save_ptz_preset":
            self._save_ptz_preset(int(payload))
        elif kind == "recall_ptz_preset":
            self._recall_ptz_preset(int(payload))
        elif kind == "ptz_home":
            self._ptz_home()
        elif kind == "ptz_menu":
            self._ptz_menu()
        elif kind == "ptz_nudge":
            # Lock the backend so a concurrent telemetry position read (capture
            # thread) can't interleave with this manual move (inference thread).
            with self._ptz_lock:
                self._drive_ptz_nudge(*payload)

    def _apply_target_fps(self, fps: float) -> None:
        """Push a live fps change onto the running frame source (best-effort)."""
        src = self._source
        if src is None:
            return
        fn = getattr(src, "set_target_fps", None)
        if not callable(fn):
            return
        try:
            fn(fps)
            log.info("camera_id=%s target fps set live to %.0f", self.camera_id, fps)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s set_target_fps failed", self.camera_id,
                      exc_info=True)

    def _save_ptz_preset(self, slot: int) -> None:
        """Store the current position into the backend's hardware preset *slot*.

        Safe no-op when no PTZ backend is configured.  Never raises into the
        command pump.
        """
        backend = self._ptz_backend
        if backend is None or not hasattr(backend, "save_preset"):
            log.debug("camera_id=%s save_ptz_preset slot=%d ignored (no backend)",
                      self.camera_id, slot)
            return
        try:
            backend.save_preset(slot)
            log.info("camera_id=%s saved PTZ preset slot=%d", self.camera_id, slot)
        except Exception:  # noqa: BLE001
            log.warning("camera_id=%s save_preset slot=%d failed", self.camera_id,
                        slot, exc_info=True)

    def _recall_ptz_preset(self, slot: int) -> None:
        """Recall the backend's hardware preset *slot*.

        Safe no-op when no PTZ backend is configured.  Never raises into the
        command pump.
        """
        backend = self._ptz_backend
        if backend is None or not hasattr(backend, "goto_preset"):
            log.debug("camera_id=%s recall_ptz_preset slot=%d ignored (no backend)",
                      self.camera_id, slot)
            return
        try:
            backend.goto_preset(slot)
            log.info("camera_id=%s recalled PTZ preset slot=%d", self.camera_id, slot)
        except Exception:  # noqa: BLE001
            log.warning("camera_id=%s goto_preset slot=%d failed", self.camera_id,
                        slot, exc_info=True)

    def _ptz_home(self) -> None:
        """Drive the backend to optical home, if it supports ``home()``.

        Opens a manual-override window so the auto loop doesn't immediately fight
        the recentre.  Safe no-op when no backend is configured or the backend
        lacks ``home()``.  Never raises into the command pump.
        """
        backend = self._ptz_backend
        if backend is None or not hasattr(backend, "home"):
            log.debug("camera_id=%s ptz_home ignored (no backend / unsupported)",
                      self.camera_id)
            return
        self._manual_override_until = time.monotonic() + _MANUAL_OVERRIDE_WINDOW_S
        try:
            backend.home()
            self._ptz_last_cmd = (0.0, 0.0, 0.0)
            log.info("camera_id=%s PTZ home", self.camera_id)
        except Exception:  # noqa: BLE001
            log.warning("camera_id=%s ptz home failed", self.camera_id, exc_info=True)

    def _ptz_menu(self) -> None:
        """Toggle the camera OSD menu, if the backend supports ``osd_menu()``.

        Safe no-op when no backend is configured or the backend lacks
        ``osd_menu()`` (most current backends do).  Never raises into the pump.
        """
        backend = self._ptz_backend
        if backend is None or not hasattr(backend, "osd_menu"):
            log.debug("camera_id=%s ptz_menu ignored (no backend / unsupported)",
                      self.camera_id)
            return
        try:
            backend.osd_menu()
            log.info("camera_id=%s PTZ OSD menu toggled", self.camera_id)
        except Exception:  # noqa: BLE001
            log.warning("camera_id=%s ptz menu failed", self.camera_id, exc_info=True)

    def _drive_ptz_nudge(self, pan: float, tilt: float, zoom: float) -> None:
        """Manual nudge: open a short auto-suspend window and drive the backend.

        Each nudge (re)opens a ~1.5 s manual-override window during which the auto
        control loop is suspended; the nudge command itself is sent straight to
        the backend so the operator's joystick/D-pad moves the camera immediately.
        After the window expires with no further nudge, auto control resumes.

        Safe no-op when no PTZ backend is configured.
        """
        self._manual_override_until = time.monotonic() + _MANUAL_OVERRIDE_WINDOW_S

        backend = self._ptz_backend
        if backend is None:
            return
        try:
            backend.move_velocity(pan, tilt, zoom)
            self._ptz_last_cmd = (pan, tilt, zoom)
            log.debug("camera_id=%s ptz nudge backend=%s pan=%.3f tilt=%.3f zoom=%.3f",
                      self.camera_id, self.config.ptz.backend, pan, tilt, zoom)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s ptz nudge failed", self.camera_id, exc_info=True)

    def _manual_override_active(self, now: float) -> bool:
        return now < self._manual_override_until

    def _drive_ptz_auto(
        self,
        tracks: list[TrackInfo],
        frame: NDArray[np.uint8] | None,
        now: float,
    ) -> None:
        """Feed the controller this tick's target error so it drives the backend.

        - No controller, or inside a manual-override window → skip (manual owns
          the camera; the controller is fed nothing so it does not fight it).
        - Tracking enabled with a resolvable target → compute the normalized
          center error (bbox-center vs frame-center, ~[-1, 1]) and the subject
          height fraction (bbox height / frame height) and step the controller
          with ``track_active=True`` so auto-zoom and the PD loop run.
        - Otherwise → step with ``track_active=False`` so the controller runs its
          coast→search behaviour and ultimately stops the backend.
        """
        ctrl = self._ptz
        if ctrl is None:
            return
        if self._manual_override_active(now):
            return

        # Global ``tracking`` switch hard-gates auto-following: when off we never
        # drive the camera toward a target (the controller is stepped idle so it
        # coasts→stops), even if a target track is locked.
        tracking_on = self._feature("tracking")
        target = self._resolve_target_track(tracks)
        # A coasting (LOST) target is a STALE box — don't chase it.  Stepping the
        # controller with track_active=False lets it run its graceful coast→search
        # →stop instead of driving the PTZ toward where the subject no longer is
        # (the "moves the camera for no reason while the box lingers" bug).
        if (
            target is not None
            and not target.lost
            and frame is not None
            and self._tracking_enabled
            and tracking_on
        ):
            err, height = self._track_error(target, frame, now)
            vel = self._estimate_aim_velocity(err, now)
            try:
                pan, tilt, zoom = ctrl.step(err, vel, height,
                                            track_active=True, t=now)
                self._ptz_last_cmd = (pan, tilt, zoom)
                self._log_ptz_cmd("auto", pan, tilt, zoom)
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s ptz auto step failed", self.camera_id,
                          exc_info=True)
        else:
            # No target → drop the velocity estimate so a re-acquire starts clean.
            self._prev_aim_err = None
            self._aim_vel = (0.0, 0.0)
            try:
                pan, tilt, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0,
                                            track_active=False, t=now)
                self._ptz_last_cmd = (pan, tilt, zoom)
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s ptz auto idle step failed", self.camera_id,
                          exc_info=True)

    def _estimate_aim_velocity(
        self, err: tuple[float, float], now: float,
    ) -> tuple[float, float]:
        """EMA-smoothed d(error)/dt in normalized units/sec for PTZ feed-forward."""
        vx = vy = 0.0
        prev = self._prev_aim_err
        if prev is not None:
            dt = now - self._prev_aim_t
            if dt > 1e-3:
                a = 0.5  # EMA: balance responsiveness vs. jitter rejection
                vx = a * ((err[0] - prev[0]) / dt) + (1.0 - a) * self._aim_vel[0]
                vy = a * ((err[1] - prev[1]) / dt) + (1.0 - a) * self._aim_vel[1]
        self._aim_vel = (vx, vy)
        self._prev_aim_err = err
        self._prev_aim_t = now
        return self._aim_vel

    def _log_ptz_cmd(self, source: str, pan: float, tilt: float, zoom: float) -> None:
        """Rate-limited DEBUG log of a PTZ command (only when it changes).

        The auto loop steps every frame; logging each step would flood the
        console, so we only emit when the command vector meaningfully changes.
        """
        if not log.isEnabledFor(logging.DEBUG):
            return
        cmd = (round(pan, 3), round(tilt, 3), round(zoom, 3))
        if cmd == self._last_logged_ptz:
            return
        self._last_logged_ptz = cmd
        log.debug("camera_id=%s ptz %s backend=%s pan=%.3f tilt=%.3f zoom=%.3f",
                  self.camera_id, source, self.config.ptz.backend, pan, tilt, zoom)

    def _resolve_target_track(self, tracks: list[TrackInfo]) -> TrackInfo | None:
        """Pick the tracked subject to follow: the explicit target, else None."""
        if self._target_track_id is None:
            return None
        for t in tracks:
            if t.track_id == self._target_track_id:
                return t
        return None

    # ── appearance ReID recovery ─────────────────────────────────────────────────

    def _ensure_reid(self) -> Any | None:
        """Lazily build the OSNet ReID matcher when enabled (None if unavailable).

        Gated on ``tracking.reid_enabled``; degrades gracefully to ``None`` when
        boxmot/torch/weights are absent so motion-only tracking still works.
        """
        if not getattr(self.config.tracking, "reid_enabled", False):
            return None
        if self._reid is None and not self._reid_probed:
            self._reid_probed = True
            try:
                from autoptz.engine.pipeline.reid import BodyReID

                reid = BodyReID(
                    threshold_hi=float(self.config.tracking.reid_threshold_hi),
                    threshold_lo=float(self.config.tracking.reid_threshold_lo),
                )
                self._reid = reid if reid.available else None
            except Exception:  # noqa: BLE001 — missing dep/weights must not crash
                log.debug("camera_id=%s BodyReID build failed", self.camera_id,
                          exc_info=True)
                self._reid = None
        return self._reid

    def _reset_reid(self) -> None:
        """Drop the appearance template (target changed / cleared)."""
        if self._reid is not None:
            try:
                self._reid.reset()
            except Exception:  # noqa: BLE001
                pass

    def _maybe_reid_recover(
        self, tracks: list[TrackInfo], frame: NDArray[np.uint8] | None, now: float,
    ) -> None:
        """Keep the target template fresh while visible; re-bind it when lost.

        While the target track is present we refresh its EMA appearance template;
        once its track_id disappears (occlusion / ID switch) we embed the current
        candidate boxes and re-bind the target onto the best appearance match —
        so the lock follows the *person*, not whichever box happens to be nearest.
        Throttled to ``_REID_INTERVAL_S`` and a no-op unless ReID is enabled.
        """
        if (self._target_track_id is None or frame is None
                or not self._feature("tracking")):
            return
        reid = self._ensure_reid()
        if reid is None or not getattr(reid, "available", False):
            return
        if now - self._last_reid_t < _REID_INTERVAL_S:
            return
        self._last_reid_t = now

        present = {t.track_id: t for t in tracks}
        if self._target_track_id in present:
            # Target visible → refresh (or seed) its appearance template.
            tgt = present[self._target_track_id]
            emb = reid.embed([_xyxy(tgt.bbox)], frame)
            if emb.size:
                reid.update_target(emb[0])
            return

        # Target lost → recover onto the best-matching present track.
        if not getattr(reid, "locked", False) or not tracks:
            return
        cand = reid.embed([_xyxy(t.bbox) for t in tracks], frame)
        if cand.size == 0:
            return
        result = reid.recover(cand)
        if result.matched and 0 <= result.best_index < len(tracks):
            new_id = tracks[result.best_index].track_id
            if new_id != self._target_track_id:
                log.info("camera_id=%s ReID recovered target → track=%d score=%.2f",
                         self.camera_id, new_id, result.best_score)
                self._target_track_id = new_id
                self._reset_pose_aim()

    def _track_error(
        self, track: TrackInfo, frame: NDArray[np.uint8], now: float | None = None,
    ) -> tuple[tuple[float, float], float]:
        """Return (normalized center error, subject-height fraction).

        Error x>0 → target is right of center; error y>0 → target is **above**
        center (image y grows downward, so we negate to match the controller's
        tilt convention where positive=up).

        The aim **centre** is *pose-first in both arm modes*: when *now* is given,
        the pose feature is on, and the estimator confidently locates the torso,
        the aim is the pose anchor (shoulder/torso midpoint, biased by the
        ``framing`` region) — so the on-screen reticle sits on the body/skeleton
        and follows the pose, never the bounding box.  Raising or extending an arm
        grows the YOLO box but does **not** move the aim.  The point is
        EMA-smoothed to suppress keypoint jitter.

        The **arms toggle** (``aim_body_mode``) changes only the *zoom* source,
        not the aim centre:

        - ``torso`` (ignore arms) → zoom on the stable shoulder→hip span, so an
          extended arm does not make the camera zoom out.
        - ``full_silhouette`` (include arms) → zoom on the full detection-box
          height, so the shot widens to fit outstretched arms.

        Fallback (pose unavailable / not confident / *now* is ``None`` — pure-bbox
        callers and unit tests): aim at the box centre-x and the ``framing``
        fraction down the box, zoom on the box height.  The region still applies
        in both arm modes (no special-cased centre).
        """
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return (0.0, 0.0), 0.0

        bb = track.bbox
        bbox_height = (bb.y2 - bb.y1) / h
        # Arms toggle: "torso" ignores arms (steady pose-torso zoom span); any
        # other value ("full_silhouette") includes arms → zoom to the full box.
        ignore_arms = getattr(self.config.tracking, "aim_body_mode", "torso") == "torso"

        aim: tuple[float, float] | None = None
        subject_height = 0.0
        aim_source = ""

        if now is not None:
            pose_aim, torso_height = self._pose_aim(track, frame, now)
            if pose_aim is not None:
                aim = pose_aim
                aim_source = "pose"
                # Zoom source follows the arms toggle; fall back to the box height
                # when the torso span isn't available so we never zoom to "0".
                subject_height = (
                    torso_height if (ignore_arms and torso_height > 0.0)
                    else bbox_height
                )

        if aim is None:
            # ── Fallback: bbox-based math (region applies in both arm modes) ────
            ax = (bb.x1 + bb.x2) * 0.5
            # Vertical aim point: fraction down from the box top per the unified
            # ``framing`` region (default upper_body=0.38 → head+torso;
            # full_body=0.50 = geometric centre).
            frac = AIM_REGION_FRACTION.get(_resolve_framing(self.config.tracking), 0.5)
            ay = bb.y1 + (bb.y2 - bb.y1) * frac
            subject_height = bbox_height
            aim_source = "bbox"
        else:
            ax, ay = aim

        if track.is_target:
            track.aim_x = float(ax)
            track.aim_y = float(ay)
            track.aim_source = aim_source

        ex = (ax - w * 0.5) / (w * 0.5)            # [-1, 1] right-positive
        ey = -((ay - h * 0.5) / (h * 0.5))         # [-1, 1] up-positive
        ex = max(-1.0, min(1.0, ex))
        ey = max(-1.0, min(1.0, ey))
        subject_height = max(0.0, min(1.0, subject_height))
        return (ex, ey), subject_height

    def _annotate_target_aim(
        self, tracks: list[TrackInfo], frame: NDArray[np.uint8] | None, now: float,
    ) -> None:
        """Populate aim telemetry for the active live target, even without PTZ."""
        if frame is None:
            return
        target = self._resolve_target_track(tracks)
        if target is None or target.lost:
            return
        try:
            self._track_error(target, frame, now)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s target aim annotation failed", self.camera_id,
                      exc_info=True)

    # ── pose-stable aim (lazy, graceful) ────────────────────────────────────────

    def _pose_aim(
        self, track: TrackInfo, frame: NDArray[np.uint8], now: float,
    ) -> tuple[tuple[float, float] | None, float]:
        """Return ((aim_x, aim_y) | None, subject_height_fraction) from pose.

        Lazily builds the pose estimator the first time tracking needs an aim
        point.  Re-estimates keypoints at most every ``_POSE_INTERVAL_S`` for the
        active target, reusing the last keypoints in between.  Returns
        ``(None, 0.0)`` whenever pose is unavailable or not confident, so the
        caller keeps the bbox-based math.  Never raises.
        """
        if not self._feature("pose"):
            return None, 0.0
        pose = self._ensure_pose()
        if pose is None or not getattr(pose, "available", False):
            return None, 0.0

        try:
            from autoptz.engine.pipeline import framing
        except Exception:  # noqa: BLE001
            return None, 0.0

        h = frame.shape[0]
        bb = track.bbox
        bbox = (bb.x1, bb.y1, bb.x2, bb.y2)

        # Drop stale keypoints if the target track changed under us.
        if self._pose_kp_track_id != track.track_id:
            self._pose_keypoints = None
            self._reset_pose_aim()

        # Re-estimate only every _POSE_INTERVAL_S; reuse last keypoints between.
        if (
            self._pose_keypoints is None
            or now - self._last_pose_t >= _POSE_INTERVAL_S
        ):
            pose_t0 = time.perf_counter()
            kps = pose.estimate(frame, bbox)
            self._pose_ms = (time.perf_counter() - pose_t0) * 1000.0
            self._last_pose_t = now
            if kps is not None:
                self._pose_keypoints = kps
                self._pose_kp_track_id = track.track_id
            else:
                self._pose_keypoints = None
                self._pose_kp_track_id = None

        kps = self._pose_keypoints
        if not kps:
            return None, 0.0

        bias = _AIM_REGION_POSE_BIAS.get(
            _resolve_framing(self.config.tracking), "upper_body",
        )
        raw_aim = framing.torso_aim_point(kps, bias=bias)
        if raw_aim is None:
            return None, 0.0

        if self._aim_smoother is None:
            self._aim_smoother = framing.AimSmoother(alpha=_POSE_AIM_ALPHA)
        aim = self._aim_smoother.update(raw_aim)

        span = framing.subject_height_from_pose(kps)
        subject_height = (span / h) if (span is not None and h > 0) else 0.0
        return aim, subject_height

    def _maybe_estimate_pose_overlay(
        self, tracks: list[TrackInfo], frame: NDArray[np.uint8] | None, now: float,
    ) -> None:
        """Populate the tracked target's pose keypoints for the overlay + aim.

        Pose is tied strictly to the **tracked subject** (the locked/selected
        target), so the skeleton and the green aim circle always describe the
        same one person — a skeleton never appears on someone with no aim circle.
        Selecting a person (clicking their box) sets the target, which is enough
        to see their skeleton; no PTZ follow required.  Throttled + cached inside
        ``_pose_aim`` (``_POSE_INTERVAL_S``), so the later aim call this tick
        reuses the same keypoints (no double inference).
        """
        if frame is None or not self._feature("pose"):
            self._pose_keypoints = None
            self._pose_kp_track_id = None
            return
        target = self._resolve_target_track(tracks)
        if target is None or target.lost:
            self._reset_pose_aim()
            return
        self._pose_aim(target, frame, now)  # side effect: fills _pose_keypoints

    def _ensure_pose(self) -> Any | None:
        """Return the pose estimator (shared pool's first, else per-worker build).

        Built/resolved once and cached (including a ``None`` failure).  Prefers
        the injected :class:`InferencePool`'s shared pose estimator so all cameras
        share one ONNX session; falls back to a per-worker build when no pool was
        injected (tests/fakes).  Built only when auto tracking first needs an aim
        point, so idle cameras and live-preview-only runs never pay the cost.  The
        estimator may itself report ``available == False`` if no model is present.
        """
        if self._pose_probed:
            return self._pose
        self._pose_probed = True

        if self._pool is not None:
            try:
                self._pose = self._pool.pose()
            except Exception:  # noqa: BLE001 — pool must never break the worker
                log.debug("camera_id=%s pool pose() failed; bbox aim only.",
                          self.camera_id, exc_info=True)
                self._pose = None
            return self._pose

        if not detection_runtime_available():
            return None
        try:
            from autoptz.engine.pipeline.pose import PoseEstimator

            self._pose = PoseEstimator()
        except Exception:  # noqa: BLE001 — pose must never break the worker
            log.debug("camera_id=%s pose estimator init failed; bbox aim only.",
                      self.camera_id, exc_info=True)
            self._pose = None
        return self._pose

    def _reset_pose_aim(self) -> None:
        """Clear the pose aim smoother + cached keypoints (on target change)."""
        self._pose_keypoints = None
        self._pose_kp_track_id = None
        self._prev_aim_err = None
        self._aim_vel = (0.0, 0.0)
        if self._aim_smoother is not None:
            try:
                self._aim_smoother.reset()
            except Exception:  # noqa: BLE001
                pass

    # ── main loop ───────────────────────────────────────────────────────────────

    def _run(self) -> None:
        src = self.config.source
        log.info(
            "camera_id=%s worker starting — source=%s addr=%s target_fps=%.0f shm=%s",
            self.camera_id, src.type, _sanitize_address(src.address),
            float(getattr(src, "fps", 0.0) or 0.0), self.shm_name,
        )
        self._open_resources()

        # Start the inference thread AFTER the source is open.  It builds the
        # heavy models itself (so a slow first-time EP compile never blocks the
        # live preview) and processes the latest captured frame.
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name=f"caminfer-{self.camera_id[:8]}", daemon=True,
        )
        self._inference_thread.start()

        last_telemetry = 0.0
        fps_window_start = time.monotonic()
        fps_window_frames = 0
        self._next_drop_log_t = time.monotonic() + _DROP_LOG_INTERVAL_S
        last_health = HealthState.OK if self._source is not None else HealthState.ERROR
        last_error: str | None = (
            None if self._source is not None else "frame source unavailable"
        )

        # If the source could not be opened at all, still emit telemetry so the
        # UI shows an error/no-signal state instead of a silent hang.
        if self._source is None:
            self._emit_telemetry(tracks=[], health=last_health, last_error=last_error)

        while not self._stop_event.is_set():
            tick_t0 = time.perf_counter()
            frame: NDArray[np.uint8] | None = None
            if self._source is not None:
                try:
                    frame = self._source.read()
                    if frame is None:
                        # A transient read() miss (decode failure / dropped
                        # frame) — count it for Camera Info.
                        self._dropped_frames += 1
                except Exception as exc:  # noqa: BLE001
                    if last_health is not HealthState.RECONNECTING:
                        log.warning("camera_id=%s frame read failed (%s); "
                                    "reconnecting", self.camera_id, exc)
                    last_health = HealthState.RECONNECTING
                    last_error = str(exc)
                    self._dropped_frames += 1
                    frame = None
            ingest_ms = (time.perf_counter() - tick_t0) * 1000.0

            now = time.monotonic()

            if frame is not None:
                if last_health is HealthState.RECONNECTING:
                    log.info("camera_id=%s frame source recovered", self.camera_id)
                self._record_frame_dims(frame)
                self._push_frame(frame)
                fps_window_frames += 1
                last_health = HealthState.OK
                last_error = None

                # Hand the newest frame to the inference thread (latest wins) so
                # detection/face never gate the capture+preview rate.
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                self._frame_ready.set()

                self._ingest_ms = ingest_ms
                # Latency = capture read + the latest inference detect+track cost.
                self._latency_ms = ingest_ms + self._detect_ms

                elapsed = now - fps_window_start
                if elapsed >= 1.0:
                    self._fps = fps_window_frames / elapsed
                    fps_window_start = now
                    fps_window_frames = 0

            # Periodic frame-drop summary (INFO) when drops continue to accrue.
            self._maybe_log_drops(now)

            # Telemetry pacing (~telemetry_hz) — report the latest inference output.
            if now - last_telemetry >= self._telemetry_period:
                self._emit_telemetry(tracks=list(self._last_tracks),
                                     health=last_health, last_error=last_error)
                last_telemetry = now

            # Small idle sleep when there is no frame to avoid a busy-spin.
            if frame is None:
                self._stop_event.wait(timeout=0.01)

        # Shutdown: wake + join the inference thread, then release resources.
        self._frame_ready.set()
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5.0)
            self._inference_thread = None
        self._close_resources()
        log.info("camera_id=%s worker stopped (frames dropped total=%d)",
                 self.camera_id, self._dropped_frames)
        # Final STOPPED telemetry so the UI reflects the camera going down.
        self._emit_telemetry(tracks=[], health=HealthState.STOPPED, last_error=None)

    def _inference_loop(self) -> None:
        """Detect → track → face → pose → PTZ on the latest captured frame.

        Runs on its own thread so the capture/preview loop is never blocked by
        model building or per-frame inference.  Builds the heavy models here (off
        the capture critical path); then consumes the freshest frame each pass,
        dropping any intermediate frames the capture thread produced meanwhile.
        """
        self._build_inference_stacks()
        last_id = 0
        while not self._stop_event.is_set():
            # Wake on a new frame, but fall through on the timeout too so commands
            # (nudge / set-target / enable-tracking) are still drained when no
            # frames are arriving — they must never depend on frame delivery.
            self._frame_ready.wait(timeout=0.05)
            self._frame_ready.clear()
            if self._stop_event.is_set():
                break
            self._drain_commands()
            with self._frame_lock:
                frame = self._latest_frame
                fid = self._latest_frame_id
            if frame is None or fid == last_id:
                continue
            last_id = fid
            now = time.monotonic()

            detect_t0 = time.perf_counter()
            tracks = self._maybe_track(frame)
            self._detect_ms = (time.perf_counter() - detect_t0) * 1000.0

            # Appearance ReID: refresh the target template while it's visible and
            # re-bind it onto the right track after an occlusion (throttled; no-op
            # unless tracking.reid_enabled and boxmot is available).
            self._maybe_reid_recover(tracks, frame, now)

            # Estimate pose for the selected person so the pose overlay shows the
            # moment you click someone — independent of whether PTZ auto-follow is
            # actively driving (which is the only place pose ran before).
            self._maybe_estimate_pose_overlay(tracks, frame, now)
            self._annotate_target_aim(tracks, frame, now)

            # Face recognition + auto-harvest (throttled internally; record the
            # cost only when a run actually executed so the badge shows real cost).
            face_t0 = time.perf_counter()
            self._maybe_identify(frame, tracks, now)
            face_dt = (time.perf_counter() - face_t0) * 1000.0
            if face_dt > 0.5:
                self._face_ms = face_dt

            # Auto PTZ control (suspended during a manual-override window).  Lock
            # the backend so a concurrent telemetry position read can't interleave.
            with self._ptz_lock:
                self._drive_ptz_auto(tracks, frame, now)

            self._last_tracks = tracks
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "camera_id=%s timings detect+track=%.1fms face=%.1fms tracks=%d "
                    "fps=%.1f", self.camera_id, self._detect_ms, self._face_ms,
                    len(tracks), self._fps,
                )

    # ── resource management ─────────────────────────────────────────────────────

    def _open_resources(self) -> None:
        # Frame source
        if self._injected_source is not None:
            self._source = self._injected_source
        else:
            try:
                self._source = build_frame_source(self.camera_id, self.config)
            except Exception:  # noqa: BLE001
                log.warning("camera_id=%s could not build frame source (cv2 missing?)",
                            self.camera_id, exc_info=True)
                self._source = None

        if self._source is not None:
            try:
                if not self._source.open():
                    log.warning("camera_id=%s frame source failed to open",
                                self.camera_id)
                    self._source = None
            except Exception:  # noqa: BLE001
                log.warning("camera_id=%s frame source open raised", self.camera_id,
                            exc_info=True)
                self._source = None

        # Shared-memory writer is created eagerly in start() (before the thread)
        # so the provider can attach immediately.  As a safety net, create it
        # here too if start() couldn't (e.g. a subclass/test that calls _run
        # directly), so the thread always has a writer when one is possible.
        if self._shm is None:
            self._create_shm_writer_eager()

        # PTZ stack (graceful: None → manual no-op + no auto control).  Light to
        # build, so it stays here; the heavy detect/face models are built on the
        # inference thread (see ``_build_inference_stacks``).
        self._build_ptz_stack()

    def _build_inference_stacks(self) -> None:
        """Build the detect + face stacks ON the inference thread.

        Kept off ``_open_resources`` / the capture loop so a slow first-time EP
        compile never stalls the live preview — the capture thread keeps reading
        frames, pushing the preview, and emitting telemetry while these models
        warm up here.  In production the shared inference pool returns models
        that were built once at supervisor start, so this is near-instant; the
        per-worker build (insightface compile) is only the no-pool fallback.
        """
        # Detection stack (graceful: None → live-preview-only).  Prefer the
        # shared inference pool's detector (one ONNX session for all cameras) +
        # a fresh PER-WORKER tracker; fall back to the per-worker build (which
        # builds its own detector) when no pool was injected (tests/fakes).
        detect = self._build_detect_stack_pooled()
        if detect is None:
            detect = _build_detect_stack(self.config)
        if detect is not None:
            self._ep = detect.ep
        self._detect = detect

        # Face / identity stack (graceful: None → no identity features).  Prefer
        # the pool's shared recogniser; fall back to a per-worker build.
        if self._injected_face_stack is not None:
            self._face = self._injected_face_stack
        else:
            face = self._build_face_stack_pooled()
            if face is None and detection_runtime_available():
                face = _build_face_stack(
                    self.config, self._injected_identity_service,
                )
            self._face = face

    def _build_detect_stack_pooled(self) -> _DetectStack | None:
        """Build a detect stack from the pool's SHARED detector + a fresh tracker.

        Returns ``None`` when no pool was injected or the pool has no detector, so
        the caller falls back to the per-worker build.  The boxmot tracker is
        always created **per-worker** here — it holds per-camera state and must
        never be shared across cameras.  Never raises.
        """
        pool = self._pool
        if pool is None:
            return None
        try:
            detector = pool.detector()
        except Exception:  # noqa: BLE001 — pool must never break the worker
            log.debug("camera_id=%s pool detector() failed; per-worker fallback.",
                      self.camera_id, exc_info=True)
            return None
        if detector is None:
            return None
        try:
            from autoptz.engine.pipeline.track import Tracker

            tracker = Tracker(
                tracker_type=self.config.tracking.tracker,
                coast_window=self.config.tracking.coast_window_ms / 1000.0,
            )
            ep = getattr(detector, "ep", "") or getattr(pool, "detector_ep", "")
            _log_detector_ready_once(
                "<shared pool>",
                getattr(detector, "ep", "") or "?",
            )
            return _DetectStack(detector=detector, tracker=tracker, ep=ep)
        except Exception:  # noqa: BLE001
            log.warning("camera_id=%s per-worker tracker init failed; "
                        "live-preview-only.", self.camera_id, exc_info=True)
            return None

    def _build_face_stack_pooled(self) -> _FaceStack | None:
        """Build a face stack from the pool's SHARED recogniser + shared gallery.

        Returns ``None`` when no pool was injected (caller then uses the
        per-worker build).  The recogniser is the pool's lock-wrapped shared
        instance; the gallery is the supervisor-injected shared
        :class:`IdentityService` (or a fresh one as a last resort).  Never raises.
        """
        pool = self._pool
        if pool is None:
            return None
        try:
            recognizer = pool.face()
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s pool face() failed; per-worker fallback.",
                      self.camera_id, exc_info=True)
            return None
        if recognizer is None:
            return None
        try:
            from autoptz.engine.identity.service import IdentityService

            service = self._injected_identity_service or IdentityService()
            return _FaceStack(recognizer=recognizer, service=service)
        except Exception:  # noqa: BLE001 — gallery build must never break the worker
            log.warning("camera_id=%s identity gallery init failed; identity off.",
                        self.camera_id, exc_info=True)
            return None

    def _build_ptz_stack(self) -> None:
        """Build a PTZController around a backend from config, if none injected.

        Never raises — a missing backend leaves PTZ disabled (manual nudges
        no-op, auto control is skipped).  The controller's background thread is
        **not** started; the worker drives it synchronously via ``step()`` each
        tick so tracking data and PTZ commands stay in lock-step on one thread.
        """
        if self._ptz is not None or self._ptz_backend is not None:
            return  # already wired (injected by a test/caller)

        try:
            from autoptz.engine.ptz.controller import PTZController
            from autoptz.engine.ptz.factory import build_backend

            backend = build_backend(self.config.ptz)
            if backend is None:
                return
            self._ptz_backend = backend
            self._ptz = PTZController(
                backend, self.config.ptz,
                coast_window_ms=int(self.config.tracking.coast_window_ms),
            )
            self._ptz_owned = True
            log.info("camera_id=%s PTZ control active (%s)", self.camera_id,
                     self.config.ptz.backend)
        except Exception:  # noqa: BLE001 — PTZ must never break the worker
            log.warning("camera_id=%s PTZ stack init failed; PTZ disabled.",
                        self.camera_id, exc_info=True)
            self._ptz = None
            self._ptz_backend = None

    def _create_shm_writer_eager(self) -> None:
        """Set ``self._shm`` from the injected writer or by creating one now.

        Idempotent and safe to call from the main thread (in ``start()``) or the
        worker thread (the ``_open_resources`` safety net).  Never raises.
        """
        if self._shm is not None:
            return
        if self._injected_shm is not None:
            self._shm = self._injected_shm
            return
        self._shm = self._create_shm_writer()

    def _create_shm_writer(self) -> ShmWriter | None:
        """Create the preview ShmWriter, reclaiming a stale leaked segment once.

        A previous process that crashed can leave the named segment behind;
        ``ShmWriter`` opens with ``create=True`` and would then raise
        ``FileExistsError``.  We unlink the orphan and retry so the camera still
        comes up with a live preview.
        """
        from autoptz.engine.runtime.shm import ShmWriter

        try:
            return ShmWriter(self.shm_name, _PREVIEW_H, _PREVIEW_W)
        except FileExistsError:
            log.warning("camera_id=%s reclaiming stale shm segment %s",
                        self.camera_id, self.shm_name)
            self._unlink_stale_shm()
            try:
                return ShmWriter(self.shm_name, _PREVIEW_H, _PREVIEW_W)
            except Exception:  # noqa: BLE001
                log.warning("camera_id=%s could not create ShmWriter %s after reclaim",
                            self.camera_id, self.shm_name, exc_info=True)
                return None
        except Exception:  # noqa: BLE001
            log.warning("camera_id=%s could not create ShmWriter %s",
                        self.camera_id, self.shm_name, exc_info=True)
            return None

    def _unlink_stale_shm(self) -> None:
        from multiprocessing.shared_memory import SharedMemory

        for name in (self.shm_name, f"{self.shm_name}__idx"):
            try:
                stale = SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s could not unlink stale shm %s",
                          self.camera_id, name, exc_info=True)

    def _close_resources(self) -> None:
        if self._source is not None:
            try:
                self._source.close()
            except Exception:  # noqa: BLE001
                pass
            self._source = None
        if self._shm is not None and self._injected_shm is None:
            # Only close shm we own; injected shm is owned by the caller/test.
            try:
                self._shm.close()
            except Exception:  # noqa: BLE001
                pass
        self._shm = None
        self._close_ptz()

    def _close_ptz(self) -> None:
        """Always halt PTZ motion on shutdown; release a controller we own.

        Order matters: stop the controller first (it stops the backend), then
        stop the raw backend as a belt-and-suspenders guarantee that motion
        halts even if no controller is present (bare-backend injection / manual-
        only configs).  A controller we built ourselves is also ``close()``d so
        its backend's hardware resources are released.
        """
        ctrl = self._ptz
        if ctrl is not None:
            try:
                if self._ptz_owned and hasattr(ctrl, "close"):
                    ctrl.close()
                elif hasattr(ctrl, "stop"):
                    ctrl.stop()
            except Exception:  # noqa: BLE001
                pass
        backend = self._ptz_backend
        if backend is not None:
            try:
                if hasattr(backend, "stop"):
                    backend.stop()
            except Exception:  # noqa: BLE001
                pass

    # ── per-frame work ──────────────────────────────────────────────────────────

    def _record_frame_dims(self, frame: NDArray[np.uint8]) -> None:
        """Cache the source frame's (width, height) for Camera Info telemetry."""
        try:
            h, w = frame.shape[:2]
            if (w, h) != (self._frame_w, self._frame_h):
                log.info("camera_id=%s stream resolution %dx%d", self.camera_id, w, h)
            self._frame_w = int(w)
            self._frame_h = int(h)
        except Exception:  # noqa: BLE001
            pass

    def _maybe_log_drops(self, now: float) -> None:
        """Emit a periodic INFO summary of accrued frame drops (rate-limited)."""
        if now < self._next_drop_log_t:
            return
        self._next_drop_log_t = now + _DROP_LOG_INTERVAL_S
        delta = self._dropped_frames - self._last_logged_drops
        self._last_logged_drops = self._dropped_frames
        if delta > 0:
            log.info(
                "camera_id=%s dropped %d frame(s) in the last %.0fs "
                "(total=%d, fps=%.1f)",
                self.camera_id, delta, _DROP_LOG_INTERVAL_S,
                self._dropped_frames, self._fps,
            )

    def _push_frame(self, frame: NDArray[np.uint8]) -> None:
        if self._shm is None:
            return
        try:
            f = self._fit_frame(frame)
            self._shm.push(f)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s shm push failed", self.camera_id, exc_info=True)

    def _fit_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Resize / coerce a BGR frame to the ShmWriter's exact dimensions."""
        assert self._shm is not None
        h, w = frame.shape[:2]
        if h == self._shm.height and w == self._shm.width and frame.dtype == np.uint8:
            return np.ascontiguousarray(frame)
        try:
            import cv2
            resized = cv2.resize(frame, (self._shm.width, self._shm.height))
            return np.ascontiguousarray(resized.astype(np.uint8))
        except Exception:  # noqa: BLE001 - cv2 absent: pad/crop with numpy
            return self._numpy_fit(frame)

    def _numpy_fit(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        assert self._shm is not None
        out = np.zeros((self._shm.height, self._shm.width, 3), dtype=np.uint8)
        src = np.atleast_3d(frame).astype(np.uint8)
        if src.shape[2] == 1:
            src = np.repeat(src, 3, axis=2)
        h = min(src.shape[0], out.shape[0])
        w = min(src.shape[1], out.shape[1])
        out[:h, :w] = src[:h, :w, :3]
        return out

    def _maybe_track(self, frame: NDArray[np.uint8] | None) -> list[TrackInfo]:
        """Run detection + tracking whenever a detector is available.

        Decoupled from ``_tracking_enabled`` on purpose: detection, overlay
        boxes, and face auto-harvest must run whenever the engine is on and the
        detector loaded — so the operator SEES it working — regardless of
        whether a PTZ-follow target is set.  ``_tracking_enabled`` only gates
        *actively driving the PTZ toward a target* (see ``_drive_ptz_auto``).

        The global ``detection`` feature switch hard-gates this whole path: with
        it off the worker produces no detections, tracks, or overlay boxes (live
        preview still runs).  Detections smaller than
        ``tracking.min_detection_size_frac`` of the frame height are dropped here
        so the engine doesn't chase/save every far-away person.
        """
        if frame is None or self._detect is None:
            return []
        if not self._feature("detection"):
            return []
        try:
            _t0 = time.perf_counter()
            detections = self._detect.detector.detect(frame)
            detections = self._filter_small_detections(detections, frame)
            _t1 = time.perf_counter()
            tracks = self._detect.tracker.update(detections, frame, fps=max(1.0, self._fps))
            _t2 = time.perf_counter()
            self._detect_ms = (_t1 - _t0) * 1000.0
            self._track_ms = (_t2 - _t1) * 1000.0
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s detect/track failed", self.camera_id, exc_info=True)
            return []

        out: list[TrackInfo] = []
        for t in tracks:
            # LOST tracks remain inside the tracker for ReID/re-acquisition, but
            # they are stale visual boxes. Do not publish them to the UI or PTZ.
            if getattr(t, "state", None) == "lost":
                continue
            # _track_identity[track_id] = (identity_id, display_name, score)
            ident = self._track_identity.get(t.track_id)
            vel = getattr(t, "velocity", (0.0, 0.0)) or (0.0, 0.0)
            out.append(TrackInfo(
                track_id=t.track_id,
                bbox=BBox(x1=t.bbox.x1, y1=t.bbox.y1, x2=t.bbox.x2, y2=t.bbox.y2),
                identity=(ident[1] if ident else None),       # NAME, for display
                identity_id=(ident[0] if ident else None),    # id, for enroll/target
                confidence=(ident[2] if ident else t.conf),
                is_target=(self._target_track_id is not None
                           and t.track_id == self._target_track_id),
                lost=False,
                vx=float(vel[0]), vy=float(vel[1]),
            ))
        return out

    def _filter_small_detections(
        self, detections: list[Any], frame: NDArray[np.uint8],
    ) -> list[Any]:
        """Drop detections shorter than ``min_detection_size_frac`` * frame height.

        Computed from the REAL frame height so it only ever fires when frame
        dimensions are known; ``min_detection_size_frac == 0.0`` disables the
        gate.  This keeps the engine from chasing/saving distant specks while
        leaving direct-to-tracker test paths (which never pass through here)
        unaffected.
        """
        frac = float(getattr(self.config.tracking, "min_detection_size_frac", 0.0))
        if frac <= 0.0 or not detections:
            return detections
        try:
            h = int(frame.shape[0])
        except Exception:  # noqa: BLE001
            return detections
        if h <= 0:
            return detections
        min_px = frac * h
        kept = [d for d in detections if (d.bbox.y2 - d.bbox.y1) >= min_px]
        if log.isEnabledFor(logging.DEBUG) and len(kept) != len(detections):
            log.debug("camera_id=%s dropped %d small detection(s) (< %.0fpx)",
                      self.camera_id, len(detections) - len(kept), min_px)
        return kept

    # ── face recognition + identity ───────────────────────────────────────────

    def _clear_face_overlay(self) -> None:
        """Drop face overlay state immediately."""
        self._last_faces = []
        self._last_faces_t = 0.0
        self._last_face_track_ids = set()

    def _expire_face_overlay(
        self, now: float, tracks: list[TrackInfo] | None = None,
    ) -> None:
        """Clear face boxes once they are stale or their owning tracks vanished."""
        if not self._last_faces:
            return
        if now - self._last_faces_t > _FACE_TTL_S:
            self._clear_face_overlay()
            return
        if tracks is None or not self._last_face_track_ids:
            return
        live_ids = {t.track_id for t in tracks}
        if not self._last_face_track_ids.issubset(live_ids):
            self._clear_face_overlay()

    def _fresh_faces_for_telemetry(self, tracks: list[TrackInfo]) -> list[FaceBox]:
        """Return only fresh face boxes for the current live tracks."""
        self._expire_face_overlay(time.monotonic(), tracks)
        return list(self._last_faces)

    def _maybe_identify(
        self, frame: NDArray[np.uint8] | None, tracks: list[TrackInfo], now: float,
    ) -> None:
        """Run the face stack a few Hz: bind tracks → identities + auto-harvest.

        Steps each face tick:
          1. Detect faces + 512-d ArcFace embeddings on the frame.
          2. For each face, find the track whose bbox contains the face centre.
          3. Match the embedding against the enabled gallery; on a hit, cache
             ``track_id → (identity_id, score)`` so the *next* telemetry annotates
             the track, and resolve identity-targeting → set the single target
             track when the wanted identity is the matched one.
          4. For a good *unmatched* face on a confirmed track, auto-harvest a new
             in-memory **unlabeled** identity (with a base64-able PNG thumbnail)
             and push it to the UI (rate-limited by a cooldown).

        Safe no-op when the face stack / insightface is unavailable, or when the
        global ``face_recognition`` feature switch is off (no detect / match /
        harvest — manual click-to-track still works).
        """
        if not self._feature("face_recognition"):
            self._clear_face_overlay()
            return
        face = self._face
        if (
            frame is None
            or face is None
            or not getattr(face.recognizer, "available", False)
        ):
            self._clear_face_overlay()
            return
        if now - self._last_face_t < _FACE_INTERVAL_S:
            self._expire_face_overlay(now, tracks)
            # Still keep identity-targeting honest even without the face stack:
            # if an explicit identity target can't be resolved we leave the
            # current track lock untouched (manual box-tracking still works).
            return
        self._last_face_t = now

        try:
            observations = face.recognizer.detect(frame)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s face detect failed", self.camera_id, exc_info=True)
            self._clear_face_overlay()
            return
        if not observations:
            self._clear_face_overlay()
            return

        matched_identity_track: dict[str, int] = {}
        pending_faces = {
            tid: self._face_for_pending_enroll(frame, observations, tracks, tid, click)
            for tid, (_iid, _name, click) in self._pending_enroll.items()
        }
        for obs in observations:
            track = self._track_for_face(obs, tracks)
            # Click-to-assign: if this face sits on a track the operator named,
            # bind its embedding (+ a fresh crop) to that identity so the person
            # is recognised later, then keep showing the assigned name.
            if (
                track is not None
                and track.track_id in self._pending_enroll
                and pending_faces.get(track.track_id) is obs
            ):
                iid, name, _click = self._pending_enroll.pop(track.track_id)
                try:
                    face.service.add_embedding(
                        iid, obs.embedding, thumbnail=_face_crop_png(frame, obs.bbox),
                    )
                except Exception:  # noqa: BLE001
                    log.debug("camera_id=%s enroll_track add_embedding failed",
                              self.camera_id, exc_info=True)
                self._track_identity[track.track_id] = (iid, name, 1.0)
                matched_identity_track[iid] = track.track_id
                rec = face.service.get(iid) if hasattr(face.service, "get") else None
                if rec is not None:
                    self._push_identity(rec)
                log.info("camera_id=%s enrolled track=%d → id=%s name=%s",
                         self.camera_id, track.track_id, iid, name)
                continue
            match = None
            try:
                # include_disabled=True so an already-harvested (disabled)
                # "Person N" is recognised and re-bound instead of being
                # re-harvested as a duplicate — this is the dedup that turns
                # "16 faces for one person" into one.
                match = face.recognizer.match(
                    obs.embedding, face.service, include_disabled=True,
                )
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s face match failed", self.camera_id,
                          exc_info=True)
            if match is not None:
                if track is not None:
                    prev = self._track_identity.get(track.track_id)
                    self._track_identity[track.track_id] = (
                        match.identity_id, match.name, match.score,
                    )
                    matched_identity_track[match.identity_id] = track.track_id
                    # Log only on a new track→identity binding to avoid per-tick
                    # spam while the same person stays in frame.
                    if prev is None or prev[0] != match.identity_id:
                        log.info(
                            "camera_id=%s identity match track=%d id=%s name=%s "
                            "score=%.2f", self.camera_id, track.track_id,
                            match.identity_id, match.name, match.score,
                        )
                # Keep the matched identity's template fresh, and occasionally
                # capture a fresh crop so the person accrues a few varied
                # candidate profile photos (rate-limited to avoid flooding).
                crop: bytes | None = None
                if now - self._last_crop_t >= _HARVEST_COOLDOWN_S:
                    crop = _face_crop_png(frame, obs.bbox)
                    self._last_crop_t = now
                try:
                    face.service.add_embedding(
                        match.identity_id, obs.embedding, thumbnail=crop,
                    )
                except Exception:  # noqa: BLE001
                    log.debug("camera_id=%s add_embedding failed", self.camera_id,
                              exc_info=True)
            elif track is not None:
                self._maybe_harvest(face, frame, obs, track, now)

        # Resolve identity-targeting ("track when found"): if the operator chose
        # an identity and it is now bound to a live track, lock the single target
        # onto that track.
        if self._target_identity_id is not None:
            tid = matched_identity_track.get(self._target_identity_id)
            if tid is not None and tid != self._target_track_id:
                log.info("camera_id=%s identity-target id=%s acquired on track=%d",
                         self.camera_id, self._target_identity_id, tid)
                self._target_track_id = tid
                self._reset_pose_aim()

        # Prune stale cache entries for tracks no longer present.
        live_ids = {t.track_id for t in tracks}
        self._track_identity = {
            k: v for k, v in self._track_identity.items() if k in live_ids
        }
        self._pending_enroll = {
            k: v for k, v in self._pending_enroll.items() if k in live_ids
        }

        # Build the face overlay payload: each detected face with the name of the
        # track it sits on (when bound).  Published in telemetry; the UI draws it
        # only when the "Face boxes" overlay is enabled.
        faces_out: list[FaceBox] = []
        face_track_ids: set[int] = set()
        for obs in observations:
            tr = self._track_for_face(obs, tracks)
            if tr is None:
                continue
            face_track_ids.add(tr.track_id)
            ident = self._track_identity.get(tr.track_id) if tr is not None else None
            bx = obs.bbox
            faces_out.append(FaceBox(
                bbox=BBox(x1=float(bx[0]), y1=float(bx[1]),
                          x2=float(bx[2]), y2=float(bx[3])),
                identity=(ident[1] if ident else None),
                score=(ident[2] if ident else 0.0),
            ))
        self._last_faces = faces_out
        self._last_faces_t = now if faces_out else 0.0
        self._last_face_track_ids = face_track_ids

    @staticmethod
    def _track_for_face(
        obs: Any, tracks: list[TrackInfo],
    ) -> TrackInfo | None:
        """Return the track whose bbox contains the face centre (closest wins)."""
        best: TrackInfo | None = None
        best_d = float("inf")
        for t in tracks:
            bb = t.bbox
            if bb.x1 <= obs.cx <= bb.x2 and bb.y1 <= obs.cy <= bb.y2:
                tcx = (bb.x1 + bb.x2) * 0.5
                tcy = (bb.y1 + bb.y2) * 0.5
                d = (tcx - obs.cx) ** 2 + (tcy - obs.cy) ** 2
                if d < best_d:
                    best_d, best = d, t
        return best

    def _face_for_pending_enroll(
        self,
        frame: NDArray[np.uint8],
        observations: list[Any],
        tracks: list[TrackInfo],
        track_id: int,
        click_norm: tuple[float, float] | None,
    ) -> Any | None:
        """Pick the face to enroll for a pending clicked track.

        If the UI provided a click point, choose the face under that point or
        nearest to it within the requested track. Without a click point, use the
        largest face in that track as a stable fallback.
        """
        candidates = []
        for obs in observations:
            tr = self._track_for_face(obs, tracks)
            if tr is None or tr.track_id != track_id or getattr(tr, "lost", False):
                continue
            candidates.append(obs)
        if not candidates:
            return None
        if click_norm is None:
            return max(candidates, key=lambda o: (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]))

        h, w = frame.shape[:2]
        px = click_norm[0] * max(1, w)
        py = click_norm[1] * max(1, h)

        def score(obs: Any) -> tuple[int, float]:
            x1, y1, x2, y2 = obs.bbox
            inside = x1 <= px <= x2 and y1 <= py <= y2
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            d2 = (cx - px) ** 2 + (cy - py) ** 2
            return (0 if inside else 1, d2)

        return min(candidates, key=score)

    def _maybe_harvest(
        self,
        face: Any,
        frame: NDArray[np.uint8],
        obs: Any,
        track: TrackInfo,
        now: float,
    ) -> None:
        """Auto-harvest a *good* unmatched face into a new unlabeled identity.

        Strict gates (all must pass) so junk identities are rare and the user
        rarely has to merge: cooldown, a comfortable crop size, a confident SCRFD
        detection, a roughly frontal pose, a sharp (non-blurry) crop, and a low
        best similarity to the WHOLE gallery (so a known person is re-bound, not
        re-harvested as a duplicate).
        """
        if now - self._last_harvest_t < _HARVEST_COOLDOWN_S:
            return
        if not self._harvest_quality_ok(face, frame, obs):
            return
        self._last_harvest_t = now
        thumbnail = _face_crop_png(frame, obs.bbox)
        try:
            rec = face.service.add_unlabeled(obs.embedding, thumbnail)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s add_unlabeled failed", self.camera_id,
                      exc_info=True)
            return
        # Bind the freshly-harvested identity to its track so telemetry shows it.
        self._track_identity[track.track_id] = (rec.id, rec.name, obs.det_score)
        log.info("camera_id=%s harvested unlabeled identity id=%s from track=%d "
                 "(det_score=%.2f)", self.camera_id, rec.id, track.track_id,
                 obs.det_score)
        self._push_identity(rec)

    def _harvest_quality_ok(
        self, face: Any, frame: NDArray[np.uint8], obs: Any,
    ) -> bool:
        """Return True iff *obs* is a clean, frontal, novel face worth harvesting.

        Applies, in cheap-to-expensive order: crop-size floor → detection-score
        floor → frontal-pose (yaw) check → gallery-novelty check → blur check.
        Each rejection logs once at DEBUG and short-circuits.  Missing landmarks
        (yaw unknown) are treated leniently — det_score + size + novelty + sharp-
        ness already guard most junk — so a detector without ``kps`` still works.
        """
        x1, y1, x2, y2 = obs.bbox
        if max(x2 - x1, y2 - y1) < _MIN_HARVEST_FACE_PX:
            return False

        if float(getattr(obs, "det_score", 0.0)) < _MIN_HARVEST_DET_SCORE:
            log.debug("camera_id=%s harvest skip: low det_score=%.2f",
                      self.camera_id, float(getattr(obs, "det_score", 0.0)))
            return False

        yaw = obs.yaw_ratio() if hasattr(obs, "yaw_ratio") else None
        if yaw is not None and yaw > _MAX_HARVEST_YAW_RATIO:
            log.debug("camera_id=%s harvest skip: non-frontal yaw_ratio=%.2f",
                      self.camera_id, yaw)
            return False

        # Novelty: only create a NEW identity when the face is confidently NOT in
        # the gallery already (low best similarity over labeled + unlabeled).
        if self._best_gallery_similarity(face, obs) > _HARVEST_NOVELTY_MAX_SIM:
            log.debug("camera_id=%s harvest skip: already similar to gallery",
                      self.camera_id)
            return False

        sharp = self._face_sharpness(frame, obs.bbox)
        if sharp is not None and sharp < _MIN_HARVEST_SHARPNESS:
            log.debug("camera_id=%s harvest skip: blurry crop (var=%.1f)",
                      self.camera_id, sharp)
            return False

        return True

    def _best_gallery_similarity(self, face: Any, obs: Any) -> float:
        """Max cosine of *obs*'s embedding against every matchable identity.

        Uses the service's per-identity ``best_score`` over ``matchable_identities``
        (labeled + unlabeled) so a face already harvested as "Person N" is not
        re-harvested.  Returns 0.0 on any failure (treat as novel).
        """
        try:
            service = face.service
            best = 0.0
            for ident in service.matchable_identities():
                score = service.best_score(ident.id, obs.embedding)
                if score > best:
                    best = score
            return best
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s gallery-novelty check failed", self.camera_id,
                      exc_info=True)
            return 0.0

    def _face_sharpness(
        self, frame: NDArray[np.uint8], bbox: tuple[float, float, float, float],
    ) -> float | None:
        """Return the Laplacian variance of the face crop (blur measure), or None.

        Higher = sharper.  ``None`` when cv2 is unavailable or the crop is
        degenerate, so the caller treats sharpness as unknown (does not reject).
        """
        try:
            import cv2  # noqa: PLC0415

            h, w = frame.shape[:2]
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            if x2 - x1 < 4 or y2 - y1 < 4:
                return None
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:  # noqa: BLE001
            return None

    def _push_identity(self, record: Any) -> None:
        """Thread-safely surface a new/updated identity to the UI."""
        cb = self._on_identity
        if cb is None:
            return
        try:
            cb(record)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s identity callback raised", self.camera_id,
                      exc_info=True)

    # ── telemetry ───────────────────────────────────────────────────────────────

    def _pose_overlay(self) -> list[PoseKeypoint]:
        """The target's last pose keypoints (pixel-space) for the pose overlay.

        Empty unless pose ran for the current target; the UI draws the skeleton
        only when the "Pose" overlay is enabled.
        """
        kps = self._pose_keypoints
        if not kps:
            return []
        if self._target_track_id is not None and self._pose_kp_track_id != self._target_track_id:
            return []
        return [PoseKeypoint(x=float(k.x), y=float(k.y), conf=float(k.conf))
                for k in kps]

    def _emit_telemetry(
        self,
        *,
        tracks: list[TrackInfo],
        health: HealthState,
        last_error: str | None,
    ) -> None:
        msg = TelemetryMsg(
            camera_id=self.camera_id,
            seq=self._seq,
            fps=self._fps,
            ep=self._ep,
            width=self._frame_w,
            height=self._frame_h,
            dropped_frames=self._dropped_frames,
            latency_ms=self._latency_ms,
            ingest_ms=self._ingest_ms,
            detect_ms=self._detect_ms,
            track_ms=self._track_ms,
            face_ms=self._face_ms,
            pose_ms=self._pose_ms,
            streaming=self._frame_w > 0,
            source_fps_cap=self._source_fps_cap(),
            tracks=tracks,
            faces=self._fresh_faces_for_telemetry(tracks),
            pose=self._pose_overlay(),
            ptz=self._ptz_state(),
            health=HealthInfo(state=health, last_error=last_error),
        )
        self._seq += 1
        try:
            self._on_telemetry(msg)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s telemetry callback raised", self.camera_id,
                      exc_info=True)

    def _source_fps_cap(self) -> float:
        """Return the source's detected hardware fps ceiling (0.0 = unknown).

        Read from the running frame source's adapter status when available; the
        UI uses it to cap its fps slider at the camera's real maximum.
        """
        src = self._source
        if src is None:
            return 0.0
        fn = getattr(src, "source_fps_cap", None)
        if not callable(fn):
            return 0.0
        try:
            cap = fn()
        except Exception:  # noqa: BLE001
            return 0.0
        return float(cap) if cap else 0.0

    def _ptz_state(self) -> PTZState:
        """Build the real PTZState for telemetry: position, motion, and state.

        Position comes from the backend when it can query (VISCA-IP / ONVIF);
        otherwise it falls back to the last velocity command sent.  ``moving`` is
        derived from the last command magnitude; ``state`` reflects manual
        override, the controller's state machine, or idle.  Empty default when no
        PTZ is configured.
        """
        backend = self._ptz_backend
        if backend is None:
            return PTZState()

        pan, tilt, zoom = self._ptz_last_cmd

        # Prefer an absolute position query when the backend supports it.  Hold
        # the PTZ lock so this read (capture thread) doesn't interleave with the
        # inference thread driving motion on the same backend.
        position = None
        try:
            with self._ptz_lock:
                position = backend.get_position()
        except Exception:  # noqa: BLE001
            position = None
        if position is not None:
            pan, tilt, zoom = position.pan, position.tilt, position.zoom

        moving = (abs(pan) > 1e-3 or abs(tilt) > 1e-3 or abs(zoom) > 1e-3)

        if self._manual_override_active(time.monotonic()):
            state = "manual"
        elif self._ptz is not None and hasattr(self._ptz, "state"):
            state = getattr(self._ptz.state, "name", str(self._ptz.state)).lower()
        else:
            state = "moving" if moving else "idle"

        return PTZState(
            pan=float(pan),
            tilt=float(tilt),
            zoom=float(zoom),
            moving=bool(moving),
            backend=str(self.config.ptz.backend),
            state=state,
        )
