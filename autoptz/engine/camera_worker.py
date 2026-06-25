"""Per-camera pipeline worker: ingest → (detect → track) → telemetry + live preview.

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

Threading model is per-thread today (capture + inference); process-per-camera is
the future hardening step (see ``supervisor.py``).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from autoptz.config.models import AIM_REGION_FRACTION
from autoptz.engine.runtime.messages import (
    BBox,
    FaceBox,
    HealthInfo,
    HealthState,
    PoseKeypoint,
    PTZState,
    QualityStateInfo,
    RuntimeEventInfo,
    RuntimeServiceInfo,
    StageTimingInfo,
    SwitchStateInfo,
    TelemetryMsg,
    TrackInfo,
    TrackingStatusInfo,
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
# Cap the preview ShmWriter push rate. The UI tile is a monitoring view (overlays
# come from the ~10 Hz telemetry, not the preview frame), so it doesn't need the
# full capture fps — capping it skips the per-frame resize/copy cost on faster
# sources. Detection/tracking use the full-rate frame on a separate path.
_PREVIEW_PUSH_FPS = 20.0
_PREVIEW_PUSH_MIN_PERIOD_S = 1.0 / _PREVIEW_PUSH_FPS

# Center Stage crop tightness per "Framing" preset → (subject fill of crop,
# max crop as a fraction of the frame). A smaller ``max_frac`` forces a tighter
# zoom even on a close subject that already fills the sensor; the live "Framing"
# dropdown (tracking.framing) picks the preset, so the user dials the shot
# without a restart. ``upper_body`` is the default head-and-shoulders look.
_CENTERSTAGE_FRAMING: dict[str, tuple[float, float]] = {
    "face": (0.86, 0.50),  # tight head/face closeup (~2.0x on a close subject)
    "head_shoulders": (0.80, 0.62),  # head + shoulders (~1.6x)
    "upper_body": (0.70, 0.74),  # head + chest (~1.35x) — default
    "full_body": (0.58, 0.94),  # whole person, gentle crop
}

_DEFAULT_TELEMETRY_HZ = 10.0

# How long a manual PTZ nudge suspends auto control before auto resumes.
_MANUAL_OVERRIDE_WINDOW_S = 1.5

# Face stack run-rate: detect/embed faces a few times a second on the target
# region, not every frame.  Period in seconds.
_FACE_INTERVAL_S = 0.25
_FACE_TTL_S = 0.12
_STAGE_WINDOW = 60
_STAGE_FRESH_S = 2.0
# Inference-hang watchdog: if the inference thread hasn't completed a tick in
# this many seconds, the capture thread halts PTZ motion and surfaces a
# DEGRADED/stalled status.  Must exceed the worst normal per-frame inference
# time (typically <200 ms even on CPU) by a large margin.
# NOTE: the value 2.0 coincidentally matches ``_STAGE_FRESH_S`` but the two
# constants are UNRELATED concepts and should be tuned independently.
_INFER_STALL_S = 2.0
_EVENT_MAX = 12
_FPS_LOG_DELTA = 0.25

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

# Throttle interval for the per-iteration crash-guard WARNINGs in the capture and
# inference loops: a persistent fault would otherwise emit a stack trace every
# tick (~hundreds/s).  At most one WARNING per interval keeps the failure visible
# without flooding the log.
_TICK_WARN_INTERVAL_S = 5.0
_INFER_WARN_INTERVAL_S = 5.0

# Capture idle/reconnect backoff: a stalled, offline, or permission-denied source
# returns no frame, and a flat 10 ms retry then spins the capture thread at ~100 Hz
# (real, measurable idle CPU per camera).  Retry fast for a short window to ride out
# a transient decode miss on a healthy source, then ramp the sleep up to a cap so a
# truly dead source costs almost nothing — resetting instantly when a frame returns
# (so recovery still happens within ~one cap interval).
_RECONNECT_FAST_RETRIES = 20  # ~0.2 s of 10 ms retries before backing off
_RECONNECT_BACKOFF_MAX_S = 0.5  # sleep cap for a sustained no-frame source


def _async_appearance_enabled() -> bool:
    """Run face + ReID on their own thread (default on; AUTOPTZ_ASYNC_APPEARANCE=0 off)."""
    import os

    return os.environ.get("AUTOPTZ_ASYNC_APPEARANCE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _appearance_guarded(method: Callable) -> Callable:
    """Serialise a target/identity method under the worker's ``_appearance_lock``.

    Lets the async appearance thread and the hot inference thread share the
    target-lock/identity state machine safely without re-indenting these long
    method bodies.  The lock is re-entrant, so nested calls (e.g. ReID →
    ``_commit_target_track``) are fine.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._appearance_lock:
            return method(self, *args, **kwargs)

    return wrapper


# How often pose keypoints are (re)estimated for the active target.  Pose runs
# only on the single target crop, not every frame; between runs the worker reuses
# the last keypoints (the bbox still tracks the person each frame, so a slightly
# stale torso point is fine and far cheaper than per-frame pose inference).
_POSE_INTERVAL_S = 0.2
_POSE_TTL_S = 0.12

# How often (seconds) to run OSNet appearance ReID: refresh the target's template
# while it's visible, or attempt recovery while it's lost.  Throttled because the
# embedder is moderately heavy and per-frame recovery is unnecessary.
_REID_INTERVAL_S = 0.25

# Conservative target-lock gates. These intentionally prefer a short hold over a
# one-frame wrong-person switch in crowded scenes.
_TARGET_HOLD_S = 1.0
_TARGET_AMBIGUOUS_HOLD_S = 0.75
_TARGET_JUMP_MIN_PX = 80.0
_TARGET_JUMP_SCALE = 0.75
_TARGET_JUMP_IOU = 0.12
_TARGET_OVERLAP_IOU = 0.08
_TARGET_CLOSE_Y_OVERLAP = 0.45
_TARGET_CLOSE_GAP_FRAC = 0.18
_POSE_BBOX_MARGIN = 0.22
_POSE_KEYPOINT_MARGIN = 0.08
_POSE_JUMP_MIN_PX = 70.0
_POSE_JUMP_SCALE = 0.65
# If the body keypoints clear this confidence while a target is crowded or its
# detector box jitters, trust the pose anchor and keep following the same body.
_POSE_TARGET_LOCK_MIN_CONF = 0.45
_FACE_LEARN_MATCH_THRESHOLD = 0.58
_FACE_TARGET_MATCH_THRESHOLD = 0.52
_REID_RECOVERY_CONFIRM_STABLE = 3
_REID_RECOVERY_CONFIRM_RESPONSIVE = 1
_REID_RECOVERY_MARGIN = 0.08
_IDENTITY_TARGET_CONFIRM = 2
# Occlusion coast: the target box height (frame fraction) must drop below this
# fraction of its recent *healthy* height to count as a sudden collapse (the
# subject covered by something) rather than the subject simply walking away.
_OCCLUSION_COLLAPSE_FRAC = 0.55
_OCCLUSION_REF_ALPHA = 0.10  # how fast the healthy-height reference tracks a gradual change

# EMA weight for the pose-derived aim point: lower = smoother/laggier.  Light
# smoothing so noisy keypoint regression doesn't jitter the framing.
_POSE_AIM_ALPHA = 0.4

# Default feature switches — every subsystem on until the supervisor pushes a
# narrower set via ``set_features``.
_DEFAULT_FEATURES: dict[str, bool] = {
    "detection": True,
    "tracking": True,
    "face_recognition": True,
    "pose": True,
    # Global master switch for appearance ReID. A camera only runs ReID when this
    # is on AND its tracking_mode is "stable" (see ``_reid_active``).
    "reid": True,
}


@dataclass
class _TargetLockState:
    """Last trusted target evidence used to suppress crowded-scene switches."""

    trusted_track_id: int | None = None
    trusted_bbox: BBox | None = None
    trusted_identity: str | None = None
    trusted_identity_id: str | None = None
    trusted_confidence: float = 0.0
    trusted_aim: tuple[float, float] | None = None
    trusted_t: float = 0.0
    status: str = "idle"
    reason: str = ""
    ambiguous_until: float = 0.0


# Frame-source abstraction, fps pacing, and source construction live in
# worker/frame_source.py; re-exported here so existing imports keep working.
from autoptz.engine.worker.frame_source import (  # noqa: E402
    FrameSource,
    _AdapterFrameSource,  # noqa: F401  re-exported for back-compat
    _index_for_unique_id,  # noqa: F401
    _parse_usb_index,  # noqa: F401
    _resolve_framing,
    _resolve_usb_device,  # noqa: F401
    _sanitize_address,
    _strip_scheme,  # noqa: F401
    build_frame_source,
)

# Detector/face stack construction + ML capability probes live in
# worker/stacks.py; re-exported here so existing imports keep working.
from autoptz.engine.worker.stacks import (  # noqa: E402
    _build_detect_stack,  # noqa: F401
    _build_face_stack,  # noqa: F401
    _DetectStack,  # noqa: F401
    _face_crop_png,  # noqa: F401
    _FaceStack,  # noqa: F401
    _log_detector_ready_once,  # noqa: F401
    _log_no_detector_once,  # noqa: F401
    _resolve_model_path,  # noqa: F401
    _xyxy,  # noqa: F401
    detection_runtime_available,  # noqa: F401
    ml_stack_available,  # noqa: F401
)

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
        self._last_preview_push_t = 0.0
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
        self._last_faces_frame_id = 0
        self._last_faces_emitted_frame_id = -1
        # Identity the operator asked us to follow ("track when found"); single
        # target per camera — supersedes an explicit track id when its identity
        # is detected on a live track.
        self._target_identity_id: str | None = config.target.identity_id
        # Slowly-tracked "healthy" target box height (frame fraction); lets the
        # drive loop tell a sudden occlusion collapse from a gradual walk-away.
        self._target_h_ref: float | None = None

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
        # Set True by the capture-thread watchdog when inference appears hung;
        # cleared automatically when inference resumes.
        self._watchdog_stalled: bool = False
        self._target_track_id: int | None = None
        self._target_lock = _TargetLockState()
        self._identity_recover_candidate: tuple[str, int, int] | None = None

        # ── pose-stable aim (lazy) ───────────────────────────────────────────────
        # Optional keypoint estimator for the active target so an extended arm
        # (which grows the person bbox) doesn't yank the framing.  Built lazily
        # the first time auto tracking actually needs an aim point, mirroring the
        # detector/face lazy build, so idle cameras pay nothing.  ``_pose_probed``
        # guards against re-attempting the build every tick once it has failed.
        self._pose: Any | None = None
        self._pose_probed = False
        self._pose_keypoints: list[Any] | None = None  # last keypoints (reused)
        self._pose_kp_track_id: int | None = None  # track they belong to
        self._last_pose_t = 0.0
        self._last_pose_overlay_t = 0.0
        self._last_pose_overlay_frame_id = 0
        self._last_pose_emitted_frame_id = -1
        self._aim_smoother: Any | None = None  # framing.AimSmoother (lazy)
        self._last_aim_framing = ""  # Frame-on at the last aim tick
        # Per-frame memo of the fused aim so _track_error (called by both the PTZ
        # loop and the aim-dot annotation each tick) advances the smoother once.
        self._aim_cache: tuple[float, int, tuple[float, float], float, float, float, str] | None = (
            None
        )

        # ── appearance ReID recovery (lazy, gated on _reid_active) ────────────────
        # OSNet body-appearance matcher used to re-bind the target onto the right
        # track after an occlusion (built lazily; ``_reid_probed`` stops us from
        # re-attempting a failed/missing-boxmot build every tick).  The template is
        # seeded/refreshed while the target is visible and queried when it's lost.
        self._reid: Any | None = None
        self._reid_probed = False
        self._last_reid_t = 0.0
        self._reid_recover_candidate: tuple[int, int] | None = None
        # Phase-stagger: seeded once on the first inference-loop tick so face,
        # ReID, and pose don't all fire on the same appearance-thread iteration.
        self._phase_seeded = False

        # Aim-error velocity estimate (normalized error units / second), fed to the
        # PTZ controller so its feed-forward + motion prediction actually engage —
        # previously the worker passed a hardcoded (0, 0), so the camera only ever
        # chased the subject's *past* position (the "laggy following" complaint).
        self._prev_aim_err: tuple[float, float] | None = None
        self._prev_aim_t: float = 0.0
        self._aim_vel: tuple[float, float] = (0.0, 0.0)

        # Ego-motion: the camera's own pan/tilt shifts every pixel, so a still
        # subject looks like it's moving.  This estimator measures that global
        # image motion (background optical flow + a learned command gain) in the
        # same normalized error space as the aim velocity, so _estimate_aim_velocity
        # can subtract it and feed the controller the subject's *world* motion —
        # the fix for hunting when the subject and camera move together.
        from autoptz.engine.pipeline.egomotion import EgoMotionEstimator

        self._ego_estimator = EgoMotionEstimator(
            gain_max=float(getattr(self.config.ptz, "ego_comp_gain_max", 8.0))
        )
        # This tick's ego velocity (error-space units/sec) and last full estimate.
        self._ego_vel: tuple[float, float] = (0.0, 0.0)
        self._ego_source: str = "none"
        # True only on a tick where ego-motion was *freshly measured*; the aim
        # feed-forward only subtracts ego when fresh, never a stale/decimated
        # estimate (subtracting a decayed value made the camera ping-pong).
        self._ego_fresh: bool = False

        # ── fused target associator (flag-gated, default OFF) ────────────────────
        # Cheap stateless instance — always constructed; only consulted when
        # config.tracking.use_target_associator is True.
        from autoptz.engine.pipeline.associator import TargetAssociator

        self._associator = TargetAssociator()

        # runtime state
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._cmd_lock = threading.Lock()
        self._cmd_queue: deque[tuple[str, Any]] = deque()
        # Manual PTZ commands (nudge/home/menu) ride a SEPARATE queue drained on
        # the capture thread, not the inference thread — a detect+track pass can
        # be tens of ms, so routing the joystick/D-pad through it added that lag
        # to every move.  The capture loop ticks at the full source rate, so this
        # makes manual PTZ feel immediate (the NDI-Studio-like responsiveness).
        self._ptz_cmd_lock = threading.Lock()
        self._ptz_cmd_queue: deque[tuple[str, Any]] = deque()
        # Wakes the capture loop's no-frame backoff sleep early so a manual PTZ
        # command (or shutdown) is applied immediately instead of waiting out the
        # backoff — keeps the joystick responsive even when the video feed is
        # stalled/reconnecting while the PTZ control channel is still live.
        self._capture_wake = threading.Event()

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
        self._inference_start = threading.Event()
        self._inference_start.set()
        self._current_inference_frame_id = 0
        # Most recent inference output, read by the capture thread for telemetry.
        self._last_tracks: list[TrackInfo] = []
        self._last_tracks_frame_id = 0
        # Serializes PTZ-backend access between the inference thread (which drives
        # motion) and the capture thread (which reads position for telemetry).
        self._ptz_lock = threading.Lock()

        # ── async appearance (face + ReID) handoff ───────────────────────────────
        # The appearance stages (face recognition + ReID recovery) are the heavy,
        # latency-tolerant, throttled passes.  Running them on the hot inference
        # thread made the detect→track→PTZ control loop wait behind a 12–20 ms face
        # pass.  When ``_async_appearance`` is on (default), they run on their own
        # thread while the hot loop does detect+track+PTZ — the two heavy costs
        # overlap.  All target/identity state they share with the hot loop
        # (_target_lock, _target_track_id, _track_identity) is serialised by the
        # single re-entrant ``_appearance_lock`` (held only inside the four
        # target/identity methods), so the brief critical sections serialise while
        # the expensive inference overlaps.  Disable with AUTOPTZ_ASYNC_APPEARANCE=0.
        self._appearance_lock = threading.RLock()
        self._async_appearance = _async_appearance_enabled()
        self._appearance_thread: threading.Thread | None = None
        self._appearance_ready = threading.Event()
        # Latest (frame, tracks, now, frame_id) published for the appearance thread.
        self._appearance_input: tuple[NDArray[np.uint8], list[TrackInfo], float, int] | None = None
        self._appearance_ms = 0.0

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
        # Feature snapshot from the previous inference tick.  Comparing it against
        # the live flags lets ``_apply_model_lifecycle`` act only on transitions
        # (build on enable / free on disable) instead of every frame.  Seeded in
        # the inference loop right after the initial stacks are built.
        self._prev_model_features: dict[str, bool] = dict(_DEFAULT_FEATURES)

        # owned resources (created in thread)
        self._source: FrameSource | None = None
        self._shm: ShmWriter | None = None
        self._vcam: Any | None = None  # VirtualCamSink (lazily created when vcam_out enabled)
        self._digital_framer: Any | None = None  # Center Stage auto-framer (lazy)
        self._cs_diag_t: float = 0.0  # throttle for the Center Stage diagnostic log
        self._detect: _DetectStack | None = None
        self._pooled_detector = False
        # True when the detector is the unified YOLO11-pose model (boxes+keypoints
        # in one pass); the worker then sources pose-aim keypoints from the
        # detections instead of running a separate pose forward pass.
        self._unified_pose_active = False
        self._detect_frame_index = 0
        # Tracks whether the detector actually ran this inference tick (vs. a
        # skip/coast frame). Used by _pose_allowed_this_tick to spread the two
        # heavy stages across separate frames when stage_spread is enabled.
        self._detected_this_tick: bool = False
        # Last detections, re-fed to the tracker on detector-skip frames (when
        # detect_interval > 1) so tracks don't age out into "no boxes".
        self._last_detections: list[Any] = []

        # System-wide CPU pressure (0–100 %) pushed by the supervisor ~1 Hz.
        # Consulted by _effective_detect_interval to throttle when the whole
        # machine is hot, even if this camera's local cost looks fine.
        self._system_cpu_pressure: float = 0.0

        self._seq = 0
        self._fps = 0.0
        self._ep = ""

        # Camera-info telemetry: last-seen source frame size and a cumulative
        # count of read() misses / decode failures.
        self._frame_w = 0
        self._frame_h = 0
        self._dropped_frames = 0
        # Inference-stage drops: frames the capture thread posted that the
        # inference thread never consumed (overwritten in the latest-frame slot
        # because inference couldn't keep up).  Previously invisible — counted here
        # so sustained overload shows up in the logs instead of silently dropping.
        self._frames_captured = 0
        self._frames_inferred = 0
        self._last_infer_t: float = 0.0  # monotonic timestamp of last completed inference tick
        self._last_logged_inf_captured = 0
        self._last_logged_inf_inferred = 0

        # Per-frame processing latency (ingest read + detect + track wall time)
        # in milliseconds, published in telemetry for the live stats overlay.
        self._latency_ms = 0.0
        self._inference_ms = 0.0
        self._ingest_ms = 0.0
        self._detect_ms = 0.0
        self._track_ms = 0.0
        # Face / pose stage cost (ms) from the most recent run that actually
        # executed (both are throttled, so they hold between runs).
        self._face_ms = 0.0
        self._pose_ms = 0.0
        self._stage_samples: dict[str, deque[float]] = {
            key: deque(maxlen=_STAGE_WINDOW)
            for key in ("ingest", "detect", "track", "face", "pose", "reid")
        }
        self._stage_last_t: dict[str, float] = {}
        # Guards _stage_samples now that "face" is appended from the appearance
        # thread while the hot thread sums it in _stage_avg (avoids a "deque
        # mutated during iteration" race).
        self._stage_lock = threading.Lock()
        self._runtime_events: deque[RuntimeEventInfo] = deque(maxlen=_EVENT_MAX)
        self._tracker_switch: SwitchStateInfo | None = None
        self._last_applied_fps: float | None = None
        self._quality_active = "auto"
        self._quality_reason = "Auto quality monitoring."
        self._quality_interval = max(1, int(getattr(config.tracking, "detect_interval", 1) or 1))

        # Periodic diagnostics bookkeeping: when the next dropped-frame summary
        # is due (monotonic seconds) and the count last reported, so we only log
        # when drops actually accrued.
        self._next_drop_log_t = 0.0
        self._last_logged_drops = 0
        # PTZ command-rate-limit so DEBUG nudge/auto lines don't spam at
        # per-frame rate — only log when the command meaningfully changes.
        self._last_logged_ptz: tuple[float, float, float] | None = None

        # Crash-safety throttles: monotonic timestamp of the last per-iteration
        # WARNING in the capture / inference loops, so a persistent fault stays
        # visible without spamming a stack trace every tick.  _infer_last_error
        # surfaces the most recent inference-stage failure for the Camera Info
        # panel.  It is cleared on the next successful detect/track tick (see
        # _maybe_track) so a transient failure does not pin a stale error onto
        # later healthy telemetry; the inference fatal handler sets a terminal
        # value so a dead inference thread stays visible while capture streams.
        self._last_tick_warn_t = 0.0
        self._last_infer_warn_t = 0.0
        self._infer_last_error: str | None = None
        # Throttle for the shm-push failure WARNING (per-camera, monotonic).
        self._last_shm_push_warn_t = 0.0

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
            target=self._run,
            name=f"camworker-{self.camera_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal stop and block until the thread exits; releases all resources."""
        self._stop_event.set()
        self._capture_wake.set()  # break a backed-off no-frame sleep promptly
        self._inference_start.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None
        if self._vcam is not None:
            self._vcam.close()
            self._vcam = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_alive(self) -> bool:
        """Return True iff the worker has been started and its capture thread is alive.

        False before ``start()``, after ``stop()``, or if the thread died unexpectedly.
        Uniform interface with ``ProcessWorkerHandle.is_alive()`` so the supervisor
        can treat both worker types identically for health monitoring.

        Delegates to :attr:`is_running` so liveness logic lives in exactly one place.
        """
        return self.is_running

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
        self,
        track_id: int,
        identity_id: str,
        name: str,
        click_x: float | None = None,
        click_y: float | None = None,
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
        self,
        callback: Callable[[IdentityRecord], None] | None,
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

    def refresh_detector_from_pool(self) -> None:
        """Point this worker at the pool's current detector after a hot-swap."""
        with self._cmd_lock:
            self._cmd_queue.append(("refresh_detector", None))

    def reload_inference_models(self) -> None:
        """Drop + rebuild detector/pose after the on-disk model cache changed.

        Unlike ``refresh_detector_from_pool`` (a hot-swap that keeps the old
        model if the new one isn't ready), this force-drops the worker's model
        references so a *removed* model truly stops drawing boxes, then rebuilds
        from the (now-refreshed) shared pool — yielding the new model, or nothing
        when the files were deleted.
        """
        with self._cmd_lock:
            self._cmd_queue.append(("reload_models", None))

    def release_inference_models(self, *, wait: float = 0.0) -> None:
        """Drop the worker's detector/pose refs so their ORT sessions can be freed.

        Unlike :meth:`reload_inference_models` this does *not* rebuild — it only
        releases, so the OS file handles are gone before the on-disk model cache
        is mutated (delete/replace fails on Windows while a handle is open).  Pass
        ``wait > 0`` to block until the inference thread confirms the release (or
        the timeout elapses); the caller then GCs and mutates the files.
        """
        done = threading.Event() if wait > 0 else None
        with self._cmd_lock:
            self._cmd_queue.append(("release_models", done))
        if done is not None:
            done.wait(timeout=wait)

    def set_inference_start_paused(self, paused: bool) -> None:
        """Pause/release heavy inference startup while preview/capture opens."""
        if paused:
            self._inference_start.clear()
        else:
            self._inference_start.set()

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

    def set_system_cpu_pressure(self, pct: float) -> None:
        """Latest system CPU% (set by the supervisor ~1 Hz), used by the governor."""
        self._system_cpu_pressure = max(0.0, float(pct))

    def _feature(self, name: str) -> bool:
        """Thread-safe read of one feature flag (default True when unset)."""
        with self._cmd_lock:
            return bool(self._features.get(name, True))

    def ptz_home(self) -> None:
        """Drive the PTZ backend to its optical home position (low-latency path)."""
        with self._ptz_cmd_lock:
            self._ptz_cmd_queue.append(("ptz_home", None))
        self._capture_wake.set()

    def ptz_menu(self) -> None:
        """Toggle the camera's on-screen-display (OSD) menu (low-latency path)."""
        with self._ptz_cmd_lock:
            self._ptz_cmd_queue.append(("ptz_menu", None))
        self._capture_wake.set()

    def ptz_nudge(self, pan: float, tilt: float, zoom: float) -> None:
        # Fast path: drained on the capture thread so the move isn't delayed by an
        # in-flight inference pass (see _drain_ptz_commands / _ptz_cmd_queue).  Wake
        # a backed-off capture sleep so the move applies now, not after the backoff.
        with self._ptz_cmd_lock:
            self._ptz_cmd_queue.append(("ptz_nudge", (float(pan), float(tilt), float(zoom))))
        self._capture_wake.set()

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
                log.warning("camera_id=%s command %s failed", self.camera_id, kind, exc_info=True)

    def _drain_ptz_commands(self) -> None:
        """Apply manual PTZ commands on the CAPTURE thread for minimal latency.

        Drives the backend under ``_ptz_lock`` (serialised with the auto loop and
        the telemetry position read), at the full capture tick — so the joystick/
        D-pad moves the camera without waiting on the inference thread.  Safe
        no-op until the backend is built (``_drive_ptz_nudge`` guards ``None``).
        """
        with self._ptz_cmd_lock:
            if not self._ptz_cmd_queue:
                return
            pending = list(self._ptz_cmd_queue)
            self._ptz_cmd_queue.clear()
        for kind, payload in pending:
            try:
                with self._ptz_lock:
                    if kind == "ptz_nudge":
                        self._drive_ptz_nudge(*payload)
                    elif kind == "ptz_home":
                        self._ptz_home()
                    elif kind == "ptz_menu":
                        self._ptz_menu()
            except Exception:  # noqa: BLE001
                log.warning("camera_id=%s ptz cmd %s failed", self.camera_id, kind, exc_info=True)

    def _apply_command(self, kind: str, payload: Any) -> None:
        if kind == "enable_tracking":
            self._tracking_enabled = bool(payload)
        elif kind == "set_target":
            # An explicit track id supersedes identity targeting for this camera.
            self._commit_target_track(payload, reset_reid=True, reason="manual")
            self._target_identity_id = None
        elif kind == "set_target_identity":
            self._target_identity_id = payload
            # Clear the explicit track lock so the identity match takes over once
            # the named person is detected ("track when found").
            self._commit_target_track(None, reset_reid=True, reason="identity")
            self._reset_pose_aim()
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
            prev_tracker = getattr(self.config.tracking, "tracker", "")
            prev_mode = getattr(self.config.tracking, "tracking_mode", "stable")
            prev_backend = getattr(self.config.ptz, "backend", "")
            prev_addr = getattr(self.config.ptz, "address", "")
            self.config = payload
            # Rebuild the PTZ backend live when the transport changes — e.g. toggling
            # Center Stage flips backend to/from "digital". Without this the digital
            # backend is never created on a live switch and Center Stage does nothing
            # until an app restart.
            new_backend = getattr(payload.ptz, "backend", "")
            new_addr = getattr(payload.ptz, "address", "")
            if new_backend != prev_backend or new_addr != prev_addr:
                self._rebuild_ptz_backend()
            # Apply an fps change from a full-config push live too, so the UI's
            # fps slider takes effect whether it routes through updateCameraConfig
            # or the dedicated setTargetFps slot.
            new_fps = getattr(payload.source, "fps", None)
            if new_fps is not None and new_fps != prev_fps:
                self._apply_target_fps(float(new_fps))
            new_tracker = getattr(payload.tracking, "tracker", "")
            if new_tracker and new_tracker != prev_tracker:
                self._rebuild_tracker(prev_tracker, new_tracker, reason="setting changed")
            new_mode = getattr(payload.tracking, "tracking_mode", "stable")
            if new_mode != prev_mode:
                # Mode now drives ReID: stable holds via ReID, responsive doesn't.
                self._reset_reid()
                self._add_event(
                    "reid",
                    f"Tracking mode set to {new_mode}: ReID "
                    f"{'on' if new_mode == 'stable' else 'off'} for this camera.",
                )
            # Push the new PTZ tuning to the live controller (gains, lead time,
            # smoothing, safe zone, loss-recovery) without an engine restart.  The
            # controller was built with the *old* cfg, so this keeps it current.
            ctrl = self._ptz
            if ctrl is not None and hasattr(ctrl, "update_config"):
                try:
                    ctrl.update_config(payload.ptz)
                except Exception:  # noqa: BLE001
                    log.debug(
                        "camera_id=%s ptz update_config failed", self.camera_id, exc_info=True
                    )
        elif kind == "set_target_fps":
            self._apply_target_fps(float(payload))
        elif kind == "refresh_detector":
            self._refresh_detector_from_pool()
        elif kind == "reload_models":
            self._reload_inference_models()
        elif kind == "release_models":
            self._release_inference_models()
            if isinstance(payload, threading.Event):
                payload.set()
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
            if (
                self._last_applied_fps is None
                or abs(self._last_applied_fps - float(fps)) >= _FPS_LOG_DELTA
            ):
                log.debug("camera_id=%s target fps set live to %.0f", self.camera_id, fps)
                self._add_event("fps", f"FPS cap set to {fps:.0f}.")
            self._last_applied_fps = float(fps)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s set_target_fps failed", self.camera_id, exc_info=True)

    def _release_inference_models(self) -> None:
        """Drop detector/pose refs *without* rebuilding so their ORT sessions free.

        Called before the on-disk model cache is mutated: once these refs and the
        shared pool's are gone (and GC runs), Windows can delete/replace the files
        that onnxruntime had open.  The subsequent ``reload_models`` rebuilds.
        """
        self._detect = None
        self._unified_pose_active = False
        self._last_detections = []
        self._pose = None
        self._pose_probed = False
        self._pose_keypoints = None
        self._pose_kp_track_id = None

    def _reload_inference_models(self) -> None:
        """Force-drop + rebuild detector/pose to match the current model cache."""
        self._detect = None
        self._unified_pose_active = False
        self._last_detections = []
        self._pose = None
        self._pose_probed = False
        self._pose_keypoints = None
        self._pose_kp_track_id = None
        if self._feature("detection"):
            self._ensure_detect_stack()
            model = (
                getattr(self._pool, "detector_model_name", "")
                or getattr(self._pool, "detector_tier", "")
                if self._pool is not None
                else ""
            )
            if self._detect is None:
                self._add_event(
                    "detector", "Detector model removed; live-preview only.", level="warning"
                )
            else:
                self._add_event(
                    "detector", f"Detector reloaded: {model or self._ep or 'shared model'}."
                )
        # Pose rebuilds lazily through _ensure_pose on its next use.

    def _refresh_detector_from_pool(self) -> None:
        """Replace the detector pointer with the pool's active detector."""
        pool = self._pool
        if pool is None:
            return
        try:
            detector = pool.detector()
        except Exception:  # noqa: BLE001
            log.debug(
                "camera_id=%s detector refresh from pool failed", self.camera_id, exc_info=True
            )
            return
        if detector is None:
            self._add_event(
                "detector", "Detector refresh failed; kept current model.", level="warning"
            )
            return
        if self._detect is None:
            self._detect = self._build_detect_stack_pooled()
        elif self._pooled_detector:
            self._detect.detector = detector
            self._detect.ep = getattr(detector, "ep", "") or getattr(pool, "detector_ep", "")
        self._ep = getattr(detector, "ep", "") or getattr(pool, "detector_ep", "")
        model = getattr(pool, "detector_model_name", "") or getattr(pool, "detector_tier", "")
        self._add_event("detector", f"Detector active: {model or self._ep or 'shared model'}.")

    def _rebuild_tracker(self, old: str, new: str, *, reason: str = "") -> None:
        """Apply a tracker backend change live, preserving identity targeting."""
        now = time.time()
        self._tracker_switch = SwitchStateInfo(
            kind="tracker",
            state="warming",
            from_value=str(old or ""),
            to_value=str(new or ""),
            active_value=str(old or ""),
            reason=reason or "Tracker setting changed.",
            ts=now,
        )
        try:
            from autoptz.engine.pipeline.track import Tracker

            tracker = Tracker(
                tracker_type=new,
                coast_window=self.config.tracking.coast_window_ms / 1000.0,
            )
            if self._detect is None:
                self._detect = self._resolve_detect_stack()
            if self._detect is not None:
                self._detect.tracker = tracker
            # Manual track ids are backend-local and may not survive a rebuild.
            # Identity targeting stays and will reacquire on the next face/ReID hit.
            if self._target_identity_id is None:
                self._commit_target_track(None, reset_reid=True, reason="tracker_rebuild")
            self._track_identity.clear()
            self._reset_pose_aim()
            self._reset_reid()
            self._tracker_switch = SwitchStateInfo(
                kind="tracker",
                state="active",
                from_value=str(old or ""),
                to_value=str(new or ""),
                active_value=str(new or ""),
                reason=reason or "Tracker setting changed.",
                ts=time.time(),
            )
            if self._target_identity_id:
                msg = f"Tracker switched to {new}; identity target will reacquire when seen."
            else:
                msg = f"Tracker switched to {new}; manual target cleared."
            self._add_event("tracker", msg)
            log.info("camera_id=%s tracker switched %s -> %s", self.camera_id, old, new)
        except Exception as exc:  # noqa: BLE001
            self._tracker_switch = SwitchStateInfo(
                kind="tracker",
                state="failed",
                from_value=str(old or ""),
                to_value=str(new or ""),
                active_value=str(old or ""),
                reason=reason or "Tracker setting changed.",
                ts=time.time(),
                error=str(exc),
            )
            self._add_event(
                "tracker", f"Tracker switch to {new} failed; kept {old}.", level="warning"
            )
            log.warning(
                "camera_id=%s tracker switch %s -> %s failed",
                self.camera_id,
                old,
                new,
                exc_info=True,
            )

    def _save_ptz_preset(self, slot: int) -> None:
        """Store the current position into the backend's hardware preset *slot*.

        Safe no-op when no PTZ backend is configured.  Never raises into the
        command pump.
        """
        backend = self._ptz_backend
        if backend is None or not hasattr(backend, "save_preset"):
            log.debug(
                "camera_id=%s save_ptz_preset slot=%d ignored (no backend)", self.camera_id, slot
            )
            return
        try:
            backend.save_preset(slot)
            log.info("camera_id=%s saved PTZ preset slot=%d", self.camera_id, slot)
        except Exception:  # noqa: BLE001
            log.warning(
                "camera_id=%s save_preset slot=%d failed", self.camera_id, slot, exc_info=True
            )

    def _recall_ptz_preset(self, slot: int) -> None:
        """Recall the backend's hardware preset *slot*.

        Safe no-op when no PTZ backend is configured.  Never raises into the
        command pump.
        """
        backend = self._ptz_backend
        if backend is None or not hasattr(backend, "goto_preset"):
            log.debug(
                "camera_id=%s recall_ptz_preset slot=%d ignored (no backend)", self.camera_id, slot
            )
            return
        try:
            backend.goto_preset(slot)
            log.info("camera_id=%s recalled PTZ preset slot=%d", self.camera_id, slot)
        except Exception:  # noqa: BLE001
            log.warning(
                "camera_id=%s goto_preset slot=%d failed", self.camera_id, slot, exc_info=True
            )

    def _ptz_home(self) -> None:
        """Drive the backend to optical home, if it supports ``home()``.

        Opens a manual-override window so the auto loop doesn't immediately fight
        the recentre.  Safe no-op when no backend is configured or the backend
        lacks ``home()``.  Never raises into the command pump.
        """
        backend = self._ptz_backend
        if backend is None or not hasattr(backend, "home"):
            log.debug("camera_id=%s ptz_home ignored (no backend / unsupported)", self.camera_id)
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
            log.debug("camera_id=%s ptz_menu ignored (no backend / unsupported)", self.camera_id)
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
            log.debug(
                "camera_id=%s ptz nudge backend=%s pan=%.3f tilt=%.3f zoom=%.3f",
                self.camera_id,
                self.config.ptz.backend,
                pan,
                tilt,
                zoom,
            )
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s ptz nudge failed", self.camera_id, exc_info=True)

    def _manual_override_active(self, now: float) -> bool:
        return now < self._manual_override_until

    @_appearance_guarded
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

        # Tell the controller how long this machine's capture+inference pipeline
        # takes so its lead time anticipates the real delay (lead_time_auto).
        ctrl.set_loop_latency(self._latency_ms / 1000.0)

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
            fh = max(1, int(frame.shape[0]))
            box_h_frac = (float(target.bbox.y2) - float(target.bbox.y1)) / fh
            occluded = self._target_box_collapsed(box_h_frac)
            err, height = self._track_error(target, frame, now, tracks=tracks)
            vel = self._estimate_aim_velocity(err, now)
            try:
                # A box that suddenly collapsed (occlusion → only a partial body
                # visible) is not a trustworthy aim — coast (track_active=False) so
                # the controller holds instead of chasing it onto the legs/last-known
                # partial position, and resumes when the subject reappears in full.
                pan, tilt, zoom = ctrl.step(err, vel, height, track_active=not occluded, t=now)
                self._ptz_last_cmd = (pan, tilt, zoom)
                self._log_ptz_cmd("auto", pan, tilt, zoom)
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s ptz auto step failed", self.camera_id, exc_info=True)
        else:
            # No target → drop the velocity estimate so a re-acquire starts clean.
            self._prev_aim_err = None
            self._aim_vel = (0.0, 0.0)
            try:
                pan, tilt, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, track_active=False, t=now)
                self._ptz_last_cmd = (pan, tilt, zoom)
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s ptz auto idle step failed", self.camera_id, exc_info=True)

    def _update_ego_motion(
        self,
        tracks: list[TrackInfo],
        frame: NDArray[np.uint8] | None,
        now: float,
    ) -> None:
        """Measure this tick's camera-induced image motion (error-space units/s).

        Stored in ``self._ego_vel`` so :meth:`_estimate_aim_velocity` can subtract
        it.  Person boxes are masked out of the optical flow so the subject's own
        motion doesn't bias the camera estimate.  No-op (zero) when disabled.
        """
        if frame is None or not getattr(self.config.ptz, "ego_comp_enabled", True):
            self._ego_vel = (0.0, 0.0)
            self._ego_source = "none"
            self._ego_fresh = False
            return
        # Sentinel default 1 so a serialized config missing this field falls back to legacy
        # every-frame behaviour; the normal Pydantic path always supplies the field default of 3.
        interval = max(1, int(getattr(self.config.ptz, "ego_comp_interval", 1)))
        if interval > 1 and (self._frames_inferred % interval) != 0:
            # Off-cadence: no fresh measurement this tick. Mark not-fresh so the aim
            # feed-forward zero-subtracts ego instead of subtracting a stale value —
            # subtracting a decayed estimate every tick is what made the camera
            # ping-pong when it and the subject both moved.
            self._ego_fresh = False
            return
        boxes = [
            (t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2)
            for t in tracks
            if getattr(t, "bbox", None) is not None
        ]
        try:
            ego = self._ego_estimator.estimate(frame, now, boxes=boxes, ptz_cmd=self._ptz_last_cmd)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s ego-motion estimate failed", self.camera_id, exc_info=True)
            self._ego_vel = (0.0, 0.0)
            self._ego_source = "none"
            self._ego_fresh = False
            return
        self._ego_vel = (ego.vx, ego.vy)
        self._ego_source = ego.source
        self._ego_fresh = True

    def _estimate_aim_velocity(
        self,
        err: tuple[float, float],
        now: float,
    ) -> tuple[float, float]:
        """EMA-smoothed d(error)/dt in normalized units/sec for PTZ feed-forward.

        The raw d(error)/dt mixes the subject's motion with the image shift the
        camera itself caused; subtracting the per-tick ego-motion estimate
        (``self._ego_vel``, also error-space units/sec) leaves the subject's
        *world* motion, so the controller's velocity feed-forward stops chasing
        the camera's own pan/tilt (the hunting/oscillation fix).

        **Window matching (the jitter fix):** ego-motion is measured only once per
        ``ego_comp_interval`` frames and spans that whole window.  The per-tick
        d(error)/dt spans a single frame, so subtracting the ego estimate only
        cancels cleanly when the two cover the SAME interval.  We therefore only
        recompute the velocity on a freshly-measured (``_ego_fresh``) tick — where
        the error delta has accrued over the same window — and HOLD the last value
        between fresh ticks.  Previously the off-cadence ticks zero-subtracted ego
        and injected the full camera-pan velocity into the feed-forward on 2 of
        every 3 frames, which is what made the camera hunt during pans.
        """
        # With ego comp ENABLED, skip the off-cadence ticks (hold last velocity)
        # so the error delta and the ego estimate always span the same window.
        # With ego comp OFF there is no multi-frame estimate to match, so compute
        # every tick as before (no ego is subtracted regardless).
        ego_on = bool(getattr(self.config.ptz, "ego_comp_enabled", True))
        if ego_on and not self._ego_fresh:
            return self._aim_vel

        vx = vy = 0.0
        prev = self._prev_aim_err
        if prev is not None:
            dt = now - self._prev_aim_t
            if dt > 1e-3:
                ego_vx, ego_vy = self._ego_vel if self._ego_fresh else (0.0, 0.0)
                raw_vx = (err[0] - prev[0]) / dt - ego_vx
                raw_vy = (err[1] - prev[1]) / dt - ego_vy
                a = 0.5  # EMA: balance responsiveness vs. jitter rejection
                vx = a * raw_vx + (1.0 - a) * self._aim_vel[0]
                vy = a * raw_vy + (1.0 - a) * self._aim_vel[1]
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
        log.debug(
            "camera_id=%s ptz %s backend=%s pan=%.3f tilt=%.3f zoom=%.3f",
            self.camera_id,
            source,
            self.config.ptz.backend,
            pan,
            tilt,
            zoom,
        )

    def _resolve_target_track(self, tracks: list[TrackInfo]) -> TrackInfo | None:
        """Pick the tracked subject to follow: the explicit target, else None."""
        if self._target_track_id is None:
            return None
        for t in tracks:
            if t.track_id == self._target_track_id:
                return t
        return None

    def _commit_target_track(
        self,
        track_id: int | None,
        *,
        reset_reid: bool = False,
        reason: str = "",
    ) -> None:
        """Commit the active target track through one state-transition path.

        Guarded by ``_appearance_lock`` (re-entrant) because both the hot thread
        (user set-target / clear) and the async appearance thread (ReID rebind)
        commit targets — this is the single serialised target-id transition.
        """
        with self._appearance_lock:
            if track_id != self._target_track_id:
                self._reset_pose_aim()
                self._target_h_ref = None
                self._target_lock = _TargetLockState(status="pending", reason=reason)
                self._reid_recover_candidate = None
                self._identity_recover_candidate = None
                if reset_reid:
                    self._reset_reid()
            elif track_id is None:
                self._target_lock = _TargetLockState(status="idle", reason=reason)
                if reset_reid:
                    self._reset_reid()
            self._target_track_id = track_id

    def _target_box_collapsed(self, height_frac: float) -> bool:
        """True when the target box height suddenly collapsed vs its recent healthy
        size — the occlusion signature (the subject covered by something leaves only
        a partial box, e.g. just the legs). A gradual shrink (the subject walking
        away) updates the reference and is NOT flagged, because the slow reference
        keeps pace; a sudden drop IS flagged, because the reference hasn't caught up.
        The caller coasts the PTZ instead of chasing the box down to the last-known
        partial position.
        """
        h = max(0.0, float(height_frac))
        ref = self._target_h_ref
        if ref is None or ref <= 0.0:
            self._target_h_ref = h
            return False
        if h < ref * _OCCLUSION_COLLAPSE_FRAC:
            return True  # leave the reference intact so a full-size reappearance recovers
        self._target_h_ref = ref + _OCCLUSION_REF_ALPHA * (h - ref)
        return False

    @staticmethod
    def _bbox_area(bb: BBox) -> float:
        return max(0.0, float(bb.x2 - bb.x1)) * max(0.0, float(bb.y2 - bb.y1))

    @staticmethod
    def _bbox_center(bb: BBox) -> tuple[float, float]:
        return ((float(bb.x1) + float(bb.x2)) * 0.5, (float(bb.y1) + float(bb.y2)) * 0.5)

    @classmethod
    def _bbox_iou(cls, a: BBox, b: BBox) -> float:
        ix1 = max(float(a.x1), float(b.x1))
        iy1 = max(float(a.y1), float(b.y1))
        ix2 = min(float(a.x2), float(b.x2))
        iy2 = min(float(a.y2), float(b.y2))
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = cls._bbox_area(a) + cls._bbox_area(b) - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def _bbox_y_overlap_frac(a: BBox, b: BBox) -> float:
        inter = max(0.0, min(float(a.y2), float(b.y2)) - max(float(a.y1), float(b.y1)))
        denom = max(1.0, min(float(a.y2 - a.y1), float(b.y2 - b.y1)))
        return inter / denom

    @staticmethod
    def _bbox_x_gap(a: BBox, b: BBox) -> float:
        if a.x2 < b.x1:
            return float(b.x1 - a.x2)
        if b.x2 < a.x1:
            return float(a.x1 - b.x2)
        return 0.0

    def _sync_target_flags(self, tracks: list[TrackInfo]) -> None:
        """Make TrackInfo.is_target reflect the current committed target id."""
        for t in tracks:
            t.is_target = self._target_track_id is not None and t.track_id == self._target_track_id

    def _target_label(self, track: TrackInfo | None = None) -> str:
        if track is not None:
            return track.identity or f"ID {track.track_id}"
        lock = self._target_lock
        if lock.trusted_identity:
            return lock.trusted_identity
        if lock.trusted_track_id is not None:
            return f"ID {lock.trusted_track_id}"
        if self._target_track_id is not None:
            return f"ID {self._target_track_id}"
        return "target"

    def _store_trusted_target(self, target: TrackInfo, now: float) -> None:
        lock = self._target_lock
        lock.trusted_track_id = target.track_id
        lock.trusted_bbox = target.bbox.model_copy()
        lock.trusted_identity = target.identity
        lock.trusted_identity_id = target.identity_id
        lock.trusted_confidence = float(target.confidence or 0.0)
        lock.trusted_t = now
        lock.status = "locked"
        lock.reason = ""

    def _target_bbox_jumped(self, target: TrackInfo) -> bool:
        prev = self._target_lock.trusted_bbox
        if prev is None:
            return False
        cur = target.bbox
        pcx, pcy = self._bbox_center(prev)
        ccx, ccy = self._bbox_center(cur)
        jump = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5
        prev_diag = max(1.0, ((prev.x2 - prev.x1) ** 2 + (prev.y2 - prev.y1) ** 2) ** 0.5)
        cur_diag = max(1.0, ((cur.x2 - cur.x1) ** 2 + (cur.y2 - cur.y1) ** 2) ** 0.5)
        limit = max(_TARGET_JUMP_MIN_PX, min(prev_diag, cur_diag) * _TARGET_JUMP_SCALE)
        return jump > limit and self._bbox_iou(prev, cur) < _TARGET_JUMP_IOU

    def _target_crowded(self, target: TrackInfo, tracks: list[TrackInfo]) -> bool:
        tb = target.bbox
        tw = max(1.0, float(tb.x2 - tb.x1))
        for other in tracks:
            if other.track_id == target.track_id or getattr(other, "lost", False):
                continue
            ob = other.bbox
            if self._bbox_iou(tb, ob) >= _TARGET_OVERLAP_IOU:
                return True
            y_overlap = self._bbox_y_overlap_frac(tb, ob)
            gap = self._bbox_x_gap(tb, ob)
            ow = max(1.0, float(ob.x2 - ob.x1))
            if y_overlap >= _TARGET_CLOSE_Y_OVERLAP and gap <= min(tw, ow) * _TARGET_CLOSE_GAP_FRAC:
                return True
        return False

    def _target_pose_trusted(
        self,
        target: TrackInfo,
        frame: NDArray[np.uint8] | None,
        now: float,
        tracks: list[TrackInfo] | None = None,
    ) -> bool:
        """Return True when target body keypoints are good enough to hold lock.

        Box overlap alone is a weak signal in real scenes: singers, speakers, and
        musicians often stand close together, so the selected person's detector
        box can overlap another person even while their torso keypoints are
        stable and inside the selected track.  In that case we should trust the
        body position and keep following instead of entering "Target blocked".

        The pose path still rejects wrong-person evidence: ``_pose_aim`` calls
        ``_pose_keypoints_consistent()``, which checks the keypoint anchor is
        inside this target box and not jumping away from the last trusted aim.
        """
        if frame is None or not self._feature("pose"):
            return False
        anchor, conf, _height = self._pose_aim(target, frame, now, tracks=tracks)
        return anchor is not None and conf >= _POSE_TARGET_LOCK_MIN_CONF

    def _mark_target_ambiguous(
        self,
        target: TrackInfo | None,
        *,
        now: float,
        reason: str,
    ) -> None:
        lock = self._target_lock
        lock.status = "ambiguous"
        lock.reason = reason
        lock.ambiguous_until = max(lock.ambiguous_until, now + _TARGET_AMBIGUOUS_HOLD_S)
        self._pose_keypoints = None
        self._pose_kp_track_id = None
        if target is None or lock.trusted_bbox is None:
            return
        target.lost = True
        target.bbox = lock.trusted_bbox.model_copy()
        target.identity = lock.trusted_identity
        target.identity_id = lock.trusted_identity_id
        target.confidence = lock.trusted_confidence
        target.vx = 0.0
        target.vy = 0.0
        if lock.trusted_aim is not None:
            target.aim_x, target.aim_y = lock.trusted_aim
            target.aim_source = "held"

    def _build_candidate_cues(
        self,
        tracks: list[TrackInfo],
    ) -> list[Any]:
        """Build ``CandidateCues`` for every non-lost track.

        Called only when ``use_target_associator`` is True.  Each cue is marked
        CONSERVATIVE — only available when the signal genuinely exists this tick:

        - **motion**: IOU of the candidate's bbox against ``_target_lock.trusted_bbox``
          (the current target's last stored bbox).  Unavailable when there is no
          current target or no stored trusted bbox.
        - **identity**: available only when ``_track_identity`` maps this track to
          the camera's configured target identity (``_target_identity_id``).
        - **pose**: unavailable here (pose is per-target; we don't have per-candidate
          pose keypoints — never run pose just for cue-building).
        - **appearance**: unavailable here (ReID template scores are not stored
          per-candidate; we don't run ReID in this path — the ReID recovery loop
          is a separate async path).
        """
        from autoptz.engine.pipeline.associator import CandidateCues, Cue

        ref_bbox = self._target_lock.trusted_bbox
        has_ref = ref_bbox is not None and self._target_track_id is not None
        cues: list[Any] = []
        for t in tracks:
            if getattr(t, "lost", False):
                continue
            # -- motion --
            if has_ref and ref_bbox is not None:
                motion_iou = self._bbox_iou(ref_bbox, t.bbox)
                motion_cue = Cue(float(motion_iou), available=True)
            else:
                motion_cue = Cue(0.0, available=False)

            # -- identity --
            # Available iff this track has a recognised identity AND it matches
            # the operator's configured target identity for this camera.
            identity_cue = Cue(0.0, available=False)
            target_iid = self._target_identity_id
            if target_iid is not None:
                entry = self._track_identity.get(t.track_id)
                if entry is not None:
                    track_iid, _name, score = entry
                    if track_iid == target_iid:
                        identity_cue = Cue(float(score), available=True)

            # -- pose --
            # Per-candidate pose is not computed here (it's per-target only).
            pose_cue = Cue(0.0, available=False)

            # -- appearance --
            # ReID template scores are not stored per-candidate for the associator
            # path (they live inside the recovery loop).  Mark unavailable.
            appearance_cue = Cue(0.0, available=False)

            cues.append(
                CandidateCues(
                    track_id=t.track_id,
                    motion=motion_cue,
                    appearance=appearance_cue,
                    pose=pose_cue,
                    identity=identity_cue,
                )
            )
        return cues

    def _apply_associator_decision(
        self,
        decision: Any,
        tracks: list[TrackInfo],
        now: float,
    ) -> None:
        """Apply a ``Decision`` returned by ``TargetAssociator.decide``.

        keep      → no change (``_target_lock`` updated by normal store path).
        switch    → commit the new target via ``_commit_target_track``.
        ambiguous → mark the current lock ambiguous (hold and freeze).
        """
        action = decision.action
        if action == "keep":
            # Refresh the trusted state for the existing target if it's visible.
            target = self._resolve_target_track(tracks)
            if target is not None and not getattr(target, "lost", False):
                self._store_trusted_target(target, now)
                # Surface associator confidence in the lock.
                self._target_lock.trusted_confidence = float(decision.confidence)
        elif action == "switch" and decision.track_id is not None:
            self._commit_target_track(decision.track_id, reason="associator")
            # Store the new target's bbox immediately so the next tick has a ref.
            new_target = next(
                (
                    t
                    for t in tracks
                    if t.track_id == decision.track_id and not getattr(t, "lost", False)
                ),
                None,
            )
            if new_target is not None:
                self._store_trusted_target(new_target, now)
                # Intentionally override detector confidence with the associator's
                # fused confidence — the fused value is what surfaces in telemetry.
                self._target_lock.trusted_confidence = float(decision.confidence)
            log.debug(
                "camera_id=%s associator switch → track=%d conf=%.2f margin=%.2f",
                self.camera_id,
                decision.track_id,
                decision.confidence,
                decision.margin,
            )
        elif action == "ambiguous":
            target = self._resolve_target_track(tracks)
            self._mark_target_ambiguous(
                target,
                now=now,
                reason="associator_ambiguous",
            )
            log.debug(
                "camera_id=%s associator ambiguous conf=%.2f margin=%.2f",
                self.camera_id,
                decision.confidence,
                decision.margin,
            )

    def _append_held_target(self, tracks: list[TrackInfo], now: float) -> None:
        lock = self._target_lock
        if self._target_track_id is None or lock.trusted_bbox is None:
            return
        if now - lock.trusted_t > _TARGET_HOLD_S:
            lock.status = "lost"
            lock.reason = "missing"
            return
        tracks.append(
            TrackInfo(
                track_id=self._target_track_id,
                bbox=lock.trusted_bbox.model_copy(),
                identity=lock.trusted_identity,
                identity_id=lock.trusted_identity_id,
                confidence=lock.trusted_confidence,
                is_target=True,
                lost=True,
                aim_x=lock.trusted_aim[0] if lock.trusted_aim is not None else None,
                aim_y=lock.trusted_aim[1] if lock.trusted_aim is not None else None,
                aim_source="held" if lock.trusted_aim is not None else "",
            )
        )
        lock.status = "coasting"
        lock.reason = "missing"

    @_appearance_guarded
    def _apply_target_lock(
        self,
        tracks: list[TrackInfo],
        frame: NDArray[np.uint8] | None,
        now: float,
    ) -> None:
        """Suppress target evidence from obvious crossings or pose-risk crowding.

        When ``config.tracking.use_target_associator`` is True the fused
        ``TargetAssociator`` drives keep/switch/ambiguous and the method returns
        early — the existing heuristic path below is completely bypassed.
        When the flag is False (default) behaviour is byte-identical to today.
        """
        self._sync_target_flags(tracks)
        if self._target_track_id is None:
            self._target_lock.status = "idle"
            return
        target = self._resolve_target_track(tracks)
        if target is None:
            self._append_held_target(tracks, now)
            return
        if target.lost:
            return

        # ── flag-gated associator path (OFF by default) ──────────────────────────
        if getattr(self.config.tracking, "use_target_associator", False):
            cues = self._build_candidate_cues(tracks)
            decision = self._associator.decide(cues, self._target_track_id)
            self._apply_associator_decision(decision, tracks, now)
            return
        # ── existing heuristic path (flag OFF — unchanged) ───────────────────────

        if (
            self._target_lock.trusted_track_id != target.track_id
            or self._target_lock.trusted_bbox is None
        ):
            self._store_trusted_target(target, now)
            return
        # Only DISTRUST the target (freeze + go ambiguous) when there's a real
        # wrong-person risk: another track is crowding it AND its box jumped to a
        # low-overlap position.  A lone subject that simply moves fast is NOT
        # ambiguous — freezing on it was the "tracking keeps blocking itself when
        # they move away too quickly" bug.  Crowd disambiguation still defers to
        # pose when the body keypoints vouch for the same person.
        crowded = self._target_crowded(target, tracks)
        if crowded and self._target_bbox_jumped(target):
            if self._target_pose_trusted(target, frame, now, tracks):
                self._store_trusted_target(target, now)
                return
            self._mark_target_ambiguous(target, now=now, reason="blocked")
            return
        self._store_trusted_target(target, now)

    # ── appearance ReID recovery ─────────────────────────────────────────────────

    def _reid_active(self) -> bool:
        """Whether appearance ReID should run for this camera right now.

        Unified control: the global ``reid`` feature must be on AND the camera's
        ``tracking_mode`` must be ``stable`` ("hold the target through crossings").
        ``responsive`` follows the freshest detection with no ReID hold.
        """
        return (
            self._feature("reid")
            and getattr(self.config.tracking, "tracking_mode", "stable") == "stable"
        )

    def _ensure_reid(self) -> Any | None:
        """Lazily build the OSNet ReID matcher when active (None if unavailable).

        Gated on :meth:`_reid_active`; degrades gracefully to ``None`` when
        boxmot/torch/weights are absent so motion-only tracking still works.
        """
        if not self._reid_active():
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
                log.debug("camera_id=%s BodyReID build failed", self.camera_id, exc_info=True)
                self._reid = None
        return self._reid

    def _reset_reid(self) -> None:
        """Drop the appearance template (target changed / cleared)."""
        self._reid_recover_candidate = None
        self._identity_recover_candidate = None
        if self._reid is not None:
            try:
                self._reid.reset()
            except Exception:  # noqa: BLE001
                pass

    @_appearance_guarded
    def _maybe_reid_recover(
        self,
        tracks: list[TrackInfo],
        frame: NDArray[np.uint8] | None,
        now: float,
    ) -> None:
        """Keep the target template fresh while visible; re-bind it when lost.

        While the target track is present we refresh its EMA appearance template;
        once its track_id disappears (occlusion / ID switch) we embed the current
        candidate boxes and re-bind the target onto the best appearance match —
        so the lock follows the *person*, not whichever box happens to be nearest.
        Throttled to ``_REID_INTERVAL_S`` and a no-op unless ReID is enabled.
        """
        if self._target_track_id is None or frame is None or not self._feature("tracking"):
            return
        if self._target_lock.status == "ambiguous":
            return
        reid = self._ensure_reid()
        if reid is None or not getattr(reid, "available", False):
            return
        if now - self._last_reid_t < self._effective_reid_interval():
            return
        self._last_reid_t = now

        visible_tracks = [t for t in tracks if not getattr(t, "lost", False)]
        present = {t.track_id: t for t in visible_tracks}
        if self._target_track_id in present:
            # Target visible → refresh (or seed) its appearance template.
            tgt = present[self._target_track_id]
            emb = reid.embed([_xyxy(tgt.bbox)], frame)
            if emb.size:
                reid.update_target(emb[0])
            return

        # Target lost → recover onto the best-matching present track.
        if not getattr(reid, "locked", False) or not visible_tracks:
            return
        cand = reid.embed([_xyxy(t.bbox) for t in visible_tracks], frame)
        if cand.size == 0:
            return
        from autoptz.engine.pipeline import reid as _reid_mod  # noqa: PLC0415

        eff_hi = _reid_mod.adaptive_threshold_hi(
            list(cand),
            base_hi=float(self.config.tracking.reid_threshold_hi),
        )
        result = reid.recover(
            cand,
            threshold=eff_hi,
            update=False,
        )
        if result.matched and 0 <= result.best_index < len(visible_tracks):
            if len(result.scores) > 1:
                ranked = sorted(result.scores, reverse=True)
                if ranked[0] - ranked[1] < _REID_RECOVERY_MARGIN:
                    self._target_lock.status = "ambiguous"
                    self._target_lock.reason = "reid_margin"
                    self._target_lock.ambiguous_until = max(
                        self._target_lock.ambiguous_until,
                        now + _TARGET_AMBIGUOUS_HOLD_S,
                    )
                    return
            new_id = visible_tracks[result.best_index].track_id
            if new_id != self._target_track_id:
                if not self._reid_rebind_allows(new_id):
                    # Named-identity target: body appearance alone must not drift the
                    # lock onto a different/unknown person — wait for a face match.
                    return
                if not self._reid_recovery_confirmed(new_id):
                    return
                # Now that the candidate is confirmed, blend the matching feature
                # into the target template exactly once.
                reid.recover(
                    cand[[result.best_index]],
                    threshold=eff_hi,
                    update=True,
                )
                log.info(
                    "camera_id=%s ReID recovered target → track=%d score=%.2f",
                    self.camera_id,
                    new_id,
                    result.best_score,
                )
                self._commit_target_track(new_id, reason="reid")

    def _reid_rebind_allows(self, new_id: int) -> bool:
        """Whether appearance ReID recovery may re-bind the target to *new_id*.

        For a target selected by *identity* (name), body appearance alone must
        never move the lock onto a different person — re-acquiring a named target
        requires a face-identity match. So a ReID re-bind is allowed only when the
        candidate track is face-confirmed as the *same* identity. A manual/clicked
        target (no identity) keeps the appearance-based re-bind it relies on.
        """
        if self._target_identity_id is None:
            return True
        entry = self._track_identity.get(new_id)
        return entry is not None and entry[0] == self._target_identity_id

    def _reid_recovery_confirmed(self, track_id: int) -> bool:
        """Return True when the active tracking mode allows rebinding now."""
        mode = getattr(self.config.tracking, "tracking_mode", "stable")
        required = (
            _REID_RECOVERY_CONFIRM_STABLE if mode == "stable" else _REID_RECOVERY_CONFIRM_RESPONSIVE
        )
        prev_id, count = self._reid_recover_candidate or (None, 0)
        count = count + 1 if prev_id == track_id else 1
        self._reid_recover_candidate = (track_id, count)
        return count >= required

    def _track_error(
        self,
        track: TrackInfo,
        frame: NDArray[np.uint8],
        now: float | None = None,
        *,
        tracks: list[TrackInfo] | None = None,
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

        # Per-frame memo: _track_error runs for BOTH the PTZ loop and the aim-dot
        # annotation each tick — compute once so the fused-aim smoother advances a
        # single step per frame (double-stepping doubled the speed/jitter).
        if now is not None and self._aim_cache is not None:
            c_now, c_tid, c_err, c_height, c_ax, c_ay, c_src = self._aim_cache
            if c_now == now and c_tid == track.track_id:
                if track.is_target:
                    track.aim_x, track.aim_y, track.aim_source = c_ax, c_ay, c_src
                return c_err, c_height

        bb = track.bbox
        bbox_height = (bb.y2 - bb.y1) / h
        # Arms toggle: "torso" ignores arms (steady pose-torso zoom span); any
        # other value ("full_silhouette") includes arms → zoom to the full box.
        ignore_arms = getattr(self.config.tracking, "aim_body_mode", "torso") == "torso"
        framing_name = _resolve_framing(self.config.tracking)

        # ── bbox anchor — always available (centre-x, framing fraction down) ─────
        ax_bbox = (bb.x1 + bb.x2) * 0.5
        ay_bbox = bb.y1 + (bb.y2 - bb.y1) * AIM_REGION_FRACTION.get(framing_name, 0.5)
        ax, ay = ax_bbox, ay_bbox
        subject_height = bbox_height
        aim_source = "bbox"

        # ── pose anchor (landmark-precise) FUSED with the bbox by confidence ─────
        # No hard switch: the dot rides the body when pose is strong and leans on
        # the box when it isn't, so it never snaps between the two.  now is None for
        # pure-bbox callers / unit tests (deterministic, un-smoothed).
        if now is not None:
            pose_anchor, pose_conf, torso_height = self._pose_aim(
                track,
                frame,
                now,
                tracks=tracks,
            )
            if pose_anchor is not None and pose_conf > 0.0:
                w_pose = max(0.0, min(1.0, pose_conf))
                ax = w_pose * pose_anchor[0] + (1.0 - w_pose) * ax_bbox
                ay = w_pose * pose_anchor[1] + (1.0 - w_pose) * ay_bbox
                aim_source = "pose" if w_pose >= 0.66 else "fused" if w_pose >= 0.25 else "bbox"
                subject_height = (
                    torso_height if (ignore_arms and torso_height > 0.0) else bbox_height
                )
            # Smooth the fused point: stable when still, snappy on a Frame-on change.
            ax, ay = self._smooth_aim((ax, ay), track, framing_name, float(max(w, h)))

        if track.is_target:
            track.aim_x = float(ax)
            track.aim_y = float(ay)
            track.aim_source = aim_source
            if not getattr(track, "lost", False):
                self._target_lock.trusted_aim = (float(ax), float(ay))

        ex = (ax - w * 0.5) / (w * 0.5)  # [-1, 1] right-positive
        ey = -((ay - h * 0.5) / (h * 0.5))  # [-1, 1] up-positive
        ex = max(-1.0, min(1.0, ex))
        ey = max(-1.0, min(1.0, ey))
        subject_height = max(0.0, min(1.0, subject_height))
        if now is not None:
            self._aim_cache = (
                now,
                track.track_id,
                (ex, ey),
                subject_height,
                float(ax),
                float(ay),
                aim_source,
            )
        return (ex, ey), subject_height

    def _smooth_aim(
        self,
        point: tuple[float, float],
        track: TrackInfo,
        framing_name: str,
        frame_extent: float,
    ) -> tuple[float, float]:
        """EMA-smooth the fused aim point in pixel space.

        Stable when the subject is still, quicker when they move, and it **snaps**
        to the new anchor when the operator changes Frame-on (so the adjustment is
        visible).  A small size-scaled deadband rejects sub-pixel keypoint jitter
        without hiding real moves."""
        try:
            from autoptz.engine.pipeline import framing
        except Exception:  # noqa: BLE001
            return point
        if self._aim_smoother is None:
            self._aim_smoother = framing.AimSmoother(alpha=_POSE_AIM_ALPHA)
        # Frame-on changed → jump to the new region rather than easing slowly.
        if framing_name != self._last_aim_framing:
            self._last_aim_framing = framing_name
            self._aim_smoother.reset()
        speed = 0.0
        try:
            vx = float(getattr(track, "vx", 0.0) or 0.0)
            vy = float(getattr(track, "vy", 0.0) or 0.0)
            speed = (vx * vx + vy * vy) ** 0.5
        except Exception:  # noqa: BLE001
            pass
        self._aim_smoother._alpha = 0.20 if speed < 3.0 else 0.6
        prev = self._aim_smoother.value
        if prev is not None and speed < 3.0:
            px, py = prev
            rx, ry = point
            deadband = max(2.0, min(12.0, frame_extent * 0.006))
            if (rx - px) ** 2 + (ry - py) ** 2 <= deadband * deadband:
                point = prev
        out = self._aim_smoother.update(point)
        return out if out is not None else point

    def _annotate_target_aim(
        self,
        tracks: list[TrackInfo],
        frame: NDArray[np.uint8] | None,
        now: float,
    ) -> None:
        """Populate aim telemetry for the active live target, even without PTZ."""
        if frame is None:
            return
        target = self._resolve_target_track(tracks)
        if target is None or target.lost:
            return
        try:
            self._track_error(target, frame, now, tracks=tracks)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s target aim annotation failed", self.camera_id, exc_info=True)

    # ── pose-stable aim (lazy, graceful) ────────────────────────────────────────

    @staticmethod
    def _point_inside_bbox_margin(
        point: tuple[float, float],
        bbox: BBox,
        margin_frac: float,
    ) -> bool:
        x, y = point
        bw = max(1.0, float(bbox.x2 - bbox.x1))
        bh = max(1.0, float(bbox.y2 - bbox.y1))
        mx, my = bw * margin_frac, bh * margin_frac
        return (
            float(bbox.x1) - mx <= x <= float(bbox.x2) + mx
            and float(bbox.y1) - my <= y <= float(bbox.y2) + my
        )

    def _pose_owned_by_track(
        self,
        kps: list[Any],
        track: TrackInfo,
        tracks: list[TrackInfo] | None,
    ) -> bool:
        """Return True when torso/head keypoints clearly belong to ``track``.

        In crowded crops a single-person pose model can choose the neighboring
        body.  Aim math must reject that evidence before it can move the reticle
        or teach target lock state.
        """
        try:
            from autoptz.engine.pipeline import framing
        except Exception:  # noqa: BLE001
            return False

        indices = (
            framing.KP_NOSE,
            framing.KP_LEFT_EYE,
            framing.KP_RIGHT_EYE,
            framing.KP_LEFT_SHOULDER,
            framing.KP_RIGHT_SHOULDER,
            framing.KP_LEFT_HIP,
            framing.KP_RIGHT_HIP,
        )
        points = []
        for idx in indices:
            if 0 <= idx < len(kps):
                kp = kps[idx]
                try:
                    if kp.usable():
                        points.append((float(kp.x), float(kp.y)))
                except Exception:  # noqa: BLE001
                    continue
        if not points:
            return False

        inside = [
            p
            for p in points
            if self._point_inside_bbox_margin(p, track.bbox, _POSE_KEYPOINT_MARGIN)
        ]
        needed = max(1, int(round(len(points) * 0.60)))
        if len(inside) < needed:
            return False

        cx = sum(x for x, _y in points) / len(points)
        cy = sum(y for _x, y in points) / len(points)
        center = (cx, cy)
        if not self._point_inside_bbox_margin(center, track.bbox, _POSE_BBOX_MARGIN):
            return False

        if not tracks:
            return True

        tcx, tcy = self._bbox_center(track.bbox)
        target_d2 = (cx - tcx) ** 2 + (cy - tcy) ** 2
        target_diag = max(
            1.0,
            ((track.bbox.x2 - track.bbox.x1) ** 2 + (track.bbox.y2 - track.bbox.y1) ** 2) ** 0.5,
        )
        ambiguous_slack = target_diag * 0.18
        for other in tracks:
            if other.track_id == track.track_id or getattr(other, "lost", False):
                continue
            if not self._point_inside_bbox_margin(center, other.bbox, _POSE_KEYPOINT_MARGIN):
                continue
            ocx, ocy = self._bbox_center(other.bbox)
            other_d2 = (cx - ocx) ** 2 + (cy - ocy) ** 2
            if other_d2 + ambiguous_slack * ambiguous_slack < target_d2:
                return False
            if self._bbox_iou(track.bbox, other.bbox) >= 0.25:
                return False
        return True

    def _pose_keypoints_consistent(
        self,
        kps: list[Any],
        track: TrackInfo,
        frame: NDArray[np.uint8],
        now: float,
        tracks: list[TrackInfo] | None = None,
    ) -> bool:
        del frame  # reserved for future image-space consistency checks
        try:
            from autoptz.engine.pipeline import framing
        except Exception:  # noqa: BLE001
            return False

        raw_aim, conf = framing.body_aim_point(
            kps,
            framing=_resolve_framing(self.config.tracking),
        )
        if raw_aim is None or conf <= 0.0:
            return False
        # A pose anchor that lands outside the box or jumps far from the last
        # trusted aim just means "don't trust the keypoints this frame" — fall
        # back to the bbox aim and KEEP FOLLOWING.  It must NOT freeze tracking
        # (the old behaviour made the camera stall whenever the subject moved).
        if not self._point_inside_bbox_margin(raw_aim, track.bbox, _POSE_BBOX_MARGIN):
            return False
        if not self._pose_owned_by_track(kps, track, tracks):
            return False

        trusted = self._target_lock.trusted_aim
        if trusted is not None:
            dx = raw_aim[0] - trusted[0]
            dy = raw_aim[1] - trusted[1]
            dist = (dx * dx + dy * dy) ** 0.5
            bb = track.bbox
            diag = max(1.0, ((bb.x2 - bb.x1) ** 2 + (bb.y2 - bb.y1) ** 2) ** 0.5)
            if dist > max(_POSE_JUMP_MIN_PX, diag * _POSE_JUMP_SCALE):
                return False
        return True

    def _pose_aim(
        self,
        track: TrackInfo,
        frame: NDArray[np.uint8],
        now: float,
        *,
        tracks: list[TrackInfo] | None = None,
    ) -> tuple[tuple[float, float] | None, float, float]:
        """Return ``(raw_anchor | None, confidence, subject_height_fraction)``.

        The *raw* (un-smoothed) landmark anchor and a 0–1 confidence, so the
        caller (:meth:`_track_error`) can **blend** it with the bbox anchor and
        smooth the fused result.  Lazily builds the pose estimator; re-estimates
        keypoints at most every ``_POSE_INTERVAL_S`` for the active target,
        reusing the last keypoints in between.  ``(None, 0.0, 0.0)`` whenever pose
        is unavailable/not confident.  Never raises.
        """
        if not self._feature("pose"):
            return None, 0.0, 0.0
        pose = self._ensure_pose()
        if pose is None or not getattr(pose, "available", False):
            return None, 0.0, 0.0

        try:
            from autoptz.engine.pipeline import framing
        except Exception:  # noqa: BLE001
            return None, 0.0, 0.0

        h, w = frame.shape[:2]
        bb = track.bbox
        bbox = (bb.x1, bb.y1, bb.x2, bb.y2)

        # Drop stale keypoints if the target track changed under us.
        if self._pose_kp_track_id != track.track_id:
            self._pose_keypoints = None
            self._reset_pose_aim()

        # Re-estimate only every _effective_pose_interval(); reuse last keypoints between.
        if (
            self._pose_keypoints is None
            or now - self._last_pose_t >= self._effective_pose_interval()
        ):
            pose_t0 = time.perf_counter()
            kps = pose.estimate(frame, bbox)
            self._pose_ms = (time.perf_counter() - pose_t0) * 1000.0
            self._record_stage("pose", self._pose_ms)
            self._last_pose_t = now
            if kps is not None and self._pose_keypoints_consistent(
                kps,
                track,
                frame,
                now,
                tracks,
            ):
                self._pose_keypoints = kps
                self._pose_kp_track_id = track.track_id
                self._last_pose_overlay_t = now
                self._last_pose_overlay_frame_id = max(1, self._current_inference_frame_id)
            else:
                self._pose_keypoints = None
                self._pose_kp_track_id = None
                self._last_pose_overlay_t = 0.0
                self._last_pose_overlay_frame_id = 0

        kps = self._pose_keypoints
        if not kps:
            return None, 0.0, 0.0

        raw_aim, conf = framing.body_aim_point(
            kps,
            framing=_resolve_framing(self.config.tracking),
        )
        if raw_aim is None:
            return None, 0.0, 0.0

        span = framing.subject_height_from_pose(kps)
        subject_height = (span / h) if (span is not None and h > 0) else 0.0
        return raw_aim, float(conf), subject_height

    def _maybe_estimate_pose_overlay(
        self,
        tracks: list[TrackInfo],
        frame: NDArray[np.uint8] | None,
        now: float,
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
            self._last_pose_overlay_t = 0.0
            self._last_pose_overlay_frame_id = 0
            self._last_pose_emitted_frame_id = -1
            return
        target = self._resolve_target_track(tracks)
        if target is None or target.lost:
            self._reset_pose_aim()
            return
        self._pose_aim(target, frame, now, tracks=tracks)  # side effect: fills _pose_keypoints

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

        # Unified mode: keypoints already came from the detection pass — expose
        # them through the PoseEstimator API so _pose_aim is unchanged, with no
        # second forward pass.
        if self._unified_pose_active:
            try:
                from autoptz.engine.pipeline.pose_detect import UnifiedPoseAdapter

                ep = getattr(self._detect.detector, "ep", "") if self._detect else ""
                self._pose = UnifiedPoseAdapter(lambda: self._last_detections, ep=ep)
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s unified pose adapter failed", self.camera_id, exc_info=True)
                self._pose = None
            return self._pose

        if self._pool is not None:
            try:
                self._pose = self._pool.pose()
            except Exception:  # noqa: BLE001 — pool must never break the worker
                log.debug(
                    "camera_id=%s pool pose() failed; bbox aim only.", self.camera_id, exc_info=True
                )
                self._pose = None
            return self._pose

        if not detection_runtime_available():
            return None
        try:
            from autoptz.engine.pipeline.pose import PoseEstimator

            self._pose = PoseEstimator()
        except Exception:  # noqa: BLE001 — pose must never break the worker
            log.debug(
                "camera_id=%s pose estimator init failed; bbox aim only.",
                self.camera_id,
                exc_info=True,
            )
            self._pose = None
        return self._pose

    def _reset_pose_aim(self) -> None:
        """Clear the pose aim smoother + cached keypoints (on target change)."""
        self._pose_keypoints = None
        self._pose_kp_track_id = None
        self._last_pose_overlay_t = 0.0
        self._last_pose_overlay_frame_id = 0
        self._last_pose_emitted_frame_id = -1
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
            self.camera_id,
            src.type,
            _sanitize_address(src.address),
            float(getattr(src, "fps", 0.0) or 0.0),
            self.shm_name,
        )
        # Crash-safety: the whole worker body runs under a try/finally so an
        # unhandled exception anywhere (including _open_resources) can never kill
        # the capture thread WITHOUT releasing resources — the finally ALWAYS
        # joins the inference thread, runs _close_resources (no leaked cv2
        # capture / shm segment), and emits a STOPPED telemetry so the UI
        # reflects the camera going down instead of a silent hang.
        last_health = HealthState.OK
        last_error: str | None = None
        try:
            self._open_resources()

            # Start the inference thread AFTER the source is open.  It builds the
            # heavy models itself (so a slow first-time EP compile never blocks the
            # live preview) and processes the latest captured frame.
            self._inference_thread = threading.Thread(
                target=self._inference_loop,
                name=f"caminfer-{self.camera_id[:8]}",
                daemon=True,
            )
            self._inference_thread.start()

            last_telemetry = 0.0
            miss_streak = 0  # consecutive no-frame reads, drives the reconnect backoff
            fps_window_start = time.monotonic()
            fps_window_frames = 0
            self._next_drop_log_t = time.monotonic() + _DROP_LOG_INTERVAL_S
            last_health = HealthState.OK if self._source is not None else HealthState.ERROR
            last_error = None if self._source is not None else "frame source unavailable"

            # If the source could not be opened at all, still emit telemetry so the
            # UI shows an error/no-signal state instead of a silent hang.
            if self._source is None:
                self._emit_telemetry(tracks=[], health=last_health, last_error=last_error)

            while not self._stop_event.is_set():
                # Per-iteration guard: a single bad frame/tick logs a throttled
                # WARNING and the loop CONTINUES rather than killing the capture
                # thread.  Only the unrecoverable case (a raise escaping here)
                # falls through to the outer handler + finally cleanup.
                try:
                    # Apply any pending manual PTZ first thing each tick (low-latency
                    # path, independent of inference) so the joystick/D-pad feels
                    # immediate.
                    self._drain_ptz_commands()

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
                                log.warning(
                                    "camera_id=%s frame read failed (%s); reconnecting",
                                    self.camera_id,
                                    exc,
                                )
                            last_health = HealthState.RECONNECTING
                            last_error = str(exc)
                            self._dropped_frames += 1
                            frame = None
                    ingest_ms = (time.perf_counter() - tick_t0) * 1000.0

                    now = time.monotonic()

                    if frame is not None:
                        self._record_stage("ingest", ingest_ms)
                        if last_health is HealthState.RECONNECTING:
                            log.info("camera_id=%s frame source recovered", self.camera_id)
                        self._record_frame_dims(frame)
                        self._push_frame(frame)
                        fps_window_frames += 1
                        miss_streak = 0  # got a frame → drop back to fast retries
                        last_health = HealthState.OK
                        last_error = None

                        # Hand the newest frame to the inference thread (latest wins)
                        # so detection/face never gate the capture+preview rate.
                        with self._frame_lock:
                            self._latest_frame = frame
                            self._latest_frame_id += 1
                        self._frames_captured += 1
                        self._frame_ready.set()

                        self._ingest_ms = ingest_ms
                        # Latency = capture read + the latest inference detect+track.
                        self._latency_ms = ingest_ms + self._inference_ms

                        elapsed = now - fps_window_start
                        if elapsed >= 1.0:
                            self._fps = fps_window_frames / elapsed
                            fps_window_start = now
                            fps_window_frames = 0

                    # Periodic frame-drop summary (INFO) when drops keep accruing.
                    self._maybe_log_drops(now)

                    # Telemetry pacing (~telemetry_hz) — report latest inference.
                    if now - last_telemetry >= self._telemetry_period:
                        self._apply_inference_watchdog(now)
                        self._emit_telemetry(
                            tracks=list(self._last_tracks),
                            health=last_health,
                            last_error=last_error,
                        )
                        last_telemetry = now

                    # No frame: retry fast briefly (transient decode miss on a
                    # healthy source), then back off a sustained no-frame source up
                    # to the cap so a stalled/offline/denied camera doesn't spin the
                    # capture thread.
                    if frame is None:
                        miss_streak += 1
                        if miss_streak <= _RECONNECT_FAST_RETRIES:
                            wait = 0.01
                        else:
                            over = miss_streak - _RECONNECT_FAST_RETRIES
                            wait = min(_RECONNECT_BACKOFF_MAX_S, 0.01 + 0.05 * over)
                        # Interruptible: a manual PTZ command (or stop) wakes this
                        # early so the joystick stays responsive even while the video
                        # feed is stalled (the next loop iteration drains the PTZ
                        # queue at the top).
                        self._capture_wake.wait(timeout=wait)
                        self._capture_wake.clear()
                except Exception as exc:  # noqa: BLE001 — one bad tick must not kill capture
                    now = time.monotonic()
                    if now - self._last_tick_warn_t >= _TICK_WARN_INTERVAL_S:
                        self._last_tick_warn_t = now
                        log.warning(
                            "camera_id=%s capture tick failed (%s); continuing",
                            self.camera_id,
                            exc,
                            exc_info=True,
                        )
                    last_health = HealthState.ERROR
                    last_error = str(exc)
                    # Brief interruptible pause so a persistent fault can't spin.
                    self._capture_wake.wait(timeout=0.05)
                    self._capture_wake.clear()
        except Exception:  # noqa: BLE001 — fatal: log + emit ERROR, finally cleans up
            log.error("camera_id=%s capture thread died", self.camera_id, exc_info=True)
            try:
                self._emit_telemetry(
                    tracks=[], health=HealthState.ERROR, last_error="capture thread error"
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
            # Shutdown: wake + join the inference thread, then release resources.
            # ALWAYS runs — even on a fatal exception above — so the cv2 capture
            # and the shared-memory segment are never leaked.
            self._frame_ready.set()
            if self._inference_thread is not None:
                self._inference_thread.join(timeout=5.0)
                self._inference_thread = None
            self._close_resources()
            log.info(
                "camera_id=%s worker stopped (frames dropped total=%d)",
                self.camera_id,
                self._dropped_frames,
            )
            # Final STOPPED telemetry so the UI reflects the camera going down.
            try:
                self._emit_telemetry(tracks=[], health=HealthState.STOPPED, last_error=None)
            except Exception:  # noqa: BLE001
                pass

    def _inference_loop(self) -> None:
        """Detect → track → face → pose → PTZ on the latest captured frame.

        Runs on its own thread so the capture/preview loop is never blocked by
        model building or per-frame inference.  Builds the heavy models here (off
        the capture critical path); then consumes the freshest frame each pass,
        dropping any intermediate frames the capture thread produced meanwhile.
        """
        while not self._stop_event.is_set() and not self._inference_start.wait(0.05):
            self._drain_commands()
        if self._stop_event.is_set():
            return
        # Crash-safety: everything from the model build onward runs under a
        # try/finally so an unhandled exception (build or a hot-loop stage) can
        # never orphan the appearance thread — the finally ALWAYS joins it.  A
        # per-iteration guard inside the while keeps one raising stage from
        # killing inference (the camera would otherwise go dark with no boxes).
        try:
            self._build_inference_stacks()
            # Seed so the first lifecycle pass sees no spurious transition.
            self._prev_model_features = self._features_snapshot()
            self._start_appearance_thread()
            last_id = 0
            while not self._stop_event.is_set():
                # Wake on a new frame, but fall through on the timeout too so
                # commands (nudge / set-target / enable-tracking) are still drained
                # when no frames are arriving — they must never depend on frame
                # delivery.
                self._frame_ready.wait(timeout=0.05)
                self._frame_ready.clear()
                if self._stop_event.is_set():
                    break
                try:
                    self._drain_commands()
                    self._apply_model_lifecycle()
                    with self._frame_lock:
                        frame = self._latest_frame
                        fid = self._latest_frame_id
                    if frame is None or fid == last_id:
                        continue
                    last_id = fid
                    self._frames_inferred += 1
                    self._current_inference_frame_id = fid
                    now = time.monotonic()
                    self._seed_subservice_phases(now)

                    detect_t0 = time.perf_counter()
                    tracks = self._maybe_track(frame)
                    self._inference_ms = (time.perf_counter() - detect_t0) * 1000.0

                    self._apply_target_lock(tracks, frame, now)

                    if self._async_appearance:
                        # Hand the heavy appearance passes (ReID + face) to their own
                        # thread so they overlap detect+track+PTZ instead of stalling
                        # the control loop.  Shared target/identity state is
                        # serialised by _appearance_lock inside those methods; the
                        # hot loop re-reads the (possibly rebound) target via
                        # _apply_target_lock each tick.  Only publish when there's
                        # actually appearance work — otherwise we wake the appearance
                        # thread every frame for two passes that both immediately
                        # no-op with face + ReID off (pure idle overhead).
                        if self._feature("face_recognition") or self._feature("reid"):
                            self._publish_appearance_input(frame, tracks, now, fid)
                    else:
                        # Inline (sync) path — preserves the exact original ordering.
                        self._maybe_reid_recover(tracks, frame, now)
                        self._apply_target_lock(tracks, frame, now)
                        face_t0 = time.perf_counter()
                        self._maybe_identify(frame, tracks, now)
                        face_dt = (time.perf_counter() - face_t0) * 1000.0
                        if face_dt > 0.5:
                            self._face_ms = face_dt
                            self._record_stage("face", face_dt)
                        self._apply_target_lock(tracks, frame, now)

                    # Estimate pose for the selected person so the pose overlay shows
                    # the moment you click someone — independent of whether PTZ
                    # auto-follow is actively driving (the only place pose ran
                    # before).
                    if self._pose_allowed_this_tick():
                        self._maybe_estimate_pose_overlay(tracks, frame, now)
                    self._annotate_target_aim(tracks, frame, now)

                    # Estimate the camera's own image motion for this tick BEFORE
                    # driving the PTZ — _drive_ptz_auto → _estimate_aim_velocity
                    # subtracts it so the feed-forward follows the subject's world
                    # motion, not the frame shift.  (_ptz_last_cmd still holds the
                    # *previous* command here — exactly the one that produced this
                    # frame's observed shift.)
                    #
                    # Ego-motion runs sparse optical flow on every frame, but its
                    # only consumer is the aim-velocity feed-forward, which only runs
                    # while actively following (tracking on).  With tracking off it
                    # burned ~15% of a core computing a value nothing reads — so gate
                    # it on tracking and keep the estimate zeroed otherwise.
                    if self._feature("tracking"):
                        self._update_ego_motion(tracks, frame, now)
                    elif self._ego_source != "none":
                        self._ego_vel = (0.0, 0.0)
                        self._ego_source = "none"

                    # Auto PTZ control (suspended during a manual-override window).
                    # Lock the backend so a concurrent telemetry position read can't
                    # interleave.
                    with self._ptz_lock:
                        self._drive_ptz_auto(tracks, frame, now)

                    self._last_tracks = tracks
                    self._last_tracks_frame_id = fid
                    # Heartbeat: advance so the capture-thread watchdog knows
                    # inference is alive.  Set AFTER all work for this frame is
                    # committed.
                    self._last_infer_t = time.monotonic()
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(
                            "camera_id=%s timings detect+track=%.1fms face=%.1fms "
                            "tracks=%d fps=%.1f",
                            self.camera_id,
                            self._detect_ms,
                            self._face_ms,
                            len(tracks),
                            self._fps,
                        )
                except Exception as exc:  # noqa: BLE001 — one stage must not kill inference
                    now = time.monotonic()
                    if now - self._last_infer_warn_t >= _INFER_WARN_INTERVAL_S:
                        self._last_infer_warn_t = now
                        log.warning(
                            "camera_id=%s inference tick failed (%s); continuing",
                            self.camera_id,
                            exc,
                            exc_info=True,
                        )
                    self._infer_last_error = str(exc)
        except Exception as exc:  # noqa: BLE001 — fatal: log; finally still joins appearance
            log.error("camera_id=%s inference thread died", self.camera_id, exc_info=True)
            # The inference thread is gone but capture may keep streaming, so the
            # UI would otherwise show a healthy-but-box-blind camera with no error.
            # Pin a terminal error onto telemetry so the still-running capture loop
            # surfaces *why* tracks stopped flowing.
            self._infer_last_error = f"inference thread stopped: {exc}"
        finally:
            # ALWAYS join the appearance thread — even on a fatal error — so it is
            # never orphaned when inference goes down.
            self._stop_appearance_thread()

    # ── async appearance (face + ReID) thread ───────────────────────────────────

    def _start_appearance_thread(self) -> None:
        """Spawn the appearance thread (face + ReID) when async mode is on."""
        if not self._async_appearance or self._appearance_thread is not None:
            return
        self._appearance_thread = threading.Thread(
            target=self._appearance_loop,
            name=f"appearance-{self.camera_id[:8]}",
            daemon=True,
        )
        self._appearance_thread.start()

    def _stop_appearance_thread(self) -> None:
        """Wake + join the appearance thread on shutdown (idempotent)."""
        self._appearance_ready.set()
        thread = self._appearance_thread
        if thread is not None:
            thread.join(timeout=3.0)
            self._appearance_thread = None

    def _publish_appearance_input(
        self,
        frame: NDArray[np.uint8],
        tracks: list[TrackInfo],
        now: float,
        fid: int,
    ) -> None:
        """Hand the latest (frame, tracks, now, id) to the appearance thread.

        A shallow copy of the tracks list is published so the hot loop can keep
        mutating its own list (telemetry annotations) without racing the
        appearance thread; the shared TrackInfo objects are only read there.
        """
        with self._frame_lock:
            self._appearance_input = (frame, list(tracks), now, fid)
        self._appearance_ready.set()

    def _appearance_loop(self) -> None:
        """Run ReID recovery + face identification off the hot inference thread.

        Consumes the freshest published (frame, tracks) — dropping intermediate
        ones like the inference loop — and runs the two heavy, throttled passes.
        All target/identity state they touch is serialised by ``_appearance_lock``
        (acquired inside the methods), so this overlaps the hot loop's detect/track
        without corrupting the target-lock state machine.
        """
        last_fid = -1
        while not self._stop_event.is_set():
            self._appearance_ready.wait(timeout=0.1)
            self._appearance_ready.clear()
            if self._stop_event.is_set():
                break
            with self._frame_lock:
                payload = self._appearance_input
            if payload is None:
                continue
            frame, tracks, now, fid = payload
            if fid == last_fid:
                continue
            last_fid = fid
            try:
                self._maybe_reid_recover(tracks, frame, now)
                t0 = time.perf_counter()
                self._maybe_identify(frame, tracks, now)
                face_dt = (time.perf_counter() - t0) * 1000.0
                if face_dt > 0.5:
                    self._face_ms = face_dt
                    self._record_stage("face", face_dt)
            except Exception:  # noqa: BLE001 — appearance must never crash the worker
                log.debug("camera_id=%s appearance pass failed", self.camera_id, exc_info=True)

    # ── resource management ─────────────────────────────────────────────────────

    def _open_resources(self) -> None:
        # Frame source
        if self._injected_source is not None:
            self._source = self._injected_source
        else:
            try:
                self._source = build_frame_source(self.camera_id, self.config)
            except Exception:  # noqa: BLE001
                log.warning(
                    "camera_id=%s could not build frame source (cv2 missing?)",
                    self.camera_id,
                    exc_info=True,
                )
                self._source = None

        if self._source is not None:
            try:
                if not self._source.open():
                    log.warning("camera_id=%s frame source failed to open", self.camera_id)
                    self._source = None
            except Exception:  # noqa: BLE001
                log.warning("camera_id=%s frame source open raised", self.camera_id, exc_info=True)
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

        Feature-aware: only the subsystems whose global switch is on are built,
        so a worker that starts with (say) detection disabled never loads the
        detector.  ``_apply_model_lifecycle`` keeps presence in sync afterwards.
        """
        if self._feature("detection"):
            self._ensure_detect_stack()
        if self._feature("face_recognition"):
            self._ensure_face_stack()

    def _resolve_detect_stack(self) -> _DetectStack | None:
        """Build a detect stack honouring pool authority.

        When a shared :class:`InferencePool` is injected it is *authoritative*:
        it already enforces the model-cache + auto-download policy
        (``allow_download=False``), so a missing model means live-preview-only —
        we must NOT fall back to the per-worker ``_build_detect_stack`` here,
        because that resolves the model with ``allow_download=True`` and would
        silently download/export a model on the inference thread (ignoring the
        operator's auto-download setting, and stalling startup/toggles).  The
        per-worker build is only for the no-pool path (tests/fakes).
        """
        if self._pool is not None:
            return self._build_detect_stack_pooled()
        self._pooled_detector = False
        return _build_detect_stack(self.config)

    def _ensure_detect_stack(self) -> None:
        """Build the detect stack if not present (graceful: ``None``).

        Prefers the shared inference pool's detector (one ONNX session for all
        cameras) + a fresh PER-WORKER tracker; the per-worker build is only used
        when no pool was injected.  See :meth:`_resolve_detect_stack`.
        """
        if self._detect is not None:
            return
        detect = self._resolve_detect_stack()
        if detect is not None:
            self._ep = detect.ep
        self._detect = detect
        # Detect unified-pose mode by detector type so both the pooled and
        # per-worker build paths are covered uniformly.
        self._unified_pose_active = False
        if detect is not None:
            try:
                from autoptz.engine.pipeline.pose_detect import PoseDetector

                self._unified_pose_active = isinstance(detect.detector, PoseDetector)
            except Exception:  # noqa: BLE001
                self._unified_pose_active = False

    def _ensure_face_stack(self) -> None:
        """Build the face/identity stack if not present (graceful: ``None``)."""
        if self._face is not None:
            return
        if self._injected_face_stack is not None:
            self._face = self._injected_face_stack
            return
        face = self._build_face_stack_pooled()
        if face is None and detection_runtime_available():
            face = _build_face_stack(
                self.config,
                self._injected_identity_service,
            )
        self._face = face

    def _features_snapshot(self) -> dict[str, bool]:
        """Atomic read of all global feature flags (default True when unset)."""
        with self._cmd_lock:
            return {k: bool(self._features.get(k, True)) for k in _DEFAULT_FEATURES}

    def _apply_model_lifecycle(self) -> None:
        """Free / rebuild heavy models to match the global feature switches.

        Acts only on transitions (compared against ``_prev_model_features``):
        turning a subsystem off drops this worker's reference so the model's
        memory is reclaimed once the shared pool releases its copy too (the
        supervisor does that in ``_on_set_features``); turning it back on
        rebuilds.  Pose and ReID rebuild lazily through ``_ensure_pose`` /
        ``_ensure_reid``, so "off" only drops the cached instance and re-arms
        their probe flags.
        """
        cur = self._features_snapshot()
        prev = self._prev_model_features
        if cur == prev:
            return

        if cur["detection"] != prev["detection"]:
            if cur["detection"]:
                self._ensure_detect_stack()
            else:
                self._detect = None
                self._unified_pose_active = False
                self._last_detections = []

        if cur["face_recognition"] != prev["face_recognition"]:
            if cur["face_recognition"]:
                self._ensure_face_stack()
            elif self._injected_face_stack is None:
                self._face = None

        if prev["pose"] and not cur["pose"]:
            self._pose = None
            self._pose_probed = False
            self._pose_keypoints = None
            self._pose_kp_track_id = None

        if prev["reid"] and not cur["reid"]:
            self._reset_reid()
            self._reid = None
            self._reid_probed = False

        self._prev_model_features = cur

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
            log.debug(
                "camera_id=%s pool detector() failed; per-worker fallback.",
                self.camera_id,
                exc_info=True,
            )
            return None
        if detector is None:
            return None
        try:
            from autoptz.engine.pipeline.track import Tracker

            tracker = Tracker(
                tracker_type=self.config.tracking.tracker,
                coast_window=self.config.tracking.coast_window_ms / 1000.0,
            )
            self._pooled_detector = True
            ep = getattr(detector, "ep", "") or getattr(pool, "detector_ep", "")
            _log_detector_ready_once(
                "<shared pool>",
                getattr(detector, "ep", "") or "?",
            )
            return _DetectStack(detector=detector, tracker=tracker, ep=ep)
        except Exception:  # noqa: BLE001
            log.warning(
                "camera_id=%s per-worker tracker init failed; live-preview-only.",
                self.camera_id,
                exc_info=True,
            )
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
            log.debug(
                "camera_id=%s pool face() failed; per-worker fallback.",
                self.camera_id,
                exc_info=True,
            )
            return None
        if recognizer is None:
            return None
        try:
            from autoptz.engine.identity.service import IdentityService

            service = self._injected_identity_service or IdentityService()
            return _FaceStack(recognizer=recognizer, service=service)
        except Exception:  # noqa: BLE001 — gallery build must never break the worker
            log.warning(
                "camera_id=%s identity gallery init failed; identity off.",
                self.camera_id,
                exc_info=True,
            )
            return None

    def _rebuild_ptz_backend(self) -> None:
        """Tear down and rebuild the PTZ controller/backend from the current config.

        Used on a live transport change (e.g. toggling Center Stage flips the
        backend to/from ``digital``). ``_build_ptz_stack`` is a no-op while a
        backend exists, so we clear it first; the rebuild then reads the new
        ``config.ptz`` and builds the right backend (a ``DigitalPTZBackend`` for
        Center Stage). Never raises.
        """
        with self._ptz_lock:
            old = self._ptz_backend
            if old is not None and self._ptz_owned:
                try:
                    old.close()
                except Exception:  # noqa: BLE001
                    log.debug(
                        "camera_id=%s ptz backend close failed", self.camera_id, exc_info=True
                    )
            self._ptz = None
            self._ptz_backend = None
            self._ptz_owned = False
            self._digital_framer = None  # fresh framer for the new backend
            self._build_ptz_stack()
        log.info(
            "camera_id=%s PTZ backend rebuilt for backend=%s",
            self.camera_id,
            getattr(self.config.ptz, "backend", "?"),
        )

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

            # For NDI cameras, hand the factory the source name so it can open a
            # dedicated PTZ receiver (the video adapter keeps its own). Without
            # this, NDI PTZ (auto-follow + manual) silently never builds.
            ndi_name: str | None = None
            src = getattr(self.config, "source", None)
            src_type = getattr(src, "type", "") if src is not None else ""
            if src_type == "ndi":
                addr = (getattr(src, "address", "") or "").strip()
                ndi_name = addr[len("ndi://") :] if addr.startswith("ndi://") else addr

            # Only USB/UVC sources auto-probe serial ports for a VISCA control
            # port — so a network camera's "auto" never grabs an unrelated
            # USB-serial device.
            backend = build_backend(self.config.ptz, ndi_name=ndi_name, is_usb=(src_type == "usb"))
            if backend is None:
                return
            self._ptz_backend = backend
            self._ptz = PTZController(
                backend,
                self.config.ptz,
                coast_window_ms=int(self.config.tracking.coast_window_ms),
            )
            self._ptz_owned = True
            log.info(
                "camera_id=%s PTZ control active (%s)", self.camera_id, self.config.ptz.backend
            )
        except Exception:  # noqa: BLE001 — PTZ must never break the worker
            log.warning(
                "camera_id=%s PTZ stack init failed; PTZ disabled.", self.camera_id, exc_info=True
            )
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
            log.warning(
                "camera_id=%s reclaiming stale shm segment %s", self.camera_id, self.shm_name
            )
            self._unlink_stale_shm()
            try:
                return ShmWriter(self.shm_name, _PREVIEW_H, _PREVIEW_W)
            except Exception:  # noqa: BLE001
                log.warning(
                    "camera_id=%s could not create ShmWriter %s after reclaim",
                    self.camera_id,
                    self.shm_name,
                    exc_info=True,
                )
                return None
        except Exception:  # noqa: BLE001
            log.warning(
                "camera_id=%s could not create ShmWriter %s",
                self.camera_id,
                self.shm_name,
                exc_info=True,
            )
            return None

    def _unlink_stale_shm(self) -> None:
        from autoptz.engine.runtime.shm import unlink_shared_memory_pair

        try:
            unlink_shared_memory_pair(self.shm_name)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s could not unlink stale shm", self.camera_id, exc_info=True)

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
                "camera_id=%s dropped %d frame(s) in the last %.0fs (total=%d, fps=%.1f)",
                self.camera_id,
                delta,
                _DROP_LOG_INTERVAL_S,
                self._dropped_frames,
                self._fps,
            )

        # Inference-stage backpressure: how many captured frames the inference
        # thread couldn't keep up with this window.  A high ratio means the
        # detector cadence (quality policy) should relax or the model is too heavy.
        cap_delta = self._frames_captured - self._last_logged_inf_captured
        inf_delta = self._frames_inferred - self._last_logged_inf_inferred
        self._last_logged_inf_captured = self._frames_captured
        self._last_logged_inf_inferred = self._frames_inferred
        skipped = cap_delta - inf_delta
        if cap_delta > 0 and skipped > 0:
            ratio = skipped / cap_delta
            if ratio >= 0.5:  # only shout when the inference thread is well behind
                log.info(
                    "camera_id=%s inference behind: processed %d/%d frames (%.0f%% skipped) "
                    "in the last %.0fs; cadence=%s",
                    self.camera_id,
                    inf_delta,
                    cap_delta,
                    ratio * 100.0,
                    _DROP_LOG_INTERVAL_S,
                    self._quality_active,
                )

    def _framed_output(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Center Stage: crop+scale the frame to auto-frame the target.

        The crop is computed directly from the current target's bounding box by
        :class:`DigitalFramer` (centre on the subject, size to frame it, smoothed)
        — NOT by integrating the velocity controller, whose conservative auto-zoom
        never engaged so the crop stayed full-frame. With no target the crop eases
        back to the whole frame. Returns the frame unchanged when no digital
        backend is active, so the caller is always safe to use the return value.
        """
        from autoptz.engine.ptz.digital import DigitalPTZBackend

        backend = self._ptz_backend
        if not isinstance(backend, DigitalPTZBackend) or frame is None:
            return frame
        import cv2

        h, w = frame.shape[:2]
        ow = int(getattr(self.config.ptz, "digital_output_w", 1280))
        oh = int(getattr(self.config.ptz, "digital_output_h", 720))
        aspect = ow / max(1, oh)
        framer = self._digital_framer
        if framer is None or abs(framer.out_aspect - aspect) > 1e-3:
            from autoptz.engine.pipeline.digital_framer import DigitalFramer

            framer = self._digital_framer = DigitalFramer(out_aspect=aspect)
        # Crop tightness follows the live "Framing" dropdown.
        framing = getattr(self.config.tracking, "framing", "upper_body")
        framer.fill, framer.max_frac = _CENTERSTAGE_FRAMING.get(
            framing, _CENTERSTAGE_FRAMING["upper_body"]
        )
        target = self._current_digital_target()
        if target is not None:
            x, y, cw, ch = framer.frame_for(target, w, h)
        else:
            x, y, cw, ch = framer.full_frame(w, h)
        nowm = time.monotonic()
        if nowm - self._cs_diag_t > 2.0:
            self._cs_diag_t = nowm
            log.info(
                "camera_id=%s center-stage: target=%s crop=%dx%d of %dx%d (tid=%s)",
                self.camera_id,
                "yes" if target is not None else "NONE",
                cw,
                ch,
                w,
                h,
                self._target_track_id,
            )
        crop = frame[y : y + ch, x : x + cw]
        if crop.size == 0:
            return frame
        return cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)

    def _current_digital_target(self) -> tuple[float, float, float, float] | None:
        """The selected target's bbox (x1,y1,x2,y2) for Center Stage, or None.

        Runs on the capture thread; a slightly stale box is fine for smooth
        framing. Prefers the live track for the current target id, but falls back
        to the maintained *trusted* target box so Center Stage keeps framing
        through track-id churn / identity re-binding (when ``_target_track_id``
        momentarily points at a track not in the latest ``_last_tracks``).
        """
        tid = self._target_track_id
        if tid is not None:
            for t in self._last_tracks or ():
                if (
                    t.track_id == tid
                    and not getattr(t, "lost", False)
                    and getattr(t, "bbox", None) is not None
                ):
                    bb = t.bbox
                    return (bb.x1, bb.y1, bb.x2, bb.y2)
        # Fallback: the last trusted target box (set whenever a target is locked,
        # by track id OR by identity), so the crop holds through brief track gaps.
        if self._target_track_id is not None or self._target_identity_id is not None:
            tb = getattr(self._target_lock, "trusted_bbox", None)
            if tb is not None:
                return (tb.x1, tb.y1, tb.x2, tb.y2)
        return None

    def _push_frame(self, frame: NDArray[np.uint8]) -> None:
        if self._shm is None:
            return
        # Cap the preview rate: skip the resize/push when the last preview frame
        # was pushed less than one preview period ago (saves CPU on >20fps sources
        # without affecting tracking, which uses the full-rate frame elsewhere).
        now = time.monotonic()
        if now - self._last_preview_push_t < _PREVIEW_PUSH_MIN_PERIOD_S:
            return
        self._last_preview_push_t = now
        try:
            framed = self._framed_output(frame)
            f = self._fit_frame(framed)
            self._shm.push(f)
            if getattr(self.config.ptz, "vcam_out", False):
                ow = int(getattr(self.config.ptz, "digital_output_w", 1280))
                oh = int(getattr(self.config.ptz, "digital_output_h", 720))
                if self._vcam is None:
                    from autoptz.engine.pipeline.vcam import VirtualCamSink

                    self._vcam = VirtualCamSink(ow, oh)
                self._vcam.send_bgr(framed)
            elif self._vcam is not None:
                # Virtual camera output was turned off — release the device so it
                # disconnects from Zoom/OBS instead of lingering on a frozen frame.
                self._vcam.close()
                self._vcam = None
        except Exception:  # noqa: BLE001
            # Was DEBUG-only: a broken preview pipe (shm/vcam) left the operator
            # staring at a frozen preview with an empty log.  Surface it as a
            # throttled WARNING so a persistent failure is visible without
            # spamming one line per pushed frame.
            if now - self._last_shm_push_warn_t >= _TICK_WARN_INTERVAL_S:
                self._last_shm_push_warn_t = now
                log.warning("camera_id=%s preview/shm push failed", self.camera_id, exc_info=True)

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
        # Intentional: no inference ran this tick, so the stale _detected_this_tick flag is
        # harmless — there are no detections to pose anyway.
        if frame is None or self._detect is None:
            return []
        if not self._feature("detection"):
            return []
        try:
            _t0 = time.perf_counter()
            self._detect_frame_index += 1
            interval = self._effective_detect_interval()
            should_detect = (
                not self._pooled_detector or (self._detect_frame_index - 1) % interval == 0
            )
            if should_detect:
                self._detected_this_tick = True
                detections = self._detect.detector.detect(frame)
                detections = self._filter_small_detections(detections, frame)
                self._last_detections = detections
            else:
                self._detected_this_tick = False
                # On detector-skip frames re-feed the previous detections so the
                # boxmot tracker keeps the person alive between detect frames.
                # Feeding [] here ages tracks out within a frame or two, which is
                # why quality floor "low"/"balanced" (interval 3/2) looked like
                # "no detection no matter how close".
                detections = self._last_detections
            _t1 = time.perf_counter()
            tracks = self._detect.tracker.update(detections, frame, fps=max(1.0, self._fps))
            _t2 = time.perf_counter()
            self._detect_ms = (_t1 - _t0) * 1000.0
            self._track_ms = (_t2 - _t1) * 1000.0
            self._record_stage("detect", self._detect_ms)
            self._record_stage("track", self._track_ms)
            # Detect/track succeeded this tick — clear any stale inference error so
            # a single transient failure does not pin last_error on every later
            # healthy telemetry (the camera would otherwise look permanently
            # faulted long after it recovered).
            self._infer_last_error = None
        except Exception as exc:  # noqa: BLE001
            # Surface the FIRST detect/track failure at WARNING (then throttle) and
            # stash it as last_error so the Camera Info panel can show *why* the
            # camera went box-blind instead of leaving an empty DEBUG-only log.
            now = time.monotonic()
            if now - self._last_infer_warn_t >= _INFER_WARN_INTERVAL_S:
                self._last_infer_warn_t = now
                log.warning("camera_id=%s detect/track failed", self.camera_id, exc_info=True)
            self._infer_last_error = str(exc)
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
            out.append(
                TrackInfo(
                    track_id=t.track_id,
                    bbox=BBox(x1=t.bbox.x1, y1=t.bbox.y1, x2=t.bbox.x2, y2=t.bbox.y2),
                    identity=(ident[1] if ident else None),  # NAME, for display
                    identity_id=(ident[0] if ident else None),  # id, for enroll/target
                    confidence=(ident[2] if ident else t.conf),
                    is_target=(
                        self._target_track_id is not None and t.track_id == self._target_track_id
                    ),
                    lost=False,
                    vx=float(vel[0]),
                    vy=float(vel[1]),
                )
            )
        return out

    def _filter_small_detections(
        self,
        detections: list[Any],
        frame: NDArray[np.uint8],
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
            log.debug(
                "camera_id=%s dropped %d small detection(s) (< %.0fpx)",
                self.camera_id,
                len(detections) - len(kept),
                min_px,
            )
        return kept

    # ── face recognition + identity ───────────────────────────────────────────

    def _clear_face_overlay(self) -> None:
        """Drop face overlay state immediately."""
        self._last_faces = []
        self._last_faces_t = 0.0
        self._last_face_track_ids = set()
        self._last_faces_frame_id = 0
        self._last_faces_emitted_frame_id = -1

    def _expire_face_overlay(
        self,
        now: float,
        tracks: list[TrackInfo] | None = None,
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
        tracks_frame_id = self._last_tracks_frame_id or self._last_faces_frame_id
        if self._last_faces_frame_id != tracks_frame_id:
            self._clear_face_overlay()
            return []
        if self._last_faces_emitted_frame_id == self._last_faces_frame_id:
            return []
        self._last_faces_emitted_frame_id = self._last_faces_frame_id
        return list(self._last_faces)

    @_appearance_guarded
    def _maybe_identify(
        self,
        frame: NDArray[np.uint8] | None,
        tracks: list[TrackInfo],
        now: float,
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
        if frame is None or face is None or not getattr(face.recognizer, "available", False):
            self._clear_face_overlay()
            return
        if now - self._last_face_t < self._effective_face_interval():
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
                        iid,
                        obs.embedding,
                        thumbnail=_face_crop_png(frame, obs.bbox),
                    )
                except Exception:  # noqa: BLE001
                    log.debug(
                        "camera_id=%s enroll_track add_embedding failed",
                        self.camera_id,
                        exc_info=True,
                    )
                self._track_identity[track.track_id] = (iid, name, 1.0)
                matched_identity_track[iid] = track.track_id
                rec = face.service.get(iid) if hasattr(face.service, "get") else None
                if rec is not None:
                    self._push_identity(rec)
                log.info(
                    "camera_id=%s enrolled track=%d → id=%s name=%s",
                    self.camera_id,
                    track.track_id,
                    iid,
                    name,
                )
                continue
            match = None
            try:
                # include_disabled=True so an already-harvested (disabled)
                # "Person N" is recognised and re-bound instead of being
                # re-harvested as a duplicate — this is the dedup that turns
                # "16 faces for one person" into one.
                match = face.recognizer.match(
                    obs.embedding,
                    face.service,
                    include_disabled=True,
                )
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s face match failed", self.camera_id, exc_info=True)
            if match is not None:
                strong_bind = match.score >= _FACE_TARGET_MATCH_THRESHOLD
                if track is not None and strong_bind:
                    prev = self._track_identity.get(track.track_id)
                    self._track_identity[track.track_id] = (
                        match.identity_id,
                        match.name,
                        match.score,
                    )
                    matched_identity_track[match.identity_id] = track.track_id
                    # Log only on a new track→identity binding to avoid per-tick
                    # spam while the same person stays in frame.
                    if prev is None or prev[0] != match.identity_id:
                        log.info(
                            "camera_id=%s identity match track=%d id=%s name=%s score=%.2f",
                            self.camera_id,
                            track.track_id,
                            match.identity_id,
                            match.name,
                            match.score,
                        )
                elif track is not None:
                    log.debug(
                        "camera_id=%s weak face match ignored for track bind "
                        "track=%d id=%s score=%.2f threshold=%.2f",
                        self.camera_id,
                        track.track_id,
                        match.identity_id,
                        match.score,
                        _FACE_TARGET_MATCH_THRESHOLD,
                    )
                # Only teach the gallery from high-confidence matches.  We still
                # treat weaker matches as "not novel" so they do not auto-harvest
                # duplicates, but we do not append their embedding to anyone.
                if match.score >= _FACE_LEARN_MATCH_THRESHOLD:
                    crop: bytes | None = None
                    if now - self._last_crop_t >= _HARVEST_COOLDOWN_S:
                        crop = _face_crop_png(frame, obs.bbox)
                        self._last_crop_t = now
                    try:
                        face.service.add_embedding(
                            match.identity_id,
                            obs.embedding,
                            thumbnail=crop,
                        )
                    except Exception:  # noqa: BLE001
                        log.debug(
                            "camera_id=%s add_embedding failed",
                            self.camera_id,
                            exc_info=True,
                        )
                else:
                    log.debug(
                        "camera_id=%s weak face match not learned id=%s score=%.2f threshold=%.2f",
                        self.camera_id,
                        match.identity_id,
                        match.score,
                        _FACE_LEARN_MATCH_THRESHOLD,
                    )
            elif track is not None:
                self._maybe_harvest(face, frame, obs, track, now)

        # Resolve identity-targeting ("track when found"): if the operator chose
        # an identity and it is now bound to a live track, lock the single target
        # onto that track.
        if self._target_identity_id is not None:
            tid = matched_identity_track.get(self._target_identity_id)
            if tid is not None and tid != self._target_track_id:
                if not self._identity_target_confirmed(self._target_identity_id, tid):
                    self._target_lock.status = "pending"
                    self._target_lock.reason = "identity_confirm"
                else:
                    log.info(
                        "camera_id=%s identity-target id=%s acquired on track=%d",
                        self.camera_id,
                        self._target_identity_id,
                        tid,
                    )
                    self._commit_target_track(tid, reason="identity")
            elif tid is not None:
                self._identity_recover_candidate = None

        # Prune stale cache entries for tracks no longer present.
        live_ids = {t.track_id for t in tracks}
        self._track_identity = {k: v for k, v in self._track_identity.items() if k in live_ids}
        self._pending_enroll = {k: v for k, v in self._pending_enroll.items() if k in live_ids}

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
            faces_out.append(
                FaceBox(
                    bbox=BBox(x1=float(bx[0]), y1=float(bx[1]), x2=float(bx[2]), y2=float(bx[3])),
                    identity=(ident[1] if ident else None),
                    score=(ident[2] if ident else 0.0),
                )
            )
        self._last_faces = faces_out
        self._last_faces_t = now if faces_out else 0.0
        self._last_face_track_ids = face_track_ids
        self._last_faces_frame_id = max(1, self._current_inference_frame_id) if faces_out else 0

    def _identity_target_confirmed(self, identity_id: str, track_id: int) -> bool:
        prev_iid, prev_tid, count = self._identity_recover_candidate or ("", -1, 0)
        count = count + 1 if prev_iid == identity_id and prev_tid == track_id else 1
        self._identity_recover_candidate = (identity_id, track_id, count)
        return count >= _IDENTITY_TARGET_CONFIRM

    @staticmethod
    def _track_for_face(
        obs: Any,
        tracks: list[TrackInfo],
    ) -> TrackInfo | None:
        """Return the track whose bbox contains the face centre (closest wins)."""
        best: TrackInfo | None = None
        best_d = float("inf")
        for t in tracks:
            if getattr(t, "lost", False):
                continue
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
        target_track = next((t for t in tracks if t.track_id == track_id), None)
        candidates = []
        for obs in observations:
            tr = self._track_for_face(obs, tracks)
            if tr is not None and tr.track_id == track_id and not getattr(tr, "lost", False):
                candidates.append(obs)
                continue
            if click_norm is None or target_track is None or getattr(target_track, "lost", False):
                continue
            h, w = frame.shape[:2]
            px = click_norm[0] * max(1, w)
            py = click_norm[1] * max(1, h)
            x1, y1, x2, y2 = obs.bbox
            if (
                target_track.bbox.x1 <= px <= target_track.bbox.x2
                and target_track.bbox.y1 <= py <= target_track.bbox.y2
                and x1 <= px <= x2
                and y1 <= py <= y2
            ):
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
            log.debug("camera_id=%s add_unlabeled failed", self.camera_id, exc_info=True)
            return
        # Bind the freshly-harvested identity to its track so telemetry shows it.
        self._track_identity[track.track_id] = (rec.id, rec.name, obs.det_score)
        log.info(
            "camera_id=%s harvested unlabeled identity id=%s from track=%d (det_score=%.2f)",
            self.camera_id,
            rec.id,
            track.track_id,
            obs.det_score,
        )
        self._push_identity(rec)

    def _harvest_quality_ok(
        self,
        face: Any,
        frame: NDArray[np.uint8],
        obs: Any,
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
            log.debug(
                "camera_id=%s harvest skip: low det_score=%.2f",
                self.camera_id,
                float(getattr(obs, "det_score", 0.0)),
            )
            return False

        yaw = obs.yaw_ratio() if hasattr(obs, "yaw_ratio") else None
        if yaw is not None and yaw > _MAX_HARVEST_YAW_RATIO:
            log.debug("camera_id=%s harvest skip: non-frontal yaw_ratio=%.2f", self.camera_id, yaw)
            return False

        # Novelty: only create a NEW identity when the face is confidently NOT in
        # the gallery already (low best similarity over labeled + unlabeled).
        if self._best_gallery_similarity(face, obs) > _HARVEST_NOVELTY_MAX_SIM:
            log.debug("camera_id=%s harvest skip: already similar to gallery", self.camera_id)
            return False

        sharp = self._face_sharpness(frame, obs.bbox)
        if sharp is not None and sharp < _MIN_HARVEST_SHARPNESS:
            log.debug("camera_id=%s harvest skip: blurry crop (var=%.1f)", self.camera_id, sharp)
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
            log.debug("camera_id=%s gallery-novelty check failed", self.camera_id, exc_info=True)
            return 0.0

    def _face_sharpness(
        self,
        frame: NDArray[np.uint8],
        bbox: tuple[float, float, float, float],
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
            log.debug("camera_id=%s identity callback raised", self.camera_id, exc_info=True)

    def _add_event(self, kind: str, message: str, *, level: str = "info") -> None:
        """Append a recent runtime event for diagnostics/history."""
        self._runtime_events.append(
            RuntimeEventInfo(
                kind=str(kind or "runtime"),
                level=str(level or "info"),
                message=str(message),
            )
        )

    def _record_stage(self, key: str, ms: float) -> None:
        """Remember a stage measurement for last/avg/p95 diagnostics."""
        if ms < 0:
            return
        with self._stage_lock:
            samples = self._stage_samples.setdefault(key, deque(maxlen=_STAGE_WINDOW))
            samples.append(float(ms))
            self._stage_last_t[key] = time.monotonic()

    def _target_fps(self) -> float:
        return max(1.0, float(getattr(self.config.source, "fps", 0.0) or 0.0))

    def _frame_budget_ms(self) -> float:
        return 1000.0 / self._target_fps()

    def _stage_avg(self, key: str) -> float:
        with self._stage_lock:
            samples = self._stage_samples.get(key)
            if not samples:
                return 0.0
            snapshot = list(samples)
        return float(sum(snapshot) / len(snapshot))

    # ── adaptive subservice cadence (CPU governor) ──────────────────────────────

    def _quality_scale(self) -> float:
        """Multiplier for subservice intervals based on active quality tier.

        ``high`` / ``auto`` (stable headroom) → 1.0 (run at the base cadence)
        ``balanced`` (near budget) → 1.25 (modest relaxation under moderate load)
        ``low`` (over budget) → 2.0 (run at half speed when the machine is over budget)

        Subservices run at FULL cadence unless the machine is near or over its
        frame budget.  The phase-stagger still smooths burst spikes at all load
        levels, so there is no need to throttle proactively at idle.
        """
        q = self._quality_active
        if q == "low":
            return 2.0
        if q == "balanced":
            return 1.25
        return 1.0  # "high", "auto", or any unrecognised value → full cadence

    def _effective_face_interval(self) -> float:
        """Face-recognition run period after quality scaling (seconds)."""
        return _FACE_INTERVAL_S * self._quality_scale()

    def _effective_reid_interval(self) -> float:
        """ReID run period after quality scaling (seconds)."""
        return _REID_INTERVAL_S * self._quality_scale()

    def _effective_pose_interval(self) -> float:
        """Pose-estimation run period after quality scaling (seconds)."""
        return _POSE_INTERVAL_S * self._quality_scale()

    def _seed_subservice_phases(self, now: float) -> None:
        """One-time phase-stagger: offset ReID and pose from face so they don't
        all fire on the same appearance-thread tick.

        Called from the inference loop the first time ``now`` is known; guarded
        by ``_phase_seeded`` so it is a no-op on every subsequent call.
        """
        if self._phase_seeded:
            return
        # Offset ReID by half a period and pose by a third so the three heavy
        # appearance passes are evenly distributed across the cadence window
        # instead of all firing at t=0.
        self._last_reid_t = now - _REID_INTERVAL_S * 0.5
        self._last_pose_t = now - _POSE_INTERVAL_S * 0.33
        self._phase_seeded = True

    def _pose_allowed_this_tick(self) -> bool:
        """False when spreading is on and the detector ran this frame (defer pose)."""
        if not getattr(self.config.tracking, "stage_spread", True):
            return True
        return not self._detected_this_tick

    def _amortized_cost_ms(self) -> float:
        """Per-displayed-frame work: each stage scaled by how often it runs.

        Detector runs every ``_quality_interval`` frames; face/pose run on a wall
        clock (``_FACE_INTERVAL_S`` / ``_POSE_INTERVAL_S``) so per-frame they cost
        proportionally less the higher the source fps.
        """
        fps = max(1.0, self._target_fps())
        detect = self._stage_avg("detect") / max(1, self._quality_interval)
        track = self._stage_avg("track")
        face = self._stage_avg("face") * min(1.0, (1.0 / _FACE_INTERVAL_S) / fps)
        pose = self._stage_avg("pose") * min(1.0, (1.0 / _POSE_INTERVAL_S) / fps)
        return detect + track + face + pose

    def _effective_detect_interval(self) -> int:
        """Return the actual detector cadence after quality policy is applied."""
        base = max(1, int(getattr(self.config.tracking, "detect_interval", 1) or 1))
        floor = str(getattr(self.config.tracking, "quality_floor", "auto") or "auto")
        if floor == "low":
            self._quality_active = "low"
            self._quality_reason = "Manual low quality floor: relaxed detector cadence."
            self._quality_interval = max(base, 3)
            return self._quality_interval
        if floor == "balanced":
            self._quality_active = "balanced"
            self._quality_reason = "Manual balanced quality floor."
            self._quality_interval = max(base, 2)
            return self._quality_interval
        if floor == "high":
            self._quality_active = "high"
            self._quality_reason = "Manual high quality floor."
            self._quality_interval = base
            return self._quality_interval

        budget = self._frame_budget_ms()
        cost = self._amortized_cost_ms()
        ratio = cost / budget if budget > 0 else 0.0
        if self._system_cpu_pressure >= 85.0:
            ratio = max(ratio, 0.92)  # Machine is hot → push toward balanced/low.
        prev = self._quality_active
        if ratio >= 0.90 or (prev in ("balanced", "low") and ratio >= 0.70):
            if ratio >= 1.10:
                interval, active = max(base, 4), "low"
                reason = "Auto quality: over frame budget; detector cadence relaxed."
            else:
                interval, active = max(base, 2), "balanced"
                reason = "Auto quality: near frame budget; detector cadence balanced."
        else:
            interval, active = base, "high"
            reason = "Auto quality: latency headroom stable; full cadence."
        if interval != self._quality_interval or active != self._quality_active:
            self._add_event("quality", reason)
        self._quality_interval = interval
        self._quality_active = active
        self._quality_reason = reason
        return interval

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
        tracks_frame_id = self._last_tracks_frame_id or self._last_pose_overlay_frame_id
        if self._last_pose_overlay_frame_id != tracks_frame_id:
            return []
        if self._last_pose_emitted_frame_id == self._last_pose_overlay_frame_id:
            return []
        if time.monotonic() - self._last_pose_overlay_t > _POSE_TTL_S:
            return []
        self._last_pose_emitted_frame_id = self._last_pose_overlay_frame_id
        return [PoseKeypoint(x=float(k.x), y=float(k.y), conf=float(k.conf)) for k in kps]

    def _ptz_status_snapshot(self, now: float) -> dict[str, float | str]:
        ctrl = self._ptz
        if ctrl is None or not hasattr(ctrl, "status_snapshot"):
            return {
                "state": "idle",
                "action": "",
                "coast_remaining_s": 0.0,
                "search_remaining_s": 0.0,
            }
        try:
            return ctrl.status_snapshot(now)
        except Exception:  # noqa: BLE001
            return {
                "state": "idle",
                "action": "",
                "coast_remaining_s": 0.0,
                "search_remaining_s": 0.0,
            }

    def _tracking_status_info(
        self,
        tracks: list[TrackInfo],
        now: float,
    ) -> TrackingStatusInfo:
        """Human-readable explanation of target lock and PTZ recovery behavior."""
        if self._target_track_id is None and self._target_identity_id is None:
            return TrackingStatusInfo()

        target = self._resolve_target_track(tracks)
        label = self._target_label(target)

        if self._manual_override_active(now):
            remaining = max(0.0, self._manual_override_until - now)
            return TrackingStatusInfo(
                state="manual",
                headline="Manual control - auto paused",
                detail=f"Auto tracking resumes in {remaining:.1f}s.",
                action="manual",
                remaining_s=remaining,
                severity="warning",
            )

        if self._watchdog_stalled:
            return TrackingStatusInfo(
                state="degraded",
                headline="Inference stalled",
                detail="inference stalled — holding",
                action="holding",
                severity="warning",
            )

        snap = self._ptz_status_snapshot(now)
        ptz_state = str(snap.get("state", "idle"))
        ptz_action = str(snap.get("action", ""))
        coast_remaining = float(snap.get("coast_remaining_s", 0.0) or 0.0)
        search_remaining = float(snap.get("search_remaining_s", 0.0) or 0.0)

        if self._target_lock.status == "ambiguous":
            remaining = max(0.0, self._target_lock.ambiguous_until - now, coast_remaining)
            return TrackingStatusInfo(
                state="ambiguous",
                headline="Target blocked",
                detail=(
                    f"Holding {label}'s last trusted position"
                    + (f" for {remaining:.1f}s." if remaining > 0.0 else ".")
                ),
                action="holding",
                remaining_s=remaining,
                severity="warning",
            )

        if target is not None and not target.lost:
            if self._tracking_enabled and self._feature("tracking"):
                return TrackingStatusInfo(
                    state="locked",
                    headline=f"Tracking {label}",
                    detail="Camera is following the confirmed target.",
                    action="tracking",
                    severity="ok",
                )
            return TrackingStatusInfo(
                state="locked",
                headline=f"Target selected: {label}",
                detail="Auto tracking is paused.",
                action="paused",
                severity="info",
            )

        if ptz_state == "coasting" or self._target_lock.status == "coasting":
            remaining = max(
                coast_remaining, max(0.0, _TARGET_HOLD_S - (now - self._target_lock.trusted_t))
            )
            return TrackingStatusInfo(
                state="coasting",
                headline="Target lost - holding position",
                detail=(
                    f"Keeping the last trusted framing for {remaining:.1f}s while "
                    "checking for the same person."
                ),
                action="holding",
                remaining_s=remaining,
                severity="warning",
            )

        if ptz_state == "searching" and ptz_action == "zooming_out":
            return TrackingStatusInfo(
                state="searching",
                headline="Searching - zooming out",
                detail=f"Widening the shot for {search_remaining:.1f}s to reacquire {label}.",
                action="zooming_out",
                remaining_s=search_remaining,
                severity="warning",
            )

        if self._target_lock.status == "pending":
            return TrackingStatusInfo(
                state="standby",
                headline="Confirming target",
                detail="Waiting for repeated identity confirmation before following.",
                action="confirming",
                severity="info",
            )

        return TrackingStatusInfo(
            state="standby",
            headline="Standing by for reacquire",
            detail=f"Waiting to confirm {label} before moving the camera.",
            action="standby",
            severity="info",
        )

    def _stage_status(self, key: str, *, enabled: bool, available: bool = True) -> str:
        if not enabled:
            return "disabled"
        if not available:
            return "idle"
        samples = self._stage_samples.get(key)
        if not samples:
            return "warming"
        age = time.monotonic() - self._stage_last_t.get(key, 0.0)
        return "active" if age <= _STAGE_FRESH_S else "stale"

    def _stage_row(
        self,
        key: str,
        name: str,
        *,
        status: str,
        cadence: str = "",
        detail: str = "",
    ) -> StageTimingInfo:
        with self._stage_lock:
            samples = list(self._stage_samples.get(key, ()))
        last = float(samples[-1]) if samples else 0.0
        avg = float(sum(samples) / len(samples)) if samples else 0.0
        if samples:
            ordered = sorted(samples)
            p95 = float(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))])
        else:
            p95 = 0.0
        last_t = self._stage_last_t.get(key, 0.0)
        age_ms = (time.monotonic() - last_t) * 1000.0 if last_t else 0.0
        budget = self._frame_budget_ms()
        return StageTimingInfo(
            key=key,
            name=name,
            status=status,
            last_ms=last,
            avg_ms=avg,
            p95_ms=p95,
            cadence=cadence,
            fresh=bool(last_t and status == "active"),
            age_ms=age_ms,
            budget_pct=(avg / budget * 100.0) if budget > 0 else 0.0,
            detail=detail,
        )

    def _stage_timings(self) -> list[StageTimingInfo]:
        detection_on = self._feature("detection")
        tracking_on = self._feature("tracking")
        face_on = self._feature("face_recognition")
        pose_on = self._feature("pose")
        detect_interval = self._effective_detect_interval()
        return [
            self._stage_row(
                "ingest",
                "Ingest",
                status=self._stage_status(
                    "ingest", enabled=True, available=self._source is not None
                ),
                cadence="every frame",
            ),
            self._stage_row(
                "detect",
                "Detector",
                status=self._stage_status(
                    "detect", enabled=detection_on, available=self._detect is not None
                ),
                cadence=f"every {detect_interval} frame{'s' if detect_interval != 1 else ''}",
            ),
            self._stage_row(
                "track",
                "Tracker",
                status=self._stage_status(
                    "track",
                    enabled=detection_on and tracking_on,
                    available=self._detect is not None,
                ),
                cadence="every inference frame",
            ),
            self._stage_row(
                "face",
                "Face",
                status=self._stage_status(
                    "face",
                    enabled=face_on,
                    available=self._face is not None,
                ),
                cadence=f"{1.0 / _FACE_INTERVAL_S:.0f} Hz",
            ),
            self._stage_row(
                "pose",
                "Pose",
                status=self._stage_status(
                    "pose",
                    enabled=pose_on,
                    available=self._pose is not None or not self._pose_probed,
                ),
                cadence=f"{1.0 / _POSE_INTERVAL_S:.0f} Hz target-only",
            ),
        ]

    def _runtime_services(self) -> list[RuntimeServiceInfo]:
        pool = self._pool
        detector_model = (
            str(getattr(pool, "detector_model_name", "") or "") if pool is not None else ""
        )
        detector_tier = str(getattr(pool, "detector_tier", "") or "") if pool is not None else ""
        detector_ep = self._ep or (
            str(getattr(pool, "detector_ep", "") or "") if pool is not None else ""
        )
        detector_error = str(getattr(pool, "detector_error", "") or "") if pool is not None else ""
        # Read cached acceleration verdict (never triggers a measurement here —
        # that runs once on a daemon thread from warmup_detector).
        _no_accel: Any = lambda: ""  # noqa: E731
        accel_summary = (
            str(getattr(pool, "detector_accel_summary", _no_accel)() or "")
            if pool is not None
            else ""
        )
        accel_verdict = (
            str(getattr(pool, "detector_accel_verdict", _no_accel)() or "")
            if pool is not None
            else ""
        )
        cap = self._source_fps_cap()
        target_fps = self._target_fps()
        tracking = self.config.tracking
        ignore_arms = getattr(tracking, "aim_body_mode", "torso") == "torso"
        # When detection is on but no detector built and the pool knows why, show
        # "failed" + the reason instead of a generic disabled/idle state — so a
        # silent model fall-back doesn't read as "the tier doesn't exist".
        det_enabled = self._feature("detection")
        det_available = self._detect is not None
        det_failed = det_enabled and not det_available and bool(detector_error)
        # Base detail for the detector row; append the accel summary when available.
        if det_failed:
            det_detail = detector_error
        else:
            det_detail = detector_model or "per-camera detector"
            if accel_summary:
                det_detail = f"{det_detail} · {accel_summary}"
        return [
            RuntimeServiceInfo(
                key="detector",
                name="Detector",
                scope="global" if pool is not None else "camera",
                configured=str(getattr(tracking, "quality_floor", "auto")),
                enabled=det_enabled,
                active=bool(det_enabled and det_available),
                state="failed"
                if det_failed
                else self._stage_status("detect", enabled=det_enabled, available=det_available),
                detail=det_detail,
                model=detector_model,
                tier=detector_tier,
                ep=detector_ep,
                confidence=accel_verdict,
            ),
            RuntimeServiceInfo(
                key="tracker",
                name="Tracker",
                configured=str(getattr(tracking, "tracker", "")),
                enabled=self._feature("tracking"),
                active=bool(self._feature("tracking") and self._detect is not None),
                state=self._stage_status(
                    "track", enabled=self._feature("tracking"), available=self._detect is not None
                ),
                backend=str(getattr(tracking, "tracker", "")),
                detail="manual targets clear on rebuild; identity targets reacquire",
            ),
            self._reid_service_row(tracking),
            RuntimeServiceInfo(
                key="face",
                name="Face",
                configured="on" if self._feature("face_recognition") else "off",
                enabled=self._feature("face_recognition"),
                active=bool(self._feature("face_recognition") and self._face is not None),
                state=self._stage_status(
                    "face",
                    enabled=self._feature("face_recognition"),
                    available=self._face is not None,
                ),
                detail="recognition and identity reacquire",
            ),
            self._pose_service_row(pool),
            RuntimeServiceInfo(
                key="framing",
                name="Framing",
                configured=str(getattr(tracking, "framing", "upper_body")),
                enabled=bool(getattr(self.config.ptz, "safe_zone_enabled", True)),
                active=bool(getattr(self.config.ptz, "safe_zone_enabled", True)),
                state="active"
                if getattr(self.config.ptz, "safe_zone_enabled", True)
                else "disabled",
                detail="ignore arms" if ignore_arms else "include arms",
            ),
            RuntimeServiceInfo(
                key="fps",
                name="FPS cap",
                configured=f"{target_fps:.0f} fps",
                enabled=True,
                active=True,
                state="active",
                detail=(f"trusted source cap {cap:.0f} fps" if cap else "source cap unknown"),
                confidence="trusted" if cap else "unknown",
            ),
        ]

    def _pose_service_row(self, pool: Any) -> RuntimeServiceInfo:
        """Pose status, surfacing a 'present but failed to load' reason.

        Mirrors the detector row: when pose is on and the model was probed but the
        estimator is unavailable with a known reason (e.g. the ORT session failed
        on this platform), report ``failed`` + that reason instead of a silent
        idle — so "downloaded but not working" explains itself.
        """
        enabled = self._feature("pose")
        pose_obj = self._pose
        loaded = pose_obj is not None and bool(getattr(pose_obj, "available", True))
        error = str(getattr(pool, "pose_error", "") or "") if pool is not None else ""
        if not error:
            error = str(getattr(pose_obj, "error", "") or "")
        failed = enabled and self._pose_probed and not loaded and bool(error)
        if failed:
            state, detail = "failed", error
        else:
            state = self._stage_status(
                "pose", enabled=enabled, available=loaded or not self._pose_probed
            )
            detail = "PTZ aim point" if enabled else "disabled"
        return RuntimeServiceInfo(
            key="pose",
            name="Pose",
            configured="on",
            enabled=enabled,
            active=bool(enabled and loaded),
            state=state,
            detail=detail,
        )

    def _reid_service_row(self, tracking: Any) -> RuntimeServiceInfo:
        """ReID status with the resolved on/off state and a plain reason.

        ReID is unified under two controls: the global ``reid`` feature and the
        per-camera ``tracking_mode`` (stable uses it, responsive doesn't).  This
        row reports the *effective* state so it never contradicts the menu."""
        feature_on = self._feature("reid")
        mode = getattr(tracking, "tracking_mode", "stable")
        wants_reid = mode == "stable"
        if not feature_on:
            state, detail = "disabled", "ReID feature off (Services)"
        elif not wants_reid:
            state, detail = "off", "Responsive mode — no ReID hold"
        elif self._reid is not None and getattr(self._reid, "available", False):
            state, detail = "active", "holding target through crossings"
        elif self._reid_probed:
            state, detail = "failed", "ReID model unavailable (boxmot/torch)"
        else:
            state, detail = "warming", "stable mode — ReID will engage on lock"
        return RuntimeServiceInfo(
            key="reid",
            name="ReID",
            configured=f"{mode} · {'on' if feature_on else 'off'}",
            enabled=bool(feature_on and wants_reid),
            active=bool(self._reid is not None and getattr(self._reid, "available", False)),
            state=state,
            detail=detail,
        )

    def _quality_state(self) -> QualityStateInfo:
        tracking = self.config.tracking
        pool = self._pool
        floor = str(getattr(tracking, "quality_floor", "auto") or "auto")
        tier = str(getattr(pool, "detector_tier", "") or "") if pool is not None else ""
        model = str(getattr(pool, "detector_model_name", "") or "") if pool is not None else ""
        switch = self._model_switch_info()
        if switch is not None and switch.state == "warming":
            reason = switch.reason or f"Switching to {switch.to_value}."
        else:
            reason = self._quality_reason
        return QualityStateInfo(
            floor=floor,
            active=self._quality_active if floor == "auto" else floor,
            reason=reason,
            detector_tier=tier,
            detector_model=model,
            tracker=str(getattr(tracking, "tracker", "")),
            detect_interval=self._quality_interval,
        )

    def _model_switch_info(self) -> SwitchStateInfo | None:
        pool = self._pool
        state_fn = getattr(pool, "switch_state", None) if pool is not None else None
        if not callable(state_fn):
            return None
        try:
            state = state_fn()
        except Exception:  # noqa: BLE001
            return None
        return SwitchStateInfo(**state) if state else None

    # ── inference-hang watchdog ───────────────────────────────────────────────

    def _inference_stalled(self, now: float) -> bool:
        """Return True iff the inference thread appears to have hung.

        Pure predicate — no side effects.  Safe to call from the capture thread
        at any time.

        Conditions (all must hold):
        - At least one inference tick has completed (``_frames_inferred > 0``),
          so we never fire on a fresh worker that hasn't started yet.
        - Tracking is enabled (``_tracking_enabled``), because a stall only
          matters when auto PTZ would be driven.
        - The heartbeat timestamp is older than ``_INFER_STALL_S`` seconds.
        """
        return (
            self._frames_inferred > 0
            and self._tracking_enabled
            and (now - self._last_infer_t) > _INFER_STALL_S
        )

    def _apply_inference_watchdog(self, now: float) -> None:
        """Watchdog action — called once per telemetry tick from the capture thread.

        If ``_inference_stalled`` is True:
        - Stops PTZ motion ONCE on the False→True transition of ``_watchdog_stalled``
          (edge-triggered) so VISCA-IP/ONVIF hardware is not flooded with stop
          packets every tick.
        - Keeps ``_watchdog_stalled`` set (level) so ``_tracking_status_info``
          continues to surface a DEGRADED/stalled state with
          ``severity="warning"`` each tick until inference recovers.

        Recovery is automatic when ``_last_infer_t`` advances (next successful
        inference tick); ``_watchdog_stalled`` is cleared and re-armed so a
        subsequent stall issues a fresh stop.
        """
        if self._inference_stalled(now):
            if not self._watchdog_stalled:  # edge: stop once on entering the stall
                backend = self._ptz_backend
                if backend is not None and hasattr(backend, "stop"):
                    try:
                        with self._ptz_lock:
                            backend.stop()
                    except Exception:  # noqa: BLE001
                        log.debug(
                            "camera_id=%s watchdog stop failed", self.camera_id, exc_info=True
                        )
            self._watchdog_stalled = True
        else:
            self._watchdog_stalled = False

    def _emit_telemetry(
        self,
        *,
        tracks: list[TrackInfo],
        health: HealthState,
        last_error: str | None,
    ) -> None:
        # Surface a recent inference-stage failure (detect/track raise) when the
        # capture path itself is healthy, so the Camera Info panel can show *why*
        # a streaming-but-box-blind camera has no tracks.
        if last_error is None and self._infer_last_error is not None:
            last_error = self._infer_last_error
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
            target_fps=self._target_fps(),
            frame_budget_ms=self._frame_budget_ms(),
            runtime_services=self._runtime_services(),
            stage_timings=self._stage_timings(),
            quality_state=self._quality_state(),
            model_switch=self._model_switch_info(),
            tracker_switch=self._tracker_switch,
            runtime_events=list(self._runtime_events),
            tracks=tracks,
            faces=self._fresh_faces_for_telemetry(tracks),
            pose=self._pose_overlay(),
            ptz=self._ptz_state(),
            tracking_status=self._tracking_status_info(tracks, time.monotonic()),
            health=HealthInfo(state=health, last_error=last_error),
        )
        self._seq += 1
        try:
            self._on_telemetry(msg)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s telemetry callback raised", self.camera_id, exc_info=True)

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

        moving = abs(pan) > 1e-3 or abs(tilt) > 1e-3 or abs(zoom) > 1e-3

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
