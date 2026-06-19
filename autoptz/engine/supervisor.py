"""Supervisor: owns CameraWorkers, routes commands, fans telemetry back to the UI.

P0 implementation
-----------------
The :class:`Supervisor` is the engine's top-level orchestrator.  It:

- Holds the :class:`EngineClient` (the UI-facing command/telemetry hub) and the
  :class:`ConfigStore`.
- ``start()`` spawns one :class:`CameraWorker` per camera currently in the
  client's camera model; ``stop()`` tears them all down.  Both are idempotent.
- Exposes :attr:`is_running` and the active inference EP (:meth:`get_best_ep`).
- Runs a *command pump* (:meth:`tick`) that calls
  ``engineClient.drain_commands()`` and routes each command to the right worker
  (``AddCamera`` → spawn, ``RemoveCamera`` → stop+drop,
  ``EnableTracking`` / ``SetTarget`` / ``PtzNudge`` / ``UpdateCameraConfig`` →
  worker).
- Pushes telemetry back to the UI via ``engineClient.push_telemetry(msg)``.

Threading
---------
Workers run on threads (one per camera) for P0.  **Future hardening:**
process-per-camera (``multiprocessing`` with shm transport) for fault isolation
and to sidestep the GIL under many cameras — the worker/telemetry contract here
is already process-safe (shm + msgpack), so this is a localized change.

The command pump can either be driven externally (``tick()`` from a GUI-thread
``QTimer`` — the default the UI uses) or by an internal daemon thread
(``start(run_pump=True)``) for headless use.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from autoptz.engine.camera_worker import _PREVIEW_H, _PREVIEW_W, CameraWorker
from autoptz.engine.runtime.inference import EP, get_best_ep
from autoptz.engine.runtime.messages import (
    AddCameraCmd,
    CmdKind,
    EnableTrackingCmd,
    EnrollIdentityCmd,
    PtzHomeCmd,
    PtzMenuCmd,
    PtzNudgeCmd,
    RecallPtzPresetCmd,
    RemoveCameraCmd,
    SavePtzPresetCmd,
    SetFeaturesCmd,
    SetTargetCmd,
    SetTargetFpsCmd,
    SetTargetIdentityCmd,
    UpdateCameraConfigCmd,
)

if TYPE_CHECKING:
    from autoptz.config.models import CameraConfig
    from autoptz.config.store import ConfigStore
    from autoptz.ui.engine_client import EngineClient

log = logging.getLogger(__name__)

_PUMP_INTERVAL_S = 0.05  # 20 Hz command pump when run internally


def _log_macos_capture_path() -> None:
    """Log (once at start) which USB capture path macOS will use.

    Native AVFoundation binds cameras by their stable ``uniqueID`` (correct
    selection); without the pyobjc extras we fall back to OpenCV's divergent
    device ordering, where selection can be unreliable.  Surfacing this up front
    makes the "wrong camera" class of bug diagnosable from the logs.
    """
    import platform

    if platform.system() != "Darwin":
        return
    try:
        from autoptz.engine.pipeline import avf_capture

        if avf_capture.is_available():
            log.info("macOS capture: native AVFoundation (bound by uniqueID).")
        else:
            log.warning(
                "macOS capture: native AVFoundation UNAVAILABLE — using OpenCV "
                "fallback; camera selection may be unreliable. Install the macOS "
                "extras: pip install -r requirements/macos.txt",
            )
    except Exception:  # noqa: BLE001 — diagnostics must never break startup
        log.debug("macOS capture-path probe failed", exc_info=True)


class Supervisor:
    """Owns and supervises one CameraWorker per camera.

    Args:
        engine_client: The UI-facing :class:`EngineClient`.  Telemetry is pushed
                       back via ``push_telemetry`` and commands are pulled via
                       ``drain_commands``.
        store:         Optional :class:`ConfigStore` (used to resolve a camera's
                       full :class:`CameraConfig`).
        worker_factory: Override the worker constructor (tests inject fakes).
                       Signature: ``(camera_id, config, on_telemetry) -> worker``.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        store: ConfigStore | None = None,
        *,
        worker_factory: Any | None = None,
    ) -> None:
        self._client = engine_client
        self._store = store
        self._worker_factory = worker_factory or self._default_worker_factory

        self._workers: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._running = False

        # One shared identity gallery across all workers so a face harvested /
        # enrolled on any camera is recognised on every camera, and CRUD from
        # the UI and worker-thread harvesting see the same records.  Built lazily
        # (and gracefully) the first time a worker spawns.
        self._identity_service: Any | None = None

        # One shared inference pool (detector + face + pose sessions) across all
        # workers so heavy models are loaded ONCE for the whole app instead of
        # per-camera.  Each worker keeps its own (stateful) tracker.  Built lazily
        # and gracefully; workers fall back to per-worker models if it's None.
        self._inference_pool: Any | None = None

        # Global ML-subsystem switches (detection / tracking / face_recognition /
        # pose), broadcast via SetFeaturesCmd and applied to every worker.
        self._features: dict[str, bool] = {}

        # internal pump thread (only when start(run_pump=True))
        self._pump_thread: threading.Thread | None = None
        self._pump_stop = threading.Event()
        self._startup_thread: threading.Thread | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def start(
        self,
        *,
        run_pump: bool = False,
        staged: bool = False,
        progress: Any | None = None,
    ) -> None:
        """Spawn a worker for every camera in the model.  Idempotent.

        If *run_pump* is True, also starts an internal daemon thread that drives
        :meth:`tick`.  When the UI owns a GUI-thread ``QTimer`` it leaves this
        False and calls :meth:`tick` itself.

        ``staged=True`` preserves the public "running" state immediately but
        opens cameras on a background startup thread in small adaptive batches.
        Workers are started with inference paused; preview shared memory and
        capture come up first, then the shared models warm once before inference
        is released. That keeps launch responsive while preserving all services
        as logically enabled.
        The UI uses this path so camera opens/model warmup do not freeze the
        main window; direct tests/headless callers keep the synchronous default.
        """
        with self._lock:
            if self._running:
                return
            self._running = True

            camera_ids = self._client.cameraModel.camera_ids()
            log.info("supervisor starting — %d camera(s), ep=%s",
                     len(camera_ids), self.active_ep)
            _log_macos_capture_path()
            if staged:
                self._start_pump_if_needed(run_pump)
                self._startup_thread = threading.Thread(
                    target=self._startup_loop,
                    args=(camera_ids, progress),
                    name="engine-startup",
                    daemon=True,
                )
                self._startup_thread.start()
                return

            for camera_id in camera_ids:
                self._spawn_worker(camera_id)

            self._start_pump_if_needed(run_pump)

    def stop(self) -> None:
        """Stop the pump and tear down all workers.  Idempotent."""
        with self._lock:
            if not self._running:
                return
            self._running = False

            log.info("supervisor stopping — tearing down %d worker(s)",
                     len(self._workers))
            self._pump_stop.set()
            pump = self._pump_thread
            self._pump_thread = None
            startup = self._startup_thread
            self._startup_thread = None

            workers = list(self._workers.items())
            self._workers.clear()

        # Join the pump outside the lock (it may call tick→lock).
        if pump is not None and pump is not threading.current_thread():
            pump.join(timeout=2.0)
        if startup is not None and startup is not threading.current_thread():
            startup.join(timeout=2.0)

        # Stop workers outside the lock so a worker thread calling back in
        # (telemetry) never deadlocks.
        for _cid, worker in workers:
            try:
                worker.stop()
            except Exception:  # noqa: BLE001
                log.warning("error stopping worker %s", _cid, exc_info=True)
            try:
                self._client.request_provider_detach(_cid)
            except Exception:  # noqa: BLE001
                log.debug("provider detach request failed for %s", _cid, exc_info=True)

    def _start_pump_if_needed(self, run_pump: bool) -> None:
        if run_pump and self._pump_thread is None:
            self._pump_stop.clear()
            self._pump_thread = threading.Thread(
                target=self._pump_loop, name="engine-cmd-pump", daemon=True,
            )
            self._pump_thread.start()

    def _startup_loop(self, camera_ids: list[str], progress: Any | None) -> None:
        """Preview-first staged camera start + model warmup."""
        total = len(camera_ids)
        adaptive_concurrency = self._adaptive_startup_concurrency()
        concurrency = 1
        self._progress(progress, active=True, phase="Opening cameras",
                       started=0, total=total)
        if total == 0:
            self._progress(progress, active=False, phase="Ready",
                           started=0, total=0)
            return
        started = 0
        for camera_id in camera_ids:
            with self._lock:
                if not self._running:
                    self._progress(progress, active=False, phase="")
                    return
                if camera_id not in self._workers:
                    self._spawn_worker(camera_id, defer_inference=True)
            started += 1
            self._progress(progress, active=True, phase="Opening cameras",
                           started=started, total=total)
            if started == 1:
                concurrency = adaptive_concurrency
                time.sleep(0.25)
            if started % concurrency == 0 and started < total:
                time.sleep(0.35)

        pool = self._ensure_inference_pool()
        self._progress(progress, active=True, phase="Warming detector",
                       started=started, total=total)
        if pool is not None:
            try:
                detector = getattr(pool, "detector", None)
                if callable(detector):
                    detector()
            except Exception:  # noqa: BLE001
                log.debug("startup detector warmup failed", exc_info=True)

        self._progress(progress, active=True, phase="Warming ReID",
                       started=started, total=total)
        self._warm_reid()

        if pool is not None:
            self._progress(progress, active=True, phase="Warming face",
                           started=started, total=total)
            try:
                face = getattr(pool, "face", None)
                if callable(face):
                    face()
            except Exception:  # noqa: BLE001
                log.debug("startup face warmup failed", exc_info=True)

            self._progress(progress, active=True, phase="Warming pose",
                           started=started, total=total)
            try:
                pose = getattr(pool, "pose", None)
                if callable(pose):
                    pose()
            except Exception:  # noqa: BLE001
                log.debug("startup pose warmup failed", exc_info=True)

        self._release_worker_inference()
        self._progress(progress, active=False, phase="Ready",
                       started=started, total=total)

    @staticmethod
    def _adaptive_startup_concurrency() -> int:
        """Aggressive camera-open concurrency constrained by current headroom."""
        cores = os.cpu_count() or 1
        try:
            from autoptz.engine.runtime.diagnostics import system_metrics

            metrics = system_metrics()
        except Exception:  # noqa: BLE001
            metrics = {"available": False}
        if not metrics.get("available"):
            return 1
        cpu = float(metrics.get("cpu_percent", 100.0) or 100.0)
        mem = float(metrics.get("mem_percent", 100.0) or 100.0)
        if cpu < 55.0 and mem < 75.0 and cores >= 8:
            return 4
        if cpu < 70.0 and mem < 85.0 and cores >= 4:
            return 2
        return 1

    @staticmethod
    def _warm_reid() -> None:
        try:
            from autoptz.engine.pipeline.track import boxmot_available

            boxmot_available()
        except Exception:  # noqa: BLE001
            log.debug("startup ReID probe failed", exc_info=True)

    @staticmethod
    def _progress(progress: Any | None, **payload: Any) -> None:
        if progress is None:
            return
        try:
            progress(**payload)
        except Exception:  # noqa: BLE001
            log.debug("startup progress callback failed", exc_info=True)

    @property
    def is_running(self) -> bool:
        return self._running

    def get_best_ep(self) -> EP:
        """Return the best available inference EP for this machine."""
        return get_best_ep()

    @property
    def active_ep(self) -> str:
        """Human-facing EP label (e.g. ``"CoreML"``, ``"CPU"``)."""
        try:
            return _ep_label(self.get_best_ep())
        except Exception:  # noqa: BLE001
            return ""

    @property
    def worker_count(self) -> int:
        with self._lock:
            return len(self._workers)

    def has_worker(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._workers

    # ── command pump ────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Drain pending UI commands and route them.  Safe to call when stopped.

        Designed to be invoked from a GUI-thread ``QTimer`` (the UI's default)
        or from the internal pump thread.
        """
        if not self._running:
            return
        for cmd in self._client.drain_commands():
            try:
                self._route(cmd)
            except Exception:  # noqa: BLE001
                log.warning("error routing command %s", getattr(cmd, "kind", "?"),
                            exc_info=True)

    def _pump_loop(self) -> None:
        while not self._pump_stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.debug("pump tick error", exc_info=True)
            elapsed = time.monotonic() - t0
            self._pump_stop.wait(max(0.0, _PUMP_INTERVAL_S - elapsed))

    # ── command routing ─────────────────────────────────────────────────────────

    def _route(self, cmd: Any) -> None:
        kind = cmd.kind
        if kind == CmdKind.ADD_CAMERA:
            self._on_add_camera(cmd)
        elif kind == CmdKind.REMOVE_CAMERA:
            self._on_remove_camera(cmd)
        elif kind == CmdKind.ENABLE_TRACKING:
            self._on_enable_tracking(cmd)
        elif kind == CmdKind.SET_TARGET:
            self._on_set_target(cmd)
        elif kind == CmdKind.SET_TARGET_IDENTITY:
            self._on_set_target_identity(cmd)
        elif kind == CmdKind.PTZ_NUDGE:
            self._on_ptz_nudge(cmd)
        elif kind == CmdKind.SET_TARGET_FPS:
            self._on_set_target_fps(cmd)
        elif kind == CmdKind.PTZ_SAVE_PRESET_SLOT:
            self._on_save_ptz_preset(cmd)
        elif kind == CmdKind.PTZ_RECALL_PRESET_SLOT:
            self._on_recall_ptz_preset(cmd)
        elif kind == CmdKind.PTZ_HOME:
            self._on_ptz_home(cmd)
        elif kind == CmdKind.PTZ_MENU:
            self._on_ptz_menu(cmd)
        elif kind == CmdKind.SET_FEATURES:
            self._on_set_features(cmd)
        elif kind == CmdKind.UPDATE_CONFIG:
            self._on_update_config(cmd)
        elif kind == CmdKind.ENROLL_IDENTITY:
            self._on_enroll_identity(cmd)
        # Remaining commands (named-preset, identities, layouts) are UI/store
        # concerns with no per-worker effect yet; they are intentionally ignored
        # here so the pump never errors on them.

    def _on_add_camera(self, cmd: AddCameraCmd) -> None:
        with self._lock:
            if cmd.camera_id and cmd.camera_id not in self._workers:
                self._spawn_worker(cmd.camera_id)

    def _on_remove_camera(self, cmd: RemoveCameraCmd) -> None:
        with self._lock:
            worker = self._workers.pop(cmd.camera_id or "", None)
        if worker is not None:
            try:
                worker.stop()
            except Exception:  # noqa: BLE001
                log.warning("error stopping removed worker %s", cmd.camera_id,
                            exc_info=True)
            try:
                self._client.request_provider_detach(cmd.camera_id or "")
            except Exception:  # noqa: BLE001
                log.debug("provider detach request failed for %s", cmd.camera_id,
                          exc_info=True)

    def _on_enable_tracking(self, cmd: EnableTrackingCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is not None:
            worker.enable_tracking(cmd.enabled)

    def _on_set_target(self, cmd: SetTargetCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is not None:
            worker.set_target(cmd.track_id)

    def _on_set_target_identity(self, cmd: SetTargetIdentityCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is not None and hasattr(worker, "set_target_identity"):
            worker.set_target_identity(cmd.identity_id)

    def _on_enroll_identity(self, cmd: EnrollIdentityCmd) -> None:
        """Bind a clicked track's face to the (new or existing) identity.

        The UI created/looked-up the identity in the shared gallery already; the
        worker captures the track's current face embedding on the next face tick
        so the named person is recognised on every camera afterwards.
        """
        worker = self._get(cmd.camera_id)
        if (worker is not None and cmd.track_id is not None
                and cmd.identity_id and hasattr(worker, "enroll_track")):
            worker.enroll_track(
                cmd.track_id, cmd.identity_id, cmd.identity_name,
                getattr(cmd, "click_x", None), getattr(cmd, "click_y", None),
            )

    def _on_ptz_nudge(self, cmd: PtzNudgeCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is not None:
            worker.ptz_nudge(cmd.pan_speed, cmd.tilt_speed, cmd.zoom_speed)

    def _on_set_target_fps(self, cmd: SetTargetFpsCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is not None and hasattr(worker, "set_target_fps"):
            worker.set_target_fps(cmd.fps)

    def _on_save_ptz_preset(self, cmd: SavePtzPresetCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is not None and hasattr(worker, "save_ptz_preset"):
            worker.save_ptz_preset(cmd.slot)

    def _on_recall_ptz_preset(self, cmd: RecallPtzPresetCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is not None and hasattr(worker, "recall_ptz_preset"):
            worker.recall_ptz_preset(cmd.slot)

    def _on_ptz_home(self, cmd: PtzHomeCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is not None and hasattr(worker, "ptz_home"):
            worker.ptz_home()

    def _on_ptz_menu(self, cmd: PtzMenuCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is not None and hasattr(worker, "ptz_menu"):
            worker.ptz_menu()

    def _on_set_features(self, cmd: SetFeaturesCmd) -> None:
        """Global feature switches → cache + apply to every worker live."""
        self._features = dict(cmd.features or {})
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            if hasattr(worker, "set_features"):
                try:
                    worker.set_features(dict(self._features))
                except Exception:  # noqa: BLE001
                    log.debug("worker.set_features failed", exc_info=True)

    def _on_update_config(self, cmd: UpdateCameraConfigCmd) -> None:
        worker = self._get(cmd.camera_id)
        if worker is None:
            return
        config = self._resolve_config(cmd.camera_id or "")
        if config is not None:
            worker.update_config(config)

    # ── worker management ───────────────────────────────────────────────────────

    def _ensure_identity_service(self) -> Any | None:
        """Build (once) the shared identity gallery backed by the store.

        Never raises — returns ``None`` if the identity stack is unavailable so
        workers fall back to per-worker / no identity features.
        """
        if self._identity_service is not None:
            return self._identity_service
        try:
            from autoptz.engine.identity.service import IdentityService

            self._identity_service = IdentityService(self._store)
            # Share it with the client so UI identity CRUD reaches the matcher.
            if hasattr(self._client, "set_identity_service"):
                self._client.set_identity_service(self._identity_service)
        except Exception:  # noqa: BLE001 — identity stack must never break startup
            log.warning("identity service init failed; identity features off.",
                        exc_info=True)
            self._identity_service = None
        return self._identity_service

    def _ensure_inference_pool(self) -> Any | None:
        """Build (once) the shared detector/face/pose inference pool.

        Never raises — returns ``None`` if the pool module/models are unavailable
        so workers fall back to building their own per-worker models (preserving
        the previous behaviour and the test fakes).
        """
        if self._inference_pool is not None:
            return self._inference_pool
        try:
            from autoptz.engine.pipeline.pool import build_inference_pool

            tier = "auto"
            try:
                getter = getattr(self._client, "getDetectorModelTier", None)
                if callable(getter):
                    tier = str(getter() or "auto")
            except Exception:  # noqa: BLE001
                tier = "auto"
            self._inference_pool = build_inference_pool(detector_tier=tier)
        except Exception:  # noqa: BLE001 — pool is an optimisation, never load-bearing
            log.warning("inference pool init failed; using per-worker models.",
                        exc_info=True)
            self._inference_pool = None
        return self._inference_pool

    def _spawn_worker(self, camera_id: str, *, defer_inference: bool = False) -> None:
        config = self._resolve_config(camera_id)
        if config is None:
            log.warning("cannot spawn worker for %s: no config", camera_id)
            return
        worker = self._worker_factory(camera_id, config, self._client.push_telemetry)
        self._workers[camera_id] = worker
        log.info("spawned worker camera_id=%s name=%s (workers=%d)",
                 camera_id, getattr(config, "name", "?"), len(self._workers))

        # Share the gallery + wire the worker→client identity push (mirrors
        # telemetry).  Done via setters so the 3-arg worker_factory contract
        # (and test fakes) stays unchanged.
        service = self._ensure_identity_service()
        if service is not None and hasattr(worker, "set_identity_service"):
            worker.set_identity_service(service)
        if hasattr(worker, "set_identity_callback") and hasattr(
            self._client, "push_identity",
        ):
            worker.set_identity_callback(self._client.push_identity)

        # Inject the shared inference pool (heavy models loaded once for all
        # cameras) and the current global feature switches.
        pool = self._ensure_inference_pool()
        if pool is not None and hasattr(worker, "set_inference_pool"):
            worker.set_inference_pool(pool)
        if self._features and hasattr(worker, "set_features"):
            worker.set_features(dict(self._features))
        if defer_inference and hasattr(worker, "set_inference_start_paused"):
            worker.set_inference_start_paused(True)

        shm_name = getattr(worker, "shm_name", f"cam_{camera_id[:8]}_preview")
        try:
            worker.start()
        except Exception:  # noqa: BLE001
            log.warning("worker for %s failed to start", camera_id, exc_info=True)
        # Ask the UI to attach the live-preview provider AFTER start() — the
        # worker creates its ShmWriter synchronously in start(), so by now the
        # segment exists and the provider's (self-healing) reader can open it.
        # The provider is self-healing regardless, so attach ordering is no
        # longer load-bearing, but emitting after start() opens the preview a
        # frame sooner.
        try:
            self._client.request_provider_attach(
                camera_id, shm_name, _PREVIEW_W, _PREVIEW_H,
            )
        except Exception:  # noqa: BLE001
            log.debug("provider attach request failed for %s", camera_id,
                      exc_info=True)

    def _release_worker_inference(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            if hasattr(worker, "set_inference_start_paused"):
                try:
                    worker.set_inference_start_paused(False)
                except Exception:  # noqa: BLE001
                    log.debug("worker inference release failed", exc_info=True)

    def _get(self, camera_id: str | None) -> Any | None:
        if not camera_id:
            return None
        with self._lock:
            return self._workers.get(camera_id)

    def _default_worker_factory(
        self, camera_id: str, config: CameraConfig, on_telemetry: Any,
    ) -> CameraWorker:
        return CameraWorker(camera_id, config, on_telemetry)

    def _resolve_config(self, camera_id: str) -> CameraConfig | None:
        """Resolve a camera's full CameraConfig.

        Prefers the live model record (already loaded from the store on
        startup); falls back to the store, then to a minimal config synthesized
        from the record's source URI so a freshly-added camera still runs.
        """
        rec = self._client.cameraModel.get_record(camera_id)
        if rec is not None and rec.camera_config is not None:
            return rec.camera_config

        if self._store is not None:
            try:
                for cam in self._store.load_cameras():
                    if cam.id == camera_id:
                        return cam
            except Exception:  # noqa: BLE001
                log.debug("store lookup failed for %s", camera_id, exc_info=True)

        if rec is not None:
            return _synthesize_config(camera_id, rec.source_uri, rec.display_name)
        return None


# ── helpers ─────────────────────────────────────────────────────────────────────


def _ep_label(ep: EP) -> str:
    """Map an EP enum to a short UI label (``CoreMLExecutionProvider`` → ``CoreML``)."""
    return ep.value.replace("ExecutionProvider", "")


def _synthesize_config(camera_id: str, source_uri: str, name: str) -> CameraConfig:
    from autoptz.config.models import CameraConfig, SourceConfig

    source_type = "usb"
    if source_uri.startswith(("rtsp://", "rtsps://")):
        source_type = "rtsp"
    elif source_uri.startswith("onvif://"):
        source_type = "onvif"
    elif source_uri.startswith("ndi://"):
        source_type = "ndi"

    return CameraConfig(
        id=camera_id,
        name=name or source_uri or camera_id,
        source=SourceConfig(type=source_type, address=source_uri),  # type: ignore[arg-type]
    )
