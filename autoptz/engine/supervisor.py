"""Supervisor: owns CameraWorkers, routes commands, fans telemetry back to the UI.

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
Each camera runs on its own threads (capture + inference) in this process.
**Future hardening:** process-per-camera (``multiprocessing`` with shm transport)
for fault isolation and to sidestep the GIL under many cameras — the
worker/telemetry contract is already process-safe (shm + msgpack), so that is a
localized change.

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

# Worker liveness / auto-restart constants
_HEALTH_SCAN_INTERVAL_S = 2.0  # how often tick() runs the health scan
_BASE_BACKOFF_S = 1.0  # initial restart back-off
_MAX_BACKOFF_S = 30.0  # maximum restart back-off (exponential cap)
_MAX_RESTART_ATTEMPTS = 5  # give up after this many consecutive failures


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
                "fallback; camera selection may be unreliable. Reinstall with "
                "`python tools/install.py --editable`.",
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

        # ── worker health monitoring + auto-restart ──────────────────────────────
        # Per-camera backoff state: cid → (attempts, next_allowed_t).
        # Cleared when a worker is healthy again or the camera is removed.
        self._restart_state: dict[str, tuple[int, float]] = {}
        # Monotonic timestamp of the last health scan (throttled to
        # _HEALTH_SCAN_INTERVAL_S so tick() doesn't scan every 50 ms).
        self._last_health_scan_t: float = 0.0

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def prime_features(self, features: dict[str, bool] | None) -> None:
        """Seed the global feature switches *before* :meth:`start` spawns workers.

        ``_spawn_worker`` applies ``self._features`` to each new worker, so seeding
        them here means a disabled service is never built at spin-up (the worker
        gates model construction on these flags).  Must be called before ``start``;
        a no-op mid-run since workers already exist.
        """
        self._features = {str(k): bool(v) for k, v in (features or {}).items()}

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
        capture come up first, then inference is released without a forced model
        warmup gate. That keeps launch responsive while preserving all services
        as logically enabled.
        The UI uses this path so camera opens/model warmup do not freeze the
        main window; direct tests/headless callers keep the synchronous default.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
            camera_ids = self._client.cameraModel.camera_ids()

        # Spawning happens OUTSIDE the lock: worker.start() opens the camera
        # (which can block for seconds), and holding self._lock across it would
        # stall any GUI-thread call that routes through the command pump.
        # _spawn_worker takes the lock only to register the worker.
        self._apply_hardware_env(len(camera_ids))
        if staged:
            log.info("supervisor starting — %d camera(s)", len(camera_ids))
            self._start_pump_if_needed(run_pump)
            startup_thread = threading.Thread(
                target=self._startup_loop,
                args=(camera_ids, progress),
                name="engine-startup",
                daemon=True,
            )
            self._startup_thread = startup_thread
            startup_thread.start()
            return

        log.info("supervisor starting — %d camera(s), ep=%s", len(camera_ids), self.active_ep)
        _log_macos_capture_path()
        for camera_id in camera_ids:
            self._spawn_worker(camera_id)

        self._start_pump_if_needed(run_pump)

    def stop(self) -> None:
        """Stop the pump and tear down all workers.  Idempotent."""
        with self._lock:
            if not self._running:
                return
            self._running = False

            log.info("supervisor stopping — tearing down %d worker(s)", len(self._workers))
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
                target=self._pump_loop,
                name="engine-cmd-pump",
                daemon=True,
            )
            self._pump_thread.start()

    def _startup_loop(self, camera_ids: list[str], progress: Any | None) -> None:
        """Preview-first staged camera start."""
        total = len(camera_ids)
        adaptive_concurrency = self._adaptive_startup_concurrency()
        concurrency = 1
        self._progress(progress, active=True, phase="Opening cameras", started=0, total=total)
        if total == 0:
            self._progress(progress, active=False, phase="Ready", started=0, total=0)
            return
        started = 0
        for camera_id in camera_ids:
            if not self._running:
                self._progress(progress, active=False, phase="")
                return
            # _spawn_worker re-checks running + dedupes under the lock, and runs
            # the slow worker.start() outside it so this loop never blocks the UI.
            self._spawn_worker(camera_id, defer_inference=True)
            started += 1
            self._progress(
                progress, active=True, phase="Opening cameras", started=started, total=total
            )
            if started == 1:
                concurrency = adaptive_concurrency
                time.sleep(0.25)
            if started % concurrency == 0 and started < total:
                time.sleep(0.35)

        self._progress(
            progress, active=True, phase="Starting detection", started=started, total=total
        )
        self._release_worker_inference()
        self._progress(progress, active=False, phase="Ready", started=started, total=total)

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

    def switch_detector_model_tier(self, tier: str, *, reason: str = "") -> None:
        """Hot-swap the shared detector model without stopping workers.

        The build runs on a background thread. The old detector remains active
        until the pool reports a successful swap, then each worker refreshes its
        detect stack to point at the new shared detector while keeping its own
        tracker state unless the tracker itself changed.
        """
        if not self._running:
            return
        pool = self._ensure_inference_pool()
        if pool is None or not hasattr(pool, "switch_detector_tier"):
            return

        def _swap() -> None:
            try:
                ok = bool(
                    pool.switch_detector_tier(
                        tier,
                        reason=reason or f"Operator selected detector tier {tier}.",
                    )
                )
            except Exception:  # noqa: BLE001
                log.warning("detector tier switch to %s failed", tier, exc_info=True)
                return
            if not ok:
                return
            with self._lock:
                workers = list(self._workers.values())
            for worker in workers:
                refresh = getattr(worker, "refresh_detector_from_pool", None)
                if callable(refresh):
                    try:
                        refresh()
                    except Exception:  # noqa: BLE001
                        log.debug("worker detector refresh failed", exc_info=True)

        threading.Thread(
            target=_swap,
            name=f"detector-swap-{str(tier or 'auto')}",
            daemon=True,
        ).start()

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
                log.warning("error routing command %s", getattr(cmd, "kind", "?"), exc_info=True)
        now = time.monotonic()
        if now - self._last_health_scan_t >= _HEALTH_SCAN_INTERVAL_S:
            self._last_health_scan_t = now
            self._scan_worker_health(now)

    def _pump_loop(self) -> None:
        while not self._pump_stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.debug("pump tick error", exc_info=True)
            elapsed = time.monotonic() - t0
            self._pump_stop.wait(max(0.0, _PUMP_INTERVAL_S - elapsed))

    # ── worker health monitoring ─────────────────────────────────────────────────

    def _scan_worker_health(self, now: float) -> None:
        """Check every worker's liveness and restart any that have died.

        Throttled by the caller (``tick()``) to run at most every
        ``_HEALTH_SCAN_INTERVAL_S`` seconds.  Uses a locked snapshot of
        ``_workers`` so the scan never holds the lock across blocking calls
        (``worker.stop()`` or ``_spawn_worker()``).
        """
        if not self._running:
            return
        with self._lock:
            snapshot = list(self._workers.items())

        for cid, worker in snapshot:
            alive = False
            try:
                alive = bool(worker.is_alive())
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s is_alive() raised", cid, exc_info=True)

            if alive:
                # Worker is healthy — clear any stale restart state.
                self._restart_state.pop(cid, None)
                continue

            # Worker appears dead.  Check back-off state.
            attempts, next_allowed_t = self._restart_state.get(cid, (0, 0.0))

            if attempts >= _MAX_RESTART_ATTEMPTS:
                # Already gave up — log once (when attempts first hits the cap
                # the counter was bumped; subsequent scans see == cap, skip).
                # The "log once" is enforced by the sentinel value stored below.
                continue

            if now < next_allowed_t:
                # Still in the back-off window — do nothing this scan.
                continue

            # Attempt a restart.
            log.info(
                "camera_id=%s capture thread died — restarting (attempt %d/%d)",
                cid,
                attempts + 1,
                _MAX_RESTART_ATTEMPTS,
            )
            try:
                worker.stop()
            except Exception:  # noqa: BLE001
                log.debug("camera_id=%s stop() before restart failed", cid, exc_info=True)

            with self._lock:
                # Guard: camera may have been explicitly removed since the snapshot.
                # Note: tick() (command routing + health scan) runs on a single
                # thread, so _on_remove_camera and this scan cannot truly interleave;
                # this re-check is belt-and-suspenders against external direct calls.
                if cid in self._workers:
                    del self._workers[cid]
                else:
                    # Removed concurrently — clear state and skip the respawn.
                    self._restart_state.pop(cid, None)
                    continue

            new_attempts = attempts + 1
            backoff = min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * (2 ** (new_attempts - 1)))
            self._restart_state[cid] = (new_attempts, now + backoff)

            if new_attempts >= _MAX_RESTART_ATTEMPTS:
                log.warning(
                    "camera_id=%s worker failed %d times — giving up on auto-restart",
                    cid,
                    new_attempts,
                )
                # Don't respawn; keep the sentinel so future scans skip this camera.
                continue

            try:
                self._spawn_worker(cid)
            except Exception:  # noqa: BLE001
                log.warning("camera_id=%s respawn failed", cid, exc_info=True)

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
        # _spawn_worker dedupes + registers under the lock and starts the worker
        # outside it (camera open can block).
        if cmd.camera_id:
            self._spawn_worker(cmd.camera_id)

    def _on_remove_camera(self, cmd: RemoveCameraCmd) -> None:
        cid = cmd.camera_id or ""
        with self._lock:
            worker = self._workers.pop(cid, None)
        # Clear any pending restart state so a removed camera isn't resurrected.
        self._restart_state.pop(cid, None)
        if worker is not None:
            try:
                worker.stop()
            except Exception:  # noqa: BLE001
                log.warning("error stopping removed worker %s", cmd.camera_id, exc_info=True)
            try:
                self._client.request_provider_detach(cid)
            except Exception:  # noqa: BLE001
                log.debug("provider detach request failed for %s", cmd.camera_id, exc_info=True)

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
        if (
            worker is not None
            and cmd.track_id is not None
            and cmd.identity_id
            and hasattr(worker, "enroll_track")
        ):
            worker.enroll_track(
                cmd.track_id,
                cmd.identity_id,
                cmd.identity_name,
                getattr(cmd, "click_x", None),
                getattr(cmd, "click_y", None),
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
        prev = dict(self._features)
        new_features = dict(cmd.features or {})
        self._features = new_features
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            if hasattr(worker, "set_features"):
                try:
                    worker.set_features(dict(new_features))
                except Exception:  # noqa: BLE001
                    log.debug("worker.set_features failed", exc_info=True)
        # Free the shared pool's copy of any subsystem just switched off so its
        # ORT session is reclaimed once the workers drop their refs too.  ReID is
        # per-worker (not pooled), so the workers' own release is sufficient.
        self._release_disabled_pool_models(prev, new_features)

    def _release_disabled_pool_models(
        self,
        prev: dict[str, bool],
        cur: dict[str, bool],
    ) -> None:
        pool = self._inference_pool
        if pool is None:
            return

        def turned_off(key: str) -> bool:
            return bool(prev.get(key, True)) and not bool(cur.get(key, True))

        releases = (
            ("detection", "release_detector"),
            ("face_recognition", "release_face"),
            ("pose", "release_pose"),
        )
        for feature, method in releases:
            if not turned_off(feature):
                continue
            fn = getattr(pool, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    log.debug("pool %s failed", method, exc_info=True)

    def release_model_sessions(self) -> None:
        """Free every detector/pose ORT session (pool + workers) before a cache mutation.

        Windows refuses to delete/replace a model file while onnxruntime still has
        it open, so the UI calls this *before* downloading/removing files: the pool
        drops its shared models and each worker synchronously releases its refs,
        then a ``gc.collect()`` finalises the sessions so the OS handles are gone.
        POSIX tolerates unlink-while-open, which is why the old (release-after)
        order only failed on Windows.
        """
        if not self._running:
            return
        pool = self._inference_pool
        if pool is not None:
            for method in ("release_detector", "release_pose"):
                fn = getattr(pool, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        log.debug("pool %s failed", method, exc_info=True)
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            release = getattr(worker, "release_inference_models", None)
            if callable(release):
                try:
                    # Block briefly so the inference thread actually drops its refs
                    # before we GC + mutate; the file-op retry covers any residual.
                    release(wait=1.0)
                except Exception:  # noqa: BLE001
                    log.debug("worker model release failed", exc_info=True)
        import gc

        gc.collect()

    def rebuild_model_sessions(self) -> None:
        """Rebuild detector/pose from the (now-refreshed) cache after a mutation.

        Each worker force-reloads from the shared pool — yielding the new model, or
        live-preview-only when the files were removed.  Pairs with
        :meth:`release_model_sessions`.
        """
        if not self._running:
            return
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            reload = getattr(worker, "reload_inference_models", None)
            if callable(reload):
                try:
                    reload()
                except Exception:  # noqa: BLE001
                    log.debug("worker model reload failed", exc_info=True)

    def apply_model_cache_changed(self) -> None:
        """Release then rebuild model sessions in one call (release-before-rebuild).

        Retained for callers that mutate the cache and only signal afterward; the
        Model Manager now calls :meth:`release_model_sessions` *before* the mutation
        and :meth:`rebuild_model_sessions` after, which is what makes delete/replace
        work on Windows.
        """
        self.release_model_sessions()
        self.rebuild_model_sessions()

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
            log.warning("identity service init failed; identity features off.", exc_info=True)
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
            self._inference_pool = build_inference_pool(
                detector_tier=tier,
                unified_pose=self._any_unified_pose(),
                allow_model_download=False,
            )
        except Exception:  # noqa: BLE001 — pool is an optimisation, never load-bearing
            log.warning("inference pool init failed; using per-worker models.", exc_info=True)
            self._inference_pool = None
        return self._inference_pool

    def _apply_hardware_env(self, camera_count: int) -> None:
        """Publish global hardware prefs into the environment before workers start.

        :func:`autoptz.engine.runtime.inference.prefs_from_env` reads these for
        every ORT session, so prefs reach the per-camera worker threads (and a
        future process-per-camera build, which would inherit the env) without
        threading them through the command schema.
        """
        import os

        from autoptz.config.models import HardwarePrefs

        try:
            raw = self._store.get_setting("hardware", {}) if self._store is not None else {}
            hw = HardwarePrefs.model_validate(raw) if raw else HardwarePrefs()
        except Exception:  # noqa: BLE001 — bad/absent prefs fall back to defaults
            hw = HardwarePrefs()

        if hw.force_ep:
            os.environ["AUTOPTZ_FORCE_EP"] = hw.force_ep
        else:
            os.environ.pop("AUTOPTZ_FORCE_EP", None)
        os.environ["AUTOPTZ_PRECISION"] = hw.precision

        if hw.intra_op_threads:
            threads = int(hw.intra_op_threads)
        else:
            # Spread cores across cameras so several workers don't oversubscribe,
            # and reserve one core for the capture/UI/paint threads so inference
            # can't starve the very pipeline that feeds it (keeps preview smooth).
            cores = os.cpu_count() or 4
            usable = max(1, cores - 1)
            threads = max(1, usable // max(1, camera_count))
        os.environ["AUTOPTZ_ORT_INTRA_THREADS"] = str(threads)

        log.info(
            "hardware prefs → env | force_ep=%s precision=%s intra_threads=%s "
            "(cores=%s, cameras=%s)",
            hw.force_ep or "auto",
            hw.precision,
            threads,
            os.cpu_count(),
            camera_count,
        )

    def _make_worker(self, camera_id: str, config: CameraConfig) -> Any:
        """Build a camera worker — a thread-based one, or (opt-in) a child process.

        Process-per-camera is used only when ``AUTOPTZ_PROCESS_PER_CAMERA`` is set
        AND no test/custom worker factory was injected, so the default and all
        tests keep the in-process threaded worker.
        """
        on_telemetry = self._client.push_telemetry
        from autoptz.engine.process_worker import (
            ProcessWorkerHandle,
            process_per_camera_enabled,
        )

        use_process = (
            self._worker_factory is self._default_worker_factory and process_per_camera_enabled()
        )
        if not use_process:
            return self._worker_factory(camera_id, config, on_telemetry)

        tier = "auto"
        try:
            getter = getattr(self._client, "getDetectorModelTier", None)
            if callable(getter):
                tier = str(getter() or "auto")
        except Exception:  # noqa: BLE001
            tier = "auto"
        db_path = ""
        try:
            db_path = str(getattr(self._store, "_path", "") or "")
        except Exception:  # noqa: BLE001
            db_path = ""
        log.info("spawning camera %s as its own PROCESS (experimental)", camera_id)
        return ProcessWorkerHandle(
            camera_id,
            config,
            on_telemetry,
            db_path=db_path,
            detector_tier=tier,
            unified_pose=self._any_unified_pose(),
        )

    def _spawn_worker(self, camera_id: str, *, defer_inference: bool = False) -> None:
        config = self._resolve_config(camera_id)
        if config is None:
            log.warning("cannot spawn worker for %s: no config", camera_id)
            return
        worker = self._make_worker(camera_id, config)
        # Register under the lock only; bail if we lost a race or the engine
        # stopped meanwhile.  Everything heavy (worker.start) runs outside it.
        with self._lock:
            if not self._running or camera_id in self._workers:
                return
            self._workers[camera_id] = worker
            worker_count = len(self._workers)
        log.info(
            "spawned worker camera_id=%s name=%s (workers=%d)",
            camera_id,
            getattr(config, "name", "?"),
            worker_count,
        )

        # Share the gallery + wire the worker→client identity push (mirrors
        # telemetry).  Done via setters so the 3-arg worker_factory contract
        # (and test fakes) stays unchanged.
        service = self._ensure_identity_service()
        if service is not None and hasattr(worker, "set_identity_service"):
            worker.set_identity_service(service)
        if hasattr(worker, "set_identity_callback") and hasattr(
            self._client,
            "push_identity",
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

        # If stop() ran while we were wiring this worker up, it has already been
        # pulled from _workers — skip start() so we don't resurrect a dead worker.
        with self._lock:
            if not self._running or self._workers.get(camera_id) is not worker:
                return
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
                camera_id,
                shm_name,
                _PREVIEW_W,
                _PREVIEW_H,
            )
        except Exception:  # noqa: BLE001
            log.debug("provider attach request failed for %s", camera_id, exc_info=True)

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
        self,
        camera_id: str,
        config: CameraConfig,
        on_telemetry: Any,
    ) -> CameraWorker:
        return CameraWorker(camera_id, config, on_telemetry)

    def _any_unified_pose(self) -> bool:
        """True iff any configured camera opts into the unified pose detector.

        The pool also honours ``AUTOPTZ_UNIFIED_POSE``; this just lets the config
        flag drive it too.  Best-effort — never raises into pool construction.
        """
        try:
            for cid in self._client.cameraModel.camera_ids():
                cfg = self._resolve_config(cid)
                if cfg is not None and getattr(cfg.tracking, "unified_pose", False):
                    return True
        except Exception:  # noqa: BLE001
            log.debug("unified-pose config scan failed", exc_info=True)
        return False

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
