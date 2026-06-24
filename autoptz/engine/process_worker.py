"""Experimental, opt-in **process-per-camera** mode.

Default OFF.  Enable with ``AUTOPTZ_PROCESS_PER_CAMERA=1``.  Each camera's
:class:`~autoptz.engine.camera_worker.CameraWorker` then runs in its own OS
process for true multi-core parallelism (GIL bypass) instead of a thread in the
GUI process.

How the boundary is crossed (the scaffolding the engine was designed around):

- **Frames** already travel through OS shared memory (:mod:`autoptz.engine.runtime.shm`),
  so a child writes the preview ring and the GUI's reader attaches by name — no
  change needed.
- **Commands** (parent→child) and **telemetry / identity events** (child→parent)
  cross via :class:`multiprocessing.Queue`.  Queues pickle their payloads, and the
  payloads here — ``CameraConfig``, ``TelemetryMsg``, ``IdentityRecord`` — are all
  picklable pydantic models, so no hand-rolled framing is needed.
- **Models** can't be shared across processes (an ORT ``InferenceSession`` is not
  picklable), so each child builds **its own** inference pool.  That trades RAM
  (≈ one model set per camera) for parallelism — which is exactly why this is
  opt-in rather than the default.

**Identity:** labeled identities converge through the shared SQLite DB (each child
opens its own connection).  Live cross-process propagation of *unlabeled* harvested
faces is not implemented in this mode yet.

**Status — EXPERIMENTAL.** The IPC + lifecycle plumbing here is unit-tested with a
synthetic source, but the throughput / RAM / real-camera + PTZ behaviour needs
validation on a real multi-camera rig before this is anything more than opt-in.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from autoptz.config.models import CameraConfig

log = logging.getLogger(__name__)

#: Sentinel command name that tells the child drain loop to stop the worker.
_STOP = "__stop__"


@dataclass
class WorkerSpec:
    """Picklable construction args for a child camera process (spawn-safe).

    Everything here must survive ``pickle`` (macOS uses the *spawn* start method),
    so it is only primitives + the frozen ``CameraConfig`` pydantic model.  The
    child rebuilds the heavy, non-picklable bits (inference pool, identity gallery,
    frame source, shm writer) itself from these values.
    """

    camera_id: str
    config: CameraConfig
    db_path: str
    detector_tier: str = "auto"
    unified_pose: bool = False
    features: dict[str, bool] = field(default_factory=dict)
    defer_inference: bool = False
    #: Test hook — drive a built-in synthetic frame source and skip model loading,
    #: so a spawned child needs neither a real camera nor ORT/insightface.
    synthetic: bool = False


class _SyntheticSource:
    """Minimal :class:`FrameSource` emitting solid-colour frames (test/headless).

    Implements only what :class:`CameraWorker` uses — ``open`` / ``read`` /
    ``close`` — so a child process can be exercised end-to-end without a camera.
    ``read`` is **paced** like a real source (real USB/NDI adapters self-pace to
    their fps); an unpaced source would spin the capture loop at thousands of fps,
    cycling the shm triple-buffer so fast a reader only ever sees torn frames.
    """

    def __init__(self, h: int = 720, w: int = 1280, value: int = 123, fps: float = 30.0) -> None:
        import numpy as np

        self._frame = np.full((h, w, 3), value, dtype=np.uint8)
        self._period = 1.0 / max(1.0, fps)

    def open(self) -> bool:
        return True

    def read(self):
        import time

        time.sleep(self._period)
        return self._frame

    def close(self) -> None:
        return None


def _safe_put(q: Any, item: Any) -> None:
    """Put on a queue, never raising into the worker (queue may be closing)."""
    try:
        q.put_nowait(item)
    except Exception:  # noqa: BLE001 — full/closed queue must not break capture
        pass


def run_camera_process(
    spec: WorkerSpec,
    cmd_q: Any,
    telemetry_q: Any,
    identity_q: Any,
) -> None:
    """Child-process entrypoint: build + run a CameraWorker, driven by ``cmd_q``.

    Top-level function so the *spawn* start method can import + pickle it.  Never
    re-raises — a child that fails to build should exit cleanly so the supervisor
    can surface/respawn it, not hang.
    """
    # Quiet the child's root logger to WARNING (its telemetry, not its logs, is
    # what the parent consumes); the parent owns the in-app log view.
    try:
        logging.getLogger().setLevel(logging.WARNING)
    except Exception:  # noqa: BLE001
        pass

    # Re-apply the OpenCV thread cap in this child (it inherited AUTOPTZ_CV2_THREADS
    # from the parent but not the in-process cv2.setNumThreads call).
    from autoptz.engine.runtime.flags import apply_opencv_thread_cap

    apply_opencv_thread_cap()

    worker = None
    try:
        from autoptz.engine.camera_worker import CameraWorker

        def _emit_telemetry(msg: Any) -> None:
            _safe_put(telemetry_q, msg)

        def _emit_identity(rec: Any) -> None:
            _safe_put(identity_q, rec)

        source = _SyntheticSource() if spec.synthetic else None
        worker = CameraWorker(
            spec.camera_id,
            spec.config,
            _emit_telemetry,
            frame_source=source,
            on_identity=_emit_identity,
        )

        if not spec.synthetic:
            _wire_models_and_identity(worker, spec)

        if spec.features:
            worker.set_features(dict(spec.features))
        if spec.defer_inference and hasattr(worker, "set_inference_start_paused"):
            worker.set_inference_start_paused(True)

        worker.start()
        _drain_commands(worker, cmd_q)
    except Exception:  # noqa: BLE001 — a child must fail visibly, never hang
        log.warning("camera process %s crashed during setup/run", spec.camera_id, exc_info=True)
    finally:
        if worker is not None:
            try:
                worker.stop()
            except Exception:  # noqa: BLE001
                log.debug("camera process %s stop failed", spec.camera_id, exc_info=True)


def _wire_models_and_identity(worker: Any, spec: WorkerSpec) -> None:
    """Build this child's OWN inference pool + identity gallery and inject them.

    Mirrors what the supervisor does per-worker in threaded mode, but the objects
    are constructed here (they can't cross the process boundary).  Best-effort:
    a missing pool/gallery degrades to live-preview/no-identity, exactly like the
    threaded path's ``None`` fallbacks.
    """
    try:
        from autoptz.engine.pipeline.pool import build_inference_pool

        pool = build_inference_pool(
            detector_tier=spec.detector_tier,
            unified_pose=spec.unified_pose,
            allow_model_download=False,
        )
        if pool is not None and hasattr(worker, "set_inference_pool"):
            worker.set_inference_pool(pool)
    except Exception:  # noqa: BLE001 — pool is an optimisation, never load-bearing
        log.warning("camera process %s: inference pool init failed", spec.camera_id, exc_info=True)

    try:
        from autoptz.config.store import ConfigStore
        from autoptz.engine.identity.service import IdentityService

        store = ConfigStore(spec.db_path) if spec.db_path else ConfigStore()
        service = IdentityService(store)
        if hasattr(worker, "set_identity_service"):
            worker.set_identity_service(service)
    except Exception:  # noqa: BLE001 — identity stack must never break the child
        log.warning("camera process %s: identity init failed", spec.camera_id, exc_info=True)


def _drain_commands(worker: Any, cmd_q: Any) -> None:
    """Apply ``(method, args, kwargs)`` commands to *worker* until the stop sentinel.

    The supervisor-side handle proxies every method call the threaded supervisor
    makes (enable_tracking, set_target, ptz_nudge, set_features, update_config, …)
    as a command tuple; here we just re-dispatch it to the real worker, so the
    worker code is identical to the threaded path.
    """
    while True:
        try:
            name, args, kwargs = cmd_q.get()
        except Exception:  # noqa: BLE001 — queue closed → shut down
            return
        if name == _STOP:
            return
        method = getattr(worker, name, None)
        if not callable(method):
            log.debug("camera process: unknown command %r", name)
            continue
        try:
            method(*args, **kwargs)
        except Exception:  # noqa: BLE001 — a bad command must not kill the process
            log.warning("camera process command %s failed", name, exc_info=True)


class ProcessWorkerHandle:
    """Supervisor-side stand-in for a :class:`CameraWorker` running in a child process.

    Exposes the **same method surface** the supervisor calls on an in-process
    worker, but each call is serialized onto the child's command queue instead of
    executed inline — so ``Supervisor`` needs only a different factory, not new
    routing.  Telemetry + identity events flow back over queues and are forwarded
    to the same callbacks the threaded path uses, via a small daemon drain thread.
    """

    def __init__(
        self,
        camera_id: str,
        config: CameraConfig,
        on_telemetry: Any,
        *,
        db_path: str,
        detector_tier: str = "auto",
        unified_pose: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.config = config
        self.shm_name = f"cam_{camera_id[:8]}_preview"  # must match CameraWorker's
        self._on_telemetry = on_telemetry
        self._on_identity: Any | None = None
        self._db_path = db_path
        self._detector_tier = detector_tier
        self._unified_pose = unified_pose

        # Buffered pre-start config (applied via WorkerSpec so the child sees it
        # BEFORE it builds inference stacks); post-start changes go via commands.
        self._features: dict[str, bool] = {}
        self._defer_inference = False
        self._started = False

        self._proc: Any | None = None
        self._cmd_q: Any | None = None
        self._telemetry_q: Any | None = None
        self._identity_q: Any | None = None
        self._drain_thread: threading.Thread | None = None
        self._drain_stop = threading.Event()

    # ── supervisor-facing injection setters (pre-start config) ──────────────────
    # The shared in-process pool/service can't cross to a child — it builds its
    # own — so these are no-ops; the data they'd carry is reconstructed there.

    def set_inference_pool(self, _pool: Any) -> None:
        return None

    def set_identity_service(self, _service: Any) -> None:
        return None

    def set_identity_callback(self, callback: Any) -> None:
        self._on_identity = callback

    def set_features(self, features: dict[str, bool] | None) -> None:
        feats = dict(features or {})
        if not self._started:
            self._features = feats
        else:
            self._send("set_features", (feats,))

    def set_inference_start_paused(self, paused: bool) -> None:
        if not self._started:
            self._defer_inference = bool(paused)
        else:
            self._send("set_inference_start_paused", (bool(paused),))

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return
        spec = WorkerSpec(
            camera_id=self.camera_id,
            config=self.config,
            db_path=self._db_path,
            detector_tier=self._detector_tier,
            unified_pose=self._unified_pose,
            features=dict(self._features),
            defer_inference=self._defer_inference,
        )
        ctx = mp.get_context("spawn")
        self._cmd_q = ctx.Queue()
        self._telemetry_q = ctx.Queue()
        self._identity_q = ctx.Queue()
        self._proc = ctx.Process(
            target=run_camera_process,
            args=(spec, self._cmd_q, self._telemetry_q, self._identity_q),
            name=f"camproc-{self.camera_id[:8]}",
            daemon=True,
        )
        self._proc.start()
        self._started = True
        self._drain_stop.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_events,
            name=f"camproc-drain-{self.camera_id[:8]}",
            daemon=True,
        )
        self._drain_thread.start()

    def is_alive(self) -> bool:
        return bool(self._proc is not None and self._proc.is_alive())

    @property
    def is_running(self) -> bool:
        return self.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        self._drain_stop.set()
        if self._cmd_q is not None:
            try:
                self._cmd_q.put((_STOP, (), {}))
            except Exception:  # noqa: BLE001
                pass
        proc = self._proc
        if proc is not None:
            try:
                proc.join(timeout=timeout)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                log.debug("camera process %s join/terminate failed", self.camera_id, exc_info=True)
        self._proc = None
        self._started = False

    # ── command proxying (every method the supervisor calls on a worker) ────────

    def _send(self, name: str, args: tuple = (), kwargs: dict | None = None) -> None:
        q = self._cmd_q
        if q is None:
            return
        try:
            q.put((name, tuple(args), dict(kwargs or {})))
        except Exception:  # noqa: BLE001 — child gone / queue closed
            log.debug("camera process %s: command %s dropped", self.camera_id, name)

    def enable_tracking(self, enabled: bool) -> None:
        self._send("enable_tracking", (bool(enabled),))

    def set_target(self, track_id: Any) -> None:
        self._send("set_target", (track_id,))

    def set_target_identity(self, identity_id: Any) -> None:
        self._send("set_target_identity", (identity_id,))

    def enroll_track(self, *args: Any) -> None:
        self._send("enroll_track", args)

    def ptz_nudge(self, pan: float, tilt: float, zoom: float) -> None:
        self._send("ptz_nudge", (float(pan), float(tilt), float(zoom)))

    def ptz_home(self) -> None:
        self._send("ptz_home")

    def ptz_menu(self) -> None:
        self._send("ptz_menu")

    def set_target_fps(self, fps: float) -> None:
        self._send("set_target_fps", (float(fps),))

    def save_ptz_preset(self, slot: int) -> None:
        self._send("save_ptz_preset", (int(slot),))

    def recall_ptz_preset(self, slot: int) -> None:
        self._send("recall_ptz_preset", (int(slot),))

    def update_config(self, config: CameraConfig) -> None:
        self.config = config
        self._send("update_config", (config,))

    def refresh_detector_from_pool(self) -> None:
        self._send("refresh_detector_from_pool")

    def reload_inference_models(self) -> None:
        self._send("reload_inference_models")

    def release_inference_models(self, *, wait: float = 0.0) -> None:
        # Best-effort across the process boundary (the on-disk model-cache mutation
        # retry in models.py covers any residual lock); the wait is not honoured.
        self._send("release_inference_models", (), {"wait": 0.0})

    # ── child → parent event pump ───────────────────────────────────────────────

    def _drain_events(self) -> None:
        """Forward telemetry + identity events from the child to the parent callbacks."""
        import queue as _queue

        while not self._drain_stop.is_set():
            got = False
            tq = self._telemetry_q
            if tq is not None:
                try:
                    msg = tq.get(timeout=0.1)
                    got = True
                    if self._on_telemetry is not None:
                        try:
                            self._on_telemetry(msg)
                        except Exception:  # noqa: BLE001
                            log.debug("telemetry forward failed", exc_info=True)
                except _queue.Empty:
                    pass
                except Exception:  # noqa: BLE001 — queue closed on stop
                    return
            iq = self._identity_q
            if iq is not None:
                try:
                    rec = iq.get_nowait()
                    got = True
                    if self._on_identity is not None:
                        try:
                            self._on_identity(rec)
                        except Exception:  # noqa: BLE001
                            log.debug("identity forward failed", exc_info=True)
                except _queue.Empty:
                    pass
                except Exception:  # noqa: BLE001
                    pass
            if not got:
                continue


def process_per_camera_enabled() -> bool:
    """True when the experimental process-per-camera mode is switched on."""
    return os.environ.get("AUTOPTZ_PROCESS_PER_CAMERA", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
