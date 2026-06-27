"""AutoPTZ Mark — the headless ramp runner + score math.

The runner ramps synthetic cameras through the real engine pipeline and reports
the most cameras the machine can sustain above an fps floor, plus a single
weighted score.  The ramp/scoring control logic is isolated behind a
``sample_fn(n) -> list[float]`` seam: the real benchmark supplies a sampler that
drives a ``Supervisor`` (see ``run_benchmark``); unit tests supply a deterministic
one so the math is verified with no real inference.

Throughput is measured by running the full pipeline at the configured cadence on
synthetic frames.  The detector runs (and incurs its inference cost) even when it
finds nothing, so the number is valid without a bundled person asset.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from autoptz.benchmark.profiles import BenchmarkProfile

log = logging.getLogger(__name__)

_NOMINAL_FPS = 30.0  # score normaliser: sustained fps / 30 fps target


@dataclass(frozen=True)
class StepResult:
    """One ramp step: N cameras run for the dwell, with observed per-camera fps."""

    cameras: int
    min_fps: float
    mean_fps: float
    per_camera_fps: list[float] = field(default_factory=list)
    sustained: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "cameras": self.cameras,
            "min_fps": round(self.min_fps, 2),
            "mean_fps": round(self.mean_fps, 2),
            "per_camera_fps": [round(f, 2) for f in self.per_camera_fps],
            "sustained": self.sustained,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """The full ramp outcome + the AutoPTZ Mark score."""

    profile: str
    weight: float
    floor_fps: float
    max_cameras: int
    sustained_cameras: int
    min_fps_at_sustained: float
    score: float
    steps: list[StepResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "weight": self.weight,
            "floor_fps": self.floor_fps,
            "max_cameras": self.max_cameras,
            "sustained_cameras": self.sustained_cameras,
            "min_fps_at_sustained": round(self.min_fps_at_sustained, 2),
            "score": self.score,
            "steps": [s.to_dict() for s in self.steps],
        }

    def summary(self) -> str:
        return (
            f"AutoPTZ Mark [{self.profile}]: score {self.score:.2f} — sustained "
            f"{self.sustained_cameras} camera(s) @ >={self.floor_fps:.0f} fps "
            f"(min {self.min_fps_at_sustained:.1f} fps at that count)."
        )


class BenchmarkRunner:
    """Ramp synthetic cameras until the min sustained fps drops below the floor."""

    def __init__(
        self,
        profile: BenchmarkProfile,
        *,
        floor_fps: float = 24.0,
        max_cameras: int = 16,
        dwell_s: float = 20.0,
        sample_fn: Callable[[int], list[float]],
        on_step: Callable[[StepResult], None] | None = None,
    ) -> None:
        self._profile = profile
        self._floor = float(floor_fps)
        self._max_cameras = max(1, int(max_cameras))
        self._dwell_s = max(0.0, float(dwell_s))
        self._sample_fn = sample_fn
        self._on_step = on_step

    def run(self) -> BenchmarkResult:
        steps: list[StepResult] = []
        sustained_cameras = 0
        min_fps_at_sustained = 0.0

        for cameras in range(1, self._max_cameras + 1):
            per_camera = [float(f) for f in self._sample_fn(cameras)]
            if per_camera:
                min_fps = min(per_camera)
                mean_fps = sum(per_camera) / len(per_camera)
            else:
                min_fps = 0.0
                mean_fps = 0.0
            sustained = min_fps >= self._floor
            step = StepResult(
                cameras=cameras,
                min_fps=min_fps,
                mean_fps=mean_fps,
                per_camera_fps=per_camera,
                sustained=sustained,
            )
            steps.append(step)
            if self._on_step is not None:
                try:
                    self._on_step(step)
                except Exception:  # noqa: BLE001 — a progress callback must not abort the run
                    log.debug("benchmark on_step callback failed", exc_info=True)
            if sustained:
                sustained_cameras = cameras
                min_fps_at_sustained = min_fps
            else:
                break  # this count failed the floor → ramp is done

        score = round(
            sustained_cameras * (min_fps_at_sustained / _NOMINAL_FPS) * self._profile.weight,
            2,
        )
        return BenchmarkResult(
            profile=self._profile.name,
            weight=self._profile.weight,
            floor_fps=self._floor,
            max_cameras=self._max_cameras,
            sustained_cameras=sustained_cameras,
            min_fps_at_sustained=min_fps_at_sustained,
            score=score,
            steps=steps,
        )


# ── real sampler: a headless Supervisor over N synthetic cameras ──────────────


def _default_fps_reader(client: Any, camera_id: str) -> float:
    """Read a camera's most recent telemetry fps from the client's model.

    ``CameraRecord.fps`` is a property over ``record.telemetry.fps`` (see
    ``autoptz/ui/list_models.py``), populated by ``CameraListModel.update_telemetry``
    on every ``push_telemetry``.  Adjust the attribute path here if that storage
    ever changes.
    """
    try:
        rec = client.cameraModel.get_record(camera_id)
    except Exception:  # noqa: BLE001
        return 0.0
    if rec is None:
        return 0.0
    return float(getattr(rec, "fps", 0.0) or 0.0)


def _add_synthetic_camera(
    client: Any,
    index: int,
    *,
    width: int = 0,
    height: int = 0,
    address: str = "anim",
) -> str:
    """Register one self-paced synthetic camera on the client's model.

    Done directly via a ``CameraRecord`` (not ``client.addCamera``, which infers a
    USB source from the URI scheme).  The 30 fps cap means the real worker paces
    the synthetic source so it never free-spins (~16000 fps would tear the shm
    triple-buffer).  A non-zero ``width``/``height`` (AutoPTZ Mark's resolution
    control) sizes the composed synthetic scene; 0 keeps the source default.

    ``address`` selects the synthetic content: the default ``"anim"`` draws moving
    synthetic people; a path to a video file (AutoPTZ Mark's bundled clip) makes
    the ``SyntheticAdapter`` loop that real clip instead (real decode, real people,
    no drawn overlay).
    """
    from autoptz.config.models import CameraConfig, SourceConfig
    from autoptz.ui.list_models import CameraRecord

    camera_id = str(uuid.uuid4())
    name = f"AutoPTZ Mark {index + 1}"
    addr = (address or "anim").strip() or "anim"
    cfg = CameraConfig(
        id=camera_id,
        name=name,
        source=SourceConfig(
            type="synthetic",
            address=addr,
            fps=30.0,
            width=int(width),
            height=int(height),
        ),
    )
    rec = CameraRecord(
        camera_id=camera_id,
        source_uri=f"synthetic://{addr}",
        display_name=name,
        camera_config=cfg,
    )
    client.cameraModel.add_camera(rec)
    return camera_id


def _default_supervisor_factory(client: Any, store: Any) -> Any:
    from autoptz.engine.supervisor import Supervisor

    return Supervisor(client, store=store)


class _SupervisorSampler:
    """Drives a headless Supervisor over N synthetic cameras and samples fps.

    The headful Mark window can **adopt** an already-built engine stack so a ramp
    reuses the ONE supervisor + pre-added cameras the Mark window already shows on
    the idle wall, instead of spinning up a second supervisor over a disjoint set
    of cameras (which doubled tiles and CPU).  Pass ``supervisor=`` (already
    primed) and ``cameras=`` (the pre-added camera ids) to adopt; ``adopted_started``
    says whether that supervisor's workers are already running so the first sample
    doesn't double-start it.
    """

    def __init__(
        self,
        profile: BenchmarkProfile,
        *,
        supervisor_factory: Callable[[Any, Any], Any] | None = None,
        client: Any | None = None,
        supervisor: Any | None = None,
        cameras: list[str] | None = None,
        adopted_started: bool = False,
        on_grow: Callable[[], str | None] | None = None,
    ) -> None:
        # When *client* is injected (the headful Mark window passes the SAME
        # EngineClient its CameraWall is bound to), the synthetic cameras land on
        # that client's model so tiles + frames actually render during the ramp.
        # The headless CLI passes None → we build a private client.
        self._profile = profile
        if client is None:
            from autoptz.ui.engine_client import EngineClient

            client = EngineClient()
        self._client = client
        # Adopt an existing supervisor (the Mark window's) when supplied so only
        # ONE stack ever runs; otherwise build a private one.
        self._adopted = supervisor is not None
        if supervisor is not None:
            self._sup = supervisor
        else:
            factory = supervisor_factory or _default_supervisor_factory
            store = getattr(client, "_store", None)
            self._sup = factory(self._client, store)
            self._sup.prime_features(dict(self._profile.features))
        # Pre-seed with adopted camera ids so _ensure_cameras never re-adds them.
        self._cameras: list[str] = list(cameras) if cameras else []
        self._started = bool(adopted_started)
        # One-time warmup gate (adopted path): the detector model loads (~8s) during
        # the first dwell, so the first measured step would otherwise read ~0 fps →
        # "0 sustained" and the ramp stops immediately.  ``_warmed`` is flipped True
        # after the first sample waits for frames to flow + the model to finish
        # loading, so step 1 measures STEADY STATE, not model-load warmup.
        self._warmed = False
        # 3DMark-style progressive ramp (adopted path only): when the ramp steps to
        # N and the wall holds fewer, call ``on_grow`` to add the next camera ONE AT
        # A TIME on the SAME client + supervisor.  ``on_grow`` returns the new id and
        # is expected to register it on the model AND spawn its worker (the Mark
        # factory's ``add_next_camera``).  None → no growth (fixed pre-added set).
        self._on_grow = on_grow

    @staticmethod
    def _drain_events() -> None:
        """Deliver queued telemetry signals from the worker thread.

        The real ``CameraWorker`` emits telemetry from its own thread, so
        ``EngineClient.push_telemetry`` marshals it onto the owning thread via a
        queued Qt signal.  Without draining the event loop the model never sees
        the update and every fps reads 0.  No-op when no ``QCoreApplication`` is
        running (the injected-fake-worker path emits synchronously).
        """
        from PySide6.QtCore import QCoreApplication

        app = QCoreApplication.instance()
        if app is not None:
            app.processEvents()

    def _ensure_cameras(self, n: int) -> None:
        while len(self._cameras) < n:
            self._cameras.append(_add_synthetic_camera(self._client, len(self._cameras)))

    @staticmethod
    def _dwell_observe(dwell_s: float) -> None:
        """Sleep the dwell while the external (GUI) pump drives the supervisor.

        Used by the adopted path: the Mark window's QTimer ticks the supervisor and
        delivers telemetry on the GUI thread, so the worker just waits before
        reading fps.  ``dwell_s == 0`` (tests) yields a single short settle so any
        already-queued telemetry has a chance to land.
        """
        time.sleep(max(0.0, dwell_s) if dwell_s > 0.0 else 0.01)

    def _warmup(
        self,
        reader: Callable[[Any, str], float],
        *,
        min_fps: float = 10.0,
        timeout_s: float = 25.0,
        settle_s: float = 1.0,
        poll_s: float = 0.2,
    ) -> None:
        """Block until the current cameras are warmed up (frames + model loaded).

        The detector model loads (~8s) during the FIRST measured dwell, so without
        this gate step 1 reads ~0 fps and the ramp stops immediately.  This polls
        the current cameras' telemetry fps (the GUI pump updates the model on the
        GUI thread; this runs on a worker thread, so we just poll + sleep) until the
        slowest camera clears ``min_fps`` (frames flowing AND the model loaded) or a
        ~``timeout_s`` budget elapses, then settles ``settle_s`` so the rolling
        average reflects steady state before the first dwell measures it.  One-shot
        (``_warmed``); ``dwell_s == 0`` tests never enter the adopted-warmup path
        with real cameras so this stays fast.
        """
        if self._warmed:
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            cams = self._cameras
            if cams:
                fps = [reader(self._client, cid) for cid in cams]
                if fps and min(fps) >= min_fps:
                    break
            time.sleep(max(0.0, poll_s))
        # Let the rolling fps average settle to steady state before the first dwell.
        time.sleep(max(0.0, settle_s))
        self._warmed = True

    def sample(
        self,
        n: int,
        *,
        dwell_s: float,
        max_ticks: int,
        tick_sleep_s: float,
        fps_reader: Callable[[Any, str], float] | None = None,
    ) -> list[float]:
        reader = fps_reader or _default_fps_reader
        if self._adopted:
            # 3DMark-style progressive ramp: grow the wall to N one camera at a time
            # (the wall started at 1).  ``on_grow`` adds the next synthetic/NDI camera
            # on the SAME client + supervisor and spawns its worker.
            if self._on_grow is not None:
                while len(self._cameras) < n:
                    cid = self._on_grow()
                    if cid is None:
                        break  # hit the camera cap
                    self._cameras.append(cid)
            # The Mark window's GUI pump (33 ms QTimer) is the SOLE driver of the
            # adopted supervisor — ticking it here too would race two threads on one
            # supervisor.  Before the FIRST dwell, wait out the one-time engine warmup
            # (model load + frames flowing) so step 1 measures steady state, not the
            # ~8s model load.  Then observe the pre-added cameras over the dwell and
            # read the first ``n`` cameras' fps (the ramp is a measurement window).
            self._warmup(reader)
            self._dwell_observe(dwell_s)
            return [reader(self._client, cid) for cid in self._cameras[:n]]
        had = len(self._cameras)
        self._ensure_cameras(n)
        if not self._started:
            # run_pump=False: we pump tick() ourselves so the dwell is bounded and
            # deterministic (no UI timer in a headless benchmark).
            self._sup.start(run_pump=False)
            self._started = True
        elif len(self._cameras) > had:
            # New cameras were appended this step.  The supervisor spawns workers
            # for the cameras present in the model at start() time, so restart it
            # to pick up the freshly added set.
            self._sup.stop()
            self._sup.start(run_pump=False)

        deadline = time.monotonic() + max(0.0, dwell_s)
        ticks = 0
        while ticks < max_ticks and (ticks == 0 or time.monotonic() < deadline):
            self._sup.tick()
            # Deliver any telemetry the worker thread queued onto this thread so
            # the model's fps reflects live frames (no-op for synchronous fakes).
            self._drain_events()
            ticks += 1
            if tick_sleep_s > 0.0:
                time.sleep(tick_sleep_s)
        # Final drain so the last queued telemetry lands before we read fps.
        self._drain_events()
        return [reader(self._client, cid) for cid in self._cameras[:n]]

    def close(self) -> None:
        # Never tear down an ADOPTED supervisor — the Mark window owns its
        # lifecycle (closeEvent → engine.stop).  Only stop a supervisor we built.
        if self._adopted:
            return
        try:
            self._sup.stop()
        except Exception:  # noqa: BLE001
            log.debug("benchmark supervisor stop failed", exc_info=True)


def run_benchmark(
    *,
    profile: str = "full",
    floor_fps: float = 24.0,
    max_cameras: int = 16,
    dwell_s: float = 20.0,
    json_path: str | None = None,
    supervisor_factory: Callable[[Any, Any], Any] | None = None,
    fps_reader: Callable[[Any, str], float] | None = None,
    max_ticks: int = 2000,
    tick_sleep_s: float = 0.005,
) -> int:
    """Run AutoPTZ Mark and print/save the report.  Returns a process exit code.

    0 on success, 2 on an unknown profile.  Prints to stdout deliberately — this
    is the CLI face of the benchmark.
    """
    from autoptz.benchmark.profiles import get_profile

    try:
        prof = get_profile(profile)
    except ValueError as exc:
        print(str(exc))
        return 2

    # The EngineClient + workers are QObjects and worker telemetry is marshalled
    # back via queued Qt signals, so a QCoreApplication must exist (and own the
    # current thread) for that delivery — and for the sampler's processEvents()
    # drain to have an event loop to pump.  Under the UI this already exists; from
    # the headless CLI we create a minimal one.  Reuse any existing instance.
    from PySide6.QtCore import QCoreApplication

    if QCoreApplication.instance() is None:
        _ = QCoreApplication(sys.argv[:1])

    print(f"AutoPTZ Mark — profile {prof.name!r} ({prof.description})")
    print(f"  floor {floor_fps:.0f} fps - max {max_cameras} cameras - dwell {dwell_s:.0f}s")

    sampler = _SupervisorSampler(prof, supervisor_factory=supervisor_factory)

    def sample_fn(n: int) -> list[float]:
        return sampler.sample(
            n,
            dwell_s=dwell_s,
            max_ticks=max_ticks,
            tick_sleep_s=tick_sleep_s,
            fps_reader=fps_reader,
        )

    def on_step(step: StepResult) -> None:
        mark = "ok " if step.sustained else "STOP"
        print(
            f"  [{mark}] {step.cameras:2d} cam(s): min {step.min_fps:5.1f} fps "
            f"- mean {step.mean_fps:5.1f} fps"
        )

    try:
        runner = BenchmarkRunner(
            prof,
            floor_fps=floor_fps,
            max_cameras=max_cameras,
            dwell_s=dwell_s,
            sample_fn=sample_fn,
            on_step=on_step,
        )
        result = runner.run()
    finally:
        sampler.close()

    print(result.summary())

    if json_path:
        import json as _json
        from pathlib import Path

        Path(json_path).write_text(_json.dumps(result.to_dict(), indent=2))
        print(f"  wrote: {json_path}")
    return 0


# ── Follow-up (NOT implemented here): GUI "AutoPTZ Mark" window ───────────────
# A future task can add an Engine > Benchmark menu action that runs run_benchmark
# on a worker QThread (never the GUI thread) and renders BenchmarkResult.steps as
# a live ramp chart + the final score.  The runner is already UI-free and returns
# a fully serialisable BenchmarkResult, so the GUI only needs to call sample_fn /
# run() off-thread and marshal StepResult updates back via a queued signal
# (mirroring EngineClient._set_startup_progress).  Out of scope for this plan.
