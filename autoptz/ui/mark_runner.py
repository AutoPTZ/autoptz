"""AutoPTZ Mark — the GUI-side ramp controller (worker thread + queued signals).

``MarkRampController`` runs the headless :class:`BenchmarkRunner` (driven by a
``sample_fn``) on a worker :class:`QThread` and re-emits progress / per-step /
final-result / error as Qt signals so the GUI thread can update the HUD.  The
``sample_factory`` seam lets tests inject a scripted ``sample_fn(n) -> list[float]``
with zero real inference (the default factory builds a real ``_SupervisorSampler``).

Cancellation is cooperative: :meth:`stop` sets a flag that :meth:`_on_step`
observes.  ``BenchmarkRunner`` swallows exceptions raised from its ``on_step``
callback (a progress callback must never abort the run), so a cancel cannot tear
the runner down mid-step; the flag instead short-circuits the *next* sample by
making it raise, which surfaces as ``error("cancelled")``.  The window treats a
"cancelled" error specially.  This keeps ``BenchmarkRunner`` untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from autoptz.benchmark.profiles import get_profile
from autoptz.benchmark.runner import BenchmarkRunner, StepResult

log = logging.getLogger(__name__)


class _MarkCancelled(Exception):
    """Internal: the user pressed Stop."""


def _default_sample_factory(
    profile: str,
    dwell_s: float,
    *,
    client: Any | None = None,
    supervisor_factory: Callable[[Any, Any], Any] | None = None,
) -> Callable[[int], list[float]]:
    """Build a real _SupervisorSampler and return its sample_fn(n).

    When *client* is supplied (the Mark window passes the SAME EngineClient its
    CameraWall is bound to), the sampler registers its synthetic cameras on that
    client so the wall's tiles render and frames flow during the ramp.
    """
    from autoptz.benchmark.runner import _SupervisorSampler

    sampler = _SupervisorSampler(
        get_profile(profile),
        client=client,
        supervisor_factory=supervisor_factory,
    )

    def sample_fn(n: int) -> list[float]:
        return sampler.sample(n, dwell_s=dwell_s, max_ticks=2000, tick_sleep_s=0.005)

    sample_fn._sampler = sampler  # type: ignore[attr-defined]  # for close()
    return sample_fn


class MarkRampController(QObject):
    progress = Signal(int, int, float)  # step_index (1-based), total, eta_s
    step_completed = Signal(object)  # StepResult
    finished = Signal(object)  # BenchmarkResult
    error = Signal(str)

    def __init__(
        self,
        *,
        profile: str = "full",
        floor_fps: float = 24.0,
        max_cameras: int = 16,
        dwell_s: float = 15.0,
        sample_factory: Callable[[], Callable[[int], list[float]]] | None = None,
        client: Any | None = None,
        supervisor_factory: Callable[[Any, Any], Any] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._profile = profile
        self._floor = float(floor_fps)
        self._max = max(1, int(max_cameras))
        self._dwell = max(0.0, float(dwell_s))
        self._sample_factory = sample_factory
        # The default sampler registers its synthetic cameras on this client; the
        # window passes the SAME client its CameraWall is bound to so tiles render.
        self._client = client
        self._supervisor_factory = supervisor_factory
        self._cancel = False
        self._thread: QThread | None = None

    def start(self) -> None:
        self._cancel = False
        # Standard worker-object pattern: the controller is moved INTO the thread,
        # so the thread must be unparented (a parented QObject cannot moveToThread).
        # We hold a strong ref in ``self._thread`` and free it via ``deleteLater``
        # only after it has truly finished — never destroy a still-running QThread
        # (that aborts with "QThread: Destroyed while thread is still running").
        thread = QThread()
        self.moveToThread(thread)
        thread.started.connect(self._run)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        self._cancel = True

    def wait(self, timeout_ms: int = 5000) -> bool:
        """Block until the worker thread has finished (for clean teardown).

        Returns True if the thread finished within *timeout_ms* (or was never
        started).  Callers on the GUI thread should pump events while waiting if
        they also need queued finished/error signals delivered first.
        """
        thread = self._thread
        if thread is None:
            return True
        return bool(thread.wait(timeout_ms))

    def _build_sample_fn(self) -> Callable[[int], list[float]]:
        if self._sample_factory is not None:
            return self._sample_factory()
        return _default_sample_factory(
            self._profile,
            self._dwell,
            client=self._client,
            supervisor_factory=self._supervisor_factory,
        )

    def _on_step(self, step: StepResult) -> None:
        self.step_completed.emit(step)
        if self._cancel:
            raise _MarkCancelled()

    def _run(self) -> None:
        sample_fn: Callable[[int], list[float]] | None = None
        try:
            sample_fn = self._build_sample_fn()
            prof = get_profile(self._profile)
            inner = sample_fn

            def wrapped(n: int) -> list[float]:
                if self._cancel:
                    raise _MarkCancelled()
                self.progress.emit(n, self._max, max(0.0, (self._max - n) * self._dwell))
                return inner(n)

            runner = BenchmarkRunner(
                prof,
                floor_fps=self._floor,
                max_cameras=self._max,
                dwell_s=self._dwell,
                sample_fn=wrapped,
                on_step=self._on_step,
            )
            result = runner.run()
            self.finished.emit(result)
        except _MarkCancelled:
            self.error.emit("cancelled")
        except Exception as exc:  # noqa: BLE001 — surfaced to the GUI, never crashes
            log.exception("AutoPTZ Mark ramp failed")
            self.error.emit(str(exc) or type(exc).__name__)
        finally:
            sampler = getattr(sample_fn, "_sampler", None) if sample_fn else None
            if sampler is not None:
                try:
                    sampler.close()
                except Exception:  # noqa: BLE001
                    log.debug("sampler close failed", exc_info=True)
            thread = self._thread
            if thread is not None:
                thread.quit()
