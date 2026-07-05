"""Latest-frame output pump — sends vcam/NDI frames OFF the capture thread.

The capture loop's only job is keeping frame cadence; the output sinks
(virtual camera, NDI sender) cost real milliseconds per frame (BGR→RGBA
conversion, resize, SDK hand-off), and paying them inline showed up as random
frame drops whenever the machine was busy. The pump owns a single "latest
frame" slot: the capture thread just parks the newest frame (cheap) and the
pump thread converts + sends at its own pace. If the pump is still busy when
the next frame arrives, the older pending frame is REPLACED — outputs always
show the newest frame and the capture thread never waits.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


class OutputSender:
    """Background sender for the per-camera output sinks (vcam / NDI)."""

    def __init__(self, name: str = "output") -> None:
        self._cond = threading.Condition()
        self._pending: tuple[NDArray[np.uint8], list[Any]] | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name=f"{name}-output-sender", daemon=True)
        self._thread.start()

    def submit(self, frame: NDArray[np.uint8], sinks: list[Any]) -> None:
        """Park *frame* for delivery to *sinks* (each needs ``send_bgr``).

        Drop-oldest: an undelivered previous frame is replaced, never queued —
        the slot holds at most one frame so memory and latency stay bounded and
        the capture thread returns immediately.
        """
        if not sinks:
            return
        with self._cond:
            if self._stop:
                return
            self._pending = (frame, list(sinks))
            self._cond.notify()

    def close(self, timeout: float = 2.0) -> None:
        """Stop the pump thread (idempotent). Sinks are closed by their owner."""
        with self._cond:
            self._stop = True
            self._pending = None
            self._cond.notify_all()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)

    # ── internals ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                frame, sinks = self._pending  # type: ignore[misc]
                self._pending = None
            for sink in sinks:
                try:
                    sink.send_bgr(frame)
                except Exception:  # noqa: BLE001 — output must never kill the pump
                    log.debug("output sink send failed", exc_info=True)
