"""Shared-memory frame source for the Qt Widgets camera tiles.

Feeds decoded preview frames to the camera tiles.  Camera-tile
widgets pull the latest frame for their camera in their repaint timer and paint
it with ``QPainter`` — no image cache, no URL cache-buster, no crossfade.

``attach(camera_id, shm_name, height, width)`` records the *intent* to read a
camera's preview segment; the :class:`ShmReader` is opened lazily on the first
``latest_qimage`` that succeeds (self-healing — the writer's segment need not
exist yet), then cached.  ``detach`` releases it when a camera is removed.

All methods run on the GUI thread (``attach``/``detach`` are delivered via a
queued connection in ``app.py``; ``latest_qimage`` is called from the tile's
paint timer), but a lock guards the maps so ordering is never a hazard.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PySide6.QtGui import QImage

from autoptz.engine.runtime.shm import ShmReader

log = logging.getLogger(__name__)


def bgr_to_qimage(bgr: np.ndarray) -> QImage:  # type: ignore[type-arg]
    """Convert a BGR numpy frame to an owned :class:`QImage` that renders correctly.

    Wraps the BGR bytes directly with ``Format_BGR888`` (no channel-reversal copy)
    and ``.copy()`` to detach from the numpy buffer — halving the per-frame
    GUI-thread copy work versus reversing BGR→RGB first.  The rendered pixels are
    identical; only the redundant copy is gone.
    """
    if not bgr.flags["C_CONTIGUOUS"]:
        bgr = np.ascontiguousarray(bgr)  # no-op for the normal (already-packed) path
    h, w = bgr.shape[:2]
    img = QImage(bgr.data, w, h, w * 3, QImage.Format.Format_BGR888)
    return img.copy()  # detach from the numpy buffer


class ShmFrameSource:
    """Registry of per-camera shm readers serving the latest frame as a QImage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._readers: dict[str, ShmReader] = {}
        # camera_id → (shm_name, height, width) intents awaiting a lazy open.
        self._intents: dict[str, tuple[str, int, int]] = {}
        self._last: dict[str, QImage] = {}

    # ── registration ───────────────────────────────────────────────────────────

    def attach(self, camera_id: str, shm_name: str, height: int, width: int) -> None:
        """Record intent to serve *camera_id* from *shm_name* (h×w)."""
        self.detach(camera_id)
        with self._lock:
            self._intents[camera_id] = (shm_name, int(height), int(width))
        log.debug("ShmFrameSource: intent %s → %s (%dx%d)", camera_id, shm_name, width, height)

    def detach(self, camera_id: str) -> None:
        with self._lock:
            self._intents.pop(camera_id, None)
            reader = self._readers.pop(camera_id, None)
            self._last.pop(camera_id, None)
        if reader is not None:
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass

    def detach_all(self) -> None:
        with self._lock:
            ids = list(self._readers) + list(self._intents)
        for cid in ids:
            self.detach(cid)

    def is_known(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._readers or camera_id in self._intents

    # ── frame access ───────────────────────────────────────────────────────────

    def latest_qimage(self, camera_id: str) -> QImage | None:
        """Return the freshest frame for *camera_id*, or the last-good, or None.

        Lazily opens the reader from the recorded intent; returns ``None`` while
        no frame has ever arrived (the tile shows its "No Signal" state then).
        Never raises.
        """
        with self._lock:
            reader = self._readers.get(camera_id)
            if reader is None:
                reader = self._try_open_reader_locked(camera_id)
            last = self._last.get(camera_id)

        if reader is None:
            return last

        try:
            result = reader.latest()
        except Exception:  # noqa: BLE001
            return last
        if result is None:
            return last  # no new frame this tick — keep the last one

        _header, frame = result  # frame: (H, W, 3) uint8 BGR
        img = bgr_to_qimage(frame)
        with self._lock:
            self._last[camera_id] = img
        return img

    def _try_open_reader_locked(self, camera_id: str) -> ShmReader | None:
        """Open + cache a reader from intent (caller holds the lock).  None until ready."""
        intent = self._intents.get(camera_id)
        if intent is None:
            return None
        shm_name, height, width = intent
        try:
            reader = ShmReader(shm_name, height, width)
        except Exception:  # noqa: BLE001 — writer segment not created yet
            return None
        self._readers[camera_id] = reader
        log.debug(
            "ShmFrameSource: opened reader %s → %s (%dx%d)", camera_id, shm_name, width, height
        )
        return reader
