"""Frame image providers: SHM → QImage bridge for QML live previews.

``ShmFrameProvider`` is registered with the QML engine as ``"image://frame/"``.
QML tiles request frames as::

    Image { source: "image://frame/" + cameraId + "?r=" + refreshTick }

The ``?r=N`` suffix forces Qt's image cache to refetch on each tick; the
provider strips it and looks up by bare camera_id.

``attach(camera_id, shm_name, h, w)`` must be called (from the main thread)
before frames will be served for a given camera.  ``detach(camera_id)`` cleans
up the ShmReader when a camera is removed.
"""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QImage
from PySide6.QtQuick import QQuickImageProvider

from autoptz.engine.runtime.shm import ShmReader

log = logging.getLogger(__name__)

_PLACEHOLDER_COLOR = QColor(20, 20, 40)   # dark navy — no-signal look


class ShmFrameProvider(QQuickImageProvider):
    """Serves latest camera frames from shared memory to QML Image items.

    Thread-safe: ``requestImage`` is called from the Qt render thread;
    ``attach``/``detach`` must be called from the main thread.
    The ShmReader is read-only and its ``latest()`` call is lock-free on the
    hot path, so concurrent access between the render thread and the engine
    thread is safe by design (see shm.py torn-read protection).
    """

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._readers: dict[str, ShmReader] = {}
        self._last_frame: dict[str, QImage] = {}
        self._placeholder = self._make_placeholder(640, 360)

    # ── registration helpers ──────────────────────────────────────────────────

    def attach(self, camera_id: str, shm_name: str, height: int, width: int) -> None:
        """Attach a ShmReader for *camera_id*.  Safe to call repeatedly."""
        self.detach(camera_id)
        try:
            reader = ShmReader(shm_name, height, width)
            self._readers[camera_id] = reader
            log.debug("ShmFrameProvider: attached %s → %s (%dx%d)", camera_id, shm_name, width, height)
        except Exception as exc:
            log.warning("ShmFrameProvider: could not attach %s (%s): %s", camera_id, shm_name, exc)

    def detach(self, camera_id: str) -> None:
        """Release the ShmReader for *camera_id* if one exists."""
        reader = self._readers.pop(camera_id, None)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
        self._last_frame.pop(camera_id, None)

    def detach_all(self) -> None:
        for cid in list(self._readers):
            self.detach(cid)

    # ── QQuickImageProvider ───────────────────────────────────────────────────

    def requestImage(self, id_str: str, size: QSize, requestedSize: QSize) -> QImage:
        """Called from the Qt render thread.  Returns a QImage immediately."""
        camera_id = id_str.split("?")[0]  # strip ?r=N cache-buster

        reader = self._readers.get(camera_id)
        if reader is None:
            return self._last_frame.get(camera_id, self._placeholder)

        result = reader.latest()
        if result is None:
            # No new frame — return last known good frame or placeholder
            return self._last_frame.get(camera_id, self._placeholder)

        _header, frame = result  # frame: (H, W, 3) uint8 BGR
        img = self._bgr_to_qimage(frame)
        self._last_frame[camera_id] = img
        return img

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bgr_to_qimage(bgr: np.ndarray) -> QImage:  # type: ignore[type-arg]
        """Convert BGR numpy frame to RGB QImage (zero-copy path where possible)."""
        rgb = bgr[..., ::-1].copy()   # BGR→RGB; copy ensures data ownership
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return img.copy()  # detach from numpy buffer

    @staticmethod
    def _make_placeholder(width: int, height: int) -> QImage:
        img = QImage(width, height, QImage.Format.Format_RGB888)
        img.fill(_PLACEHOLDER_COLOR)
        return img
