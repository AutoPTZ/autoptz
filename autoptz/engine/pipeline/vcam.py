"""Virtual-camera output sink (optional pyvirtualcam). No-op when unavailable."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

_PVC_AVAILABLE: bool | None = None


def pick_interpolation(src_size: tuple[int, int], dst_size: tuple[int, int]) -> int:
    """Choose the best ``cv2`` interpolation for a resize from *src* to *dst*.

    Both sizes are ``(width, height)``. The Center-Stage crop is almost always
    *upscaled* to the output (a window of the frame blown up), which looks soft
    under ``INTER_LINEAR`` — so:

    * **Up** (dst larger in either dimension) → ``INTER_CUBIC`` (sharper).
    * **Down** (dst smaller in both dimensions) → ``INTER_AREA`` (best for shrink).
    * **~1:1** → ``INTER_LINEAR`` (cheap, no quality loss when not scaling).
    """
    import cv2

    sw, sh = src_size
    dw, dh = dst_size
    if dw == sw and dh == sh:
        return int(cv2.INTER_LINEAR)
    # If we are growing in *either* dimension, favour upscale sharpness.
    if dw > sw or dh > sh:
        return int(cv2.INTER_CUBIC)
    return int(cv2.INTER_AREA)


def _probe_pyvirtualcam() -> bool:
    global _PVC_AVAILABLE
    if _PVC_AVAILABLE is None:
        try:
            import pyvirtualcam  # noqa: F401

            _PVC_AVAILABLE = True
        except Exception:  # noqa: BLE001
            _PVC_AVAILABLE = False
    return bool(_PVC_AVAILABLE)


class VirtualCamSink:
    def __init__(self, width: int, height: int, fps: float = 30.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self._cam = None
        self.available = False
        if not _probe_pyvirtualcam():
            return
        try:
            import pyvirtualcam

            self._cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=max(1.0, fps),
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            self.available = True
        except Exception:  # noqa: BLE001 — no virtual-cam driver installed, etc.
            log.info("virtual camera unavailable; Center Stage output disabled", exc_info=True)
            self._cam = None
            self.available = False

    def send_bgr(self, frame: NDArray[np.uint8]) -> None:
        if self._cam is None:
            return
        try:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                import cv2

                interp = pick_interpolation(
                    (int(frame.shape[1]), int(frame.shape[0])), (self.width, self.height)
                )
                frame = cv2.resize(frame, (self.width, self.height), interpolation=interp)
            self._cam.send(frame)
        except Exception:  # noqa: BLE001 — never let output break the pipeline
            log.debug("virtual camera send failed", exc_info=True)

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.close()
            finally:
                self._cam = None
                self.available = False
