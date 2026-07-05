"""NDI output sink — publish the Center Stage feed as a network NDI source.

Mirrors :class:`autoptz.engine.pipeline.vcam.VirtualCamSink` but sends over the
network instead of a local virtual-camera device, so ANY computer on the LAN can
receive the framed feed with a free NDI receiver (NDI Tools, OBS, vMix, a
monitor). Uses cyndilib's ``Sender`` — the same SDK already shipped for NDI input
(so no new dependency, no license conflict). Import-guarded: cyndilib is absent in
the repo .venv / CI, where this degrades to an unavailable no-op.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

try:  # import-guard: cyndilib is conda-only, absent in .venv/CI
    from cyndilib.sender import Sender as _Sender
    from cyndilib.video_frame import VideoSendFrame as _VideoSendFrame
    from cyndilib.wrapper import FourCC as _FourCC

    _CYNDILIB_OK = True
except Exception:  # noqa: BLE001 — any import failure → feature unavailable
    _Sender = None  # type: ignore[assignment,misc]
    _VideoSendFrame = None  # type: ignore[assignment,misc]
    _FourCC = None  # type: ignore[assignment,misc]
    _CYNDILIB_OK = False


def ndi_output_available() -> bool:
    """True when cyndilib is importable so an NDI output can be created."""
    return _CYNDILIB_OK


def bgr_to_rgba_flat(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a contiguous BGR ``(H, W, 3)`` frame to a flat RGBA buffer for NDI."""
    h, w = frame.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = frame[..., 2]
    rgba[..., 1] = frame[..., 1]
    rgba[..., 2] = frame[..., 0]
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba).ravel()


class NDISendSink:
    """Send BGR frames as an NDI source named *name*. No-op when cyndilib is absent."""

    def __init__(self, width: int, height: int, name: str, fps: float = 30.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.ndi_name = name
        self._sender = None
        self.available = False
        if not _CYNDILIB_OK:
            return
        try:
            self._sender = _Sender(ndi_name=name, clock_video=True)
            vf = _VideoSendFrame()
            vf.set_resolution(self.width, self.height)
            vf.set_fourcc(_FourCC.RGBA)
            vf.set_frame_rate(int(round(max(1.0, fps))))
            self._sender.set_video_frame(vf)
            self._sender.open()
            self.available = True
        except Exception:  # noqa: BLE001 — NDI runtime missing / name clash, etc.
            log.info("NDI output unavailable; Center Stage NDI feed disabled", exc_info=True)
            self._sender = None
            self.available = False

    def send_bgr(self, frame: NDArray[np.uint8]) -> None:
        if self._sender is None:
            return
        try:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                import cv2

                from autoptz.engine.pipeline.vcam import pick_interpolation

                interp = pick_interpolation(
                    (int(frame.shape[1]), int(frame.shape[0])), (self.width, self.height)
                )
                frame = cv2.resize(frame, (self.width, self.height), interpolation=interp)
            self._sender.write_video(bgr_to_rgba_flat(frame))
        except Exception:  # noqa: BLE001 — never let output break the pipeline
            log.debug("NDI output send failed", exc_info=True)

    def close(self) -> None:
        if self._sender is not None:
            try:
                self._sender.close()
            finally:
                self._sender = None
                self.available = False
