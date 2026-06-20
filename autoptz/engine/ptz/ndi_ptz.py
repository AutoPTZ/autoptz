"""NDI PTZ backend via cyndilib.

Requires the NDI SDK runtime and the cyndilib package
(listed in requirements/macos.txt / requirements/gpu-nvidia.txt).
Degrades gracefully with a clear ImportError if cyndilib is absent.

NDI PTZ does not expose absolute pan/tilt position queries;
get_position() always returns None.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from autoptz.engine.ptz.base import PTZBackend, PTZCaps, PTZState

log = logging.getLogger(__name__)


def _require_cyndilib() -> Any:
    try:
        import cyndilib

        return cyndilib
    except ImportError as exc:
        raise ImportError(
            "cyndilib is required for NDI PTZ control.  "
            "Install it and the NDI SDK runtime: "
            "pip install cyndilib"
        ) from exc


class NDIPTZBackend(PTZBackend):
    """NDI PTZ backend wrapping cyndilib's Receiver PTZ API.

    Args:
        receiver: A ``cyndilib.Receiver`` instance already connected to an NDI source.
                  The caller owns the receiver lifetime; close() does NOT destroy it.
    """

    def __init__(self, receiver: Any) -> None:
        super().__init__()
        _require_cyndilib()  # validate import before any work
        self._recv = receiver
        self._lock = threading.Lock()
        self.caps = PTZCaps(
            continuous_pan_tilt=True,
            continuous_zoom=True,
            native_presets=True,
            query_position=False,
            absolute_pan_tilt=False,
            absolute_zoom=True,
        )
        log.info("NDIPTZBackend ready (receiver=%r)", receiver)

    # ── PTZBackend interface ──────────────────────────────────────────────────

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        with self._lock:
            # cyndilib: recv_ptz_pan_tilt_speed(recv, pan_speed, tilt_speed)
            # NDI convention: positive pan = right, positive tilt = up
            self._recv.recv_ptz_pan_tilt_speed(pan, tilt)
            if abs(zoom) > 1e-4:
                self._recv.recv_ptz_zoom_speed(zoom)
            else:
                self._recv.recv_ptz_zoom_speed(0.0)

    def move_absolute(self, pan: float, tilt: float, zoom: float) -> None:
        with self._lock:
            # NDI absolute: pan/tilt [0,1] (0.5=centre), zoom [0,1]
            # Convert from our [-1,1] / [0,1]
            ndi_pan = (pan + 1.0) / 2.0
            ndi_tilt = (tilt + 1.0) / 2.0
            self._recv.recv_ptz_pan_tilt(ndi_pan, ndi_tilt)
            self._recv.recv_ptz_zoom(zoom)

    def stop(self) -> None:
        with self._lock:
            try:
                self._recv.recv_ptz_pan_tilt_speed(0.0, 0.0)
                self._recv.recv_ptz_zoom_speed(0.0)
            except Exception:
                pass

    def get_position(self) -> PTZState | None:
        return None  # NDI PTZ has no standard position inquiry

    def goto_preset(self, idx: int) -> None:
        with self._lock:
            self._recv.recv_ptz_preset_recall(idx, 1.0)

    def save_preset(self, idx: int) -> None:
        with self._lock:
            self._recv.recv_ptz_preset_store(idx)

    def home(self) -> None:
        """Best-effort NDI home.

        NDI PTZ has no dedicated "home" command.  Many NDI cameras map preset 0
        to the home/default view, so we recall preset 0 if the receiver exposes
        the call; otherwise this is a safe no-op.
        """
        with self._lock:
            try:
                self._recv.recv_ptz_preset_recall(0, 1.0)
            except Exception:
                log.debug("NDI home (preset 0 recall) unsupported/failed", exc_info=True)

    def osd_menu(self) -> None:
        """NDI PTZ has no on-screen-display menu command — safe no-op."""
        log.debug("NDI backend has no OSD-menu command; ignoring osd_menu()")

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        log.info("NDIPTZBackend closed (receiver not destroyed — caller owns it)")
