"""VISCA over IP backend (TCP).

Supports two wire formats:
  ``"sony"`` — 8-byte header per Sony/Panasonic VISCA-over-IP spec (port 52381).
  ``"raw"``  — plain VISCA bytes over TCP; works with PTZOptics, BirdDog, Lumens.

Preset memory, zoom, and pan/tilt all use standard VISCA serial commands wrapped
in the chosen transport framing.  Position inquiry (InqPanTiltPos) is issued on
get_position() for cameras that answer it; others return None.
"""
from __future__ import annotations

import logging
import socket
import struct
import threading

from autoptz.engine.ptz.base import (
    PTZBackend,
    PTZCaps,
    PTZState,
    visca_home_cmd,
    visca_menu_cmd,
    visca_pantilt_cmd,
    visca_preset_recall_cmd,
    visca_preset_set_cmd,
    visca_stop_cmd,
    visca_zoom_cmd,
    visca_zoom_stop_cmd,
)

log = logging.getLogger(__name__)

# Sony VISCA-over-IP payload type for VISCA commands
_SONY_TYPE_CMD = 0x0100
# PanTiltPosInq: 81 09 06 12 FF
_INQ_PANTILT = bytes([0x81, 0x09, 0x06, 0x12, 0xFF])
# ZoomPosInq:   81 09 04 47 FF
_INQ_ZOOM = bytes([0x81, 0x09, 0x04, 0x47, 0xFF])


class ViscaIPBackend(PTZBackend):
    """VISCA-over-TCP PTZ backend.

    Args:
        host:    Camera IP address or hostname.
        port:    TCP port (default 52381 for Sony; PTZOptics uses 5678).
        mode:    ``"sony"`` or ``"raw"`` framing (default ``"raw"``).
        timeout: Socket connection/recv timeout in seconds.
    """

    def __init__(
        self,
        host: str,
        port: int = 52381,
        mode: str = "raw",
        timeout: float = 2.0,
    ) -> None:
        super().__init__()
        if mode not in ("sony", "raw"):
            raise ValueError(f"mode must be 'sony' or 'raw', got {mode!r}")
        self._mode = mode
        self._lock = threading.Lock()
        self._seq = 0  # VISCA-over-IP sequence counter (Sony mode)

        self.caps = PTZCaps(
            continuous_pan_tilt=True,
            continuous_zoom=True,
            native_presets=True,
            query_position=True,  # best-effort; cameras that don't answer return None
        )
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)
        log.info("ViscaIP connected to %s:%d (%s mode)", host, port, mode)

    # ── framing ───────────────────────────────────────────────────────────────

    def _frame(self, payload: bytes) -> bytes:
        if self._mode == "sony":
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            header = struct.pack(">HHI", _SONY_TYPE_CMD, len(payload), self._seq)
            return header + payload
        return payload  # raw: send plain VISCA bytes

    def _send(self, visca_cmd: bytes) -> None:
        with self._lock:
            self._sock.sendall(self._frame(visca_cmd))

    def _query(self, visca_inq: bytes, response_len: int) -> bytes | None:
        """Send inquiry and read fixed-length response; returns None on error."""
        try:
            with self._lock:
                self._sock.sendall(self._frame(visca_inq))
                if self._mode == "sony":
                    # Sony: 8-byte header before payload
                    hdr = self._sock.recv(8)
                    if len(hdr) < 8:
                        return None
                    _ptype, plen, _seq = struct.unpack(">HHI", hdr)
                    return self._sock.recv(plen)
                else:
                    return self._sock.recv(response_len)
        except Exception:
            return None

    # ── PTZBackend interface ──────────────────────────────────────────────────

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        self._send(visca_pantilt_cmd(pan, tilt))
        self._send(visca_zoom_cmd(zoom))

    def stop(self) -> None:
        self._send(visca_stop_cmd())
        self._send(visca_zoom_stop_cmd())

    def get_position(self) -> PTZState | None:
        # Pan/tilt inquiry returns: 90 50 0w 0w 0w 0w 0x 0x 0x 0x FF (11 bytes)
        resp = self._query(_INQ_PANTILT, 11)
        if resp is None or len(resp) < 11 or resp[0] != 0x90:
            return None
        try:
            # nibble-packed signed 16-bit values
            pan_raw = (resp[2] << 12) | (resp[3] << 8) | (resp[4] << 4) | resp[5]
            tilt_raw = (resp[6] << 12) | (resp[7] << 8) | (resp[8] << 4) | resp[9]
            # convert to signed
            if pan_raw > 0x7FFF:
                pan_raw -= 0x10000
            if tilt_raw > 0x7FFF:
                tilt_raw -= 0x10000
            # normalize to [-1, 1] assuming ±0x8700 pan range, ±0x4B00 tilt
            pan_n = _clamp(pan_raw / 0x8700, -1.0, 1.0)
            tilt_n = _clamp(tilt_raw / 0x4B00, -1.0, 1.0)
        except Exception:
            return None

        # Zoom inquiry
        zoom_resp = self._query(_INQ_ZOOM, 7)
        zoom_n = 0.0
        if zoom_resp is not None and len(zoom_resp) >= 7 and zoom_resp[0] == 0x90:
            try:
                z_raw = (zoom_resp[2] << 12) | (zoom_resp[3] << 8) | (zoom_resp[4] << 4) | zoom_resp[5]
                zoom_n = _clamp(z_raw / 0x4000, 0.0, 1.0)
            except Exception:
                pass

        return PTZState(pan=pan_n, tilt=tilt_n, zoom=zoom_n)

    def goto_preset(self, idx: int) -> None:
        self._send(visca_preset_recall_cmd(idx))

    def save_preset(self, idx: int) -> None:
        self._send(visca_preset_set_cmd(idx))

    def home(self) -> None:
        """Drive pan/tilt to the camera's optical home position."""
        self._send(visca_home_cmd())

    def osd_menu(self) -> None:
        """Toggle the camera's on-screen-display menu (no-op if unsupported)."""
        self._send(visca_menu_cmd())

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        log.info("ViscaIP closed")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
