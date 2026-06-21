"""USB / serial VISCA backend.

Wraps a pyserial connection to any Sony/compatible VISCA camera (EVI-D100, etc.).
Normalized [-1, 1] pan/tilt/zoom maps to VISCA speed bytes 0x01–0x18 / 0x01–0x14 / 0x01–0x07.
Presets are native VISCA memory commands (81 01 04 3F 01/02 MM FF).
"""

from __future__ import annotations

import logging
from typing import Any

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


class ViscaUSBBackend(PTZBackend):
    """Serial VISCA PTZ backend.

    Args:
        port: Serial port string, e.g. ``"/dev/tty.usbserial-1420"`` or ``"COM3"``.
        baud: Baud rate (default 9600 for most Sony cameras).
        address: VISCA camera address 1–7 (rarely needs changing).
    """

    def __init__(self, port: str, baud: int = 9600, address: int = 1) -> None:
        super().__init__()
        import serial  # pyserial — already in requirements/base.txt

        self.caps = PTZCaps(
            continuous_pan_tilt=True,
            continuous_zoom=True,
            native_presets=True,
            query_position=False,
        )
        # address byte: camera 1 = 0x81, camera 2 = 0x82, …
        self._addr = 0x80 | (address & 0x07)
        self._ser: Any = serial.Serial(port, baud, timeout=0.1)
        log.info("ViscaUSB opened %s @ %d baud", port, baud)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _send(self, cmd: bytes) -> None:
        """Write VISCA command; drain any pending ACK/completion bytes first."""
        # Patch address byte (index 0) to match configured camera address
        patched = bytes([self._addr]) + cmd[1:]
        pending = self._ser.in_waiting
        if pending:
            self._ser.read(pending)
        self._ser.write(patched)

    # ── PTZBackend interface ──────────────────────────────────────────────────

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        self._send(visca_pantilt_cmd(pan, tilt))
        self._send(visca_zoom_cmd(zoom))

    def stop(self) -> None:
        self._send(visca_stop_cmd())
        self._send(visca_zoom_stop_cmd())

    def get_position(self) -> PTZState | None:
        return None  # serial VISCA has no reliable position inquiry on D100

    def goto_preset(self, idx: int) -> None:
        self._send(visca_preset_recall_cmd(idx))

    def save_preset(self, idx: int) -> None:
        self._send(visca_preset_set_cmd(idx))

    def home(self) -> None:
        """Drive camera to its optical home position."""
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
            self._ser.close()
        except Exception:
            pass
        log.info("ViscaUSB closed")
