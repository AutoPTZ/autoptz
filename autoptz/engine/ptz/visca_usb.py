"""USB / serial VISCA backend.

Wraps a pyserial connection to any Sony/compatible VISCA camera (EVI-D100, etc.).
Normalized [-1, 1] pan/tilt/zoom maps to VISCA speed bytes 0x01–0x18 / 0x01–0x14 / 0x01–0x07.
Presets are native VISCA memory commands (81 01 04 3F 01/02 MM FF).

Auto-reconnect: on SerialException or OSError the port is closed and a
ReconnectPolicy gate is consulted; if the backoff interval has elapsed a single
reconnect is attempted and the command is retried once.  Neither move_velocity
nor stop ever raise into the caller.

Safety (halt-on-reconnect): when a reconnect succeeds, a stop is always sent
FIRST before retrying any command.  This ensures a camera that was mid-pan on
disconnect halts rather than continuing a stale continuous-move.  Sending a
stop to an already-stopped camera is a safe no-op.
"""

from __future__ import annotations

import logging
import time
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
from autoptz.engine.ptz.reconnect import ReconnectPolicy

log = logging.getLogger(__name__)

# Import serial at module level so tests can monkeypatch visca_usb.serial
import serial  # noqa: E402  pyserial — already in requirements/base.txt


class ViscaUSBBackend(PTZBackend):
    """Serial VISCA PTZ backend.

    Args:
        port: Serial port string, e.g. ``"/dev/tty.usbserial-1420"`` or ``"COM3"``.
        baud: Baud rate (default 9600 for most Sony cameras).
        address: VISCA camera address 1–7 (rarely needs changing).
    """

    def __init__(self, port: str, baud: int = 9600, address: int = 1) -> None:
        super().__init__()
        self.caps = PTZCaps(
            continuous_pan_tilt=True,
            continuous_zoom=True,
            native_presets=True,
            query_position=False,
        )
        # address byte: camera 1 = 0x81, camera 2 = 0x82, …
        self._addr = 0x80 | (address & 0x07)
        self._port = port
        self._baud = baud
        self._policy = ReconnectPolicy()
        self._connected: bool = False
        self._ser: Any = None
        self._open()

    # ── connection management ─────────────────────────────────────────────────

    def _open(self) -> None:
        """Open (or reopen) the serial port; sets _connected=True on success."""
        self._ser = serial.Serial(self._port, self._baud, timeout=0.1)
        self._connected = True
        log.info("ViscaUSB opened %s @ %d baud", self._port, self._baud)

    def _close_ser(self) -> None:
        """Close the serial port without raising."""
        self._connected = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    @property
    def connected(self) -> bool:
        """True while the serial port is believed to be live."""
        return self._connected

    # ── helpers ───────────────────────────────────────────────────────────────

    def _send(self, cmd: bytes) -> None:
        """Write VISCA command; drain any pending ACK/completion bytes first.

        On SerialException or OSError the port is closed and a reconnect is
        attempted once (subject to backoff policy).  Never raises.
        """
        # Patch address byte (index 0) to match configured camera address
        patched = bytes([self._addr]) + cmd[1:]

        # Fast path: serial port is open and healthy.
        if self._ser is not None:
            try:
                self._do_write(patched)
                return
            except (serial.SerialException, OSError) as _exc:
                log.debug("ViscaUSB write failed (will reconnect): %s", _exc)
            # Transport error on the existing port — close it.
            self._close_ser()

        # Port is None (either from a previous failure or just closed above).
        # Check whether the policy allows a reconnect attempt right now.
        now = time.monotonic()
        if not self._policy.should_attempt(now):
            return  # still in backoff window; swallow silently

        # Attempt exactly one reconnect.
        try:
            self._open()
            self._policy.record_success()
        except (serial.SerialException, OSError) as exc:
            log.warning("ViscaUSB reconnect to %s failed: %s", self._port, exc)
            self._policy.record_failure(time.monotonic())
            return

        # Safety halt-on-reconnect: always send a stop immediately after
        # reconnect so a camera that was mid-pan on disconnect halts before
        # we retry any new command.  Sending a stop to an already-stopped
        # camera is a safe no-op.
        stop_patched = bytes([self._addr]) + visca_stop_cmd()[1:]
        zoom_stop_patched = bytes([self._addr]) + visca_zoom_stop_cmd()[1:]
        try:
            self._do_write(stop_patched)
            self._do_write(zoom_stop_patched)
        except (serial.SerialException, OSError):
            pass  # best-effort; the retry below will handle further errors

        # Retry the command once on the fresh port.
        try:
            self._do_write(patched)
        except (serial.SerialException, OSError) as exc:
            log.warning("ViscaUSB retry write failed: %s", exc)
            self._close_ser()
            self._policy.record_failure(time.monotonic())

    def _do_write(self, patched: bytes) -> None:
        """Actually write to the serial port (factored out for clarity)."""
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
        self._close_ser()
        log.info("ViscaUSB closed")
