"""Safe VISCA-over-USB serial auto-discovery.

A USB PTZ camera typically presents **two** USB interfaces: a UVC *video* device
(opened by the frame source) and a companion USB-serial *control* port — e.g.
``/dev/cu.usbserial-XXXX`` (macOS), ``/dev/ttyUSB0`` (Linux), or ``COM3``
(Windows) — that speaks VISCA.  Auto-probe never tried serial ports, so a plain
USB camera got no PTZ backend and every move/menu command silently no-op'd.

This module finds that port without moving the camera: it sends the VISCA
``CAM_VersionInq`` (``81 09 00 02 FF``) — a read-only inquiry — and treats a port
as a VISCA camera only when it answers with a valid completion frame.  It also
detects the **baud** the camera answers on (many newer USB PTZ cameras run at
115200 and silently ignore everything at the legacy 9600 default).

Everything here is best-effort and must never raise into the engine.
"""

from __future__ import annotations

import logging
import time

import serial  # pyserial — already in requirements/base.txt
from serial import SerialException
from serial.tools import list_ports

log = logging.getLogger(__name__)

# CAM_VersionInq — a read-only inquiry that does NOT move the camera.
_VISCA_VERSION_INQ = bytes([0x81, 0x09, 0x00, 0x02, 0xFF])

# Bauds to try, in order.  9600 is the classic Sony/VISCA default; 115200 is
# common on modern USB PTZ heads; 38400 is the occasional middle ground.
DEFAULT_BAUDS: tuple[int, ...] = (9600, 115200, 38400)

# Substrings (case-insensitive) that mark a real USB-serial adapter.  We never
# poke Bluetooth, debug-console, or other virtual ports.
_PORT_INCLUDE = ("usbserial", "usbmodem", "wchusbserial", "slab", "ttyusb", "ttyacm", "ftdi")
_PORT_EXCLUDE = ("bluetooth", "debug-console", "debug-modem")


def is_visca_reply(data: bytes) -> bool:
    """True if *data* looks like a VISCA reply frame.

    A VISCA reply starts with an address byte ``z0`` where ``z`` is ``9``–``F``
    (camera address 1–7 → ``0x90``–``0xF0``) and ends with the ``0xFF``
    terminator.  Requiring both makes a false positive from line noise unlikely.
    """
    return bool(data) and data[-1] == 0xFF and (data[0] & 0xF0) >= 0x90


def probe_visca_baud(
    port: str,
    bauds: tuple[int, ...] = DEFAULT_BAUDS,
    *,
    settle_s: float = 0.25,
    open_timeout_s: float = 0.5,
    read_bytes: int = 16,
) -> int | None:
    """Return the first baud at which *port* answers a VISCA version inquiry.

    Opens *port* at each baud, sends the (non-moving) ``CAM_VersionInq``, and
    returns that baud when a valid VISCA reply comes back.  Returns ``None`` if
    no baud answers, the port is busy, or the port is absent.  Never raises.
    """
    for baud in bauds:
        ser = None
        try:
            ser = serial.Serial(port, baud, timeout=open_timeout_s)
            ser.reset_input_buffer()
            ser.write(_VISCA_VERSION_INQ)
            time.sleep(settle_s)
            pending = ser.in_waiting
            data = ser.read(pending if pending else read_bytes)
            if is_visca_reply(data):
                log.debug("VISCA reply on %s @ %d baud: %s", port, baud, data.hex(" "))
                return baud
        except (SerialException, OSError) as exc:
            log.debug("VISCA probe %s @ %d failed: %s", port, baud, exc)
        except Exception:  # noqa: BLE001 — discovery must never raise
            log.debug("VISCA probe %s @ %d errored", port, baud, exc_info=True)
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:  # noqa: BLE001
                    pass
    return None


def candidate_ports() -> list[str]:
    """Enumerate serial ports that look like USB-serial adapters.

    Filters out Bluetooth / debug / other virtual ports so we never write to
    something that isn't a camera.  Best-effort: returns ``[]`` if enumeration
    is unavailable.
    """
    try:
        ports = list(list_ports.comports())
    except Exception:  # noqa: BLE001
        log.debug("serial port enumeration failed", exc_info=True)
        return []
    out: list[str] = []
    for info in ports:
        dev = getattr(info, "device", "") or ""
        low = dev.lower()
        if any(bad in low for bad in _PORT_EXCLUDE):
            continue
        if low.startswith("com") or any(tok in low for tok in _PORT_INCLUDE):
            out.append(dev)
    return out


def discover_visca_usb(
    *,
    bauds: tuple[int, ...] = DEFAULT_BAUDS,
    settle_s: float = 0.25,
) -> tuple[str, int] | None:
    """Scan USB-serial ports for a VISCA camera; return ``(port, baud)`` or ``None``.

    Returns the first port that answers a VISCA version inquiry.  Never raises.
    """
    for port in candidate_ports():
        baud = probe_visca_baud(port, bauds, settle_s=settle_s)
        if baud is not None:
            log.info("Discovered VISCA-USB camera on %s @ %d baud", port, baud)
            return port, baud
    return None
