"""PTZBackend abstract interface, capability/state dataclasses, and shared VISCA helpers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ── capability flags ──────────────────────────────────────────────────────────


@dataclass
class PTZCaps:
    """Capability flags reported by each backend."""

    continuous_pan_tilt: bool = True
    continuous_zoom: bool = True
    absolute_pan_tilt: bool = False
    absolute_zoom: bool = False
    native_presets: bool = False
    query_position: bool = False
    # normalized speed ceilings (backend may not reach 1.0)
    pan_speed_max: float = 1.0
    tilt_speed_max: float = 1.0
    zoom_speed_max: float = 1.0


# ── position snapshot ─────────────────────────────────────────────────────────


@dataclass
class PTZState:
    """Snapshot of camera position; all values normalized."""

    pan: float = 0.0  # [-1, 1]  left → right
    tilt: float = 0.0  # [-1, 1]  down → up
    zoom: float = 0.0  # [0,  1]  wide → tele
    timestamp: float = field(default_factory=time.monotonic)


# ── abstract backend ──────────────────────────────────────────────────────────


class PTZBackend(ABC):
    """Common interface for NDI, VISCA-IP, VISCA-USB, and ONVIF PTZ backends.

    All speed/position values are normalized:
      pan/tilt  [-1, 1]   negative=left/down, positive=right/up
      zoom      [-1, 1]   negative=wide, positive=tele   (velocity)
                [0,  1]   wide → tele                    (absolute)
    """

    caps: PTZCaps

    def __init__(self) -> None:
        self.caps = PTZCaps()

    @abstractmethod
    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        """Send continuous-velocity command.  Repeated until stop() or a new call."""

    def move_absolute(self, pan: float, tilt: float, zoom: float) -> None:
        """Move to absolute normalized position.  Optional; raises if not supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support absolute moves")

    @abstractmethod
    def stop(self) -> None:
        """Halt all motion immediately.  Must be safe to call from any thread."""

    def get_position(self) -> PTZState | None:
        """Return current position, or None if the backend cannot query it."""
        return None

    @abstractmethod
    def goto_preset(self, idx: int) -> None:
        """Recall preset slot *idx* (0-based).  Native if supported, else software."""

    @abstractmethod
    def save_preset(self, idx: int) -> None:
        """Save current position to preset slot *idx* (0-based)."""

    def home(self) -> None:
        """Drive the camera to its optical home position.

        Optional capability — the default is a safe no-op so backends/cameras
        that cannot home simply ignore the request.
        """
        return None

    def osd_menu(self) -> None:
        """Open (or toggle) the camera's on-screen-display menu.

        Optional capability — the default is a safe no-op so backends/cameras
        without an OSD menu silently ignore the request.
        """
        return None

    @abstractmethod
    def close(self) -> None:
        """Release hardware resources.  No further calls allowed after this."""

    # context-manager convenience
    def __enter__(self) -> PTZBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ── VISCA byte helpers (used by visca_usb + visca_ip) ─────────────────────────


def visca_pantilt_cmd(pan: float, tilt: float) -> bytes:
    """Build an 81 01 06 01 VV WW PP QQ FF pan/tilt drive command.

    pan  [-1, 1]: negative=left (0x01), positive=right (0x02)
    tilt [-1, 1]: positive=up   (0x01), negative=down  (0x02)
    """
    if abs(pan) < 0.01:
        pan_speed, pan_dir = 0x15, 0x03  # stop
    else:
        pan_speed = max(0x01, min(0x18, round(abs(pan) * 0x18)))
        pan_dir = 0x01 if pan < 0 else 0x02

    if abs(tilt) < 0.01:
        tilt_speed, tilt_dir = 0x15, 0x03  # stop
    else:
        tilt_speed = max(0x01, min(0x14, round(abs(tilt) * 0x14)))
        tilt_dir = 0x01 if tilt > 0 else 0x02

    return bytes([0x81, 0x01, 0x06, 0x01, pan_speed, tilt_speed, pan_dir, tilt_dir, 0xFF])


def visca_zoom_cmd(zoom: float) -> bytes:
    """Build an 81 01 04 07 ZZ FF zoom drive command.

    zoom [-1, 1]: negative=wide, positive=tele; 0=stop.
    """
    if abs(zoom) < 0.01:
        zoom_byte = 0x00
    elif zoom > 0:
        speed = max(0x01, min(0x07, round(zoom * 7)))
        zoom_byte = 0x20 | speed  # tele at variable speed
    else:
        speed = max(0x01, min(0x07, round(abs(zoom) * 7)))
        zoom_byte = 0x30 | speed  # wide at variable speed
    return bytes([0x81, 0x01, 0x04, 0x07, zoom_byte, 0xFF])


def visca_stop_cmd() -> bytes:
    return bytes([0x81, 0x01, 0x06, 0x01, 0x15, 0x15, 0x03, 0x03, 0xFF])


def visca_zoom_stop_cmd() -> bytes:
    return bytes([0x81, 0x01, 0x04, 0x07, 0x00, 0xFF])


def visca_preset_set_cmd(idx: int) -> bytes:
    """81 01 04 3F 01 MM FF — store preset MM."""
    return bytes([0x81, 0x01, 0x04, 0x3F, 0x01, idx & 0xFF, 0xFF])


def visca_preset_recall_cmd(idx: int) -> bytes:
    """81 01 04 3F 02 MM FF — recall preset MM."""
    return bytes([0x81, 0x01, 0x04, 0x3F, 0x02, idx & 0xFF, 0xFF])


def visca_home_cmd() -> bytes:
    """81 01 06 04 FF — drive pan/tilt to the optical home position."""
    return bytes([0x81, 0x01, 0x06, 0x04, 0xFF])


def visca_menu_cmd() -> bytes:
    """81 01 06 06 10 FF — toggle the camera's on-screen-display (OSD) menu.

    This is the common Sony/PTZOptics "Menu" key shortcut (Datascreen on).
    Cameras without an OSD menu ignore the command, so it is safe to send.
    """
    return bytes([0x81, 0x01, 0x06, 0x06, 0x10, 0xFF])
