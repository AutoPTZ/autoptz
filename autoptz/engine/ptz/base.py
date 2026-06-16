"""PTZBackend abstract interface shared by all PTZ implementations.

Phase 5 implementation target.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class PTZBackend(ABC):
    """Common interface for NDI, VISCA-IP, VISCA-USB, and ONVIF PTZ backends."""

    @abstractmethod
    def move(self, pan_speed: float, tilt_speed: float) -> None:
        """Continuous move at normalized speeds in [-1, 1]."""

    @abstractmethod
    def stop(self) -> None:
        """Stop all motion immediately."""

    @abstractmethod
    def zoom(self, speed: float) -> None:
        """Continuous zoom at normalized speed in [-1, 1]."""

    @abstractmethod
    def go_to_preset(self, name: str) -> None:
        """Recall a named preset position."""

    @abstractmethod
    def save_preset(self, name: str) -> None:
        """Save current position as a named preset."""

    @abstractmethod
    def close(self) -> None:
        """Release hardware resources."""
