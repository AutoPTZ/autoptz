"""Update checking and OS-specific release installation."""

from __future__ import annotations

from autoptz.update.checker import UpdateInfo, check_for_update
from autoptz.update.installer import DownloadedUpdate, download_update, launch_update

__all__ = ["DownloadedUpdate", "UpdateInfo", "check_for_update", "download_update", "launch_update"]
