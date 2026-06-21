"""Download and launch AutoPTZ release assets for the current OS.

The checker decides *which* release is newer.  This module handles the concrete
"get the right file and start it" work in a UI-independent way so it is easy to
test and safe to call from a background Qt task.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from autoptz.update.checker import UpdateInfo

log = logging.getLogger(__name__)

_TIMEOUT_S = 30.0
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DownloadedUpdate:
    """A downloaded release asset ready to launch."""

    path: Path
    asset_name: str
    version: str


def default_download_dir() -> Path:
    """Return the user-visible folder where updater downloads are stored."""
    root = Path.home() / "Downloads"
    if not root.exists():
        root = Path.home()
    return root / "AutoPTZ Updates"


def _safe_asset_name(name: str) -> str:
    """Return a filename safe to create in the download directory."""
    clean = Path(name).name.strip()
    return clean or "AutoPTZ-update"


def download_update(
    info: UpdateInfo,
    *,
    dest_dir: Path | None = None,
    opener: object | None = None,
) -> DownloadedUpdate:
    """Download this OS's release asset and return its local path.

    Raises ``RuntimeError`` with a user-readable message when the release has no
    usable asset for this OS or the network/file write fails.
    """
    asset = info.asset_for_platform()
    if asset is None:
        raise RuntimeError("No AutoPTZ installer asset is available for this operating system.")
    asset_name, url = asset
    target_dir = dest_dir or default_download_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / _safe_asset_name(asset_name)
    tmp_path = path.with_suffix(path.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "AutoPTZ-Updater"})
    opener_obj = opener or urllib.request.urlopen
    try:
        with opener_obj(request, timeout=_TIMEOUT_S) as response:  # type: ignore[misc]  # noqa: S310
            with tmp_path.open("wb") as out:
                _copy_stream(response, out)
        tmp_path.replace(path)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(f"Could not download AutoPTZ update: {exc}") from exc

    return DownloadedUpdate(path=path, asset_name=asset_name, version=info.version)


def _copy_stream(src: BinaryIO, dst: BinaryIO) -> None:
    """Copy bytes without loading the installer into memory."""
    while True:
        chunk = src.read(_CHUNK_SIZE)
        if not chunk:
            return
        dst.write(chunk)


def launch_update(path: Path) -> None:
    """Launch a downloaded installer/app bundle for the current OS."""
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])  # noqa: S603,S607
            return
        if path.name.lower().endswith(".appimage"):
            mode = path.stat().st_mode
            path.chmod(mode | 0o755)
            subprocess.Popen([str(path)])  # noqa: S603
            return
        subprocess.Popen(["xdg-open", str(path)])  # noqa: S603,S607
    except OSError as exc:
        log.debug("launch update failed", exc_info=True)
        raise RuntimeError(f"Could not launch AutoPTZ update: {exc}") from exc
