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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from autoptz.update.checker import UpdateInfo

log = logging.getLogger(__name__)

_TIMEOUT_S = 30.0
_CHUNK_SIZE = 1024 * 1024

#: ``progress(bytes_downloaded, total_bytes)`` — ``total`` is 0 when the server
#: doesn't send Content-Length.
DownloadProgress = Callable[[int, int], None]


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
    progress: DownloadProgress | None = None,
) -> DownloadedUpdate:
    """Download this OS's release asset and return its local path.

    ``progress(downloaded, total)`` is called as bytes arrive so the UI can show
    a real progress bar (``total`` is 0 when the server omits Content-Length).

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
            total = _content_length(response)
            with tmp_path.open("wb") as out:
                _copy_stream(response, out, total, progress)
        tmp_path.replace(path)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(f"Could not download AutoPTZ update: {exc}") from exc

    return DownloadedUpdate(path=path, asset_name=asset_name, version=info.version)


def _content_length(response: object) -> int:
    try:
        return int(response.headers.get("Content-Length") or 0)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return 0


def _copy_stream(
    src: BinaryIO,
    dst: BinaryIO,
    total: int = 0,
    progress: DownloadProgress | None = None,
) -> None:
    """Copy bytes without loading the installer into memory, reporting progress."""
    done = 0
    while True:
        chunk = src.read(_CHUNK_SIZE)
        if not chunk:
            return
        dst.write(chunk)
        done += len(chunk)
        if progress is not None:
            try:
                progress(done, total)
            except Exception:  # noqa: BLE001 — a UI callback must never break the download
                pass


def launch_update(path: Path) -> None:
    """Launch a downloaded installer/app bundle for the current OS.

    Windows installs **silently** (Inno Setup ``/VERYSILENT``): it closes the
    running app, updates in place, and relaunches — no wizard clicks, the
    closest to a seamless "it just updated" experience.  macOS opens the ``.dmg``
    (the user drags to Applications) and Linux runs the new AppImage.
    """
    try:
        if sys.platform == "win32":
            if path.suffix.lower() == ".exe":
                # Inno Setup silent flags: no wizard, auto-close the running app,
                # relaunch when done.  Detached so this process can quit.
                subprocess.Popen(  # noqa: S603
                    [
                        str(path),
                        "/VERYSILENT",
                        "/SUPPRESSMSGBOXES",
                        "/NORESTART",
                        "/RESTARTAPPLICATIONS",
                    ]
                )
            else:
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
