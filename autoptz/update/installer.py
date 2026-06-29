"""Download and launch AutoPTZ release assets for the current OS.

The checker decides *which* release is newer.  This module handles the concrete
"get the right file and start it" work in a UI-independent way so it is easy to
test and safe to call from a background Qt task.

Security hardening (see checker.py for the TLS context fix rationale):

* :func:`download_update` uses the same certifi-backed SSL context as the
  checker so HTTPS is verified even in frozen (PyInstaller) builds that lack the
  system trust store.
* :func:`verify_sha256` checks the downloaded installer against a ``SHA256SUMS``
  (or ``<asset>.sha256``) file fetched from the same release before
  :func:`launch_update` is called.  A tampered or corrupted installer is never
  launched.  If the release has no checksum asset a ``WARNING`` is emitted and
  the update proceeds (so existing releases without checksums still work), but
  the gap is never silent.
"""

from __future__ import annotations

import hashlib
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

from autoptz.update.checker import UpdateInfo, _ssl_context

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


def verify_sha256(path: Path, expected_hex: str) -> bool:
    """Return ``True`` iff the SHA-256 digest of *path* matches *expected_hex*.

    Both digests are lower-cased before comparison.  The file is read in
    ``_CHUNK_SIZE`` chunks so the installer is never fully loaded into memory.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower() == expected_hex.strip().lower()


def _find_checksum_asset(
    asset_name: str,
    assets: tuple[tuple[str, str], ...],
) -> tuple[str, str] | None:
    """Return ``(name, url)`` for a checksum asset matching *asset_name*, or None.

    Looks for two conventions (in preference order):

    1. A ``<asset_name>.sha256`` sibling asset.
    2. A ``SHA256SUMS`` (or ``sha256sums``) manifest asset.
    """
    # 1. Per-file sidecar: e.g. "AutoPTZ-2.1.0-setup.exe.sha256"
    sidecar = asset_name.lower() + ".sha256"
    for name, url in assets:
        if name.lower() == sidecar:
            return name, url
    # 2. Aggregate manifest: "SHA256SUMS" or "sha256sums"
    for name, url in assets:
        if name.lower() in ("sha256sums", "sha256sums.txt"):
            return name, url
    return None


def _parse_sha256sums(content: str, asset_name: str) -> str | None:
    """Extract the hex digest for *asset_name* from a ``SHA256SUMS``-style file.

    Handles both ``<hash>  <name>`` (two-space) and ``<hash> <name>`` formats.
    If the file is a single-line sidecar (just the hash), returns that directly.
    """
    name_lower = asset_name.lower()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 1:
            # Single-line sidecar: the entire content is the hash.
            return parts[0].lower()
        digest, fname = parts[0], parts[1].lstrip("* \t")
        fname_lower = fname.lower()
        basename_lower = fname.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if fname_lower == name_lower or basename_lower == name_lower:
            return digest.lower()
    return None


def _fetch_text(url: str, opener: object | None = None) -> str:
    """Fetch *url* as UTF-8 text using the certifi TLS context.

    Raises ``RuntimeError`` on network/TLS/HTTP errors.
    """
    ctx = _ssl_context()
    request = urllib.request.Request(url, headers={"User-Agent": "AutoPTZ-Updater"})
    opener_obj = opener or urllib.request.urlopen
    try:
        with opener_obj(request, timeout=_TIMEOUT_S, context=ctx) as resp:  # type: ignore[misc]  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc


def download_update(
    info: UpdateInfo,
    *,
    dest_dir: Path | None = None,
    opener: object | None = None,
    progress: DownloadProgress | None = None,
) -> DownloadedUpdate:
    """Download this OS's release asset, verify its SHA-256 hash, and return its local path.

    ``progress(downloaded, total)`` is called as bytes arrive so the UI can show
    a real progress bar (``total`` is 0 when the server omits Content-Length).

    Raises ``RuntimeError`` with a user-readable message when the release has no
    usable asset for this OS, the network/file write fails, or the checksum
    verification fails.  A missing checksum asset logs a ``WARNING`` and proceeds.
    """
    asset = info.asset_for_platform()
    if asset is None:
        raise RuntimeError("No AutoPTZ installer asset is available for this operating system.")
    asset_name, url = asset
    target_dir = dest_dir or default_download_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / _safe_asset_name(asset_name)
    tmp_path = path.with_suffix(path.suffix + ".part")

    ctx = _ssl_context()
    request = urllib.request.Request(url, headers={"User-Agent": "AutoPTZ-Updater"})
    opener_obj = opener or urllib.request.urlopen
    try:
        with opener_obj(request, timeout=_TIMEOUT_S, context=ctx) as response:  # type: ignore[misc]  # noqa: S310
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

    # ── Checksum verification ──────────────────────────────────────────────────
    checksum_asset = _find_checksum_asset(asset_name, info.assets)
    if checksum_asset is None:
        log.warning(
            "AutoPTZ update: no SHA256SUMS or .sha256 asset found for release %s — "
            "installer integrity could not be verified before launch.",
            info.version,
        )
    else:
        _cksum_name, cksum_url = checksum_asset
        try:
            cksum_text = _fetch_text(cksum_url, opener=opener)
        except RuntimeError as exc:
            raise RuntimeError(f"Could not fetch checksum for AutoPTZ update: {exc}") from exc
        expected = _parse_sha256sums(cksum_text, asset_name)
        if expected is None:
            raise RuntimeError(
                f"Checksum file for AutoPTZ {info.version} did not contain an entry "
                f"for '{asset_name}'. Update aborted to prevent running an unverified installer."
            )
        if not verify_sha256(path, expected):
            # Remove the tampered/corrupt file so it is never launched.
            try:
                path.unlink()
            except OSError:
                pass
            raise RuntimeError(
                f"SHA-256 mismatch for '{asset_name}' — the downloaded installer may be "
                "corrupted or tampered with. Update aborted."
            )
        log.info("AutoPTZ update: SHA-256 verified for %s (%s)", asset_name, info.version)

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

    Windows installs with Inno Setup ``/SILENT``: **no wizard pages**, but a
    visible progress window so the update is obviously happening (``/VERYSILENT``
    showed nothing at all, so the user couldn't tell it was working or done).  The
    installer closes the running app, updates in place, and **relaunches it
    itself** via a silent-only ``[Run]`` entry — see ``packaging/autoptz.iss``.

    We deliberately do *not* pass ``/RESTARTAPPLICATIONS``: the app self-quits
    right after launching the installer, so the Restart Manager never registers it
    and that flag was a no-op — worse, it could double-launch once the installer's
    own relaunch entry works.  macOS opens the ``.dmg`` (the user drags to
    Applications) and Linux runs the new AppImage.
    """
    try:
        if sys.platform == "win32":
            if path.suffix.lower() == ".exe":
                # Inno Setup silent flags: no wizard pages, suppress prompts (so
                # closing the running app is automatic), don't reboot.  The
                # installer relaunches AutoPTZ when done (its silent [Run] entry).
                # Detached so this process can quit.
                subprocess.Popen(  # noqa: S603
                    [
                        str(path),
                        "/SILENT",
                        "/SUPPRESSMSGBOXES",
                        "/NORESTART",
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
