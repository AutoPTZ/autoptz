"""Check GitHub Releases for a newer AutoPTZ version.

Pure and network-tolerant: :func:`check_for_update` never raises — any failure
(offline, rate-limited, bad JSON) returns ``None`` so the UI simply shows
"up to date" or nothing. Download/install lives in :mod:`autoptz.update.installer`
so release parsing stays pure and easy to test.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

log = logging.getLogger(__name__)

DEFAULT_REPO = "AutoPTZ/autoptz"
_API_RELEASES = "https://api.github.com/repos/{repo}/releases?per_page=20"
_TIMEOUT_S = 6.0
_USER_AGENT = "AutoPTZ-Updater"


@dataclass(frozen=True)
class UpdateInfo:
    """A release newer than the running version."""

    version: str  # normalized, e.g. "2.1.0"
    tag: str  # raw tag, e.g. "v2.1.0"
    name: str  # release title
    body: str  # markdown release notes
    html_url: str  # release page (what "Download" opens)
    is_prerelease: bool
    assets: tuple[tuple[str, str], ...] = ()  # (filename, browser_download_url)

    def asset_for_platform(self) -> tuple[str, str] | None:
        """Return ``(filename, url)`` for this OS (and arch on macOS), or None."""
        if sys.platform == "darwin":
            # Releases ship separate arm64 and x86_64 dmgs; prefer the one matching
            # this Mac, falling back to any .dmg (e.g. an older single-arch release).
            import platform

            arch = platform.machine().lower()  # "arm64" or "x86_64"
            dmgs = [(n, u) for n, u in self.assets if n.lower().endswith(".dmg")]
            for name, url in dmgs:
                if arch in name.lower():
                    return name, url
            return dmgs[0] if dmgs else None
        if sys.platform == "win32":
            exts: tuple[str, ...] = ("setup.exe", ".msi", ".exe")
        else:
            exts = (".appimage", ".deb", ".tar.gz")
        for name, url in self.assets:
            if name.lower().endswith(exts):
                return name, url
        return None

    def asset_url_for_platform(self) -> str | None:
        """Direct download URL for the current OS asset, or None if absent."""
        asset = self.asset_for_platform()
        return asset[1] if asset else None


def _repo() -> str:
    return os.environ.get("AUTOPTZ_UPDATE_REPO", DEFAULT_REPO)


def _fetch_releases() -> list[dict] | None:
    url = _API_RELEASES.format(repo=_repo())
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        log.info("update check: fetch failed (%s)", exc)
        return None
    return data if isinstance(data, list) else None


def _parse_release(raw: dict) -> tuple[Version, UpdateInfo] | None:
    tag = str(raw.get("tag_name") or "").strip()
    if not tag:
        return None
    try:
        ver = Version(tag.lstrip("vV"))
    except InvalidVersion:
        return None
    assets = tuple(
        (str(a.get("name") or ""), str(a.get("browser_download_url") or ""))
        for a in raw.get("assets", [])
        if a.get("browser_download_url")
    )
    info = UpdateInfo(
        version=str(ver),
        tag=tag,
        name=str(raw.get("name") or tag),
        body=str(raw.get("body") or ""),
        html_url=str(raw.get("html_url") or ""),
        is_prerelease=bool(raw.get("prerelease")),
        assets=assets,
    )
    return ver, info


def check_for_update(current: str, *, include_prereleases: bool = False) -> UpdateInfo | None:
    """Return the newest release strictly newer than *current*, else ``None``.

    Never raises. ``include_prereleases`` controls whether beta/rc builds count.
    """
    try:
        cur = Version(str(current).lstrip("vV"))
    except InvalidVersion:
        log.info("update check: unparseable current version %r", current)
        return None

    releases = _fetch_releases()
    if not releases:
        return None

    best: UpdateInfo | None = None
    best_ver: Version | None = None
    for raw in releases:
        if not isinstance(raw, dict) or raw.get("draft"):
            continue
        if raw.get("prerelease") and not include_prereleases:
            continue
        parsed = _parse_release(raw)
        if parsed is None:
            continue
        ver, info = parsed
        if ver <= cur:
            continue
        if best_ver is None or ver > best_ver:
            best, best_ver = info, ver

    return best
