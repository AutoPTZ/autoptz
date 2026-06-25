"""Check GitHub Releases for a newer AutoPTZ version.

:func:`check_for_update_result` never raises and reports a **three-way** outcome
— update-available / up-to-date / failed(reason) — so the UI can tell "you're on
the latest" apart from "the check couldn't run" (offline, TLS, rate-limited).
The legacy :func:`check_for_update` (``UpdateInfo | None``) is kept as a thin
wrapper for callers that only care about an available update.

HTTPS uses certifi's CA bundle when present so the check works in frozen
(PyInstaller) builds, where a missing system trust store caused TLS verification
to fail silently — the root cause of "Intel Macs can't check for updates".

Download/install lives in :mod:`autoptz.update.installer` so release parsing
stays pure and easy to test.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal

from packaging.version import InvalidVersion, Version

log = logging.getLogger(__name__)

DEFAULT_REPO = "AutoPTZ/autoptz"
_API_RELEASES = "https://api.github.com/repos/{repo}/releases?per_page=20"
_TIMEOUT_S = 6.0
_USER_AGENT = "AutoPTZ-Updater"

#: Why a check failed — drives the user-facing message and whether to retry soon.
ErrorKind = Literal["network", "timeout", "tls", "rate_limited", "parse", "bad_version"]
CheckStatus = Literal["update_available", "up_to_date", "failed"]


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
        """Return ``(filename, url)`` for this OS (and arch on macOS), or None.

        On macOS this is **architecture-safe**: an Intel Mac never falls back to
        an arm64-only build (or vice versa) — handing the wrong-arch installer to
        a user just produces a binary that won't run.  A genuinely arch-agnostic
        dmg (no ``arm64``/``x86_64`` marker, e.g. an old universal release) is
        still accepted.
        """
        if sys.platform == "darwin":
            import platform

            arch = platform.machine().lower()  # "arm64" or "x86_64"
            if arch == "x86_64":
                want = ("x86_64", "x64", "intel")
            elif arch == "arm64":
                want = ("arm64", "aarch64", "apple", "silicon")
            else:  # unknown arch → only an exact substring match
                want = (arch,)
            dmgs = [(n, u) for n, u in self.assets if n.lower().endswith(".dmg")]
            # 1. An asset that names this architecture.
            for name, url in dmgs:
                if any(tok in name.lower() for tok in want):
                    return name, url
            # 2. An arch-agnostic dmg (no arch marker at all) — legacy/universal.
            markers = ("arm64", "aarch64", "x86_64", "x64", "intel", "apple", "silicon")
            for name, url in dmgs:
                if not any(m in name.lower() for m in markers):
                    return name, url
            # 3. Only wrong-arch dmgs exist → no safe asset for this Mac.
            return None
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


@dataclass(frozen=True)
class CheckResult:
    """Outcome of an update check.

    ``status`` is one of ``update_available`` / ``up_to_date`` / ``failed``.
    ``info`` is set only when an update is available; ``error`` / ``error_kind``
    are set only on failure.
    """

    status: CheckStatus
    info: UpdateInfo | None = None
    error: str | None = None
    error_kind: ErrorKind | None = None

    @property
    def has_update(self) -> bool:
        return self.status == "update_available"

    @property
    def ok(self) -> bool:
        """True unless the check failed (i.e. the result is trustworthy)."""
        return self.status != "failed"


def _repo() -> str:
    return os.environ.get("AUTOPTZ_UPDATE_REPO", DEFAULT_REPO)


def _ssl_context() -> ssl.SSLContext | None:
    """Prefer certifi's CA bundle so HTTPS verifies in frozen builds.

    PyInstaller bundles may lack the system trust store, so ``urllib`` could not
    verify ``api.github.com`` → ``CERTIFICATE_VERIFY_FAILED`` → the check failed
    silently (notably on Intel macOS).  Falls back to the default context, then
    to ``None`` (urllib's default) if even that can't be built.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 — certifi missing → use the system default
        try:
            return ssl.create_default_context()
        except Exception:  # noqa: BLE001
            return None


def _fetch_releases() -> tuple[list[dict] | None, ErrorKind | None, str | None]:
    """Fetch the releases list. Returns ``(releases, error_kind, error_message)``.

    On success ``error_kind``/``error_message`` are ``None``; on failure
    ``releases`` is ``None`` and the error is classified for the UI.  Never raises.
    """
    url = _API_RELEASES.format(repo=_repo())
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    ctx = _ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S, context=ctx) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            return (
                None,
                "rate_limited",
                "GitHub's update API is rate-limited right now. Try again later.",
            )
        log.info("update check: HTTP %s", exc.code)
        return None, "network", f"The update server returned an error (HTTP {exc.code})."
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        if "CERTIFICATE_VERIFY" in reason.upper() or "SSL" in reason.upper():
            log.info("update check: TLS failure (%s)", reason)
            return None, "tls", "Couldn't establish a secure connection to the update server."
        log.info("update check: network failure (%s)", reason)
        return None, "network", "Couldn't reach the update server. Check your internet connection."
    except TimeoutError:
        return None, "timeout", "The update server timed out. Check your internet connection."
    except ValueError:
        return None, "parse", "The update server sent an unexpected response."
    except OSError as exc:
        log.info("update check: OS error (%s)", exc)
        return None, "network", "Couldn't reach the update server. Check your internet connection."
    if not isinstance(data, list):
        return None, "parse", "The update server sent an unexpected response."
    return data, None, None


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


def check_for_update_result(current: str, *, include_prereleases: bool = False) -> CheckResult:
    """Check for a newer release and report a three-way :class:`CheckResult`.

    Never raises.  ``include_prereleases`` controls whether beta/rc builds count.
    """
    try:
        cur = Version(str(current).lstrip("vV"))
    except InvalidVersion:
        log.info("update check: unparseable current version %r", current)
        return CheckResult(
            status="failed",
            error_kind="bad_version",
            error=f"Could not read the running version ({current!r}).",
        )

    releases, kind, message = _fetch_releases()
    if releases is None:
        return CheckResult(status="failed", error_kind=kind, error=message)

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

    if best is not None:
        return CheckResult(status="update_available", info=best)
    return CheckResult(status="up_to_date")


def check_for_update(current: str, *, include_prereleases: bool = False) -> UpdateInfo | None:
    """Return the newest release strictly newer than *current*, else ``None``.

    Thin wrapper over :func:`check_for_update_result`; ``None`` covers both
    "up to date" and "check failed" (use the result form to tell them apart).
    """
    result = check_for_update_result(current, include_prereleases=include_prereleases)
    return result.info if result.has_update else None
