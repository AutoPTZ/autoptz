"""Tests for update checking and OS-specific asset download (network mocked)."""

from __future__ import annotations

from pathlib import Path

import autoptz.update.checker as checker
import autoptz.update.installer as installer
from autoptz.update.checker import UpdateInfo, check_for_update
from autoptz.update.installer import download_update, launch_update


def _release(tag: str, *, prerelease: bool = False, assets: list[str] | None = None) -> dict:
    return {
        "tag_name": tag,
        "name": f"AutoPTZ {tag}",
        "body": "notes",
        "html_url": f"https://github.com/AutoPTZ/autoptz/releases/tag/{tag}",
        "prerelease": prerelease,
        "draft": False,
        "assets": [
            {"name": n, "browser_download_url": f"https://example/{n}"} for n in (assets or [])
        ],
    }


def _patch(monkeypatch, releases: list[dict] | None) -> None:
    # _fetch_releases now returns (releases, error_kind, error_message); None
    # releases models a network failure.
    if releases is None:
        monkeypatch.setattr(
            checker, "_fetch_releases", lambda: (None, "network", "Couldn't reach the server.")
        )
    else:
        monkeypatch.setattr(checker, "_fetch_releases", lambda: (releases, None, None))


def test_newer_stable_returns_info(monkeypatch) -> None:
    _patch(monkeypatch, [_release("v2.1.0"), _release("v2.0.0")])
    info = check_for_update("2.0.0")
    assert isinstance(info, UpdateInfo)
    assert info.version == "2.1.0"
    assert info.tag == "v2.1.0"


def test_same_version_returns_none(monkeypatch) -> None:
    _patch(monkeypatch, [_release("v2.0.0")])
    assert check_for_update("2.0.0") is None


def test_older_release_returns_none(monkeypatch) -> None:
    _patch(monkeypatch, [_release("v1.9.0")])
    assert check_for_update("2.0.0") is None


def test_prerelease_excluded_by_default(monkeypatch) -> None:
    _patch(monkeypatch, [_release("v2.1.0-rc1", prerelease=True)])
    assert check_for_update("2.0.0") is None


def test_prerelease_included_when_requested(monkeypatch) -> None:
    _patch(monkeypatch, [_release("v2.1.0-rc1", prerelease=True)])
    info = check_for_update("2.0.0", include_prereleases=True)
    assert info is not None and info.is_prerelease


def test_picks_highest_of_many(monkeypatch) -> None:
    _patch(monkeypatch, [_release("v2.0.1"), _release("v2.3.0"), _release("v2.1.0")])
    info = check_for_update("2.0.0")
    assert info is not None and info.version == "2.3.0"


def test_network_failure_returns_none(monkeypatch) -> None:
    _patch(monkeypatch, None)
    assert check_for_update("2.0.0") is None


def test_bad_current_version_returns_none(monkeypatch) -> None:
    _patch(monkeypatch, [_release("v2.1.0")])
    assert check_for_update("not-a-version") is None


def test_draft_releases_ignored(monkeypatch) -> None:
    draft = _release("v9.9.9")
    draft["draft"] = True
    _patch(monkeypatch, [draft, _release("v2.0.0")])
    assert check_for_update("2.0.0") is None


def test_asset_url_for_platform(monkeypatch) -> None:
    info = UpdateInfo(
        version="2.1.0",
        tag="v2.1.0",
        name="x",
        body="",
        html_url="https://example/rel",
        is_prerelease=False,
        assets=(
            ("AutoPTZ-2.1.0.dmg", "https://example/dmg"),
            ("AutoPTZ-2.1.0-setup.exe", "https://example/exe"),
            ("AutoPTZ-2.1.0.AppImage", "https://example/appimage"),
        ),
    )
    url = info.asset_url_for_platform()
    # Whatever this OS is, it should resolve to one of the provided assets.
    assert url in {"https://example/dmg", "https://example/exe", "https://example/appimage"}
    asset = info.asset_for_platform()
    assert asset is not None
    assert asset[1] == url


def test_asset_for_platform_macos_prefers_matching_arch(monkeypatch) -> None:
    import platform as _platform

    monkeypatch.setattr(checker.sys, "platform", "darwin")
    info = UpdateInfo(
        version="2.1.0",
        tag="v2.1.0",
        name="x",
        body="",
        html_url="https://example/rel",
        is_prerelease=False,
        assets=(
            ("AutoPTZ-2.1.0-macos-x86_64.dmg", "https://example/intel"),
            ("AutoPTZ-2.1.0-macos-arm64.dmg", "https://example/arm"),
        ),
    )
    monkeypatch.setattr(_platform, "machine", lambda: "arm64")
    assert info.asset_for_platform() == ("AutoPTZ-2.1.0-macos-arm64.dmg", "https://example/arm")
    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    assert info.asset_for_platform()[1] == "https://example/intel"

    # An older single, unlabeled dmg still resolves via the fallback.
    one = UpdateInfo(
        version="2.0.0",
        tag="v2.0.0",
        name="x",
        body="",
        html_url="https://example/rel",
        is_prerelease=False,
        assets=(("AutoPTZ-2.0.0.dmg", "https://example/any"),),
    )
    assert one.asset_for_platform() == ("AutoPTZ-2.0.0.dmg", "https://example/any")


def test_update_manager_prerelease_opt_in() -> None:
    """include_prereleases defaults to opt-out (False) and the setter persists.

    Stable users must not be offered pre-releases unless they explicitly opt in,
    matching check_for_update()'s own default.
    """
    from PySide6.QtCore import QCoreApplication

    from autoptz.ui.update_manager import UpdateManager

    if QCoreApplication.instance() is None:
        QCoreApplication([])

    store: dict[str, object] = {}

    class _Client:
        def getSetting(self, key: str, default: object) -> object:
            return store.get(key, default)

        def setSetting(self, key: str, value: object) -> None:
            store[key] = value

    mgr = UpdateManager(_Client(), "2.0.0")
    assert mgr.include_prereleases is False  # opt-out by default
    mgr.set_include_prereleases(True)
    assert mgr.include_prereleases is True
    assert store["update_include_prereleases"] is True
    mgr.set_include_prereleases(False)
    assert mgr.include_prereleases is False


def test_download_update_writes_platform_asset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(checker.sys, "platform", "linux")
    info = UpdateInfo(
        version="2.1.0",
        tag="v2.1.0",
        name="x",
        body="",
        html_url="https://example/rel",
        is_prerelease=False,
        assets=(("AutoPTZ-2.1.0-linux-x86_64.AppImage", "https://example/appimage"),),
    )

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _n: int) -> bytes:
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return b"appimage-bytes"

    def _open(_request, timeout: float):
        assert timeout > 0
        return _Response()

    result = download_update(info, dest_dir=tmp_path, opener=_open)

    assert result.version == "2.1.0"
    assert result.path.name == "AutoPTZ-2.1.0-linux-x86_64.AppImage"
    assert result.path.read_bytes() == b"appimage-bytes"


def test_download_update_reports_progress(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(checker.sys, "platform", "linux")
    info = UpdateInfo(
        version="2.1.0",
        tag="v2.1.0",
        name="x",
        body="",
        html_url="https://example/rel",
        is_prerelease=False,
        assets=(("AutoPTZ-2.1.0-linux-x86_64.AppImage", "https://example/appimage"),),
    )
    chunks = [b"a" * 1000, b"b" * 1000, b"c" * 500]
    total = sum(len(c) for c in chunks)

    class _Response:
        headers = {"Content-Length": str(total)}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _n: int) -> bytes:
            return chunks.pop(0) if chunks else b""

    seen: list[tuple[int, int]] = []
    result = download_update(
        info,
        dest_dir=tmp_path,
        opener=lambda *_a, **_k: _Response(),
        progress=lambda d, t: seen.append((d, t)),
    )

    assert result.path.read_bytes() == b"a" * 1000 + b"b" * 1000 + b"c" * 500
    # Cumulative byte counts, each carrying the known total, ending exactly at total.
    assert [d for d, _ in seen] == [1000, 2000, 2500]
    assert all(t == total for _, t in seen)
    assert seen[-1] == (total, total)


# ── launch_update: Windows silent-install flags ─────────────────────────────────


def _capture_windows_launch(monkeypatch) -> list[str]:  # noqa: ANN001
    monkeypatch.setattr(installer.sys, "platform", "win32")
    captured: list[list[str]] = []
    monkeypatch.setattr(installer.subprocess, "Popen", lambda args, *a, **k: captured.append(args))
    launch_update(Path("C:/Users/x/Downloads/AutoPTZ-2.1.0-windows-x64-setup.exe"))
    assert captured, "Popen was not called"
    return captured[0]


def test_windows_launch_uses_silent_not_verysilent(monkeypatch) -> None:  # noqa: ANN001
    args = _capture_windows_launch(monkeypatch)
    # /SILENT shows a progress window (visible feedback); /VERYSILENT showed nothing.
    assert "/SILENT" in args
    assert "/VERYSILENT" not in args


def test_windows_launch_drops_restartapplications(monkeypatch) -> None:  # noqa: ANN001
    # The installer now relaunches via its own silent [Run] entry; passing
    # /RESTARTAPPLICATIONS was a no-op (app self-quits) and risks a double launch.
    args = _capture_windows_launch(monkeypatch)
    assert "/RESTARTAPPLICATIONS" not in args


def test_windows_launch_keeps_no_wizard_and_no_reboot(monkeypatch) -> None:  # noqa: ANN001
    args = _capture_windows_launch(monkeypatch)
    assert "/SUPPRESSMSGBOXES" in args
    assert "/NORESTART" in args
    assert str(args[0]).endswith(".exe")
