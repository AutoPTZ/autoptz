"""Tests for update checking and OS-specific asset download (network mocked)."""

from __future__ import annotations

import autoptz.update.checker as checker
from autoptz.update.checker import UpdateInfo, check_for_update
from autoptz.update.installer import download_update


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
    monkeypatch.setattr(checker, "_fetch_releases", lambda: releases)


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
