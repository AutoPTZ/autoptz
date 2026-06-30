"""Tests for installer.py security hardening: TLS context + SHA-256 verification.

All network I/O is mocked — no real GitHub calls are made.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

import autoptz.update.installer as installer
from autoptz.update.checker import UpdateInfo
from autoptz.update.installer import (
    _find_checksum_asset,
    _parse_sha256sums,
    download_update,
    verify_sha256,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_info(
    assets: list[tuple[str, str]],
    *,
    platform: str = "linux",
) -> UpdateInfo:
    return UpdateInfo(
        version="2.2.0",
        tag="v2.2.0",
        name="AutoPTZ v2.2.0",
        body="",
        html_url="https://example/rel",
        is_prerelease=False,
        assets=tuple(assets),
    )


class _FakeResponse:
    """Minimal file-like object that `opener` returns inside a with-block."""

    def __init__(self, data: bytes, *, headers: dict | None = None) -> None:
        self._data = data
        self._pos = 0
        self.headers = headers or {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            chunk = self._data[self._pos :]
        else:
            chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


# ── verify_sha256 ─────────────────────────────────────────────────────────────


def test_verify_sha256_correct(tmp_path: Path) -> None:
    data = b"hello world"
    f = tmp_path / "file.bin"
    f.write_bytes(data)
    assert verify_sha256(f, _sha256(data)) is True


def test_verify_sha256_wrong_hash(tmp_path: Path) -> None:
    data = b"hello world"
    f = tmp_path / "file.bin"
    f.write_bytes(data)
    assert verify_sha256(f, "0" * 64) is False


def test_verify_sha256_case_insensitive(tmp_path: Path) -> None:
    data = b"Case Test"
    f = tmp_path / "file.bin"
    f.write_bytes(data)
    digest_upper = _sha256(data).upper()
    assert verify_sha256(f, digest_upper) is True


def test_verify_sha256_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    assert verify_sha256(f, _sha256(b"")) is True


# ── _find_checksum_asset ──────────────────────────────────────────────────────


def test_find_checksum_asset_sidecar() -> None:
    assets = (
        ("AutoPTZ-2.2.0-setup.exe", "https://example/exe"),
        ("AutoPTZ-2.2.0-setup.exe.sha256", "https://example/exe.sha256"),
    )
    result = _find_checksum_asset("AutoPTZ-2.2.0-setup.exe", assets)
    assert result == ("AutoPTZ-2.2.0-setup.exe.sha256", "https://example/exe.sha256")


def test_find_checksum_asset_sidecar_accepts_intel_macos_alias() -> None:
    assets = (("AutoPTZ-2.2.0-macos-x86_64.dmg.sha256", "https://example/intel.sha256"),)
    result = _find_checksum_asset("AutoPTZ-2.2.0-macos-intel.dmg", assets)
    assert result == ("AutoPTZ-2.2.0-macos-x86_64.dmg.sha256", "https://example/intel.sha256")


def test_find_checksum_asset_sha256sums() -> None:
    assets = (
        ("AutoPTZ-2.2.0-setup.exe", "https://example/exe"),
        ("SHA256SUMS", "https://example/SHA256SUMS"),
    )
    result = _find_checksum_asset("AutoPTZ-2.2.0-setup.exe", assets)
    assert result == ("SHA256SUMS", "https://example/SHA256SUMS")


def test_find_checksum_asset_sidecar_preferred_over_manifest() -> None:
    assets = (
        ("AutoPTZ-2.2.0-setup.exe", "https://example/exe"),
        ("AutoPTZ-2.2.0-setup.exe.sha256", "https://example/exe.sha256"),
        ("SHA256SUMS", "https://example/SHA256SUMS"),
    )
    result = _find_checksum_asset("AutoPTZ-2.2.0-setup.exe", assets)
    assert result is not None
    assert result[0] == "AutoPTZ-2.2.0-setup.exe.sha256"


def test_find_checksum_asset_none_when_absent() -> None:
    assets = (("AutoPTZ-2.2.0-setup.exe", "https://example/exe"),)
    assert _find_checksum_asset("AutoPTZ-2.2.0-setup.exe", assets) is None


# ── _parse_sha256sums ──────────────────────────────────────────────────────────


def test_parse_sha256sums_two_space_format() -> None:
    content = "aabbcc  AutoPTZ-2.2.0-setup.exe\nddeeff  AutoPTZ-2.2.0.dmg\n"
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-setup.exe") == "aabbcc"


def test_parse_sha256sums_one_space_format() -> None:
    content = "aabbcc AutoPTZ-2.2.0-setup.exe\n"
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-setup.exe") == "aabbcc"


def test_parse_sha256sums_sidecar_single_line() -> None:
    """A sidecar .sha256 file that contains only the hash."""
    assert _parse_sha256sums("abc123\n", "anything") == "abc123"


def test_parse_sha256sums_star_prefix() -> None:
    """BSD-style: hash *filename"""
    content = "aabbcc *AutoPTZ-2.2.0-setup.exe\n"
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-setup.exe") == "aabbcc"


def test_parse_sha256sums_not_found_returns_none() -> None:
    content = "aabbcc  SomethingElse.dmg\n"
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-setup.exe") is None


def test_parse_sha256sums_case_insensitive_filename() -> None:
    content = "aabbcc  autoptz-2.2.0-setup.exe\n"
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-setup.exe") == "aabbcc"


def test_parse_sha256sums_matches_manifest_path_basename() -> None:
    content = (
        "aabbcc  artifacts/macos-x86_64/AutoPTZ-2.2.0-macos-x86_64.dmg\n"
        "ddeeff  artifacts\\windows\\AutoPTZ-2.2.0-windows-x64-setup.exe\n"
    )
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-macos-x86_64.dmg") == "aabbcc"
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-windows-x64-setup.exe") == "ddeeff"


def test_parse_sha256sums_accepts_intel_macos_alias() -> None:
    content = "aabbcc  artifacts/macos-x86_64/AutoPTZ-2.2.0-macos-x86_64.dmg\n"
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-macos-intel.dmg") == "aabbcc"
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-macos-x64.dmg") == "aabbcc"


def test_parse_sha256sums_intel_alias_does_not_match_arm64() -> None:
    content = "aabbcc  AutoPTZ-2.2.0-macos-arm64.dmg\n"
    assert _parse_sha256sums(content, "AutoPTZ-2.2.0-macos-intel.dmg") is None


# ── download_update: TLS context is passed to urlopen ────────────────────────


def test_download_uses_certifi_ssl_context(tmp_path: Path, monkeypatch) -> None:
    """download_update must pass the certifi SSL context to urlopen, never None."""
    import ssl

    monkeypatch.setattr(installer.sys, "platform", "linux")

    installer_data = b"fake-appimage"
    info = _make_info(
        [("AutoPTZ-2.2.0-linux.AppImage", "https://example/appimage")],
    )

    ctx_passed: list[ssl.SSLContext | None] = []

    def _opener(request, *, timeout: float, context: ssl.SSLContext | None = None) -> _FakeResponse:
        ctx_passed.append(context)
        return _FakeResponse(installer_data)

    download_update(info, dest_dir=tmp_path, opener=_opener)

    assert len(ctx_passed) == 1, "opener was not called"
    assert ctx_passed[0] is not None, "SSL context must not be None"
    assert isinstance(ctx_passed[0], ssl.SSLContext)


# ── download_update: correct hash → success ───────────────────────────────────


def test_download_correct_hash_succeeds(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(installer.sys, "platform", "linux")

    installer_data = b"real-appimage-bytes"
    good_hash = _sha256(installer_data)
    sha256_content = f"{good_hash}  AutoPTZ-2.2.0-linux.AppImage\n"

    info = _make_info(
        [
            ("AutoPTZ-2.2.0-linux.AppImage", "https://example/appimage"),
            ("SHA256SUMS", "https://example/SHA256SUMS"),
        ],
    )

    urls_fetched: list[str] = []

    def _opener(request, *, timeout: float, context=None) -> _FakeResponse:
        url = request.full_url
        urls_fetched.append(url)
        if "SHA256SUMS" in url:
            return _FakeResponse(sha256_content.encode())
        return _FakeResponse(installer_data)

    result = download_update(info, dest_dir=tmp_path, opener=_opener)
    assert result.path.read_bytes() == installer_data
    assert "https://example/SHA256SUMS" in urls_fetched


# ── download_update: wrong hash → raise, file deleted ────────────────────────


def test_download_wrong_hash_raises_and_removes_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(installer.sys, "platform", "linux")

    installer_data = b"legit-appimage"
    wrong_hash = "0" * 64
    sha256_content = f"{wrong_hash}  AutoPTZ-2.2.0-linux.AppImage\n"

    info = _make_info(
        [
            ("AutoPTZ-2.2.0-linux.AppImage", "https://example/appimage"),
            ("SHA256SUMS", "https://example/SHA256SUMS"),
        ],
    )

    def _opener(request, *, timeout: float, context=None) -> _FakeResponse:
        url = request.full_url
        if "SHA256SUMS" in url:
            return _FakeResponse(sha256_content.encode())
        return _FakeResponse(installer_data)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_update(info, dest_dir=tmp_path, opener=_opener)

    # The downloaded file must be removed so launch_update can never be reached.
    assert not (tmp_path / "AutoPTZ-2.2.0-linux.AppImage").exists()


# ── download_update: no checksum asset → warning, no error ───────────────────


def test_download_no_checksum_warns_and_proceeds(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(installer.sys, "platform", "linux")

    installer_data = b"appimage-no-checksum"
    info = _make_info(
        [("AutoPTZ-2.2.0-linux.AppImage", "https://example/appimage")],
        # No SHA256SUMS asset in the release.
    )

    def _opener(request, *, timeout: float, context=None) -> _FakeResponse:
        return _FakeResponse(installer_data)

    with caplog.at_level(logging.WARNING, logger="autoptz.update.installer"):
        result = download_update(info, dest_dir=tmp_path, opener=_opener)

    assert result.path.read_bytes() == installer_data
    assert any(
        "SHA256SUMS" in record.message or "integrity" in record.message.lower()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ), "Expected a WARNING about missing checksum asset"


# ── download_update: tampered → launch_update is never reached ───────────────


def test_tampered_installer_is_not_launched(tmp_path: Path, monkeypatch) -> None:
    """launch_update must never be called when the hash doesn't match."""
    monkeypatch.setattr(installer.sys, "platform", "linux")

    installer_data = b"tampered-bytes"
    sha256_content = f"{'0' * 64}  AutoPTZ-2.2.0-linux.AppImage\n"

    info = _make_info(
        [
            ("AutoPTZ-2.2.0-linux.AppImage", "https://example/appimage"),
            ("SHA256SUMS", "https://example/SHA256SUMS"),
        ],
    )

    def _opener(request, *, timeout: float, context=None) -> _FakeResponse:
        url = request.full_url
        if "SHA256SUMS" in url:
            return _FakeResponse(sha256_content.encode())
        return _FakeResponse(installer_data)

    launched: list[Path] = []
    monkeypatch.setattr(installer, "launch_update", lambda p: launched.append(p))

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download = download_update(info, dest_dir=tmp_path, opener=_opener)
        installer.launch_update(download.path)

    assert launched == [], "launch_update must not be called after a hash mismatch"
