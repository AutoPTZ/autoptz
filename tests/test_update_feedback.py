"""Update-check feedback: distinct outcomes, error classification, arch safety.

Covers the changes that make the updater observable and correct:

* ``check_for_update_result`` returns a 3-way result — update-available /
  up-to-date / failed(reason) — instead of collapsing "no update" and "network
  error" into a single ``None``.
* Network/timeout/TLS/rate-limit/parse failures are classified with a
  user-facing message (so the UI can say *why* it failed, not "up to date").
* ``asset_for_platform`` never hands an Intel Mac an arm64-only build (and vice
  versa) — it returns ``None`` rather than the wrong architecture.
* ``UpdateManager`` routes a failed check to ``checkFailed`` (not ``upToDate``),
  emits ``checkStarted``, and only arms the 24h throttle on a *successful* check.
"""

from __future__ import annotations

import autoptz.update.checker as checker
from autoptz.update.checker import CheckResult, UpdateInfo, check_for_update_result


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


def _ok(monkeypatch, releases: list[dict]) -> None:
    monkeypatch.setattr(checker, "_fetch_releases", lambda: (releases, None, None))


def _fail(monkeypatch, kind: str, msg: str) -> None:
    monkeypatch.setattr(checker, "_fetch_releases", lambda: (None, kind, msg))


# ─────────────────────────────────────────────────────────────────────────────
# Structured result: update_available / up_to_date / failed
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckResult:
    def test_update_available(self, monkeypatch) -> None:
        _ok(monkeypatch, [_release("v2.1.0")])
        r = check_for_update_result("2.0.0")
        assert r.status == "update_available"
        assert r.has_update is True
        assert isinstance(r.info, UpdateInfo) and r.info.version == "2.1.0"

    def test_up_to_date_is_distinct_from_failure(self, monkeypatch) -> None:
        _ok(monkeypatch, [_release("v2.0.0")])
        r = check_for_update_result("2.0.0")
        assert r.status == "up_to_date"
        assert r.has_update is False
        assert r.error is None and r.error_kind is None

    def test_network_failure_is_failed_not_up_to_date(self, monkeypatch) -> None:
        _fail(monkeypatch, "network", "Couldn't reach the update server.")
        r = check_for_update_result("2.0.0")
        assert r.status == "failed"
        assert r.error_kind == "network"
        assert r.error and "reach" in r.error.lower()

    def test_tls_failure_classified(self, monkeypatch) -> None:
        _fail(monkeypatch, "tls", "Couldn't verify a secure connection.")
        r = check_for_update_result("2.0.0")
        assert r.status == "failed" and r.error_kind == "tls"

    def test_bad_current_version_is_failed(self, monkeypatch) -> None:
        _ok(monkeypatch, [_release("v2.1.0")])
        r = check_for_update_result("not-a-version")
        assert r.status == "failed" and r.error_kind == "bad_version"


# ─────────────────────────────────────────────────────────────────────────────
# Error classification at the fetch boundary
# ─────────────────────────────────────────────────────────────────────────────


class TestFetchClassification:
    def test_tls_error_classified_as_tls(self, monkeypatch) -> None:
        import urllib.error

        def boom(req, timeout, context=None):  # noqa: ARG001
            raise urllib.error.URLError(
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
            )

        monkeypatch.setattr(checker.urllib.request, "urlopen", boom)
        releases, kind, msg = checker._fetch_releases()
        assert releases is None
        assert kind == "tls"
        assert msg

    def test_timeout_classified(self, monkeypatch) -> None:
        def boom(req, timeout, context=None):  # noqa: ARG001
            raise TimeoutError("timed out")

        monkeypatch.setattr(checker.urllib.request, "urlopen", boom)
        _releases, kind, _msg = checker._fetch_releases()
        assert kind == "timeout"

    def test_generic_urlerror_is_network(self, monkeypatch) -> None:
        import urllib.error

        def boom(req, timeout, context=None):  # noqa: ARG001
            raise urllib.error.URLError("Name or service not known")

        monkeypatch.setattr(checker.urllib.request, "urlopen", boom)
        _releases, kind, _msg = checker._fetch_releases()
        assert kind == "network"


# ─────────────────────────────────────────────────────────────────────────────
# Intel-Mac architecture safety
# ─────────────────────────────────────────────────────────────────────────────


def _info(assets: tuple[tuple[str, str], ...]) -> UpdateInfo:
    return UpdateInfo(
        version="2.3.0",
        tag="v2.3.0",
        name="x",
        body="",
        html_url="https://example/rel",
        is_prerelease=False,
        assets=assets,
    )


class TestArchSafety:
    def test_intel_never_gets_arm64_only_build(self, monkeypatch) -> None:
        import platform as _platform

        monkeypatch.setattr(checker.sys, "platform", "darwin")
        monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
        info = _info((("AutoPTZ-2.3.0-macos-arm64.dmg", "https://example/arm"),))
        # No x86_64 asset → must NOT fall back to the arm64 binary.
        assert info.asset_for_platform() is None

    def test_arm_never_gets_intel_only_build(self, monkeypatch) -> None:
        import platform as _platform

        monkeypatch.setattr(checker.sys, "platform", "darwin")
        monkeypatch.setattr(_platform, "machine", lambda: "arm64")
        info = _info((("AutoPTZ-2.3.0-macos-x86_64.dmg", "https://example/intel"),))
        assert info.asset_for_platform() is None

    def test_intel_alias_matches(self, monkeypatch) -> None:
        import platform as _platform

        monkeypatch.setattr(checker.sys, "platform", "darwin")
        monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
        info = _info((("AutoPTZ-2.3.0-macos-intel.dmg", "https://example/intel"),))
        assert info.asset_for_platform() == (
            "AutoPTZ-2.3.0-macos-intel.dmg",
            "https://example/intel",
        )

    def test_unmarked_dmg_still_resolves(self, monkeypatch) -> None:
        import platform as _platform

        monkeypatch.setattr(checker.sys, "platform", "darwin")
        monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
        info = _info((("AutoPTZ-2.0.0.dmg", "https://example/any"),))
        assert info.asset_for_platform() == ("AutoPTZ-2.0.0.dmg", "https://example/any")


# ─────────────────────────────────────────────────────────────────────────────
# UpdateManager routing + throttle
# ─────────────────────────────────────────────────────────────────────────────


def _manager(store: dict, current: str = "2.0.0"):
    from PySide6.QtCore import QCoreApplication

    from autoptz.ui.update_manager import UpdateManager

    if QCoreApplication.instance() is None:
        QCoreApplication([])

    class _Client:
        def getSetting(self, key, default):
            return store.get(key, default)

        def setSetting(self, key, value):
            store[key] = value

    return UpdateManager(_Client(), current)


class TestManagerRouting:
    def test_failed_check_routes_to_checkFailed_not_upToDate(self, monkeypatch) -> None:
        mgr = _manager({})
        seen = {"failed": None, "uptodate": 0}
        mgr.checkFailed.connect(lambda msg, manual: seen.__setitem__("failed", (msg, manual)))
        mgr.upToDate.connect(lambda manual: seen.__setitem__("uptodate", seen["uptodate"] + 1))
        mgr._on_finished(CheckResult(status="failed", error="offline", error_kind="network"), True)
        assert seen["failed"] == ("offline", True)
        assert seen["uptodate"] == 0

    def test_up_to_date_routes_to_upToDate(self) -> None:
        mgr = _manager({})
        seen = {"uptodate": None}
        mgr.upToDate.connect(lambda manual: seen.__setitem__("uptodate", manual))
        mgr._on_finished(CheckResult(status="up_to_date"), True)
        assert seen["uptodate"] is True

    def test_throttle_not_armed_on_failure(self) -> None:
        store: dict = {}
        mgr = _manager(store)
        mgr._on_finished(CheckResult(status="failed", error="x", error_kind="network"), False)
        # A failed check must NOT stamp last-check (else startup won't retry for 24h).
        assert "update_last_check" not in store or not store["update_last_check"]

    def test_throttle_armed_on_success(self) -> None:
        store: dict = {}
        mgr = _manager(store)
        mgr._on_finished(CheckResult(status="up_to_date"), False)
        assert float(store.get("update_last_check", 0)) > 0
