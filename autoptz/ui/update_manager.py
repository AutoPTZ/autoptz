"""Background update checks and downloads wired to the GUI.

Runs :func:`autoptz.update.checker.check_for_update` off the GUI thread via
``QThreadPool`` and emits results back on the GUI thread. Persists its prefs
(auto-check, last-check time, skipped version) through the engine client's
settings API.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

log = logging.getLogger(__name__)

#: Don't auto-check more than once per this interval (manual checks ignore it).
_CHECK_INTERVAL_S = 24 * 3600


class _CheckSignals(QObject):
    finished = Signal(object, bool)  # (CheckResult, manual)


class _DownloadSignals(QObject):
    progress = Signal(int, int)  # (bytes_downloaded, total_bytes)
    finished = Signal(object)  # DownloadedUpdate
    failed = Signal(str)


class _CheckTask(QRunnable):
    def __init__(
        self, current: str, include_prereleases: bool, manual: bool, signals: _CheckSignals
    ) -> None:
        super().__init__()
        self._current = current
        self._pre = include_prereleases
        self._manual = manual
        self._signals = signals

    def run(self) -> None:
        from autoptz.update.checker import CheckResult, check_for_update_result

        try:
            result = check_for_update_result(self._current, include_prereleases=self._pre)
        except Exception as exc:  # noqa: BLE001 — never let a worker crash the pool
            log.debug("update check task failed", exc_info=True)
            result = CheckResult(
                status="failed", error_kind="network", error=str(exc) or "Update check failed."
            )
        self._signals.finished.emit(result, self._manual)


class _DownloadTask(QRunnable):
    def __init__(self, info: object, signals: _DownloadSignals) -> None:
        super().__init__()
        self._info = info
        self._signals = signals

    def run(self) -> None:
        from autoptz.update.installer import download_update

        try:
            result = download_update(
                self._info,  # type: ignore[arg-type]
                progress=lambda done, total: self._signals.progress.emit(int(done), int(total)),
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("update download failed", exc_info=True)
            self._signals.failed.emit(str(exc) or "Update download failed.")
            return
        self._signals.finished.emit(result)


class UpdateManager(QObject):
    """Owns update-check scheduling, settings, and result routing."""

    checkStarted = Signal(bool)  # manual? — a check just began (show "Checking…")
    updateAvailable = Signal(object)  # UpdateInfo
    upToDate = Signal(bool)  # manual?
    checkFailed = Signal(str, bool)  # (reason, manual?) — distinct from upToDate
    downloadStarted = Signal(object)  # UpdateInfo
    downloadProgress = Signal(int, int)  # (bytes_downloaded, total_bytes)
    downloadFinished = Signal(object)  # DownloadedUpdate
    downloadFailed = Signal(str)

    def __init__(self, client: Any, current_version: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._current = current_version
        self._signals = _CheckSignals(self)
        self._download_signals = _DownloadSignals(self)
        self._signals.finished.connect(self._on_finished)
        self._download_signals.progress.connect(self.downloadProgress.emit)
        self._download_signals.finished.connect(self.downloadFinished.emit)
        self._download_signals.failed.connect(self.downloadFailed.emit)

    # ── settings ────────────────────────────────────────────────────────────────
    def _get(self, key: str, default: Any) -> Any:
        try:
            return self._client.getSetting(key, default)
        except Exception:  # noqa: BLE001
            return default

    def _set(self, key: str, value: Any) -> None:
        try:
            self._client.setSetting(key, value)
        except Exception:  # noqa: BLE001
            log.debug("could not persist %s", key, exc_info=True)

    @property
    def auto_check_enabled(self) -> bool:
        return bool(self._get("update_auto_check", True))

    def set_auto_check(self, on: bool) -> None:
        self._set("update_auto_check", bool(on))

    @property
    def include_prereleases(self) -> bool:
        # Opt-in: stable users are only offered stable releases. Matches the
        # checker's own default (check_for_update(..., include_prereleases=False))
        # and the release workflow's "opted into pre-releases" intent.
        return bool(self._get("update_include_prereleases", False))

    def set_include_prereleases(self, on: bool) -> None:
        self._set("update_include_prereleases", bool(on))

    def skip_version(self, version: str) -> None:
        self._set("update_skip_version", version)

    # ── checks ──────────────────────────────────────────────────────────────────
    def maybe_check_on_startup(self) -> None:
        """Auto-check if enabled and not checked within the throttle window."""
        if not self.auto_check_enabled:
            return
        last = float(self._get("update_last_check", 0) or 0)
        if time.time() - last < _CHECK_INTERVAL_S:
            return
        self._start(manual=False)

    def check_now(self) -> None:
        """Manual check (Help menu): always runs, always reports a result."""
        self._start(manual=True)

    def _start(self, *, manual: bool) -> None:
        # NB: the 24h throttle is stamped on SUCCESS (in _on_finished), not here —
        # a failed/offline check must not suppress the next startup retry.
        self.checkStarted.emit(manual)
        task = _CheckTask(self._current, self.include_prereleases, manual, self._signals)
        QThreadPool.globalInstance().start(task)

    def download(self, info: object) -> None:
        """Download the OS-specific update asset in the background."""
        self.downloadStarted.emit(info)
        task = _DownloadTask(info, self._download_signals)
        QThreadPool.globalInstance().start(task)

    @Slot(object, bool)
    def _on_finished(self, result: object, manual: bool) -> None:
        status = getattr(result, "status", "failed")

        # A failed check is reported as such — never as "up to date" — and does
        # NOT arm the 24h throttle, so the next launch retries.
        if status == "failed":
            reason = str(getattr(result, "error", "") or "The update check could not be completed.")
            log.info("update check failed (%s): %s", getattr(result, "error_kind", "?"), reason)
            self.checkFailed.emit(reason, manual)
            return

        # Successful check (up-to-date or update-available) → remember when.
        self._set("update_last_check", time.time())

        if status != "update_available":
            self.upToDate.emit(manual)
            return

        info = getattr(result, "info", None)
        version = getattr(info, "version", "")
        if not manual:
            skip = str(self._get("update_skip_version", "") or "")
            if skip and skip == version:
                log.info("update %s available but skipped by user", version)
                return
        self.updateAvailable.emit(info)
