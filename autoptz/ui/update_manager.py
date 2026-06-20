"""Background update checks wired to the GUI (notify-only).

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
    finished = Signal(object, bool)  # (UpdateInfo | None, manual)


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
        from autoptz.update.checker import check_for_update

        try:
            info = check_for_update(self._current, include_prereleases=self._pre)
        except Exception:  # noqa: BLE001 — never let a worker crash the pool
            log.debug("update check task failed", exc_info=True)
            info = None
        self._signals.finished.emit(info, self._manual)


class UpdateManager(QObject):
    """Owns update-check scheduling, settings, and result routing."""

    updateAvailable = Signal(object)  # UpdateInfo
    upToDate = Signal(bool)  # manual?

    def __init__(self, client: Any, current_version: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._current = current_version
        self._signals = _CheckSignals(self)
        self._signals.finished.connect(self._on_finished)

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
        return bool(self._get("update_include_prereleases", True))

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
        self._set("update_last_check", time.time())
        task = _CheckTask(self._current, self.include_prereleases, manual, self._signals)
        QThreadPool.globalInstance().start(task)

    @Slot(object, bool)
    def _on_finished(self, info: object, manual: bool) -> None:
        if info is None:
            self.upToDate.emit(manual)
            return
        version = getattr(info, "version", "")
        if not manual:
            skip = str(self._get("update_skip_version", "") or "")
            if skip and skip == version:
                log.info("update %s available but skipped by user", version)
                return
        self.updateAvailable.emit(info)
