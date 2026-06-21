"""Continuous NDI source discovery via cyndilib Finder.

``NDIDiscovery`` runs a background thread that polls the NDI network for
sources and fires registered callbacks whenever a source appears or disappears.
Callbacks receive ``("added"|"removed", NDISource)``.

NDI sources can join or leave at any time (e.g. an encoder starts/stops),
so this runs continuously — not just at startup.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

DiscoveryEvent = Literal["added", "removed"]
NDICallback = Callable[[DiscoveryEvent, "NDISource"], None]


@dataclass(frozen=True)
class NDISource:
    """Immutable description of one NDI source seen on the network."""

    name: str  # full NDI name, e.g. "LAPTOP (NDI CAMERA)"
    url_address: str = ""  # optional host:port hint from the SDK


class NDIDiscovery:
    """Continuously discovers NDI sources on the local network.

    Uses a cyndilib ``Finder`` polled at ``poll_interval`` seconds.
    Fires ``on_change`` callbacks from the discovery thread; callers
    should not do heavy work in callbacks.

    cyndilib (and the NDI SDK runtime) must be installed.  If not present,
    ``start()`` logs a warning and returns immediately without raising.
    """

    def __init__(self, poll_interval: float = 2.0) -> None:
        self._poll_interval = poll_interval
        self._callbacks: list[NDICallback] = []
        self._known: dict[str, NDISource] = {}  # name → source
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Callback registration ─────────────────────────────────────────────────

    def on_change(self, callback: NDICallback) -> None:
        """Register a callback fired on source add/remove events."""
        with self._lock:
            self._callbacks.append(callback)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the discovery thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        try:
            import cyndilib  # noqa: F401  # verify available before starting thread
        except ImportError:
            log.warning(
                "NDIDiscovery: cyndilib not available; NDI discovery disabled. "
                "Install cyndilib and the NDI SDK runtime."
            )
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ndi-discovery", daemon=True)
        self._thread.start()
        log.info("NDIDiscovery started (poll_interval=%.1f s)", self._poll_interval)

    def stop(self) -> None:
        """Stop the discovery thread and block until it exits."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._poll_interval + 2.0)
            self._thread = None
        log.info("NDIDiscovery stopped")

    @property
    def sources(self) -> list[NDISource]:
        """Return a snapshot of currently known NDI sources."""
        with self._lock:
            return list(self._known.values())

    # ── Discovery loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            from cyndilib.finder import Finder  # noqa: PLC0415
        except ImportError:
            log.error("NDIDiscovery: cyndilib unavailable in thread; exiting")
            return

        finder = Finder()
        finder.open()

        # Initial settle: NDI sources may not appear immediately after open
        time.sleep(min(self._poll_interval, 1.0))

        try:
            while not self._stop_event.is_set():
                self._poll(finder)
                self._stop_event.wait(timeout=self._poll_interval)
        finally:
            finder.close()

    def _poll(self, finder: object) -> None:
        """Compare current Finder results against known set; fire callbacks."""
        try:
            current: dict[str, NDISource] = {}
            for src in finder.iter_sources():  # type: ignore[union-attr]
                name = str(src)
                url = getattr(src, "url_address", "")
                current[name] = NDISource(name=name, url_address=str(url))
        except Exception as exc:  # noqa: BLE001
            log.debug("NDIDiscovery poll error: %s", exc)
            return

        with self._lock:
            known = dict(self._known)

        added = {n: s for n, s in current.items() if n not in known}
        removed = {n: s for n, s in known.items() if n not in current}

        for source in added.values():
            log.info("NDIDiscovery: source added %r", source.name)
            self._fire("added", source)

        for source in removed.values():
            log.info("NDIDiscovery: source removed %r", source.name)
            self._fire("removed", source)

        if added or removed:
            with self._lock:
                self._known = current

    def _fire(self, event: DiscoveryEvent, source: NDISource) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(event, source)
            except Exception as exc:  # noqa: BLE001
                log.error("NDIDiscovery callback raised: %s", exc)
