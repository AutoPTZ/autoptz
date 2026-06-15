"""ONVIF WS-Discovery for IP cameras — continuous add/remove events.

``ONVIFDiscovery`` uses WS-Discovery (RFC 4/WS-DD) multicast probes
(239.255.255.250:3702) to find ONVIF-capable devices on the LAN.

Discovery runs continuously:
- An initial probe fires on ``start()``.
- Subsequent probes repeat every ``rescan_interval`` seconds to catch cameras
  that come online after startup.
- Devices that stop responding for ``miss_threshold`` consecutive scans are
  considered removed.

If ``wsdiscovery`` is not installed, ``start()`` logs a warning and returns
without raising (ONVIF discovery is best-effort; cameras can also be added
manually via RTSP URL).

Callbacks receive ``("added"|"removed", ONVIFDevice)`` from the discovery
thread — do not do heavy work inside them.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

log = logging.getLogger(__name__)

DiscoveryEvent = Literal["added", "removed"]
ONVIFCallback = Callable[[DiscoveryEvent, "ONVIFDevice"], None]

# Remove a device after this many missed scans (to handle transient losses)
_MISS_THRESHOLD = 3


@dataclass(frozen=True)
class ONVIFDevice:
    """Immutable description of one ONVIF device found via WS-Discovery."""

    xaddrs: tuple[str, ...]  # ONVIF service endpoint URLs
    types: tuple[str, ...]   # WS-Discovery device types
    scopes: tuple[str, ...]  # WS-Discovery scopes (contain make/model hints)

    @property
    def primary_xaddr(self) -> str:
        return self.xaddrs[0] if self.xaddrs else ""

    @property
    def host(self) -> str:
        url = self.primary_xaddr
        try:
            return urlparse(url).hostname or url
        except Exception:  # noqa: BLE001
            return url


def _service_key(xaddrs: list[str]) -> str:
    """Stable key for a discovered service (first xaddr, normalised)."""
    return xaddrs[0].rstrip("/") if xaddrs else ""


class ONVIFDiscovery:
    """Continuously discovers ONVIF cameras via WS-Discovery multicast.

    Requires the ``wsdiscovery`` package (``pip install wsdiscovery``).
    """

    def __init__(self, rescan_interval: float = 30.0) -> None:
        self._rescan_interval = rescan_interval
        self._callbacks: list[ONVIFCallback] = []
        self._known: dict[str, ONVIFDevice] = {}   # key → device
        self._misses: dict[str, int] = {}           # key → missed scan count
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Callback registration ─────────────────────────────────────────────────

    def on_change(self, callback: ONVIFCallback) -> None:
        """Register a callback fired on device add/remove events."""
        with self._lock:
            self._callbacks.append(callback)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the discovery thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        try:
            import wsdiscovery  # noqa: F401
        except ImportError:
            log.warning(
                "ONVIFDiscovery: wsdiscovery not installed; ONVIF auto-discovery disabled. "
                "Install with: pip install wsdiscovery"
            )
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="onvif-discovery", daemon=True
        )
        self._thread.start()
        log.info(
            "ONVIFDiscovery started (rescan_interval=%.0f s)", self._rescan_interval
        )

    def stop(self) -> None:
        """Stop the discovery thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._rescan_interval + 5.0)
            self._thread = None
        log.info("ONVIFDiscovery stopped")

    @property
    def devices(self) -> list[ONVIFDevice]:
        """Return a snapshot of currently known ONVIF devices."""
        with self._lock:
            return list(self._known.values())

    # ── Discovery loop ────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            from wsdiscovery import WSDiscovery  # noqa: PLC0415
        except ImportError:
            log.error("ONVIFDiscovery: wsdiscovery unavailable in thread; exiting")
            return

        wsd = WSDiscovery()
        wsd.start()

        try:
            # First scan immediately
            self._scan(wsd)

            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=self._rescan_interval)
                if not self._stop_event.is_set():
                    self._scan(wsd)
        finally:
            wsd.stop()

    def _scan(self, wsd: object) -> None:
        """Run a WS-Discovery search and diff against the known set."""
        try:
            services = wsd.searchServices(timeout=3)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            log.debug("ONVIFDiscovery scan error: %s", exc)
            return

        current_keys: set[str] = set()

        for svc in services:
            try:
                xaddrs = list(svc.getXAddrs() or [])
                if not xaddrs:
                    continue
                key = _service_key(xaddrs)
                types = tuple(str(t) for t in (svc.getTypes() or []))
                scopes = tuple(str(s) for s in (svc.getScopes() or []))
                current_keys.add(key)

                device = ONVIFDevice(
                    xaddrs=tuple(xaddrs),
                    types=types,
                    scopes=scopes,
                )

                with self._lock:
                    is_new = key not in self._known
                    self._known[key] = device
                    self._misses.pop(key, None)

                if is_new:
                    log.info(
                        "ONVIFDiscovery: device added host=%r xaddr=%r",
                        device.host, device.primary_xaddr,
                    )
                    self._fire("added", device)

            except Exception as exc:  # noqa: BLE001
                log.debug("ONVIFDiscovery: error parsing service: %s", exc)

        # Increment miss counter for absent devices; remove after threshold.
        # Collect removals first so _fire is called outside the lock.
        to_remove: list[ONVIFDevice] = []
        with self._lock:
            stale_keys = set(self._known) - current_keys
            for key in stale_keys:
                self._misses[key] = self._misses.get(key, 0) + 1
                if self._misses[key] >= _MISS_THRESHOLD:
                    dev = self._known.pop(key)
                    self._misses.pop(key, None)
                    log.info("ONVIFDiscovery: device removed host=%r", dev.host)
                    to_remove.append(dev)

        for dev in to_remove:
            self._fire("removed", dev)

    def _fire(self, event: DiscoveryEvent, device: ONVIFDevice) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(event, device)
            except Exception as exc:  # noqa: BLE001
                log.error("ONVIFDiscovery callback raised: %s", exc)
