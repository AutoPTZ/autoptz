"""USB camera hot-plug detection — continuous device add/remove events.

``USBDiscovery`` runs a background polling loop that probes VideoCapture
indices (0–N) and fires registered callbacks whenever the set of available
USB cameras changes.

Platform notes
--------------
- **macOS**: AVFoundation reports only cameras the OS hands to us.
  Plugging a camera triggers a new index becoming openable.
- **Windows**: MSMF device indices are stable while the device is present.
- **Linux**: V4L2 device files appear in ``/dev/video*``.
  If ``pyudev`` is installed the loop is supplemented with real udev events
  for sub-second latency; the poll loop is kept as a safety net.

Callbacks receive ``("added"|"removed", USBDevice)`` and are fired from the
discovery thread — do not do heavy work in them.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2

log = logging.getLogger(__name__)

DiscoveryEvent = Literal["added", "removed"]
USBCallback = Callable[[DiscoveryEvent, "USBDevice"], None]

_MAX_PROBE_INDEX = 10  # probe indices 0..N-1
_PROBE_TIMEOUT_MS = 200  # cv2 property to limit open latency on absent indices


@dataclass(frozen=True)
class USBDevice:
    """Immutable description of one USB camera seen by OpenCV."""

    index: int             # VideoCapture index
    path: str = ""         # device path on Linux, empty on other platforms
    display_name: str = "" # best-effort name (platform-specific)


# ── Platform helpers ──────────────────────────────────────────────────────────

def _cv2_backend_for_probe() -> int:
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        return cv2.CAP_MSMF
    return cv2.CAP_V4L2


def _probe_indices(max_index: int = _MAX_PROBE_INDEX) -> set[int]:
    """Return the set of VideoCapture indices that open successfully."""
    backend = _cv2_backend_for_probe()
    found: set[int] = set()
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            found.add(i)
        cap.release()
    return found


def _v4l2_device_path(index: int) -> str:
    return f"/dev/video{index}"


def _display_name(index: int) -> str:
    """Best-effort device name retrieval (Linux only via v4l2-ctl or name file)."""
    if platform.system() != "Linux":
        return ""
    import subprocess  # noqa: PLC0415
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", f"/dev/video{index}", "--info"],
            capture_output=True, text=True, timeout=1.0,
        )
        for line in result.stdout.splitlines():
            if "Card type" in line:
                return line.split(":", 1)[-1].strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _build_device(index: int) -> USBDevice:
    path = _v4l2_device_path(index) if platform.system() == "Linux" else ""
    return USBDevice(index=index, path=path, display_name=_display_name(index))


# ── Discovery ─────────────────────────────────────────────────────────────────

class USBDiscovery:
    """Continuously monitors USB cameras via polling (all platforms).

    On Linux also hooks ``pyudev`` for faster notification.
    """

    def __init__(self, poll_interval: float = 3.0) -> None:
        self._poll_interval = poll_interval
        self._callbacks: list[USBCallback] = []
        self._known: dict[int, USBDevice] = {}  # index → device
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._udev_monitor: object | None = None

    # ── Callback registration ─────────────────────────────────────────────────

    def on_change(self, callback: USBCallback) -> None:
        """Register a callback fired on device add/remove events."""
        with self._lock:
            self._callbacks.append(callback)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the discovery thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="usb-discovery", daemon=True
        )
        self._thread.start()
        log.info("USBDiscovery started (poll_interval=%.1f s)", self._poll_interval)

    def stop(self) -> None:
        """Stop the discovery thread."""
        self._stop_event.set()
        self._stop_udev()
        if self._thread:
            self._thread.join(timeout=self._poll_interval + 2.0)
            self._thread = None
        log.info("USBDiscovery stopped")

    @property
    def devices(self) -> list[USBDevice]:
        """Return a snapshot of currently known USB cameras."""
        with self._lock:
            return list(self._known.values())

    # ── pyudev integration (Linux) ────────────────────────────────────────────

    def _start_udev(self) -> None:
        if platform.system() != "Linux":
            return
        try:
            import pyudev  # noqa: PLC0415

            context = pyudev.Context()
            monitor = pyudev.Monitor.from_netlink(context)
            monitor.filter_by(subsystem="video4linux")
            monitor.start()
            self._udev_monitor = monitor

            observer_thread = threading.Thread(
                target=self._udev_loop, name="usb-udev", daemon=True
            )
            observer_thread.start()
            log.debug("USBDiscovery: pyudev observer active")
        except ImportError:
            log.debug("USBDiscovery: pyudev not installed; using polling only")
        except Exception as exc:  # noqa: BLE001
            log.debug("USBDiscovery: udev setup failed: %s", exc)

    def _udev_loop(self) -> None:
        monitor = self._udev_monitor
        if monitor is None:
            return
        for _device in monitor:  # type: ignore[union-attr]
            if self._stop_event.is_set():
                break
            # A v4l2 event appeared; re-poll shortly to let the OS settle
            time.sleep(0.3)
            self._poll()

    def _stop_udev(self) -> None:
        self._udev_monitor = None  # udev loop checks stop_event

    # ── Polling loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        self._start_udev()

        # Initial probe
        self._poll()

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self._poll_interval)
            if not self._stop_event.is_set():
                self._poll()

    def _poll(self) -> None:
        """Diff current probe results against known set; fire callbacks."""
        try:
            current_indices = _probe_indices()
        except Exception as exc:  # noqa: BLE001
            log.debug("USBDiscovery poll error: %s", exc)
            return

        with self._lock:
            known = dict(self._known)

        added_indices = current_indices - set(known)
        removed_indices = set(known) - current_indices

        for i in added_indices:
            dev = _build_device(i)
            log.info("USBDiscovery: device added index=%d path=%r", i, dev.path)
            self._fire("added", dev)
            with self._lock:
                self._known[i] = dev

        for i in removed_indices:
            dev = known[i]
            log.info("USBDiscovery: device removed index=%d path=%r", i, dev.path)
            self._fire("removed", dev)
            with self._lock:
                self._known.pop(i, None)

    def _fire(self, event: DiscoveryEvent, device: USBDevice) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(event, device)
            except Exception as exc:  # noqa: BLE001
                log.error("USBDiscovery callback raised: %s", exc)
