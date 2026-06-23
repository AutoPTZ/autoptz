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

import contextlib
import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2

log = logging.getLogger(__name__)

DiscoveryEvent = Literal["added", "removed"]
USBCallback = Callable[[DiscoveryEvent, "USBDevice"], None]

_MAX_PROBE_INDEX = 6  # hard cap; probing also stops at the first absent index
_PROBE_TIMEOUT_MS = 200  # cv2 property to limit open latency on absent indices

# Warn at most once when AVFoundation (pyobjc) is missing, so we don't spam the
# log on every rescan.
_warned_no_avf = False


@contextlib.contextmanager
def _suppressed_stderr():
    """Silence C-level stderr while probing.

    OpenCV prints "OpenCV: out device of bound" / "camera failed to properly
    initialize" straight to the C stderr (fd 2) for absent indices — Python
    logging can't catch it, so we temporarily redirect the fd to /dev/null.
    """
    try:
        fd = sys.stderr.fileno()
    except Exception:  # noqa: BLE001 — no real fd (e.g. captured stderr)
        yield
        return
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


@dataclass(frozen=True)
class USBDevice:
    """Immutable description of one USB camera seen by OpenCV."""

    index: int  # VideoCapture index
    path: str = ""  # device path on Linux, empty on other platforms
    display_name: str = ""  # best-effort name (platform-specific)


# ── Platform helpers ──────────────────────────────────────────────────────────


def _cv2_backend_for_probe() -> int:
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        return cv2.CAP_MSMF
    return cv2.CAP_V4L2


def _probe_indices(max_index: int = _MAX_PROBE_INDEX) -> set[int]:
    """Return the set of VideoCapture indices that actually open **and read**.

    We do not trust ``cap.isOpened()`` alone — on macOS a stale/absent index can
    report opened yet never deliver a frame (the source of the phantom "USB 0-3"
    and the out-of-bound errors).  We confirm one real frame before counting an
    index, and release immediately so we never hold the device open.
    """
    backend = _cv2_backend_for_probe()
    found: set[int] = set()
    # Probe a small, bounded range with C-level stderr silenced — this is what
    # kills the "OpenCV: out device of bound" spam on macOS when AVFoundation
    # (pyobjc) is unavailable. A bad index is skipped, not fatal.
    with _suppressed_stderr():
        for i in range(max_index):
            cap = None
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        found.add(i)
            except Exception:  # noqa: BLE001 — a bad index must not abort the scan
                log.debug("probe index %d raised", i, exc_info=True)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
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
            capture_output=True,
            text=True,
            timeout=1.0,
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


# ── One-shot enumeration (FROZEN contract for the UI's scanUSBCameras) ─────────


def _probed_fallback_cameras() -> list[dict]:
    """Enumerate cameras by probing actually-openable VideoCapture indices.

    Used when platform enumeration (AVFoundation / DirectShow) is unavailable.
    Only indices that open **and** deliver a real frame are returned, so the UI
    never shows phantom "Camera 0-3" entries for indices that can't be opened.

    Each entry is the FROZEN ``{name, unique_id, index, is_continuity}`` dict;
    ``unique_id`` is ``None`` because a bare index has no stable id.
    """
    try:
        indices = sorted(_probe_indices())
    except Exception:  # noqa: BLE001 — probing must never raise
        log.debug("camera index probe failed; returning no devices.", exc_info=True)
        return []
    return [
        {
            "name": f"Camera {i}",
            "unique_id": None,
            "index": i,
            "is_continuity": False,
            "source_label": "USB",
        }
        for i in indices
    ]


def _macos_source_label(av_module: object, device: object, is_continuity: bool) -> str:
    """Map an AVFoundation device to a friendly source label.

    ``Built-in`` / ``Continuity Camera`` / ``Desk View`` / ``External`` — shown in
    the UI instead of the opaque ``usb://N`` index.  Falls back to ``Camera`` when
    the device type can't be read.  Never raises.
    """
    if is_continuity:
        return "Continuity Camera"
    try:
        dtype = device.deviceType()
        label_for = {
            getattr(av_module, "AVCaptureDeviceTypeBuiltInWideAngleCamera", None): "Built-in",
            getattr(av_module, "AVCaptureDeviceTypeDeskViewCamera", None): "Desk View",
            getattr(av_module, "AVCaptureDeviceTypeExternal", None): "External",
            getattr(av_module, "AVCaptureDeviceTypeExternalUnknown", None): "External",
        }
        label = label_for.get(dtype)
        if label:
            return label
    except Exception:  # noqa: BLE001
        pass
    return "Camera"


def _enumerate_macos_cameras() -> list[dict] | None:
    """Enumerate macOS video devices via AVFoundation.

    Returns a list of ``{name, unique_id, index, is_continuity}`` dicts in
    AVFoundation discovery order (which matches OpenCV's ``CAP_AVFOUNDATION``
    index order), or ``None`` if PyObjC / AVFoundation is unavailable so the
    caller can fall back.
    """
    try:
        import AVFoundation  # type: ignore  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — PyObjC / framework not installed
        global _warned_no_avf
        if not _warned_no_avf:
            log.warning(
                "pyobjc-framework-AVFoundation is not installed in this Python — "
                "camera names fall back to generic indices. Install it "
                "(`pip install pyobjc-framework-AVFoundation`) for real device names.",
            )
            _warned_no_avf = True
        return None

    # Build the list of device types to discover.  Older PyObjC / macOS may not
    # expose every constant (e.g. Continuity Camera), so probe defensively.
    type_names = [
        "AVCaptureDeviceTypeBuiltInWideAngleCamera",
        "AVCaptureDeviceTypeExternalUnknown",  # legacy external USB cameras
        "AVCaptureDeviceTypeExternal",  # macOS 14+ external cameras
        "AVCaptureDeviceTypeContinuityCamera",  # iPhone Continuity Camera
        "AVCaptureDeviceTypeDeskViewCamera",  # Continuity Desk View
    ]
    device_types = []
    for name in type_names:
        const = getattr(AVFoundation, name, None)
        if const is not None:
            device_types.append(const)
    if not device_types:
        log.debug("No AVFoundation device-type constants found; falling back.")
        return None

    try:
        media_type = AVFoundation.AVMediaTypeVideo
        session = AVFoundation.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
            device_types,
            media_type,
            AVFoundation.AVCaptureDevicePositionUnspecified,
        )
        devices = session.devices()
    except Exception:  # noqa: BLE001 — any AVFoundation runtime error
        log.debug("AVFoundation discovery session failed; falling back.", exc_info=True)
        return None

    continuity_const = getattr(
        AVFoundation,
        "AVCaptureDeviceTypeContinuityCamera",
        None,
    )

    cameras: list[dict] = []
    seen_ids: set[str] = set()
    for index, dev in enumerate(devices):
        try:
            name = str(dev.localizedName())
            unique_id = str(dev.uniqueID())
        except Exception:  # noqa: BLE001
            name = f"Camera {index}"
            unique_id = None
        # De-duplicate: the same physical device can surface under more than one
        # requested device-type bucket.  Capture now binds by ``unique_id``, so a
        # duplicate id is the *same* camera — list it once.
        if unique_id is not None and unique_id in seen_ids:
            continue
        if unique_id is not None:
            seen_ids.add(unique_id)
        is_continuity = False
        try:
            if continuity_const is not None and dev.deviceType() == continuity_const:
                is_continuity = True
            elif "continuity" in name.lower() or "iphone" in name.lower():
                is_continuity = True
        except Exception:  # noqa: BLE001
            pass
        # Keep ``name`` the raw AVFoundation ``localizedName`` (FROZEN contract) —
        # the UI layer (``scanUSBCameras``) is what appends a Continuity label, so
        # suffixing here would double it.  ``is_continuity`` carries the flag.
        cameras.append(
            {
                "name": name,
                "unique_id": unique_id,
                "index": index,
                "is_continuity": is_continuity,
                "source_label": _macos_source_label(AVFoundation, dev, is_continuity),
            }
        )
    return cameras


def usb_enumeration_is_cheap() -> bool:
    """True when camera enumeration is a cheap metadata listing (no device opens).

    macOS + AVFoundation lists devices via a discovery session *without* opening
    them, so it is safe for the UI to poll periodically for hotplug.  The
    cross-platform fallback (:func:`_probed_fallback_cameras`) instead OPENS each
    ``VideoCapture`` index to confirm a real frame — far too costly to poll — so
    callers should refresh on-demand only when this returns ``False``.
    """
    if platform.system() != "Darwin":
        return False
    try:
        import AVFoundation  # type: ignore  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001 — PyObjC / framework not installed
        return False
    return True


def enumerate_cameras() -> list[dict]:
    """Return discovered video cameras as ``{name, unique_id, index, is_continuity}``.

    FROZEN contract — the UI's ``scanUSBCameras()`` consumes exactly these keys:

    - ``name`` (``str``):           human-facing device name.
    - ``unique_id`` (``str|None``): stable per-device id (macOS ``uniqueID``); may
                                    be ``None`` on platforms / fallbacks without one.
    - ``index`` (``int``):          enumeration index to open the device with
                                    (matches OpenCV's capture index order).
    - ``is_continuity`` (``bool``): ``True`` for an iPhone Continuity Camera.
    - ``source_label`` (``str``):   friendly source kind ("Built-in", "Continuity
                                    Camera", "Desk View", "External", "USB") shown
                                    in the UI instead of the opaque ``usb://N``.

    macOS uses AVFoundation; if PyObjC / AVFoundation is unavailable (or on
    other platforms) it gracefully falls back to **probing actually-openable
    VideoCapture indices** (confirming a real frame each) rather than blindly
    listing 0–N.  Never raises.
    """
    if platform.system() == "Darwin":
        try:
            cams = _enumerate_macos_cameras()
        except Exception:  # noqa: BLE001 — enumeration must never raise
            log.debug("macOS camera enumeration raised; falling back.", exc_info=True)
            cams = None
        if cams is not None:
            return cams
    # TODO(Windows): add a native enumerator for friendly device names.
    # Fallback (no platform enumeration): only real, openable indices.
    return _probed_fallback_cameras()


def cameras_by_unique_id() -> dict[str, dict]:
    """Return discovered cameras keyed by ``unique_id`` (entries that have one).

    A convenience over :func:`enumerate_cameras` for the capture layer (e.g.
    ``pipeline/avf_capture.py``), which binds devices by ``uniqueID`` rather than
    index — this lets it confirm a chosen id is still present, and read its name /
    Continuity flag, without re-implementing AVFoundation enumeration.  Entries
    whose ``unique_id`` is ``None`` (index-only fallbacks) are omitted.  Never
    raises.
    """
    out: dict[str, dict] = {}
    try:
        for cam in enumerate_cameras():
            uid = cam.get("unique_id")
            if uid:
                out[uid] = cam
    except Exception:  # noqa: BLE001 — enumeration must never raise
        log.debug("cameras_by_unique_id enumeration failed", exc_info=True)
    return out


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
        self._thread = threading.Thread(target=self._run, name="usb-discovery", daemon=True)
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

            observer_thread = threading.Thread(target=self._udev_loop, name="usb-udev", daemon=True)
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
