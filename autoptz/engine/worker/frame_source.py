"""Synchronous frame-source abstraction + fps pacing for the camera worker.

The worker owns the capture loop (so it can also feed detection), driving an
ingest ``SourceAdapter`` through this thin synchronous wrapper. ``_pace_read``
throttles worker-owned reads to the configured target fps with a deadline
accumulator (see its docstring for why the naive sleep-then-read doubled the
period).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from autoptz.config.models import TrackingConfig

if TYPE_CHECKING:
    from autoptz.config.models import CameraConfig

log = logging.getLogger(__name__)


def _resolve_framing(tracking: TrackingConfig) -> str:
    """Return the framing preset for aim/pose.

    ``tracking.framing`` is the single user-facing control for shot composition.
    """
    return getattr(tracking, "framing", TrackingConfig.model_fields["framing"].default)


# ── frame source abstraction ───────────────────────────────────────────────────


@runtime_checkable
class FrameSource(Protocol):
    """Minimal synchronous frame source the worker drives itself.

    Implementations open a device/stream, hand back one BGR frame per
    ``read()`` (or ``None`` on a transient miss), and release on ``close()``.
    Tests inject a fake implementation; production wraps the ingest adapters.
    """

    def open(self) -> bool:
        """Open the source.  Return ``True`` on success."""
        ...

    def read(self) -> NDArray[np.uint8] | None:
        """Return one BGR (H, W, 3) frame, or ``None`` on a transient miss."""
        ...

    def close(self) -> None:
        """Release the source."""
        ...


class _AdapterFrameSource:
    """Adapt an ingest ``SourceAdapter`` subclass to the synchronous FrameSource.

    The ingest adapters expose synchronous ``_open`` / ``_read_frame`` /
    ``_close`` primitives (their own capture thread is *not* started here — the
    worker owns the loop so it can also feed detection).
    """

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        # Next wall-clock instant a read should fire (monotonic seconds). 0.0 =
        # cadence not yet started / needs a resync on the next read.
        self._next_deadline = 0.0

    def open(self) -> bool:
        ok = bool(self._adapter._open())
        # Restart the pacing cadence cleanly after every (re)connect so a stall
        # gap doesn't leave a stale deadline that bursts catch-up frames.
        self._next_deadline = 0.0
        return ok

    def read(self) -> NDArray[np.uint8] | None:
        self._pace_read()
        return self._adapter._read_frame()

    def close(self) -> None:
        try:
            self._adapter._close()
        except Exception:  # noqa: BLE001
            log.debug("frame source _close raised", exc_info=True)

    def set_target_fps(self, fps: float) -> None:
        """Forward a live fps change to the wrapped ingest adapter (best-effort)."""
        fn = getattr(self._adapter, "set_target_fps", None)
        if callable(fn):
            fn(fps)
        # Re-anchor the cadence so the new rate takes effect from the next frame
        # rather than inheriting the old deadline.
        self._next_deadline = 0.0

    def _pace_read(self) -> None:
        """Pace worker-owned reads to the adapter's target fps with a deadline
        accumulator.

        The previous implementation slept a full ``1/target`` period measured
        from when the *previous read completed*, then ``_read_frame`` blocked
        again waiting for the next hardware frame — the two stacked and roughly
        halved the delivered rate (slider 30 → ~15, 15 → ~10).

        Instead we advance a fixed ``next_deadline`` by one period each frame and
        only sleep until that instant.  A read that blocks (OpenCV) or the rest
        of the worker loop absorbs into the period instead of adding to it: when
        the target meets/exceeds the source rate we never sleep and the hardware
        paces us; when the target is lower we throttle down to exactly it.
        """
        target = float(getattr(self._adapter, "_target_fps", 0.0) or 0.0)
        if target <= 0.0:
            self._next_deadline = 0.0
            return
        period = 1.0 / max(1.0, target)
        now = time.monotonic()
        if self._next_deadline <= 0.0:
            # First frame after (re)connect or rate change: anchor from now.
            self._next_deadline = now + period
            return
        wait = self._next_deadline - now
        if wait > 0.001:
            time.sleep(wait)
            now = self._next_deadline
        self._next_deadline += period
        # If we've fallen more than a period behind (slow read / stall / paused
        # worker), resync to the present so we don't fire a burst of frames.
        if self._next_deadline < now - period:
            self._next_deadline = now + period

    def source_fps_cap(self) -> float | None:
        """Return the adapter's detected hardware fps ceiling, or ``None``."""
        try:
            return self._adapter.status.source_fps_cap
        except Exception:  # noqa: BLE001
            return None


def build_frame_source(camera_id: str, config: CameraConfig) -> FrameSource:
    """Construct the right ingest-adapter-backed FrameSource for *config*.

    Importing ``ingest`` pulls in ``cv2``; if that is unavailable we raise so
    the worker can fall back to a no-signal state without crashing the engine.
    """
    from autoptz.engine.pipeline.ingest import (
        NDIAdapter,
        RTSPAdapter,
        SyntheticAdapter,
        USBAdapter,
    )

    source = config.source
    target_fps = source.fps
    stall = config.reconnect.stall_timeout_s

    if source.type == "synthetic":
        adapter: Any = SyntheticAdapter(
            camera_id,
            address=source.address,
            target_fps=target_fps,
            stall_timeout=stall,
        )
    elif source.type == "usb":
        dev = _resolve_usb_device(source)
        adapter = USBAdapter(
            camera_id,
            source=dev,
            target_fps=target_fps,
            stall_timeout=stall,
            unique_id=getattr(source, "unique_id", None),
        )
    elif source.type in ("rtsp", "onvif"):
        adapter = RTSPAdapter(
            camera_id, url=source.address, target_fps=target_fps, stall_timeout=stall
        )
    elif source.type == "ndi":
        adapter = NDIAdapter(
            camera_id,
            ndi_name=_strip_scheme(source.address, "ndi://"),
            target_fps=target_fps,
            stall_timeout=stall,
        )
    else:  # pragma: no cover - validated by pydantic Literal
        adapter = USBAdapter(camera_id, source=0, target_fps=target_fps, stall_timeout=stall)

    return _AdapterFrameSource(adapter)


def _resolve_usb_device(source: Any) -> int | str:
    """Resolve the *fallback* cv2 source (index/path) for a USB ``SourceConfig``.

    This is only the fallback the ``USBAdapter`` uses when it cannot resolve a
    stable ``unique_id`` to a verified capture index at open time (off macOS, or
    when the device is gone).  When a ``unique_id`` is present we still pre-seed
    a best-effort current index from discovery so a non-macOS adapter — which
    has no uniqueID resolution — opens the right enumeration slot; macOS gets the
    authoritative verified resolution inside :class:`USBAdapter`.
    """
    unique_id = getattr(source, "unique_id", None)
    if unique_id:
        idx = _index_for_unique_id(unique_id)
        if idx is not None:
            return idx
        log.debug(
            "USB unique_id=%s not in current enumeration; falling back to address index.",
            unique_id,
        )
    return _parse_usb_index(getattr(source, "address", ""))


def _index_for_unique_id(unique_id: str) -> int | None:
    """Look up the enumerated camera index whose ``unique_id`` matches.

    Returns ``None`` if enumeration is unavailable or no device matches.
    """
    try:
        from autoptz.engine.discovery.usb import enumerate_cameras

        for cam in enumerate_cameras():
            if cam.get("unique_id") == unique_id:
                return int(cam["index"])
    except Exception:  # noqa: BLE001 — enumeration must never break source build
        log.debug("USB enumeration lookup failed for unique_id=%s", unique_id, exc_info=True)
    return None


def _parse_usb_index(address: str) -> int | str:
    """Map a ``usb://N`` address (or bare index/path) to a cv2 source."""
    raw = _strip_scheme(address, "usb://")
    if raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        return raw  # device path


def _strip_scheme(address: str, scheme: str) -> str:
    return address[len(scheme) :] if address.startswith(scheme) else address


def _sanitize_address(address: str | None) -> str:
    """Strip any ``user:pass@`` credentials from a URL for safe logging.

    ``rtsp://user:pass@host/stream`` → ``rtsp://host/stream``.  Non-URL
    addresses (bare USB indices/paths) pass through unchanged.  Never raises.
    """
    if not address:
        return ""
    try:
        import re

        return re.sub(r"(\w+://)[^@/]*@", r"\1", str(address))
    except Exception:  # noqa: BLE001
        return str(address)
