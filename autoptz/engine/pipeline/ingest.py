"""Source adapters: USB, RTSP/FFmpeg (HW decode), NDI (cyndilib frame-sync).

All adapters share the ``SourceAdapter`` ABC:

- ``start()`` / ``stop()`` — manage an internal daemon capture thread
- ``status``                — snapshot of current state

The capture thread runs ``_run()`` which:

1. Calls ``_open()``; on failure backs off exponentially (1 s → 2 → 4 … 30 s)
2. Reads frames via ``_read_frame()``; paces to ``target_fps``
3. Pushes BGR frames into the injected ``ShmWriter`` (resizing to fit)
4. Detects stalls (no valid frame for ``> stall_timeout`` s) → reconnect
5. On ``stop_event`` calls ``_close()`` and exits

Subclasses implement only ``_open``, ``_read_frame``, and ``_close``.

Optional dependencies:
- ``av`` (PyAV) for RTSPAdapter HW decode — falls back to cv2 if absent
- ``cyndilib`` for NDIAdapter — raises ImportError in ``_open`` if absent
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from autoptz.engine.runtime.messages import BBox, GroundTruthPerson
from autoptz.engine.runtime.shm import ShmWriter

log = logging.getLogger(__name__)

# AutoPTZ Mark accuracy bench: when set, the drawn synthetic scene also publishes
# per-frame ground-truth person boxes (the painted silhouettes' true positions).
# Off by default so the field stays empty (zero payload/overhead) for normal runs.
_MARK_GT_ENV = "AUTOPTZ_MARK_GT"
_NDI_SDK_TELEMETRY_PROBE_INTERVAL_S = 1.0
_NDI_DROP_EST_WINDOW_S = 5.0
_NDI_DROP_EST_TOLERANCE_FPS = 1.5
_NDI_DROP_EST_TOLERANCE_RATIO = 0.15


def _mark_gt_enabled() -> bool:
    return os.environ.get(_MARK_GT_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _ndi_source_drop_estimate(source_fps: float, delivered_fps: float, dt_s: float) -> int:
    """Estimate source-side NDI drops only for severe sustained shortfall.

    NDI FrameSync is latest-frame based: a receiver that wakes a little late can
    report 29-30 fps with jitter even when neither AutoPTZ nor the SDK dropped a
    frame. Treat small gaps as pacing jitter, and reserve this estimate for the
    collapse class that matters to the 8-stream gate.
    """
    source = max(0.0, float(source_fps))
    delivered = max(0.0, float(delivered_fps))
    dt = max(0.0, float(dt_s))
    if source <= 0.0 or delivered <= 0.0 or dt <= 0.0:
        return 0
    gap_fps = source - delivered
    allowed_gap_fps = max(_NDI_DROP_EST_TOLERANCE_FPS, source * _NDI_DROP_EST_TOLERANCE_RATIO)
    if gap_fps <= allowed_gap_fps:
        return 0
    return max(0, int(round(gap_fps * dt)))


# ── Optional-dependency probes (lazy, cached) ──────────────────────────────────

_AV_AVAILABLE: bool | None = None
_NDI_AVAILABLE: bool | None = None


def _probe_av() -> bool:
    global _AV_AVAILABLE
    if _AV_AVAILABLE is None:
        try:
            import av as _av  # noqa: F401

            _AV_AVAILABLE = True
        except ImportError:
            _AV_AVAILABLE = False
    return bool(_AV_AVAILABLE)


def _probe_ndi() -> bool:
    global _NDI_AVAILABLE
    if _NDI_AVAILABLE is None:
        try:
            import cyndilib as _ndi  # noqa: F401

            _NDI_AVAILABLE = True
        except ImportError:
            _NDI_AVAILABLE = False
    return bool(_NDI_AVAILABLE)


# ── Status ─────────────────────────────────────────────────────────────────────


class AdapterState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AdapterStatus:
    state: AdapterState = AdapterState.STOPPED
    fps: float = 0.0
    frames_total: int = 0
    last_error: str | None = None
    # Trusted source fps ceiling, or ``None`` until known. Low current/default
    # stream-rate readings are deliberately left unknown so the UI does not cap
    # itself to a false "max".
    source_fps_cap: float | None = None


# ── Reconnect back-off constants ────────────────────────────────────────────────

_BACKOFF_MIN = 1.0  # seconds
_BACKOFF_MAX = 30.0
_BACKOFF_FACTOR = 2.0
_STALL_TIMEOUT_DEFAULT = 5.0  # seconds without a frame → stalled

# One-time guard so the "native macOS capture unavailable" warning (which means
# camera selection may be unreliable) is logged once per process, not per open.
_WARNED_NO_NATIVE_AVF = False

# Set once native AVFoundation capture is confirmed to start a session but deliver
# no frames (a fragile PyObjC path on some builds).  Subsequent opens then skip the
# native attempt and use OpenCV directly, so cameras don't each pay the native
# first-frame wait before falling back.
_NATIVE_AVF_BROKEN = False


# ── Abstract base ──────────────────────────────────────────────────────────────


class SourceAdapter(ABC):
    """Base for all ingest adapters. Subclasses implement _open / _read_frame / _close."""

    def __init__(
        self,
        camera_id: str,
        shm_writer: ShmWriter | None = None,
        target_fps: float = 30.0,
        stall_timeout: float = _STALL_TIMEOUT_DEFAULT,
    ) -> None:
        self.camera_id = camera_id
        self._shm = shm_writer
        self._target_fps = max(1.0, target_fps)
        self._stall_timeout = stall_timeout

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = AdapterStatus()
        self._status_lock = threading.Lock()

    # ── live tuning ───────────────────────────────────────────────────────────

    def set_target_fps(self, fps: float) -> None:
        """Change the pacing target fps **live** (thread-safe).

        The capture loop re-reads ``self._target_fps`` every tick, so lowering it
        immediately widens the per-frame sleep budget — actually reducing
        capture/detection work without a reconnect.  Clamped to ``[1, fps_cap]``
        when a hardware cap is known so we never pace faster than the source can
        deliver.  Subclasses that hold an open capture may also nudge the device's
        own ``CAP_PROP_FPS``; the pacing change alone is sufficient to slow work.
        """
        cap = self.status.source_fps_cap
        target = max(1.0, float(fps))
        if cap is not None and cap > 0:
            target = min(target, cap)
        self._target_fps = target

    def _set_source_fps_cap(self, cap: float | None) -> None:
        """Record the detected hardware fps ceiling (None when unknown/absurd)."""
        with self._status_lock:
            self._status.source_fps_cap = cap

    def delivery_metrics(self) -> dict[str, float | int | str]:
        """Per-source frame-delivery telemetry (Phase 0a).

        No-op default — only :class:`NDIAdapter` tracks real values (NDI's
        FrameSync never signals "no new frame", so severe source-vs-delivered
        shortfall is estimated over a smoothed rolling window).  Every other
        adapter has direct frame-miss accounting already, so it returns the
        zero/unknown defaults here.
        """
        return {
            "frames_delivered": 0,
            "frames_dropped_est": 0,
            "delivered_fps": 0.0,
            "source_fps": 0.0,
            "duplicate_frames": 0,
            "stale_frames": 0,
            "ndi_queue_depth": -1,
            "ndi_queue_audio": -1,
            "ndi_queue_metadata": -1,
            "ndi_total_video_frames": 0,
            "ndi_dropped_video_frames": 0,
            "ndi_total_audio_frames": 0,
            "ndi_dropped_audio_frames": 0,
            "ndi_total_metadata_frames": 0,
            "ndi_dropped_metadata_frames": 0,
            "ndi_connections": -1,
            "ndi_fourcc": "",
            "ndi_buffer_ms": 0.0,
            "ndi_conversion_ms": 0.0,
            "ndi_copy_ms": 0.0,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the internal capture thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"ingest-{self.camera_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and block until it exits."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None
        self._set_state(AdapterState.STOPPED)

    @property
    def status(self) -> AdapterStatus:
        with self._status_lock:
            s = self._status
            return AdapterStatus(
                state=s.state,
                fps=s.fps,
                frames_total=s.frames_total,
                last_error=s.last_error,
                source_fps_cap=s.source_fps_cap,
            )

    # ── Subclass interface ─────────────────────────────────────────────────────

    @abstractmethod
    def _open(self) -> bool:
        """Attempt to open the source. Return True on success."""

    @abstractmethod
    def _read_frame(self) -> NDArray[np.uint8] | None:
        """Read one BGR frame (H×W×3) or None on transient failure."""

    @abstractmethod
    def _close(self) -> None:
        """Tear down the connection/capture device."""

    # ── Capture loop ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        backoff = _BACKOFF_MIN

        while not self._stop_event.is_set():
            self._set_state(AdapterState.STARTING)

            success = False
            try:
                success = self._open()
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))

            if not success:
                self._set_state(AdapterState.RECONNECTING)
                log.info(
                    "camera_id=%s open failed; retrying in %.0fs",
                    self.camera_id,
                    backoff,
                )
                if self._stop_event.wait(timeout=backoff):
                    break
                backoff = min(backoff * _BACKOFF_FACTOR, _BACKOFF_MAX)
                continue

            backoff = _BACKOFF_MIN  # reset on successful open
            self._set_state(AdapterState.RUNNING)

            last_good_ts = time.monotonic()
            fps_window_start = time.monotonic()
            fps_window_frames = 0

            while not self._stop_event.is_set():
                t0 = time.monotonic()
                frame: NDArray[np.uint8] | None = None

                try:
                    frame = self._read_frame()
                except Exception as exc:  # noqa: BLE001
                    self._set_error(str(exc))
                    break  # trigger outer reconnect loop

                now = time.monotonic()

                if frame is not None:
                    last_good_ts = now
                    self._deliver(frame)
                    fps_window_frames += 1

                    elapsed = now - fps_window_start
                    if elapsed >= 1.0:
                        with self._status_lock:
                            self._status.fps = fps_window_frames / elapsed
                        fps_window_start = now
                        fps_window_frames = 0
                else:
                    # Check for stall
                    if now - last_good_ts > self._stall_timeout:
                        log.warning(
                            "camera_id=%s stalled (%.1f s without frame); reconnecting",
                            self.camera_id,
                            now - last_good_ts,
                        )
                        self._set_state(AdapterState.RECONNECTING)
                        break  # trigger outer reconnect loop

                # FPS pacing: sleep the remaining budget for this frame slot.
                # ``_target_fps`` is re-read here every tick so a live
                # ``set_target_fps()`` change takes effect without a reconnect.
                frame_interval = 1.0 / max(1.0, self._target_fps)
                elapsed_read = time.monotonic() - t0
                sleep_for = frame_interval - elapsed_read
                if sleep_for > 1e-3:
                    self._stop_event.wait(timeout=sleep_for)

            try:
                self._close()
            except Exception as exc:  # noqa: BLE001
                log.debug("camera_id=%s _close() raised: %s", self.camera_id, exc)

        self._set_state(AdapterState.STOPPED)

    def _deliver(self, frame: NDArray[np.uint8]) -> None:
        """Push frame to shm (resizing to ShmWriter dims if needed)."""
        with self._status_lock:
            self._status.frames_total += 1

        if self._shm is None:
            return

        h, w = frame.shape[:2]
        if h != self._shm.height or w != self._shm.width:
            frame = cv2.resize(frame, (self._shm.width, self._shm.height))

        self._shm.push(frame)

    def _set_state(self, state: AdapterState) -> None:
        with self._status_lock:
            self._status.state = state

    def _set_error(self, msg: str) -> None:
        log.error("camera_id=%s adapter error: %s", self.camera_id, msg)
        with self._status_lock:
            self._status.state = AdapterState.ERROR
            self._status.last_error = msg


# ── USB Adapter ────────────────────────────────────────────────────────────────

# A trusted source-FPS ceiling must be inside this band. OpenCV's CAP_PROP_FPS
# often reports the current/default stream rate, not a real maximum, so USB/OpenCV
# only publishes a cap when a high-rate probe gets a high-rate response.
_FPS_CAP_MIN = 1.0
_FPS_CAP_MAX = 240.0
_FPS_TRUSTED_PROBE_MIN = 45.0
_FPS_PROBE_CANDIDATES = (240.0, 120.0, 90.0, 60.0, 50.0)


def _read_cv2_fps(cap: cv2.VideoCapture) -> float:
    try:
        reported = float(cap.get(cv2.CAP_PROP_FPS))
    except Exception:  # noqa: BLE001
        return 0.0
    return reported if _FPS_CAP_MIN <= reported <= _FPS_CAP_MAX else 0.0


def _trusted_cv2_fps_reading(cap: cv2.VideoCapture) -> float:
    """Trust only high CAP_PROP_FPS reads as a max; low reads are current-rate hints."""
    reported = _read_cv2_fps(cap)
    return reported if reported >= _FPS_TRUSTED_PROBE_MIN else 0.0


def _cv2_usb_backend() -> int:
    """Pick the best cv2 VideoCapture backend for USB cameras on this platform."""
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        return cv2.CAP_MSMF
    return cv2.CAP_V4L2


def _macos_index_for_unique_id(unique_id: str) -> int | None:
    """Resolve a macOS AVFoundation ``uniqueID`` → the OpenCV capture index.

    The bug this fixes: ``discovery/usb.py`` enumerates with the modern
    ``AVCaptureDeviceDiscoverySession`` (which *includes* Desk View / Continuity
    pseudo-cameras), but OpenCV's ``CAP_AVFOUNDATION`` opens device **N** from the
    *legacy* ``[AVCaptureDevice devicesWithMediaType:]`` list — a different,
    shorter, differently-ordered enumeration.  Trusting the discovery-session
    index against OpenCV therefore opens the wrong physical camera (classic
    symptom: picking the iPhone Continuity Camera opens the built-in webcam).

    We re-resolve here against the **exact list OpenCV 4.x iterates**: video
    devices plus muxed devices, sorted by ``uniqueID``.  The returned index is
    therefore the verified position of the device whose ``uniqueID`` matches.
    Returns ``None`` when PyObjC/AVFoundation is unavailable or the id is no
    longer present.  Never raises.
    """
    try:
        import AVFoundation  # type: ignore  # noqa: PLC0415

        video = list(
            AVFoundation.AVCaptureDevice.devicesWithMediaType_(
                AVFoundation.AVMediaTypeVideo,
            )
        )
        muxed = list(
            AVFoundation.AVCaptureDevice.devicesWithMediaType_(
                AVFoundation.AVMediaTypeMuxed,
            )
        )
        devices = video + muxed
        # OpenCV preserves system ordering by sorting the combined list by
        # uniqueID before indexing into it. Match that exactly.
        devices.sort(key=lambda d: str(d.uniqueID()))
        for idx, dev in enumerate(devices):
            try:
                if str(dev.uniqueID()) == unique_id:
                    return idx
            except Exception:  # noqa: BLE001 — a flaky device must not abort
                continue
    except Exception:  # noqa: BLE001 — PyObjC/framework absent or runtime error
        log.debug(
            "AVFoundation uniqueID→index resolution unavailable for %s", unique_id, exc_info=True
        )
    return None


class USBAdapter(SourceAdapter):
    """Capture from a USB/built-in camera.

    On **macOS**, when a stable AVFoundation ``unique_id`` is available, capture
    goes through the native :class:`~autoptz.engine.pipeline.avf_capture.AVFCapture`
    (an ``AVCaptureSession`` bound directly by ``uniqueID``) — this opens the
    *exact* physical camera the user picked, sidestepping OpenCV's divergent
    AVFoundation index ordering (the classic "pick the iPhone, get the webcam"
    bug).  Everywhere else — non-macOS, PyObjC absent, or no ``unique_id`` — it
    falls back to ``cv2.VideoCapture`` (AVFoundation/MSMF/V4L2), behaving exactly
    as before.

    The native vs. OpenCV choice is made per ``_open()`` (and re-made on every
    reconnect), so a transient PyObjC/device hiccup degrades to OpenCV without a
    restart.  All of ``set_target_fps``/``source_fps_cap``/stall handling/``status``
    are provided by :class:`SourceAdapter` and are identical for both paths.

    Args:
        source:    device index (int) or device path/URI (str) — the fallback used
                   when ``unique_id`` is absent or the native path is unavailable.
        unique_id: stable macOS AVFoundation ``uniqueID`` for the device.  When
                   given (and on macOS with PyObjC), it binds the native capture
                   to the correct physical camera; on the OpenCV fallback it is
                   re-resolved to the verified OpenCV capture index.
    """

    def __init__(
        self,
        camera_id: str,
        source: int | str = 0,
        shm_writer: ShmWriter | None = None,
        target_fps: float = 30.0,
        stall_timeout: float = _STALL_TIMEOUT_DEFAULT,
        unique_id: str | None = None,
    ) -> None:
        super().__init__(camera_id, shm_writer, target_fps, stall_timeout)
        self._source = source
        self._unique_id = unique_id
        self._cap: cv2.VideoCapture | None = None
        # Native AVFoundation capture (macOS, uniqueID, PyObjC present); None when
        # the OpenCV path is in use for this open.
        self._avf: object | None = None

    def _use_native_avf(self) -> bool:
        """Decide whether to open via native AVFoundation rather than OpenCV.

        True only on macOS, with a ``unique_id`` present, and when the PyObjC
        AVFoundation capture path imports cleanly.  Any of those missing → use
        OpenCV.  Never raises (a probe failure means "no native path").
        """
        global _WARNED_NO_NATIVE_AVF
        if platform.system() != "Darwin" or not self._unique_id:
            return False
        try:
            from autoptz.engine.pipeline import avf_capture  # noqa: PLC0415

            if avf_capture.is_available():
                return True
        except Exception:  # noqa: BLE001 — import/probe failure → OpenCV fallback
            log.debug(
                "camera_id=%s native AVF probe failed; using OpenCV", self.camera_id, exc_info=True
            )
        # macOS + a uniqueID but no native path: capture must fall back to OpenCV,
        # whose AVFoundation device ordering diverges from the picker — selection
        # can be unreliable.  Warn once, loudly, so it shows in the Logs panel.
        if not _WARNED_NO_NATIVE_AVF:
            _WARNED_NO_NATIVE_AVF = True
            log.warning(
                "Native macOS camera capture is unavailable (PyObjC AVFoundation "
                "missing) — camera selection may be unreliable. Reinstall with "
                "`python tools/install.py --editable`.",
            )
        return False

    def _resolve_source(self) -> int | str | None:
        """Pick the cv2 source to open, preferring a verified uniqueID index.

        On macOS, a stored ``unique_id`` is re-resolved to the OpenCV capture
        index every open (so it tracks index reshuffles across reconnects /
        Continuity Camera coming and going).  Off macOS — or when no
        ``unique_id`` is stored — we use the integer index / path the adapter was
        built with, exactly as before.

        **macOS, uniqueID set but unresolvable → return ``None`` (refuse).**  The
        stored ``usb://N`` address is an index from the *modern*
        ``AVCaptureDeviceDiscoverySession`` enumeration, which does NOT line up
        with OpenCV's *legacy* ``devicesWithMediaType:`` ordering.  Opening that
        index would confidently start a **different physical camera** (the classic
        "pick the built-in webcam, get the Continuity Camera").  Rather than guess
        wrong, we refuse: the caller fails the open so the worker shows no-signal
        and retries — the correct camera comes back once it can be verified by
        ``uniqueID`` (native path) or matched in the legacy list.
        """
        if self._unique_id and platform.system() == "Darwin":
            idx = _macos_index_for_unique_id(self._unique_id)
            if idx is not None:
                if idx != self._source:
                    log.info(
                        "camera_id=%s resolved uniqueID=%s → capture index %d",
                        self.camera_id,
                        self._unique_id,
                        idx,
                    )
                return idx
            log.warning(
                "camera_id=%s uniqueID=%s could not be matched to an OpenCV "
                "capture index (PyObjC/AVFoundation unavailable?); refusing to "
                "open a possibly-wrong device. Run `python tools/install.py "
                "--editable` for reliable macOS camera selection.",
                self.camera_id,
                self._unique_id,
            )
            return None
        return self._source

    def _open(self) -> bool:
        # Native AVFoundation binds the EXACT device by uniqueID (the correct
        # selection), so we PREFER it.  But on some PyObjC builds the sample-buffer
        # delegate never fires and the session delivers no frames — so if native
        # fails to deliver, we fall back to OpenCV, which DOES deliver frames.  The
        # fallback is now SAFE: ``_resolve_source`` maps the uniqueID to the
        # verified OpenCV capture index (and refuses rather than open the wrong
        # device when it can't), so OpenCV opens the right camera too — it just
        # can't reach devices missing from its shorter legacy list (e.g. Desk View).
        #
        # ``_NATIVE_AVF_BROKEN`` caches a confirmed no-frame native result so the
        # other cameras (and reconnects) skip the native wait and go straight to
        # the working OpenCV path.
        global _NATIVE_AVF_BROKEN
        if self._use_native_avf() and not _NATIVE_AVF_BROKEN:
            if self._open_avf():
                return True
            _NATIVE_AVF_BROKEN = True
            log.warning(
                "camera_id=%s native AVFoundation delivered no frames — using "
                "OpenCV for this and subsequent cameras.",
                self.camera_id,
            )
        return self._open_cv()

    def _open_avf(self) -> bool:
        """Open the device natively via AVFoundation, bound by ``uniqueID``.

        Returns False (never raises) when the device is gone or AVFoundation
        errors, so the caller can drop to the OpenCV path / reconnect loop.
        Publishes the device's reported hardware fps as the ``source_fps_cap``.
        """
        from autoptz.engine.pipeline.avf_capture import AVFCapture  # noqa: PLC0415

        cap = AVFCapture(str(self._unique_id))
        if not cap.open():
            return False
        # Surface the hardware fps ceiling (UI slider cap) and clamp our pacing
        # to it, mirroring the OpenCV ``_detect_fps_cap`` behaviour.
        reported = cap.fps
        if reported is not None and _FPS_CAP_MIN <= reported <= _FPS_CAP_MAX:
            self._set_source_fps_cap(reported)
            if self._target_fps > reported:
                log.info(
                    "camera_id=%s clamping target_fps %.0f → source cap %.0f",
                    self.camera_id,
                    self._target_fps,
                    reported,
                )
                self._target_fps = reported
        else:
            self._set_source_fps_cap(None)
        self._avf = cap
        log.info(
            "camera_id=%s USBAdapter opened uniqueID=%s via native AVFoundation",
            self.camera_id,
            self._unique_id,
        )
        return True

    def _open_cv(self) -> bool:
        backend = _cv2_usb_backend()
        source = self._resolve_source()
        if source is None:
            # macOS uniqueID could not be verified against the OpenCV capture
            # list — opening the stored discovery index would start the wrong
            # camera, so we bail with a clear, UI-visible reason (surfaced via
            # health ``last_error``) instead.
            self._set_error(
                "Camera could not be verified; reinstall with python tools/install.py "
                "--editable for reliable macOS selection."
            )
            return False
        cap = cv2.VideoCapture(source, backend)  # type: ignore[call-overload]
        if not cap.isOpened():
            cap.release()
            log.warning(
                "camera_id=%s USBAdapter: cannot open source %r",
                self.camera_id,
                source,
            )
            return False
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s USB cv2 buffer-size set failed", self.camera_id, exc_info=True)
        # Probe for a trusted source fps ceiling. A plain CAP_PROP_FPS read is
        # usually just the current/default rate, so low readings stay "unknown".
        self._detect_fps_cap(cap)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        self._cap = cap
        log.info("camera_id=%s USBAdapter opened source %r", self.camera_id, source)
        return True

    def _detect_fps_cap(self, cap: cv2.VideoCapture) -> None:
        """Probe for a credible USB/OpenCV source FPS ceiling.

        Many drivers report the active/default FPS (often 15 or 30) when asked
        for ``CAP_PROP_FPS``. Treating that as "max" makes the UI lie. Instead we
        ask for a few high rates and trust only high responses; otherwise the cap
        remains unknown and the UI uses its generous fallback.
        """
        best = _trusted_cv2_fps_reading(cap)
        for requested in _FPS_PROBE_CANDIDATES:
            try:
                cap.set(cv2.CAP_PROP_FPS, requested)
            except Exception:  # noqa: BLE001
                continue
            reported = _read_cv2_fps(cap)
            if reported >= max(_FPS_TRUSTED_PROBE_MIN, requested * 0.75):
                best = max(best, reported)
        self._apply_trusted_fps_cap(best)

    def _apply_trusted_fps_cap(self, cap: float) -> None:
        if _FPS_CAP_MIN <= cap <= _FPS_CAP_MAX:
            self._set_source_fps_cap(cap)
            if self._target_fps > cap:
                log.info(
                    "camera_id=%s clamping target_fps %.0f → source cap %.0f",
                    self.camera_id,
                    self._target_fps,
                    cap,
                )
                self._target_fps = cap
        else:
            self._set_source_fps_cap(None)

    def _read_frame(self) -> NDArray[np.uint8] | None:
        # Native AVFoundation path: ``read()`` returns the newest BGR frame, or
        # ``(False, None)`` on a transient gap.  A persistent None (device asleep
        # / unplugged) lets the base loop's stall timer fire → reconnect, which
        # rebinds by uniqueID.
        if self._avf is not None:
            ok, frame = self._avf.read()  # type: ignore[attr-defined]
            if not ok or frame is None:
                return None
            return frame
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame  # type: ignore[return-value]

    def _close(self) -> None:
        if self._avf is not None:
            try:
                self._avf.release()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — teardown must not raise
                log.debug("camera_id=%s AVF release raised", self.camera_id, exc_info=True)
            self._avf = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ── RTSP Adapter ───────────────────────────────────────────────────────────────


def _hw_decode_codec() -> tuple[str, dict[str, str]]:
    """Return (codec_name, extra_options) for PyAV hardware-accelerated decode."""
    system = platform.system()
    if system == "Darwin":
        # VideoToolbox — pass an empty codec name; av picks h264 by default, VT kicks in via hwaccel
        return "", {"vt_require_sw": "false"}
    if system == "Windows":
        return "h264_d3d11va", {}
    # Linux: prefer NVDEC if available
    return "h264_cuvid", {}


class RTSPAdapter(SourceAdapter):
    """Capture from an RTSP/IP camera via PyAV (FFmpeg) with HW decode.

    Falls back to ``cv2.VideoCapture`` if PyAV is not installed.

    Args:
        url:       RTSP/RTMP/HTTP stream URL.
        transport: RTSP transport protocol (``"tcp"`` or ``"udp"``).
        hw_decode: Request hardware-accelerated decode (VideoToolbox/NVDEC/D3D11VA).
    """

    def __init__(
        self,
        camera_id: str,
        url: str,
        shm_writer: ShmWriter | None = None,
        target_fps: float = 30.0,
        stall_timeout: float = _STALL_TIMEOUT_DEFAULT,
        transport: str = "tcp",
        hw_decode: bool = True,
    ) -> None:
        super().__init__(camera_id, shm_writer, target_fps, stall_timeout)
        self._url = url
        self._transport = transport
        self._hw_decode = hw_decode

        self._container: object | None = None
        self._frame_iter: Iterator[NDArray[np.uint8]] | None = None
        self._cap: cv2.VideoCapture | None = None  # cv2 fallback

    def _open(self) -> bool:
        if _probe_av():
            return self._open_av()
        return self._open_cv()

    def _open_av(self) -> bool:
        import av  # noqa: PLC0415

        options: dict[str, str] = {
            "rtsp_transport": self._transport,
            "fflags": "nobuffer",
            "flags": "low_delay",
            "stimeout": "5000000",  # socket read timeout in µs (5 s)
            "max_delay": "0",
            "reorder_queue_size": "0",
            "probesize": "32768",
            "analyzeduration": "0",
        }

        try:
            container = av.open(self._url, options=options, timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("camera_id=%s RTSP av.open failed: %s", self.camera_id, exc)
            return False

        streams = container.streams.video
        if not streams:
            container.close()
            return False

        video_stream = streams[0]
        # RTSP stream metadata reports the current stream cadence, not the
        # camera/source maximum. Do not publish it as a hard cap.
        self._set_source_fps_cap(None)

        if self._hw_decode:
            hw_codec, hw_opts = _hw_decode_codec()
            if hw_codec:
                try:
                    video_stream.codec_context.codec = av.Codec(hw_codec)
                    for k, v in hw_opts.items():
                        video_stream.codec_context.options[k] = v  # type: ignore[index]
                except Exception:  # noqa: BLE001
                    log.debug(
                        "camera_id=%s HW codec %r unavailable; using software decode",
                        self.camera_id,
                        hw_codec,
                    )

        self._container = container
        self._frame_iter = self._iter_frames_av(container, video_stream)
        log.info("camera_id=%s RTSPAdapter opened %r via PyAV", self.camera_id, self._url)
        return True

    def _open_cv(self) -> bool:
        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            cap.release()
            log.warning(
                "camera_id=%s RTSPAdapter: cv2 fallback could not open %r",
                self.camera_id,
                self._url,
            )
            return False
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            log.debug("camera_id=%s RTSP cv2 buffer-size set failed", self.camera_id, exc_info=True)
        # OpenCV exposes the current RTSP stream rate here, not a source max.
        self._set_source_fps_cap(None)
        self._cap = cap
        log.info(
            "camera_id=%s RTSPAdapter opened %r via cv2 (PyAV not installed)",
            self.camera_id,
            self._url,
        )
        return True

    @staticmethod
    def _iter_frames_av(container: object, video_stream: object) -> Iterator[NDArray[np.uint8]]:
        """Yield decoded BGR frames from a PyAV container."""
        import av  # noqa: PLC0415

        try:
            for packet in container.demux(video_stream):  # type: ignore[union-attr]
                if packet.size == 0:  # EOS sentinel
                    break
                for frame in packet.decode():
                    yield frame.to_ndarray(format="bgr24")
        except av.FFmpegError as exc:
            log.debug("RTSPAdapter PyAV error during demux: %s", exc)

    def _read_frame(self) -> NDArray[np.uint8] | None:
        if self._frame_iter is not None:
            try:
                return next(self._frame_iter)
            except StopIteration:
                return None

        if self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None  # type: ignore[return-value]

        return None

    def _close(self) -> None:
        self._frame_iter = None
        if self._container is not None:
            try:
                self._container.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            self._container = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ── NDI Adapter ────────────────────────────────────────────────────────────────


def _ndi_color_format_pref() -> str:
    """Which NDI receive color format to request (``AUTOPTZ_NDI_COLOR_FORMAT``).

    * ``fastest`` (default) — ``RecvColorFormat.fastest``: the SDK hands back its
      cheapest native format (usually 8-bit UYVY); we do the single YUV→BGR pass
      ourselves, skipping a full-frame color conversion.  This is the lighter path,
      closer to how NDI Monitor takes native YUV to the GPU.
    * ``bgra`` — ``RecvColorFormat.BGRX_BGRA``: the NDI SDK converts the source's
      native YUV to BGRA on the CPU for every frame, then we strip alpha to BGR.
      Universal but pays that extra conversion inside the SDK; kept as an escape
      hatch (set ``AUTOPTZ_NDI_COLOR_FORMAT=bgra``) for any source the native path
      misbehaves on.

    ``_read_frame`` dispatches on the actual FourCC either way, and self-heals to
    ``bgra`` on reconnect if a source delivers a 16-bit format we can't convert
    cheaply — so ``fastest`` is safe as the default.
    """
    import os

    val = os.environ.get("AUTOPTZ_NDI_COLOR_FORMAT", "fastest").strip().lower()
    return "bgra" if val in ("bgra", "bgrx", "bgr") else "fastest"


def _ndi_fourcc_name(vf: object) -> str:
    """Best-effort FourCC name for a captured video frame (e.g. ``"UYVY"``).

    Empty string when cyndilib can't report it (older builds), so the converter
    falls back to its size-based heuristic.
    """
    try:
        fc = vf.get_fourcc()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return ""
    return str(getattr(fc, "name", fc)).upper()


def _ndi_source_stamp(vf: object) -> object | None:
    """Best-effort source-frame identity for duplicate/stale accounting.

    cyndilib/SDK builds expose this under different names (often timecode or a
    timestamp).  Only scalar values are trusted; mocks/proxy objects are ignored so
    telemetry stays conservative when the wrapper does not expose a real stamp.
    """
    for name in (
        "timestamp",
        "timecode",
        "time_code",
        "frame_timestamp",
        "frame_time",
        "source_timestamp",
    ):
        try:
            val = getattr(vf, name)
        except Exception:  # noqa: BLE001
            continue
        try:
            if callable(val):
                val = val()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(val, int | float | str | bytes):
            return val.decode(errors="replace") if isinstance(val, bytes) else val
    for name in ("get_timestamp", "get_timecode", "get_time_code"):
        try:
            fn = getattr(vf, name)
        except Exception:  # noqa: BLE001
            continue
        if not callable(fn):
            continue
        try:
            val = fn()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(val, int | float | str | bytes):
            return val.decode(errors="replace") if isinstance(val, bytes) else val
    return None


def _ndi_frame_to_bgr(
    arr: NDArray[np.uint8], fourcc: str, h: int, w: int
) -> NDArray[np.uint8] | None:
    """Convert a flat NDI frame buffer to a contiguous BGR image.

    Dispatches on the actual FourCC so the ``fastest`` (native-format) receive
    path is correct for every layout the SDK can hand back — not just the BGRA we
    used to assume.  Returns ``None`` for the 16-bit P216/PA16 families (no cheap
    OpenCV path) so the caller can fall back to the universal BGRA receive format.

    Falls back to the original size-based heuristic when the FourCC is unknown, so
    the default BGRA path is byte-for-byte unchanged.
    """
    size = int(arr.size)
    px = h * w
    f = fourcc.upper() if fourcc else ""

    # Packed 32-bit (4 bytes/pixel): BGRA/BGRX vs RGBA/RGBX.
    if f in ("BGRA", "BGRX") and size == px * 4:
        return cv2.cvtColor(arr.reshape((h, w, 4)), cv2.COLOR_BGRA2BGR)
    if f in ("RGBA", "RGBX") and size == px * 4:
        return cv2.cvtColor(arr.reshape((h, w, 4)), cv2.COLOR_RGBA2BGR)
    # Packed 4:2:2, 8-bit (2 bytes/pixel).
    if f == "UYVY" and size == px * 2:
        return cv2.cvtColor(arr.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_UYVY)
    # UYVY + 8-bit alpha plane (3 bytes/pixel): convert the UYVY part, drop alpha.
    if f == "UYVA" and size == px * 3:
        return cv2.cvtColor(arr[: px * 2].reshape((h, w, 2)), cv2.COLOR_YUV2BGR_UYVY)
    # Planar / semi-planar 4:2:0 (1.5 bytes/pixel).
    if f in ("NV12", "I420", "YV12") and size == px * 3 // 2:
        code = {
            "NV12": cv2.COLOR_YUV2BGR_NV12,
            "I420": cv2.COLOR_YUV2BGR_I420,
            "YV12": cv2.COLOR_YUV2BGR_YV12,
        }[f]
        return cv2.cvtColor(arr.reshape((h * 3 // 2, w)), code)
    # 16-bit YUV (P216/PA16): no cheap OpenCV path — signal a BGRA fall back.
    if f in ("P216", "PA16"):
        return None

    # Unknown FourCC (older cyndilib): preserve the original size-based heuristic.
    if size == px * 4:
        return cv2.cvtColor(arr.reshape((h, w, 4)), cv2.COLOR_BGRA2BGR)
    if size == px * 2:
        return cv2.cvtColor(arr.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_UYVY)
    if size == px * 3:
        return arr.reshape((h, w, 3))
    return None


class NDIAdapter(SourceAdapter):
    """Capture from an NDI source using cyndilib FrameSync.

    Args:
        ndi_name: NDI source name as advertised on the network,
                  e.g. ``"LAPTOP (NDI CAMERA)"``; obtained from ``NDIDiscovery``.
    """

    def __init__(
        self,
        camera_id: str,
        ndi_name: str,
        shm_writer: ShmWriter | None = None,
        target_fps: float = 30.0,
        stall_timeout: float = _STALL_TIMEOUT_DEFAULT,
        discover_timeout: float = 6.0,
    ) -> None:
        super().__init__(camera_id, shm_writer, target_fps, stall_timeout)
        self._ndi_name = ndi_name
        self._discover_timeout = float(discover_timeout)
        self._receiver: object | None = None
        self._finder: object | None = None
        self._video_frame: object | None = None
        # "bgra" | "fastest" — see _ndi_color_format_pref. Flipped to "bgra" at
        # runtime if a source delivers a format we can't convert cheaply, so the
        # next reconnect self-heals to the universal SDK-converted path.
        self._color_format_pref = _ndi_color_format_pref()

        # ── Phase 0a: per-source frame-drop + queue telemetry ─────────────────
        # FrameSync ALWAYS hands back the latest frame (never "no new frame"), so
        # true drops can't be read directly.  Estimate only severe sustained
        # shortfall over a smoothed rolling window from ``source_fps``
        # (advertised, or peak-delivered fallback) vs ``delivered_fps``.  Guarded
        # by the inherited ``_status_lock``.
        self._delivered = 0  # frames delivered this rolling window
        self._win_t0 = 0.0  # monotonic start of the current window
        self._win_delivered0 = 0  # cumulative delivered at window start
        self._delivered_fps = 0.0
        self._source_fps = 0.0
        self._frames_dropped_est = 0  # cumulative estimated drops since (re)connect
        self._queue_depth = -1  # SDK queue depth, -1 when the SDK exposes none
        self._queue_audio = -1
        self._queue_metadata = -1
        self._total_video_frames = 0
        self._dropped_video_frames = 0
        self._total_audio_frames = 0
        self._dropped_audio_frames = 0
        self._total_metadata_frames = 0
        self._dropped_metadata_frames = 0
        self._ndi_connections = -1
        self._last_fourcc = ""
        self._buffer_ms = 0.0
        self._conversion_ms = 0.0
        self._copy_ms = 0.0
        self._duplicate_frames = 0
        self._stale_frames = 0
        self._last_source_stamp: object | None = None
        self._last_source_change_t = 0.0
        self._next_sdk_telemetry_probe_t = 0.0

    def _reset_delivery_metrics(self) -> None:
        """Clear the rolling-window drop/queue state (on (re)connect/close)."""
        with self._status_lock:
            self._delivered = 0
            self._win_t0 = 0.0
            self._win_delivered0 = 0
            self._delivered_fps = 0.0
            self._source_fps = 0.0
            self._frames_dropped_est = 0
            self._queue_depth = -1
            self._queue_audio = -1
            self._queue_metadata = -1
            self._total_video_frames = 0
            self._dropped_video_frames = 0
            self._total_audio_frames = 0
            self._dropped_audio_frames = 0
            self._total_metadata_frames = 0
            self._dropped_metadata_frames = 0
            self._ndi_connections = -1
            self._last_fourcc = ""
            self._buffer_ms = 0.0
            self._conversion_ms = 0.0
            self._copy_ms = 0.0
            self._duplicate_frames = 0
            self._stale_frames = 0
            self._last_source_stamp = None
            self._last_source_change_t = 0.0
            self._next_sdk_telemetry_probe_t = 0.0

    def _open(self) -> bool:
        self._reset_delivery_metrics()
        if not _probe_ndi():
            self._set_error("cyndilib not available — install cyndilib.")
            return False

        try:
            # cyndilib ≥0.1: capture is pull-based via ``Receiver.frame_sync`` (a
            # ``FrameSync``); ``BGRX_BGRA`` delivers ready-to-use BGRA frames.
            from cyndilib.finder import Finder  # noqa: PLC0415
            from cyndilib.receiver import Receiver  # noqa: PLC0415
            from cyndilib.video_frame import VideoFrameSync  # noqa: PLC0415
            from cyndilib.wrapper.ndi_recv import (  # noqa: PLC0415
                RecvBandwidth,
                RecvColorFormat,
            )

            finder = Finder()
            finder.open()
            # NDI discovery is eventually-consistent: a single snapshot can miss a
            # source that's actually present, so poll for it for a few seconds
            # before giving up.
            source = self._resolve_source(finder)
            if source is None:
                known = [str(s) for s in finder.iter_sources()]
                finder.close()
                self._set_error(f"NDI source {self._ndi_name!r} not on the network.")
                log.warning(
                    "camera_id=%s NDI source %r not found (seen: %s)",
                    self.camera_id,
                    self._ndi_name,
                    known,
                )
                return False

            if self._color_format_pref == "fastest":
                color_format = RecvColorFormat.fastest
            else:
                color_format = RecvColorFormat.BGRX_BGRA
            receiver = Receiver(
                color_format=color_format,
                bandwidth=RecvBandwidth.highest,
            )
            log.info(
                "camera_id=%s NDIAdapter color_format=%s",
                self.camera_id,
                self._color_format_pref,
            )
            video_frame = VideoFrameSync()
            receiver.frame_sync.set_video_frame(video_frame)
            receiver.set_source(source)

            # Keep the finder open for the receiver's lifetime so the resolved
            # source stays valid.
            self._finder = finder
            self._receiver = receiver
            self._video_frame = video_frame
            log.info("camera_id=%s NDIAdapter connected to %r", self.camera_id, self._ndi_name)
            return True

        except Exception as exc:  # noqa: BLE001
            log.warning("camera_id=%s NDIAdapter open failed: %s", self.camera_id, exc)
            self._set_error(f"NDI open failed: {exc}")
            return False

    def _resolve_source(self, finder: object) -> object | None:
        """Poll *finder* for ``self._ndi_name`` until it appears or the timeout.

        NDI sources blink in and out of any single discovery poll, so we wait
        (refreshing each iteration) rather than trusting one snapshot.
        """
        deadline = time.monotonic() + self._discover_timeout
        while time.monotonic() < deadline:
            try:
                finder.wait_for_sources(1.0)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — older API without wait_for_sources
                time.sleep(0.5)
            try:
                src = finder.get_source(self._ndi_name)  # type: ignore[attr-defined]
                if src is not None:
                    return src
            except Exception:  # noqa: BLE001
                pass
            for src in finder.iter_sources():  # type: ignore[attr-defined]
                if str(src) == self._ndi_name:
                    return src
        return None

    def _read_frame(self) -> NDArray[np.uint8] | None:
        receiver = self._receiver
        vf = self._video_frame
        if receiver is None or vf is None:
            return None
        try:
            # Non-blocking: fills the *registered* video frame (``vf``) with the
            # latest frame.  capture_video() takes a FrameFormat (default
            # progressive), NOT the frame — the frame was set via set_video_frame.
            receiver.frame_sync.capture_video()  # type: ignore[union-attr]
            w = int(getattr(vf, "xres", 0))
            h = int(getattr(vf, "yres", 0))
            if w <= 0 or h <= 0:
                return None

            t_buffer = time.perf_counter()
            data = vf.get_array()  # type: ignore[union-attr]
            if data is None or len(data) == 0:
                return None
            arr: NDArray[np.uint8] = np.asarray(data, dtype=np.uint8).reshape(-1)
            buffer_ms = (time.perf_counter() - t_buffer) * 1000.0

            # Dispatch on the actual FourCC so the native ("fastest") receive path
            # is correct for whatever the SDK hands back, not just BGRA.
            fourcc = _ndi_fourcc_name(vf)
            t_convert = time.perf_counter()
            bgr = _ndi_frame_to_bgr(arr, fourcc, h, w)
            conversion_ms = (time.perf_counter() - t_convert) * 1000.0
            if bgr is not None:
                t_copy = time.perf_counter()
                out = np.ascontiguousarray(bgr)
                copy_ms = (time.perf_counter() - t_copy) * 1000.0
                self._note_delivered(
                    vf,
                    fourcc=fourcc,
                    buffer_ms=buffer_ms,
                    conversion_ms=conversion_ms,
                    copy_ms=copy_ms,
                    source_stamp=_ndi_source_stamp(vf),
                )
                return out

            # Unsupported native format (16-bit P216/PA16) on the fastest path:
            # self-heal by flipping to the universal BGRA format so the next
            # reconnect re-opens with the SDK doing the conversion.  Drop this
            # frame (the stall/reconnect path will re-open shortly).
            if self._color_format_pref != "bgra":
                log.warning(
                    "camera_id=%s NDI native format %r has no cheap BGR path; "
                    "falling back to BGRA on reconnect",
                    self.camera_id,
                    fourcc or f"{arr.size // max(1, h * w)}bpp",
                )
                self._color_format_pref = "bgra"
            return None

        except Exception as exc:  # noqa: BLE001
            log.debug("camera_id=%s NDI read_frame error: %s", self.camera_id, exc)
            return None

    def _note_delivered(
        self,
        vf: object,
        *,
        fourcc: str,
        buffer_ms: float,
        conversion_ms: float,
        copy_ms: float,
        source_stamp: object | None,
    ) -> None:
        """Account one delivered frame and roll the smoothed drop-estimate window.

        ``source_fps`` prefers the source's advertised ``frame_rate_N/frame_rate_D``
        (guarded), else falls back to the *peak* delivered fps observed so far.
        Drops are accrued only when source-vs-delivered shortfall is severe enough
        to represent a sustained receiver/source mismatch rather than normal
        scheduler jitter.  All writes are under the inherited status lock; never
        raises.
        """
        now = time.monotonic()
        with self._status_lock:
            self._delivered += 1
            self._last_fourcc = str(fourcc or "").upper()
            self._buffer_ms = float(buffer_ms)
            self._conversion_ms = float(conversion_ms)
            self._copy_ms = float(copy_ms)
            self._maybe_probe_sdk_telemetry(now)
            # Advertised source rate (guarded) → peak-delivered fallback.
            adv = 0.0
            try:
                rate_n = float(getattr(vf, "frame_rate_N", 0) or 0)
                rate_d = float(getattr(vf, "frame_rate_D", 0) or 0)
                if rate_n > 0 and rate_d > 0:
                    adv = rate_n / rate_d
            except Exception:  # noqa: BLE001 — telemetry must never fail the read
                adv = 0.0
            if adv > 0.0:
                self._source_fps = max(self._source_fps, adv)

            if source_stamp is not None:
                if self._last_source_stamp is None or source_stamp != self._last_source_stamp:
                    self._last_source_stamp = source_stamp
                    self._last_source_change_t = now
                else:
                    self._duplicate_frames += 1
                    source_rate = adv if adv > 0.0 else self._source_fps
                    if source_rate <= 0.0:
                        source_rate = float(self._target_fps)
                    stale_after_s = max(0.1, 2.5 / max(1.0, source_rate))
                    if (
                        self._last_source_change_t > 0.0
                        and now - self._last_source_change_t >= stale_after_s
                    ):
                        self._stale_frames += 1

            if self._win_t0 <= 0.0:
                self._win_t0 = now
                self._win_delivered0 = self._delivered
                return
            dt = now - self._win_t0
            if dt < _NDI_DROP_EST_WINDOW_S:
                return
            delivered = self._delivered - self._win_delivered0
            self._delivered_fps = delivered / dt if dt > 0 else 0.0

            self._source_fps = max(self._source_fps, self._delivered_fps)

            self._frames_dropped_est += _ndi_source_drop_estimate(
                self._source_fps,
                self._delivered_fps,
                dt,
            )

            # Reset the window.
            self._win_t0 = now
            self._win_delivered0 = self._delivered

    def _maybe_probe_sdk_telemetry(self, now: float) -> None:
        """Sample optional SDK counters without putting them on every frame."""
        if now < self._next_sdk_telemetry_probe_t:
            return
        self._next_sdk_telemetry_probe_t = now + _NDI_SDK_TELEMETRY_PROBE_INTERVAL_S
        self._queue_depth, self._queue_audio, self._queue_metadata = self._probe_queue_depths()
        self._ndi_connections = self._probe_no_connections()
        perf = self._probe_performance_counters()
        self._total_video_frames = perf.get("total_video", self._total_video_frames)
        self._dropped_video_frames = perf.get("dropped_video", self._dropped_video_frames)
        self._total_audio_frames = perf.get("total_audio", self._total_audio_frames)
        self._dropped_audio_frames = perf.get("dropped_audio", self._dropped_audio_frames)
        self._total_metadata_frames = perf.get("total_metadata", self._total_metadata_frames)
        self._dropped_metadata_frames = perf.get("dropped_metadata", self._dropped_metadata_frames)

    @staticmethod
    def _as_int(value: object) -> int | None:
        """Convert scalar counter values without letting mocks become integers."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return None
        return None

    @classmethod
    def _field_int(cls, obj: object, *names: str) -> int | None:
        """Read the first scalar field found from a dict/object/callable."""
        for name in names:
            try:
                if isinstance(obj, dict):
                    value = obj.get(name)
                else:
                    value = getattr(obj, name)
            except Exception:  # noqa: BLE001
                continue
            try:
                if callable(value):
                    value = value()
            except Exception:  # noqa: BLE001
                continue
            out = cls._as_int(value)
            if out is not None:
                return out
        return None

    def _probe_queue_depths(self) -> tuple[int, int, int]:
        """Best-effort SDK queue-depth probe; -1 when nothing exposes one.

        cyndilib does not expose a queue/performance API, but newer SDK builds
        or wrappers might.  Try a couple of likely accessors and swallow anything
        that raises — this is pure telemetry and must never break the read.
        """
        receiver = self._receiver
        try:
            fn = getattr(receiver, "get_queue_depth", None)
            if callable(fn):
                val = self._as_int(fn())
                if val is not None:
                    return (val, -1, -1)
        except Exception:  # noqa: BLE001
            pass
        for name in ("get_queue", "get_queue_depths", "recv_get_queue"):
            try:
                fn = getattr(receiver, name, None)
                data = fn() if callable(fn) else None
            except Exception:  # noqa: BLE001
                continue
            video = self._field_int(data, "video_frames", "video", "video_frames_queue")
            audio = self._field_int(data, "audio_frames", "audio", "audio_frames_queue")
            meta = self._field_int(data, "metadata_frames", "metadata", "metadata_frames_queue")
            if video is not None or audio is not None or meta is not None:
                return (
                    video if video is not None else -1,
                    audio if audio is not None else -1,
                    meta if meta is not None else -1,
                )
        try:
            fs = getattr(receiver, "frame_sync", None)
            qd = getattr(fs, "queue_depth", None)
            val = self._as_int(qd)
            if val is not None:
                return (val, -1, -1)
        except Exception:  # noqa: BLE001
            pass
        return (-1, -1, -1)

    def _probe_no_connections(self) -> int:
        """Best-effort NDI receiver connection-count probe; -1 when unknown."""
        receiver = self._receiver
        for name in ("get_no_connections", "no_connections", "num_connections"):
            try:
                value = getattr(receiver, name)
            except Exception:  # noqa: BLE001
                continue
            try:
                if callable(value):
                    value = value()
            except Exception:  # noqa: BLE001
                continue
            out = self._as_int(value)
            if out is not None:
                return out
        return -1

    def _probe_performance_counters(self) -> dict[str, int]:
        """Best-effort NDIlib_recv_get_performance-style counters."""
        receiver = self._receiver
        data: object | None = None
        for name in ("get_performance", "get_recv_performance", "recv_get_performance"):
            try:
                fn = getattr(receiver, name, None)
                if callable(fn):
                    data = fn()
                    break
            except Exception:  # noqa: BLE001
                data = None
        if data is None:
            try:
                data = getattr(receiver, "performance", None)
            except Exception:  # noqa: BLE001
                data = None
        return self._parse_performance_counters(data)

    @classmethod
    def _parse_performance_counters(cls, data: object) -> dict[str, int]:
        """Parse likely cyndilib/native performance-counter shapes."""
        if data is None:
            return {}

        if isinstance(data, tuple | list) and len(data) >= 2:
            total, dropped = data[0], data[1]
            return cls._merge_performance_halves(total=total, dropped=dropped)

        total = None
        dropped = None
        if isinstance(data, dict):
            total = data.get("total") or data.get("total_frames")
            dropped = data.get("dropped") or data.get("dropped_frames")
        else:
            total = getattr(data, "total", None) or getattr(data, "total_frames", None)
            dropped = getattr(data, "dropped", None) or getattr(data, "dropped_frames", None)
        merged = cls._merge_performance_halves(total=total, dropped=dropped)
        if merged:
            return merged

        out: dict[str, int] = {}
        for dest, names in {
            "total_video": ("total_video_frames", "video_frames", "video"),
            "dropped_video": ("dropped_video_frames", "video_frames_dropped", "video_dropped"),
            "total_audio": ("total_audio_frames", "audio_frames", "audio"),
            "dropped_audio": ("dropped_audio_frames", "audio_frames_dropped", "audio_dropped"),
            "total_metadata": ("total_metadata_frames", "metadata_frames", "metadata"),
            "dropped_metadata": (
                "dropped_metadata_frames",
                "metadata_frames_dropped",
                "metadata_dropped",
            ),
        }.items():
            val = cls._field_int(data, *names)
            if val is not None:
                out[dest] = val
        return out

    @classmethod
    def _merge_performance_halves(cls, *, total: object, dropped: object) -> dict[str, int]:
        out: dict[str, int] = {}
        for dest, obj, names in (
            ("total_video", total, ("video_frames", "video", "total_video_frames")),
            ("dropped_video", dropped, ("video_frames", "video", "dropped_video_frames")),
            ("total_audio", total, ("audio_frames", "audio", "total_audio_frames")),
            ("dropped_audio", dropped, ("audio_frames", "audio", "dropped_audio_frames")),
            ("total_metadata", total, ("metadata_frames", "metadata", "total_metadata_frames")),
            (
                "dropped_metadata",
                dropped,
                ("metadata_frames", "metadata", "dropped_metadata_frames"),
            ),
        ):
            val = cls._field_int(obj, *names)
            if val is not None:
                out[dest] = val
        return out

    def delivery_metrics(self) -> dict[str, float | int | str]:
        """Return the tracked rolling-window delivery telemetry (Phase 0a)."""
        with self._status_lock:
            return {
                "frames_delivered": int(self._delivered),
                "frames_dropped_est": int(self._frames_dropped_est),
                "delivered_fps": float(self._delivered_fps),
                "source_fps": float(self._source_fps),
                "duplicate_frames": int(self._duplicate_frames),
                "stale_frames": int(self._stale_frames),
                "ndi_queue_depth": int(self._queue_depth),
                "ndi_queue_audio": int(self._queue_audio),
                "ndi_queue_metadata": int(self._queue_metadata),
                "ndi_total_video_frames": int(self._total_video_frames),
                "ndi_dropped_video_frames": int(self._dropped_video_frames),
                "ndi_total_audio_frames": int(self._total_audio_frames),
                "ndi_dropped_audio_frames": int(self._dropped_audio_frames),
                "ndi_total_metadata_frames": int(self._total_metadata_frames),
                "ndi_dropped_metadata_frames": int(self._dropped_metadata_frames),
                "ndi_connections": int(self._ndi_connections),
                "ndi_fourcc": str(self._last_fourcc),
                "ndi_buffer_ms": float(self._buffer_ms),
                "ndi_conversion_ms": float(self._conversion_ms),
                "ndi_copy_ms": float(self._copy_ms),
            }

    def _close(self) -> None:
        receiver = self._receiver
        if receiver is not None:
            try:
                receiver.disconnect()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
        finder = self._finder
        if finder is not None:
            try:
                finder.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
        self._receiver = None
        self._finder = None
        self._video_frame = None
        self._reset_delivery_metrics()


# ── Synthetic Adapter (test / scaling, no camera or OS permission needed) ────────


def _find_people_sample() -> str | None:
    """Best-effort path to a bundled people image (ultralytics asset), else None.

    Used only when present at runtime; AutoPTZ never redistributes these samples.
    """
    try:
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.find_spec("ultralytics")
        if spec and spec.submodule_search_locations:
            base = Path(next(iter(spec.submodule_search_locations)))
            for rel in ("assets/zidane.jpg", "assets/bus.jpg"):
                cand = base / rel
                if cand.exists():
                    return str(cand)
    except Exception:  # noqa: BLE001
        pass
    return None


class SyntheticAdapter(SourceAdapter):
    """Procedural / file-backed synthetic source — needs no camera or OS permission.

    Built for scaling tests and headless CPU validation: it fabricates a moving
    BGR scene at the pipeline resolution so the *whole* pipeline (capture handoff →
    detection → tracking → pose → face → ego-motion → paint) runs exactly as for a
    real camera, but deterministically and on any machine (no AVFoundation/USB/NDI,
    no camera permission).  ``address`` selects the content:

    * a path to a **video** file  → looped frame-by-frame (real motion + people);
    * a path to an **image** file → panned/zoomed to synthesise motion;
    * ``""`` / ``"anim"``          → a procedurally generated animated scene;
    * ``"people"``                 → a bundled ultralytics people sample if present.

    The scene is panned every frame so motion-dependent stages (tracking, pose,
    ego-motion optical flow) get genuine work rather than a static image.
    """

    def __init__(
        self,
        camera_id: str,
        *,
        address: str = "",
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
        shm_writer: ShmWriter | None = None,
        stall_timeout: float = _STALL_TIMEOUT_DEFAULT,
        people: bool | None = None,
    ) -> None:
        super().__init__(camera_id, shm_writer, target_fps, stall_timeout)
        self._address = (address or "").strip()
        self._w = int(shm_writer.width) if shm_writer is not None else int(width)
        self._h = int(shm_writer.height) if shm_writer is not None else int(height)
        self._base: NDArray[np.uint8] | None = None
        self._video: object | None = None
        self._idx = 0
        # Effective delivered-fps telemetry (logged every few seconds): when the
        # downstream pipeline can't keep up, the worker pulls slower than target
        # and this drops below the configured fps — the frame-drop signal.
        self._fps_t0 = 0.0
        self._fps_n0 = 0
        # Per-camera phase offset so N synthetic cameras don't move in lockstep
        # (keeps detection/tracking/ego work uncorrelated across cameras).
        self._phase = (abs(hash(camera_id)) % 997) / 997.0 * 2.0 * float(np.pi)
        # People silhouettes: on by default for the procedural/anim scene so the
        # detector/tracker/center-stage have moving targets to engage and show.
        if people is None:
            people = self._address in ("", "anim", "synthetic")
        self._people = bool(people)
        # Latest synthetic ground truth (AutoPTZ Mark bench). Populated per-frame
        # only for the drawn scene when ``AUTOPTZ_MARK_GT`` is on; otherwise stays
        # empty so the telemetry field carries no payload.
        self._latest_gt: list[GroundTruthPerson] = []
        # Stable per-camera count (2–4) + per-person phase so silhouettes drift apart.
        self._n_people = 2 + (abs(hash(camera_id)) % 3)  # 2..4
        # Per-person deterministic motion descriptors: varied speed, path shape, and an
        # occasional smooth entry/exit glide.  Seeded by (camera_id, k) so the scene is
        # lively yet frame-for-frame reproducible (tests assert per-camera decorrelation
        # + replay determinism + detector-friendly silhouettes).
        self._people_motion: list[dict[str, float]] = []
        for k in range(self._n_people):
            seed = abs(hash((camera_id, "person", k)))
            self._people_motion.append(
                {
                    "speed": 0.6 + (seed % 100) / 100.0 * 0.9,  # 0.6..1.5
                    "path": float(seed % 3),  # 0 sine, 1 lissajous, 2 drift
                    "amp_x": 0.22 + ((seed >> 3) % 100) / 100.0 * 0.20,  # 0.22..0.42
                    "amp_y": 0.06 + ((seed >> 7) % 100) / 100.0 * 0.12,  # 0.06..0.18
                    "phase": (seed % 997) / 997.0 * 2.0 * float(np.pi),
                    "exit_period": 4.0 + float(seed % 4),  # smooth off/on cycle (s)
                }
            )

    def _resolve_path(self) -> str | None:
        addr = self._address
        if addr in ("", "anim", "synthetic"):
            return None
        if addr == "people":
            return _find_people_sample()
        return addr

    def _open(self) -> bool:
        self._idx = 0
        path = self._resolve_path()
        if path and os.path.exists(path):
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ok, _ = cap.read()
                count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if ok and count and count > 1.5:  # a real (multi-frame) video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._video = cap
                    # Surface the clip's native cadence as the source fps cap (info
                    # only — pacing still uses self._target_fps).  AutoPTZ Mark feeds
                    # a 24/30/60 clip and this makes its true rate observable.
                    src_fps = cap.get(cv2.CAP_PROP_FPS)
                    if src_fps and 0.0 < src_fps < 240.0:
                        self._set_source_fps_cap(float(src_fps))
                        log.info(
                            "camera_id=%s SyntheticAdapter clip native fps=%.1f "
                            "(pacing target=%.0f)",
                            self.camera_id,
                            src_fps,
                            self._target_fps,
                        )
                    log.info("camera_id=%s SyntheticAdapter looping video %s", self.camera_id, path)
                    return True
                cap.release()
            img = cv2.imread(path)
            if img is not None:
                self._base = cv2.resize(img, (self._w, self._h))
                log.info("camera_id=%s SyntheticAdapter animating image %s", self.camera_id, path)
                return True
        # Drawn ("anim") ground-truth scene: paint a plain/neutral procedural
        # background (``_base`` stays None → the soft gradient in ``_compose``) with
        # the moving people silhouettes on top.  We deliberately DO NOT load a bundled
        # sample photo here anymore — that left a stray picture floating behind the
        # drawn people (and AutoPTZ never redistributes those samples).  The GT
        # geometry / silhouettes are unchanged, so the accuracy bench is unaffected.
        log.info(
            "camera_id=%s SyntheticAdapter procedural scene %dx%d@%.0ffps",
            self.camera_id,
            self._w,
            self._h,
            self._target_fps,
        )
        return True

    def _read_frame(self) -> NDArray[np.uint8] | None:
        self._idx += 1
        now = time.monotonic()
        if self._fps_t0 == 0.0:
            self._fps_t0, self._fps_n0 = now, self._idx
        elif now - self._fps_t0 >= 5.0:
            eff = (self._idx - self._fps_n0) / (now - self._fps_t0)
            # INFO normally (shows in the in-app log panel); WARNING under the test
            # debug flag so headless scaling runs capture it on stderr.
            emit = (
                log.warning
                if os.environ.get("AUTOPTZ_SYNTH_DEBUG", "").strip().lower()
                in ("1", "true", "yes", "on")
                else log.info
            )
            emit(
                "camera_id=%s synthetic effective fps=%.1f (target=%.0f)",
                self.camera_id,
                eff,
                self._target_fps,
            )
            self._fps_t0, self._fps_n0 = now, self._idx
        cap = self._video
        if cap is not None:
            # Clip / video source: real (not synthesised) people — no ground truth.
            self._latest_gt = []
            ok, frm = cap.read()  # type: ignore[union-attr]
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # type: ignore[union-attr]
                ok, frm = cap.read()  # type: ignore[union-attr]
            if not ok or frm is None:
                return None
            if frm.shape[1] != self._w or frm.shape[0] != self._h:
                frm = cv2.resize(frm, (self._w, self._h))
            return np.ascontiguousarray(frm)
        # Drawn scene: optionally publish ground truth for the AutoPTZ Mark bench
        # using the SAME frame clock (self._idx) the scene is painted from, so GT
        # aligns to the delivered frame.  Gated by env so the default path is free.
        if self._people and _mark_gt_enabled():
            t = self._idx / max(1.0, self._target_fps)
            self._latest_gt = self._compute_people_ground_truth(t)
        else:
            self._latest_gt = []
        return self._compose()

    def _compose(self) -> NDArray[np.uint8]:
        w, h, i = self._w, self._h, self._idx
        t = i / max(1.0, self._target_fps)
        # Pan (simulated camera/ego motion): slow sinusoid + tiny high-freq jitter.
        dx = int(round(0.04 * w * np.sin(t * 0.8 + self._phase) + 2.0 * np.sin(t * 11.0)))
        dy = int(round(0.03 * h * np.cos(t * 0.6 + self._phase)))
        if self._base is not None:
            frame = np.roll(self._base, (dy, dx), axis=(0, 1))
        else:
            # Plain/neutral studio-grey background — a soft vertical gradient (top
            # lighter, bottom darker) with a faint slowly-drifting horizontal sheen so
            # the optical-flow / ego-motion stages still see texture, but no garish
            # colours and (deliberately) no bundled photo behind the people.
            col = np.linspace(70, 110, h, dtype=np.float32).reshape(h, 1)
            sheen = 12.0 * np.cos((np.arange(w, dtype=np.float32) / max(1.0, w)) * 6.28 + t)
            grey = np.clip(col + sheen.reshape(1, w), 0, 255).astype(np.uint8)
            frame = np.repeat(grey[:, :, np.newaxis], 3, axis=2)
        # A moving foreground: person silhouettes (default) so detection→tracking→
        # center-stage engage, or the legacy plain block when people are disabled.
        if self._people:
            self._draw_people(frame, t)
        else:
            bx = int((0.5 + 0.35 * np.sin(t * 1.3 + self._phase)) * w)
            by = int((0.5 + 0.25 * np.cos(t * 1.1)) * h)
            cv2.rectangle(frame, (bx - 60, by - 120), (bx + 60, by + 120), (30, 30, 220), -1)
        return np.ascontiguousarray(frame)

    def _people_boxes(self, t: float) -> list[tuple[int, int, int, int, int, str]]:
        """Pure silhouette geometry for the drawn scene at scene-time ``t``.

        Returns one ``(person_id, cx, cy, person_h, half_w, path_type)`` row per
        person (including momentarily off-frame ones — the off-frame clip is the
        caller's ``-half_w <= cx <= w + half_w`` test).  Motion is a function of
        ``(camera_id, person_index, frame_index)`` only, so the same camera
        replayed from frame 0 is byte-identical.  This is the single source of
        truth consumed by BOTH :meth:`_draw_people` (the painted pixels) and
        :meth:`_compute_people_ground_truth` (the bench's true boxes), so the
        ground truth always aligns with what the detector actually sees.
        Silhouettes stay ≥ ``0.42*h`` tall so the real detector still fires.
        """
        h, w = self._h, self._w
        person_h = int(0.42 * h)  # detector-friendly height (>> noise floor)
        half_w = max(8, int(person_h * 0.16))
        boxes: list[tuple[int, int, int, int, int, str]] = []
        for k in range(self._n_people):
            m = self._people_motion[k]
            tt = t * m["speed"] + m["phase"]
            if m["path"] == 0.0:  # gentle sine sweep
                fx = 0.5 + m["amp_x"] * np.sin(tt)
                fy = 0.55 + m["amp_y"] * np.cos(tt * 0.8)
            elif m["path"] == 1.0:  # lissajous (figure-eight feel)
                fx = 0.5 + m["amp_x"] * np.sin(tt * 1.3)
                fy = 0.55 + m["amp_y"] * np.sin(tt * 0.9 + 1.1)
            else:  # slow lateral drift + bob
                fx = 0.5 + m["amp_x"] * np.sin(tt * 0.5) + 0.06 * np.sin(tt * 4.0)
                fy = 0.55 + m["amp_y"] * np.cos(tt * 0.6)
            # Occasional smooth entry/exit: ease the person off the right edge and back.
            cycle = (t % m["exit_period"]) / m["exit_period"]
            if cycle > 0.85:  # last 15% of the cycle: glide off-frame
                fx = fx + (cycle - 0.85) / 0.15 * 0.8
            cx = int(min(1.2, max(-0.2, fx)) * w)
            cy = int(min(0.95, max(0.20, fy)) * h)
            boxes.append((k, cx, cy, person_h, half_w, str(int(m["path"]))))
        return boxes

    def _draw_people(self, frame: NDArray[np.uint8], t: float) -> None:
        """Draw N moving silhouettes from the shared :meth:`_people_boxes` geometry.

        Visuals are unchanged from before the refactor — the per-person motion now
        lives in :meth:`_people_boxes` so the ground truth uses the very same boxes.
        """
        w = self._w
        for _pid, cx, cy, person_h, half_w, _path in self._people_boxes(t):
            head_r = max(6, int(person_h * 0.13))
            if cx < -half_w or cx > w + half_w:
                continue  # fully off-frame this instant
            top = cy - person_h // 2
            # legs
            cv2.rectangle(frame, (cx - half_w, cy), (cx - 2, top + person_h), (60, 60, 70), -1)
            cv2.rectangle(frame, (cx + 2, cy), (cx + half_w, top + person_h), (60, 60, 70), -1)
            # torso
            cv2.rectangle(
                frame, (cx - half_w, top + 2 * head_r), (cx + half_w, cy), (140, 90, 60), -1
            )
            # head
            cv2.circle(frame, (cx, top + head_r), head_r, (170, 150, 130), -1)

    def _compute_people_ground_truth(self, t: float) -> list[GroundTruthPerson]:
        """Ground-truth person boxes for the drawn scene at scene-time ``t``.

        Built from the SAME :meth:`_people_boxes` geometry the scene is painted
        from, so each GT box matches a painted silhouette's outer extent.
        ``visible`` mirrors the drawn off-frame clip (False once a silhouette has
        glided off-frame this tick), and ``path_type`` is the person's motion-path
        id.  One entry per person (visible or not).
        """
        w = self._w
        out: list[GroundTruthPerson] = []
        for pid, cx, cy, person_h, half_w, path_type in self._people_boxes(t):
            top = cy - person_h // 2
            bbox = BBox(
                x1=float(cx - half_w),
                y1=float(top),
                x2=float(cx + half_w),
                y2=float(top + person_h),
            )
            visible = -half_w <= cx <= w + half_w
            out.append(
                GroundTruthPerson(person_id=pid, bbox=bbox, visible=visible, path_type=path_type)
            )
        return out

    def latest_ground_truth(self) -> list[GroundTruthPerson]:
        """Most-recent synthetic ground truth for the current frame (bench only).

        Empty unless this is the drawn scene AND ``AUTOPTZ_MARK_GT`` is on — clip /
        video / image sources never publish ground truth.  The camera worker reads
        this when stamping telemetry on the bench.
        """
        return list(self._latest_gt)

    def _close(self) -> None:
        cap = self._video
        if cap is not None:
            try:
                cap.release()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            self._video = None
