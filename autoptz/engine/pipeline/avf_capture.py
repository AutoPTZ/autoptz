"""Native AVFoundation video capture, keyed by the stable device ``uniqueID``.

Why this exists
---------------
On macOS the device the user *picks* comes from the **modern**
``AVCaptureDeviceDiscoverySession`` (see ``engine/discovery/usb.py``), but
OpenCV's ``cv2.VideoCapture(index, CAP_AVFOUNDATION)`` opens devices in a
*different* (legacy ``devicesWithMediaType:``) enumeration order.  Picking the
iPhone Continuity Camera could therefore open the built-in webcam, and names
would mismatch.

This module bypasses OpenCV on macOS entirely: given a device's AVFoundation
``uniqueID`` it opens *that exact device* via a native PyObjC
``AVCaptureSession`` + ``AVCaptureVideoDataOutput`` and delivers BGR frames
(numpy ``ndarray``).  Binding by ``uniqueID`` (not index) means reconnects after
sleep/lock re-find the same physical camera even if the enumeration order
shifted while the Continuity Camera came and went.

Frame contract (matches ``USBAdapter`` in ``ingest.py``)
--------------------------------------------------------
- ``open() -> bool``                — start the session; True on success.
- ``read() -> (ok, frame)``         — newest BGR ``HxWx3 uint8`` frame, or
                                      ``(False, None)`` when none is available
                                      yet / the device stalled.  Non-blocking-ish:
                                      waits up to a short grace period for the
                                      first/next frame so the adapter loop pacing
                                      stays in control.
- ``release() -> None``             — tear the session down.
- ``fps`` (property, ``float|None``)— reported hardware fps if known.

Graceful degradation
--------------------
``is_available()`` returns False when PyObjC / AVFoundation is missing so the
caller can fall back to the OpenCV path.  ``open()`` returns False (never raises)
for a missing device or any AVFoundation runtime error.  Continuity Camera
disappearance (sleep/lock) shows up as ``read()`` returning ``(False, None)``
indefinitely → the adapter's stall timer fires and triggers a reconnect, at
which point we rebind by ``uniqueID``.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# ── Optional-dependency probe (lazy, cached) ────────────────────────────────────

_AVF_AVAILABLE: bool | None = None

# Pixel format we request from AVFoundation.  32BGRA gives us exactly the BGR
# channel order OpenCV/the pipeline expects (after dropping the alpha plane),
# avoiding a YUV→BGR conversion on the hot path.
_PIXEL_FORMAT_32BGRA = 0x42475241  # 'BGRA' FourCC == kCVPixelFormatType_32BGRA

# How long ``read()`` waits for a fresh frame before returning ``(False, None)``.
# Short enough that the adapter loop keeps control of pacing/stall detection,
# long enough to ride out normal inter-frame gaps at low fps.
_READ_WAIT_TIMEOUT = 0.5  # seconds

# How long ``open()`` waits for the FIRST decoded frame before giving up so the
# caller can fall back to OpenCV.  Kept short: on builds where the native
# sample-buffer delegate never fires, we want to detect that and switch to the
# working OpenCV path quickly rather than stalling each camera for seconds.
_FIRST_FRAME_TIMEOUT = 1.5  # seconds

# One-time diagnostics so a silent pixel-extraction failure is visible in the log
# (and we can see which extraction strategy worked) without spamming per frame.
_DECODE_LOGGED = False

_FPS_CAP_MIN = 1.0
_FPS_CAP_MAX = 240.0


def is_available() -> bool:
    """Return True iff native AVFoundation capture can actually run here.

    Cached after the first probe.  Never raises — a missing framework simply
    yields False so the caller can fall back to OpenCV.

    Requires only the **core** PyObjC frameworks (AVFoundation + objc + Quartz +
    Foundation); the GCD dispatch queue the sample-buffer output needs is obtained
    via :func:`_make_dispatch_queue`, which falls back to ``libSystem`` through
    ctypes when the ``pyobjc-framework-libdispatch`` module is absent.  That last
    package is the one most often missing from an environment — making it optional
    is what lets native (correct, uniqueID-based) selection work everywhere instead
    of silently degrading to OpenCV's unreliable device ordering.
    """
    global _AVF_AVAILABLE
    if _AVF_AVAILABLE is None:
        try:
            # Probe every framework ``_open_impl`` / the sample-buffer path needs,
            # so the adapter's native-vs-OpenCV decision is correct up front and
            # we don't pick the native path only to fail at open().
            import AVFoundation  # type: ignore  # noqa: F401, PLC0415
            import objc  # type: ignore  # noqa: F401, PLC0415
            import Quartz  # type: ignore  # noqa: F401, PLC0415 — CoreMedia/CoreVideo
            from Foundation import NSObject  # type: ignore  # noqa: F401, PLC0415

            # A usable GCD queue is mandatory for the sample-buffer delegate; if we
            # can't make one (neither pyobjc nor ctypes), native capture can't run.
            _AVF_AVAILABLE = _make_dispatch_queue(b"autoptz.avf.probe") is not None
        except Exception:  # noqa: BLE001 — framework absent / load error
            _AVF_AVAILABLE = False
    return bool(_AVF_AVAILABLE)


_ACCESS_LOCK = threading.Lock()
_ACCESS_RESULT: bool | None = None
_ACCESS_EVENT = threading.Event()
_ACCESS_PENDING = False


def ensure_camera_access(timeout: float = 30.0) -> bool:
    """Ensure macOS camera access is granted, prompting once if undetermined.

    THIS is the difference between "session starts but no frames ever arrive" and
    working video: an ``AVCaptureSession`` started while authorization is
    ``notDetermined`` silently delivers nothing.  We check the status and, on the
    first call, fire ``requestAccessForMediaType:completionHandler:`` (the system
    prompt); every caller waits on the shared result so all camera workers proceed
    together once the user responds.  Returns True only when access is authorized.

    Never raises.  ``denied``/``restricted`` returns False (the adapter then shows
    a clear no-signal; the user grants access in System Settings).
    """
    global _ACCESS_RESULT, _ACCESS_PENDING
    try:
        import AVFoundation  # type: ignore  # noqa: PLC0415

        media = AVFoundation.AVMediaTypeVideo
        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(media)
        if status == 3:  # AVAuthorizationStatusAuthorized
            return True
        if status in (1, 2):  # restricted / denied
            log.warning(
                "Camera access is %s — grant it in System Settings › Privacy & "
                "Security › Camera, then restart, for live video.",
                "restricted" if status == 1 else "denied",
            )
            return False
        # notDetermined → request once; all callers wait on the shared result.
        with _ACCESS_LOCK:
            if _ACCESS_RESULT is not None:
                return _ACCESS_RESULT
            if not _ACCESS_PENDING:
                _ACCESS_PENDING = True
                log.info("Requesting macOS camera access (one-time prompt)…")

                def _handler(granted: bool) -> None:
                    global _ACCESS_RESULT
                    _ACCESS_RESULT = bool(granted)
                    _ACCESS_EVENT.set()

                AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                    media,
                    _handler,
                )
        _ACCESS_EVENT.wait(timeout)
        return bool(_ACCESS_RESULT)
    except Exception:  # noqa: BLE001 — never break capture on the access probe
        log.debug("camera access check failed", exc_info=True)
        return False


def _make_dispatch_queue(label: bytes):  # noqa: ANN202 — opaque dispatch object
    """Create a serial GCD dispatch queue, returning a PyObjC-usable object.

    Prefers the ``pyobjc-framework-libdispatch`` binding when present.  When it
    is not (a common gap — the other pyobjc frameworks are usually installed but
    this one is missed), falls back to calling ``dispatch_queue_create`` directly
    in ``libSystem`` via ctypes and wrapping the returned pointer with
    ``objc.objc_object`` — which yields a genuine ``OS_dispatch_queue`` that
    ``setSampleBufferDelegate_queue_`` accepts.  Returns ``None`` only if both
    paths fail.  Never raises.
    """
    try:
        from libdispatch import dispatch_queue_create  # type: ignore  # noqa: PLC0415

        return dispatch_queue_create(label, None)
    except Exception:  # noqa: BLE001 — module absent: fall through to ctypes/GCD
        pass
    try:
        import ctypes
        import ctypes.util

        import objc  # type: ignore  # noqa: PLC0415

        lib = ctypes.CDLL(ctypes.util.find_library("System") or "/usr/lib/libSystem.B.dylib")
        lib.dispatch_queue_create.restype = ctypes.c_void_p
        lib.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        raw = lib.dispatch_queue_create(label, None)
        if not raw:
            return None
        return objc.objc_object(c_void_p=raw)
    except Exception:  # noqa: BLE001
        log.debug("dispatch queue creation failed (pyobjc + ctypes)", exc_info=True)
        return None


def _device_for_unique_id(unique_id: str):  # noqa: ANN202 — PyObjC opaque type
    """Return the ``AVCaptureDevice`` whose ``uniqueID`` matches, or ``None``.

    Uses AVFoundation's own lookup (``deviceWithUniqueID:``) so we always bind to
    the exact physical device regardless of any enumeration ordering.  Never
    raises; returns ``None`` when the framework is unavailable or the device is
    no longer present (e.g. a Continuity Camera that went to sleep).
    """
    try:
        import AVFoundation  # type: ignore  # noqa: PLC0415

        dev = AVFoundation.AVCaptureDevice.deviceWithUniqueID_(unique_id)
        return dev  # may be None when absent
    except Exception:  # noqa: BLE001
        log.debug("AVFoundation deviceWithUniqueID_ failed for %s", unique_id, exc_info=True)
        return None


class AVFCapture:
    """Native AVFoundation capture for a single device, bound by ``uniqueID``.

    The capture session pushes sample buffers on an AVFoundation-managed dispatch
    queue; we copy the newest frame's pixels into a numpy array under a lock.
    ``read()`` hands back the most recent frame, dropping older ones — exactly
    the "latest frame wins" semantics OpenCV's ``VideoCapture.read()`` provides
    for live cameras, so the adapter loop behaves identically.

    Args:
        unique_id: AVFoundation ``uniqueID`` of the device to open.
    """

    def __init__(self, unique_id: str) -> None:
        self._unique_id = unique_id

        self._session: object | None = None
        self._delegate: object | None = None
        self._output: object | None = None
        self._input: object | None = None
        self._queue: object | None = None

        self._lock = threading.Lock()
        self._frame: NDArray[np.uint8] | None = None
        self._frame_event = threading.Event()
        self._fps: float | None = None

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float | None:
        """Reported hardware fps ceiling, or ``None`` if unknown/absurd."""
        return self._fps

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Build and start the capture session.  Return True on success.

        Never raises — a missing framework, a device that is gone, or any
        AVFoundation runtime error returns False so the caller can fall back or
        retry via the adapter's reconnect loop.
        """
        if not is_available():
            log.debug("AVFCapture unavailable (PyObjC AVFoundation missing).")
            return False
        try:
            return self._open_impl()
        except Exception as exc:  # noqa: BLE001 — opening must never crash ingest
            log.warning("AVFCapture open failed for uniqueID=%s: %s", self._unique_id, exc)
            self.release()
            return False

    def _open_impl(self) -> bool:
        import AVFoundation  # type: ignore  # noqa: PLC0415
        import objc  # type: ignore  # noqa: PLC0415
        from Foundation import NSObject  # type: ignore  # noqa: PLC0415

        # Without camera authorization the session starts but delivers no frames.
        if not ensure_camera_access():
            log.warning("AVFCapture: no camera access — cannot open %s.", self._unique_id)
            return False

        device = _device_for_unique_id(self._unique_id)
        if device is None:
            log.warning("AVFCapture: device uniqueID=%s not found.", self._unique_id)
            return False

        # Build the capture input from the device.
        error = objc.nil
        dev_input, error = AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(
            device, None
        )
        if dev_input is None:
            log.warning("AVFCapture: cannot create input for %s (%s)", self._unique_id, error)
            return False

        session = AVFoundation.AVCaptureSession.alloc().init()
        if not session.canAddInput_(dev_input):
            log.warning("AVFCapture: session cannot add input for %s", self._unique_id)
            return False
        session.addInput_(dev_input)

        # Video data output delivering 32BGRA sample buffers.
        output = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
        try:
            from Quartz import kCVPixelBufferPixelFormatTypeKey  # type: ignore  # noqa: PLC0415

            pixel_key = kCVPixelBufferPixelFormatTypeKey
        except Exception:  # noqa: BLE001 — fall back to the string key name
            pixel_key = "PixelFormatType"
        output.setVideoSettings_({pixel_key: _PIXEL_FORMAT_32BGRA})
        # Drop stale frames so ``read()`` always sees the newest one (live cam).
        output.setAlwaysDiscardsLateVideoFrames_(True)

        if not session.canAddOutput_(output):
            log.warning("AVFCapture: session cannot add output for %s", self._unique_id)
            return False
        session.addOutput_(output)

        # Sample-buffer delegate: copies the newest frame into a numpy array.
        delegate = _build_delegate(NSObject, self)
        queue = _make_dispatch_queue(b"autoptz.avf.capture")
        if queue is None:
            log.warning("AVFCapture: could not create a dispatch queue for %s", self._unique_id)
            return False
        output.setSampleBufferDelegate_queue_(delegate, queue)

        self._session = session
        self._delegate = delegate
        self._output = output
        self._input = dev_input
        # Keep the queue alive for the session's lifetime (the output retains it,
        # but holding our own reference avoids any wrapper-dealloc surprises).
        self._queue = queue

        self._detect_fps(device)

        session.startRunning()

        # Only declare success if a REAL frame actually decodes within a short
        # grace window.  AVFoundation will happily start a session that never
        # delivers a usable buffer (e.g. pixel-buffer extraction unsupported in
        # this PyObjC build) — without this check the adapter would sit on "No
        # Signal" forever instead of falling back to the OpenCV path.
        if not self._frame_event.wait(timeout=_FIRST_FRAME_TIMEOUT):
            log.warning(
                "AVFCapture: session for uniqueID=%s started but delivered no "
                "frame within %.1fs (camera in use elsewhere, or access not "
                "granted?) — will retry.",
                self._unique_id,
                _FIRST_FRAME_TIMEOUT,
            )
            self.release()
            return False

        log.info(
            "AVFCapture opened uniqueID=%s (%s) — first frame OK",
            self._unique_id,
            _safe_name(device),
        )
        return True

    def _detect_fps(self, device: object) -> None:
        """Read the device's active-format max frame rate into ``self._fps``.

        A 0 / absurd reading leaves ``fps`` as ``None`` (unknown) so the adapter
        keeps its requested target.  Never raises.
        """
        try:
            fmt = device.activeFormat()  # type: ignore[attr-defined]
            ranges = fmt.videoSupportedFrameRateRanges()
            best = 0.0
            for r in ranges:
                best = max(best, float(r.maxFrameRate()))
            if _FPS_CAP_MIN <= best <= _FPS_CAP_MAX:
                self._fps = best
            else:
                self._fps = None
        except Exception:  # noqa: BLE001
            self._fps = None

    def release(self) -> None:
        """Stop the session and drop all references.  Idempotent, never raises."""
        session = self._session
        output = self._output
        try:
            if output is not None:
                output.setSampleBufferDelegate_queue_(None, None)
        except Exception:  # noqa: BLE001
            pass
        try:
            if session is not None and session.isRunning():
                session.stopRunning()
        except Exception:  # noqa: BLE001
            pass
        self._session = None
        self._delegate = None
        self._output = None
        self._input = None
        self._queue = None
        with self._lock:
            self._frame = None
        self._frame_event.clear()

    # ── Frame access ────────────────────────────────────────────────────────────

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        """Return ``(ok, frame)`` for the newest available BGR frame.

        Mirrors ``cv2.VideoCapture.read()``: ``(True, frame)`` with the most
        recent frame, or ``(False, None)`` if no frame arrived within a short
        grace window.  A persistent ``(False, None)`` (device asleep/unplugged)
        is what surfaces a stall to the adapter loop, which then reconnects and
        rebinds by ``uniqueID``.
        """
        # Wait briefly for a frame to land, then take the freshest one.  We clear
        # the event before reading so the next ``read`` waits for a *new* frame
        # rather than re-returning the same one at a tight loop.
        if self._frame_event.wait(timeout=_READ_WAIT_TIMEOUT):
            with self._lock:
                frame = self._frame
                self._frame_event.clear()
            if frame is not None:
                return True, frame
        return False, None

    # ── Delegate callback (called on the AVFoundation dispatch queue) ────────────

    def _on_sample_buffer(self, pixel_bgr: NDArray[np.uint8]) -> None:
        """Store the newest decoded BGR frame (called from the capture queue)."""
        with self._lock:
            self._frame = pixel_bgr
        self._frame_event.set()


# ── Delegate class factory ──────────────────────────────────────────────────────

# Cache the dynamically-created delegate class so we register the Objective-C
# class exactly once per process (re-registering the same name raises with
# "Cannot allocateClassPair for _AVFSampleDelegate").  The lock makes the
# create-once guard thread-safe: with multiple cameras, two worker threads used
# to enter the ``is None`` branch concurrently and the loser crashed trying to
# register an already-registered class — which then dropped that camera to the
# OpenCV path (the root of the wrong-name/address drift).
_DELEGATE_CLASS: object | None = None
_DELEGATE_LOCK = threading.Lock()


def _build_delegate(nsobject_base: object, owner: AVFCapture):  # noqa: ANN202
    """Create (once) and instantiate the sample-buffer delegate for *owner*.

    The delegate implements
    ``captureOutput:didOutputSampleBuffer:fromConnection:`` and forwards each
    frame's pixels (converted to a contiguous BGR ``ndarray``) to
    ``owner._on_sample_buffer``.  All pixel-buffer access is guarded so a malformed
    sample never propagates an exception into the AVFoundation runtime.
    """
    global _DELEGATE_CLASS
    import objc  # type: ignore  # noqa: PLC0415

    with _DELEGATE_LOCK:
        if _DELEGATE_CLASS is None:

            class _AVFSampleDelegate(nsobject_base):  # type: ignore[misc, valid-type]
                def initWithOwner_(self, owner):  # noqa: N802
                    self = objc.super(_AVFSampleDelegate, self).init()
                    if self is None:
                        return None
                    self._owner = owner
                    return self

                def captureOutput_didOutputSampleBuffer_fromConnection_(  # noqa: N802
                    self,
                    _output,
                    sample_buffer,
                    _connection,
                ) -> None:
                    try:
                        frame = _sample_buffer_to_bgr(sample_buffer)
                    except Exception:  # noqa: BLE001 — never raise into AVF runtime
                        log.debug("AVFCapture: sample-buffer decode failed", exc_info=True)
                        return
                    if frame is not None and self._owner is not None:
                        self._owner._on_sample_buffer(frame)

            _DELEGATE_CLASS = _AVFSampleDelegate

    return _DELEGATE_CLASS.alloc().initWithOwner_(owner)  # type: ignore[union-attr]


def _sample_buffer_to_bgr(sample_buffer) -> NDArray[np.uint8] | None:  # noqa: ANN001
    """Convert a ``CMSampleBuffer`` (32BGRA) into a contiguous BGR ndarray.

    Returns ``None`` if the buffer has no image data.  The alpha channel is
    dropped, leaving an ``HxWx3`` uint8 BGR frame.  The returned array owns its
    memory (a copy), so it remains valid after the pixel buffer is unlocked.
    """
    import Quartz  # type: ignore  # noqa: PLC0415

    image_buffer = Quartz.CMSampleBufferGetImageBuffer(sample_buffer)
    if image_buffer is None:
        return None

    Quartz.CVPixelBufferLockBaseAddress(image_buffer, 0)
    try:
        width = int(Quartz.CVPixelBufferGetWidth(image_buffer))
        height = int(Quartz.CVPixelBufferGetHeight(image_buffer))
        bytes_per_row = int(Quartz.CVPixelBufferGetBytesPerRow(image_buffer))
        base = Quartz.CVPixelBufferGetBaseAddress(image_buffer)
        if base is None or width == 0 or height == 0:
            return None

        # ``base`` is a pointer to the locked pixel buffer.  How PyObjC hands it
        # back varies by build (objc.varlist / a buffer object / a raw integer
        # address), so try each strategy.  Honour the (possibly padded) row
        # stride, then drop padding + the alpha plane and COPY into a tight BGR
        # array we own (valid after the buffer is unlocked).
        n = bytes_per_row * height
        flat = _base_to_uint8(base, n)
        if flat is None:
            _log_decode_once(False, f"no buffer strategy worked for base={type(base).__name__!r}")
            return None
        rows = flat.reshape((height, bytes_per_row))
        bgra = rows[:, : width * 4].reshape((height, width, 4))
        bgr: NDArray[np.uint8] = np.ascontiguousarray(bgra[:, :, :3])
        _log_decode_once(True, f"{width}x{height} stride={bytes_per_row}")
        return bgr
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(image_buffer, 0)


def _base_to_uint8(base, n: int) -> NDArray[np.uint8] | None:  # noqa: ANN001
    """Wrap a CVPixelBuffer base address as a length-``n`` uint8 view.

    PyObjC may return the base address as an ``objc.varlist`` (has
    ``as_buffer``), a Python buffer/memoryview, or a raw integer pointer.  Try
    each; return ``None`` if none work.  The caller copies before unlocking.
    """
    # objc.varlist → typed view of known length.
    try:
        as_buffer = getattr(base, "as_buffer", None)
        if callable(as_buffer):
            return np.frombuffer(as_buffer(n), dtype=np.uint8, count=n)
    except Exception:  # noqa: BLE001
        pass
    # Already buffer-protocol compatible (memoryview / bytes-like).
    try:
        return np.frombuffer(base, dtype=np.uint8, count=n)
    except Exception:  # noqa: BLE001
        pass
    # Raw integer pointer → read via ctypes.
    try:
        import ctypes  # noqa: PLC0415

        addr = int(base)
        if addr:
            cbuf = (ctypes.c_uint8 * n).from_address(addr)
            return np.frombuffer(cbuf, dtype=np.uint8, count=n)
    except Exception:  # noqa: BLE001
        pass
    return None


def _log_decode_once(ok: bool, detail: str) -> None:
    """Log the first sample-buffer decode outcome at INFO (once per process)."""
    global _DECODE_LOGGED
    if _DECODE_LOGGED:
        return
    _DECODE_LOGGED = True
    if ok:
        log.info("AVFCapture: first frame decoded (%s)", detail)
    else:
        log.warning("AVFCapture: frame decode FAILED (%s) — will fall back to OpenCV", detail)


def _safe_name(device: object) -> str:
    """Best-effort ``localizedName`` of an AVCaptureDevice for logging."""
    try:
        return str(device.localizedName())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return "<unknown>"
