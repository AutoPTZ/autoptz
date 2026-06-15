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
import platform
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

from autoptz.engine.runtime.shm import ShmWriter

log = logging.getLogger(__name__)

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


# ── Reconnect back-off constants ────────────────────────────────────────────────

_BACKOFF_MIN = 1.0    # seconds
_BACKOFF_MAX = 30.0
_BACKOFF_FACTOR = 2.0
_STALL_TIMEOUT_DEFAULT = 5.0  # seconds without a frame → stalled


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
        frame_interval = 1.0 / self._target_fps

        while not self._stop_event.is_set():
            self._set_state(AdapterState.STARTING)

            success = False
            try:
                success = self._open()
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))

            if not success:
                self._set_state(AdapterState.RECONNECTING)
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

                # FPS pacing: sleep the remaining budget for this frame slot
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

def _cv2_usb_backend() -> int:
    """Pick the best cv2 VideoCapture backend for USB cameras on this platform."""
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        return cv2.CAP_MSMF
    return cv2.CAP_V4L2


class USBAdapter(SourceAdapter):
    """Capture from a USB/built-in camera via OpenCV (AVFoundation/MSMF/V4L2).

    Args:
        source: device index (int) or device path/URI (str).
    """

    def __init__(
        self,
        camera_id: str,
        source: int | str = 0,
        shm_writer: ShmWriter | None = None,
        target_fps: float = 30.0,
        stall_timeout: float = _STALL_TIMEOUT_DEFAULT,
    ) -> None:
        super().__init__(camera_id, shm_writer, target_fps, stall_timeout)
        self._source = source
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> bool:
        backend = _cv2_usb_backend()
        cap = cv2.VideoCapture(self._source, backend)  # type: ignore[call-overload]
        if not cap.isOpened():
            cap.release()
            log.warning(
                "camera_id=%s USBAdapter: cannot open source %r",
                self.camera_id, self._source,
            )
            return False
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        self._cap = cap
        log.info("camera_id=%s USBAdapter opened source %r", self.camera_id, self._source)
        return True

    def _read_frame(self) -> NDArray[np.uint8] | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame  # type: ignore[return-value]

    def _close(self) -> None:
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
            "stimeout": "5000000",   # socket read timeout in µs (5 s)
            "max_delay": "500000",
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
                        self.camera_id, hw_codec,
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
                self.camera_id, self._url,
            )
            return False
        self._cap = cap
        log.info(
            "camera_id=%s RTSPAdapter opened %r via cv2 (PyAV not installed)",
            self.camera_id, self._url,
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
    ) -> None:
        super().__init__(camera_id, shm_writer, target_fps, stall_timeout)
        self._ndi_name = ndi_name
        self._receiver: object | None = None
        self._framesync: object | None = None

    def _open(self) -> bool:
        if not _probe_ndi():
            self._set_error("cyndilib not available — install it and the NDI SDK runtime.")
            return False

        try:
            from cyndilib.finder import Finder  # noqa: PLC0415
            from cyndilib.framesync import FrameSyncReceiver  # noqa: PLC0415
            from cyndilib.receiver import Receiver, RecvBandwidth  # noqa: PLC0415

            # Verify the source is currently visible
            finder = Finder()
            finder.open()
            time.sleep(0.5)
            known_names = [str(src) for src in finder.iter_sources()]
            finder.close()

            if self._ndi_name not in known_names:
                log.warning(
                    "camera_id=%s NDI source %r not found on network (seen: %s)",
                    self.camera_id, self._ndi_name, known_names,
                )
                return False

            receiver = Receiver(bandwidth=RecvBandwidth.highest)  # type: ignore[call-arg]
            framesync = FrameSyncReceiver(receiver)

            # Connect to the named source
            for src in Finder().iter_sources():  # type: ignore[call-arg]
                if str(src) == self._ndi_name:
                    receiver.set_source(src)
                    break

            self._receiver = receiver
            self._framesync = framesync
            log.info("camera_id=%s NDIAdapter connected to %r", self.camera_id, self._ndi_name)
            return True

        except Exception as exc:  # noqa: BLE001
            log.warning("camera_id=%s NDIAdapter open failed: %s", self.camera_id, exc)
            return False

    def _read_frame(self) -> NDArray[np.uint8] | None:
        if self._framesync is None:
            return None
        try:
            from cyndilib.video_frame import VideoFrameSync  # noqa: PLC0415

            vf = VideoFrameSync()
            self._framesync.capture_video(vf)  # type: ignore[union-attr]

            data = vf.get_array()
            if data is None or len(data) == 0:
                return None

            arr: NDArray[np.uint8] = np.asarray(data, dtype=np.uint8)
            # NDI typically delivers UYVY or BGRA; convert to BGR
            if arr.ndim == 1:
                # Packed UYVY (4:2:2): 2 bytes per pixel; reshape then convert
                h = vf.yres  # type: ignore[union-attr]
                w = vf.xres  # type: ignore[union-attr]
                arr = arr.reshape((h, w, 2))
                arr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_UYVY)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

            return arr

        except Exception as exc:  # noqa: BLE001
            log.debug("camera_id=%s NDI read_frame error: %s", self.camera_id, exc)
            return None

    def _close(self) -> None:
        for attr in ("_framesync", "_receiver"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, attr, None)
