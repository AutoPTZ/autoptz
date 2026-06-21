"""NDI PTZ backend via cyndilib (``Receiver.ptz``).

Requires the NDI SDK runtime and the cyndilib package; degrades gracefully with a
clear error if absent.

Two ways to drive PTZ:

* pass a ``receiver`` already connected to the source (shared with the video
  adapter) — ``close()`` then leaves it alone (the caller owns it), or
* pass an ``ndi_name`` and the backend opens its **own** low-bandwidth receiver
  to that source and owns its lifetime (``close()`` disconnects it).

NDI PTZ exposes no absolute pan/tilt position query, so ``get_position()`` is
always ``None``.

Sign convention gotcha: cyndilib's ``ptz.pan_and_tilt(pan_speed, tilt_speed)``
treats **+pan_speed as LEFT** and -pan_speed as right (per the NDI SDK), the
opposite of every other backend and of the controller's "+pan = right".  This
backend therefore **negates pan** so a positive controller command pans right on
NDI too.  Tilt already matches (+ = up).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from autoptz.engine.ptz.base import PTZBackend, PTZCaps, PTZState

log = logging.getLogger(__name__)


def _require_cyndilib() -> Any:
    try:
        import cyndilib

        return cyndilib
    except ImportError as exc:
        raise ImportError(
            "cyndilib is required for NDI PTZ control.  Install it and the NDI SDK "
            "runtime: pip install cyndilib"
        ) from exc


def _connect_ptz_receiver(ndi_name: str, timeout: float) -> tuple[Any, Any]:
    """Open a low-bandwidth receiver connected to *ndi_name*; return (receiver, finder).

    Polls discovery (NDI is eventually-consistent) for up to *timeout* seconds.
    Raises ``RuntimeError`` if the source never appears.  The finder is returned so
    the caller can keep it open for the receiver's lifetime and close it later.
    """
    from cyndilib.finder import Finder
    from cyndilib.receiver import Receiver
    from cyndilib.wrapper.ndi_recv import RecvBandwidth, RecvColorFormat

    finder = Finder()
    finder.open()
    source = None
    deadline = time.monotonic() + timeout
    while source is None and time.monotonic() < deadline:
        try:
            finder.wait_for_sources(1.0)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — older API without wait_for_sources
            time.sleep(0.5)
        try:
            source = finder.get_source(ndi_name)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            source = None
        if source is None:
            for src in finder.iter_sources():  # type: ignore[attr-defined]
                if str(src) == ndi_name:
                    source = src
                    break
    if source is None:
        finder.close()
        raise RuntimeError(f"NDI source {ndi_name!r} not found on the network.")

    # Lowest bandwidth: PTZ control rides the connection; we don't decode video
    # here (the ingest adapter has its own receiver for frames).
    receiver = Receiver(color_format=RecvColorFormat.BGRX_BGRA, bandwidth=RecvBandwidth.lowest)
    receiver.set_source(source)
    return receiver, finder


class NDIPTZBackend(PTZBackend):
    """NDI PTZ backend wrapping cyndilib's ``Receiver.ptz`` API.

    Args:
        ndi_name:         NDI source name to connect a dedicated PTZ receiver to
                          (used when ``receiver`` is not supplied).
        receiver:         An already-connected ``cyndilib.Receiver`` to ride
                          instead of opening our own.  When given, ``close()``
                          leaves it alone (the caller owns it).
        discover_timeout: Seconds to wait for ``ndi_name`` to appear when opening
                          our own receiver.
    """

    def __init__(
        self,
        ndi_name: str = "",
        receiver: Any | None = None,
        *,
        discover_timeout: float = 5.0,
    ) -> None:
        super().__init__()
        _require_cyndilib()
        self._lock = threading.Lock()
        self._owns_receiver = receiver is None
        self._finder: Any | None = None
        if receiver is None:
            if not ndi_name:
                raise ValueError("NDIPTZBackend needs an ndi_name or a receiver.")
            receiver, self._finder = _connect_ptz_receiver(ndi_name, discover_timeout)
        self._recv = receiver
        self._ptz = receiver.ptz  # cyndilib PTZ controller
        # PTZ-capability metadata isn't available the instant we connect; let the
        # connection settle so the first commands land and the check is accurate.
        if not self._wait_ptz_ready(timeout=1.5):
            log.warning(
                "NDIPTZBackend: source %r reports no PTZ support after settle; "
                "commands will still be attempted.",
                ndi_name or receiver,
            )
        self.caps = PTZCaps(
            continuous_pan_tilt=True,
            continuous_zoom=True,
            native_presets=True,
            query_position=False,
            absolute_pan_tilt=True,
            absolute_zoom=True,
        )
        log.info("NDIPTZBackend ready (source=%r, owns_receiver=%s)", ndi_name, self._owns_receiver)

    # ── PTZBackend interface ──────────────────────────────────────────────────

    def _wait_ptz_ready(self, timeout: float) -> bool:
        """Poll ``is_ptz_supported`` until True or *timeout* (connection settle)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if bool(self._recv.is_ptz_supported()):
                    return True
            except Exception:  # noqa: BLE001
                return True  # can't tell → assume yes; calls no-op if not
            time.sleep(0.1)
        return False

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        with self._lock:
            try:
                # NDI: +pan_speed is LEFT, so negate to keep "+pan = right".
                self._ptz.pan_and_tilt(-float(pan), float(tilt))
                self._ptz.zoom(float(zoom))
            except Exception:  # noqa: BLE001 — never let a PTZ hiccup crash the loop
                log.debug("NDI move_velocity failed", exc_info=True)

    def move_absolute(self, pan: float, tilt: float, zoom: float) -> None:
        with self._lock:
            try:
                # Controller pan is "+ = right"; NDI absolute pan is "+ = left".
                self._ptz.set_pan_and_tilt_values(-float(pan), float(tilt))
                self._ptz.set_zoom_level(float(zoom))
            except Exception:  # noqa: BLE001
                log.debug("NDI move_absolute failed", exc_info=True)

    def stop(self) -> None:
        with self._lock:
            try:
                self._ptz.pan_and_tilt(0.0, 0.0)
                self._ptz.zoom(0.0)
            except Exception:  # noqa: BLE001
                pass

    def get_position(self) -> PTZState | None:
        return None  # NDI PTZ has no standard position inquiry

    def goto_preset(self, idx: int) -> None:
        with self._lock:
            try:
                self._ptz.recall_preset(idx, 1.0)
            except Exception:  # noqa: BLE001
                log.debug("NDI goto_preset failed", exc_info=True)

    def save_preset(self, idx: int) -> None:
        with self._lock:
            try:
                self._ptz.store_preset(idx)
            except Exception:  # noqa: BLE001
                log.debug("NDI save_preset failed", exc_info=True)

    def home(self) -> None:
        """Best-effort NDI home: many cameras map preset 0 to home/default."""
        with self._lock:
            try:
                self._ptz.recall_preset(0, 1.0)
            except Exception:  # noqa: BLE001
                log.debug("NDI home (preset 0 recall) unsupported/failed", exc_info=True)

    def osd_menu(self) -> None:
        """NDI PTZ has no on-screen-display menu command — safe no-op."""
        log.debug("NDI backend has no OSD-menu command; ignoring osd_menu()")

    def close(self) -> None:
        try:
            self.stop()
        except Exception:  # noqa: BLE001
            pass
        # Only tear down the connection if we opened it (shared receivers belong
        # to the caller / video adapter).
        if self._owns_receiver:
            for obj in (self._recv, self._finder):
                if obj is None:
                    continue
                try:
                    if obj is self._recv:
                        obj.disconnect()
                    else:
                        obj.close()
                except Exception:  # noqa: BLE001
                    pass
            self._finder = None
        log.info("NDIPTZBackend closed (owns_receiver=%s)", self._owns_receiver)
