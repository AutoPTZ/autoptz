"""AutoPTZ Mark — fake NDI sources at scale (import-guarded; needs cyndilib).

Broadcasts the synthetic person frames as local NDI sources so Mark's NDI
source-mode ingests them through the REAL NDIAdapter path.  cyndilib is absent
in the repo .venv / CI, so this module degrades to "unavailable" there and its
tests skip.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:  # import-guard: cyndilib is conda-only, absent in .venv/CI
    from cyndilib.sender import Sender as _Sender
    from cyndilib.video_frame import VideoSendFrame as _VideoSendFrame
    from cyndilib.wrapper import FourCC as _FourCC

    _CYNDILIB_OK = True
except Exception:  # noqa: BLE001 — any import failure → feature unavailable
    _Sender = None  # type: ignore[assignment,misc]
    _VideoSendFrame = None  # type: ignore[assignment,misc]
    _FourCC = None  # type: ignore[assignment,misc]
    _CYNDILIB_OK = False


def ndi_sim_available() -> bool:
    return _CYNDILIB_OK


class MarkNDISender:
    def __init__(
        self, index: int, *, width: int = 1280, height: int = 720, fps: float = 30.0
    ) -> None:
        if not _CYNDILIB_OK:
            raise RuntimeError("cyndilib not available; NDI sim disabled")
        from autoptz.engine.pipeline.ingest import SyntheticAdapter

        self._index = index
        self._w, self._h = int(width), int(height)
        self._name = f"AutoPTZ Mark Cam {index + 1}"
        # Reuse the real synthetic scene generator (person silhouettes).
        self._adapter = SyntheticAdapter(
            f"ndi-sim-{index}", address="anim", width=width, height=height, target_fps=fps
        )
        self._adapter._open()
        self._sender = _Sender(ndi_name=self._name, clock_video=True)
        vf = _VideoSendFrame()
        vf.set_resolution(self._w, self._h)
        vf.set_fourcc(_FourCC.RGBA)
        vf.set_frame_rate(int(round(fps)))
        self._sender.set_video_frame(vf)
        self._open_flag = False

    @property
    def ndi_name(self) -> str:
        return self._name

    def open(self) -> None:
        if not self._open_flag:
            self._sender.open()
            self._open_flag = True

    def push_next(self) -> None:
        bgr = self._adapter._read_frame()
        if bgr is None:
            return
        # BGR (H,W,3) -> RGBA (H,W,4), flattened 1-D uint8 for write_video.
        rgba = np.empty((self._h, self._w, 4), dtype=np.uint8)
        rgba[..., 0] = bgr[..., 2]
        rgba[..., 1] = bgr[..., 1]
        rgba[..., 2] = bgr[..., 0]
        rgba[..., 3] = 255
        self._sender.write_video(np.ascontiguousarray(rgba).ravel())

    def num_connections(self, timeout_ms: int = 0) -> int:
        try:
            return int(self._sender.get_num_connections(timeout_ms))
        except Exception:  # noqa: BLE001
            return 0

    def close(self) -> None:
        try:
            self._sender.close()
        except Exception:  # noqa: BLE001
            log.debug("ndi sender close failed", exc_info=True)
        try:
            self._adapter._close()
        except Exception:  # noqa: BLE001
            pass


class MarkNDIFleet:
    def __init__(self, n: int, *, width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        if not _CYNDILIB_OK:
            raise RuntimeError("cyndilib not available; NDI sim disabled")
        self._senders = [
            MarkNDISender(i, width=width, height=height, fps=fps) for i in range(max(0, n))
        ]

    def names(self) -> list[str]:
        return [s.ndi_name for s in self._senders]

    def open(self) -> None:
        for s in self._senders:
            s.open()

    def pump_once(self) -> None:
        for s in self._senders:
            s.push_next()

    def close(self) -> None:
        for s in self._senders:
            s.close()
