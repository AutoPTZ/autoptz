"""AutoPTZ Mark — fake NDI sources at scale (import-guarded; needs cyndilib).

Broadcasts the synthetic person frames as local NDI sources so Mark's NDI
source-mode ingests them through the REAL NDIAdapter path.  cyndilib is absent
in the repo .venv / CI, so this module degrades to "unavailable" there and its
tests skip.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

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


def _resolve_full_name(short: str, discovered: list[str]) -> str | None:
    """Match a sender's short NDI name to its full discovered name.

    NDI advertises ``"HOSTNAME (short name)"``, so ``"AutoPTZ Mark Cam 1"`` appears
    as ``"PRINCES-MBP (AutoPTZ Mark Cam 1)"``.  Match exact, suffix ``"(short)"``,
    or ``"(short)"`` contained — returns the full name, or None if not yet seen.
    """
    token = f"({short})"
    for full in discovered:
        if full == short or full.endswith(token) or token in full:
            return full
    return None


class MarkNDIFleet:
    def __init__(self, n: int, *, width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        if not _CYNDILIB_OK:
            raise RuntimeError("cyndilib not available; NDI sim disabled")
        self._senders = [
            MarkNDISender(i, width=width, height=height, fps=fps) for i in range(max(0, n))
        ]

    def names(self) -> list[str]:
        return [s.ndi_name for s in self._senders]

    def full_names(self, *, timeout_s: float = 5.0) -> list[str]:
        """Discover each sender's FULL hostname-prefixed NDI name.

        NDI advertises a source as ``"HOSTNAME (short name)"``, so a sender created
        as ``"AutoPTZ Mark Cam 1"`` appears on the network as e.g.
        ``"PRINCES-MBP (AutoPTZ Mark Cam 1)"``.  The ingest (``NDIAdapter``) matches
        the FULL name exactly, so cameras MUST be registered with these — using the
        short name is why NDI mode produced no streams.  Opens + pumps the senders
        while a Finder resolves them; falls back to the short name on any miss.
        """
        short = self.names()
        if not short:
            return []
        self.open()
        from cyndilib.finder import Finder  # noqa: PLC0415

        finder = Finder()
        finder.open()
        resolved: dict[str, str] = {}
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        try:
            while time.monotonic() < deadline and len(resolved) < len(short):
                self.pump_once()  # keep broadcasting so discovery sees them
                try:
                    finder.wait_for_sources(0.3)
                except Exception:  # noqa: BLE001
                    pass
                discovered = [str(s) for s in finder.iter_sources()]
                for s in short:
                    if s not in resolved:
                        full = _resolve_full_name(s, discovered)
                        if full is not None:
                            resolved[s] = full
                time.sleep(0.05)
        finally:
            try:
                finder.close()
            except Exception:  # noqa: BLE001
                log.debug("ndi finder close failed", exc_info=True)
        out = [resolved.get(s, s) for s in short]
        log.info("MarkNDIFleet resolved full NDI names: %s", out)
        return out

    def open(self) -> None:
        for s in self._senders:
            s.open()

    def pump_once(self) -> None:
        for s in self._senders:
            s.push_next()

    def close(self) -> None:
        for s in self._senders:
            s.close()


def _add_ndi_camera(client: Any, ndi_name: str, index: int) -> str:
    """Register one NDI camera (``ndi://<name>``) on the client's model.

    Built directly via a ``CameraRecord`` with a ``type="ndi"`` source so the real
    ``NDIAdapter`` ingests the matching :class:`MarkNDISender` broadcast.
    """
    from autoptz.config.models import CameraConfig, SourceConfig
    from autoptz.ui.list_models import CameraRecord

    camera_id = str(uuid.uuid4())
    uri = f"ndi://{ndi_name}"
    cfg = CameraConfig(
        id=camera_id,
        name=ndi_name,
        source=SourceConfig(type="ndi", address=uri),
    )
    rec = CameraRecord(
        camera_id=camera_id,
        source_uri=uri,
        display_name=ndi_name,
        camera_config=cfg,
    )
    client.cameraModel.add_camera(rec)
    return camera_id


class MarkNDIFleetSampler:
    """Sample fps over N real NDI sources (a :class:`MarkNDIFleet`) on a client.

    Mirrors ``runner._SupervisorSampler`` but ingests through the REAL NDIAdapter:
    it broadcasts a :class:`MarkNDIFleet` and registers ``ndi://`` cameras on the
    injected ``EngineClient`` (the Mark window's, so its CameraWall renders the
    tiles), then drives a headless Supervisor while pumping the fleet each tick.

    Requires cyndilib; constructing it without it raises (callers gate on
    :func:`ndi_sim_available`).  Validated live by the user — cyndilib is absent in
    CI/.venv, so the unit suite only asserts the import-guard + factory wiring.
    """

    def __init__(
        self,
        profile: Any,
        *,
        client: Any,
        supervisor_factory: Callable[[Any, Any], Any] | None = None,
        supervisor: Any | None = None,
        fleet: MarkNDIFleet | None = None,
        cameras: list[str] | None = None,
        adopted_started: bool = False,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        on_grow: Callable[[], str | None] | None = None,
    ) -> None:
        if not _CYNDILIB_OK:
            raise RuntimeError("cyndilib not available; NDI sim disabled")
        from autoptz.benchmark.runner import _default_supervisor_factory

        self._profile = profile
        self._client = client
        self._w, self._h, self._fps = int(width), int(height), float(fps)
        self._factory = supervisor_factory or _default_supervisor_factory
        # Adopt the Mark window's fleet + supervisor + pre-added cameras so only ONE
        # NDI fleet broadcasts and only ONE supervisor runs (no duplicate sources).
        self._adopted = supervisor is not None
        self._fleet = fleet
        self._sup = supervisor
        self._cameras = list(cameras) if cameras else []
        self._started = bool(adopted_started)
        # Progressive ramp (adopted path): grow the registered NDI cameras one at a
        # time as the ramp steps up.  ``on_grow`` registers the next ndi:// camera on
        # the client + spawns its worker (the Mark factory's add_next_camera).  The
        # full fleet of SENDERS is already broadcasting; only the registration grows.
        self._on_grow = on_grow

    @staticmethod
    def _drain_events() -> None:
        from PySide6.QtCore import QCoreApplication

        app = QCoreApplication.instance()
        if app is not None:
            app.processEvents()

    def _ensure_fleet(self, n: int) -> None:
        if self._fleet is None:
            self._fleet = MarkNDIFleet(n, width=self._w, height=self._h, fps=self._fps)
            self._fleet.open()
            for i, name in enumerate(self._fleet.names()):
                self._cameras.append(_add_ndi_camera(self._client, name, i))

    def sample(
        self,
        n: int,
        *,
        dwell_s: float,
        max_ticks: int,
        tick_sleep_s: float,
        fps_reader: Callable[[Any, str], float] | None = None,
    ) -> list[float]:
        from autoptz.benchmark.runner import _default_fps_reader

        reader = fps_reader or _default_fps_reader
        if self._adopted:
            # 3DMark-style progressive ramp: register the next ndi:// cameras one at a
            # time (the senders already broadcast the full fleet).
            if self._on_grow is not None:
                while len(self._cameras) < n:
                    cid = self._on_grow()
                    if cid is None:
                        break
                    self._cameras.append(cid)
            # The Mark window's GUI pump drives the adopted supervisor AND pumps the
            # adopted fleet — ticking/pumping here would race two threads.  Just wait
            # the dwell, then read the first ``n`` pre-added NDI cameras' fps.
            assert self._fleet is not None
            time.sleep(max(0.0, dwell_s) if dwell_s > 0.0 else 0.01)
            self._drain_events()
            return [reader(self._client, cid) for cid in self._cameras[:n]]
        # NDI senders/receivers can't be reconfigured per-step cheaply, so build
        # the full fleet up to the max count once on the first sample.
        self._ensure_fleet(n)
        assert self._fleet is not None
        if self._sup is None:
            store = getattr(self._client, "_store", None)
            self._sup = self._factory(self._client, store)
            self._sup.prime_features(dict(self._profile.features))
        if not self._started:
            self._sup.start(run_pump=False)
            self._started = True

        deadline = time.monotonic() + max(0.0, dwell_s)
        ticks = 0
        while ticks < max_ticks and (ticks == 0 or time.monotonic() < deadline):
            self._fleet.pump_once()
            self._sup.tick()
            self._drain_events()
            ticks += 1
            if tick_sleep_s > 0.0:
                time.sleep(tick_sleep_s)
        self._drain_events()
        return [reader(self._client, cid) for cid in self._cameras[:n]]

    def close(self) -> None:
        # Never tear down an ADOPTED supervisor/fleet — the Mark window owns them.
        if self._adopted:
            return
        if self._sup is not None:
            try:
                self._sup.stop()
            except Exception:  # noqa: BLE001
                log.debug("ndi sampler supervisor stop failed", exc_info=True)
        if self._fleet is not None:
            try:
                self._fleet.close()
            except Exception:  # noqa: BLE001
                log.debug("ndi sampler fleet close failed", exc_info=True)


def ndi_sample_factory(
    profile: str,
    dwell_s: float,
    *,
    client: Any,
    max_cameras: int,
    supervisor_factory: Callable[[Any, Any], Any] | None = None,
) -> Callable[[int], list[float]]:
    """Build a :class:`MarkNDIFleetSampler` and return its ``sample_fn(n)``.

    Raises if cyndilib is unavailable (callers gate on :func:`ndi_sim_available`).
    """
    from autoptz.benchmark.profiles import get_profile

    sampler = MarkNDIFleetSampler(
        get_profile(profile),
        client=client,
        supervisor_factory=supervisor_factory,
    )

    def sample_fn(n: int) -> list[float]:
        return sampler.sample(n, dwell_s=dwell_s, max_ticks=2000, tick_sleep_s=0.005)

    sample_fn._sampler = sampler  # type: ignore[attr-defined]  # for close()
    return sample_fn
