"""MarkEngineFactory — a fully isolated second engine stack for AutoPTZ Mark.

The Mark demo must show ONLY fake cameras and must never touch the user's real
EngineClient/ConfigStore (sharing them is the bug that made real cameras appear
in the Mark wall and let closing Mark kill the app).  This factory builds a
throwaway ConfigStore on a temp file, its own EngineClient + Supervisor, and
populates only synthetic (or fake-NDI) cameras.  The GUI owns a 33 ms QTimer
that calls :meth:`tick`; on close the GUI stops that timer FIRST, then
:meth:`stop`.
"""

from __future__ import annotations

import logging
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt

from autoptz.benchmark.profiles import get_profile
from autoptz.benchmark.runner import _add_synthetic_camera
from autoptz.config.store import ConfigStore
from autoptz.ui.engine_client import EngineClient
from autoptz.ui.frames import ShmFrameSource
from autoptz.ui.mark_session import MarkSession

log = logging.getLogger(__name__)


def _default_supervisor_factory(client: Any, store: Any) -> Any:
    from autoptz.engine.supervisor import Supervisor

    return Supervisor(client, store=store)


class MarkEngineFactory:
    """Own a throwaway store + client + supervisor populated with fake cameras only."""

    def __init__(
        self,
        session: MarkSession,
        *,
        supervisor_factory: Any | None = None,
    ) -> None:
        self._session = session
        # Throwaway store on a temp FILE (never :memory:, never default_db_path()).
        self._tmpdir = Path(tempfile.mkdtemp(prefix="autoptz-mark-"))
        self._store = ConfigStore(db_path=self._tmpdir / "mark.db", debounce_s=0.0)
        # Prime the chosen detector tier on the ISOLATED store BEFORE the client loads
        # it (EngineClient reads "detector_model_tier" at construction).  The session
        # maps the model choice → tier ("auto"/"nano"/"small" → auto/fast/balanced),
        # so the supervisor picks it up at start without touching the main app's tier.
        self._store.set_setting("detector_model_tier", session.detector_tier())
        self._client = EngineClient(store=self._store)
        factory = supervisor_factory or _default_supervisor_factory
        self._supervisor = factory(self._client, self._store)
        self._supervisor.prime_features(dict(get_profile(session.profile).features))
        self._ndi_fleet: Any | None = None
        self._started = False
        # The Mark wall binds to THIS frame source (NOT the main app's).  Wiring the
        # isolated client's provider attach/detach to it is what makes the synthetic
        # workers' shm actually render — without it every tile stayed blank.  Mirrors
        # autoptz/ui/app.py: providerAttachRequested carries (cid, shm, w, h) while
        # ShmFrameSource.attach takes (cid, shm, height, width), so the lambda swaps.
        self._frame_source = ShmFrameSource()
        self._client.providerAttachRequested.connect(
            lambda cid, shm, w, h: self._frame_source.attach(cid, shm, h, w),
            Qt.ConnectionType.QueuedConnection,
        )
        self._client.providerDetachRequested.connect(
            self._frame_source.detach,
            Qt.ConnectionType.QueuedConnection,
        )
        # The camera ids pre-added to the idle wall — the ramp ADOPTS these (and the
        # supervisor below) so only ONE engine stack ever runs (no doubled tiles).
        self._camera_ids: list[str] = []
        self._setup_fake_cameras()

    @property
    def client(self) -> Any:
        return self._client

    @property
    def frame_source(self) -> ShmFrameSource:
        """The isolated frame source the Mark wall binds to (own shm readers)."""
        return self._frame_source

    @property
    def store(self) -> Any:
        return self._store

    @property
    def supervisor(self) -> Any:
        return self._supervisor

    @property
    def camera_ids(self) -> list[str]:
        """The pre-added camera ids on the idle wall (the ramp adopts these)."""
        return list(self._camera_ids)

    @property
    def ndi_fleet(self) -> Any:
        return self._ndi_fleet

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def max_cameras(self) -> int:
        return max(1, int(self._session.max_cameras))

    def _setup_fake_cameras(self) -> None:
        """Register only the FIRST camera (3DMark-style progressive wall).

        The wall starts at one camera and grows via :meth:`add_next_camera` as the
        ramp advances.  For NDI the full fleet of senders is still built up front
        (senders can't be reconfigured per-step cheaply and all broadcast at once),
        but only the first ``ndi://`` camera is registered on the client; the rest
        are registered one at a time by :meth:`add_next_camera`.
        """
        n = self.max_cameras
        self._ndi_names: list[str] = []
        if self._session.source == "ndi":
            from autoptz.benchmark.ndi_sim import (
                MarkNDIFleet,
                _add_ndi_camera,
                ndi_sim_available,
            )

            if ndi_sim_available():
                self._ndi_fleet = MarkNDIFleet(n)
                self._ndi_names = list(self._ndi_fleet.names())
                if self._ndi_names:
                    self._camera_ids.append(_add_ndi_camera(self._client, self._ndi_names[0], 0))
                return
            log.warning("NDI requested but cyndilib unavailable; using synthetic cameras.")
        w, h = self._session.resolution_size()
        self._camera_ids.append(
            _add_synthetic_camera(
                self._client, 0, width=w, height=h, address=self._synthetic_address()
            )
        )

    def _synthetic_address(self) -> str:
        """The SyntheticAdapter address for this session's source.

        ``clip`` → the bundled real-people clip path (the adapter loops the real
        decode); any other (synthetic) source → ``"anim"`` (drawn synthetic people).

        When ``clip`` is requested but the bundled clip is missing (e.g. a fresh
        clone / CI checkout where the asset isn't present), fall back to the drawn
        scene *with a warning* rather than letting the SyntheticAdapter silently
        degrade — the advertised "real people" demo shouldn't fail quietly.
        """
        if self._session.is_clip():
            if self._session.clip_available():
                return self._session.clip_path()
            log.warning(
                "Mark clip source requested but bundled clip is missing (%s); "
                "falling back to drawn synthetic people.",
                self._session.clip_path(),
            )
        return "anim"

    def add_next_camera(self) -> str | None:
        """Add the next fake camera to the wall (progressive ramp).  None at the cap.

        Registers the next synthetic/NDI camera on the isolated client and, when the
        supervisor is already running, spawns just that one worker (no full restart),
        which emits the provider-attach so the new tile starts rendering.
        """
        index = len(self._camera_ids)
        if index >= self.max_cameras:
            return None
        if self._session.source == "ndi" and self._ndi_fleet is not None:
            if index >= len(self._ndi_names):
                return None
            from autoptz.benchmark.ndi_sim import _add_ndi_camera

            cid = _add_ndi_camera(self._client, self._ndi_names[index], index)
        else:
            w, h = self._session.resolution_size()
            cid = _add_synthetic_camera(
                self._client, index, width=w, height=h, address=self._synthetic_address()
            )
        self._camera_ids.append(cid)
        # Bring up just the new worker if the engine is already running.
        if self._started:
            spawn = getattr(self._supervisor, "_spawn_worker", None)
            if callable(spawn):
                try:
                    spawn(cid)
                except Exception:  # noqa: BLE001
                    log.debug("mark add_next_camera worker spawn failed", exc_info=True)
        # _add_synthetic_camera/_add_ndi_camera mutate cameraModel DIRECTLY (not via
        # client.addCamera), so the client's cameraAdded signal — which the wall
        # listens to, to build the new tile + reflow — is never emitted.  Emit it
        # here (this runs on the GUI thread via MarkWindow._grow_one_slot) so the
        # new camera actually appears as a tile (3DMark-style growing wall).
        try:
            self._client.cameraAdded.emit(cid)
        except Exception:  # noqa: BLE001
            log.debug("mark cameraAdded emit failed", exc_info=True)
        return cid

    def auto_track_targets(self, *, seed: int = 0) -> None:
        """Auto-track a seeded random target per camera so Center Stage engages.

        Only meaningful for the **full** profile (which runs detection + tracking);
        the streams profile has no tracks to follow, so this is a no-op there.  Uses
        the engine's existing target-set path (``client.setTarget``): a small seeded
        track id (1..3) per camera position lands on one of the early synthetic
        tracks so the demo visibly locks on rather than sitting idle.  Deterministic
        for a given seed (keyed by camera position, not the random uuid).
        """
        if not get_profile(self._session.profile).features.get("tracking", False):
            return
        rng = random.Random(seed)
        setter = getattr(self._client, "setTarget", None)
        if not callable(setter):
            return
        for cid in self._camera_ids:
            track_id = rng.randint(1, 3)
            try:
                setter(cid, track_id)
            except Exception:  # noqa: BLE001 — a demo target must never crash the engine
                log.debug("mark auto-track set_target failed for %s", cid, exc_info=True)

    def start(self) -> None:
        # NDI senders must broadcast BEFORE the NDIAdapter polls for sources.
        if self._ndi_fleet is not None:
            self._ndi_fleet.open()
        self._supervisor.start(run_pump=False, staged=True)
        self._started = True
        # The factory starts the supervisor directly (bypassing client.startEngine),
        # so reflect the running state on the isolated client for the status bar.
        self._client._engine_running = True
        try:
            self._client.engineStateChanged.emit()
        except Exception:  # noqa: BLE001
            log.debug("mark engineStateChanged emit failed", exc_info=True)

    def tick(self) -> None:
        # Keep the adopted NDI fleet broadcasting each GUI tick (the ramp adopts
        # this fleet rather than building a second one).
        if self._ndi_fleet is not None:
            try:
                self._ndi_fleet.pump_once()
            except Exception:  # noqa: BLE001
                log.debug("mark NDI fleet pump failed", exc_info=True)
        sup = self._supervisor
        if sup is not None and getattr(sup, "is_running", False):
            sup.tick()

    def stop(self) -> None:
        self._started = False
        try:
            self._client._engine_running = False
            self._client.engineStateChanged.emit()
        except Exception:  # noqa: BLE001
            log.debug("mark engine-state clear failed", exc_info=True)
        try:
            self._supervisor.stop()
        except Exception:  # noqa: BLE001
            log.debug("mark supervisor stop failed", exc_info=True)
        # Release every shm reader/intent so the discarded session leaks nothing.
        try:
            self._frame_source.detach_all()
        except Exception:  # noqa: BLE001
            log.debug("mark frame source detach failed", exc_info=True)
        if self._ndi_fleet is not None:
            try:
                self._ndi_fleet.close()
            except Exception:  # noqa: BLE001
                log.debug("mark NDI fleet close failed", exc_info=True)
            self._ndi_fleet = None
        try:
            self._store.close()
        except Exception:  # noqa: BLE001
            log.debug("mark store close failed", exc_info=True)
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            log.debug("mark tempdir cleanup failed", exc_info=True)
