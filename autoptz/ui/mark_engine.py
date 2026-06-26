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
import shutil
import tempfile
from pathlib import Path
from typing import Any

from autoptz.benchmark.profiles import get_profile
from autoptz.benchmark.runner import _add_synthetic_camera
from autoptz.config.store import ConfigStore
from autoptz.ui.engine_client import EngineClient
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
        self._client = EngineClient(store=self._store)
        factory = supervisor_factory or _default_supervisor_factory
        self._supervisor = factory(self._client, self._store)
        self._supervisor.prime_features(dict(get_profile(session.profile).features))
        self._ndi_fleet: Any | None = None
        self._setup_fake_cameras()

    @property
    def client(self) -> Any:
        return self._client

    @property
    def store(self) -> Any:
        return self._store

    @property
    def supervisor(self) -> Any:
        return self._supervisor

    def _setup_fake_cameras(self) -> None:
        n = max(1, int(self._session.max_cameras))
        if self._session.source == "ndi":
            from autoptz.benchmark.ndi_sim import (
                MarkNDIFleet,
                _add_ndi_camera,
                ndi_sim_available,
            )

            if ndi_sim_available():
                self._ndi_fleet = MarkNDIFleet(n)
                for i, name in enumerate(self._ndi_fleet.names()):
                    _add_ndi_camera(self._client, name, i)
                return
            log.warning("NDI requested but cyndilib unavailable; using synthetic cameras.")
        for i in range(n):
            _add_synthetic_camera(self._client, i)

    def start(self) -> None:
        # NDI senders must broadcast BEFORE the NDIAdapter polls for sources.
        if self._ndi_fleet is not None:
            self._ndi_fleet.open()
        self._supervisor.start(run_pump=False, staged=True)

    def tick(self) -> None:
        sup = self._supervisor
        if sup is not None and getattr(sup, "is_running", False):
            sup.tick()

    def stop(self) -> None:
        try:
            self._supervisor.stop()
        except Exception:  # noqa: BLE001
            log.debug("mark supervisor stop failed", exc_info=True)
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
