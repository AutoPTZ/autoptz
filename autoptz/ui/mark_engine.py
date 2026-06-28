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
import os
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

# The drawn ("anim") synthetic-people scene is NO LONGER a user-selectable source —
# Mark always shows the SELECTED clip (real footage).  The drawn scene survives ONLY
# as the ground-truth scene for the accuracy bench, gated by this env (it pairs with
# the ingest adapter's matching AUTOPTZ_MARK_GT gate that also publishes per-frame
# ground-truth boxes).  Off by default, so cameras play the real clip.
_MARK_GT_ENV = "AUTOPTZ_MARK_GT"


def _mark_gt_enabled() -> bool:
    return os.environ.get(_MARK_GT_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _default_supervisor_factory(client: Any, store: Any) -> Any:
    from autoptz.engine.supervisor import Supervisor

    return Supervisor(client, store=store)


def _make_transcode_cache() -> Any:
    """Build the per-scene transcode cache (lazy import — it pulls in cv2).

    A module-level seam so tests can inject a stub cache before the factory's
    ``_setup_fake_cameras`` resolves the first camera's address.
    """
    from autoptz.engine.pipeline.transcode_cache import TranscodeCache

    return TranscodeCache()


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
        # Per-scene transcode cache: the picked resolution/fps drives a native
        # cached variant of the master clip (built once, reused after).  Resolved
        # variant address is memoised so growing the wall never rebuilds.
        self._cache = _make_transcode_cache()
        self._resolved_clip_address: str | None = None
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
                w, h = self._session.resolution_size()
                # NDI broadcasts the SELECTED clip (real footage at the chosen
                # resolution/fps) — the engine's resolved transcode variant — NOT the
                # drawn scene.  ``frame_source`` is None only when the clip is missing,
                # in which case each sender degrades to the drawn ("anim") scene.
                self._ndi_fleet = MarkNDIFleet(
                    n,
                    width=w,
                    height=h,
                    fps=self._session.target_fps(),
                    frame_source=self._ndi_frame_source(),
                )
                # Register the FULL hostname-prefixed names ("HOST (AutoPTZ Mark Cam
                # N)") the network advertises — the NDIAdapter matches the full name,
                # so the short names left every NDI tile blank.  full_names() opens +
                # discovers the senders (falls back to short names on timeout).
                self._ndi_names = self._ndi_fleet.full_names()
                if self._ndi_names:
                    cid = _add_ndi_camera(self._client, self._ndi_names[0], 0)
                    self._camera_ids.append(cid)
                    self._activate_camera_ai(cid)
                return
            log.warning("NDI requested but cyndilib unavailable; using synthetic cameras.")
        w, h = self._session.resolution_size()
        native_fps = self._synthetic_native_fps()
        if native_fps is not None:
            log.info(
                "Mark clip %r → synthetic source @ %.0f fps",
                self._session.clip_info().id,
                native_fps,
            )
        cid = _add_synthetic_camera(
            self._client,
            0,
            width=w,
            height=h,
            address=self._synthetic_address(),
            native_fps=native_fps,
        )
        self._camera_ids.append(cid)
        # Bring the first tile up tracking + auto-framing (full profile only).
        self._activate_camera_ai(cid)

    def _synthetic_native_fps(self) -> float | None:
        """The selected clip's native fps for the per-camera synthetic source.

        Mark always plays the selected clip (the drawn source is removed), so this
        is the clip's own cadence whenever the asset is present.  Returns ``None``
        (default 30 fps pacing) when the ground-truth drawn scene is in use
        (``AUTOPTZ_MARK_GT``) or the bundled clip is missing — the drawn scene has
        no native cadence of its own.
        """
        if _mark_gt_enabled():
            return None
        if self._session.clip_available():
            return self._session.clip_info().native_fps
        return None

    def _synthetic_address(self) -> str:
        """The SyntheticAdapter address fed to every per-camera (clip + NDI) source.

        Mark shows the SELECTED clip's real footage, so this returns a transcode-
        cached variant of the bundled clip at the picked (resolution, fps), built
        once and reused (the adapter loops the real decode).  The drawn ("anim")
        scene is no longer a user source — it is returned ONLY when the accuracy
        bench is on (``AUTOPTZ_MARK_GT``), as the ground-truth scene.

        When the bundled clip is missing (a fresh clone / CI checkout without the
        asset), fall back to the drawn scene *with a warning* rather than letting the
        SyntheticAdapter silently degrade — the advertised "real people" demo
        shouldn't fail quietly.
        """
        if _mark_gt_enabled():
            return "anim"
        if self._session.clip_available():
            return self._resolve_clip_variant_address()
        log.warning(
            "Mark clip is missing (%s); falling back to the drawn synthetic scene.",
            self._session.clip_path(),
        )
        return "anim"

    def _ndi_frame_source(self) -> str | None:
        """The clip variant path each NDI sender broadcasts (None → drawn scene).

        NDI mode shows the SELECTED clip's real footage at the chosen resolution/fps,
        not the drawn ("anim") scene — so it reuses the same transcode-cached variant
        clip mode resolves (built once, falling back to the raw master on a build
        failure).  Returns None only when the bundled clip is absent, so each sender
        degrades to the drawn scene transparently instead of broadcasting a dead path.
        """
        if self._session.clip_available():
            return self._resolve_clip_variant_address()
        log.warning(
            "Mark NDI requested but bundled clip is missing (%s); broadcasting the "
            "drawn synthetic scene instead.",
            self._session.clip_path(),
        )
        return None

    def _resolve_clip_variant_address(self) -> str:
        """The path to feed the synthetic camera for the selected clip.

        Consults the transcode cache for a variant at the session's
        (resolution, fps); on a miss it builds that variant ONCE (slow — seconds
        — but cached after, and it happens once at Mark setup), then reuses the
        resolved path for every camera on the wall.  A build failure falls back
        to the raw master clip path so the demo never crashes on a transcode
        error (worst case: the master plays at its native res/fps).
        """
        # Memoised: growing the wall must reuse the resolved variant, never rebuild.
        if self._resolved_clip_address is not None:
            return self._resolved_clip_address

        meta = self._session.clip_info()
        target_res = self._session.resolution_size()
        target_fps = self._session.target_fps()
        master_path = self._session.clip_path()

        cached = self._cache.get_cached_variant(meta.id, target_res, target_fps)
        if cached is not None:
            self._resolved_clip_address = str(cached)
            return self._resolved_clip_address

        log.info("preparing scene variant…")
        try:
            built = self._cache.build_cached_variant(
                meta.id,
                master_path=master_path,
                master_res=meta.native_resolution,
                master_fps=meta.native_fps,
                target_res=target_res,
                target_fps=target_fps,
            )
            self._resolved_clip_address = str(built)
        except Exception:  # noqa: BLE001 — a transcode failure must never crash the demo
            log.warning(
                "Mark scene variant build failed for clip %r; falling back to the raw master clip.",
                meta.id,
                exc_info=True,
            )
            self._resolved_clip_address = master_path
        return self._resolved_clip_address

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
                self._client,
                index,
                width=w,
                height=h,
                address=self._synthetic_address(),
                native_fps=self._synthetic_native_fps(),
            )
        self._camera_ids.append(cid)
        # Re-apply tracking + Center Stage so a newly-grown tile comes up following
        # and auto-framing immediately (full profile only; no-op for streams).
        self._activate_camera_ai(cid)
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

    def _full_profile(self) -> bool:
        """True when this session's profile runs detection + tracking (Center Stage)."""
        return bool(get_profile(self._session.profile).features.get("tracking", False))

    def _activate_camera_ai(self, cid: str) -> None:
        """Turn ON tracking AND Center Stage for one camera (full profile only).

        So every tile — including newly-grown ones — visibly tracks and auto-frames.
        Tracking uses the engine's ``enableTracking`` path; "Center Stage" is the
        engine's multi-person group-framing knob (``tracking.group_framing``), enabled
        via ``updateCameraConfigPatch`` so the camera auto-widens to keep people in
        shot.  No-op for the streams profile (tracking stays OFF there) and resilient:
        a demo activation must never crash the engine.
        """
        if not self._full_profile():
            return
        enabler = getattr(self._client, "enableTracking", None)
        if callable(enabler):
            try:
                enabler(cid, True)
            except Exception:  # noqa: BLE001 — a demo activation must never crash
                log.debug("mark enableTracking failed for %s", cid, exc_info=True)
        patcher = getattr(self._client, "updateCameraConfigPatch", None)
        if callable(patcher):
            try:
                patcher(cid, {"tracking": {"group_framing": True}})
            except Exception:  # noqa: BLE001 — a demo activation must never crash
                log.debug("mark Center Stage enable failed for %s", cid, exc_info=True)

    def auto_track_targets(self, *, seed: int = 0) -> None:
        """Auto-track a seeded target per camera + turn ON tracking and Center Stage.

        Only meaningful for the **full** profile (which runs detection + tracking);
        the streams profile has no tracks to follow, so this is a no-op there.  For
        each camera it (1) enables tracking + Center Stage via
        :meth:`_activate_camera_ai` so the tile follows and auto-frames, and (2) sets
        a seeded target via the engine's existing ``client.setTarget`` path: a small
        seeded track id (1..3) per camera position lands on one of the early tracks so
        the demo visibly locks on rather than sitting idle.  Deterministic for a given
        seed (keyed by camera position, not the random uuid).
        """
        if not self._full_profile():
            return
        rng = random.Random(seed)
        setter = getattr(self._client, "setTarget", None)
        for cid in self._camera_ids:
            self._activate_camera_ai(cid)
            track_id = rng.randint(1, 3)
            if not callable(setter):
                continue
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
