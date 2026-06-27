"""MarkEngineFactory: isolated stack, fake-only cameras, clean teardown, main store untouched."""

from __future__ import annotations

from pathlib import Path

from autoptz.config.store import ConfigStore, default_db_path
from autoptz.ui.mark_session import MarkSession


class _FakeSupervisor:
    def __init__(self, client, store):
        self.client = client
        self.store = store
        self.started = False
        self.stopped = False
        self.ticks = 0
        self._features = None

    def prime_features(self, features):  # mirror real API
        self._features = features

    def start(self, *, run_pump=False, staged=False, progress=None):
        self.started = True

    def tick(self):
        self.ticks += 1

    def stop(self):
        self.stopped = True

    @property
    def is_running(self):
        return self.started and not self.stopped


def _factory():
    from autoptz.ui import mark_engine

    made = {}

    def fake_sup(client, store):
        s = _FakeSupervisor(client, store)
        made["sup"] = s
        return s

    eng = mark_engine.MarkEngineFactory(
        MarkSession(source="synthetic", max_cameras=3),
        supervisor_factory=fake_sup,
    )
    return eng, made


def test_store_is_isolated_not_the_default_db():
    eng, _ = _factory()
    try:
        assert isinstance(eng.store, ConfigStore)
        assert Path(eng.store._path).resolve() != default_db_path().resolve()
    finally:
        eng.stop()


def test_starts_with_one_camera_for_progressive_ramp():
    """3DMark-style: the idle wall starts at ONE synthetic camera, not all N.

    Cameras are added one at a time as the ramp advances (via add_next_camera),
    so the wall visibly grows rather than showing N blank tiles up front.
    """
    eng, _ = _factory()  # session max_cameras=3
    try:
        ids = eng.client.cameraModel.camera_ids()
        assert len(ids) == 1
        rec = eng.client.cameraModel.get_record(ids[0])
        assert rec.camera_config.source.type == "synthetic"
        assert eng.camera_ids == ids
    finally:
        eng.stop()


def test_add_next_camera_grows_wall_and_caps_at_max():
    eng, made = _factory()  # session max_cameras=3
    eng.start()
    try:
        assert len(eng.client.cameraModel.camera_ids()) == 1
        cid2 = eng.add_next_camera()
        assert cid2 is not None
        assert len(eng.client.cameraModel.camera_ids()) == 2
        cid3 = eng.add_next_camera()
        assert cid3 is not None
        assert len(eng.client.cameraModel.camera_ids()) == 3
        # Capped at session.max_cameras (3): no further growth.
        assert eng.add_next_camera() is None
        assert len(eng.client.cameraModel.camera_ids()) == 3
        # All synthetic.
        for cid in eng.client.cameraModel.camera_ids():
            rec = eng.client.cameraModel.get_record(cid)
            assert rec.camera_config.source.type == "synthetic"
    finally:
        eng.stop()


def test_add_next_camera_emits_cameraAdded_for_the_wall():
    """The wall builds tiles from the client's ``cameraAdded`` signal, but the
    progressive add mutates the camera model DIRECTLY — so ``add_next_camera`` must
    emit ``cameraAdded`` itself, otherwise the grown cameras never become tiles
    (the observed "1 tile for N cams" bug: model count grows, wall stays at one)."""
    eng, _ = _factory()
    eng.start()
    seen: list[str] = []
    eng.client.cameraAdded.connect(seen.append)
    try:
        cid2 = eng.add_next_camera()
        cid3 = eng.add_next_camera()
        assert cid2 in seen and cid3 in seen
    finally:
        eng.stop()


def test_start_stop_reflect_engine_running_for_status_bar():
    """The factory starts the supervisor directly (not ``client.startEngine``), so it
    must mark the isolated client running/stopped — else the Mark status bar reads
    'Engine stopped' through the whole demo."""
    eng, _ = _factory()
    try:
        eng.start()
        assert eng.client.engineRunning is True
    finally:
        eng.stop()
    assert eng.client.engineRunning is False


def test_auto_track_sets_targets_full_profile(qapp):
    """The full profile auto-tracks a (seeded) target per camera so Center Stage
    visibly engages.  Uses the engine's existing target-set path (client.setTarget),
    which records the target on the camera model and enqueues a SetTargetCmd."""
    from autoptz.ui import mark_engine

    # A recording supervisor that also routes set_target to a fake worker.
    targets: dict[str, int] = {}

    class _RecSup(_FakeSupervisor):
        pass

    eng = mark_engine.MarkEngineFactory(
        MarkSession(profile="full", source="synthetic", max_cameras=3),
        supervisor_factory=lambda c, s: _RecSup(c, s),
    )
    eng.start()
    try:
        # Grow the wall to 3 cameras, then auto-track.
        eng.add_next_camera()
        eng.add_next_camera()
        eng.auto_track_targets(seed=1234)
        ids = eng.client.cameraModel.camera_ids()
        assert len(ids) == 3
        for cid in ids:
            rec = eng.client.cameraModel.get_record(cid)
            # A target track id was committed on every camera (non-None, >= 1).
            assert rec.target_track_id is not None
            assert rec.target_track_id >= 1
            targets[cid] = rec.target_track_id
        # Deterministic for the same seed (a fresh engine yields the same per-index
        # targets, keyed by position not id since ids are random uuids).
        eng2 = mark_engine.MarkEngineFactory(
            MarkSession(profile="full", source="synthetic", max_cameras=3),
            supervisor_factory=lambda c, s: _RecSup(c, s),
        )
        eng2.start()
        try:
            eng2.add_next_camera()
            eng2.add_next_camera()
            eng2.auto_track_targets(seed=1234)
            t1 = [
                eng.client.cameraModel.get_record(c).target_track_id
                for c in eng.client.cameraModel.camera_ids()
            ]
            t2 = [
                eng2.client.cameraModel.get_record(c).target_track_id
                for c in eng2.client.cameraModel.camera_ids()
            ]
            assert t1 == t2
        finally:
            eng2.stop()
    finally:
        eng.stop()


def test_auto_track_noop_when_not_full_profile(qapp):
    """The streams profile (no inference) does NOT auto-track — nothing to follow."""
    from autoptz.ui import mark_engine

    eng = mark_engine.MarkEngineFactory(
        MarkSession(profile="streams", source="synthetic", max_cameras=2),
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    eng.start()
    try:
        eng.auto_track_targets(seed=1)
        for cid in eng.client.cameraModel.camera_ids():
            rec = eng.client.cameraModel.get_record(cid)
            assert rec.target_track_id is None
    finally:
        eng.stop()


def test_model_choice_primes_detector_tier():
    """The session model selects the isolated client's detector tier.

    "auto" → "auto" (default), "nano" → "fast", "small" → "balanced".  The tier is
    set on the isolated store BEFORE the client loads it, so the supervisor picks
    it up at start without touching the main app's tier.
    """
    from autoptz.ui import mark_engine

    for model, expected in (("auto", "auto"), ("nano", "fast"), ("small", "balanced")):
        eng = mark_engine.MarkEngineFactory(
            MarkSession(source="synthetic", max_cameras=2, model=model),
            supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
        )
        try:
            assert eng.client.getDetectorModelTier() == expected
            # Persisted on the ISOLATED store only.
            assert eng.store.get_setting("detector_model_tier", "auto") == expected
        finally:
            eng.stop()


def test_synthetic_cameras_use_session_resolution():
    """The pre-added + grown synthetic cameras carry the session's resolution size."""
    from autoptz.ui import mark_engine

    eng = mark_engine.MarkEngineFactory(
        MarkSession(source="synthetic", max_cameras=3, resolution="1080p"),
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    try:
        cid = eng.client.cameraModel.camera_ids()[0]
        src = eng.client.cameraModel.get_record(cid).camera_config.source
        assert (src.width, src.height) == (1920, 1080)
    finally:
        eng.stop()


def test_clip_source_registers_cameras_with_clip_path(monkeypatch):
    """A clip-source factory registers synthetic cameras whose address is the
    bundled clip path, so the SyntheticAdapter loops the real clip (real decode,
    real people) instead of drawing synthetic people."""
    from autoptz.ui import mark_engine

    # Force the bundled clip "present" so this asserts the clip-path wiring even on a
    # checkout where the asset isn't installed (it's an untracked bundled asset).
    monkeypatch.setattr(MarkSession, "clip_available", lambda self: True)

    session = MarkSession(source="clip", max_cameras=3, resolution="1080p")
    clip = session.clip_path()
    eng = mark_engine.MarkEngineFactory(
        session,
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    eng.start()
    try:
        # Pre-added first camera carries the clip path + session resolution.
        ids = eng.client.cameraModel.camera_ids()
        assert len(ids) == 1
        rec = eng.client.cameraModel.get_record(ids[0])
        assert rec.camera_config.source.type == "synthetic"
        assert rec.camera_config.source.address == clip
        assert (rec.camera_config.source.width, rec.camera_config.source.height) == (1920, 1080)
        # Grown cameras keep the clip path too.
        cid2 = eng.add_next_camera()
        assert cid2 is not None
        rec2 = eng.client.cameraModel.get_record(cid2)
        assert rec2.camera_config.source.address == clip
    finally:
        eng.stop()


def test_clip_source_falls_back_to_anim_when_clip_missing(monkeypatch):
    """When the bundled clip is absent, a clip-source session degrades to the drawn
    ("anim") scene instead of silently feeding the SyntheticAdapter a dead path."""
    from autoptz.ui import mark_engine

    # Simulate a checkout without the bundled clip.
    monkeypatch.setattr(MarkSession, "clip_available", lambda self: False)

    session = MarkSession(source="clip", max_cameras=2, resolution="1080p")
    eng = mark_engine.MarkEngineFactory(
        session,
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    eng.start()
    try:
        ids = eng.client.cameraModel.camera_ids()
        rec = eng.client.cameraModel.get_record(ids[0])
        assert rec.camera_config.source.type == "synthetic"
        # Falls back to the drawn scene, NOT the (missing) clip path.
        assert rec.camera_config.source.address == "anim"
    finally:
        eng.stop()


def test_start_then_stop_is_clean():
    eng, made = _factory()
    eng.start()
    eng.tick()
    eng.tick()
    assert made["sup"].started and made["sup"].ticks == 2
    eng.stop()
    assert made["sup"].stopped
    # store connection closed + temp dir removed
    assert eng.store._conn is None


def test_main_store_untouched(tmp_path):
    # A real ConfigStore on a separate path must be unaffected by Mark's lifecycle.
    main = ConfigStore(db_path=tmp_path / "main.db")
    main.set_setting("engine_running", True)
    eng, _ = _factory()
    eng.start()
    eng.stop()
    assert main.get_setting("engine_running") is True
    main.close()


def test_features_primed_from_profile():
    eng, made = _factory()
    try:
        # The fake supervisor records what prime_features received.
        assert isinstance(made["sup"]._features, dict)
        assert made["sup"]._features  # non-empty (profile feature flags)
    finally:
        eng.stop()


def test_factory_owns_a_frame_source():
    """The factory exposes its OWN ShmFrameSource (the Mark wall binds to it).

    The blank-tile bug was the factory owning no frame source: synthetic workers
    wrote to the Mark engine's shm but nothing was attached to read it, so every
    tile stayed blank.  The factory must own one so the window can bind its wall.
    """
    from autoptz.ui.frames import ShmFrameSource

    eng, _ = _factory()
    try:
        assert isinstance(eng.frame_source, ShmFrameSource)
    finally:
        eng.stop()


def test_provider_attach_detach_wired_to_frame_source(qapp):
    """The isolated client's provider attach/detach drive the factory's frame source.

    Mirrors app.py's wiring (~304-311): providerAttachRequested(cid,shm,w,h) →
    frames.attach(cid,shm,h,w) and providerDetachRequested → frames.detach, both
    queued.  Without this the Mark wall never reads the synthetic workers' shm.
    """
    from PySide6.QtCore import QCoreApplication

    eng, _ = _factory()
    try:
        frames = eng.frame_source
        # Emit an attach on the isolated client; QueuedConnection delivers on pump.
        eng.client.providerAttachRequested.emit("cam-1", "cam_abc_preview", 1280, 720)
        QCoreApplication.instance().processEvents()
        assert frames.is_known("cam-1")
        # The h/w order is honored (attach takes height,width; signal carries w,h).
        shm_name, height, width = frames._intents["cam-1"]
        assert (shm_name, height, width) == ("cam_abc_preview", 720, 1280)
        # Detach removes it.
        eng.client.providerDetachRequested.emit("cam-1")
        QCoreApplication.instance().processEvents()
        assert not frames.is_known("cam-1")
    finally:
        eng.stop()


def test_stop_detaches_frame_source(qapp):
    from PySide6.QtCore import QCoreApplication

    eng, _ = _factory()
    eng.client.providerAttachRequested.emit("cam-9", "cam_xyz_preview", 1280, 720)
    QCoreApplication.instance().processEvents()
    assert eng.frame_source.is_known("cam-9")
    eng.stop()
    # Teardown releases every reader/intent so the discarded session leaks nothing.
    assert not eng.frame_source.is_known("cam-9")
