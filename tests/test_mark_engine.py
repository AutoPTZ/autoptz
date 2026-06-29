"""MarkEngineFactory: isolated stack, fake-only cameras, clean teardown, main store untouched."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


@pytest.fixture(autouse=True)
def _no_clip_by_default(monkeypatch):
    """Default the bundled clip to ABSENT so the generic lifecycle tests (``_factory``)
    deterministically use the drawn scene and never depend on the developer's real
    transcode cache or trigger a (slow) variant build.  Clip-specific tests override
    this with their own ``monkeypatch.setattr(MarkSession, "clip_available", ...)``.
    """
    monkeypatch.setattr(MarkSession, "clip_available", lambda self: False)


def _factory():
    """A clip-source engine whose clip is absent (see ``_no_clip_by_default``), so it
    deterministically falls back to the drawn scene — the right default now that the
    user-facing 'synthetic' drawn source is removed (source is only 'clip' | 'ndi')."""
    from autoptz.ui import mark_engine

    made = {}

    def fake_sup(client, store):
        s = _FakeSupervisor(client, store)
        made["sup"] = s
        return s

    eng = mark_engine.MarkEngineFactory(
        MarkSession(source="clip", max_cameras=3),
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
        MarkSession(profile="full", source="clip", max_cameras=3),
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
            MarkSession(profile="full", source="clip", max_cameras=3),
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
        MarkSession(profile="streams", source="clip", max_cameras=2),
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


def _center_stage_on(eng, cid) -> bool:
    """Center Stage = the engine's group_framing knob on the camera's tracking config."""
    rec = eng.client.cameraModel.get_record(cid)
    return bool(rec.camera_config.tracking.group_framing)


def test_full_profile_enables_tracking_and_center_stage_per_camera(qapp):
    """The full profile must turn ON tracking AND Center Stage (group_framing) on EVERY
    camera, so each tile visibly tracks + auto-frames — not just set a target."""
    from autoptz.ui import mark_engine

    eng = mark_engine.MarkEngineFactory(
        MarkSession(profile="full", source="clip", max_cameras=3),
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    eng.start()
    try:
        eng.add_next_camera()
        eng.add_next_camera()
        eng.auto_track_targets(seed=7)
        ids = eng.client.cameraModel.camera_ids()
        assert len(ids) == 3
        for cid in ids:
            rec = eng.client.cameraModel.get_record(cid)
            assert rec.tracking_enabled is True, f"tracking not enabled for {cid}"
            assert _center_stage_on(eng, cid), f"Center Stage not on for {cid}"
    finally:
        eng.stop()


def test_grown_camera_gets_tracking_and_center_stage_full_profile(qapp):
    """Newly-grown tiles must ALSO come up with tracking + Center Stage ON (re-applied
    as the wall grows) — the dynamic activation, not only the initial cameras."""
    from autoptz.ui import mark_engine

    eng = mark_engine.MarkEngineFactory(
        MarkSession(profile="full", source="clip", max_cameras=3),
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    eng.start()
    try:
        cid2 = eng.add_next_camera()
        assert cid2 is not None
        rec2 = eng.client.cameraModel.get_record(cid2)
        # The grown camera came up tracking + auto-framing without an auto_track pass.
        assert rec2.tracking_enabled is True
        assert _center_stage_on(eng, cid2)
    finally:
        eng.stop()


def test_streams_profile_keeps_tracking_and_center_stage_off(qapp):
    """The streams profile keeps tracking OFF and never turns on Center Stage — even on
    grown tiles and after an auto_track pass."""
    from autoptz.ui import mark_engine

    eng = mark_engine.MarkEngineFactory(
        MarkSession(profile="streams", source="clip", max_cameras=3),
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    eng.start()
    try:
        eng.add_next_camera()
        eng.auto_track_targets(seed=3)
        for cid in eng.client.cameraModel.camera_ids():
            rec = eng.client.cameraModel.get_record(cid)
            assert rec.tracking_enabled is False
            assert _center_stage_on(eng, cid) is False
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
            MarkSession(source="clip", max_cameras=2, model=model),
            supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
        )
        try:
            assert eng.client.getDetectorModelTier() == expected
            # Persisted on the ISOLATED store only.
            assert eng.store.get_setting("detector_model_tier", "auto") == expected
        finally:
            eng.stop()


def test_native_fps_to_synthetic_camera(monkeypatch):
    """The selected clip's native fps flows to the synthetic cameras' source.

    A "cinematic_60" clip session sets source.fps == 60.0 on the registered cameras.
    This verifies the registry-metadata path (native_fps from clip_info()) drives
    the source fps regardless of whether the asset is on disk, so we force the clip
    "present" rather than depending on the (parallel-produced) file existing.
    """
    from autoptz.ui import mark_engine

    monkeypatch.setattr(MarkSession, "clip_available", lambda self: True)
    # Stub the cache so the (res, fps) variant resolves deterministically without
    # depending on the developer's real transcode cache or triggering a slow build.
    _stub_cache_factory(monkeypatch, _StubCache(cached="/tmp/mark-cache/cinematic_60/var.mp4"))
    eng = mark_engine.MarkEngineFactory(
        MarkSession(source="clip", clip_id="cinematic_60", max_cameras=3),
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    eng.start()
    try:
        ids = eng.client.cameraModel.camera_ids()
        assert len(ids) == 1
        rec = eng.client.cameraModel.get_record(ids[0])
        assert rec.camera_config.source.fps == 60.0
        # Grown cameras keep the native fps too.
        cid2 = eng.add_next_camera()
        assert cid2 is not None
        assert eng.client.cameraModel.get_record(cid2).camera_config.source.fps == 60.0
    finally:
        eng.stop()


def test_synthetic_cameras_use_session_resolution():
    """The pre-added + grown synthetic cameras carry the session's resolution size."""
    from autoptz.ui import mark_engine

    eng = mark_engine.MarkEngineFactory(
        MarkSession(source="clip", max_cameras=3, resolution="1080p"),
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
    # Stub the transcode cache so it deterministically falls back to the raw master
    # (cache miss + build raises) regardless of any real on-disk cached variant —
    # this test asserts the master clip path reaches the camera, not the variant.
    _stub_cache_factory(monkeypatch, _StubCache(cached=None, build_raises=True))

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


def test_drawn_anim_scene_only_reachable_via_gt_env(monkeypatch):
    """The drawn ("anim") scene is NO LONGER a user source: it is reachable only as
    the ground-truth scene gated by AUTOPTZ_MARK_GT.  With the clip present and GT on,
    the cameras draw the synthetic scene; with GT off they broadcast the real clip."""
    from autoptz.ui import mark_engine

    monkeypatch.setattr(MarkSession, "clip_available", lambda self: True)
    cached = "/tmp/mark-cache/crowd/1920x1080_30fps.mp4"
    _stub_cache_factory(monkeypatch, _StubCache(cached=cached))
    monkeypatch.setenv("AUTOPTZ_MARK_GT", "1")

    session = MarkSession(source="clip", max_cameras=2, resolution="1080p")
    eng = mark_engine.MarkEngineFactory(
        session,
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    try:
        rec = eng.client.cameraModel.get_record(eng.client.cameraModel.camera_ids()[0])
        # GT on → the drawn ground-truth scene, even though a real clip is present.
        assert rec.camera_config.source.address == "anim"
    finally:
        eng.stop()


def test_clip_present_feeds_real_clip_not_drawn_scene_without_gt(monkeypatch):
    """Without AUTOPTZ_MARK_GT, a clip session feeds the REAL clip variant to every
    camera — never the drawn 'anim' scene (the drawn source is removed)."""
    from autoptz.ui import mark_engine

    monkeypatch.delenv("AUTOPTZ_MARK_GT", raising=False)
    monkeypatch.setattr(MarkSession, "clip_available", lambda self: True)
    cached = "/tmp/mark-cache/crowd/1920x1080_30fps.mp4"
    _stub_cache_factory(monkeypatch, _StubCache(cached=cached))

    session = MarkSession(source="clip", max_cameras=2, resolution="1080p")
    eng = mark_engine.MarkEngineFactory(
        session,
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    try:
        rec = eng.client.cameraModel.get_record(eng.client.cameraModel.camera_ids()[0])
        assert rec.camera_config.source.address == cached
        assert rec.camera_config.source.address != "anim"
    finally:
        eng.stop()


class _StubCache:
    """A stand-in TranscodeCache recording calls for the cache-wiring tests."""

    def __init__(self, *, cached=None, built=None, build_raises=False):
        self._cached = cached
        self._built = built
        self._build_raises = build_raises
        self.get_calls: list[tuple] = []
        self.build_calls: list[dict] = []

    def get_cached_variant(self, clip_id, target_res, target_fps):
        self.get_calls.append((clip_id, target_res, target_fps))
        return self._cached

    def build_cached_variant(
        self, clip_id, master_path, master_res, master_fps, target_res, target_fps
    ):
        self.build_calls.append(
            {
                "clip_id": clip_id,
                "master_path": master_path,
                "master_res": master_res,
                "master_fps": master_fps,
                "target_res": target_res,
                "target_fps": target_fps,
            }
        )
        if self._build_raises:
            raise RuntimeError("transcode boom")
        return self._built


def _stub_cache_factory(monkeypatch, cache):
    """Make ``MarkEngineFactory.__init__`` build ``cache`` instead of a real one.

    The factory builds its cache via the module-level ``_make_transcode_cache``
    helper, so patching it injects the stub BEFORE ``_setup_fake_cameras`` runs
    (the cache must be in place when the first camera's address is resolved).
    """
    from autoptz.ui import mark_engine

    monkeypatch.setattr(mark_engine, "_make_transcode_cache", lambda: cache)


def test_clip_uses_cached_variant_as_camera_address(monkeypatch):
    """A cache HIT feeds the cached variant path to the synthetic camera (not the
    raw master): the picked resolution/fps drives a native cached variant."""
    from autoptz.ui import mark_engine

    monkeypatch.setattr(MarkSession, "clip_available", lambda self: True)
    cached = "/tmp/mark-cache/crowd/1920x1080_30fps.mp4"
    cache = _StubCache(cached=cached)
    _stub_cache_factory(monkeypatch, cache)

    session = MarkSession(source="clip", max_cameras=3, resolution="1080p")
    eng = mark_engine.MarkEngineFactory(
        session,
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    try:
        ids = eng.client.cameraModel.camera_ids()
        assert len(ids) == 1
        rec = eng.client.cameraModel.get_record(ids[0])
        assert rec.camera_config.source.address == cached
        # The cache was consulted with the session's resolution + target fps.
        assert cache.get_calls
        clip_id, res, fps = cache.get_calls[-1]
        assert clip_id == session.clip_info().id
        assert res == session.resolution_size()
        assert fps == session.target_fps()
        # A cache hit must NOT trigger a (slow) build.
        assert cache.build_calls == []
    finally:
        eng.stop()


def test_clip_cache_miss_builds_variant_once_then_uses_it(monkeypatch):
    """On a cache MISS the engine builds the variant exactly once and feeds the
    built path; subsequent cameras reuse it without rebuilding."""
    from autoptz.ui import mark_engine

    monkeypatch.setattr(MarkSession, "clip_available", lambda self: True)
    built = "/tmp/mark-cache/crowd/built_variant.mp4"
    cache = _StubCache(cached=None, built=built)
    _stub_cache_factory(monkeypatch, cache)

    session = MarkSession(source="clip", max_cameras=3, resolution="1080p")
    eng = mark_engine.MarkEngineFactory(
        session,
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    eng.start()
    try:
        ids = eng.client.cameraModel.camera_ids()
        assert len(ids) == 1
        rec = eng.client.cameraModel.get_record(ids[0])
        assert rec.camera_config.source.address == built
        # Built exactly once with the master metadata + session targets.
        assert len(cache.build_calls) == 1
        call = cache.build_calls[0]
        assert call["clip_id"] == session.clip_info().id
        assert str(call["master_path"]) == session.clip_path()
        assert call["master_res"] == session.clip_info().native_resolution
        assert call["master_fps"] == session.clip_info().native_fps
        assert call["target_res"] == session.resolution_size()
        assert call["target_fps"] == session.target_fps()
        # Growing the wall reuses the built path WITHOUT a second build.
        cid2 = eng.add_next_camera()
        assert cid2 is not None
        rec2 = eng.client.cameraModel.get_record(cid2)
        assert rec2.camera_config.source.address == built
        assert len(cache.build_calls) == 1
    finally:
        eng.stop()


def test_clip_build_failure_falls_back_to_master(monkeypatch):
    """If building the variant raises, the engine falls back to the raw master
    clip path (the demo must never crash on a transcode failure)."""
    from autoptz.ui import mark_engine

    monkeypatch.setattr(MarkSession, "clip_available", lambda self: True)
    cache = _StubCache(cached=None, build_raises=True)
    _stub_cache_factory(monkeypatch, cache)

    session = MarkSession(source="clip", max_cameras=2, resolution="1080p")
    clip = session.clip_path()
    eng = mark_engine.MarkEngineFactory(
        session,
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    try:
        ids = eng.client.cameraModel.camera_ids()
        rec = eng.client.cameraModel.get_record(ids[0])
        # No exception bubbled; address is the raw master clip path.
        assert rec.camera_config.source.address == clip
        assert len(cache.build_calls) == 1
    finally:
        eng.stop()


# ── NDI broadcasts the selected clip (not the drawn scene) ────────────────────


class _RecFleet:
    """A recording stand-in for MarkNDIFleet (cyndilib absent in CI/.venv)."""

    calls: list[dict] = []

    def __init__(self, n, *, width, height, fps=30.0, frame_source=None):
        type(self).calls.append(
            {
                "n": n,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_source": frame_source,
            }
        )
        self._n = n

    def full_names(self, **_kwargs):
        return [f"HOST (AutoPTZ Mark Cam {i + 1})" for i in range(self._n)]

    def open(self):
        pass

    def pump_once(self):
        pass

    def close(self):
        pass


def _patch_ndi_fleet(monkeypatch):
    """Make the NDI branch usable without cyndilib: available + a recording fleet."""
    from autoptz.benchmark import ndi_sim

    _RecFleet.calls = []
    monkeypatch.setattr(ndi_sim, "ndi_sim_available", lambda: True)
    monkeypatch.setattr(ndi_sim, "MarkNDIFleet", _RecFleet)
    # The first ndi:// camera is registered on the client during setup; keep the
    # real _add_ndi_camera so the camera model is populated as in production.


def test_ndi_fleet_built_with_clip_variant_frame_source(monkeypatch):
    """NDI mode must broadcast the SELECTED clip (real footage), not the drawn scene.

    The factory resolves the transcode-cached variant for (clip_id, resolution, fps)
    and threads it into MarkNDIFleet as ``frame_source`` so every NDI tile shows the
    same real video as clip mode."""
    from autoptz.ui import mark_engine

    monkeypatch.setattr(MarkSession, "clip_available", lambda self: True)
    cached = "/tmp/mark-cache/crowd/1280x720_30fps.mp4"
    cache = _StubCache(cached=cached)
    _stub_cache_factory(monkeypatch, cache)
    _patch_ndi_fleet(monkeypatch)

    session = MarkSession(source="ndi", max_cameras=3, resolution="720p")
    eng = mark_engine.MarkEngineFactory(
        session,
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    try:
        assert _RecFleet.calls, "the NDI fleet was never built"
        call = _RecFleet.calls[-1]
        # Built at the chosen resolution and fed the resolved clip variant.
        assert call["frame_source"] == cached
        assert (call["width"], call["height"]) == session.resolution_size()
    finally:
        eng.stop()


def test_ndi_fleet_frame_source_falls_back_to_master_when_variant_fails(monkeypatch):
    """If the variant can't build, NDI broadcasts the raw master clip (never crashes)."""
    from autoptz.ui import mark_engine

    monkeypatch.setattr(MarkSession, "clip_available", lambda self: True)
    cache = _StubCache(cached=None, build_raises=True)
    _stub_cache_factory(monkeypatch, cache)
    _patch_ndi_fleet(monkeypatch)

    session = MarkSession(source="ndi", max_cameras=2, resolution="720p")
    master = session.clip_path()
    eng = mark_engine.MarkEngineFactory(
        session,
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    try:
        call = _RecFleet.calls[-1]
        assert call["frame_source"] == master
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


def test_provider_attach_detach_wired_to_frame_source(qapp, wait_until):
    """The isolated client's provider attach/detach drive the factory's frame source.

    Mirrors app.py's wiring (~304-311): providerAttachRequested(cid,shm,w,h) →
    frames.attach(cid,shm,h,w) and providerDetachRequested → frames.detach, both
    queued.  Without this the Mark wall never reads the synthetic workers' shm.
    """
    eng, _ = _factory()
    try:
        frames = eng.frame_source
        # Emit an attach on the isolated client; QueuedConnection delivers on pump.
        eng.client.providerAttachRequested.emit("cam-1", "cam_abc_preview", 1280, 720)
        wait_until(
            lambda: frames.is_known("cam-1"),
            timeout=2.0,
            message="provider attach signal was not delivered",
            pump_qt=True,
        )
        # The h/w order is honored (attach takes height,width; signal carries w,h).
        shm_name, height, width = frames._intents["cam-1"]
        assert (shm_name, height, width) == ("cam_abc_preview", 720, 1280)
        # Detach removes it.
        eng.client.providerDetachRequested.emit("cam-1")
        wait_until(
            lambda: not frames.is_known("cam-1"),
            timeout=2.0,
            message="provider detach signal was not delivered",
            pump_qt=True,
        )
    finally:
        eng.stop()


def test_stop_detaches_frame_source(qapp, wait_until):
    eng, _ = _factory()
    eng.client.providerAttachRequested.emit("cam-9", "cam_xyz_preview", 1280, 720)
    wait_until(
        lambda: eng.frame_source.is_known("cam-9"),
        timeout=2.0,
        message="provider attach signal was not delivered",
        pump_qt=True,
    )
    eng.stop()
    # Teardown releases every reader/intent so the discarded session leaks nothing.
    assert not eng.frame_source.is_known("cam-9")


# ── PHASE 1 (Step 1.5): leak-proof stop + started-last start ──────────────────


def test_stop_continues_cleanup_when_supervisor_stop_raises():
    """A failing supervisor.stop() must NOT block the rest of teardown: the frame
    source is still detached and the temp dir is still removed."""
    from autoptz.ui import mark_engine

    class _BoomStopSup(_FakeSupervisor):
        def stop(self):
            raise RuntimeError("supervisor stop boom")

    eng = mark_engine.MarkEngineFactory(
        MarkSession(source="clip", max_cameras=2),
        supervisor_factory=lambda c, s: _BoomStopSup(c, s),
    )
    tmpdir = eng._tmpdir
    detached = {"n": 0}
    real_detach = eng.frame_source.detach_all

    def _spy_detach():
        detached["n"] += 1
        return real_detach()

    eng._frame_source.detach_all = _spy_detach  # type: ignore[method-assign]
    eng.stop()  # must not raise despite the supervisor blowing up
    assert detached["n"] == 1  # frame source still detached
    assert eng.store._conn is None  # store still closed
    assert not tmpdir.exists()  # temp dir still removed


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows file-locking keeps the forced-open sqlite handle so rmtree can't "
    "remove the dir; the product does best-effort ignore_errors cleanup. The "
    "close-failure-doesn't-block-cleanup ordering is verified on POSIX.",
)
def test_stop_removes_tmpdir_even_if_store_close_raises():
    """A failing store.close() must NOT block the temp-dir removal."""
    from autoptz.ui import mark_engine

    eng = mark_engine.MarkEngineFactory(
        MarkSession(source="clip", max_cameras=2),
        supervisor_factory=lambda c, s: _FakeSupervisor(c, s),
    )
    tmpdir = eng._tmpdir

    def _boom_close():
        raise RuntimeError("store close boom")

    eng._store.close = _boom_close  # type: ignore[method-assign]
    eng.stop()  # must not raise
    assert not tmpdir.exists()  # temp dir gone despite the store close failing


def test_start_does_not_set_started_if_supervisor_start_raises():
    """If supervisor.start() raises, is_started must stay False (a half-started
    engine must not advertise itself as running)."""
    from autoptz.ui import mark_engine

    class _BoomStartSup(_FakeSupervisor):
        def start(self, *, run_pump=False, staged=False, progress=None):
            raise RuntimeError("supervisor start boom")

    eng = mark_engine.MarkEngineFactory(
        MarkSession(source="clip", max_cameras=2),
        supervisor_factory=lambda c, s: _BoomStartSup(c, s),
    )
    try:
        with pytest.raises(RuntimeError):
            eng.start()
        assert eng.is_started is False
    finally:
        eng.stop()
