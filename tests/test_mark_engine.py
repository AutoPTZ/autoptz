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


def test_only_fake_cameras_registered():
    eng, _ = _factory()
    try:
        ids = eng.client.cameraModel.camera_ids()
        assert len(ids) == 3
        # All synthetic, named "AutoPTZ Mark N"
        for cid in ids:
            rec = eng.client.cameraModel.get_record(cid)
            assert rec.camera_config.source.type == "synthetic"
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
