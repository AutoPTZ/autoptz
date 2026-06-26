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
