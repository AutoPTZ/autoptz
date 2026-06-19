"""Phase 8 tests: face/ReID identity engine + EngineClient identity API.

All tests are headless and mock the heavy ML stack (insightface, boxmot/OSNet)
with fakes — no model download, no network, no GPU.  They cover:

- gallery CRUD: enroll / label / rename / delete / setEnabled / merge
- retention policy: labeled-only persisted (unlabeled never written to the DB)
- auto-harvest: an unmatched good face creates an unlabeled identity + thumbnail
- target-by-identity: the worker locks the single target onto the matched track
- the FROZEN IdentityListModel roles incl. the base64 ``thumbnail`` data URI
- graceful no-op when insightface is absent (FaceRecognizer disabled)
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

import numpy as np
import pytest

from autoptz.config.models import CameraConfig, IdentityRecord, SourceConfig
from autoptz.engine.pipeline.identify import (
    FaceRecognizer,
    cosine,
    embedding_from_bytes,
    embedding_to_bytes,
    normalize,
)

# ── fakes ───────────────────────────────────────────────────────────────────


class _FakeFace:
    """Duck-typed insightface Face: bbox + normed_embedding + det_score."""

    def __init__(self, bbox, emb, det_score=0.95):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.normed_embedding = normalize(np.asarray(emb, dtype=np.float32))
        self.det_score = det_score


class _FakeApp:
    """Duck-typed FaceAnalysis: returns a fixed list of faces from .get()."""

    def __init__(self, faces=None):
        self._faces = faces or []

    def set_faces(self, faces):
        self._faces = faces

    def get(self, frame):  # noqa: ARG002
        return list(self._faces)


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize(rng.standard_normal(dim).astype(np.float32))


def _make_store(tmp_path: Path):
    from autoptz.config.store import ConfigStore
    return ConfigStore(db_path=tmp_path / "identity.db", debounce_s=0)


def _make_service(tmp_path: Path | None = None):
    from autoptz.engine.identity.service import IdentityService
    store = _make_store(tmp_path) if tmp_path else None
    return IdentityService(store), store


# ── embedding helpers ─────────────────────────────────────────────────────────


class TestEmbeddingHelpers:
    def test_normalize_unit_length(self):
        v = normalize(np.array([3.0, 4.0]))
        assert float(np.linalg.norm(v)) == pytest.approx(1.0)

    def test_normalize_zero_safe(self):
        v = normalize(np.zeros(4))
        assert not np.any(np.isnan(v))

    def test_bytes_roundtrip(self):
        v = _vec(1)
        back = embedding_from_bytes(embedding_to_bytes(v))
        assert cosine(v, back) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_identical_is_one(self):
        v = _vec(2)
        assert cosine(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_mismatched_shapes_zero(self):
        assert cosine(np.ones(4), np.ones(8)) == 0.0


# ── FaceRecognizer ──────────────────────────────────────────────────────────────


class TestFaceRecognizer:
    def test_absent_insightface_is_graceful(self, monkeypatch):
        # Simulate insightface being unavailable (the package may now be
        # installed in the venv) so we exercise the graceful-degradation path:
        # _try_init must swallow the ImportError, disable, and no-op.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "insightface" or name.startswith("insightface."):
                raise ImportError("simulated: insightface unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rec = FaceRecognizer()
        assert rec.available is False
        assert rec.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []

    def test_injected_app_available(self):
        rec = FaceRecognizer(_app=_FakeApp())
        assert rec.available is True

    def test_detect_extracts_observations(self):
        emb = _vec(3)
        app = _FakeApp([_FakeFace((10, 20, 60, 90), emb)])
        rec = FaceRecognizer(_app=app)
        obs = rec.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(obs) == 1
        assert obs[0].bbox == (10.0, 20.0, 60.0, 90.0)
        assert cosine(obs[0].embedding, emb) == pytest.approx(1.0, abs=1e-5)

    def test_match_hits_enrolled_identity(self):
        service, _ = _make_service()
        emb = _vec(4)
        ident = service.enroll("Alice", emb)
        rec = FaceRecognizer(_app=_FakeApp(), match_threshold=0.35)
        m = rec.match(emb, service)
        assert m is not None
        assert m.identity_id == ident.id
        assert m.score == pytest.approx(1.0, abs=1e-5)

    def test_match_misses_below_threshold(self):
        service, _ = _make_service()
        service.enroll("Alice", _vec(5))
        rec = FaceRecognizer(_app=_FakeApp(), match_threshold=0.99)
        assert rec.match(_vec(6), service) is None

    def test_match_skips_disabled_identity(self):
        service, _ = _make_service()
        emb = _vec(7)
        ident = service.enroll("Alice", emb)
        service.set_enabled(ident.id, False)
        rec = FaceRecognizer(_app=_FakeApp(), match_threshold=0.1)
        assert rec.match(emb, service) is None

    def test_match_include_disabled_recognises_harvested(self):
        # Recognition / dedup path: an auto-harvested (disabled) identity MUST be
        # matched when include_disabled=True, so the same face is recognised
        # instead of being re-harvested as a duplicate.
        service, _ = _make_service()
        emb = _vec(7)
        u = service.add_unlabeled(emb)              # disabled by policy
        rec = FaceRecognizer(_app=_FakeApp(), match_threshold=0.35)
        assert rec.match(emb, service) is None      # enabled-only: skipped
        m = rec.match(emb, service, include_disabled=True)
        assert m is not None and m.identity_id == u.id

    def test_harvest_dedup_one_identity_per_person(self):
        # Simulate two face ticks for the same person: the second must recognise
        # the first harvested identity (no duplicate). A different face harvests
        # a second identity.
        service, _ = _make_service()
        rec = FaceRecognizer(_app=_FakeApp(), match_threshold=0.35)
        person_a = _vec(40)
        # tick 1: unmatched → harvest
        assert rec.match(person_a, service, include_disabled=True) is None
        a = service.add_unlabeled(person_a)
        # tick 2: same person → recognised, no new harvest
        m = rec.match(person_a, service, include_disabled=True)
        assert m is not None and m.identity_id == a.id
        assert len(service.all_identities()) == 1
        # a different person → unmatched → would harvest a second identity
        person_b = _vec(41)
        assert rec.match(person_b, service, include_disabled=True) is None

    def test_matchable_identities_includes_disabled(self):
        service, _ = _make_service()
        service.add_unlabeled(_vec(42))             # disabled
        en = service.enroll("Bob", _vec(43))        # enabled
        assert {r.id for r in service.matchable_identities()} == {
            r.id for r in service.all_identities()
        }
        assert en.id in {r.id for r in service.enabled_identities()}
        assert len(service.enabled_identities()) == 1


# ── IdentityService gallery CRUD + retention ──────────────────────────────────


class TestIdentityServiceCRUD:
    def test_enroll_labeled_and_enabled(self):
        service, _ = _make_service()
        rec = service.enroll("Alice", _vec(10))
        assert rec.labeled is True
        assert rec.enabled is True
        assert rec in service.labeled_identities()

    def test_add_unlabeled_is_memory_only(self):
        service, _ = _make_service()
        rec = service.add_unlabeled(_vec(11), thumbnail=b"png")
        assert rec.labeled is False
        assert rec.enabled is False
        assert rec.name.startswith("Person ")
        assert rec.thumbnail == b"png"
        assert rec in service.unlabeled_identities()

    def test_expire_unlabeled_drops_stale_keeps_labeled(self):
        from datetime import timedelta

        from autoptz.engine.identity.service import _now

        service, _ = _make_service()
        labeled = service.enroll("Alice", _vec(10))
        stale = service.add_unlabeled(_vec(11))
        fresh = service.add_unlabeled(_vec(12))
        old = _now() - timedelta(seconds=120)
        # Age the stale unlabeled + the labeled record into the past.
        service._records[stale.id] = service._records[stale.id].model_copy(
            update={"updated_at": old})
        service._records[labeled.id] = service._records[labeled.id].model_copy(
            update={"updated_at": old})

        removed = service.expire_unlabeled(60)
        assert stale.id in removed
        assert service.get(stale.id) is None      # stale unlabeled forgotten
        assert service.get(fresh.id) is not None   # recent unlabeled kept
        assert service.get(labeled.id) is not None  # named identity never expires

    def test_expire_unlabeled_zero_age_is_noop(self):
        service, _ = _make_service()
        service.add_unlabeled(_vec(13))
        assert service.expire_unlabeled(0) == []

    def test_label_promotes_unlabeled(self):
        service, _ = _make_service()
        u = service.add_unlabeled(_vec(12))
        promoted = service.label(u.id, "Bob")
        assert promoted is not None
        assert promoted.labeled is True
        assert promoted.enabled is True
        assert promoted.name == "Bob"

    def test_label_blank_rejected(self):
        service, _ = _make_service()
        u = service.add_unlabeled(_vec(13))
        assert service.label(u.id, "   ") is None

    def test_rename(self):
        service, _ = _make_service()
        rec = service.enroll("Alice", _vec(14))
        assert service.rename(rec.id, "Alicia")
        assert service.get(rec.id).name == "Alicia"

    def test_set_enabled(self):
        service, _ = _make_service()
        rec = service.enroll("Alice", _vec(15))
        assert service.set_enabled(rec.id, False)
        assert service.get(rec.id).enabled is False

    def test_delete(self):
        service, _ = _make_service()
        rec = service.enroll("Alice", _vec(16))
        assert service.delete(rec.id)
        assert service.get(rec.id) is None

    def test_version_bumps_on_mutation(self):
        service, _ = _make_service()
        v0 = service.version
        service.enroll("Alice", _vec(17))
        assert service.version > v0

    def test_merge_folds_embeddings_and_thumbnail(self):
        service, _ = _make_service()
        keep = service.enroll("Alice", _vec(20), thumbnail=None)
        drop = service.enroll("Alice2", _vec(21), thumbnail=b"thumb")
        merged = service.merge(keep.id, drop.id)
        assert merged is not None
        assert merged.id == keep.id
        # keep now matches drop's old embedding too
        assert service.best_score(keep.id, embedding_from_bytes(drop.embeddings[0])) \
            == pytest.approx(1.0, abs=1e-5)
        # drop's thumbnail inherited (keep had none)
        assert merged.thumbnail == b"thumb"
        # drop is gone
        assert service.get(drop.id) is None

    def test_merge_same_id_noop(self):
        service, _ = _make_service()
        keep = service.enroll("Alice", _vec(22))
        assert service.merge(keep.id, keep.id) is None

    def test_remove_thumbnail_drops_only_that_photo(self):
        service, _ = _make_service()
        rec = service.enroll("Alice", _vec(23))
        # Accrue three candidate photos (thumbnail set the profile to the first).
        for n in range(3):
            service.add_embedding(rec.id, _vec(24 + n), thumbnail=f"p{n}".encode())
        cur = service.get(rec.id)
        assert cur.thumbnails == [b"p0", b"p1", b"p2"]
        assert cur.thumbnail == b"p0"            # first accrued photo is profile
        # Drop the middle (non-profile) photo: only it goes, profile unchanged.
        assert service.remove_thumbnail(rec.id, 1) is True
        cur = service.get(rec.id)
        assert cur.thumbnails == [b"p0", b"p2"]
        assert cur.thumbnail == b"p0"

    def test_remove_thumbnail_repairs_profile_to_next(self):
        service, _ = _make_service()
        rec = service.enroll("Alice", _vec(25))
        for n in range(2):
            service.add_embedding(rec.id, _vec(26 + n), thumbnail=f"q{n}".encode())
        assert service.get(rec.id).thumbnail == b"q0"
        # Removing the profile photo repairs it to the next remaining shot.
        assert service.remove_thumbnail(rec.id, 0) is True
        cur = service.get(rec.id)
        assert cur.thumbnails == [b"q1"]
        assert cur.thumbnail == b"q1"

    def test_remove_last_thumbnail_clears_profile(self):
        service, _ = _make_service()
        rec = service.enroll("Alice", _vec(27))
        service.add_embedding(rec.id, _vec(28), thumbnail=b"only")
        assert service.remove_thumbnail(rec.id, 0) is True
        cur = service.get(rec.id)
        assert cur.thumbnails == []
        assert cur.thumbnail is None

    def test_remove_thumbnail_leaves_embeddings_intact(self):
        # Thumbnails and embeddings are NOT index-aligned — pruning a photo must
        # not drop any recognition template.
        service, _ = _make_service()
        rec = service.enroll("Alice", _vec(29))            # 1 embedding
        service.add_embedding(rec.id, _vec(30), thumbnail=b"a")  # +1 embedding, +photo
        before = len(service.get(rec.id).embeddings)
        assert service.remove_thumbnail(rec.id, 0) is True
        assert len(service.get(rec.id).embeddings) == before

    def test_remove_thumbnail_out_of_range_returns_false(self):
        service, _ = _make_service()
        rec = service.enroll("Alice", _vec(31))
        assert service.remove_thumbnail(rec.id, 0) is False        # no photos
        assert service.remove_thumbnail("nope", 0) is False         # no identity

    def test_remove_thumbnail_persists_for_labeled(self, tmp_path):
        service, store = _make_service(tmp_path)
        rec = service.enroll("Alice", _vec(32))
        for n in range(2):
            service.add_embedding(rec.id, _vec(33 + n), thumbnail=f"r{n}".encode())
        assert service.remove_thumbnail(rec.id, 0) is True
        loaded = {i.id: i for i in store.load_identities()}[rec.id]
        assert loaded.thumbnails == [b"r1"]
        assert loaded.thumbnail == b"r1"

    def test_store_delete_identity_photo_repacks(self, tmp_path):
        # The explicit single-photo store path: removing a row re-packs the
        # remaining photos' idx so they stay contiguous.
        from autoptz.config.models import IdentityRecord
        store = _make_store(tmp_path)
        rec = IdentityRecord(name="Alice", thumbnails=[b"a", b"b", b"c"])
        store.save_identity(rec)
        assert store.delete_identity_photo(rec.id, 1) is True
        reloaded = {i.id: i for i in store.load_identities()}[rec.id]
        assert reloaded.thumbnails == [b"a", b"c"]
        assert store.delete_identity_photo(rec.id, 5) is False   # out of range


# ── retention policy: labeled-only persisted ─────────────────────────────────


class TestRetentionPolicy:
    def test_labeled_persists_unlabeled_does_not(self, tmp_path):
        service, store = _make_service(tmp_path)
        service.enroll("Alice", _vec(30))          # labeled → persisted
        service.add_unlabeled(_vec(31))            # unlabeled → memory only
        persisted = store.load_identities()
        assert len(persisted) == 1
        assert persisted[0].name == "Alice"

    def test_label_promotes_into_db(self, tmp_path):
        service, store = _make_service(tmp_path)
        u = service.add_unlabeled(_vec(32))
        assert store.load_identities() == []
        service.label(u.id, "Carol")
        persisted = store.load_identities()
        assert [i.name for i in persisted] == ["Carol"]

    def test_unlabeled_vanish_on_reload(self, tmp_path):
        service, store = _make_service(tmp_path)
        service.enroll("Alice", _vec(33))
        service.add_unlabeled(_vec(34))
        # Simulate "restart": a fresh service over the same store.
        from autoptz.engine.identity.service import IdentityService
        fresh = IdentityService(store)
        names = sorted(r.name for r in fresh.all_identities())
        assert names == ["Alice"]   # unlabeled "Person N" is gone

    def test_enabled_roundtrips_through_store(self, tmp_path):
        service, store = _make_service(tmp_path)
        rec = service.enroll("Alice", _vec(35))
        service.set_enabled(rec.id, False)
        loaded = {i.id: i for i in store.load_identities()}
        assert loaded[rec.id].enabled is False


# ── BodyReID (OSNet via boxmot) ────────────────────────────────────────────────


class _FakeReIDBackend:
    """Duck-typed boxmot ReID backend: maps each box to a fixed embedding."""

    def __init__(self, embeddings):
        self._embeddings = np.asarray(embeddings, dtype=np.float32)

    def get_features(self, xyxys, frame):  # noqa: ARG002
        n = len(xyxys)
        return self._embeddings[:n]


class TestBodyReID:
    def test_absent_backend_is_graceful(self):
        from autoptz.engine.pipeline.reid import BodyReID
        r = BodyReID(weights=Path("does-not-exist.pt"))
        # boxmot present but weights/network missing → disabled, no raise
        assert r.embed([(0, 0, 10, 10)], np.zeros((20, 20, 3), np.uint8)).size == 0

    def test_recover_picks_best_match_with_hysteresis(self):
        from autoptz.engine.pipeline.reid import BodyReID
        target = _vec(40)
        interloper = _vec(41)
        backend = _FakeReIDBackend([interloper, target])
        r = BodyReID(_backend=backend, threshold_hi=0.7, threshold_lo=0.4)
        r.set_target(target)
        feats = r.embed([(0, 0, 5, 5), (5, 5, 9, 9)],
                        np.zeros((10, 10, 3), np.uint8))
        result = r.recover(feats)
        assert result.matched is True
        assert result.best_index == 1   # the target crop, not the interloper

    def test_recover_no_template_returns_unmatched(self):
        from autoptz.engine.pipeline.reid import BodyReID, ReIDResult
        r = BodyReID(_backend=_FakeReIDBackend([_vec(42)]))
        res = r.recover(np.atleast_2d(_vec(42)))
        assert isinstance(res, ReIDResult)
        assert res.matched is False


# ── IdentityListModel FROZEN roles ─────────────────────────────────────────────


class TestIdentityListModelRoles:
    def test_role_names(self):
        from autoptz.ui.engine_client import IdentityListModel
        m = IdentityListModel()
        names = {bytes(v).decode() for v in m.roleNames().values()}
        assert {"identityId", "identityName", "thumbnail", "enabled", "labeled"} <= names

    def test_thumbnail_role_is_base64_data_uri(self):
        from autoptz.ui.engine_client import IdentityListModel
        png = b"\x89PNG\r\n\x1a\n_fake_png_bytes"
        rec = IdentityRecord(name="Alice", thumbnail=png)
        m = IdentityListModel()
        m.add_identity(rec)
        idx = m.index(0)
        uri = m.data(idx, IdentityListModel.ThumbnailRole)
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",", 1)[1])
        assert decoded == png

    def test_thumbnail_empty_when_absent(self):
        from autoptz.ui.engine_client import IdentityListModel
        m = IdentityListModel()
        m.add_identity(IdentityRecord(name="Alice"))
        assert m.data(m.index(0), IdentityListModel.ThumbnailRole) == ""

    def test_enabled_and_labeled_roles(self):
        from autoptz.ui.engine_client import IdentityListModel
        m = IdentityListModel()
        m.add_identity(IdentityRecord(name="P1", enabled=False, labeled=False))
        idx = m.index(0)
        assert m.data(idx, IdentityListModel.EnabledRole) is False
        assert m.data(idx, IdentityListModel.LabeledRole) is False

    def test_add_identity_upserts_by_id(self):
        from autoptz.ui.engine_client import IdentityListModel
        m = IdentityListModel()
        rec = IdentityRecord(name="Person 1", enabled=False, labeled=False)
        m.add_identity(rec)
        # re-push same id with a thumbnail → updates in place, no new row
        m.add_identity(rec.model_copy(update={"thumbnail": b"x"}))
        assert m.rowCount() == 1
        assert m.data(m.index(0), IdentityListModel.ThumbnailRole) != ""


# ── EngineClient identity slots (FROZEN API) ──────────────────────────────────


def _make_client(tmp_path=None):
    from autoptz.ui.engine_client import EngineClient
    if tmp_path:
        store = _make_store(tmp_path)
        return EngineClient(store=store), store
    return EngineClient(), None


@pytest.fixture(scope="module")
def qapp():
    import sys

    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    yield app


class TestEngineClientIdentityAPI:
    def test_has_frozen_slots(self, qapp):
        client, _ = _make_client()
        for name in ("setTargetIdentity", "labelIdentity",
                     "mergeIdentities", "setIdentityEnabled",
                     "enrollIdentity", "renameIdentity", "deleteIdentity"):
            assert hasattr(client, name), name

    def test_push_identity_adds_to_model(self, qapp):
        client, _ = _make_client()
        rec = IdentityRecord(name="Person 1", enabled=False, labeled=False,
                             thumbnail=b"png")
        client.push_identity(rec)
        assert client._identity_model.rowCount() == 1
        m = client._identity_model
        assert m.data(m.index(0), m.ThumbnailRole).startswith("data:image/png;base64,")

    def test_label_identity_promotes(self, qapp, tmp_path):
        client, store = _make_client(tmp_path)
        rec = IdentityRecord(name="Person 1", enabled=False, labeled=False)
        client.push_identity(rec)
        client.labelIdentity(rec.id, "Dana")
        updated = client._identity_model.get(rec.id)
        assert updated.name == "Dana"
        assert updated.labeled is True
        assert updated.enabled is True
        # labeling persists (labeled-only retention)
        assert any(i.name == "Dana" for i in store.load_identities())

    def test_set_identity_enabled(self, qapp):
        client, _ = _make_client()
        cid = client.addCamera("usb://0", "Cam")
        client.enrollIdentity(cid, "Alice", 1)
        iid = client._identity_model.get_all()[0].id
        client.setIdentityEnabled(iid, False)
        assert client._identity_model.get(iid).enabled is False

    def test_merge_identities_removes_drop(self, qapp):
        client, _ = _make_client()
        a = IdentityRecord(name="A", embeddings=[embedding_to_bytes(_vec(50))])
        b = IdentityRecord(name="B", embeddings=[embedding_to_bytes(_vec(51))],
                           thumbnail=b"t")
        client.push_identity(a)
        client.push_identity(b)
        client.mergeIdentities(a.id, b.id)
        assert client._identity_model.get(b.id) is None
        assert client._identity_model.get(a.id) is not None

    def test_set_target_identity_enqueues_command(self, qapp):
        from autoptz.engine.runtime.messages import CmdKind
        client, _ = _make_client()
        cid = client.addCamera("usb://0", "Cam")
        client.drain_commands()
        client.setTargetIdentity(cid, "id-123")
        cmds = client.drain_commands()
        target = [c for c in cmds if c.kind == CmdKind.SET_TARGET_IDENTITY]
        assert len(target) == 1
        assert target[0].identity_id == "id-123"

    def test_set_target_identity_updates_config(self, qapp):
        client, _ = _make_client()
        cid = client.addCamera("usb://0", "Cam")
        client.setTargetIdentity(cid, "id-xyz")
        rec = client.get_camera(cid)
        assert rec.camera_config.target.identity_id == "id-xyz"
        assert rec.camera_config.target.mode == "identity"

    def test_set_target_identity_empty_clears(self, qapp):
        client, _ = _make_client()
        cid = client.addCamera("usb://0", "Cam")
        client.setTargetIdentity(cid, "id-xyz")
        client.setTargetIdentity(cid, "")
        rec = client.get_camera(cid)
        assert rec.camera_config.target.identity_id is None


# ── CameraWorker auto-harvest + target-by-identity ────────────────────────────


def _worker_config():
    return CameraConfig(
        id="cam-test-1234abcd",
        name="Test",
        source=SourceConfig(type="usb", address="usb://0"),
    )


def _gray(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 255, (480, 640, 3))).astype(np.uint8)


class _StubTrack:
    """Minimal TrackInfo-like object for direct worker-method tests."""

    def __init__(self, track_id, bbox):
        from autoptz.engine.runtime.messages import BBox
        self.track_id = track_id
        self.bbox = BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
        self.identity = None
        self.confidence = 0.0
        self.is_target = False


class TestWorkerAutoHarvest:
    def _worker(self, service, recognizer, harvested):
        from autoptz.engine.camera_worker import CameraWorker, _FaceStack
        w = CameraWorker(
            "cam-test-1234abcd",
            _worker_config(),
            on_telemetry=lambda m: None,
            on_identity=harvested.append,
            face_stack=_FaceStack(recognizer=recognizer, service=service),
        )
        w._face = _FaceStack(recognizer=recognizer, service=service)
        return w

    def test_unmatched_face_creates_unlabeled_identity_with_thumbnail(self):
        service, _ = _make_service()
        # A face that matches nothing in the (empty) gallery.
        app = _FakeApp([_FakeFace((280, 180, 380, 320), _vec(60))])
        rec = FaceRecognizer(_app=app, match_threshold=0.5)
        harvested = []
        w = self._worker(service, rec, harvested)
        track = _StubTrack(1, (250, 150, 400, 460))
        frame = _gray(1)
        w._maybe_identify(frame, [track], now=100.0)
        # one unlabeled identity harvested + pushed
        assert len(harvested) == 1
        assert harvested[0].labeled is False
        assert harvested[0].thumbnail is not None   # cv2 present → PNG crop
        assert service.unlabeled_identities()

    def test_matched_face_annotates_track_identity(self):
        service, _ = _make_service()
        emb = _vec(61)
        ident = service.enroll("Alice", emb)
        app = _FakeApp([_FakeFace((280, 180, 380, 320), emb)])
        rec = FaceRecognizer(_app=app, match_threshold=0.4)
        w = self._worker(service, rec, [])
        track = _StubTrack(7, (250, 150, 400, 460))
        w._maybe_identify(_gray(2), [track], now=200.0)
        assert w._track_identity.get(7) is not None
        assert w._track_identity[7][0] == ident.id

    def test_face_overlay_expires_and_clears_when_track_disappears(self):
        service, _ = _make_service()
        emb = _vec(612)
        service.enroll("Alice", emb)
        app = _FakeApp([_FakeFace((280, 180, 380, 320), emb)])
        rec = FaceRecognizer(_app=app, match_threshold=0.4)
        w = self._worker(service, rec, [])
        track = _StubTrack(7, (250, 150, 400, 460))
        w._maybe_identify(_gray(21), [track], now=210.0)
        assert len(w._last_faces) == 1

        assert w._fresh_faces_for_telemetry([]) == []
        w._maybe_identify(_gray(21), [track], now=211.0)
        assert len(w._last_faces) == 1
        w._last_faces_t = time.monotonic() - 1.0
        assert w._fresh_faces_for_telemetry([track]) == []

    def test_face_overlay_publishes_once_per_inference_frame(self):
        service, _ = _make_service()
        emb = _vec(613)
        service.enroll("Alice", emb)
        app = _FakeApp([_FakeFace((280, 180, 380, 320), emb)])
        rec = FaceRecognizer(_app=app, match_threshold=0.4)
        w = self._worker(service, rec, [])
        track = _StubTrack(7, (250, 150, 400, 460))
        w._current_inference_frame_id = 12
        w._last_tracks_frame_id = 12
        w._maybe_identify(_gray(22), [track], now=time.monotonic())

        assert len(w._fresh_faces_for_telemetry([track])) == 1
        assert w._fresh_faces_for_telemetry([track]) == []

    def test_pending_enroll_uses_clicked_face_not_first_face(self):
        service, _ = _make_service()
        target_emb = _vec(610)
        wrong_emb = _vec(611)
        ident = service.enroll("Alice", None, identity_id="id-click")
        app = _FakeApp([
            _FakeFace((260, 170, 320, 240), wrong_emb),
            _FakeFace((350, 250, 430, 340), target_emb),
        ])
        rec = FaceRecognizer(_app=app, match_threshold=0.99)
        w = self._worker(service, rec, [])
        track = _StubTrack(7, (240, 140, 450, 380))
        # Click near the second face; without point-specific selection the first
        # face in detector order would be enrolled instead.
        w._apply_command("enroll_track", (7, ident.id, "Alice", (390 / 640, 295 / 480)))
        w._maybe_identify(_gray(22), [track], now=220.0)
        assert service.best_score(ident.id, target_emb) == pytest.approx(1.0, abs=1e-5)
        assert service.best_score(ident.id, wrong_emb) < 0.5

    def test_target_by_identity_locks_matched_track(self):
        service, _ = _make_service()
        emb = _vec(62)
        ident = service.enroll("Alice", emb)
        app = _FakeApp([_FakeFace((280, 180, 380, 320), emb)])
        rec = FaceRecognizer(_app=app, match_threshold=0.4)
        w = self._worker(service, rec, [])
        # operator asked to follow Alice by identity
        w.set_target_identity(ident.id)
        w._drain_commands()
        track = _StubTrack(9, (250, 150, 400, 460))
        w._maybe_identify(_gray(3), [track], now=300.0)
        # the single target is now locked to the track bound to Alice
        assert w._target_track_id == 9

    def test_disabled_recognizer_is_noop(self):
        service, _ = _make_service()
        rec = FaceRecognizer()   # insightface absent → unavailable
        harvested = []
        w = self._worker(service, rec, harvested)
        track = _StubTrack(1, (0, 0, 100, 200))
        w._maybe_identify(_gray(4), [track], now=400.0)
        assert harvested == []
        assert w._track_identity == {}
