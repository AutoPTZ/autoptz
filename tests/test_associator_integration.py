"""Integration tests for the flag-gated TargetAssociator in CameraWorker.

Tests confirm:
  1. Flag OFF: ``_associator.decide`` is NEVER called; existing heuristic path runs.
  2. Flag ON — clear single target: decision is "keep", no ID-switch.
  3. Flag ON — two look-alike crossing tracks (similar motion, no distinguishing
     cue): decision is "ambiguous", no ID-switch.
  4. Flag ON — a track carrying the target identity clearly leads: decision is
     "switch", target is rebound.
  5. ``_build_candidate_cues``: cue availability rules respected.
"""

from __future__ import annotations

import pytest

from autoptz.config.models import CameraConfig, SourceConfig, TrackingConfig
from autoptz.engine.runtime.messages import BBox, TrackInfo

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_config(*, use_target_associator: bool = False) -> CameraConfig:
    return CameraConfig(
        id="test-cam-assoc1234",
        name="Test",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(use_target_associator=use_target_associator),
    )


def _make_worker(config: CameraConfig):
    from autoptz.engine.camera_worker import CameraWorker

    return CameraWorker(
        "test-cam-assoc1234",
        config,
        on_telemetry=lambda m: None,
    )


def _track(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackInfo:
    return TrackInfo(
        track_id=track_id,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _bbox(x1: float, y1: float, x2: float, y2: float) -> BBox:
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


# ── TC-1: Flag OFF — associator is NEVER consulted ─────────────────────────────


class TestFlagOff:
    def test_decide_never_called_when_flag_off(self, monkeypatch):
        """With use_target_associator=False, _associator.decide must never be called."""
        config = _make_config(use_target_associator=False)
        worker = _make_worker(config)

        # Inject a target
        worker._target_track_id = 1
        worker._target_lock.trusted_track_id = 1
        worker._target_lock.trusted_bbox = _bbox(100, 100, 200, 300)
        worker._target_lock.trusted_t = 1000.0
        worker._target_lock.status = "locked"

        def _explode(candidates, current_target_id):  # noqa: ARG001
            raise AssertionError("_associator.decide must NOT be called when flag is OFF")

        monkeypatch.setattr(worker._associator, "decide", _explode)

        track = _track(1, 100, 100, 200, 300)
        # Should NOT raise; the heuristic path runs without calling decide.
        worker._apply_target_lock([track], None, 1001.0)

    def test_existing_lock_tests_still_pass_with_flag_off(self):
        """Smoke-check: heuristic path still stores trusted target normally."""
        config = _make_config(use_target_associator=False)
        worker = _make_worker(config)

        worker._target_track_id = 5
        track = _track(5, 50, 50, 150, 200)
        worker._apply_target_lock([track], None, 2000.0)

        # First call seeds trusted_track_id (trusted_bbox was None).
        assert worker._target_lock.trusted_track_id == 5
        assert worker._target_lock.trusted_bbox is not None


# ── TC-2: Flag ON — single clear target → keep ─────────────────────────────────


class TestFlagOnKeep:
    def test_single_clear_target_kept(self, monkeypatch):
        """Only track matches the current target with good IOU → keep (no switch)."""
        config = _make_config(use_target_associator=True)
        worker = _make_worker(config)

        worker._target_track_id = 1
        worker._target_lock.trusted_track_id = 1
        worker._target_lock.trusted_bbox = _bbox(100, 100, 200, 300)
        worker._target_lock.trusted_t = 5000.0
        worker._target_lock.status = "locked"

        # Track 1 is at essentially the same position → high IOU → keep.
        track = _track(1, 102, 98, 202, 302)

        decided = []

        real_decide = worker._associator.decide

        def spy_decide(candidates, current_target_id):
            d = real_decide(candidates, current_target_id)
            decided.append(d)
            return d

        monkeypatch.setattr(worker._associator, "decide", spy_decide)

        worker._apply_target_lock([track], None, 5001.0)

        assert len(decided) == 1
        assert decided[0].action == "keep"
        # Target ID must not have changed.
        assert worker._target_track_id == 1


# ── TC-3: Flag ON — two look-alike tracks → ambiguous ─────────────────────────


class TestFlagOnAmbiguous:
    def test_two_lookalike_tracks_ambiguous_no_switch(self):
        """Real associator returns 'ambiguous' when two look-alike tracks are close.

        Geometry (motion-only — no identity/appearance/pose cues available):
          trusted_bbox (ref) = (100, 100, 200, 300)  [100×200 box, area=20 000]

          track2 (ID=2, challenger)  at (102, 100, 202, 300):
              intersection = 98×200 = 19 600, union = 20 400
              IOU ≈ 0.9608  ← best candidate

          track1 (ID=1, current target, runner-up) at (105, 100, 205, 300):
              intersection = 95×200 = 19 000, union = 21 000
              IOU ≈ 0.9048

          margin = 0.9608 − 0.9048 ≈ 0.056 < ambiguous_margin (0.08)
          → Rule 2b fires: ambiguous (look-alike hold).

        With only the motion cue available, association_confidence collapses to
        the raw IOU (single-cue weighted average = IOU).
        """
        config = _make_config(use_target_associator=True)
        worker = _make_worker(config)

        # Current target is track 1; trusted_bbox is the reference for IOU.
        worker._target_track_id = 1
        worker._target_lock.trusted_track_id = 1
        worker._target_lock.trusted_bbox = _bbox(100, 100, 200, 300)
        worker._target_lock.trusted_t = 6000.0
        worker._target_lock.status = "locked"

        # Challenger (track 2) leads the current target (track 1) by ~0.056 IOU —
        # below ambiguous_margin (0.08), so the REAL associator returns "ambiguous".
        track1 = _track(1, 105, 100, 205, 300)  # current target; IOU≈0.9048
        track2 = _track(2, 102, 100, 202, 300)  # challenger;     IOU≈0.9608

        # Sanity-check: the real decide (not monkeypatched) returns "ambiguous".
        from autoptz.engine.pipeline.associator import CandidateCues, Cue

        ref = worker._target_lock.trusted_bbox
        cue1 = CandidateCues(
            track_id=1,
            motion=Cue(worker._bbox_iou(ref, track1.bbox), available=True),
            appearance=Cue(0.0, available=False),
            pose=Cue(0.0, available=False),
            identity=Cue(0.0, available=False),
        )
        cue2 = CandidateCues(
            track_id=2,
            motion=Cue(worker._bbox_iou(ref, track2.bbox), available=True),
            appearance=Cue(0.0, available=False),
            pose=Cue(0.0, available=False),
            identity=Cue(0.0, available=False),
        )
        direct_decision = worker._associator.decide([cue1, cue2], current_target_id=1)
        assert direct_decision.action == "ambiguous", (
            f"Expected real associator to return 'ambiguous', got {direct_decision!r}"
        )

        prev_id = worker._target_track_id
        worker._apply_target_lock([track1, track2], None, 6001.0)

        # No ID-switch — target must remain track 1.
        assert worker._target_track_id == prev_id
        # Lock must be frozen as ambiguous (real associator drove this, not a mock).
        assert worker._target_lock.status == "ambiguous"


# ── TC-4: Flag ON — identity cue leads → switch ───────────────────────────────


class TestFlagOnSwitch:
    def test_identity_cue_drives_switch(self, monkeypatch):
        """A track carrying the target identity clearly dominates → switch."""
        config = _make_config(use_target_associator=True)
        worker = _make_worker(config)

        # Current target is track 1.
        worker._target_track_id = 1
        worker._target_identity_id = "identity-alice"
        worker._target_lock.trusted_track_id = 1
        worker._target_lock.trusted_bbox = _bbox(100, 100, 200, 300)
        worker._target_lock.trusted_t = 7000.0
        worker._target_lock.status = "locked"

        # Track 2 carries the target identity with high confidence.
        worker._track_identity[2] = ("identity-alice", "Alice", 0.95)

        track1 = _track(1, 100, 100, 200, 300)
        track2 = _track(2, 300, 100, 400, 300)  # different position

        # Force the associator to return "switch" to track 2.
        from autoptz.engine.pipeline.associator import Decision

        forced_switch = Decision(
            action="switch",
            track_id=2,
            confidence=0.82,
            margin=0.25,
        )
        monkeypatch.setattr(
            worker._associator,
            "decide",
            lambda candidates, current_target_id: forced_switch,
        )

        worker._apply_target_lock([track1, track2], None, 7001.0)

        # Target must have switched to track 2.
        assert worker._target_track_id == 2
        # Lock must be fully committed after the switch.
        assert worker._target_lock.trusted_track_id == 2
        assert worker._target_lock.status == "locked"
        assert worker._target_lock.trusted_bbox is not None


# ── TC-5: _build_candidate_cues availability rules ────────────────────────────


class TestBuildCandidateCues:
    def _make_worker_with_ref(
        self,
        *,
        trusted_bbox: BBox | None,
        target_track_id: int | None,
        target_identity_id: str | None = None,
    ):
        config = _make_config(use_target_associator=True)
        worker = _make_worker(config)
        worker._target_track_id = target_track_id
        worker._target_identity_id = target_identity_id
        worker._target_lock.trusted_bbox = trusted_bbox
        return worker

    def test_motion_available_when_ref_bbox_exists(self):
        """motion cue is available when trusted_bbox and current target are set."""
        worker = self._make_worker_with_ref(
            trusted_bbox=_bbox(100, 100, 200, 300),
            target_track_id=1,
        )
        track = _track(1, 102, 98, 202, 302)
        cues = worker._build_candidate_cues([track])
        assert len(cues) == 1
        assert cues[0].motion.available is True
        assert 0.0 < cues[0].motion.value <= 1.0  # high IOU

    def test_motion_unavailable_when_no_ref_bbox(self):
        """motion cue is unavailable when trusted_bbox is None (no prior target)."""
        worker = self._make_worker_with_ref(
            trusted_bbox=None,
            target_track_id=None,
        )
        track = _track(1, 100, 100, 200, 300)
        cues = worker._build_candidate_cues([track])
        assert len(cues) == 1
        assert cues[0].motion.available is False

    def test_identity_available_only_for_matching_track(self):
        """identity cue available only for a track carrying the target identity."""
        worker = self._make_worker_with_ref(
            trusted_bbox=_bbox(100, 100, 200, 300),
            target_track_id=1,
            target_identity_id="identity-alice",
        )
        # Track 2 carries the target identity; track 3 does not.
        worker._track_identity[2] = ("identity-alice", "Alice", 0.90)
        worker._track_identity[3] = ("identity-bob", "Bob", 0.85)
        track2 = _track(2, 200, 100, 300, 300)
        track3 = _track(3, 400, 100, 500, 300)
        cues_by_id = {c.track_id: c for c in worker._build_candidate_cues([track2, track3])}

        assert cues_by_id[2].identity.available is True
        assert cues_by_id[2].identity.value == pytest.approx(0.90)
        assert cues_by_id[3].identity.available is False

    def test_identity_unavailable_when_no_target_identity(self):
        """No target_identity_id set → identity cue always unavailable."""
        worker = self._make_worker_with_ref(
            trusted_bbox=_bbox(100, 100, 200, 300),
            target_track_id=1,
            target_identity_id=None,
        )
        worker._track_identity[1] = ("identity-alice", "Alice", 0.95)
        track = _track(1, 100, 100, 200, 300)
        cues = worker._build_candidate_cues([track])
        assert cues[0].identity.available is False

    def test_pose_and_appearance_always_unavailable(self):
        """pose and appearance cues are never available from _build_candidate_cues."""
        worker = self._make_worker_with_ref(
            trusted_bbox=_bbox(100, 100, 200, 300),
            target_track_id=1,
        )
        track = _track(1, 100, 100, 200, 300)
        cues = worker._build_candidate_cues([track])
        assert len(cues) == 1
        assert cues[0].pose.available is False
        assert cues[0].appearance.available is False

    def test_lost_tracks_excluded(self):
        """Tracks with lost=True are excluded from candidate cues."""
        worker = self._make_worker_with_ref(
            trusted_bbox=_bbox(100, 100, 200, 300),
            target_track_id=1,
        )
        live_track = _track(1, 100, 100, 200, 300)
        lost_track = _track(2, 300, 100, 400, 300)
        lost_track.lost = True
        cues = worker._build_candidate_cues([live_track, lost_track])
        assert len(cues) == 1
        assert cues[0].track_id == 1


# ── TC-6: Config model — flag field ──────────────────────────────────────────


class TestConfigFlag:
    def test_default_is_false(self):
        """use_target_associator defaults to False in TrackingConfig."""
        tc = TrackingConfig()
        assert tc.use_target_associator is False

    def test_can_be_set_true(self):
        tc = TrackingConfig(use_target_associator=True)
        assert tc.use_target_associator is True

    def test_camera_config_default_false(self):
        cam = CameraConfig(name="Cam")
        assert cam.tracking.use_target_associator is False
