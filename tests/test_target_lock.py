"""Target-lock correctness tests.

Bug D — a target selected by *identity* (name) must never be re-bound to a
different person by the appearance-only ReID recovery path. ReID may hold/refresh
the current track, but re-acquiring a *named* target requires a face-identity
match, not body appearance — otherwise the camera drifts onto the next
visually-similar person, then the next unknown track.
"""

from __future__ import annotations

import numpy as np

from autoptz.config.models import CameraConfig, SourceConfig, TrackingConfig
from autoptz.engine.runtime.messages import BBox, TrackInfo


def _worker(identity_id: str | None = None):
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(
        id="cam-lock-00000001",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(tracking_mode="stable"),
    )
    w = CameraWorker("cam-lock-00000001", cfg, on_telemetry=lambda m: None)
    w._target_identity_id = identity_id
    return w


class TestReidRebindIdentityGate:
    def test_named_target_allows_rebind_only_to_same_identity(self):
        w = _worker(identity_id="A")
        w._track_identity = {5: ("A", "Alice", 0.9), 7: ("B", "Bob", 0.8)}
        assert w._reid_rebind_allows(5) is True  # same identity → allowed
        assert w._reid_rebind_allows(7) is False  # different known person → blocked

    def test_named_target_blocks_rebind_to_unknown_track(self):
        w = _worker(identity_id="A")
        w._track_identity = {}
        # Unknown track (no confirmed identity) → blocked; wait for a face match.
        assert w._reid_rebind_allows(9) is False

    def test_manual_target_allows_appearance_rebind(self):
        w = _worker(identity_id=None)
        w._track_identity = {7: ("B", "Bob", 0.8)}
        # No identity target (clicked/manual): appearance re-bind is the intended behaviour.
        assert w._reid_rebind_allows(7) is True
        assert w._reid_rebind_allows(9) is True


class TestTargetBoxCollapsed:
    """Bug C — a target box that suddenly collapses (occlusion → only legs/partial
    visible) must be flagged so the camera coasts instead of chasing the shrinking
    box down to the last-known partial position. A gradual shrink (the subject
    walking away) must NOT be flagged."""

    def test_first_call_sets_reference_not_collapsed(self):
        w = _worker()
        assert w._target_box_collapsed(0.5) is False
        assert w._target_h_ref == 0.5

    def test_gradual_shrink_not_flagged_and_tracks_reference(self):
        w = _worker()
        w._target_box_collapsed(0.5)
        for h in (0.47, 0.44, 0.41, 0.38):
            assert w._target_box_collapsed(h) is False
        # The healthy-height reference followed the gradual change downward.
        assert w._target_h_ref < 0.5

    def test_sudden_collapse_is_flagged_and_recovers(self):
        w = _worker()
        w._target_box_collapsed(0.5)
        assert w._target_box_collapsed(0.2) is True  # sudden drop below the reference
        # Reference is left intact on collapse, so the subject reappearing at full
        # size is immediately trusted again.
        assert w._target_box_collapsed(0.5) is False


def _track(track_id: int, bbox: BBox, *, confidence: float = 0.9) -> TrackInfo:
    return TrackInfo(track_id=track_id, bbox=bbox, confidence=confidence)


class TestTargetBoxEvidence:
    def test_unusable_ptz_box_rejects_degenerate_and_tiny_boxes(self):
        w = _worker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert w._target_box_usable_for_ptz(_track(1, BBox(x1=10, y1=10, x2=10, y2=20)), frame) is False
        assert w._target_box_usable_for_ptz(_track(1, BBox(x1=10, y1=10, x2=20, y2=20)), frame) is False
        assert w._target_box_usable_for_ptz(_track(1, BBox(x1=10, y1=10, x2=120, y2=140)), frame) is True

    def test_single_bbox_teleport_is_held_until_confirmed(self):
        w = _worker()
        w._target_track_id = 1
        first = _track(1, BBox(x1=100, y1=100, x2=200, y2=320))
        w._apply_target_lock([first], frame=None, now=1.0)
        assert w._target_lock.status == "locked"

        jump = _track(1, BBox(x1=500, y1=100, x2=600, y2=320))
        w._apply_target_lock([jump], frame=None, now=1.1)
        assert w._target_lock.status == "ambiguous"
        assert w._target_lock.reason == "bbox_jump"
        assert jump.lost is True
        assert jump.bbox.x1 == 100

        confirmed = _track(1, BBox(x1=505, y1=102, x2=605, y2=322))
        w._apply_target_lock([confirmed], frame=None, now=1.2)
        assert w._target_lock.status == "locked"
        assert w._target_lock.trusted_bbox is not None
        assert w._target_lock.trusted_bbox.x1 == 505
