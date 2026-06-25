"""Center Stage group-framing wiring in CameraWorker (digital crop path only).

Covers the D1 group-vs-explicit-lock rule and the union helper that
``_current_digital_target`` uses:

  * group framing OFF → single-target behaviour (unchanged);
  * group framing ON, no explicit lock, several people → the UNION of their
    boxes (wider than any single one);
  * group framing ON, one person → just that person's box;
  * an explicitly locked target (by id or identity) ALWAYS wins, even with
    group framing on.
"""

from __future__ import annotations

from autoptz.config.models import CameraConfig, SourceConfig, TrackingConfig
from autoptz.engine.camera_worker import CameraWorker, _TargetLockState
from autoptz.engine.runtime.messages import BBox, TrackInfo


def _make_worker(*, group_framing: bool = False, identity_id: str | None = None) -> CameraWorker:
    config = CameraConfig(
        id="test-cam-group12345",
        name="Test",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(group_framing=group_framing),
    )
    if identity_id is not None:
        config = config.model_copy(
            update={"target": config.target.model_copy(update={"identity_id": identity_id})}
        )
    return CameraWorker("test-cam-group12345", config, on_telemetry=lambda m: None)


def _track(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackInfo:
    return TrackInfo(track_id=track_id, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


class TestGroupUnionHelper:
    """``_group_union_bbox`` is pure — test it directly."""

    def test_none_for_no_people(self):
        assert CameraWorker._group_union_bbox([]) is None

    def test_single_person_is_that_box(self):
        tracks = [_track(1, 300, 400, 500, 670)]
        assert CameraWorker._group_union_bbox(tracks) == (300.0, 400.0, 500.0, 670.0)

    def test_union_of_several_people(self):
        tracks = [_track(1, 300, 400, 500, 670), _track(2, 1400, 380, 1600, 690)]
        u = CameraWorker._group_union_bbox(tracks)
        assert u == (300.0, 380.0, 1600.0, 690.0)

    def test_lost_people_are_ignored(self):
        live = _track(1, 300, 400, 500, 670)
        lost = _track(2, 1400, 380, 1600, 690)
        lost.lost = True
        u = CameraWorker._group_union_bbox([live, lost])
        # The lost person drops out → union is just the live one.
        assert u == (300.0, 400.0, 500.0, 670.0)


class TestGroupFramingOff:
    def test_off_with_no_lock_returns_none(self):
        w = _make_worker(group_framing=False)
        w._last_tracks = [_track(1, 300, 400, 500, 670), _track(2, 1400, 380, 1600, 690)]
        # No explicit lock and group framing off → no digital target (full frame).
        assert w._current_digital_target() is None


class TestGroupFramingOn:
    def test_union_when_no_lock_and_multiple_people(self):
        w = _make_worker(group_framing=True)
        w._target_track_id = None
        w._last_tracks = [_track(1, 300, 400, 500, 670), _track(2, 1400, 380, 1600, 690)]
        target = w._current_digital_target()
        assert target == (300.0, 380.0, 1600.0, 690.0)
        # Wider than either single person box.
        assert (target[2] - target[0]) > (500 - 300)
        assert (target[2] - target[0]) > (1600 - 1400)

    def test_single_person_is_single_box(self):
        w = _make_worker(group_framing=True)
        w._target_track_id = None
        w._last_tracks = [_track(7, 800, 400, 1000, 700)]
        assert w._current_digital_target() == (800.0, 400.0, 1000.0, 700.0)


class TestExplicitLockWins:
    def test_locked_track_id_beats_group(self):
        # Group framing ON, but an explicit track id is locked → follow that one
        # person, NOT the union.
        w = _make_worker(group_framing=True)
        w._target_track_id = 2
        w._last_tracks = [_track(1, 300, 400, 500, 670), _track(2, 1400, 380, 1600, 690)]
        target = w._current_digital_target()
        assert target == (1400.0, 380.0, 1600.0, 690.0)  # just the locked person

    def test_locked_identity_beats_group_via_trusted_bbox(self):
        # Identity locked but its live track has churned out of _last_tracks: the
        # explicit-lock path falls back to the trusted bbox, NOT the group union.
        w = _make_worker(group_framing=True, identity_id="person-abc")
        w._target_track_id = None
        w._target_lock = _TargetLockState()
        w._target_lock.trusted_bbox = BBox(x1=900, y1=420, x2=1100, y2=720)
        w._last_tracks = [_track(1, 300, 400, 500, 670), _track(2, 1400, 380, 1600, 690)]
        target = w._current_digital_target()
        assert target == (900.0, 420.0, 1100.0, 720.0)  # the locked identity, not the union
