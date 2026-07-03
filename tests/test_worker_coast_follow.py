"""R-1 worker coast-follow: the PTZ briefly follows a coasted LOST target.

The tracker (autoptz/engine/pipeline/track.py) already coasts LOST tracks
along a damped pre-loss velocity (see tests/test_track.py::TestPredictiveCoast).
This file covers the WORKER side that makes that coast reach the camera:

  1. ``_maybe_track`` publishes the locked target's coasted LOST track (instead
     of filtering it out like every other LOST track).
  2. ``_drive_ptz_auto`` follows that coasted target while it is still moving
     fast enough to be worth chasing (``|v| >= _COAST_FOLLOW_MIN_V``), and
     falls back to today's hold→coast→search once it has decayed below that.
  3. ``_apply_target_lock`` / ``_append_held_target`` do not synthesize a
     second, frozen copy of the target now that the coasted one is published.

Follows the ``_bare_worker`` harness pattern from test_camera_worker_framing.py
and the fake detect/tracker stack pattern from test_worker_crash_safety.py.
"""

from __future__ import annotations

import math

import numpy as np

from autoptz.config.models import CameraConfig
from autoptz.engine.camera_worker import _COAST_FOLLOW_MIN_V, CameraWorker
from autoptz.engine.pipeline.track import TrackState
from autoptz.engine.runtime.messages import BBox, TrackInfo


def _bare_worker() -> CameraWorker:
    cfg = CameraConfig(id="cam-coast", name="Cam Coast")
    return CameraWorker("cam-coast", cfg, on_telemetry=lambda m: None)


def _frame() -> np.ndarray:
    return np.zeros((720, 1000, 3), dtype=np.uint8)


def _bbox(x1: float, y1: float, x2: float, y2: float) -> BBox:
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _track(track_id: int, bbox: BBox, *, lost: bool = False, vx: float = 0.0, vy: float = 0.0):
    return TrackInfo(track_id=track_id, bbox=bbox, lost=lost, vx=vx, vy=vy)


class _FakeController:
    """Minimal ``_ptz`` stand-in: only the methods ``_drive_ptz_auto`` touches
    before handing off to ``_publish_ptz`` (which the tests monkeypatch)."""

    def set_loop_latency(self, seconds: float) -> None:
        pass


def _worker_with_target(track_id: int = 7):
    w = _bare_worker()
    w._ptz = _FakeController()
    w._tracking_enabled = True
    w._target_track_id = track_id
    w._manual_override_active = lambda now: False  # type: ignore[assignment]
    w._feature = lambda name: True  # type: ignore[assignment]
    return w


# ── (a)/(b)/(d): _drive_ptz_auto coast-follow ──────────────────────────────────


class TestDrivePtzAutoCoastFollow:
    def test_lost_moving_target_drives_ptz_active(self) -> None:
        """(a) lost + moving (|v| >= 1.0) → driven with track_active=True using
        the coasted box error."""
        w = _worker_with_target()
        calls: list[dict] = []
        w._publish_ptz = lambda ctrl, err, vel, height, **kw: calls.append(  # type: ignore[assignment]
            {"err": err, "vel": vel, "height": height, **kw}
        )
        frame = _frame()
        # Coasted box well right-of-center, moving right at 5 px/frame.
        target = _track(7, _bbox(800, 300, 900, 500), lost=True, vx=5.0, vy=0.0)
        w._drive_ptz_auto([target], frame, now=1000.0)

        assert calls, "no PTZ command published for a lost+moving target"
        last = calls[-1]
        assert last["track_active"] is True
        # Coasted bbox is right of center → positive x error.
        assert last["err"][0] > 0.0

    def test_lost_stationary_target_does_not_drive(self) -> None:
        """(b) lost + stationary (|v| < 1.0) → track_active=False (no drive)."""
        w = _worker_with_target()
        calls: list[dict] = []
        w._publish_ptz = lambda ctrl, err, vel, height, **kw: calls.append(  # type: ignore[assignment]
            {"err": err, "vel": vel, "height": height, **kw}
        )
        frame = _frame()
        target = _track(7, _bbox(800, 300, 900, 500), lost=True, vx=0.2, vy=0.1)
        assert math.hypot(0.2, 0.1) < _COAST_FOLLOW_MIN_V
        w._drive_ptz_auto([target], frame, now=1000.0)

        assert calls, "no PTZ command published"
        assert calls[-1]["track_active"] is False

    def test_tracking_feature_disabled_stays_inactive_even_if_moving(self) -> None:
        """(d) ``tracking`` feature off → inactive even with a moving lost target."""
        w = _worker_with_target()
        w._feature = lambda name: False  # type: ignore[assignment]
        calls: list[dict] = []
        w._publish_ptz = lambda ctrl, err, vel, height, **kw: calls.append(  # type: ignore[assignment]
            {"err": err, "vel": vel, "height": height, **kw}
        )
        frame = _frame()
        target = _track(7, _bbox(800, 300, 900, 500), lost=True, vx=5.0, vy=0.0)
        w._drive_ptz_auto([target], frame, now=1000.0)

        assert calls, "no PTZ command published"
        assert calls[-1]["track_active"] is False

    def test_min_velocity_threshold_is_one_pixel_per_frame(self) -> None:
        assert _COAST_FOLLOW_MIN_V == 1.0

    def test_non_target_lost_tracks_still_ignored(self) -> None:
        """No behaviour change for LOST tracks that are NOT the locked target —
        _resolve_target_track must not pick them up."""
        w = _worker_with_target(track_id=7)
        calls: list[dict] = []
        w._publish_ptz = lambda ctrl, err, vel, height, **kw: calls.append(  # type: ignore[assignment]
            {"err": err, "vel": vel, "height": height, **kw}
        )
        frame = _frame()
        other = _track(9, _bbox(100, 100, 200, 200), lost=True, vx=5.0, vy=0.0)
        w._drive_ptz_auto([other], frame, now=1000.0)

        assert calls, "no PTZ command published"
        assert calls[-1]["track_active"] is False


# ── (c): exactly one TrackInfo carries the target id while coasting ───────────


class TestApplyTargetLockNoHeldDuplicate:
    def test_coasted_target_present_no_held_duplicate_appended(self) -> None:
        """(c) With the coasted LOST target already published, _apply_target_lock
        (via _append_held_target) must NOT synthesize a second frozen copy."""
        w = _bare_worker()
        w._target_track_id = 7
        # Prime the lock with a previous trusted sighting so _append_held_target
        # WOULD fire if the target were missing (target is None) -- it must not
        # fire here because the coasted track is present.
        w._target_lock.trusted_track_id = 7
        w._target_lock.trusted_bbox = _bbox(700, 300, 800, 500)
        w._target_lock.trusted_t = 999.999

        tracks = [_track(7, _bbox(800, 300, 900, 500), lost=True, vx=5.0, vy=0.0)]
        w._apply_target_lock(tracks, _frame(), now=1000.0)

        matching = [t for t in tracks if t.track_id == 7]
        assert len(matching) == 1, (
            f"expected exactly one TrackInfo for the target, got {len(matching)}"
        )
        assert matching[0].lost is True


# ── requirement 1: _maybe_track publishes the target's coasted LOST track ─────


class _FakeDetector:
    def detect(self, frame):  # noqa: ANN001, ANN202
        return []


class _FakeTracker:
    """Fake boxmot-wrapper tracker returning canned Track objects."""

    def __init__(self, tracks):  # noqa: ANN001
        self._tracks = tracks

    def update(self, detections, frame, fps=30.0):  # noqa: ANN001, ANN202
        return self._tracks


class _FakeDetectStack:
    def __init__(self, tracks):  # noqa: ANN001
        self.detector = _FakeDetector()
        self.tracker = _FakeTracker(tracks)


class _FakeRawTrack:
    """Stand-in for autoptz.engine.pipeline.track.Track (the tracker's own
    output type), shaped with the attributes _maybe_track's loop reads."""

    def __init__(self, track_id, bbox, state, conf=0.9, velocity=(0.0, 0.0)):  # noqa: ANN001
        self.track_id = track_id
        self.bbox = bbox
        self.state = state
        self.conf = conf
        self.velocity = velocity


def test_maybe_track_publishes_coasted_target_lost_track() -> None:
    """Requirement 1: the loop keeps filtering LOST tracks EXCEPT the locked
    target, which it emits with lost=True, its coasted bbox, and vx/vy from
    the tracker's velocity."""
    w = _bare_worker()
    w._target_track_id = 7
    target_raw = _FakeRawTrack(
        7, _bbox(800, 300, 900, 500), TrackState.LOST, conf=0.8, velocity=(5.0, -2.0)
    )
    other_lost = _FakeRawTrack(9, _bbox(100, 100, 200, 200), TrackState.LOST, velocity=(1.0, 1.0))
    confirmed = _FakeRawTrack(3, _bbox(10, 10, 60, 60), TrackState.CONFIRMED, conf=0.95)
    w._detect = _FakeDetectStack([target_raw, other_lost, confirmed])  # type: ignore[assignment]

    out = w._maybe_track(_frame())

    ids = {t.track_id: t for t in out}
    assert 9 not in ids, "non-target LOST tracks must still be filtered out"
    assert 3 in ids and ids[3].lost is False
    assert 7 in ids, "the locked target's coasted LOST track must be published"
    published = ids[7]
    assert published.lost is True
    assert published.bbox.x1 == 800 and published.bbox.x2 == 900
    assert published.vx == 5.0
    assert published.vy == -2.0
