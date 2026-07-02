"""Unit tests for autoptz.engine.pipeline.track.

BoxMOT is NOT required — the tracker implementation is injected via ``_impl``
so all state-machine and lifecycle logic can be tested with a plain mock.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from autoptz.engine.pipeline.detect import BBox, Detection
from autoptz.engine.pipeline.track import (
    Track,
    Tracker,
    TrackerType,
    TrackState,
    _probe_boxmot,
)

# ── Mock BoxMOT implementation ────────────────────────────────────────────────


def _make_impl(track_rows: list[list[float]] | None = None) -> MagicMock:
    """Return a mock BoxMOT tracker whose update() returns *track_rows*.

    Each row should be [x1,y1,x2,y2,track_id,conf,cls(,det_idx)].
    """
    impl = MagicMock()
    if track_rows is None:
        impl.update.return_value = np.empty((0, 7), dtype=np.float32)
    else:
        arr = np.array(track_rows, dtype=np.float32) if track_rows else np.empty((0, 7), np.float32)
        impl.update.return_value = arr
    return impl


# ── Fixtures ───────────────────────────────────────────────────────────────────

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _det(x1, y1, x2, y2, conf=0.9) -> Detection:
    return Detection(BBox(x1, y1, x2, y2), conf, 0)


# ── Tracker basics ─────────────────────────────────────────────────────────────


class TestTrackerBasics:
    def test_no_detections_no_tracks(self) -> None:
        tracker = Tracker(_impl=_make_impl([]))
        tracks = tracker.update([], FRAME)
        assert tracks == []

    def test_single_track_returned(self) -> None:
        impl = _make_impl([[10, 20, 100, 200, 1, 0.9, 0]])
        tracker = Tracker(_impl=impl)
        dets = [_det(10, 20, 100, 200)]
        tracks = tracker.update(dets, FRAME)
        assert len(tracks) == 1
        t = tracks[0]
        assert t.track_id == 1
        assert t.bbox.x1 == pytest.approx(10.0)
        assert t.conf == pytest.approx(0.9)

    def test_two_tracks_returned(self) -> None:
        impl = _make_impl(
            [
                [10, 20, 100, 200, 1, 0.9, 0],
                [300, 50, 400, 300, 2, 0.85, 0],
            ]
        )
        tracker = Tracker(_impl=impl)
        tracks = tracker.update([_det(10, 20, 100, 200), _det(300, 50, 400, 300)], FRAME)
        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert ids == {1, 2}

    def test_active_count(self) -> None:
        impl = _make_impl([[10, 10, 100, 200, 7, 0.9, 0]])
        tracker = Tracker(_impl=impl)
        tracker.update([_det(10, 10, 100, 200)], FRAME)
        assert tracker.active_count >= 1

    def test_reset_clears_state(self) -> None:
        impl = _make_impl([[10, 10, 100, 200, 3, 0.9, 0]])
        tracker = Tracker(_impl=impl)
        tracker.update([_det(10, 10, 100, 200)], FRAME)
        tracker.reset()
        assert tracker.active_count == 0


# ── Track lifecycle ────────────────────────────────────────────────────────────


class TestTrackLifecycle:
    def test_new_track_is_tentative_with_min_hits_2(self) -> None:
        impl = MagicMock()
        impl.update.return_value = np.array([[10, 20, 100, 200, 1, 0.9, 0]], dtype=np.float32)
        tracker = Tracker(_impl=impl, min_hits=2)
        tracks = tracker.update([_det(10, 20, 100, 200)], FRAME)
        assert tracks[0].state == TrackState.TENTATIVE

    def test_track_confirmed_after_min_hits(self) -> None:
        track_row = np.array([[10, 20, 100, 200, 1, 0.9, 0]], dtype=np.float32)
        impl = MagicMock()
        impl.update.return_value = track_row
        tracker = Tracker(_impl=impl, min_hits=2)
        tracker.update([_det(10, 20, 100, 200)], FRAME)  # hits=1 → tentative
        tracks = tracker.update([_det(10, 20, 100, 200)], FRAME)  # hits=2 → confirmed
        assert tracks[0].state == TrackState.CONFIRMED

    def test_min_hits_1_immediately_confirmed(self) -> None:
        impl = _make_impl([[10, 20, 100, 200, 1, 0.9, 0]])
        tracker = Tracker(_impl=impl, min_hits=1)
        tracks = tracker.update([_det(10, 20, 100, 200)], FRAME)
        assert tracks[0].state == TrackState.CONFIRMED

    def test_track_enters_lost_when_missing(self) -> None:
        """Track present frame 1, absent frame 2 → LOST on frame 2."""
        impl = MagicMock()
        # Frame 1: track present
        impl.update.side_effect = [
            np.array([[10, 20, 100, 200, 1, 0.9, 0]], dtype=np.float32),
            np.empty((0, 7), dtype=np.float32),  # frame 2: gone from BoxMOT
        ]
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=1.0)
        tracker.update([_det(10, 20, 100, 200)], FRAME, fps=10.0)  # coast_max=10
        tracks = tracker.update([], FRAME, fps=10.0)
        lost = [t for t in tracks if t.state == TrackState.LOST]
        assert len(lost) == 1
        assert lost[0].track_id == 1

    def test_track_removed_after_coast_window(self) -> None:
        """After coast_max_frames without detection, track is REMOVED (not returned)."""
        impl = MagicMock()

        # Frame 1: track appears
        impl.update.side_effect = [
            np.array([[10, 20, 100, 200, 1, 0.9, 0]], dtype=np.float32),
        ] + [np.empty((0, 7), dtype=np.float32)] * 10

        tracker = Tracker(_impl=impl, min_hits=1, coast_window=0.5)
        tracker.update([_det(10, 20, 100, 200)], FRAME, fps=10.0)  # coast_max = 5 frames

        # Run 6 empty frames (> coast_max_frames=5) to trigger removal
        tracks_by_frame: list[list[Track]] = []
        for _ in range(6):
            tracks_by_frame.append(tracker.update([], FRAME, fps=10.0))

        # After coast window expires, track should disappear
        final = tracks_by_frame[-1]
        assert not any(t.track_id == 1 for t in final)

    def test_reacquired_track_exits_lost(self) -> None:
        """A track re-detected while in LOST state should return to CONFIRMED."""
        impl = MagicMock()
        row = np.array([[10, 20, 100, 200, 1, 0.9, 0]], dtype=np.float32)
        impl.update.side_effect = [
            row,  # frame 1: present
            np.empty((0, 7), dtype=np.float32),  # frame 2: missing
            row,  # frame 3: re-detected
        ]
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=1.0)
        tracker.update([_det(10, 20, 100, 200)], FRAME, fps=10.0)  # confirmed
        tracker.update([], FRAME, fps=10.0)  # lost
        tracks = tracker.update([_det(10, 20, 100, 200)], FRAME, fps=10.0)

        confirmed = [t for t in tracks if t.track_id == 1 and t.state == TrackState.CONFIRMED]
        assert len(confirmed) == 1

    def test_age_increments_each_frame(self) -> None:
        row = np.array([[10, 20, 100, 200, 5, 0.9, 0]], dtype=np.float32)
        impl = MagicMock()
        impl.update.return_value = row
        tracker = Tracker(_impl=impl, min_hits=1)
        for _i in range(3):
            tracks = tracker.update([_det(10, 20, 100, 200)], FRAME)
        assert tracks[0].age == 3

    def test_hits_counter_increments(self) -> None:
        row = np.array([[10, 20, 100, 200, 5, 0.9, 0]], dtype=np.float32)
        impl = MagicMock()
        impl.update.return_value = row
        tracker = Tracker(_impl=impl)
        for _ in range(5):
            tracks = tracker.update([_det(10, 20, 100, 200)], FRAME)
        assert tracks[0].hits == 5


# ── Velocity ──────────────────────────────────────────────────────────────────


class TestVelocity:
    def test_first_frame_velocity_zero(self) -> None:
        impl = _make_impl([[100, 100, 200, 300, 1, 0.9, 0]])
        tracker = Tracker(_impl=impl, min_hits=1)
        tracks = tracker.update([_det(100, 100, 200, 300)], FRAME)
        assert tracks[0].velocity == (0.0, 0.0)

    def test_moving_right_positive_vx(self) -> None:
        """Track moves +50px right between frames."""
        impl = MagicMock()
        impl.update.side_effect = [
            np.array([[100, 100, 200, 300, 1, 0.9, 0]], dtype=np.float32),  # cx=150
            np.array([[150, 100, 250, 300, 1, 0.9, 0]], dtype=np.float32),  # cx=200
        ]
        tracker = Tracker(_impl=impl, min_hits=1)
        tracker.update([_det(100, 100, 200, 300)], FRAME)
        tracks = tracker.update([_det(150, 100, 250, 300)], FRAME)
        vx, vy = tracks[0].velocity
        assert vx == pytest.approx(50.0)
        assert vy == pytest.approx(0.0)

    def test_stationary_track_zero_velocity(self) -> None:
        row = np.array([[100, 100, 200, 300, 1, 0.9, 0]], dtype=np.float32)
        impl = MagicMock()
        impl.update.return_value = row
        tracker = Tracker(_impl=impl, min_hits=1)
        tracker.update([_det(100, 100, 200, 300)], FRAME)
        tracks = tracker.update([_det(100, 100, 200, 300)], FRAME)
        vx, vy = tracks[0].velocity
        assert vx == pytest.approx(0.0)
        assert vy == pytest.approx(0.0)


# ── TrackerType handling ───────────────────────────────────────────────────────


class TestTrackerTypeEnum:
    def test_bytetrack_string(self) -> None:
        tracker = Tracker(_impl=_make_impl(), tracker_type="bytetrack")
        assert tracker._tracker_type == TrackerType.BYTETRACK

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError):
            Tracker(_impl=_make_impl(), tracker_type="nosuchtracker")  # type: ignore[arg-type]


# ── BoxMOT unavailability ──────────────────────────────────────────────────────


class TestBoxMOTUnavailable:
    def test_probe_returns_bool(self) -> None:
        result = _probe_boxmot()
        assert isinstance(result, bool)

    def test_no_impl_no_boxmot_falls_back_to_iou_tracker(self) -> None:
        """Without boxmot installed, update() must NOT raise — it degrades to the
        built-in lightweight IoU tracker so detection/boxes still work."""
        import autoptz.engine.pipeline.track as track_mod
        from autoptz.engine.pipeline.track import _SimpleIoUTracker

        orig = track_mod._BOXMOT_AVAILABLE
        track_mod._BOXMOT_AVAILABLE = False
        try:
            tracker = Tracker()
            tracker._impl_pending = True
            tracker._impl = None
            # First detection → a stable confirmed track, no exception.
            tracks = tracker.update([_det(10, 20, 100, 200)], FRAME, fps=30.0)
            assert isinstance(tracker._impl, _SimpleIoUTracker)
            assert len(tracks) == 1
            first_id = tracks[0].track_id
            # A nudged box on the next frame keeps the same id (IoU association).
            tracks2 = tracker.update([_det(14, 24, 104, 204)], FRAME, fps=30.0)
            assert tracks2[0].track_id == first_id
        finally:
            track_mod._BOXMOT_AVAILABLE = orig


# ── Edge cases ─────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_update_with_8col_output(self) -> None:
        """BoxMOT sometimes returns 8 columns [… det_idx]; wrapper must handle it."""
        row8 = np.array([[10, 20, 100, 200, 1, 0.9, 0, -1]], dtype=np.float32)
        impl = MagicMock()
        impl.update.return_value = row8
        tracker = Tracker(_impl=impl, min_hits=1)
        tracks = tracker.update([_det(10, 20, 100, 200)], FRAME)
        assert len(tracks) == 1

    def test_update_with_empty_frame(self) -> None:
        impl = _make_impl([])
        tracker = Tracker(_impl=impl)
        tracks = tracker.update([], np.zeros((0, 0, 3), dtype=np.uint8))
        assert tracks == []

    def test_multiple_lost_tracks_all_returned(self) -> None:
        impl = MagicMock()
        impl.update.side_effect = [
            np.array(
                [
                    [10, 10, 100, 200, 1, 0.9, 0],
                    [300, 10, 400, 200, 2, 0.8, 0],
                ],
                dtype=np.float32,
            ),
            np.empty((0, 7), dtype=np.float32),  # both gone
        ]
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=2.0)
        tracker.update([_det(10, 10, 100, 200), _det(300, 10, 400, 200)], FRAME, fps=1.0)
        tracks = tracker.update([], FRAME, fps=1.0)
        lost = [t for t in tracks if t.state == TrackState.LOST]
        assert len(lost) == 2
        ids = {t.track_id for t in lost}
        assert ids == {1, 2}

    def test_fps_param_affects_coast_max_frames(self) -> None:
        impl = MagicMock()
        impl.update.side_effect = [
            np.array([[10, 10, 100, 200, 1, 0.9, 0]], dtype=np.float32),
        ] + [np.empty((0, 7), dtype=np.float32)] * 4

        tracker = Tracker(_impl=impl, min_hits=1, coast_window=0.5)
        # At 2 fps, coast_max = 1 frame
        tracker.update([_det(10, 10, 100, 200)], FRAME, fps=2.0)
        tracker.update([], FRAME, fps=2.0)  # frames_lost=1 → at coast_max
        t3 = tracker.update([], FRAME, fps=2.0)  # frames_lost=2 > coast_max → removed
        alive = [t for t in t3 if t.track_id == 1]
        assert len(alive) == 0


# ── Predictive coast (LOST tracks keep moving along their last velocity) ──────


def _moving_rows(cx_values: list[float], *, cy: float = 150.0, w: float = 40.0, h: float = 100.0):
    """One [1,7] BoxMOT row per centre-x value, for a box of fixed size."""
    return [
        np.array(
            [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 1, 0.9, 0]],
            dtype=np.float32,
        )
        for cx in cx_values
    ]


def _run_moving_then_lost(
    tracker: Tracker,
    impl: MagicMock,
    cx_values: list[float],
    lost_frames: int,
    fps: float = 10.0,
) -> list[list[Track]]:
    """Feed *cx_values* active frames then *lost_frames* empty frames.

    Returns the per-frame track lists for the lost frames only.
    """
    impl.update.side_effect = (
        _moving_rows(cx_values) + [np.empty((0, 7), dtype=np.float32)] * lost_frames
    )
    for cx in cx_values:
        tracker.update([_det(cx - 20, 100, cx + 20, 200)], FRAME, fps=fps)
    out: list[list[Track]] = []
    for _ in range(lost_frames):
        out.append(tracker.update([], FRAME, fps=fps))
    return out


class TestPredictiveCoast:
    """A LOST track must coast along its pre-loss velocity (damped), not freeze.

    Rationale: upstream ByteTrack/BoT-SORT Kalman-predict lost tracks with the
    positional velocity intact; emitting a frozen bbox with velocity=(0,0) was
    the direct cause of PTZ bounce after brief occlusions.
    """

    def test_lost_track_keeps_pre_loss_velocity(self) -> None:
        """First LOST frame reports (damped) pre-loss velocity, not zero."""
        impl = MagicMock()
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=1.0)
        cxs = [100.0 + 10.0 * i for i in range(8)]  # +10 px/frame
        lost_frames = _run_moving_then_lost(tracker, impl, cxs, lost_frames=1)
        lost = [t for t in lost_frames[0] if t.state == TrackState.LOST]
        assert len(lost) == 1
        vx, vy = lost[0].velocity
        assert vx > 5.0, f"LOST velocity should carry pre-loss motion, got vx={vx}"
        assert vx <= 10.5
        assert abs(vy) < 1.0

    def test_lost_bbox_coasts_forward(self) -> None:
        """The LOST bbox centre keeps advancing in the direction of motion."""
        impl = MagicMock()
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=1.0)
        cxs = [100.0 + 10.0 * i for i in range(8)]  # last active cx = 170
        frames = _run_moving_then_lost(tracker, impl, cxs, lost_frames=3)
        centres = []
        for tracks in frames:
            (lost,) = [t for t in tracks if t.state == TrackState.LOST]
            centres.append(lost.bbox.cx)
        assert centres[0] > 170.0 + 2.0, f"first coast frame should advance, got {centres[0]}"
        assert centres[0] < centres[1] < centres[2], f"coast should keep advancing: {centres}"
        total = centres[2] - 170.0
        assert 10.0 < total < 35.0, f"3-frame damped coast should advance 10-35px, got {total}"

    def test_coast_velocity_decays(self) -> None:
        """Coast velocity is damped toward zero (bounded drift on wrong guesses)."""
        impl = MagicMock()
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=2.0)
        cxs = [100.0 + 10.0 * i for i in range(8)]
        frames = _run_moving_then_lost(tracker, impl, cxs, lost_frames=8)
        vxs = []
        for tracks in frames:
            (lost,) = [t for t in tracks if t.state == TrackState.LOST]
            vxs.append(lost.velocity[0])
        assert vxs[-1] < vxs[0], f"velocity should decay across the coast window: {vxs}"
        assert vxs[-1] < 3.0, f"velocity should decay to near zero, got {vxs[-1]}"

    def test_stationary_lost_track_does_not_drift(self) -> None:
        impl = MagicMock()
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=1.0)
        cxs = [100.0] * 8  # stationary
        frames = _run_moving_then_lost(tracker, impl, cxs, lost_frames=5)
        for tracks in frames:
            (lost,) = [t for t in tracks if t.state == TrackState.LOST]
            assert abs(lost.bbox.cx - 100.0) < 1.0
            assert abs(lost.velocity[0]) < 0.5

    def test_coasted_centre_clamped_to_frame(self) -> None:
        """Coast must not extrapolate the box centre outside the frame."""
        impl = MagicMock()
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=3.0)
        cxs = [560.0 + 15.0 * i for i in range(5)]  # heading for the 640px edge
        frames = _run_moving_then_lost(tracker, impl, cxs, lost_frames=20)
        for tracks in frames:
            lost = [t for t in tracks if t.state == TrackState.LOST]
            if not lost:
                continue
            assert lost[0].bbox.cx <= 640.0, f"coasted centre left the frame: {lost[0].bbox.cx}"

    def test_coasted_box_keeps_size(self) -> None:
        impl = MagicMock()
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=1.0)
        cxs = [100.0 + 10.0 * i for i in range(8)]
        frames = _run_moving_then_lost(tracker, impl, cxs, lost_frames=3)
        for tracks in frames:
            (lost,) = [t for t in tracks if t.state == TrackState.LOST]
            assert (lost.bbox.x2 - lost.bbox.x1) == pytest.approx(40.0, abs=0.1)
            assert (lost.bbox.y2 - lost.bbox.y1) == pytest.approx(100.0, abs=0.1)

    def test_reacquire_after_coast_has_no_velocity_spike(self) -> None:
        """Re-detection after a 3-frame gap must not report a gap-spanning velocity.

        Without coast, prev-centre freezes at the loss point and the first
        re-acquired frame reports the whole gap as one frame of motion
        (~40 px/frame here) — which whips the PTZ feed-forward.
        """
        impl = MagicMock()
        tracker = Tracker(_impl=impl, min_hits=1, coast_window=1.0)
        cxs = [100.0 + 10.0 * i for i in range(8)]  # last active cx=170, v=10
        rows = _moving_rows(cxs)
        rows += [np.empty((0, 7), dtype=np.float32)] * 3
        rows += _moving_rows([210.0])  # reappears where constant motion predicts
        impl.update.side_effect = rows
        for cx in cxs:
            tracker.update([_det(cx - 20, 100, cx + 20, 200)], FRAME, fps=10.0)
        for _ in range(3):
            tracker.update([], FRAME, fps=10.0)
        tracks = tracker.update([_det(190, 100, 230, 200)], FRAME, fps=10.0)
        (t,) = [x for x in tracks if x.track_id == 1]
        assert t.state == TrackState.CONFIRMED
        assert t.velocity[0] < 25.0, f"re-acquire velocity spike: vx={t.velocity[0]}"
        assert t.velocity[0] > 0.0


class TestTrackerCreationFps:
    """The BoxMOT tracker is created lazily on the FIRST frame — when the ingest
    fps is still ~0 (frame intervals not yet timed).  A degenerate creation fps
    permanently collapses BoT-SORT's lost-track survival window to ~0 frames
    (``max_time_lost = frame_rate/30 * track_buffer``), so a single missed
    detection drops the track and the reappearing person gets a NEW id —
    id-switch / target-loss / PTZ bounce after any brief occlusion.
    """

    def test_warmup_fps_floored_to_sane_default(self) -> None:
        from autoptz.engine.pipeline.track import _tracker_creation_fps

        # The caller floors at max(1.0, fps), so warmup creation fps is ~1.0.
        assert _tracker_creation_fps(0.0) == 30.0
        assert _tracker_creation_fps(1.0) == 30.0
        assert _tracker_creation_fps(2.0) == 30.0

    def test_real_fps_preserved(self) -> None:
        from autoptz.engine.pipeline.track import _tracker_creation_fps

        assert _tracker_creation_fps(24.0) == 24.0
        assert _tracker_creation_fps(30.0) == 30.0
        assert _tracker_creation_fps(60.0) == 60.0

    def test_botsort_lost_window_survives_warmup_creation(self) -> None:
        """Regression: a BoT-SORT tracker created during warmup (fps≈1) must keep a
        non-zero lost-track survival window so boxmot can re-associate a person who
        reappears within the coast window (was ``max_time_lost == 0``)."""
        if not _probe_boxmot():
            pytest.skip("boxmot not installed")
        tracker = Tracker(tracker_type=TrackerType.BOTSORT, coast_window=1.5)
        tracker.update([], FRAME, fps=1.0)  # first frame during warmup
        impl = tracker._impl
        assert getattr(impl, "max_time_lost", 0) >= 30
