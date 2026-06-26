"""Assertion group 1: track ID continuity across detect_interval skips."""

from __future__ import annotations

import numpy as np
import pytest

import autoptz.engine.pipeline.track as track_mod
from autoptz.engine.pipeline.track import Tracker, TrackState, _SimpleIoUTracker
from tests.synthetic_tracking import (
    FRAME,
    constant_velocity_centres,
    detections_for_centres,
    make_mock_impl,
    tracker_rows_for_centres,
)

FPS = 30.0


def _interval_schedule(n_frames: int, interval: int) -> set[int]:
    """Frame indices on which PersonDetector would actually run inference.

    detect() runs on frames 1, N+1, 2N+1, … (1-based); 0-based that is i % N == 0.
    """
    return {i for i in range(n_frames) if i % interval == 0}


@pytest.mark.parametrize("interval", [1, 2, 3])
def test_track_id_stable_through_skips_via_mock_impl(interval: int) -> None:
    """A confirmed track keeps one ID and never re-IDs across detect skips."""
    n = 18
    centres = constant_velocity_centres(x0=200.0, vx=4.0, frames=n)
    detect_frames = _interval_schedule(n, interval)
    # Mock impl: emit the SAME track id on detection frames, empty on skip frames
    # (the tracker coasts on skip frames just as in production).
    rows = []
    for i in range(n):
        if i in detect_frames:
            rows.append(tracker_rows_for_centres([centres[i]], track_id=1)[0])
        else:
            rows.append(np.empty((0, 7), dtype=np.float32))
    tracker = Tracker(_impl=make_mock_impl(rows), min_hits=1, coast_window=1.5)

    seen_ids: set[int] = set()
    confirmed_or_lost = 0
    for i in range(n):
        dets = detections_for_centres([centres[i]])[0] if i in detect_frames else []
        tracks = tracker.update(dets, FRAME, fps=FPS)
        live = [t for t in tracks if t.track_id == 1]
        for t in tracks:
            seen_ids.add(t.track_id)
        if live and live[0].state in (TrackState.CONFIRMED, TrackState.LOST):
            confirmed_or_lost += 1

    assert seen_ids == {1}  # no spurious new IDs across the whole run
    # The track is present (confirmed or coasting) on every frame: min_hits=1 so
    # frame 0 is already CONFIRMED, giving exactly n frames total.
    assert confirmed_or_lost == n


@pytest.mark.parametrize("interval", [2, 3])
def test_track_confirmed_persists_during_skip_gap(interval: int) -> None:
    """On a skip frame the track is still returned as CONFIRMED or LOST, not gone."""
    n = 12
    centres = constant_velocity_centres(x0=200.0, vx=3.0, frames=n)
    detect_frames = _interval_schedule(n, interval)
    rows = [
        tracker_rows_for_centres([centres[i]], track_id=1)[0]
        if i in detect_frames
        else np.empty((0, 7), dtype=np.float32)
        for i in range(n)
    ]
    tracker = Tracker(_impl=make_mock_impl(rows), min_hits=1, coast_window=1.5)
    states_on_skip: list[TrackState] = []
    for i in range(n):
        dets = detections_for_centres([centres[i]])[0] if i in detect_frames else []
        tracks = tracker.update(dets, FRAME, fps=FPS)
        if i not in detect_frames and i > 0:
            live = [t for t in tracks if t.track_id == 1]
            assert live, f"track lost entirely on skip frame {i}"
            states_on_skip.append(live[0].state)
    # Every skip-frame appearance is a coast (LOST) — never a removal/new id.
    # states_on_skip must be non-empty: interval>=2 guarantees at least one skip frame.
    assert states_on_skip
    assert all(s == TrackState.LOST for s in states_on_skip)


def test_id_stable_through_skips_via_iou_fallback() -> None:
    """Without boxmot, the built-in IoU tracker keeps a stable id across a 1-frame
    skip when boxes overlap (real coast path on a minimal install)."""
    orig = track_mod._BOXMOT_AVAILABLE
    track_mod._BOXMOT_AVAILABLE = False
    try:
        # Slow drift so consecutive *detected* boxes still overlap (IoU >= 0.3)
        # even with one skipped frame between them.
        centres = constant_velocity_centres(x0=300.0, vx=2.0, frames=6)
        detect_frames = {0, 2, 4}
        tracker = Tracker(min_hits=1, coast_window=2.0)
        tracker._impl_pending = True
        tracker._impl = None
        first_id = None
        detect_frames_with_track = 0
        for i in range(6):
            dets = detections_for_centres([centres[i]])[0] if i in detect_frames else []
            tracks = tracker.update(dets, FRAME, fps=FPS)
            assert isinstance(tracker._impl, _SimpleIoUTracker)
            live = [t for t in tracks if t.state != TrackState.LOST]
            if i in detect_frames and live:
                detect_frames_with_track += 1
                if first_id is None:
                    first_id = live[0].track_id
                else:
                    assert live[0].track_id == first_id  # no re-ID across the skip
        # Track must have been seen on at least 2 detect frames: a regression that drops
        # the track entirely (live == []) on every frame after the first would be silent
        # without this guard.
        assert detect_frames_with_track >= 2, (
            f"track only appeared on {detect_frames_with_track} detect frame(s); "
            "expected it to survive across skips"
        )
    finally:
        track_mod._BOXMOT_AVAILABLE = orig
