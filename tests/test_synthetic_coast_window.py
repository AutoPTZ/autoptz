"""Assertion group 3: coast window — LOST then REMOVED at the coast_max boundary."""

from __future__ import annotations

import numpy as np
import pytest

from autoptz.engine.pipeline.detect import BBox, Detection
from autoptz.engine.pipeline.track import Tracker, TrackState
from tests.synthetic_tracking import FRAME, box_at, make_mock_impl


def _present_then_gone(n_gone: int, track_id: int = 1) -> list[np.ndarray]:
    b = box_at(320.0, 240.0)
    present = np.array([[b.x1, b.y1, b.x2, b.y2, float(track_id), 0.9, 0.0]], dtype=np.float32)
    return [present] + [np.empty((0, 7), dtype=np.float32)] * n_gone


@pytest.mark.parametrize(
    ("coast_window", "fps"),
    [(0.5, 10.0), (1.0, 10.0), (1.5, 30.0), (0.2, 5.0)],
)
def test_lost_then_removed_at_boundary(coast_window: float, fps: float) -> None:
    coast_max = max(1, int(coast_window * fps))
    n_gone = coast_max + 2
    tracker = Tracker(
        _impl=make_mock_impl(_present_then_gone(n_gone)),
        min_hits=1,
        coast_window=coast_window,
    )
    # Frame 0: confirm. Any non-empty detection list is fine; the mock impl drives
    # the returned rows, so the detection content is ignored.
    confirm_det = [Detection(BBox(0, 0, 1, 1), 0.9, 0)]
    tracks0 = tracker.update(confirm_det, FRAME, fps=fps)
    assert any(t.track_id == 1 and t.state == TrackState.CONFIRMED for t in tracks0)

    lost_frames = 0
    removed_frame: int | None = None
    for i in range(1, n_gone + 1):
        tracks = tracker.update([], FRAME, fps=fps)
        live = [t for t in tracks if t.track_id == 1]
        if live:
            assert live[0].state == TrackState.LOST  # coasting, not confirmed
            lost_frames += 1
        elif removed_frame is None:
            removed_frame = i

    # The track stayed LOST for at least one frame, then was REMOVED (dropped).
    assert lost_frames >= 1
    assert removed_frame is not None
    # Removal happens once frames_lost exceeds coast_max — at coast_max+1 ticks.
    assert removed_frame <= coast_max + 1


def test_track_returned_as_lost_immediately_after_miss() -> None:
    tracker = Tracker(_impl=make_mock_impl(_present_then_gone(3)), min_hits=1, coast_window=1.0)
    tracker.update([Detection(BBox(0, 0, 1, 1), 0.9, 0)], FRAME, fps=10.0)
    tracks = tracker.update([], FRAME, fps=10.0)
    lost = [t for t in tracks if t.state == TrackState.LOST and t.track_id == 1]
    assert len(lost) == 1
    assert lost[0].velocity == (0.0, 0.0)  # LOST tracks report zero velocity
