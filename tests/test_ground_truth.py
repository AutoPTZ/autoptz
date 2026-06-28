"""Slice 4 — synthetic ground truth + the pure GroundTruthComparator.

These tests are headless (no Qt): the synthetic ground truth is derived from the
SAME pure geometry helper that paints the drawn ``anim`` scene, so it aligns
frame-for-frame with what the detector sees.  The comparator is pure MOT-style
math (greedy IOU match → MOTA/MOTP/miss/id-switch).
"""

from __future__ import annotations

import os

from autoptz.benchmark.ground_truth import GroundTruthComparator
from autoptz.engine.pipeline.ingest import SyntheticAdapter
from autoptz.engine.runtime.messages import BBox, GroundTruthPerson


def _adapter(camera_id: str = "cam-gt", w: int = 640, h: int = 480) -> SyntheticAdapter:
    a = SyntheticAdapter(camera_id, address="anim", width=w, height=h, target_fps=30.0)
    a._open()
    return a


def _bbox_from_geom(cx: int, cy: int, person_h: int, half_w: int) -> tuple[float, ...]:
    """Mirror the drawn silhouette's outer extent (x1, y1, x2, y2)."""
    top = cy - person_h // 2
    return (cx - half_w, top, cx + half_w, top + person_h)


# ── synthetic ground truth ──────────────────────────────────────────────────────


class TestSyntheticGroundTruth:
    def test_gt_positions_byte_match_people_boxes(self) -> None:
        """GT bbox geometry is derived from the SAME helper that draws the scene."""
        a = _adapter()
        for idx in (1, 7, 23, 60):
            a._idx = idx
            t = a._idx / max(1.0, a._target_fps)
            boxes = a._people_boxes(t)
            gt = a._compute_people_ground_truth(t)
            # one GT entry per person geometry row (visible or not)
            assert len(gt) == len(boxes)
            for (pid, cx, cy, person_h, half_w, path_type), g in zip(boxes, gt, strict=True):
                assert g.person_id == pid
                assert g.path_type == path_type
                x1, y1, x2, y2 = _bbox_from_geom(cx, cy, person_h, half_w)
                assert (g.bbox.x1, g.bbox.y1, g.bbox.x2, g.bbox.y2) == (x1, y1, x2, y2)
        a._close()

    def test_offframe_person_is_not_visible(self) -> None:
        """visible mirrors the drawn off-frame clip (cx outside [-half_w, w+half_w])."""
        a = _adapter()
        w = a._w
        # Search the exit cycle for a frame where some person glides off-frame.
        found_offframe = False
        for idx in range(1, 400):
            a._idx = idx
            t = a._idx / max(1.0, a._target_fps)
            boxes = a._people_boxes(t)
            gt = a._compute_people_ground_truth(t)
            for (_pid, cx, _cy, _ph, half_w, _pt), g in zip(boxes, gt, strict=True):
                onframe = -half_w <= cx <= w + half_w
                assert g.visible is onframe
                if not onframe:
                    found_offframe = True
        a._close()
        assert found_offframe, "expected at least one off-frame person across the exit cycle"

    def test_gt_is_deterministic_per_camera_and_frame(self) -> None:
        a1 = _adapter("cam-det-gt")
        a2 = _adapter("cam-det-gt")
        for idx in (1, 5, 19, 88):
            a1._idx = a2._idx = idx
            t = idx / max(1.0, a1._target_fps)
            g1 = a1._compute_people_ground_truth(t)
            g2 = a2._compute_people_ground_truth(t)
            assert [x.model_dump() for x in g1] == [x.model_dump() for x in g2]
        a1._close()
        a2._close()

    def test_latest_gt_attr_empty_without_flag(self) -> None:
        """No env flag → reading frames never populates the latest-GT attribute."""
        os.environ.pop("AUTOPTZ_MARK_GT", None)
        a = _adapter()
        for _ in range(5):
            a._read_frame()
        assert a.latest_ground_truth() == []
        a._close()

    def test_latest_gt_attr_populated_with_flag(self) -> None:
        os.environ["AUTOPTZ_MARK_GT"] = "1"
        try:
            a = _adapter()
            for _ in range(5):
                a._read_frame()
            gt = a.latest_ground_truth()
            assert gt, "flag on → drawn scene should publish ground truth"
            assert all(isinstance(g, GroundTruthPerson) for g in gt)
        finally:
            os.environ.pop("AUTOPTZ_MARK_GT", None)
            a._close()

    def test_no_gt_for_clip_source(self, tmp_path) -> None:
        """Only the drawn scene has GT; a video/clip source publishes an empty list."""
        os.environ["AUTOPTZ_MARK_GT"] = "1"
        try:
            import cv2
            import numpy as np

            clip = tmp_path / "clip.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(clip), fourcc, 30.0, (64, 48))
            for _ in range(6):
                vw.write(np.zeros((48, 64, 3), dtype=np.uint8))
            vw.release()
            if not clip.exists():
                import pytest

                pytest.skip("no mp4v writer available")
            a = SyntheticAdapter(
                "cam-clip", address=str(clip), width=64, height=48, target_fps=30.0
            )
            a._open()
            for _ in range(4):
                a._read_frame()
            assert a.latest_ground_truth() == []
            a._close()
        finally:
            os.environ.pop("AUTOPTZ_MARK_GT", None)


# ── GroundTruthComparator (pure MOT math) ───────────────────────────────────────


def _gt(
    pid: int, x1: float, y1: float, x2: float, y2: float, path_type: str = "0"
) -> GroundTruthPerson:
    return GroundTruthPerson(
        person_id=pid, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), visible=True, path_type=path_type
    )


class _Track:
    """Minimal track stand-in: track_id + bbox attribute, like TrackInfo."""

    def __init__(self, track_id: int, bbox: BBox) -> None:
        self.track_id = track_id
        self.bbox = bbox


def _track(track_id: int, x1: float, y1: float, x2: float, y2: float) -> _Track:
    return _Track(track_id, BBox(x1=x1, y1=y1, x2=x2, y2=y2))


class TestGroundTruthComparator:
    def test_perfect_tracker_echoes_gt(self) -> None:
        comp = GroundTruthComparator()
        for _ in range(30):
            gt = [_gt(1, 0, 0, 10, 20), _gt(2, 50, 50, 60, 80)]
            tracks = [_track(1, 0, 0, 10, 20), _track(2, 50, 50, 60, 80)]
            comp.on_frame(tracks, gt)
        m = comp.finalize()
        assert m["miss_rate"] == 0.0
        assert m["id_switch_rate"] == 0.0
        assert m["motp"] < 1e-9
        assert m["mota"] == 1.0

    def test_iou_above_threshold_matches(self) -> None:
        comp = GroundTruthComparator()
        # heavy overlap (IOU > 0.5) → matched, no miss
        comp.on_frame([_track(1, 0, 0, 10, 10)], [_gt(1, 1, 1, 11, 11)])
        m = comp.finalize()
        assert m["miss_rate"] == 0.0

    def test_iou_below_threshold_no_match(self) -> None:
        comp = GroundTruthComparator()
        # tiny overlap (IOU < 0.3) → no match: a miss + a false positive
        comp.on_frame([_track(1, 0, 0, 10, 10)], [_gt(1, 8, 8, 18, 18)])
        m = comp.finalize()
        assert m["miss_rate"] == 1.0

    def test_miss_rate_one_when_no_detections(self) -> None:
        comp = GroundTruthComparator()
        for _ in range(10):
            comp.on_frame([], [_gt(1, 0, 0, 10, 10)])
        assert comp.finalize()["miss_rate"] == 1.0

    def test_id_switch_increments_on_id_flip(self) -> None:
        comp = GroundTruthComparator()
        # frames 1..5: gt person 1 tracked by track 7
        for _ in range(5):
            comp.on_frame([_track(7, 0, 0, 10, 10)], [_gt(1, 0, 0, 10, 10)])
        # frame 6: same gt person, DIFFERENT track id → one id switch
        comp.on_frame([_track(99, 0, 0, 10, 10)], [_gt(1, 0, 0, 10, 10)])
        m = comp.finalize()
        assert m["id_switch_rate"] > 0.0

    def test_metrics_finite_over_300_frames(self) -> None:
        import math

        a = _adapter("cam-300")
        comp = GroundTruthComparator()
        for idx in range(1, 301):
            a._idx = idx
            t = idx / max(1.0, a._target_fps)
            gt = [g for g in a._compute_people_ground_truth(t) if g.visible]
            # "lazy" tracker: echo every other gt as a track (induces misses + matches)
            tracks = [
                _track(g.person_id, g.bbox.x1, g.bbox.y1, g.bbox.x2, g.bbox.y2)
                for j, g in enumerate(gt)
                if j % 2 == 0
            ]
            comp.on_frame(tracks, gt)
        a._close()
        m = comp.finalize()
        for key in ("miss_rate", "id_switch_rate", "motp", "mota"):
            assert math.isfinite(m[key]), f"{key} not finite: {m[key]}"
