"""AutoPTZ Mark — ground-truth accuracy comparator (pure, no Qt).

Compares a tracker's per-frame output against the synthetic scene's ground truth
with the standard MOT bookkeeping: greedy IOU matching (descending IOU, ≥ 0.5)
accumulates matches, false positives, misses, localization error, and identity
switches.  :meth:`GroundTruthComparator.finalize` reduces those to the familiar
CLEAR-MOT summary metrics (miss rate, id-switch rate, MOTP, MOTA).

Everything here is pure math over plain objects, so it runs headless and stays
trivially testable: tracks need only ``track_id`` + ``bbox`` (e.g. ``TrackInfo``),
ground truth is ``GroundTruthPerson`` (``person_id`` + ``bbox``).
"""

from __future__ import annotations

from typing import Any

# IOU at/above which a track box and a ground-truth box may be matched.
_MATCH_IOU = 0.5


def _iou(a: Any, b: Any) -> float:
    """Intersection-over-union of two boxes exposing ``x1/y1/x2/y2``."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class GroundTruthComparator:
    """Accumulate CLEAR-MOT statistics frame-by-frame, then summarise.

    Feed each processed frame's tracks + ground truth to :meth:`on_frame`; call
    :meth:`finalize` once at the end for the summary dict.
    """

    def __init__(self, match_iou: float = _MATCH_IOU) -> None:
        self._match_iou = float(match_iou)
        self._gt_total = 0  # total ground-truth objects across all frames
        self._matches = 0  # gt boxes matched to a track this frame
        self._false_pos = 0  # tracks with no gt match
        self._misses = 0  # gt boxes with no track match
        self._id_switches = 0
        self._dist_sum = 0.0  # Σ (1 - IOU) over matched pairs → MOTP numerator
        # gt person_id → track_id it was last matched to (for id-switch detection).
        self._last_assignment: dict[int, int] = {}

    def on_frame(self, tracks: Any, gt_people: Any) -> None:
        """Greedy-match one frame's *tracks* against *gt_people*.

        ``tracks`` items expose ``track_id`` + ``bbox``; ``gt_people`` items expose
        ``person_id`` + ``bbox``.  Both are iterated as plain sequences.
        """
        track_list = list(tracks)
        gt_list = list(gt_people)
        self._gt_total += len(gt_list)

        # Build every candidate (iou, gt_index, track_index) pair above threshold,
        # then greedily consume them in descending-IOU order so the best overlaps
        # win and each box is used at most once.
        candidates: list[tuple[float, int, int]] = []
        for gi, g in enumerate(gt_list):
            for ti, tr in enumerate(track_list):
                iou = _iou(g.bbox, tr.bbox)
                if iou >= self._match_iou:
                    candidates.append((iou, gi, ti))
        candidates.sort(key=lambda c: c[0], reverse=True)

        gt_used: set[int] = set()
        tr_used: set[int] = set()
        for iou, gi, ti in candidates:
            if gi in gt_used or ti in tr_used:
                continue
            gt_used.add(gi)
            tr_used.add(ti)
            self._matches += 1
            self._dist_sum += 1.0 - iou
            pid = int(gt_list[gi].person_id)
            tid = int(track_list[ti].track_id)
            prev = self._last_assignment.get(pid)
            if prev is not None and prev != tid:
                self._id_switches += 1
            self._last_assignment[pid] = tid

        self._misses += len(gt_list) - len(gt_used)
        self._false_pos += len(track_list) - len(tr_used)

    def finalize(self) -> dict[str, float]:
        """Reduce accumulated counts to the CLEAR-MOT summary metrics.

        Returns ``{miss_rate, id_switch_rate, motp, mota}``:

        * ``miss_rate``      = misses / gt_total
        * ``id_switch_rate`` = id_switches / gt_total
        * ``motp``           = mean (1 - IOU) over matched pairs (0 = pixel-perfect)
        * ``mota``           = 1 - (misses + false_pos + id_switches) / gt_total
        """
        gt = self._gt_total
        if gt <= 0:
            return {"miss_rate": 0.0, "id_switch_rate": 0.0, "motp": 0.0, "mota": 1.0}
        miss_rate = self._misses / gt
        id_switch_rate = self._id_switches / gt
        motp = self._dist_sum / self._matches if self._matches else 0.0
        mota = 1.0 - (self._misses + self._false_pos + self._id_switches) / gt
        return {
            "miss_rate": miss_rate,
            "id_switch_rate": id_switch_rate,
            "motp": motp,
            "mota": mota,
        }
