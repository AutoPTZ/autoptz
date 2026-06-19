"""BoxMOT tracker wrapper: BoT-SORT (default), DeepOCSORT, ByteTrack.

Design
------
- Thin wrapper around a BoxMOT tracker instance; the concrete tracker is
  either created from ``boxmot`` (required at runtime) or injected via
  ``_impl`` (for tests).

- Exposes ``list[Track]`` with a proper four-state lifecycle:
    ``TENTATIVE → CONFIRMED → LOST (coasting) → REMOVED``

  BoxMOT only returns *active* tracks from ``update()``.  Lost tracks
  (within the coast window) and removed tracks are managed here so the
  camera worker can enter coast-mode when the target disappears and
  re-acquire rather than spawn a new ID.

- Velocity is estimated from consecutive bbox-centre deltas.  On the
  first frame a track appears, velocity is ``(0, 0)``.

- Camera-motion compensation (CMC): BoT-SORT's built-in ECC/optical-flow
  CMC is enabled by default; ByteTrack has no CMC.

- ``min_hits``: frames of consecutive detections before TENTATIVE → CONFIRMED
  (default 1 for BoT-SORT/DeepOCSORT, 3 for ByteTrack).

Track output array convention (BoxMOT ≥ 12)
-------------------------------------------
``update()`` returns ``[N, 7|8]`` per row:
    ``[x1, y1, x2, y2, track_id, conf, class_id (, det_idx)]``

We require at least 7 columns; column 7 (det_idx) is ignored if absent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from autoptz.engine.pipeline.detect import BBox, Detection, detections_to_numpy

log = logging.getLogger(__name__)

# ── Optional BoxMOT probe ──────────────────────────────────────────────────────

_BOXMOT_AVAILABLE: bool | None = None


def _probe_boxmot() -> bool:
    global _BOXMOT_AVAILABLE
    if _BOXMOT_AVAILABLE is None:
        try:
            import boxmot as _  # noqa: F401
            _BOXMOT_AVAILABLE = True
            _quiet_boxmot_logging()
        except ImportError:
            _BOXMOT_AVAILABLE = False
    return bool(_BOXMOT_AVAILABLE)


def _quiet_boxmot_logging() -> None:
    """Silence BoxMOT's chatty output (per-tracker config dumps at INFO and the
    'ECC did not converge' warnings on low-texture frames) so it doesn't flood
    the console.  BoxMOT ≥ 19 logs through the stdlib ``logging`` module on a
    ``"boxmot"`` logger with its own Rich handler; raising its level to ERROR
    keeps genuine failures visible while dropping the noise."""
    try:
        blog = logging.getLogger("boxmot")
        blog.setLevel(logging.ERROR)
        # boxmot resets the logger level during construction but keeps its Rich
        # handler — pinning the handler to ERROR filters the init config dump too.
        for h in blog.handlers:
            h.setLevel(logging.ERROR)
    except Exception:  # noqa: BLE001
        pass


def boxmot_available() -> bool:
    """Public capability probe: is the BoxMOT tracker backend importable?"""
    return _probe_boxmot()


# One-time guard so the "using lightweight fallback tracker" line is logged
# once per process rather than once per camera worker.
_LOGGED_FALLBACK_TRACKER = False


def _log_fallback_tracker_once() -> None:
    global _LOGGED_FALLBACK_TRACKER
    if _LOGGED_FALLBACK_TRACKER:
        return
    _LOGGED_FALLBACK_TRACKER = True
    log.info(
        "boxmot not installed — using built-in lightweight IoU tracker "
        "(detection + boxes work; install boxmot for occlusion-robust BoT-SORT "
        "+ ReID re-acquisition).",
    )


# ── Lightweight fallback tracker (no boxmot) ────────────────────────────────────


class _SimpleIoUTracker:
    """Greedy-IoU multi-object tracker used when ``boxmot`` is not installed.

    Mirrors BoxMOT's ``update(dets, frame) -> [N, 7]`` contract
    (``[x1, y1, x2, y2, track_id, conf, cls]``) so :class:`Tracker` wraps it
    transparently.  Motion-only: it associates this frame's detections to the
    previous frame's boxes by IoU and keeps stable integer ids — no appearance
    ReID and no Kalman prediction.  Enough to draw and follow person boxes on a
    minimal install; install ``boxmot`` for occlusion-robust BoT-SORT.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self._iou = iou_threshold
        self._max_age = max(1, int(max_age))
        self._next_id = 1
        # track_id → (xyxy box, frames-since-last-seen)
        self._tracks: dict[int, tuple[NDArray[np.float32], int]] = {}

    @staticmethod
    def _iou_matrix(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        ax1, ay1, ax2, ay2 = (a[:, i][:, None] for i in range(4))
        bx1, by1, bx2, by2 = (b[:, i][None, :] for i in range(4))
        iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
        ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)

    def update(self, dets: NDArray[np.float32], frame: Any = None) -> NDArray[np.float32]:
        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim != 2 or dets.shape[0] == 0:
            self._age_and_prune(set())
            return np.empty((0, 7), dtype=np.float32)

        boxes = dets[:, :4]
        track_ids = list(self._tracks.keys())
        assigned: dict[int, int] = {}  # det index → track id
        if track_ids:
            track_boxes = np.array([self._tracks[t][0] for t in track_ids], dtype=np.float32)
            iou = self._iou_matrix(boxes, track_boxes)
            pairs = sorted(
                ((float(iou[i, j]), i, j)
                 for i in range(iou.shape[0]) for j in range(iou.shape[1])),
                reverse=True,
            )
            used_det: set[int] = set()
            used_trk: set[int] = set()
            for score, i, j in pairs:
                if score < self._iou:
                    break
                if i in used_det or j in used_trk:
                    continue
                used_det.add(i)
                used_trk.add(j)
                assigned[i] = track_ids[j]

        out_rows: list[list[float]] = []
        seen: set[int] = set()
        for i in range(dets.shape[0]):
            tid = assigned.get(i)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            self._tracks[tid] = (boxes[i].copy(), 0)
            seen.add(tid)
            conf = float(dets[i, 4]) if dets.shape[1] > 4 else 0.0
            cls = float(dets[i, 5]) if dets.shape[1] > 5 else 0.0
            out_rows.append([
                float(boxes[i, 0]), float(boxes[i, 1]),
                float(boxes[i, 2]), float(boxes[i, 3]),
                float(tid), conf, cls,
            ])

        self._age_and_prune(seen)
        return np.array(out_rows, dtype=np.float32)

    def _age_and_prune(self, seen: set[int]) -> None:
        for tid in list(self._tracks.keys()):
            if tid in seen:
                continue
            box, age = self._tracks[tid]
            age += 1
            if age > self._max_age:
                del self._tracks[tid]
            else:
                self._tracks[tid] = (box, age)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ── Track data model ───────────────────────────────────────────────────────────

class TrackState(str, Enum):
    TENTATIVE = "tentative"   # newly created, not yet confirmed
    CONFIRMED = "confirmed"   # seen ≥ min_hits times consecutively
    LOST = "lost"             # no detection match; coasting on Kalman prediction
    REMOVED = "removed"       # coast window expired; track is gone


@dataclass
class Track:
    """One tracked person.  Snapshot from a single ``Tracker.update()`` call."""

    track_id: int
    bbox: BBox
    conf: float
    state: TrackState
    age: int                          # total frames since first seen
    hits: int                         # total detection matches (not counting lost frames)
    velocity: tuple[float, float]     # (vx, vy) pixels/frame, estimated from consecutive centres


# ── Tracker type ──────────────────────────────────────────────────────────────

class TrackerType(str, Enum):
    BOTSORT = "botsort"
    DEEPOCSORT = "deepocsort"
    BYTETRACK = "bytetrack"


# ── BoxMOT factory ────────────────────────────────────────────────────────────

def _create_boxmot_tracker(
    tracker_type: TrackerType,
    reid_weights: Path | None,
    device: str,
    fps: float,
    max_age: int,
) -> Any:
    """Instantiate the requested BoxMOT tracker.

    Targets the BoxMOT ≥ 11 API (tested against 19.x): tracker classes live in
    ``boxmot.trackers`` (not the top-level package), and constructors take
    ``reid_model`` + ``with_reid`` rather than the old ``reid_weights`` /
    ``device`` / ``half`` / ``per_class`` / ``max_age`` arguments.

    ``reid_weights`` is optional for all trackers; if absent, appearance ReID is
    disabled (motion-only tracking) — the trackers still run on boxes alone.

    When ``boxmot`` is not installed (or its API is incompatible) callers fall
    back to the built-in :class:`_SimpleIoUTracker` so detection/tracking still
    runs on a minimal install — it never hard-fails.
    """
    if not _probe_boxmot():
        _log_fallback_tracker_once()
        return _SimpleIoUTracker(max_age=max_age)

    buffer = int(max_age)
    rate = max(1, int(fps))

    def _instantiate(use_reid: bool) -> Any:
        """Build the tracker for the active BoxMOT API.  ``use_reid`` toggles
        appearance ReID (weights download + extra inference)."""
        weights = reid_weights if use_reid else None
        # BoxMOT ≥ 11 exposes tracker classes from ``boxmot.trackers`` with a
        # ``reid_model`` / ``with_reid`` constructor.  Older 10.x releases
        # exported them at the top level with ``reid_weights`` / ``device``.
        try:
            from boxmot.trackers import BotSort, ByteTrack, DeepOcSort  # noqa: PLC0415

            if tracker_type == TrackerType.BYTETRACK:
                return ByteTrack(
                    track_thresh=0.45, match_thresh=0.8,
                    track_buffer=buffer, frame_rate=rate,
                )
            if tracker_type == TrackerType.DEEPOCSORT:
                return DeepOcSort(reid_model=weights, embedding_off=not use_reid)
            return BotSort(
                reid_model=weights, with_reid=use_reid,
                track_buffer=buffer, frame_rate=rate, cmc_method="ecc",
            )
        except ImportError:
            import boxmot  # noqa: PLC0415

            if tracker_type == TrackerType.BYTETRACK:
                return boxmot.ByteTrack(
                    track_thresh=0.45, match_thresh=0.8,
                    track_buffer=buffer, frame_rate=rate,
                )
            if tracker_type == TrackerType.DEEPOCSORT:
                return boxmot.DeepOcSort(
                    reid_weights=weights, device=device,
                    half=False, per_class=False, max_age=max_age,
                )
            return boxmot.BotSort(
                reid_weights=weights, device=device,
                half=False, per_class=False, max_age=max_age,
            )

    want_reid = reid_weights is not None
    try:
        tracker = _instantiate(want_reid)
    except Exception:  # noqa: BLE001
        if want_reid:
            # ReID weights/download failed — fall back to motion-only so tracking
            # still works (never let an optional appearance model kill tracking).
            log.warning("ReID init failed; falling back to motion-only tracking.",
                        exc_info=True)
            tracker = _instantiate(False)
        else:
            raise
    # BoxMOT re-raises its logger to INFO during construction; re-silence it now
    # so the per-frame ECC warnings don't reach the console.
    _quiet_boxmot_logging()
    return tracker


# ── Track-state machine ────────────────────────────────────────────────────────

@dataclass
class _TrackRecord:
    """Mutable bookkeeping for one track across frames."""
    age: int = 0
    hits: int = 0
    frames_lost: int = 0
    prev_cx: float = 0.0
    prev_cy: float = 0.0
    last_bbox: BBox | None = None
    last_conf: float = 0.0


# ── Tracker ───────────────────────────────────────────────────────────────────

class Tracker:
    """Wraps a BoxMOT tracker and manages track lifecycle + velocity.

    Args:
        tracker_type:  ``"botsort"`` (default), ``"deepocsort"``, ``"bytetrack"``.
        reid_weights:  Path to an OSNet ONNX / PT weight file (Phase 4 adds this).
        min_hits:      Consecutive detections before TENTATIVE → CONFIRMED.
        coast_window:  Seconds to keep a lost track before marking REMOVED.
        device:        Torch/ORT device string (``"cpu"``, ``"cuda:0"``, …).
        _impl:         Pre-built tracker implementation (for tests/CI).
    """

    def __init__(
        self,
        tracker_type: TrackerType | str = TrackerType.BOTSORT,
        reid_weights: Path | None = None,
        min_hits: int = 1,
        coast_window: float = 1.5,
        device: str = "cpu",
        _impl: Any = None,
    ) -> None:
        self._tracker_type = TrackerType(tracker_type)
        self._reid_weights = reid_weights
        self._min_hits = min_hits
        self._coast_window = coast_window
        self._device = device

        # Resolved on first update() when fps is known
        self._fps: float = 30.0
        self._coast_max_frames: int = max(1, int(coast_window * 30.0))

        if _impl is not None:
            self._impl = _impl
        else:
            # Deferred: created on first update() so fps is available
            self._impl = None
            self._impl_pending = True

        # Per-track bookkeeping (this wrapper, not BoxMOT internal state)
        self._records: dict[int, _TrackRecord] = {}
        # Tracks currently in LOST state: id → (last_bbox, last_conf, frames_lost)
        self._lost: dict[int, _TrackRecord] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        detections: list[Detection],
        frame: NDArray[np.uint8],
        fps: float = 30.0,
    ) -> list[Track]:
        """Update the tracker for one frame.

        Args:
            detections: Output of ``PersonDetector.detect()``; may be ``[]``
                        on non-inference frames (tracker uses Kalman prediction).
            frame:      BGR frame (used by BoT-SORT CMC and ReID crops).
            fps:        Current ingest fps; used to convert coast_window to frames.

        Returns a list of all non-REMOVED tracks (TENTATIVE, CONFIRMED, LOST).
        """
        self._fps = fps
        self._coast_max_frames = max(1, int(self._coast_window * fps))

        # Lazy init now that fps is known
        if getattr(self, "_impl_pending", False) and self._impl is None:
            self._impl = _create_boxmot_tracker(
                self._tracker_type,
                self._reid_weights,
                self._device,
                fps,
                self._coast_max_frames,
            )
            self._impl_pending = False

        dets_np = detections_to_numpy(detections)  # [N, 6]
        raw = self._impl.update(dets_np, frame)
        # BoxMOT ≥ 11 returns a ``TrackResults`` object; older versions and the
        # built-in fallback return a plain ndarray.  Coerce to ndarray either way
        # so ``_reconcile`` can index/iterate uniformly.
        raw_tracks: NDArray[np.float32] = np.asarray(raw, dtype=np.float32)

        return self._reconcile(raw_tracks)

    def reset(self) -> None:
        """Clear all track state (use on source change)."""
        if self._impl is not None and hasattr(self._impl, "reset"):
            self._impl.reset()
        self._records.clear()
        self._lost.clear()
        if hasattr(self, "_impl_pending"):
            self._impl = None
            self._impl_pending = True

    @property
    def active_count(self) -> int:
        """Number of currently active (non-REMOVED) tracks."""
        return len(self._records) + len(self._lost)

    # ── Internal reconciliation ────────────────────────────────────────────────

    def _reconcile(self, raw: NDArray[np.float32]) -> list[Track]:
        """Reconcile BoxMOT output with our lifecycle records."""

        # Normalise output to [N, 7] minimum
        if raw is None or len(raw) == 0:
            active_ids: set[int] = set()
        else:
            raw = np.atleast_2d(raw)
            if raw.shape[1] < 7:
                # Pad with zeros if fewer columns than expected
                pad = np.zeros((raw.shape[0], 7 - raw.shape[1]), dtype=np.float32)
                raw = np.concatenate([raw, pad], axis=1)
            active_ids = {int(r[4]) for r in raw}

        # Advance age for all existing records
        for rec in self._records.values():
            rec.age += 1

        # Process tracks returned by BoxMOT
        tracks_out: list[Track] = []
        seen_ids: set[int] = set()

        for row in ([] if len(active_ids) == 0 else raw):
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            tid = int(row[4])
            conf = float(row[5])

            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            seen_ids.add(tid)

            if tid not in self._records:
                # New track — may have been lost before (re-acquired)
                if tid in self._lost:
                    rec = self._lost.pop(tid)
                    rec.frames_lost = 0
                else:
                    # age=1: this is its first frame
                    rec = _TrackRecord(age=1, prev_cx=bbox.cx, prev_cy=bbox.cy)
                self._records[tid] = rec

            rec = self._records[tid]
            rec.hits += 1
            rec.frames_lost = 0

            vx = bbox.cx - rec.prev_cx
            vy = bbox.cy - rec.prev_cy
            rec.prev_cx = bbox.cx
            rec.prev_cy = bbox.cy
            rec.last_bbox = bbox
            rec.last_conf = conf

            state = (
                TrackState.CONFIRMED if rec.hits >= self._min_hits else TrackState.TENTATIVE
            )

            tracks_out.append(Track(
                track_id=tid,
                bbox=bbox,
                conf=conf,
                state=state,
                age=rec.age,
                hits=rec.hits,
                velocity=(vx, vy),
            ))

        # Move tracks that disappeared this frame from active → lost
        gone_ids = set(self._records) - seen_ids
        for tid in gone_ids:
            rec = self._records.pop(tid)
            rec.frames_lost = 1
            self._lost[tid] = rec

        # Advance coast counter; emit LOST tracks; prune REMOVED
        to_remove: list[int] = []
        for tid, rec in self._lost.items():
            if rec.frames_lost > self._coast_max_frames:
                to_remove.append(tid)
                continue
            rec.frames_lost += 1
            if rec.last_bbox is not None:
                tracks_out.append(Track(
                    track_id=tid,
                    bbox=rec.last_bbox,
                    conf=rec.last_conf,
                    state=TrackState.LOST,
                    age=rec.age,
                    hits=rec.hits,
                    velocity=(0.0, 0.0),
                ))

        for tid in to_remove:
            del self._lost[tid]
            log.debug("track %d REMOVED (coast expired)", tid)

        return tracks_out
