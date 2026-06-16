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
        except ImportError:
            _BOXMOT_AVAILABLE = False
    return bool(_BOXMOT_AVAILABLE)


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

    ``reid_weights`` is optional for all trackers; if absent, appearance
    ReID is skipped (motion-only tracking).  ReID models are added in
    Phase 4.
    """
    if not _probe_boxmot():
        raise ImportError(
            "boxmot is required for Tracker: pip install boxmot\n"
            "See requirements/base.txt for the pinned version."
        )

    import boxmot  # noqa: PLC0415

    # boxmot resolves None reid_weights gracefully for ByteTrack;
    # for BoT-SORT / DeepOCSORT it disables appearance association.
    rw = reid_weights  # may be None

    if tracker_type == TrackerType.BYTETRACK:
        return boxmot.ByteTrack(
            track_thresh=0.45,
            match_thresh=0.8,
            track_buffer=max_age,
            frame_rate=int(fps),
        )
    if tracker_type == TrackerType.DEEPOCSORT:
        return boxmot.DeepOcSort(
            reid_weights=rw,
            device=device,
            half=False,
            per_class=False,
            max_age=max_age,
        )
    # Default: BoT-SORT with CMC
    return boxmot.BotSort(
        reid_weights=rw,
        device=device,
        half=False,
        per_class=False,
        max_age=max_age,
    )


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
        raw_tracks: NDArray[np.float32] = self._impl.update(dets_np, frame)

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
