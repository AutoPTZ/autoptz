#!/usr/bin/env python3
"""Detection + tracking benchmark on a recorded video clip.

Usage::

    # With a real YOLO26 model
    python tools/bench/track_clip.py video.mp4 --model autoptz/models/yolo26n.onnx

    # Synthetic model (no model file needed — for CI / quick sanity checks)
    python tools/bench/track_clip.py video.mp4 --synthetic

    # Override tracker
    python tools/bench/track_clip.py video.mp4 --model yolo26n.onnx --tracker bytetrack

    # Write annotated output
    python tools/bench/track_clip.py video.mp4 --model yolo26n.onnx --output out.mp4

Press Ctrl+C to stop early; partial results are still reported.

Reported metrics
----------------
- ``avg_det_fps``   Average detections per second (ORT inference only).
- ``avg_total_fps`` Wall-clock throughput including decode + postprocess.
- ``n_unique_ids``  Total distinct track IDs seen (lower ≈ more stable).
- ``id_switches``   Heuristic ID-switch count (new ID whose bbox overlaps
                    a previously-seen track by > ``--iou-thr``).
- ``tracks_1s``     Tracks that lived ≥ ``--min-track-s`` seconds (stable).
- ``avg_tracks``    Average active (non-REMOVED) tracks per frame.
- ``occlusion_recoveries`` Tracks re-detected after ≥ 1 frame of LOST state.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
log = logging.getLogger("bench.track_clip")


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def run_clip(
    video_path: str | Path,
    model_path: str | Path | None,
    synthetic: bool = False,
    tracker_type: str = "botsort",
    detect_interval: int = 1,
    conf_threshold: float = 0.25,
    input_size: int = 640,
    min_track_s: float = 1.0,
    iou_thr: float = 0.5,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
) -> dict[str, float]:
    from autoptz.engine.pipeline.detect import (  # noqa: PLC0415
        PersonDetector,
        make_synthetic_detector_session,
    )
    from autoptz.engine.pipeline.track import Tracker  # noqa: PLC0415

    # ── Build detector ─────────────────────────────────────────────────────────
    if synthetic:
        log.info("Using synthetic detector (no real model)")
        session = make_synthetic_detector_session(input_size=input_size)
        detector = PersonDetector(
            _session=session,
            input_size=input_size,
            conf_threshold=conf_threshold,
            detect_interval=detect_interval,
        )
    elif model_path is not None:
        detector = PersonDetector(
            model_path=model_path,
            input_size=input_size,
            conf_threshold=conf_threshold,
            detect_interval=detect_interval,
        )
    else:
        log.error("Provide --model or --synthetic")
        sys.exit(1)

    log.info("Detector EP: %s", detector.ep)

    # ── Build tracker ──────────────────────────────────────────────────────────
    tracker = Tracker(tracker_type=tracker_type, min_hits=1, coast_window=1.5)
    log.info("Tracker: %s", tracker_type)

    # ── Open video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        sys.exit(1)

    source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Source: %s  fps=%.1f  frames=%d", video_path, source_fps, total_frames)

    # ── Output writer ──────────────────────────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    if output_path:
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (W, H))

    # ── Bookkeeping ────────────────────────────────────────────────────────────
    # id → (first_frame, last_frame, last_bbox)
    id_records: dict[int, dict] = {}
    # ids currently in LOST state last frame
    prev_lost_ids: set[int] = set()
    # prev xyxy per id (for ID-switch heuristic)
    prev_bboxes: dict[int, tuple[float, float, float, float]] = {}
    all_ids_seen: set[int] = set()
    id_switches = 0
    occlusion_recoveries = 0

    frame_num = 0
    det_times: list[float] = []
    total_times: list[float] = []
    tracks_per_frame: list[int] = []
    min_track_frames = int(min_track_s * source_fps)

    try:
        while True:
            t_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame_num += 1
            if max_frames and frame_num > max_frames:
                break

            # Detect
            t_det = time.perf_counter()
            dets = detector.detect(frame)
            det_elapsed = time.perf_counter() - t_det
            if dets or (frame_num % detect_interval == 0):
                det_times.append(det_elapsed)

            # Track
            tracks = tracker.update(dets, frame, fps=source_fps)
            total_times.append(time.perf_counter() - t_start)

            # ── Metrics per frame ──────────────────────────────────────────────
            active = [t for t in tracks if t.state.value in ("tentative", "confirmed")]
            lost_now = {t.track_id for t in tracks if t.state.value == "lost"}

            tracks_per_frame.append(len(active))

            for t in active:
                tid = t.track_id
                bb = t.bbox.as_xyxy()

                if tid not in all_ids_seen:
                    # New ID — check if it overlaps a previous track (heuristic ID switch)
                    for prev_id, prev_bb in list(prev_bboxes.items()):
                        if prev_id not in {tx.track_id for tx in active} and _iou(bb, prev_bb) > iou_thr:
                            id_switches += 1
                            break
                    all_ids_seen.add(tid)

                if tid in prev_lost_ids:
                    occlusion_recoveries += 1

                if tid not in id_records:
                    id_records[tid] = {"first": frame_num, "last": frame_num}
                else:
                    id_records[tid]["last"] = frame_num

                prev_bboxes[tid] = bb

            prev_lost_ids = lost_now

            # ── Annotate output ────────────────────────────────────────────────
            if writer is not None:
                _annotate(frame, tracks)
                writer.write(frame)

            if frame_num % 100 == 0:
                fps_so_far = frame_num / sum(total_times) if total_times else 0.0
                log.info(
                    "frame=%d  ids=%d  switches=%d  fps=%.1f",
                    frame_num, len(all_ids_seen), id_switches, fps_so_far,
                )

    except KeyboardInterrupt:
        log.info("Stopped early at frame %d", frame_num)

    cap.release()
    if writer:
        writer.release()

    # ── Compute final metrics ──────────────────────────────────────────────────
    avg_det_fps = (1.0 / (sum(det_times) / len(det_times))) if det_times else 0.0
    avg_total_fps = frame_num / sum(total_times) if total_times else 0.0
    stable_tracks = sum(
        1 for rec in id_records.values()
        if (rec["last"] - rec["first"]) >= min_track_frames
    )

    metrics = {
        "frames": float(frame_num),
        "avg_det_fps": avg_det_fps,
        "avg_total_fps": avg_total_fps,
        "n_unique_ids": float(len(all_ids_seen)),
        "id_switches": float(id_switches),
        "tracks_1s": float(stable_tracks),
        "avg_tracks": float(np.mean(tracks_per_frame)) if tracks_per_frame else 0.0,
        "occlusion_recoveries": float(occlusion_recoveries),
    }
    return metrics


def _annotate(frame: np.ndarray, tracks: list) -> None:
    from autoptz.engine.pipeline.track import TrackState  # noqa: PLC0415
    _COLOURS = {
        TrackState.CONFIRMED: (0, 220, 0),
        TrackState.TENTATIVE: (200, 200, 0),
        TrackState.LOST: (0, 100, 220),
    }
    for t in tracks:
        x1, y1, x2, y2 = int(t.bbox.x1), int(t.bbox.y1), int(t.bbox.x2), int(t.bbox.y2)
        colour = _COLOURS.get(t.state, (128, 128, 128))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"#{t.track_id} {t.state.value[0].upper()} {t.conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def _print_report(metrics: dict[str, float], path: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  Benchmark: {path}")
    print(f"{'─' * 55}")
    print(f"  Frames processed     : {int(metrics['frames'])}")
    print(f"  Avg detect fps       : {metrics['avg_det_fps']:.1f}")
    print(f"  Avg total fps        : {metrics['avg_total_fps']:.1f}")
    print(f"  Unique track IDs     : {int(metrics['n_unique_ids'])}")
    print(f"  Heuristic ID switches: {int(metrics['id_switches'])}")
    print(f"  Stable tracks (≥1 s) : {int(metrics['tracks_1s'])}")
    print(f"  Avg active tracks/fr : {metrics['avg_tracks']:.1f}")
    print(f"  Occlusion recoveries : {int(metrics['occlusion_recoveries'])}")
    print(f"{'─' * 55}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="AutoPTZ v2 — detect + track benchmark on a video clip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", help="YOLO26 ONNX model path")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic constant-output model (no model file needed)")
    parser.add_argument("--tracker", default="botsort",
                        choices=["botsort", "deepocsort", "bytetrack"],
                        help="BoxMOT tracker (default: botsort)")
    parser.add_argument("--detect-interval", type=int, default=1,
                        help="Run detection every N frames (default: 1)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--input-size", type=int, default=640,
                        help="Model input size in pixels (default: 640)")
    parser.add_argument("--min-track-s", type=float, default=1.0,
                        help="Min seconds for a track to count as stable (default: 1.0)")
    parser.add_argument("--iou-thr", type=float, default=0.5,
                        help="IoU threshold for ID-switch detection (default: 0.5)")
    parser.add_argument("--output", help="Write annotated video to this path")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (useful for quick checks)")

    args = parser.parse_args(argv)

    metrics = run_clip(
        video_path=args.video,
        model_path=args.model,
        synthetic=args.synthetic,
        tracker_type=args.tracker,
        detect_interval=args.detect_interval,
        conf_threshold=args.conf,
        input_size=args.input_size,
        min_track_s=args.min_track_s,
        iou_thr=args.iou_thr,
        output_path=args.output,
        max_frames=args.max_frames,
    )

    _print_report(metrics, args.video)


if __name__ == "__main__":
    main()
