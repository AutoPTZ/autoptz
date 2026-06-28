"""Unit tests for engine-reported quality metrics (pure, no Qt).

These verify ``PerCameraQualityAccumulator`` / ``QualityMetrics`` and the
``StepResult`` / ``BenchmarkResult`` JSON shape with quality riding along.  All
math is fed synthetic ``TelemetryMsg`` streams built from real engine messages,
so it is verified with no inference and no event loop.
"""

from __future__ import annotations

import json

from autoptz.benchmark.profiles import get_profile
from autoptz.benchmark.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    PerCameraQualityAccumulator,
    QualityMetrics,
    StepResult,
)
from autoptz.engine.runtime.messages import BBox, TelemetryMsg, TrackInfo


def _bbox() -> BBox:
    return BBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)


def _msg(
    seq: int,
    *,
    fps: float = 30.0,
    tracks: list[TrackInfo] | None = None,
    dropped_frames: int = 0,
) -> TelemetryMsg:
    return TelemetryMsg(
        camera_id="cam-a",
        seq=seq,
        fps=fps,
        dropped_frames=dropped_frames,
        tracks=tracks or [],
    )


def _target(
    track_id: int,
    *,
    lost: bool = False,
    confidence: float = 0.9,
    identity_id: str | None = None,
) -> TrackInfo:
    return TrackInfo(
        track_id=track_id,
        bbox=_bbox(),
        identity_id=identity_id,
        confidence=confidence,
        is_target=True,
        lost=lost,
    )


def _non_target(track_id: int, *, confidence: float = 0.5) -> TrackInfo:
    return TrackInfo(
        track_id=track_id,
        bbox=_bbox(),
        confidence=confidence,
        is_target=False,
        lost=False,
    )


class TestAcquireLostReacquireIdSwitch:
    """A scripted lifecycle: acquire@10, lost@25, reacquire@35, id-switch@50."""

    def _run_stream(self) -> QualityMetrics:
        fps = 10.0  # 1 frame == 0.1 s, keeps the timing arithmetic exact
        acc = PerCameraQualityAccumulator(fps_hint=fps)
        # Frames 1..9: no target present yet.
        for seq in range(1, 10):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_non_target(99)]))
        # Frame 10: first acquire (target present + not lost).
        for seq in range(10, 25):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(5)]))
        # Frames 25..34: target LOST (10 frames -> 1.0 s lost).
        for seq in range(25, 35):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(5, lost=True)]))
        # Frame 35: reacquire (same track id 5).
        for seq in range(35, 50):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(5)]))
        # Frame 50+: id switch 5 -> 6 (still target, not lost).
        for seq in range(50, 60):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(6)]))
        return acc.finalize()

    def test_time_to_first_acquire(self) -> None:
        q = self._run_stream()
        # 9 non-target frames observed before acquire at frame 10 -> 0.9 s @ 10 fps.
        assert q.time_to_first_acquire_s is not None
        assert abs(q.time_to_first_acquire_s - 0.9) < 1e-6

    def test_lost_event_and_durations(self) -> None:
        q = self._run_stream()
        assert q.lost_event_count == 1
        # 10 lost frames @ 10 fps -> 1.0 s total and longest.
        assert abs(q.total_lost_duration_s - 1.0) < 1e-6
        assert abs(q.longest_lost_duration_s - 1.0) < 1e-6

    def test_reacquire_count(self) -> None:
        q = self._run_stream()
        assert q.reacquire_count == 1

    def test_id_switch_count(self) -> None:
        q = self._run_stream()
        assert q.id_switch_count == 1


class TestTotalVsLongestLost:
    def test_two_lost_events_total_and_longest(self) -> None:
        fps = 10.0
        acc = PerCameraQualityAccumulator(fps_hint=fps)
        seq = 1
        # Acquire.
        for _ in range(5):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(1)]))
            seq += 1
        # Short lost: 2 frames -> 0.2 s.
        for _ in range(2):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(1, lost=True)]))
            seq += 1
        # Found.
        for _ in range(5):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(1)]))
            seq += 1
        # Long lost: 4 frames -> 0.4 s (the longest single event).
        for _ in range(4):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(1, lost=True)]))
            seq += 1
        # Found again.
        for _ in range(3):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(1)]))
            seq += 1
        q = acc.finalize()
        assert q.lost_event_count == 2
        assert abs(q.total_lost_duration_s - 0.6) < 1e-6  # 0.2 + 0.4
        assert abs(q.longest_lost_duration_s - 0.4) < 1e-6
        assert q.reacquire_count == 2


class TestLostByDisappearance:
    def test_target_disappearing_from_tracks_counts_as_lost(self) -> None:
        fps = 10.0
        acc = PerCameraQualityAccumulator(fps_hint=fps)
        seq = 1
        for _ in range(4):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(7)]))
            seq += 1
        # Target track vanishes from msg.tracks entirely (no is_target track) -> lost.
        for _ in range(3):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_non_target(99)]))
            seq += 1
        # It comes back.
        for _ in range(4):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(7)]))
            seq += 1
        q = acc.finalize()
        assert q.lost_event_count == 1
        assert abs(q.total_lost_duration_s - 0.3) < 1e-6  # 3 vanished frames
        assert q.reacquire_count == 1


class TestIdSwitchOnTrackId:
    def test_track_id_change_5_to_6(self) -> None:
        fps = 30.0
        acc = PerCameraQualityAccumulator(fps_hint=fps)
        acc.on_telemetry(_msg(1, fps=fps, tracks=[_target(5)]))
        acc.on_telemetry(_msg(2, fps=fps, tracks=[_target(5)]))
        acc.on_telemetry(_msg(3, fps=fps, tracks=[_target(6)]))  # switch
        acc.on_telemetry(_msg(4, fps=fps, tracks=[_target(6)]))
        q = acc.finalize()
        assert q.id_switch_count == 1


class TestHoldPct:
    def test_hold_pct_80_of_100(self) -> None:
        fps = 30.0
        acc = PerCameraQualityAccumulator(fps_hint=fps)
        # 80 held frames (target present, not lost) + 20 not-held -> 80%.
        for _ in range(80):
            acc.on_telemetry(_msg(1, fps=fps, tracks=[_target(1)]))
        for _ in range(20):
            acc.on_telemetry(_msg(1, fps=fps, tracks=[_non_target(99)]))
        q = acc.finalize()
        assert abs(q.target_hold_pct - 80.0) < 1e-6


class TestMeanConfidence:
    def test_rolling_mean_skips_non_target_and_lost(self) -> None:
        fps = 30.0
        acc = PerCameraQualityAccumulator(fps_hint=fps)
        # Held frames contribute confidence: 1.0 and 0.6 -> mean 0.8.
        acc.on_telemetry(_msg(1, fps=fps, tracks=[_target(1, confidence=1.0)]))
        acc.on_telemetry(_msg(2, fps=fps, tracks=[_target(1, confidence=0.6)]))
        # A non-target frame (confidence ignored).
        acc.on_telemetry(_msg(3, fps=fps, tracks=[_non_target(99, confidence=0.2)]))
        # A lost target frame (confidence ignored even though is_target).
        acc.on_telemetry(_msg(4, fps=fps, tracks=[_target(1, lost=True, confidence=0.99)]))
        q = acc.finalize()
        assert abs(q.mean_target_confidence - 0.8) < 1e-6


class TestNeverAcquired:
    def test_never_acquired_metrics(self) -> None:
        fps = 30.0
        acc = PerCameraQualityAccumulator(fps_hint=fps)
        for _ in range(20):
            acc.on_telemetry(_msg(1, fps=fps, tracks=[_non_target(99)]))
        q = acc.finalize()
        assert q.time_to_first_acquire_s is None
        assert q.target_hold_pct == 0.0
        assert q.mean_target_confidence == 0.0
        assert q.lost_event_count == 0
        assert q.reacquire_count == 0
        # to_dict emits null for the never-acquired sentinel (JSON-friendly).
        d = q.to_dict()
        assert d["time_to_first_acquire_s"] is None
        assert json.loads(json.dumps(d))["time_to_first_acquire_s"] is None

    def test_zero_frames_is_safe(self) -> None:
        q = PerCameraQualityAccumulator(fps_hint=30.0).finalize()
        assert q.time_to_first_acquire_s is None
        assert q.target_hold_pct == 0.0
        assert q.mean_target_confidence == 0.0


class TestDroppedFramesAndFps:
    def test_dropped_frames_tracked_and_fps_observed(self) -> None:
        fps = 25.0
        acc = PerCameraQualityAccumulator(fps_hint=10.0)  # hint differs from msg.fps
        for seq in range(1, 11):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(1)], dropped_frames=seq))
        q = acc.finalize()
        # Latest cumulative dropped_frames wins.
        assert q.dropped_frames == 10
        # Observed mean fps comes from msg.fps (25), not the hint (10).
        assert abs(q.fps - 25.0) < 1e-6

    def test_fps_falls_back_to_hint_when_msg_has_no_fps(self) -> None:
        acc = PerCameraQualityAccumulator(fps_hint=12.0)
        for seq in range(1, 6):
            acc.on_telemetry(_msg(seq, fps=0.0, tracks=[_target(1)]))
        q = acc.finalize()
        assert abs(q.fps - 12.0) < 1e-6


class TestQualityMetricsToDict:
    def test_to_dict_round_trips_through_json(self) -> None:
        fps = 10.0
        acc = PerCameraQualityAccumulator(fps_hint=fps)
        for seq in range(1, 11):
            acc.on_telemetry(_msg(seq, fps=fps, tracks=[_target(1)]))
        q = acc.finalize()
        assert isinstance(q, QualityMetrics)
        d = q.to_dict()
        round_tripped = json.loads(json.dumps(d))
        # All documented keys present.
        for key in (
            "time_to_first_acquire_s",
            "total_lost_duration_s",
            "longest_lost_duration_s",
            "lost_event_count",
            "reacquire_count",
            "id_switch_count",
            "target_hold_pct",
            "mean_target_confidence",
            "fps",
            "dropped_frames",
        ):
            assert key in round_tripped


# ── StepResult / BenchmarkResult carry the quality dict in to_dict() ──────────


class TestStepResultQuality:
    def test_step_result_to_dict_emits_per_camera_quality(self) -> None:
        q = PerCameraQualityAccumulator(fps_hint=10.0)
        for seq in range(1, 11):
            q.on_telemetry(_msg(seq, fps=10.0, tracks=[_target(1)]))
        quality = {"cam-a": q.finalize().to_dict()}
        step = StepResult(
            cameras=1,
            min_fps=30.0,
            mean_fps=30.0,
            per_camera_fps=[30.0],
            sustained=True,
            per_camera_quality=quality,
        )
        d = step.to_dict()
        assert "per_camera_quality" in d
        assert d["per_camera_quality"]["cam-a"]["target_hold_pct"] == 100.0
        # JSON round-trips.
        json.loads(json.dumps(d))

    def test_step_result_quality_defaults_empty(self) -> None:
        step = StepResult(cameras=1, min_fps=30.0, mean_fps=30.0, per_camera_fps=[30.0])
        assert step.per_camera_quality == {}
        assert step.to_dict()["per_camera_quality"] == {}


class TestBenchmarkResultQuality:
    def test_benchmark_result_to_dict_round_trips_with_quality(self) -> None:
        q = PerCameraQualityAccumulator(fps_hint=10.0)
        for seq in range(1, 11):
            q.on_telemetry(_msg(seq, fps=10.0, tracks=[_target(1)]))
        step = StepResult(
            cameras=1,
            min_fps=30.0,
            mean_fps=30.0,
            per_camera_fps=[30.0],
            sustained=True,
            per_camera_quality={"cam-a": q.finalize().to_dict()},
        )
        result = BenchmarkResult(
            profile="full",
            weight=1.0,
            floor_fps=24.0,
            max_cameras=1,
            sustained_cameras=1,
            min_fps_at_sustained=30.0,
            score=1.0,
            steps=[step],
        )
        d = result.to_dict()
        data = json.loads(json.dumps(d))
        assert data["steps"][0]["per_camera_quality"]["cam-a"]["target_hold_pct"] == 100.0


class TestRunnerWiresQualityReader:
    def test_run_puts_quality_reader_output_into_step(self) -> None:
        prof = get_profile("full")

        def sample_fn(n: int) -> list[float]:
            return [30.0] * n if n < 2 else [10.0] * n

        def quality_reader() -> dict[str, dict]:
            return {"cam-x": {"target_hold_pct": 75.0}}

        runner = BenchmarkRunner(
            prof,
            sample_fn=sample_fn,
            floor_fps=24.0,
            max_cameras=4,
            dwell_s=0.0,
            quality_reader=quality_reader,
        )
        result = runner.run()
        # Every recorded step carries the quality snapshot.
        for step in result.steps:
            assert step.per_camera_quality == {"cam-x": {"target_hold_pct": 75.0}}

    def test_run_without_quality_reader_leaves_empty(self) -> None:
        prof = get_profile("full")
        runner = BenchmarkRunner(
            prof,
            sample_fn=lambda n: [10.0] * n,
            floor_fps=24.0,
            max_cameras=2,
            dwell_s=0.0,
        )
        result = runner.run()
        assert all(s.per_camera_quality == {} for s in result.steps)
