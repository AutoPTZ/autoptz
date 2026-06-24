"""CPU subservice governor — adaptive cadence + phase-stagger tests.

Tests the quality-scale helpers and one-time phase-stagger seeding added to
CameraWorker.  No real threads are started — the worker is constructed but
never ``start()``-ed.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoptz.engine.camera_worker import (
    _FACE_INTERVAL_S,
    _POSE_INTERVAL_S,
    _REID_INTERVAL_S,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────


class _FakeSource:
    """Minimal frame source — never called in these tests."""

    def open(self) -> bool:
        return True

    def read(self):
        return np.full((240, 320, 3), 100, dtype=np.uint8)

    def close(self) -> None:
        pass


@pytest.fixture()
def worker():
    from autoptz.config.models import CameraConfig, SourceConfig
    from autoptz.engine.camera_worker import CameraWorker

    config = CameraConfig(
        id="govtest01abcd",
        name="Gov",
        source=SourceConfig(type="usb", address="usb://0"),
    )
    w = CameraWorker(
        "govtest01abcd",
        config,
        lambda _m: None,
        frame_source=_FakeSource(),
    )
    # Worker is intentionally NOT started — pure attribute inspection.
    return w


# ─────────────────────────────────────────────────────────────────────────────
# _quality_scale
# ─────────────────────────────────────────────────────────────────────────────


class TestQualityScale:
    def test_high_returns_1_0(self, worker) -> None:
        worker._quality_active = "high"
        assert worker._quality_scale() == pytest.approx(1.0)

    def test_balanced_returns_1_25(self, worker) -> None:
        worker._quality_active = "balanced"
        assert worker._quality_scale() == pytest.approx(1.25)

    def test_auto_returns_1_0(self, worker) -> None:
        worker._quality_active = "auto"
        assert worker._quality_scale() == pytest.approx(1.0)

    def test_low_returns_2_0(self, worker) -> None:
        worker._quality_active = "low"
        assert worker._quality_scale() == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Effective interval getters
# ─────────────────────────────────────────────────────────────────────────────


class TestEffectiveIntervals:
    @pytest.mark.parametrize(
        "quality, scale",
        [
            ("high", 1.0),
            ("balanced", 1.25),
            ("auto", 1.0),
            ("low", 2.0),
        ],
    )
    def test_face_interval(self, worker, quality, scale) -> None:
        worker._quality_active = quality
        assert worker._effective_face_interval() == pytest.approx(_FACE_INTERVAL_S * scale)

    @pytest.mark.parametrize(
        "quality, scale",
        [
            ("high", 1.0),
            ("balanced", 1.25),
            ("auto", 1.0),
            ("low", 2.0),
        ],
    )
    def test_reid_interval(self, worker, quality, scale) -> None:
        worker._quality_active = quality
        assert worker._effective_reid_interval() == pytest.approx(_REID_INTERVAL_S * scale)

    @pytest.mark.parametrize(
        "quality, scale",
        [
            ("high", 1.0),
            ("balanced", 1.25),
            ("auto", 1.0),
            ("low", 2.0),
        ],
    )
    def test_pose_interval(self, worker, quality, scale) -> None:
        worker._quality_active = quality
        assert worker._effective_pose_interval() == pytest.approx(_POSE_INTERVAL_S * scale)


# ─────────────────────────────────────────────────────────────────────────────
# Regression: high quality == base constants exactly
# ─────────────────────────────────────────────────────────────────────────────


class TestHighQualityRegression:
    """At high quality the effective intervals must equal the base constants."""

    def test_face_equals_base(self, worker) -> None:
        worker._quality_active = "high"
        assert worker._effective_face_interval() == _FACE_INTERVAL_S

    def test_reid_equals_base(self, worker) -> None:
        worker._quality_active = "high"
        assert worker._effective_reid_interval() == _REID_INTERVAL_S

    def test_pose_equals_base(self, worker) -> None:
        worker._quality_active = "high"
        assert worker._effective_pose_interval() == _POSE_INTERVAL_S


# ─────────────────────────────────────────────────────────────────────────────
# Phase-stagger: _seed_subservice_phases
# ─────────────────────────────────────────────────────────────────────────────


class TestPhaseStagger:
    def test_seed_sets_reid_offset(self, worker) -> None:
        """_last_reid_t is set to now - _REID_INTERVAL_S * 0.5."""
        worker._phase_seeded = False
        now = 1000.0
        worker._seed_subservice_phases(now)
        assert worker._last_reid_t == pytest.approx(now - _REID_INTERVAL_S * 0.5)

    def test_seed_sets_pose_offset(self, worker) -> None:
        """_last_pose_t is set to now - _POSE_INTERVAL_S * 0.33."""
        worker._phase_seeded = False
        now = 1000.0
        worker._seed_subservice_phases(now)
        assert worker._last_pose_t == pytest.approx(now - _POSE_INTERVAL_S * 0.33)

    def test_seed_marks_seeded(self, worker) -> None:
        worker._phase_seeded = False
        worker._seed_subservice_phases(1000.0)
        assert worker._phase_seeded is True

    def test_seed_is_idempotent(self, worker) -> None:
        """Calling seed a second time must NOT overwrite the timestamps."""
        worker._phase_seeded = False
        now = 1000.0
        worker._seed_subservice_phases(now)
        # Advance time and call again — values must not change.
        worker._seed_subservice_phases(now + 999.0)
        assert worker._last_reid_t == pytest.approx(now - _REID_INTERVAL_S * 0.5)
        assert worker._last_pose_t == pytest.approx(now - _POSE_INTERVAL_S * 0.33)

    def test_reid_becomes_due_before_face(self, worker) -> None:
        """After seeding, reid fires ~0.125 s before face (half-period offset).

        Both are seeded at the same real ``now``.  Face has _last_face_t=0,
        so it becomes due at now (well in the past already).  ReID was seeded
        with _last_reid_t = now - 0.5 * _REID_INTERVAL_S, so it becomes due at
        now + 0.5 * _REID_INTERVAL_S (= 0.125 s later).

        The test confirms the timing arithmetic rather than waiting real time.
        """
        # Seed with last_face_t at 0 (default) and now = 1000.0.
        worker._phase_seeded = False
        now = 1000.0
        worker._seed_subservice_phases(now)

        # face is already due because _last_face_t==0 (>> interval ago)
        face_elapsed = now - worker._last_face_t  # 1000 s >> 0.25 s → due
        reid_elapsed = now - worker._last_reid_t  # exactly 0.5 * interval → not yet due

        assert face_elapsed > _FACE_INTERVAL_S, "face should already be overdue"
        assert reid_elapsed < _REID_INTERVAL_S, "reid should not yet be due immediately after seed"

        # The seeding must not touch the face timestamp — it stays at its default 0.0.
        assert worker._last_face_t == 0.0

        # The next reid fire-time is last_reid_t + _REID_INTERVAL_S
        reid_next_due = worker._last_reid_t + _REID_INTERVAL_S
        # That is now + 0.5 * _REID_INTERVAL_S = 1000.125
        assert reid_next_due == pytest.approx(now + _REID_INTERVAL_S * 0.5)

    def test_pose_becomes_due_after_seed(self, worker) -> None:
        """After seeding, pose fires ~0.066 s after the seed moment.

        Pose is seeded with _last_pose_t = now - 0.33 * _POSE_INTERVAL_S, so
        the next due time is last_pose_t + _POSE_INTERVAL_S
                           = now - 0.33*_POSE_INTERVAL_S + _POSE_INTERVAL_S
                           = now + 0.67 * _POSE_INTERVAL_S
        With the default _POSE_INTERVAL_S=0.2 that is now + 0.134 s.
        """
        worker._phase_seeded = False
        now = 1000.0
        worker._seed_subservice_phases(now)

        pose_elapsed = now - worker._last_pose_t  # exactly 0.33 * interval → not yet due
        assert pose_elapsed < _POSE_INTERVAL_S, "pose should not yet be due immediately after seed"

        # The next pose fire-time is last_pose_t + _POSE_INTERVAL_S
        pose_next_due = worker._last_pose_t + _POSE_INTERVAL_S
        # That is now + (1 - 0.33) * _POSE_INTERVAL_S = now + 0.67 * _POSE_INTERVAL_S
        assert pose_next_due == pytest.approx(now + _POSE_INTERVAL_S * 0.67)


# ─────────────────────────────────────────────────────────────────────────────
# Ego-motion decimation: _update_ego_motion cadence gate
# ─────────────────────────────────────────────────────────────────────────────


def _worker(**ptz):
    from autoptz.config.models import CameraConfig, PTZConfig, SourceConfig, TrackingConfig

    cfg = CameraConfig(
        id="cam-cpu-000001",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(),
        ptz=PTZConfig(**ptz),
    )
    from autoptz.engine.camera_worker import CameraWorker

    return CameraWorker("cam-cpu-000001", cfg, on_telemetry=lambda m: None)


def test_ego_runs_every_nth_frame():
    w = _worker(ego_comp_enabled=True, ego_comp_interval=3)
    calls = {"n": 0}

    class _Est:
        def estimate(self, *a, **k):
            calls["n"] += 1
            from autoptz.engine.pipeline.egomotion import EgoMotion

            return EgoMotion(vx=0.1, vy=0.0, source="flow", confidence=1.0)

    w._ego_estimator = _Est()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    for i in range(6):
        w._frames_inferred = i  # drives the decimation phase
        w._update_ego_motion([], frame, float(i))
    # 6 frames, interval 3 → flow ran on frames 0 and 3 only.
    assert calls["n"] == 2
    # Between runs the estimate is reused (non-zero), not blanked to "none".
    assert w._ego_source == "flow"


# ─────────────────────────────────────────────────────────────────────────────
# Stage-spread: pose skips on frames where the detector ran
# ─────────────────────────────────────────────────────────────────────────────


def test_pose_skips_on_a_detect_frame():
    from autoptz.config.models import CameraConfig, PTZConfig, SourceConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(
        id="cam-cpu-000002",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(stage_spread=True),
        ptz=PTZConfig(),
    )
    w = CameraWorker("cam-cpu-000002", cfg, on_telemetry=lambda m: None)
    w._detected_this_tick = True
    assert w._pose_allowed_this_tick() is False
    w._detected_this_tick = False
    assert w._pose_allowed_this_tick() is True


# ─────────────────────────────────────────────────────────────────────────────
# Amortized cost + hysteresis
# ─────────────────────────────────────────────────────────────────────────────


def test_amortized_cost_divides_detect_by_interval():
    from autoptz.config.models import CameraConfig, PTZConfig, SourceConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(
        id="cam-cpu-000003",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(),
        ptz=PTZConfig(),
    )
    w = CameraWorker("cam-cpu-000003", cfg, on_telemetry=lambda m: None)
    w._quality_interval = 4
    w._stage_samples_override = {"detect": 40.0, "track": 2.0, "face": 0.0, "pose": 0.0}
    w._stage_avg = lambda k: w._stage_samples_override.get(k, 0.0)
    # Detect amortized over interval 4 → 10ms, + track 2ms = 12ms, NOT 42ms.
    assert abs(w._amortized_cost_ms() - 12.0) < 0.51


# ─────────────────────────────────────────────────────────────────────────────
# System-CPU-aware governor
# ─────────────────────────────────────────────────────────────────────────────


def test_high_system_cpu_relaxes_even_when_local_cost_low():
    from autoptz.config.models import CameraConfig, PTZConfig, SourceConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(
        id="cam-cpu-000004",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(),
        ptz=PTZConfig(),
    )
    w = CameraWorker("cam-cpu-000004", cfg, on_telemetry=lambda m: None)
    w._stage_avg = lambda k: 1.0  # trivially low local cost
    w.set_system_cpu_pressure(95.0)
    w._effective_detect_interval()
    assert w._quality_active in ("balanced", "low")
