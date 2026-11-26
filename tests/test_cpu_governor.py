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


# ─────────────────────────────────────────────────────────────────────────────
# Auto detector-cadence governor must not oscillate
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoCadenceStability:
    """The measured cost is amortized by the *current* interval, so judging an
    upgrade by the current-cadence cost is self-referential: at interval 4 the
    cost looks cheap, it upgrades to 2, which is over budget, so it drops to 4 —
    field telemetry showed ~7 flips/s (512 in 70 s) on a live NDI camera."""

    @staticmethod
    def _costs(worker, monkeypatch, **ms: float) -> None:
        monkeypatch.setattr(worker, "_stage_avg", lambda key: float(ms.get(key, 0.0)))

    def test_constant_load_settles_without_flapping(self, worker, monkeypatch) -> None:
        # 30 fps → 33.3 ms budget. interval 2: (40/2 + 20)/33.3 = 1.20 (over);
        # interval 4: (40/4 + 20)/33.3 = 0.90 — the old code flipped 4↔2 forever.
        self._costs(worker, monkeypatch, detect=40.0, track=20.0)
        intervals = [worker._effective_detect_interval() for _ in range(30)]
        assert intervals[-1] == 4
        flips = sum(1 for a, b in zip(intervals, intervals[1:], strict=False) if a != b)
        assert flips <= 2, intervals
        quality_events = [e for e in worker._runtime_events if e.kind == "quality"]
        assert len(quality_events) <= 3

    def test_recovers_full_cadence_when_load_drops(self, worker, monkeypatch) -> None:
        self._costs(worker, monkeypatch, detect=40.0, track=20.0)
        for _ in range(5):
            worker._effective_detect_interval()
        assert worker._quality_interval == 4
        self._costs(worker, monkeypatch, detect=10.0, track=3.0)  # (10+3)/33.3 = 0.39
        intervals = [worker._effective_detect_interval() for _ in range(10)]
        assert intervals[-1] == 1
        assert worker._quality_active == "high"

    def test_upgrades_only_when_next_cadence_fits(self, worker, monkeypatch) -> None:
        # At interval 4 cost is 0.6, but interval 2 would be (30/2+18)/33.3 = 0.99:
        # still too close to the 1.10 downgrade line → stay low (no ping-pong).
        self._costs(worker, monkeypatch, detect=30.0, track=18.0)
        worker._quality_interval, worker._quality_active = 4, "low"
        assert all(worker._effective_detect_interval() == 4 for _ in range(10))

    def test_async_face_not_charged_to_frame_budget(self, worker, monkeypatch) -> None:
        # Field numbers (live NDI): face ran 190 ms on the appearance thread at
        # 4 Hz. Off the frame path, it must not push the detector to every 4th
        # frame. Its CPU cost is still covered by the system-CPU-pressure clamp.
        self._costs(worker, monkeypatch, detect=27.0, track=5.0, pose=17.0, face=190.0)
        worker._async_appearance = True
        for _ in range(5):
            worker._effective_detect_interval()
        assert worker._quality_interval == 2

    def test_sync_face_is_charged_to_frame_budget(self, worker, monkeypatch) -> None:
        self._costs(worker, monkeypatch, detect=27.0, track=5.0, pose=17.0, face=190.0)
        worker._async_appearance = False  # inline face really blocks the frame
        for _ in range(5):
            worker._effective_detect_interval()
        assert worker._quality_interval == 4

    def test_frame_backlog_relaxes_cadence_and_holds(self, worker, monkeypatch) -> None:
        # Live NDI, 6+ people: projected cost looked fine (async face excluded) but
        # frames aged to p90 ~165 ms at full cadence. Measured frame age is the
        # ground truth: relax, and don't re-upgrade until a cooldown passes.
        import autoptz.engine.camera_worker as cw

        clock = [1000.0]
        monkeypatch.setattr(cw, "_governor_clock", lambda: clock[0])
        self._costs(worker, monkeypatch, detect=10.0, track=3.0, age=90.0)
        worker._effective_detect_interval()
        assert worker._quality_interval >= 2
        relaxed = worker._quality_interval

        # Backlog clears, but we're inside the cooldown → no upgrade yet.
        self._costs(worker, monkeypatch, detect=10.0, track=3.0, age=20.0)
        clock[0] += 3.0
        assert worker._effective_detect_interval() == relaxed

        # After the cooldown, light load + no backlog → full cadence again.
        clock[0] += cw._BACKLOG_UPGRADE_COOLDOWN_S + 0.1
        for _ in range(3):
            worker._effective_detect_interval()
        assert worker._quality_interval == 1

    def test_no_backlog_no_change(self, worker, monkeypatch) -> None:
        self._costs(worker, monkeypatch, detect=10.0, track=3.0, age=25.0)
        assert worker._effective_detect_interval() == 1
