"""Ego-motion freshness gate — the ping-pong fix.

Bug: ego-motion is decimated (optical flow every Nth inference frame). On the
off-cadence frames the cached estimate is reused, but `_estimate_aim_velocity`
ran every tick and unconditionally SUBTRACTED that stale/decayed estimate from
the per-frame aim velocity — producing a sawtooth "phantom" world-velocity that
the controller's predictive lead chased, i.e. the camera ping-ponged when it AND
the subject moved. Fix: only subtract ego when it was freshly measured this tick.
"""

from __future__ import annotations

import numpy as np

from autoptz.config.models import CameraConfig, PTZConfig, SourceConfig, TrackingConfig


def _worker(**ptz):
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(
        id="cam-ego-00000001",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(),
        ptz=PTZConfig(**ptz),
    )
    return CameraWorker("cam-ego-00000001", cfg, on_telemetry=lambda m: None)


class TestEgoFreshnessGate:
    def test_stale_ego_is_not_subtracted(self):
        w = _worker()
        w._prev_aim_err = (0.0, 0.0)
        w._prev_aim_t = 0.0
        w._aim_vel = (0.0, 0.0)
        w._ego_vel = (0.5, 0.0)  # a (stale) cached ego estimate
        w._ego_fresh = False  # decimated/off-cadence → not freshly measured
        vx, _ = w._estimate_aim_velocity((0.1, 0.0), 0.1)
        # raw d(err)/dt = 1.0, ego NOT subtracted, EMA(0.5) from 0 → 0.5
        assert abs(vx - 0.5) < 1e-6

    def test_fresh_ego_is_subtracted(self):
        w = _worker()
        w._prev_aim_err = (0.0, 0.0)
        w._prev_aim_t = 0.0
        w._aim_vel = (0.0, 0.0)
        w._ego_vel = (0.5, 0.0)
        w._ego_fresh = True  # freshly measured this tick
        vx, _ = w._estimate_aim_velocity((0.1, 0.0), 0.1)
        # raw = 1.0 - 0.5 = 0.5, EMA(0.5) from 0 → 0.25
        assert abs(vx - 0.25) < 1e-6

    def test_update_sets_freshness_per_branch(self):
        w = _worker(ego_comp_enabled=True, ego_comp_interval=3)

        class _Est:
            def estimate(self, *a, **k):
                from autoptz.engine.pipeline.egomotion import EgoMotion

                return EgoMotion(vx=0.2, vy=0.0, source="flow", confidence=1.0)

        w._ego_estimator = _Est()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Measured frame (frames_inferred % 3 == 0) → fresh.
        w._frames_inferred = 0
        w._update_ego_motion([], frame, 0.0)
        assert w._ego_fresh is True

        # Off-cadence frame → not fresh (stale estimate must not be trusted).
        w._frames_inferred = 1
        w._update_ego_motion([], frame, 0.1)
        assert w._ego_fresh is False

        # Disabled → not fresh.
        w2 = _worker(ego_comp_enabled=False)
        w2._update_ego_motion([], frame, 0.0)
        assert w2._ego_fresh is False
