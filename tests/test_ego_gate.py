"""Ego-motion freshness gate — the pan-jitter fix.

Bug: ego-motion is decimated (optical flow every Nth inference frame, default 3).
`_estimate_aim_velocity` ran every tick: on the off-cadence frames it computed the
per-tick d(error)/dt but zero-subtracted ego, injecting the FULL camera-pan
velocity into the controller's feed-forward on 2 of every 3 frames — so the camera
hunted (find/lose/find) during pans.

Fix (window matching): the ego estimate spans `interval` frames, so the subject
velocity must be measured over the SAME window for the subtraction to cancel.  We
only recompute on a freshly-measured (`_ego_fresh`) tick and HOLD the last value
between fresh ticks (when ego comp is enabled).  With ego comp OFF there is no
multi-frame estimate to match, so velocity is computed every tick as before.
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
    def test_stale_ego_tick_holds_and_does_not_inject_raw_velocity(self):
        # Off-cadence tick (ego enabled): the camera-pan-contaminated d(error)/dt
        # must NOT be injected.  The last subject velocity (0.0 here) is held.
        w = _worker()
        w._prev_aim_err = (0.0, 0.0)
        w._prev_aim_t = 0.0
        w._aim_vel = (0.0, 0.0)
        w._ego_vel = (0.5, 0.0)  # a (stale) cached ego estimate
        w._ego_fresh = False  # decimated/off-cadence → not freshly measured
        vx, _ = w._estimate_aim_velocity((0.1, 0.0), 0.1)
        assert abs(vx - 0.0) < 1e-6  # held, not the raw d(err)/dt of 1.0

    def test_offcadence_holds_last_velocity_when_ego_enabled(self):
        # Window-matched fix: on an off-cadence (not-fresh) tick with ego comp
        # ENABLED, the per-tick d(error)/dt is camera-pan-contaminated (ego spans
        # `interval` frames, the error delta spans 1) — so instead of injecting it
        # we HOLD the last subject velocity and do not roll the prev marker, so the
        # next fresh tick's delta spans the same window the ego estimate covers.
        w = _worker(ego_comp_enabled=True, ego_comp_interval=3)
        w._prev_aim_err = (0.0, 0.0)
        w._prev_aim_t = 0.0
        w._aim_vel = (0.2, 0.0)  # last good (subject) velocity
        w._ego_fresh = False  # off-cadence
        vx, _ = w._estimate_aim_velocity((0.1, 0.0), 0.1)
        assert abs(vx - 0.2) < 1e-6  # held, not the raw d(err)/dt of 1.0
        assert w._prev_aim_err == (0.0, 0.0)  # prev NOT rolled
        assert w._prev_aim_t == 0.0

    def test_offcadence_still_computes_when_ego_disabled(self):
        # With ego comp OFF there is no multi-frame estimate to window-match, so
        # the per-tick velocity is still computed every tick (legacy behaviour).
        w = _worker(ego_comp_enabled=False)
        w._prev_aim_err = (0.0, 0.0)
        w._prev_aim_t = 0.0
        w._aim_vel = (0.0, 0.0)
        w._ego_fresh = False
        vx, _ = w._estimate_aim_velocity((0.1, 0.0), 0.1)
        assert abs(vx - 0.5) < 1e-6  # raw 1.0, EMA(0.5) from 0 → 0.5

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
