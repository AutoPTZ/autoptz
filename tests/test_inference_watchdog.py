"""Inference-hang watchdog tests (R2′).

Tests for:
  - ``CameraWorker._inference_stalled`` — pure predicate, no threads.
  - ``CameraWorker._apply_inference_watchdog`` — action: PTZ stop + status.
  - ``CameraWorker._tracking_status_info`` — surfaces ``degraded`` when stalled.

All tests construct a ``CameraWorker`` WITHOUT starting it (no threads, no
camera hardware, no ML models).  State is set directly on the worker instance
and the methods under test are called synchronously.
"""

from __future__ import annotations

import threading
from typing import Any  # noqa: UP035

import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────


def _camera_config(camera_id: str = "watchdog1234abcd", *, tracking_on: bool = True):
    """Build a minimal CameraConfig with tracking on or off."""
    from autoptz.config.models import CameraConfig, SourceConfig, TargetConfig

    return CameraConfig(
        id=camera_id,
        name="Watchdog Cam",
        source=SourceConfig(type="usb", address="usb://0"),
        target=TargetConfig(mode="identity" if tracking_on else "off"),
    )


class _FakeSource:
    """Minimal FrameSource stub — never called in these tests."""

    def open(self) -> bool:
        return True

    def read(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


def _make_worker(camera_id: str = "watchdog1234abcd", *, tracking_on: bool = True):
    """Construct a CameraWorker without starting it."""
    from autoptz.engine.camera_worker import CameraWorker

    return CameraWorker(
        camera_id,
        _camera_config(camera_id, tracking_on=tracking_on),
        lambda _msg: None,
        frame_source=_FakeSource(),
    )


class _FakePTZBackend:
    """Records stop() calls; no real hardware."""

    def __init__(self) -> None:
        self.stop_calls: int = 0

    def stop(self) -> None:
        self.stop_calls += 1

    def move_velocity(self, pan: float, tilt: float, zoom: float) -> None:  # noqa: D401
        pass


# ── _inference_stalled tests ──────────────────────────────────────────────────


class TestInferenceStalled:
    """Unit tests for the pure ``_inference_stalled`` predicate."""

    def _worker(self, **kw: Any):
        return _make_worker(**kw)

    def test_false_when_fresh_no_inferred_frames(self) -> None:
        """Stall predicate is False on a fresh worker (no frames inferred yet)."""
        w = self._worker()
        w._last_infer_t = 0.0
        w._frames_inferred = 0
        assert w._inference_stalled(now=100.0) is False

    def test_false_when_tracking_disabled(self) -> None:
        """Stall predicate is False when tracking is off — nothing to protect."""
        w = self._worker(tracking_on=False)
        w._frames_inferred = 10
        w._last_infer_t = 0.0  # very old heartbeat
        assert w._inference_stalled(now=100.0) is False

    def test_false_when_heartbeat_is_recent(self) -> None:
        """Stall predicate is False when the heartbeat is within the threshold."""
        from autoptz.engine import camera_worker as cw

        w = self._worker()
        w._frames_inferred = 5
        now = 100.0
        # heartbeat only 0.5 s ago — well within _INFER_STALL_S (2.0 s)
        w._last_infer_t = now - (cw._INFER_STALL_S - 0.5)
        assert w._inference_stalled(now=now) is False

    def test_true_when_stale_and_started_and_tracking_on(self) -> None:
        """Stall predicate is True when inference started and heartbeat is old."""
        from autoptz.engine import camera_worker as cw

        w = self._worker()
        w._frames_inferred = 1
        now = 100.0
        # heartbeat exactly at the boundary (just over the threshold)
        w._last_infer_t = now - cw._INFER_STALL_S - 0.01
        assert w._inference_stalled(now=now) is True

    def test_true_with_many_frames_and_old_heartbeat(self) -> None:
        """Stall with many inferred frames and a very old heartbeat."""
        w = self._worker()
        w._frames_inferred = 9999
        w._last_infer_t = 0.0
        assert w._inference_stalled(now=100.0) is True

    def test_boundary_exactly_at_threshold_is_false(self) -> None:
        """Exactly at the threshold should NOT be stalled (strict greater-than)."""
        from autoptz.engine import camera_worker as cw

        w = self._worker()
        w._frames_inferred = 1
        now = 100.0
        w._last_infer_t = now - cw._INFER_STALL_S  # exactly at threshold
        assert w._inference_stalled(now=now) is False

    def test_recovery_after_heartbeat_advances(self) -> None:
        """Once the heartbeat is updated, the stall clears automatically."""
        from autoptz.engine import camera_worker as cw

        w = self._worker()
        w._frames_inferred = 5
        now = 100.0
        # Initially stalled
        w._last_infer_t = now - cw._INFER_STALL_S - 1.0
        assert w._inference_stalled(now=now) is True
        # Inference resumes — heartbeat advances
        w._last_infer_t = now - 0.05
        assert w._inference_stalled(now=now) is False


class TestInferenceStallAge:
    def _worker(self, **kw: Any):
        return _make_worker(**kw)

    def test_zero_when_no_inferred_frames(self) -> None:
        w = self._worker()
        w._frames_inferred = 0
        w._last_infer_t = 0.0
        assert w._inference_stall_age(now=100.0) == 0.0

    def test_zero_when_tracking_disabled(self) -> None:
        w = self._worker(tracking_on=False)
        w._frames_inferred = 10
        w._last_infer_t = 0.0
        assert w._inference_stall_age(now=100.0) == 0.0

    def test_reports_age_when_running(self) -> None:
        w = self._worker()
        w._frames_inferred = 5
        w._last_infer_t = 90.0
        assert abs(w._inference_stall_age(now=100.0) - 10.0) < 1e-6

    def test_never_negative(self) -> None:
        w = self._worker()
        w._frames_inferred = 5
        w._last_infer_t = 105.0  # clock skew / future heartbeat
        assert w._inference_stall_age(now=100.0) == 0.0


def test_healthinfo_field_defaults_and_round_trips() -> None:
    from autoptz.engine.runtime.messages import HealthInfo, TelemetryMsg

    assert HealthInfo().inference_stall_age_s == 0.0
    msg = TelemetryMsg(
        camera_id="c1",
        seq=0,
        health=HealthInfo(inference_stall_age_s=3.5),
    )
    restored = TelemetryMsg.from_msgpack(msg.to_msgpack())
    assert restored.health.inference_stall_age_s == 3.5


# ── _apply_inference_watchdog action tests ────────────────────────────────────


class TestApplyInferenceWatchdog:
    """Tests for the watchdog action method."""

    def _stalled_worker_with_fake_ptz(self) -> tuple[Any, _FakePTZBackend]:
        """Return a worker in a stalled state with a fake PTZ backend."""
        from autoptz.engine import camera_worker as cw

        w = _make_worker()
        w._frames_inferred = 5
        # Make it stalled
        w._last_infer_t = 0.0

        backend = _FakePTZBackend()
        w._ptz_backend = backend
        w._ptz_lock = threading.Lock()

        now = cw._INFER_STALL_S + 10.0
        return w, backend, now

    def test_watchdog_calls_backend_stop_when_stalled(self) -> None:
        """When stalled, _apply_inference_watchdog must call backend.stop()."""
        w, backend, now = self._stalled_worker_with_fake_ptz()
        w._apply_inference_watchdog(now)
        assert backend.stop_calls >= 1

    def test_watchdog_sets_stalled_flag(self) -> None:
        """When stalled, _watchdog_stalled must be set to True."""
        w, _backend, now = self._stalled_worker_with_fake_ptz()
        w._apply_inference_watchdog(now)
        assert w._watchdog_stalled is True

    def test_watchdog_stops_once_per_stall_episode(self) -> None:
        """stop() is called ONCE on entry to a stall; re-armed after recovery."""
        from autoptz.engine import camera_worker as cw

        w, backend, now = self._stalled_worker_with_fake_ptz()

        # --- first stall episode ---
        # Three consecutive ticks while continuously stalled → stop called once.
        w._apply_inference_watchdog(now)
        w._apply_inference_watchdog(now)
        w._apply_inference_watchdog(now)
        assert w._watchdog_stalled is True
        assert backend.stop_calls == 1

        # --- recovery ---
        # Advance the heartbeat so _inference_stalled returns False.
        w._last_infer_t = now - 0.05
        w._apply_inference_watchdog(now)
        assert w._watchdog_stalled is False
        assert backend.stop_calls == 1  # no new stop on recovery

        # --- second stall episode ---
        # Drop the heartbeat again; watchdog must re-arm and stop once more.
        w._last_infer_t = 0.0
        stall_now2 = now + cw._INFER_STALL_S + 1.0
        w._apply_inference_watchdog(stall_now2)
        w._apply_inference_watchdog(stall_now2)
        assert w._watchdog_stalled is True
        assert backend.stop_calls == 2

    def test_watchdog_no_stop_without_ptz_backend(self) -> None:
        """Watchdog with no PTZ backend configured must not raise."""
        from autoptz.engine import camera_worker as cw

        w = _make_worker()
        w._frames_inferred = 5
        w._last_infer_t = 0.0
        w._ptz_backend = None

        now = cw._INFER_STALL_S + 5.0
        w._apply_inference_watchdog(now)  # must not raise
        assert w._watchdog_stalled is True

    def test_watchdog_clears_flag_on_recovery(self) -> None:
        """After inference resumes (heartbeat advances), watchdog clears the flag."""
        from autoptz.engine import camera_worker as cw

        w, _backend, _now = self._stalled_worker_with_fake_ptz()
        stalled_now = cw._INFER_STALL_S + 10.0

        # Trigger stall
        w._apply_inference_watchdog(stalled_now)
        assert w._watchdog_stalled is True

        # Simulate inference recovery: heartbeat advances
        w._last_infer_t = stalled_now - 0.05
        # Watchdog detects recovery
        w._apply_inference_watchdog(stalled_now)
        assert w._watchdog_stalled is False

    def test_watchdog_no_stop_when_not_stalled(self) -> None:
        """If inference is healthy, backend.stop() must NOT be called."""
        w = _make_worker()
        backend = _FakePTZBackend()
        w._ptz_backend = backend
        w._ptz_lock = threading.Lock()

        now = 100.0
        w._frames_inferred = 10
        w._last_infer_t = now - 0.1  # very recent heartbeat

        w._apply_inference_watchdog(now)
        assert backend.stop_calls == 0
        assert w._watchdog_stalled is False


# ── _tracking_status_info degraded state tests ───────────────────────────────


class TestTrackingStatusInfoWhenStalled:
    """Tests that _tracking_status_info surfaces the stalled/degraded state."""

    def _worker_with_target(self) -> Any:
        """Build a worker that has a target set (status would be non-idle)."""
        w = _make_worker()
        # Simulate having a target identity set so status is non-trivial
        w._target_identity_id = "identity-abc"
        w._target_track_id = None
        return w

    def test_status_degraded_when_watchdog_stalled(self) -> None:
        """When _watchdog_stalled is True, tracking status must be 'degraded'."""
        w = self._worker_with_target()
        w._watchdog_stalled = True
        status = w._tracking_status_info([], now=100.0)
        assert status.state == "degraded"

    def test_status_detail_mentions_stall(self) -> None:
        """The detail field must mention 'inference stalled'."""
        w = self._worker_with_target()
        w._watchdog_stalled = True
        status = w._tracking_status_info([], now=100.0)
        assert "inference stalled" in status.detail.lower()

    def test_status_severity_is_warning(self) -> None:
        """The severity must be 'warning' so the UI highlights it."""
        w = self._worker_with_target()
        w._watchdog_stalled = True
        status = w._tracking_status_info([], now=100.0)
        assert status.severity == "warning"

    def test_status_action_is_holding(self) -> None:
        """Action must be 'holding' — PTZ has been stopped."""
        w = self._worker_with_target()
        w._watchdog_stalled = True
        status = w._tracking_status_info([], now=100.0)
        assert status.action == "holding"

    def test_status_idle_when_no_target(self) -> None:
        """With no target, the stall state is not reached (idle short-circuit)."""
        w = _make_worker()
        # No target at all
        w._target_identity_id = None
        w._target_track_id = None
        w._watchdog_stalled = True
        status = w._tracking_status_info([], now=100.0)
        # The early-return path fires — state is 'idle' (default)
        assert status.state == "idle"

    def test_status_normal_when_not_stalled(self) -> None:
        """When watchdog is not stalled, the normal tracking state is returned."""
        w = self._worker_with_target()
        w._watchdog_stalled = False
        status = w._tracking_status_info([], now=100.0)
        assert status.state != "degraded"
