"""Phase 0 ENGINE instrumentation — worker-side wiring.

Covers:
  * Feature 0a — per-source frame-drop + queue telemetry passthrough
    (``_AdapterFrameSource.delivery_metrics`` and the 5 ``TelemetryMsg`` fields
    populated by ``CameraWorker._emit_telemetry``).
  * Feature 0b — true end-to-end latency probe (capture_age / command_send /
    actuation / end_to_end), the ``AUTOPTZ_TRUE_LATENCY_LEAD`` flag gate, and the
    4 ``TelemetryMsg`` latency fields.

No cameras, no real models — the worker is built bare (never ``.start()``ed) and
a fake frame source / controller is injected.
"""

from __future__ import annotations

import autoptz.engine.camera_worker as cw
from autoptz.config.models import CameraConfig
from autoptz.engine.camera_worker import CameraWorker
from autoptz.engine.runtime.messages import HealthState, TelemetryMsg


def _bare_worker() -> CameraWorker:
    cfg = CameraConfig(id="cam-lp", name="Cam LP")
    return CameraWorker("cam-lp", cfg, on_telemetry=lambda m: None)


# ── Feature 0a: frame_source passthrough ───────────────────────────────────────


class TestFrameSourceDeliveryPassthrough:
    def test_passthrough_returns_adapter_metrics(self) -> None:
        class _Adapter:
            def delivery_metrics(self) -> dict[str, float | int]:
                return {
                    "frames_delivered": 7,
                    "frames_dropped_est": 2,
                    "delivered_fps": 28.0,
                    "source_fps": 30.0,
                    "ndi_queue_depth": -1,
                }

        src = cw._AdapterFrameSource(_Adapter())
        assert src.delivery_metrics()["frames_dropped_est"] == 2

    def test_passthrough_returns_empty_when_method_absent(self) -> None:
        class _Adapter:  # no delivery_metrics
            pass

        src = cw._AdapterFrameSource(_Adapter())
        assert src.delivery_metrics() == {}

    def test_passthrough_returns_empty_when_method_raises(self) -> None:
        class _Adapter:
            def delivery_metrics(self) -> dict[str, float | int]:
                raise RuntimeError("boom")

        src = cw._AdapterFrameSource(_Adapter())
        assert src.delivery_metrics() == {}


# ── Feature 0a: _emit_telemetry populates the 5 delivery fields ─────────────────


class _FakeDeliverySource:
    """A frame source exposing only delivery_metrics() for telemetry tests."""

    def __init__(self, metrics: dict[str, float | int]) -> None:
        self._metrics = metrics

    def delivery_metrics(self) -> dict[str, float | int]:
        return self._metrics


def test_emit_telemetry_includes_delivery_fields() -> None:
    w = _bare_worker()
    w._source = _FakeDeliverySource(
        {
            "frames_delivered": 123,
            "frames_dropped_est": 9,
            "delivered_fps": 27.5,
            "source_fps": 30.0,
            "ndi_queue_depth": 4,
        }
    )
    captured: list[TelemetryMsg] = []
    w._on_telemetry = captured.append
    w._emit_telemetry(tracks=[], health=HealthState.OK, last_error=None)

    assert captured
    msg = captured[0]
    assert msg.frames_delivered == 123
    assert msg.frames_dropped_est == 9
    assert msg.delivered_fps == 27.5
    assert msg.source_fps == 30.0
    assert msg.ndi_queue_depth == 4


def test_emit_telemetry_falls_back_to_worker_counters() -> None:
    """No delivery_metrics → frames_delivered falls back to _frames_captured."""
    w = _bare_worker()
    w._source = None  # no source → _delivery_metrics() returns {}
    w._frames_captured = 55
    w._fps = 24.0
    captured: list[TelemetryMsg] = []
    w._on_telemetry = captured.append
    w._emit_telemetry(tracks=[], health=HealthState.OK, last_error=None)

    msg = captured[0]
    assert msg.frames_delivered == 55
    assert msg.frames_dropped_est == 0
    assert msg.delivered_fps == 24.0
    assert msg.source_fps == 0.0
    assert msg.ndi_queue_depth == -1


# ── Feature 0b: true end-to-end latency probe ──────────────────────────────────


class _CaptureLatencyController:
    """Records the seconds passed to set_loop_latency + supports the drive path."""

    def __init__(self) -> None:
        self.loop_latency_s: float | None = None
        self._cmd_send_ms = 0.0

    def set_loop_latency(self, seconds: float) -> None:
        self.loop_latency_s = seconds

    def last_cmd_send_ms(self) -> float:
        return self._cmd_send_ms


def _drive_with_flag(monkeypatch, *, flag_on: bool) -> _CaptureLatencyController:
    """Run _drive_ptz_auto once with a fake controller and the flag set, returning
    the controller so the test can read which latency was fed to set_loop_latency."""
    if flag_on:
        monkeypatch.setenv("AUTOPTZ_TRUE_LATENCY_LEAD", "1")
    else:
        monkeypatch.delenv("AUTOPTZ_TRUE_LATENCY_LEAD", raising=False)

    w = _bare_worker()
    ctrl = _CaptureLatencyController()
    w._ptz = ctrl
    # Known measured legacy latency + a stamped capture timestamp.
    w._latency_ms = 50.0  # legacy ingest+inference latency → 0.050 s
    w._manual_override_active = lambda now: False  # type: ignore[assignment]
    # No target resolvable → the idle branch runs but set_loop_latency is fed first.
    w._resolve_target_track = lambda tracks: None  # type: ignore[assignment]
    w._feature = lambda name: True  # type: ignore[assignment]
    now = 1000.0
    # Capture frame is 30 ms old at command time.
    w._current_inference_capture_ts = now - 0.030
    w._drive_ptz_auto([], None, now)
    return ctrl


class TestTrueLatencyLeadFlag:
    def test_flag_off_feeds_legacy_latency(self, monkeypatch) -> None:
        ctrl = _drive_with_flag(monkeypatch, flag_on=False)
        # Default OFF: the legacy self._latency_ms/1000 is fed, unchanged.
        assert ctrl.loop_latency_s == 0.050

    def test_flag_on_feeds_end_to_end_latency(self, monkeypatch) -> None:
        ctrl = _drive_with_flag(monkeypatch, flag_on=True)
        # ON: capture_age (30) + command_send (0, no send yet) + actuation (40)
        # = 70 ms → 0.070 s. Distinct from the legacy 0.050.
        assert ctrl.loop_latency_s is not None
        assert abs(ctrl.loop_latency_s - 0.070) < 1e-6


class TestCaptureAgePlumbing:
    def test_capture_age_computed_from_stamped_ts(self, monkeypatch) -> None:
        monkeypatch.delenv("AUTOPTZ_TRUE_LATENCY_LEAD", raising=False)
        w = _bare_worker()
        w._ptz = _CaptureLatencyController()
        w._manual_override_active = lambda now: False  # type: ignore[assignment]
        w._resolve_target_track = lambda tracks: None  # type: ignore[assignment]
        w._feature = lambda name: True  # type: ignore[assignment]
        now = 2000.0
        w._current_inference_capture_ts = now - 0.025  # 25 ms old
        w._drive_ptz_auto([], None, now)
        # capture_age_ms is stamped onto the worker for telemetry (~25 ms).
        assert abs(w._capture_age_ms - 25.0) < 1.0

    def test_capture_age_zero_when_unstamped(self, monkeypatch) -> None:
        monkeypatch.delenv("AUTOPTZ_TRUE_LATENCY_LEAD", raising=False)
        w = _bare_worker()
        w._ptz = _CaptureLatencyController()
        w._manual_override_active = lambda now: False  # type: ignore[assignment]
        w._resolve_target_track = lambda tracks: None  # type: ignore[assignment]
        w._feature = lambda name: True  # type: ignore[assignment]
        w._current_inference_capture_ts = 0.0  # never stamped
        w._drive_ptz_auto([], None, 3000.0)
        assert w._capture_age_ms == 0.0


class TestTrueLatencyFlagHelper:
    def test_flag_helper_default_off(self, monkeypatch) -> None:
        monkeypatch.delenv("AUTOPTZ_TRUE_LATENCY_LEAD", raising=False)
        assert cw._true_latency_lead_enabled() is False

    def test_flag_helper_on(self, monkeypatch) -> None:
        monkeypatch.setenv("AUTOPTZ_TRUE_LATENCY_LEAD", "1")
        assert cw._true_latency_lead_enabled() is True


class TestEmitTelemetryLatencyFields:
    def test_emit_telemetry_includes_latency_decomposition(self) -> None:
        w = _bare_worker()
        w._capture_age_ms = 30.0
        w._command_send_ms = 5.0
        w._end_to_end_ms = 75.0
        captured: list[TelemetryMsg] = []
        w._on_telemetry = captured.append
        w._emit_telemetry(tracks=[], health=HealthState.OK, last_error=None)
        msg = captured[0]
        assert msg.capture_age_ms == 30.0
        assert msg.command_send_ms == 5.0
        assert msg.actuation_estimate_ms == w.config.ptz.actuation_estimate_ms
        assert msg.end_to_end_ms == 75.0
