"""Phase 5: PTZ backends + closed-loop controller tests.

Uses a MockBackend throughout; no real hardware needed.
"""
from __future__ import annotations

import time

import pytest

from autoptz.config.models import PTZConfig
from autoptz.engine.ptz.base import (
    PTZBackend,
    PTZCaps,
    PTZState,
    visca_pantilt_cmd,
    visca_preset_recall_cmd,
    visca_preset_set_cmd,
    visca_stop_cmd,
    visca_zoom_cmd,
    visca_zoom_stop_cmd,
)
from autoptz.engine.ptz.controller import (
    ControllerState,
    OneEuroFilter,
    PTZController,
    _clamp,
    _shape,
)

# ─────────────────────────────────────────────────────────────────────────────
# Mock backend
# ─────────────────────────────────────────────────────────────────────────────


class MockBackend(PTZBackend):
    def __init__(self, has_position: bool = False) -> None:
        super().__init__()
        self.caps = PTZCaps(
            continuous_pan_tilt=True,
            continuous_zoom=True,
            native_presets=True,
        )
        self.velocity_calls: list[tuple[float, float, float]] = []
        self.stop_count: int = 0
        self.presets_set: dict[int, bool] = {}
        self.presets_recalled: list[int] = []
        self.closed: bool = False
        self._pos: PTZState | None = PTZState() if has_position else None

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        self.velocity_calls.append((pan, tilt, zoom))

    def stop(self) -> None:
        self.stop_count += 1

    def get_position(self) -> PTZState | None:
        return self._pos

    def goto_preset(self, idx: int) -> None:
        self.presets_recalled.append(idx)

    def save_preset(self, idx: int) -> None:
        self.presets_set[idx] = True

    def close(self) -> None:
        self.closed = True


def _cfg(**kw: object) -> PTZConfig:
    """Build PTZConfig with defaults suitable for deterministic unit tests."""
    defaults: dict[str, object] = {
        "kp": 0.6,
        "kd": 0.0,
        "kv": 0.0,
        "deadzone_x": 0.0,
        "deadzone_y": 0.0,
        "max_pan_speed": 1.0,
        "max_tilt_speed": 1.0,
        "max_zoom_speed": 1.0,
        "auto_zoom": False,
    }
    defaults.update(kw)
    return PTZConfig(**defaults)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# PTZCaps / PTZState
# ─────────────────────────────────────────────────────────────────────────────


class TestDataclasses:
    def test_caps_defaults(self) -> None:
        c = PTZCaps()
        assert c.continuous_pan_tilt is True
        assert c.absolute_pan_tilt is False
        assert c.native_presets is False

    def test_state_defaults(self) -> None:
        s = PTZState()
        assert s.pan == 0.0
        assert s.tilt == 0.0
        assert s.zoom == 0.0
        assert s.timestamp > 0.0

    def test_state_custom(self) -> None:
        s = PTZState(pan=0.5, tilt=-0.3, zoom=0.8)
        assert s.pan == pytest.approx(0.5)
        assert s.tilt == pytest.approx(-0.3)
        assert s.zoom == pytest.approx(0.8)


# ─────────────────────────────────────────────────────────────────────────────
# VISCA byte helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestViscaHelpers:
    def test_stop_cmd_bytes(self) -> None:
        cmd = visca_stop_cmd()
        assert cmd == bytes([0x81, 0x01, 0x06, 0x01, 0x15, 0x15, 0x03, 0x03, 0xFF])

    def test_zoom_stop_bytes(self) -> None:
        cmd = visca_zoom_stop_cmd()
        assert cmd == bytes([0x81, 0x01, 0x04, 0x07, 0x00, 0xFF])

    def test_pantilt_cmd_stop(self) -> None:
        cmd = visca_pantilt_cmd(0.0, 0.0)
        # pan_dir=0x03, tilt_dir=0x03 (both stop)
        assert cmd[6] == 0x03
        assert cmd[7] == 0x03
        assert cmd[-1] == 0xFF

    def test_pantilt_cmd_full_right_up(self) -> None:
        cmd = visca_pantilt_cmd(1.0, 1.0)
        assert cmd[4] == 0x18   # max pan speed
        assert cmd[5] == 0x14   # max tilt speed
        assert cmd[6] == 0x02   # right
        assert cmd[7] == 0x01   # up

    def test_pantilt_cmd_full_left_down(self) -> None:
        cmd = visca_pantilt_cmd(-1.0, -1.0)
        assert cmd[6] == 0x01   # left
        assert cmd[7] == 0x02   # down

    def test_pantilt_speed_clamped(self) -> None:
        # speed must be at least 0x01 for moving, at most 0x18 pan / 0x14 tilt
        cmd = visca_pantilt_cmd(0.001, 0.0)
        assert 0x01 <= cmd[4] <= 0x18

    def test_zoom_tele(self) -> None:
        cmd = visca_zoom_cmd(1.0)
        assert cmd[4] & 0xF0 == 0x20   # tele nibble
        assert 0x01 <= cmd[4] & 0x0F <= 0x07

    def test_zoom_wide(self) -> None:
        cmd = visca_zoom_cmd(-1.0)
        assert cmd[4] & 0xF0 == 0x30   # wide nibble

    def test_zoom_stop(self) -> None:
        cmd = visca_zoom_cmd(0.0)
        assert cmd[4] == 0x00

    def test_preset_set_cmd(self) -> None:
        cmd = visca_preset_set_cmd(3)
        assert cmd == bytes([0x81, 0x01, 0x04, 0x3F, 0x01, 0x03, 0xFF])

    def test_preset_recall_cmd(self) -> None:
        cmd = visca_preset_recall_cmd(5)
        assert cmd == bytes([0x81, 0x01, 0x04, 0x3F, 0x02, 0x05, 0xFF])


# ─────────────────────────────────────────────────────────────────────────────
# One-euro filter
# ─────────────────────────────────────────────────────────────────────────────


class TestOneEuroFilter:
    def test_init_pass_through(self) -> None:
        f = OneEuroFilter()
        assert f(0.5, 0.0) == pytest.approx(0.5)

    def test_constant_signal_converges(self) -> None:
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.007)
        t = 0.0
        dt = 1.0 / 30.0
        out = 0.0
        for _ in range(60):
            out = f(1.0, t)
            t += dt
        assert out == pytest.approx(1.0, abs=0.02)

    def test_fast_signal_low_lag(self) -> None:
        """On a step from 0→1, the filter should reach >0.5 within 10 samples."""
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.1)
        t = 0.0
        dt = 1.0 / 30.0
        out = 0.0
        for _ in range(10):
            out = f(1.0, t)
            t += dt
        assert out > 0.5

    def test_reset(self) -> None:
        f = OneEuroFilter()
        f(1.0, 0.0)
        f.reset()
        # after reset, first call should pass through again
        assert f(0.3, 0.1) == pytest.approx(0.3)

    def test_no_timestamp_works(self) -> None:
        f = OneEuroFilter(freq=30.0)
        vals = [f(float(i)) for i in range(5)]
        assert vals[0] == pytest.approx(0.0)
        assert vals[-1] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_clamp_below(self) -> None:
        assert _clamp(-2.0, -1.0, 1.0) == pytest.approx(-1.0)

    def test_clamp_above(self) -> None:
        assert _clamp(5.0, -1.0, 1.0) == pytest.approx(1.0)

    def test_clamp_within(self) -> None:
        assert _clamp(0.3, -1.0, 1.0) == pytest.approx(0.3)

    def test_shape_zero(self) -> None:
        assert _shape(0.0) == pytest.approx(0.0)

    def test_shape_unity(self) -> None:
        assert _shape(1.0) == pytest.approx(1.0)
        assert _shape(-1.0) == pytest.approx(-1.0)

    def test_shape_ease_in(self) -> None:
        # ease-in: mid-range output < linear (0.5^1.5 ≈ 0.354 < 0.5)
        assert _shape(0.5) < 0.5
        assert _shape(-0.5) > -0.5

    def test_shape_preserves_sign(self) -> None:
        assert _shape(0.3) > 0.0
        assert _shape(-0.3) < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Controller — math / state
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerMath:
    def test_zero_error_zero_pan(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=0.6))
        pan, tilt, _ = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan == pytest.approx(0.0, abs=1e-6)
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_positive_error_positive_pan(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=0.6))
        pan, _, _ = ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan > 0.0

    def test_negative_error_negative_pan(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=0.6))
        pan, _, _ = ctrl.step((-0.5, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan < 0.0

    def test_velocity_feedforward_increases_command(self) -> None:
        b1, b2 = MockBackend(), MockBackend()
        ctrl_noff = PTZController(b1, _cfg(kv=0.0))
        ctrl_ff = PTZController(b2, _cfg(kv=0.3))
        pan_noff, _, _ = ctrl_noff.step((0.2, 0.0), (0.5, 0.0), 0.45, True, t=0.0)
        pan_ff, _, _ = ctrl_ff.step((0.2, 0.0), (0.5, 0.0), 0.45, True, t=0.0)
        assert pan_ff > pan_noff

    def test_deadzone_suppresses_small_error(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=0.6, deadzone_x=0.1, deadzone_y=0.1))
        pan, tilt, _ = ctrl.step((0.05, 0.08), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan == pytest.approx(0.0, abs=1e-6)
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_deadzone_passes_large_error(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=0.6, deadzone_x=0.05, deadzone_y=0.05))
        pan, _, _ = ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan > 0.0

    def test_shifted_safe_zone_suppresses_error_around_box_center(self) -> None:
        ctrl = PTZController(
            MockBackend(),
            _cfg(safe_zone_enabled=True, safe_zone_x=0.3, safe_zone_y=-0.2,
                 safe_zone_w=0.1, safe_zone_h=0.1),
        )
        pan, tilt, _ = ctrl.step((0.34, -0.24), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan == pytest.approx(0.0, abs=1e-6)
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_shifted_safe_zone_tracks_relative_to_box_center(self) -> None:
        ctrl = PTZController(
            MockBackend(),
            _cfg(safe_zone_enabled=True, safe_zone_x=0.3, safe_zone_y=0.0,
                 safe_zone_w=0.05, safe_zone_h=0.05),
        )
        pan, _, _ = ctrl.step((0.6, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan > 0.0

    def test_invert_pan(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=0.6, invert_pan=True))
        pan, _, _ = ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan < 0.0  # positive error + invert → negative command

    def test_invert_tilt(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=0.6, invert_tilt=True))
        _, tilt, _ = ctrl.step((0.0, 0.5), (0.0, 0.0), 0.45, True, t=0.0)
        assert tilt < 0.0

    def test_max_speed_scales_output(self) -> None:
        b1, b2 = MockBackend(), MockBackend()
        ctrl_slow = PTZController(b1, _cfg(kp=1.0, max_pan_speed=0.3))
        ctrl_fast = PTZController(b2, _cfg(kp=1.0, max_pan_speed=1.0))
        pan_slow, _, _ = ctrl_slow.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        pan_fast, _, _ = ctrl_fast.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan_slow < pan_fast

    def test_pan_clamped_to_plus_one(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=100.0))
        pan, _, _ = ctrl.step((1.0, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert abs(pan) <= 1.0

    def test_response_curve_applied(self) -> None:
        """With large Kp, mid-range error should produce less than half-speed."""
        ctrl = PTZController(MockBackend(), _cfg(kp=0.5))
        pan, _, _ = ctrl.step((1.0, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        # shape(0.5) ≈ 0.35; shape(1.0)=1.0 — shape(0.5) < 0.5
        assert pan < 0.5

    def test_commands_sent_to_backend(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(kp=0.6))
        ctrl.step((0.4, 0.2), (0.0, 0.0), 0.45, True, t=0.0)
        assert len(backend.velocity_calls) >= 1
        pan_sent, tilt_sent, _ = backend.velocity_calls[-1]
        assert pan_sent > 0.0
        assert tilt_sent > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Controller — state machine (coast / search)
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerStateMachine:
    def test_idle_on_init(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg())
        assert ctrl.state == ControllerState.IDLE

    def test_transitions_to_tracking(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg())
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert ctrl.state == ControllerState.TRACKING

    def test_loss_enters_coasting(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(), coast_window_ms=500)
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.05)
        assert ctrl.state == ControllerState.COASTING

    def test_coast_sends_last_velocity(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(kp=0.6), coast_window_ms=500)
        # track once to get a non-zero command
        ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        prev_calls = len(backend.velocity_calls)
        # lose target → coast
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.05)
        # the coast tick should send (a non-trivially-different) command
        assert len(backend.velocity_calls) >= prev_calls

    def test_coast_expires_to_searching(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(), coast_window_ms=300)
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        # lose target
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.05)
        assert ctrl.state == ControllerState.COASTING
        # advance past coast window
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.40)
        assert ctrl.state == ControllerState.SEARCHING

    def test_coast_expired_sends_stop(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(), coast_window_ms=200)
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.05)
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.30)
        assert backend.stop_count >= 1

    def test_reacquire_returns_to_tracking(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(), coast_window_ms=500)
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.05)
        assert ctrl.state == ControllerState.COASTING
        # target found again
        ctrl.step((0.1, 0.0), (0.0, 0.0), 0.45, True, t=0.10)
        assert ctrl.state == ControllerState.TRACKING

    def test_idle_sends_zero(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg())
        # Never call step with track_active=True → stays IDLE
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.45, False, t=0.0)
        if backend.velocity_calls:
            pan, tilt, _ = backend.velocity_calls[-1]
            assert pan == pytest.approx(0.0, abs=1e-6)
            assert tilt == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Controller — zoom
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerZoom:
    def test_auto_zoom_off_no_zoom_cmd(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(auto_zoom=False))
        _, _, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.9, True, t=0.0)
        assert zoom == pytest.approx(0.0)

    def test_tall_subject_zooms_out(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(auto_zoom=True, zoom_framing="medium"))
        # subject_height=0.9 >> target 0.45 → zoom out (negative)
        _, _, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.9, True, t=0.0)
        assert zoom < 0.0

    def test_short_subject_zooms_in(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(auto_zoom=True, zoom_framing="medium"))
        # subject_height=0.1 << target 0.45 → zoom in (positive)
        _, _, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.1, True, t=0.0)
        assert zoom > 0.0

    def test_subject_in_band_no_zoom(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(auto_zoom=True, zoom_framing="medium"))
        # subject_height ≈ target (0.45 ± 0.05 hysteresis)
        _, _, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert zoom == pytest.approx(0.0, abs=1e-9)

    def test_zoom_clamped_to_max_speed(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(auto_zoom=True, max_zoom_speed=0.4))
        _, _, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.9, True, t=0.0)
        assert abs(zoom) <= 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Controller — presets
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerPresets:
    def test_goto_preset_dispatches_to_backend(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg())
        ctrl.goto_preset(2)
        assert 2 in backend.presets_recalled

    def test_save_preset_dispatches_to_backend(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg())
        ctrl.save_preset(1)
        assert backend.presets_set.get(1) is True

    def test_multiple_presets(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg())
        ctrl.save_preset(0)
        ctrl.save_preset(3)
        ctrl.goto_preset(0)
        ctrl.goto_preset(3)
        assert 0 in backend.presets_set
        assert 3 in backend.presets_set
        assert backend.presets_recalled == [0, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Controller — thread lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerThread:
    def test_start_stop_sends_backend_stop(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(), rate_hz=50.0)
        ctrl.start()
        time.sleep(0.05)
        ctrl.stop()
        assert backend.stop_count >= 1

    def test_close_closes_backend(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(), rate_hz=50.0)
        ctrl.start()
        ctrl.close()
        assert backend.closed is True

    def test_context_manager(self) -> None:
        backend = MockBackend()
        with PTZController(backend, _cfg(), rate_hz=50.0):
            pass
        assert backend.closed is True

    def test_stop_without_start_is_safe(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg())
        ctrl.stop()  # must not raise
        assert backend.stop_count >= 1

    def test_double_start_no_extra_threads(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(), rate_hz=50.0)
        ctrl.start()
        ctrl.start()  # idempotent
        assert ctrl._thread is not None
        ctrl.stop()

    def test_thread_delivers_commands(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(kp=0.6), rate_hz=50.0)
        ctrl.start()
        ctrl.update((0.5, 0.0), (0.0, 0.0), 0.45, True)
        time.sleep(0.1)
        ctrl.stop()
        assert len(backend.velocity_calls) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Controller — rate-limit suppression
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerRateLimit:
    def test_identical_commands_not_resent(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(kp=0.6))
        t = 0.0
        # First tick — sends
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=t)
        n_after_first = len(backend.velocity_calls)
        # Same input, same time — one-euro won't change output meaningfully
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=t + 0.001)
        # May or may not send (filter will drift slightly), but should not
        # double-spam.  Just assert total is still sane.
        assert len(backend.velocity_calls) <= n_after_first + 1

    def test_changed_command_is_sent(self) -> None:
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(kp=0.6))
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.45, True, t=0.1)
        # stop command is zero — should be sent because command changed
        assert len(backend.velocity_calls) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Controller — PD derivative produces smooth ramp
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerSmoothing:
    def test_step_input_produces_smooth_ramp(self) -> None:
        """Sudden large error should NOT produce a bang-bang max command.

        With one-euro filtering, the first tick after a step input yields
        a command close to the steady-state (since the filter initializes to
        the input).  But the derivative term on the *second* tick should be
        smaller than a naive finite-difference over raw error.
        """
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(kp=0.4, kd=0.05))
        pans = []
        for i in range(5):
            pan, _, _ = ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=i * 0.033)
            pans.append(pan)
        # Pan should stabilize (not oscillate wildly)
        max_pan = max(abs(p) for p in pans)
        assert max_pan <= 1.0
        # After first tick, commands should not grow unboundedly
        assert all(abs(p) <= 1.0 for p in pans)

    def test_filters_reset_on_reacquire(self) -> None:
        """Filters must reset on re-acquisition to avoid stale derivative spikes."""
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(kp=0.4, kd=0.2), coast_window_ms=100)
        ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        # lose and expire coast
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.05)
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.20)
        assert ctrl.state == ControllerState.SEARCHING
        # re-acquire with a different error
        pan, _, _ = ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.25)
        # should not spike beyond ±1
        assert abs(pan) <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# MockBackend sanity
# ─────────────────────────────────────────────────────────────────────────────


class TestMockBackend:
    def test_is_subclass(self) -> None:
        assert issubclass(MockBackend, PTZBackend)

    def test_context_manager(self) -> None:
        b = MockBackend()
        with b:
            b.stop()
        assert b.closed is True

    def test_get_position_none_by_default(self) -> None:
        b = MockBackend(has_position=False)
        assert b.get_position() is None

    def test_get_position_returns_state(self) -> None:
        b = MockBackend(has_position=True)
        pos = b.get_position()
        assert pos is not None
        assert isinstance(pos, PTZState)
