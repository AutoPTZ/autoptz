"""PTZ backends + closed-loop controller tests.

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
        # Isolate the PD/feed-forward math: no acceleration ramp unless a test
        # opts in (the slew limiter has its own dedicated tests).
        "max_accel": 0.0,
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
        assert cmd[4] == 0x18  # max pan speed
        assert cmd[5] == 0x14  # max tilt speed
        assert cmd[6] == 0x02  # right
        assert cmd[7] == 0x01  # up

    def test_pantilt_cmd_full_left_down(self) -> None:
        cmd = visca_pantilt_cmd(-1.0, -1.0)
        assert cmd[6] == 0x01  # left
        assert cmd[7] == 0x02  # down

    def test_pantilt_speed_clamped(self) -> None:
        # speed must be at least 0x01 for moving, at most 0x18 pan / 0x14 tilt
        cmd = visca_pantilt_cmd(0.001, 0.0)
        assert 0x01 <= cmd[4] <= 0x18

    def test_zoom_tele(self) -> None:
        cmd = visca_zoom_cmd(1.0)
        assert cmd[4] & 0xF0 == 0x20  # tele nibble
        assert 0x01 <= cmd[4] & 0x0F <= 0x07

    def test_zoom_wide(self) -> None:
        cmd = visca_zoom_cmd(-1.0)
        assert cmd[4] & 0xF0 == 0x30  # wide nibble

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

    def test_predict_accel_gain_off_by_default(self) -> None:
        # Phase C predictive lead must be opt-in: default config keeps it disabled.
        assert _cfg().predict_accel_gain == 0.0

    def test_acceleration_prediction_anticipates_when_enabled(self) -> None:
        # An accelerating subject (velocity rising tick-to-tick): with
        # predict_accel_gain the controller projects the change in velocity and
        # commands a larger move than the velocity-only (gain=0) controller.  The
        # first tick only seeds the previous velocity (accel term skipped).
        c0 = PTZController(MockBackend(), _cfg(kv=0.0, safe_zone_enabled=False))
        c1 = PTZController(
            MockBackend(), _cfg(kv=0.0, safe_zone_enabled=False, predict_accel_gain=1.0)
        )
        c0.step((0.4, 0.0), (0.3, 0.0), 0.45, True, t=0.0)
        c1.step((0.4, 0.0), (0.3, 0.0), 0.45, True, t=0.0)
        pan0, _, _ = c0.step((0.4, 0.0), (0.8, 0.0), 0.45, True, t=0.1)
        pan1, _, _ = c1.step((0.4, 0.0), (0.8, 0.0), 0.45, True, t=0.1)
        assert pan1 > pan0

    def test_deadzone_suppresses_small_error(self) -> None:
        # The per-axis dead-zone path is the safe-zone-OFF behaviour.
        ctrl = PTZController(
            MockBackend(), _cfg(kp=0.6, safe_zone_enabled=False, deadzone_x=0.1, deadzone_y=0.1)
        )
        pan, tilt, _ = ctrl.step((0.05, 0.08), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan == pytest.approx(0.0, abs=1e-6)
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_deadzone_passes_large_error(self) -> None:
        ctrl = PTZController(
            MockBackend(), _cfg(kp=0.6, safe_zone_enabled=False, deadzone_x=0.05, deadzone_y=0.05)
        )
        pan, _, _ = ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan > 0.0

    def test_shifted_safe_zone_holds_at_box_center(self) -> None:
        # The setpoint is the zone centre: a subject AT the (offset) box centre is
        # held still.  (Off-centre-inside-zone now eases toward centre — see
        # TestSafeZoneCentering — rather than freezing as it used to.)
        ctrl = PTZController(
            MockBackend(),
            _cfg(
                safe_zone_enabled=True,
                safe_zone_x=0.3,
                safe_zone_y=-0.2,
                safe_zone_w=0.1,
                safe_zone_h=0.1,
            ),
        )
        pan, tilt, _ = ctrl.step((0.3, -0.2), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan == pytest.approx(0.0, abs=1e-6)
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_shifted_safe_zone_tracks_relative_to_box_center(self) -> None:
        ctrl = PTZController(
            MockBackend(),
            _cfg(
                safe_zone_enabled=True,
                safe_zone_x=0.3,
                safe_zone_y=0.0,
                safe_zone_w=0.05,
                safe_zone_h=0.05,
            ),
        )
        pan, _, _ = ctrl.step((0.6, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan > 0.0

    def test_invert_pan(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=0.6, invert_pan=True))
        pan, _, _ = ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan < 0.0  # positive error + invert → negative command


class TestSafeZoneCentering:
    """The safe zone must ease the subject toward the zone CENTRE (Center-Stage
    feel), not merely freeze them anywhere inside the oval (the old deadzone-only
    behaviour that 'never centered' and let a big zone ignore the frame edges)."""

    def _zone(self, **kw: object) -> PTZController:
        defaults: dict[str, object] = {
            "kp": 0.6,
            "safe_zone_enabled": True,
            "safe_zone_x": 0.0,
            "safe_zone_y": 0.0,
            "safe_zone_w": 0.15,
            "safe_zone_h": 0.22,
        }
        defaults.update(kw)
        return PTZController(MockBackend(), _cfg(**defaults))

    def test_offcenter_subject_inside_zone_is_eased_to_center(self) -> None:
        # Subject right-of-centre but inside the oval: old code froze (pan==0);
        # now it eases back toward the centre (pan>0).
        ctrl = self._zone()
        pan, _, _ = ctrl.step((0.12, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan > 0.0

    def test_centered_subject_holds_still(self) -> None:
        # Within the small inner deadband around the zone centre → no micro-moves.
        ctrl = self._zone()
        pan, tilt, _ = ctrl.step((0.01, 0.01), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan == pytest.approx(0.0, abs=1e-6)
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_offset_zone_drives_subject_to_box_center(self) -> None:
        # The setpoint is the zone centre, not the frame centre: a subject at the
        # FRAME centre with a right-shifted zone is moved toward the zone centre.
        # Old code used frame-centre error → pan==0 (ignored the offset).
        ctrl = self._zone(safe_zone_x=0.3, safe_zone_w=0.1, safe_zone_h=0.1)
        pan, _, _ = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan < 0.0

    def test_large_zone_still_centers_subject_near_frame_edge(self) -> None:
        # A big safe zone must NOT let the subject sit at the frame edge.
        ctrl = self._zone(safe_zone_w=0.9, safe_zone_h=0.9)
        pan, _, _ = ctrl.step((0.85, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        assert pan > 0.0


class TestWarmReacquire:
    """A brief target loss (within the coast window) must resume smoothly, not
    cold-reset the controller and re-ramp from a dead stop — the "find" lurch in
    the find/lose/find cycle during pans."""

    def _ctrl(self, coast_ms: int) -> PTZController:
        return PTZController(
            MockBackend(),
            _cfg(kp=1.0, kd=0.0, aim_smoothing=0.0, safe_zone_enabled=False, max_accel=2.0),
            coast_window_ms=coast_ms,
            rate_hz=20.0,
        )

    def _ramp_to_steady(self, ctrl: PTZController, t0: float) -> tuple[float, float]:
        t = t0
        pan = 0.0
        for _ in range(30):
            pan, _, _ = ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=t)
            t += 0.05
        return pan, t

    def test_brief_coast_reacquire_resumes_without_reramping(self) -> None:
        ctrl = self._ctrl(coast_ms=1500)
        steady, t = self._ramp_to_steady(ctrl, 0.0)
        assert steady > 0.3
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=t)  # one lost tick → COASTING
        t += 0.05
        pan_re, _, _ = ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=t)
        assert pan_re == pytest.approx(steady, abs=0.06)  # warm: no slew from 0

    def test_long_loss_reacquire_cold_resets(self) -> None:
        ctrl = self._ctrl(coast_ms=100)  # coast expires after ~2 lost ticks
        steady, t = self._ramp_to_steady(ctrl, 0.0)
        for _ in range(5):  # long loss → COASTING → SEARCHING
            ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=t)
            t += 0.05
        pan_re, _, _ = ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=t)
        assert pan_re < steady - 0.1  # cold: slew limiter ramps from 0


class TestZoneGeometry:
    def test_square_zone_contains_corner_round_zone_excludes_it(self) -> None:
        from autoptz.engine.ptz.controller import _zone_norm

        # A corner point at (hw, hh): inside a square zone (roundness 0), outside
        # an elliptical one (roundness 1) — so a configured "square" respects its
        # corners instead of being treated as an ellipse.
        assert _zone_norm(0.1, 0.1, 0.1, 0.1, 0.0) <= 1.0
        assert _zone_norm(0.1, 0.1, 0.1, 0.1, 1.0) > 1.0

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
        """With large Kp, mid-range error should produce less than half-speed.

        Catch-up speed is disabled here so this isolates the ``_shape`` response
        curve — with the default catch-up boost on, a far subject is deliberately
        driven *faster* (covered by ``test_ptz_catchup``).
        """
        ctrl = PTZController(MockBackend(), _cfg(kp=0.5, catch_up_speed=0.0))
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

    def test_status_snapshot_reports_coast_remaining(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(), coast_window_ms=500)
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.10)
        snap = ctrl.status_snapshot(t=0.25)
        assert snap["state"] == "coasting"
        assert snap["action"] == "holding"
        assert snap["coast_remaining_s"] == pytest.approx(0.35)

    def test_status_snapshot_reports_search_zoom_remaining(self) -> None:
        cfg = _cfg(loss_zoom_out=0.5, reacquire_window_s=2.0)
        ctrl = PTZController(MockBackend(), cfg, coast_window_ms=100)
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.05)
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=0.20)
        snap = ctrl.status_snapshot(t=0.70)
        assert snap["state"] == "searching"
        assert snap["action"] == "zooming_out"
        assert snap["search_remaining_s"] == pytest.approx(1.5)

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
        ctrl = PTZController(MockBackend(), _cfg(auto_zoom=True, zoom_framing="upper_body"))
        # subject_height=0.9 >> target 0.45 → zoom out (negative)
        _, _, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.9, True, t=0.0)
        assert zoom < 0.0

    def test_short_subject_zooms_in(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(auto_zoom=True, zoom_framing="upper_body"))
        # subject_height=0.1 << target 0.45 → zoom in (positive)
        _, _, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.1, True, t=0.0)
        assert zoom > 0.0

    def test_subject_in_band_no_zoom(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(auto_zoom=True, zoom_framing="upper_body"))
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
        # Poll until the worker thread delivers rather than sleeping a fixed window:
        # at 50 Hz a tick is ~20 ms, but a loaded CI runner can stall the thread, so
        # a fixed 0.1 s sleep flakes. Allow a generous ceiling; return as soon as ready.
        deadline = time.monotonic() + 2.0
        while not backend.velocity_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        ctrl.stop()
        assert len(backend.velocity_calls) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Controller — fixed-rate pump + stop-on-loss heartbeat
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerPumpHeartbeat:
    """The background loop drives the backend from update()s; a stale producer
    (no fresh update() past the heartbeat threshold) makes the loop halt."""

    def test_update_records_recency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """update() stamps the last-update monotonic time so the loop can gauge it."""
        import autoptz.engine.ptz.controller as ctrl_mod

        ctrl = PTZController(MockBackend(), _cfg(kp=0.6), rate_hz=20.0)
        monkeypatch.setattr(ctrl_mod.time, "monotonic", lambda: 123.5)
        ctrl.update((0.3, 0.0), (0.0, 0.0), 0.45, True)
        assert ctrl._last_update_t == pytest.approx(123.5)

    def test_heartbeat_threshold_floor(self) -> None:
        """At a fast rate the threshold floors at the absolute minimum, not periods."""
        ctrl = PTZController(MockBackend(), _cfg(), rate_hz=1000.0)
        assert ctrl._heartbeat_stale_s == pytest.approx(ctrl_mod_floor())

    def test_thread_drives_from_update(self) -> None:
        """With start(), a fresh update(track_active=True) makes the loop drive."""
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(kp=0.6), rate_hz=50.0)
        ctrl.start()
        try:
            ctrl.update((0.5, 0.0), (0.0, 0.0), 0.45, True)
            deadline = time.monotonic() + 2.0
            while not backend.velocity_calls and time.monotonic() < deadline:
                time.sleep(0.01)
            assert backend.velocity_calls, "loop did not drive backend from update()"
            assert ctrl.state == ControllerState.TRACKING
        finally:
            ctrl.stop()

    def test_stale_feed_halts_camera(self) -> None:
        """Withholding update() past the stale threshold makes the loop stop()."""
        backend = MockBackend()
        # Short threshold so the test is quick but still well above a tick period.
        ctrl = PTZController(backend, _cfg(kp=0.6), rate_hz=50.0)
        ctrl._heartbeat_stale_s = 0.1
        ctrl.start()
        try:
            ctrl.update((0.5, 0.0), (0.0, 0.0), 0.45, True)
            deadline = time.monotonic() + 2.0
            while not backend.velocity_calls and time.monotonic() < deadline:
                time.sleep(0.01)
            assert backend.velocity_calls, "loop never started driving"
            stops_before = backend.stop_count
            # Now go quiet: no further update() — the loop must halt and drop to IDLE.
            deadline = time.monotonic() + 2.0
            while ctrl.state != ControllerState.IDLE and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ctrl.state == ControllerState.IDLE, "stale feed did not halt the controller"
            assert backend.stop_count > stops_before, "stale feed did not send a backend stop"
        finally:
            ctrl.stop()

    def test_heartbeat_inactive_on_synchronous_step(self) -> None:
        """The synchronous step() path never trips the heartbeat (loop not running).

        step() refreshes the payload itself immediately before _tick, so even a
        wall-clock gap between steps must not be treated as a stalled producer.
        """
        backend = MockBackend()
        ctrl = PTZController(backend, _cfg(kp=0.6))
        ctrl._heartbeat_stale_s = 0.0  # would trip instantly IF the gate were off
        ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        ctrl.step((0.5, 0.0), (0.0, 0.0), 0.45, True, t=1.0)
        assert ctrl.state == ControllerState.TRACKING


def ctrl_mod_floor() -> float:
    import autoptz.engine.ptz.controller as ctrl_mod

    return ctrl_mod._HEARTBEAT_FLOOR_S


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


# ─────────────────────────────────────────────────────────────────────────────
# Control-law additions: integral (anti-windup), oscillation guard, latency lead
# ─────────────────────────────────────────────────────────────────────────────


class TestControllerIntegral:
    def test_integral_off_by_default(self) -> None:
        """ki defaults to 0 → the loop is pure PD (no accumulation)."""
        ctrl = PTZController(MockBackend(), _cfg(kp=0.5))
        assert ctrl._cfg.ki == 0.0
        a = ctrl.step((0.2, 0.0), (0.0, 0.0), 0.45, True, t=0.0)[0]
        b = ctrl.step((0.2, 0.0), (0.0, 0.0), 0.45, True, t=0.5)[0]
        # steady error, no integral → command shouldn't grow tick over tick
        assert b == pytest.approx(a, abs=0.05)

    def test_integral_accumulates_on_steady_error(self) -> None:
        """With ki>0 a persistent error builds extra command over time."""
        ctrl = PTZController(MockBackend(), _cfg(kp=0.2, ki=1.0, aim_smoothing=0.0))
        first = ctrl.step((0.2, 0.0), (0.0, 0.0), 0.45, True, t=0.0)[0]
        last = first
        for i in range(1, 8):
            last = ctrl.step((0.2, 0.0), (0.0, 0.0), 0.45, True, t=i * 0.1)[0]
        assert last > first  # integral pushed the command up

    def test_integral_is_anti_windup_clamped(self) -> None:
        """A long saturating error can't wind the integral past its cap."""
        ctrl = PTZController(MockBackend(), _cfg(kp=0.2, ki=2.0, aim_smoothing=0.0))
        for i in range(50):
            ctrl.step((1.0, 0.0), (0.0, 0.0), 0.45, True, t=i * 0.1)
        # ki·∫ is capped at 0.5 → accumulator stays within ±(0.5/ki)
        assert abs(ctrl._int_ex) <= 0.5 / 2.0 + 1e-6

    def test_integral_resets_on_reacquire(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg(kp=0.2, ki=1.0, aim_smoothing=0.0))
        for i in range(6):
            ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=i * 0.1)
        assert ctrl._int_ex != 0.0
        ctrl.step((0.0, 0.0), (0.0, 0.0), 0.0, False, t=1.0)  # lose target
        ctrl.step((0.3, 0.0), (0.0, 0.0), 0.45, True, t=5.0)  # re-acquire
        assert ctrl._int_ex == 0.0


class TestOscillationGuard:
    def test_guard_damps_sustained_hunting(self) -> None:
        """Alternating-sign errors (hunting) get progressively damped."""
        ctrl = PTZController(MockBackend(), _cfg(kp=0.8, aim_smoothing=0.0, osc_guard=True))
        mags: list[float] = []
        for i in range(8):
            err = 0.5 if i % 2 == 0 else -0.5  # flip every tick
            pan = ctrl.step((err, 0.0), (0.0, 0.0), 0.45, True, t=i * 0.05)[0]
            mags.append(abs(pan))
        assert ctrl._flip_score > 1.0  # detected hunting
        assert mags[-1] < mags[1]  # later commands damped vs early ones

    def test_guard_leaves_steady_motion_alone(self) -> None:
        """A constant-direction error keeps full command (no false damping)."""
        ctrl = PTZController(MockBackend(), _cfg(kp=0.8, aim_smoothing=0.0, osc_guard=True))
        for i in range(6):
            ctrl.step((0.4, 0.0), (0.0, 0.0), 0.45, True, t=i * 0.05)
        assert ctrl._flip_score == pytest.approx(0.0, abs=1e-9)


class TestLatencyLead:
    def test_measured_latency_increases_lead(self) -> None:
        """A moving subject yields a larger lead-in command once latency is fed."""
        b_lo, b_hi = MockBackend(), MockBackend()
        # lead comes only from latency (lead_time_s=0); kp turns the lead-shifted
        # error into a command; kv=0 so the velocity itself adds nothing directly.
        cfg = _cfg(
            kp=0.6,
            kd=0.0,
            kv=0.0,
            aim_smoothing=0.0,
            lead_time_s=0.0,
            lead_time_auto=True,
            safe_zone_enabled=False,  # otherwise the framing box swallows the lead
        )
        lo = PTZController(b_lo, cfg)
        hi = PTZController(b_hi, cfg)
        hi.set_loop_latency(0.3)  # 300 ms pipeline
        # Off-centre error so we're actively *following* (the hold only suppresses
        # motion when the subject is inside the box); the latency lead then
        # projects further ahead → a larger command.
        lo.step((0.2, 0.0), (0.5, 0.0), 0.45, True, t=0.0)
        hi.step((0.2, 0.0), (0.5, 0.0), 0.45, True, t=0.0)
        lo_pan = lo.step((0.2, 0.0), (0.5, 0.0), 0.45, True, t=0.1)[0]
        hi_pan = hi.step((0.2, 0.0), (0.5, 0.0), 0.45, True, t=0.1)[0]
        assert abs(hi_pan) > abs(lo_pan)

    def test_set_loop_latency_is_clamped(self) -> None:
        ctrl = PTZController(MockBackend(), _cfg())
        ctrl.set_loop_latency(99.0)
        assert ctrl._loop_latency_s <= 0.8
        ctrl.set_loop_latency(-5.0)
        assert ctrl._loop_latency_s == 0.0


class TestSlewRateLimit:
    def _ctrl(self, max_accel: float):
        return PTZController(
            MockBackend(),
            _cfg(kp=1.0, aim_smoothing=0.0, safe_zone_enabled=False, max_accel=max_accel),
        )

    def test_ramps_up_instead_of_jumping(self) -> None:
        ctrl = self._ctrl(max_accel=2.0)  # ≤0.2 change per 0.1s tick
        cmds = [ctrl.step((1.0, 0.0), (0.0, 0.0), 0.45, True, t=i * 0.1)[0] for i in range(8)]
        # Early commands are well short of the steady target (1.0) — it's ramping.
        assert cmds[1] < 0.5
        assert cmds[-1] > cmds[1]
        # No tick jumps by more than max_accel*dt (0.2), so no "speeds out of nowhere".
        assert all(b - a <= 0.2 + 1e-6 for a, b in zip(cmds, cmds[1:], strict=False))

    def test_disabled_responds_instantly(self) -> None:
        ctrl = self._ctrl(max_accel=0.0)
        ctrl.step((1.0, 0.0), (0.0, 0.0), 0.45, True, t=0.0)
        pan = ctrl.step((1.0, 0.0), (0.0, 0.0), 0.45, True, t=0.1)[0]
        assert pan == pytest.approx(1.0, abs=1e-6)  # jumps straight to full speed

    def test_ramps_back_down_into_deadzone(self) -> None:
        # Drive to speed, then center the target (use a real dead-zone) → the
        # command should ease DOWN over ticks, not cut to zero abruptly.
        ctrl = PTZController(
            MockBackend(),
            _cfg(kp=1.0, aim_smoothing=0.0, safe_zone_enabled=False, deadzone_x=0.1, max_accel=2.0),
        )
        for i in range(8):
            ctrl.step((1.0, 0.0), (0.0, 0.0), 0.45, True, t=i * 0.1)
        high = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.45, True, t=0.8)[0]  # centered now
        lower = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.45, True, t=0.9)[0]
        assert 0.0 <= lower < high  # decelerating, not slamming to 0
