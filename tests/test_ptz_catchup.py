"""Dynamic "catch-up" tracking speed.

The PTZ control loop scales each axis' speed ceiling by an error-proportional
boost: far from the target framing the camera speeds UP to catch the subject;
near the setpoint it stays at the configured speed for precision.  These tests
cover the boost curve, the config control, and the end-to-end effect on the
controller command.
"""

from __future__ import annotations

import pytest

from autoptz.config.models import PTZConfig
from autoptz.engine.ptz.base import PTZBackend, PTZCaps
from autoptz.engine.ptz.controller import (
    _DYN_E_REF,
    PTZController,
    _catch_up_boost,
)


class _Backend(PTZBackend):
    def __init__(self) -> None:
        super().__init__()
        self.caps = PTZCaps(continuous_pan_tilt=True, continuous_zoom=True)

    def move_velocity(self, pan, tilt, zoom=0.0):  # pragma: no cover - trivial
        pass

    def stop(self):  # pragma: no cover - trivial
        pass

    def goto_preset(self, idx):  # pragma: no cover - trivial
        pass

    def save_preset(self, idx):  # pragma: no cover - trivial
        pass

    def close(self):  # pragma: no cover - trivial
        pass


def _cfg(**kw):
    # Strip the control loop of confounders so the boost is the only variable:
    # no safe-zone setpoint shaping, no deadzone, no slew, no osc damping.
    base = {
        "safe_zone_enabled": False,
        "deadzone_x": 0.0,
        "deadzone_y": 0.0,
        "max_accel": 0.0,
        "osc_guard": False,
        "aim_smoothing": 0.0,
        "kp": 0.8,
        "kd": 0.0,
        "kv": 0.0,
        "max_pan_speed": 0.7,
        "max_tilt_speed": 0.7,
    }
    base.update(kw)
    return PTZConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Boost curve
# ─────────────────────────────────────────────────────────────────────────────


class TestCatchUpBoost:
    def test_strength_zero_is_unity_everywhere(self) -> None:
        assert _catch_up_boost(0.0, 0.0) == 1.0
        assert _catch_up_boost(0.5, 0.0) == 1.0
        assert _catch_up_boost(1.0, 0.0) == 1.0

    def test_no_boost_at_setpoint(self) -> None:
        # Centred subject (zero error) → configured speed, never boosted.
        assert _catch_up_boost(0.0, 1.0) == 1.0

    def test_boost_grows_with_error(self) -> None:
        b_small = _catch_up_boost(0.1, 0.6)
        b_large = _catch_up_boost(0.5, 0.6)
        assert b_large > b_small > 1.0

    def test_boost_saturates_beyond_reference(self) -> None:
        # At/after the reference error the boost is maxed and stops growing.
        at_ref = _catch_up_boost(_DYN_E_REF, 0.6)
        beyond = _catch_up_boost(1.0, 0.6)
        assert at_ref == pytest.approx(beyond)

    def test_sign_of_error_does_not_matter(self) -> None:
        assert _catch_up_boost(-0.4, 0.6) == pytest.approx(_catch_up_boost(0.4, 0.6))


# ─────────────────────────────────────────────────────────────────────────────
# Config control
# ─────────────────────────────────────────────────────────────────────────────


class TestCatchUpConfig:
    def test_default_is_on(self) -> None:
        assert PTZConfig().catch_up_speed == pytest.approx(0.6)

    def test_configurable_within_bounds(self) -> None:
        assert PTZConfig(catch_up_speed=1.0).catch_up_speed == 1.0
        assert PTZConfig(catch_up_speed=0.0).catch_up_speed == 0.0

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            PTZConfig(catch_up_speed=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end effect on the controller command
# ─────────────────────────────────────────────────────────────────────────────


class TestCatchUpControllerEffect:
    def test_far_subject_pans_faster_with_catch_up(self) -> None:
        # Same large horizontal error: catch-up ON must command a faster pan than OFF.
        off = PTZController(_Backend(), _cfg(catch_up_speed=0.0))
        on = PTZController(_Backend(), _cfg(catch_up_speed=0.6))
        pan_off, _, _ = off.step((0.8, 0.0), (0.0, 0.0), 0.4, True, t=0.0)
        pan_on, _, _ = on.step((0.8, 0.0), (0.0, 0.0), 0.4, True, t=0.0)
        assert pan_on > pan_off > 0.0

    def test_catch_up_is_per_axis(self) -> None:
        # Error only horizontal → pan is boosted, tilt stays put (≈0) regardless.
        on = PTZController(_Backend(), _cfg(catch_up_speed=0.8))
        pan, tilt, _ = on.step((0.8, 0.0), (0.0, 0.0), 0.4, True, t=0.0)
        assert pan > 0.0
        assert tilt == pytest.approx(0.0, abs=1e-6)

    def test_zero_catch_up_matches_baseline(self) -> None:
        # catch_up_speed=0 must be byte-identical to the pre-feature command.
        a = PTZController(_Backend(), _cfg(catch_up_speed=0.0))
        b = PTZController(_Backend(), _cfg(catch_up_speed=0.0))
        pa = a.step((0.6, -0.3), (0.0, 0.0), 0.4, True, t=0.0)
        pb = b.step((0.6, -0.3), (0.0, 0.0), 0.4, True, t=0.0)
        assert pa == pytest.approx(pb)
