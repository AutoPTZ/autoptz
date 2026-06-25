"""Tests for B1 (Tracking Speed preset) and B2 (nonlinear dead-band).

B1: TrackingSpeed enum, SPEED_PROFILES map, apply_speed_profile helper,
    match_speed_preset UI helper.
B2: _soft_deadband with band>0 is value-continuous AND slope-continuous at the
    dead-zone edge; band=0 reproduces the original hard-knee exactly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoptz.config.models import (
    SPEED_PROFILES,
    PTZConfig,
    TrackingSpeed,
    apply_speed_profile,
)
from autoptz.engine.ptz.controller import _smoothstep, _soft_deadband
from autoptz.ui.widgets.properties_panel import match_speed_preset

# ── B1: TrackingSpeed enum ────────────────────────────────────────────────────


class TestTrackingSpeedEnum:
    def test_all_four_members_exist(self) -> None:
        assert TrackingSpeed.CALM.value == "calm"
        assert TrackingSpeed.NORMAL.value == "normal"
        assert TrackingSpeed.FAST.value == "fast"
        assert TrackingSpeed.SPORT.value == "sport"

    def test_is_str_enum(self) -> None:
        # TrackingSpeed inherits from str: its .value is a str and equality with
        # the bare string works.
        assert TrackingSpeed.NORMAL.value == "normal"
        assert TrackingSpeed("normal") == TrackingSpeed.NORMAL
        # str-subclass: the enum member itself compares equal to its value string.
        assert TrackingSpeed.NORMAL == "normal"


# ── B1: SPEED_PROFILES map ────────────────────────────────────────────────────


class TestSpeedProfiles:
    _SIX_KEYS = frozenset(
        ("max_pan_speed", "max_tilt_speed", "kp", "aim_smoothing", "max_accel", "catch_up_speed")
    )

    def test_all_four_presets_present(self) -> None:
        for speed in TrackingSpeed:
            assert speed in SPEED_PROFILES

    def test_each_preset_has_all_six_keys(self) -> None:
        for speed, profile in SPEED_PROFILES.items():
            assert self._SIX_KEYS == frozenset(profile.keys()), f"{speed} missing keys"

    def test_normal_matches_ptzconfig_defaults(self) -> None:
        """NORMAL must equal the existing PTZConfig field defaults exactly."""
        defaults = PTZConfig()
        profile = SPEED_PROFILES[TrackingSpeed.NORMAL]
        assert profile["max_pan_speed"] == pytest.approx(defaults.max_pan_speed)
        assert profile["max_tilt_speed"] == pytest.approx(defaults.max_tilt_speed)
        assert profile["kp"] == pytest.approx(defaults.kp)
        assert profile["aim_smoothing"] == pytest.approx(defaults.aim_smoothing)
        assert profile["max_accel"] == pytest.approx(defaults.max_accel)
        assert profile["catch_up_speed"] == pytest.approx(defaults.catch_up_speed)

    def test_preset_ordering_is_monotone(self) -> None:
        """Speed should increase from CALM → NORMAL → FAST → SPORT."""
        order = [TrackingSpeed.CALM, TrackingSpeed.NORMAL, TrackingSpeed.FAST, TrackingSpeed.SPORT]
        speeds = [SPEED_PROFILES[s]["max_pan_speed"] for s in order]
        kps = [SPEED_PROFILES[s]["kp"] for s in order]
        assert speeds == sorted(speeds)
        assert kps == sorted(kps)

    def test_all_values_in_field_range(self) -> None:
        """Every preset value must be within the PTZConfig field's ge/le bounds."""
        for speed, profile in SPEED_PROFILES.items():
            assert 0.0 <= profile["max_pan_speed"] <= 1.0, f"{speed} max_pan_speed out of range"
            assert 0.0 <= profile["max_tilt_speed"] <= 1.0, f"{speed} max_tilt_speed out of range"
            assert profile["kp"] >= 0.0, f"{speed} kp negative"
            assert 0.0 <= profile["aim_smoothing"] <= 1.0, f"{speed} aim_smoothing out of range"
            assert 0.0 <= profile["max_accel"] <= 50.0, f"{speed} max_accel out of range"
            assert 0.0 <= profile["catch_up_speed"] <= 1.0, f"{speed} catch_up_speed out of range"


# ── B1: apply_speed_profile helper ───────────────────────────────────────────


class TestApplySpeedProfile:
    def test_normal_returns_default_values(self) -> None:
        """apply_speed_profile(cfg, NORMAL) must equal the PTZConfig defaults."""
        base = PTZConfig()
        updated = apply_speed_profile(base, TrackingSpeed.NORMAL)
        profile = SPEED_PROFILES[TrackingSpeed.NORMAL]
        assert updated.max_pan_speed == pytest.approx(profile["max_pan_speed"])
        assert updated.max_tilt_speed == pytest.approx(profile["max_tilt_speed"])
        assert updated.kp == pytest.approx(profile["kp"])
        assert updated.aim_smoothing == pytest.approx(profile["aim_smoothing"])
        assert updated.max_accel == pytest.approx(profile["max_accel"])
        assert updated.catch_up_speed == pytest.approx(profile["catch_up_speed"])

    def test_sets_tracking_speed_field(self) -> None:
        cfg = apply_speed_profile(PTZConfig(), TrackingSpeed.FAST)
        assert cfg.tracking_speed == TrackingSpeed.FAST

    def test_preserves_other_fields(self) -> None:
        """Fields not in the six-knob profile must be unchanged."""
        base = PTZConfig(invert_pan=True, backend="ndi", deadzone_x=0.08)
        updated = apply_speed_profile(base, TrackingSpeed.CALM)
        assert updated.invert_pan is True
        assert updated.backend == "ndi"
        assert updated.deadzone_x == pytest.approx(0.08)

    def test_returns_new_frozen_instance(self) -> None:
        """apply_speed_profile must return a *new* frozen PTZConfig, not mutate."""
        base = PTZConfig()
        updated = apply_speed_profile(base, TrackingSpeed.SPORT)
        assert updated is not base
        # Original still has default tracking_speed=None
        assert base.tracking_speed is None
        # Confirm frozen (can't set attributes)
        with pytest.raises((ValidationError, TypeError)):
            updated.kp = 99.0  # type: ignore[misc]

    def test_all_four_presets_apply(self) -> None:
        base = PTZConfig()
        for speed in TrackingSpeed:
            cfg = apply_speed_profile(base, speed)
            profile = SPEED_PROFILES[speed]
            assert cfg.kp == pytest.approx(profile["kp"])

    def test_round_trip_via_model_dump(self) -> None:
        """apply_speed_profile result must survive a model_dump / model_validate round-trip."""
        cfg = apply_speed_profile(PTZConfig(), TrackingSpeed.FAST)
        data = cfg.model_dump()
        reloaded = PTZConfig.model_validate(data)
        assert reloaded.tracking_speed == TrackingSpeed.FAST
        assert reloaded.kp == pytest.approx(cfg.kp)


# ── B1: PTZConfig tracking_speed field ───────────────────────────────────────


class TestPTZConfigTrackingSpeedField:
    def test_default_is_none(self) -> None:
        assert PTZConfig().tracking_speed is None

    def test_can_set_to_each_preset(self) -> None:
        for speed in TrackingSpeed:
            cfg = PTZConfig(tracking_speed=speed)
            assert cfg.tracking_speed == speed

    def test_accepts_string_value(self) -> None:
        """The field must coerce a string "calm" → TrackingSpeed.CALM (str-enum)."""
        cfg = PTZConfig.model_validate({"tracking_speed": "calm"})
        assert cfg.tracking_speed == TrackingSpeed.CALM

    def test_accepts_none(self) -> None:
        cfg = PTZConfig.model_validate({"tracking_speed": None})
        assert cfg.tracking_speed is None


# ── B1: PTZConfig nonlinear_band field ───────────────────────────────────────


class TestPTZConfigNonlinearBandField:
    def test_default_is_positive(self) -> None:
        """The default band must be > 0 so new installs get smooth exit."""
        assert PTZConfig().nonlinear_band > 0.0

    def test_zero_is_valid(self) -> None:
        cfg = PTZConfig(nonlinear_band=0.0)
        assert cfg.nonlinear_band == 0.0

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            PTZConfig(nonlinear_band=-0.01)


# ── B1: match_speed_preset UI helper ─────────────────────────────────────────


class TestMatchSpeedPreset:
    def _ptz_dict_for(self, speed: TrackingSpeed) -> dict[str, float]:
        return dict(SPEED_PROFILES[speed])

    def test_normal_profile_matches_normal(self) -> None:
        assert match_speed_preset(self._ptz_dict_for(TrackingSpeed.NORMAL)) == TrackingSpeed.NORMAL

    def test_calm_profile_matches_calm(self) -> None:
        assert match_speed_preset(self._ptz_dict_for(TrackingSpeed.CALM)) == TrackingSpeed.CALM

    def test_fast_profile_matches_fast(self) -> None:
        assert match_speed_preset(self._ptz_dict_for(TrackingSpeed.FAST)) == TrackingSpeed.FAST

    def test_sport_profile_matches_sport(self) -> None:
        assert match_speed_preset(self._ptz_dict_for(TrackingSpeed.SPORT)) == TrackingSpeed.SPORT

    def test_custom_values_return_none(self) -> None:
        d = self._ptz_dict_for(TrackingSpeed.NORMAL)
        d["kp"] = 0.42  # not matching any preset
        assert match_speed_preset(d) is None

    def test_empty_dict_does_not_crash(self) -> None:
        # No keys → falls back to each profile's own defaults for comparison →
        # always matches the first preset (CALM) since an empty dict trivially
        # satisfies any profile.  The important thing is no exception is raised.
        result = match_speed_preset({})
        assert result is not None  # some preset will always match an empty dict

    def test_partial_dict_uses_profile_default_for_missing(self) -> None:
        # Supplying only some keys: missing keys default to profile value → match.
        d = {"max_pan_speed": 0.7}  # only one key, NORMAL's value
        # With all other keys defaulting to NORMAL's profile values → matches.
        assert match_speed_preset(d) == TrackingSpeed.NORMAL


# ── B2: _smoothstep helper ────────────────────────────────────────────────────


class TestSmoothstep:
    def test_zero_at_zero(self) -> None:
        assert _smoothstep(0.0) == pytest.approx(0.0)

    def test_one_at_one(self) -> None:
        assert _smoothstep(1.0) == pytest.approx(1.0)

    def test_half_at_half(self) -> None:
        assert _smoothstep(0.5) == pytest.approx(0.5)

    def test_clamped_below(self) -> None:
        assert _smoothstep(-1.0) == pytest.approx(0.0)

    def test_clamped_above(self) -> None:
        assert _smoothstep(2.0) == pytest.approx(1.0)

    def test_zero_derivative_at_endpoints(self) -> None:
        """Smoothstep must have zero derivative at 0 and 1 (C1-continuity anchor)."""
        eps = 1e-5
        d_at_0 = (_smoothstep(eps) - _smoothstep(0.0)) / eps
        d_at_1 = (_smoothstep(1.0) - _smoothstep(1.0 - eps)) / eps
        assert abs(d_at_0) < 0.01
        assert abs(d_at_1) < 0.01

    def test_monotone_increasing(self) -> None:
        pts = [_smoothstep(t / 100.0) for t in range(101)]
        assert all(b >= a for a, b in zip(pts, pts[1:], strict=False))


# ── B2: _soft_deadband with band=0 (backward compat) ─────────────────────────


class TestSoftDeadbandHardKnee:
    """With band=0 the function must reproduce the original behaviour exactly."""

    def test_zero_within_deadband(self) -> None:
        assert _soft_deadband(0.03, 0.05, 0.0) == pytest.approx(0.0)
        assert _soft_deadband(-0.03, 0.05, 0.0) == pytest.approx(0.0)

    def test_at_boundary(self) -> None:
        assert _soft_deadband(0.05, 0.05, 0.0) == pytest.approx(0.0)

    def test_linear_excess_outside(self) -> None:
        # d=0.10, dead=0.05 → excess=0.05
        assert _soft_deadband(0.10, 0.05, 0.0) == pytest.approx(0.05)
        assert _soft_deadband(-0.10, 0.05, 0.0) == pytest.approx(-0.05)

    def test_sign_preserving(self) -> None:
        pos = _soft_deadband(0.2, 0.05, 0.0)
        neg = _soft_deadband(-0.2, 0.05, 0.0)
        assert pos > 0.0
        assert neg < 0.0
        assert pos == pytest.approx(-neg)

    def test_zero_dead_is_identity(self) -> None:
        for d in (-0.5, 0.0, 0.7):
            assert _soft_deadband(d, 0.0, 0.0) == pytest.approx(d)

    def test_default_band_is_zero_compatible(self) -> None:
        """Calling without explicit band= argument must behave as band=0."""
        assert _soft_deadband(0.1, 0.05) == pytest.approx(_soft_deadband(0.1, 0.05, 0.0))


# ── B2: _soft_deadband with band>0 (nonlinear transition) ────────────────────


class TestSoftDeadbandNonlinear:
    """With band>0 the function must be value-continuous AND slope-continuous
    (C1) at the dead-zone edge."""

    _DEAD = 0.05
    _BAND = 0.04

    def _f(self, d: float) -> float:
        return _soft_deadband(d, self._DEAD, self._BAND)

    # Value-continuity: f(dead-ε) → 0, f(dead+ε) → 0 (no step at edge)

    def test_zero_just_inside_edge(self) -> None:
        eps = 1e-6
        assert self._f(self._DEAD - eps) == pytest.approx(0.0, abs=1e-6)

    def test_near_zero_just_outside_edge(self) -> None:
        eps = 1e-6
        assert abs(self._f(self._DEAD + eps)) < 1e-4  # smoothstep(ε/band) ≈ 0

    # Slope-continuity: slope just outside the edge ≈ 0 (not the full linear slope)

    def test_slope_near_zero_at_dead_edge(self) -> None:
        """The slope at the dead-zone edge must be ≈ 0 (not the hard-knee value 1)."""
        eps = 1e-4
        slope = (self._f(self._DEAD + eps) - self._f(self._DEAD)) / eps
        # With band>0 and smoothstep zero-derivative at 0, slope must be near 0.
        assert abs(slope) < 0.05

    def test_slope_approaches_one_beyond_band(self) -> None:
        """Beyond dead+band the slope must approach 1 (pure linear excess)."""
        eps = 1e-4
        d_far = self._DEAD + self._BAND + 0.05
        slope = (self._f(d_far + eps) - self._f(d_far)) / eps
        assert abs(slope - 1.0) < 0.01

    # Far output approaches linear excess

    def test_far_output_matches_linear_excess(self) -> None:
        """Well beyond the transition zone, output == |d| - dead (linear excess)."""
        d = self._DEAD + self._BAND + 0.10
        expected_linear = d - self._DEAD
        assert self._f(d) == pytest.approx(expected_linear, abs=1e-6)

    def test_negative_side_symmetric(self) -> None:
        for d in (0.06, 0.08, 0.12):
            assert self._f(d) == pytest.approx(-self._f(-d), abs=1e-9)

    # band=0 exact equivalence

    def test_band_zero_matches_original_outside(self) -> None:
        """band=0 must be exactly equivalent to the original hard-knee for a range of d."""
        for d in (-0.2, -0.1, 0.0, 0.08, 0.15):
            assert _soft_deadband(d, self._DEAD, 0.0) == pytest.approx(
                _soft_deadband(d, self._DEAD), abs=1e-12
            )

    # No discontinuity in value across the transition zone

    def test_value_continuous_across_transition(self) -> None:
        """Sample 200 points through the transition; max step must be tiny."""
        prev = self._f(self._DEAD)
        max_step = 0.0
        for i in range(1, 201):
            d = self._DEAD + i * self._BAND / 200.0
            cur = self._f(d)
            max_step = max(max_step, abs(cur - prev))
            prev = cur
        # 200 steps across band=0.04 → each step ≈ 0.0002; allow 10× headroom.
        assert max_step < 0.002

    def test_within_transition_output_less_than_linear_excess(self) -> None:
        """Inside the band the nonlinear output must be less than the linear excess
        (the ramp suppresses the initial gain)."""
        d = self._DEAD + self._BAND * 0.5  # midpoint of transition
        linear = d - self._DEAD
        nonlinear = self._f(d)
        assert nonlinear < linear
