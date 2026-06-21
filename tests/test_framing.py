"""Unit tests for autoptz.engine.pipeline.framing.

Pure aim-point / subject-height math + EMA smoothing for pose-stable framing.
No model required — keypoints are synthetic, so these run anywhere.
"""

from __future__ import annotations

import math

from autoptz.engine.pipeline.framing import (
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_NOSE,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    AimSmoother,
    Keypoint,
    body_aim_point,
    head_point,
    hip_midpoint,
    shoulder_midpoint,
    subject_height_from_pose,
    torso_aim_point,
)

# COCO-17 has 17 keypoints; build a full list with a low-conf default and fill
# in the torso points we care about.
_LOW = Keypoint(0.0, 0.0, 0.0)


def _pose(
    *,
    ls: tuple[float, float, float] | None = None,
    rs: tuple[float, float, float] | None = None,
    lh: tuple[float, float, float] | None = None,
    rh: tuple[float, float, float] | None = None,
) -> list[Keypoint]:
    """Build a 17-keypoint list with only the torso anchors populated."""
    kps = [_LOW] * 17
    if ls is not None:
        kps[KP_LEFT_SHOULDER] = Keypoint(*ls)
    if rs is not None:
        kps[KP_RIGHT_SHOULDER] = Keypoint(*rs)
    if lh is not None:
        kps[KP_LEFT_HIP] = Keypoint(*lh)
    if rh is not None:
        kps[KP_RIGHT_HIP] = Keypoint(*rh)
    return kps


# A symmetric standing person: shoulders at y=100, hips at y=300, centred at x=200.
_STANDING = _pose(
    ls=(170.0, 100.0, 0.9),
    rs=(230.0, 100.0, 0.9),
    lh=(180.0, 300.0, 0.9),
    rh=(220.0, 300.0, 0.9),
)


class TestMidpoints:
    def test_shoulder_midpoint(self) -> None:
        assert shoulder_midpoint(_STANDING) == (200.0, 100.0)

    def test_hip_midpoint(self) -> None:
        assert hip_midpoint(_STANDING) == (200.0, 300.0)

    def test_low_confidence_points_ignored(self) -> None:
        # Only one usable shoulder → midpoint is that single point.
        kps = _pose(ls=(170.0, 100.0, 0.9), rs=(230.0, 100.0, 0.1))
        assert shoulder_midpoint(kps) == (170.0, 100.0)

    def test_no_usable_points_returns_none(self) -> None:
        assert shoulder_midpoint(_pose()) is None
        assert hip_midpoint(_pose()) is None


class TestTorsoAimPoint:
    def test_upper_body_is_shoulder_midpoint(self) -> None:
        assert torso_aim_point(_STANDING, bias="upper_body") == (200.0, 100.0)

    def test_full_body_is_shoulder_hip_midpoint(self) -> None:
        # Torso centre: x=200, y=(100+300)/2 = 200.
        assert torso_aim_point(_STANDING, bias="full_body") == (200.0, 200.0)

    def test_face_lifts_above_shoulders(self) -> None:
        ax, ay = torso_aim_point(_STANDING, bias="face")
        assert ax == 200.0
        # span = 200, lift = 0.45 → 100 - 90 = 10.
        assert ay == 100.0 - 200.0 * 0.45

    def test_head_shoulders_lifts_less_than_face(self) -> None:
        _, ay_face = torso_aim_point(_STANDING, bias="face")
        _, ay_hs = torso_aim_point(_STANDING, bias="head_shoulders")
        assert ay_hs > ay_face  # head_shoulders sits lower (more shoulder)

    def test_arm_motion_does_not_move_aim(self) -> None:
        """The whole point: moving arms must not change the torso aim point."""
        before = torso_aim_point(_STANDING, bias="upper_body")
        # Raise both wrists/elbows wildly — torso anchors are unchanged.
        moved = list(_STANDING)
        moved[7] = Keypoint(120.0, 40.0, 0.9)  # left_elbow up high
        moved[9] = Keypoint(110.0, 10.0, 0.9)  # left_wrist way up
        moved[10] = Keypoint(290.0, 10.0, 0.9)  # right_wrist way up
        after = torso_aim_point(moved, bias="upper_body")
        assert before == after

    def test_falls_back_to_shoulders_when_hips_missing(self) -> None:
        kps = _pose(ls=(170.0, 100.0, 0.9), rs=(230.0, 100.0, 0.9))
        assert torso_aim_point(kps, bias="full_body") == (200.0, 100.0)

    def test_falls_back_to_hips_when_shoulders_missing(self) -> None:
        kps = _pose(lh=(180.0, 300.0, 0.9), rh=(220.0, 300.0, 0.9))
        assert torso_aim_point(kps, bias="upper_body") == (200.0, 300.0)

    def test_none_when_no_torso(self) -> None:
        assert torso_aim_point(_pose(), bias="upper_body") is None


class TestSubjectHeight:
    def test_span_scaled(self) -> None:
        # shoulder→hip span = 200 px, factor 3.3 → 660.
        assert subject_height_from_pose(_STANDING) == 200.0 * 3.3

    def test_none_without_both_anchors(self) -> None:
        only_shoulders = _pose(ls=(170.0, 100.0, 0.9), rs=(230.0, 100.0, 0.9))
        assert subject_height_from_pose(only_shoulders) is None

    def test_invariant_to_arm_motion(self) -> None:
        before = subject_height_from_pose(_STANDING)
        moved = list(_STANDING)
        moved[9] = Keypoint(110.0, 10.0, 0.9)  # raise a wrist
        assert subject_height_from_pose(moved) == before


class TestAimSmoother:
    def test_first_sample_passes_through(self) -> None:
        s = AimSmoother(alpha=0.4)
        assert s.update((100.0, 200.0)) == (100.0, 200.0)

    def test_ema_blends_toward_new_sample(self) -> None:
        s = AimSmoother(alpha=0.5)
        s.update((0.0, 0.0))
        x, y = s.update((100.0, 200.0))
        assert math.isclose(x, 50.0)
        assert math.isclose(y, 100.0)

    def test_none_holds_last_value(self) -> None:
        s = AimSmoother(alpha=0.5)
        s.update((10.0, 20.0))
        assert s.update(None) == (10.0, 20.0)

    def test_alpha_one_is_no_smoothing(self) -> None:
        s = AimSmoother(alpha=1.0)
        s.update((0.0, 0.0))
        assert s.update((100.0, 100.0)) == (100.0, 100.0)

    def test_reset_clears_state(self) -> None:
        s = AimSmoother(alpha=0.5)
        s.update((10.0, 20.0))
        s.reset()
        assert s.value is None
        assert s.update((5.0, 5.0)) == (5.0, 5.0)

    def test_converges_over_many_samples(self) -> None:
        s = AimSmoother(alpha=0.3)
        s.update((0.0, 0.0))
        last = (0.0, 0.0)
        for _ in range(100):
            last = s.update((100.0, 100.0))
        assert math.isclose(last[0], 100.0, abs_tol=1e-3)
        assert math.isclose(last[1], 100.0, abs_tol=1e-3)


# A standing person WITH a head landmark (nose at y=40, above the shoulders).
_STANDING_HEAD = _pose(
    ls=(170.0, 100.0, 0.9),
    rs=(230.0, 100.0, 0.9),
    lh=(180.0, 300.0, 0.9),
    rh=(220.0, 300.0, 0.9),
)
_STANDING_HEAD[KP_NOSE] = Keypoint(200.0, 40.0, 0.9)


class TestHeadPoint:
    def test_nose_is_preferred(self) -> None:
        assert head_point(_STANDING_HEAD) == (200.0, 40.0)

    def test_none_without_head_landmarks(self) -> None:
        assert head_point(_STANDING) is None  # no nose/eyes/ears


class TestBodyAimPoint:
    """Landmark-precise, framing-aware anchor used by the fused aim dot."""

    def test_regions_are_distinct_and_ordered_top_to_bottom(self) -> None:
        ys = {}
        for fr in ("face", "head_shoulders", "upper_body", "full_body"):
            pt, conf = body_aim_point(_STANDING_HEAD, framing=fr)
            assert pt is not None and conf > 0.0
            ys[fr] = pt[1]
        # Higher on the body (smaller y) for tighter framings.
        assert ys["face"] < ys["head_shoulders"] < ys["upper_body"] < ys["full_body"]

    def test_face_is_the_head(self) -> None:
        pt, _ = body_aim_point(_STANDING_HEAD, framing="face")
        assert pt == (200.0, 40.0)

    def test_full_body_is_person_centre_hips(self) -> None:
        # Person centre ≈ the hips (stable, ≈ the bbox centre), NOT the torso
        # midpoint (which sits too high / above the true centre).
        pt, _ = body_aim_point(_STANDING_HEAD, framing="full_body")
        assert pt == (200.0, 300.0)

    def test_full_body_zero_conf_without_hips(self) -> None:
        # Hips gone → don't trust pose for the person centre; conf 0 so the caller
        # falls back to the stable bbox centre instead of jumping to the shoulders.
        no_hips = _pose(ls=(170.0, 100.0, 0.9), rs=(230.0, 100.0, 0.9))
        no_hips[KP_NOSE] = Keypoint(200.0, 40.0, 0.9)
        _pt, conf = body_aim_point(no_hips, framing="full_body")
        assert conf == 0.0

    def test_confidence_zero_when_region_landmarks_missing(self) -> None:
        # No head landmarks → "face" can't be confidently located → conf 0 so the
        # caller leans entirely on the bbox anchor.
        _pt, conf = body_aim_point(_STANDING, framing="face")
        assert conf == 0.0

    def test_none_when_no_usable_keypoints(self) -> None:
        pt, conf = body_aim_point([_LOW] * 17, framing="upper_body")
        assert pt is None and conf == 0.0
