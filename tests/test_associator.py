"""Tests for TargetAssociator — synthetic scenario harness.

No real frames are needed: all cues are constructed explicitly so every
branch of the association logic can be exercised in isolation.

Coverage targets
----------------
association_confidence
  TC-01  All four cues present → exact weighted average
  TC-02  One cue unavailable → excluded from denominator, not zero-penalised
  TC-03  Only motion available → returns exactly the motion value
  TC-04  No cues available → 0.0
  TC-05  Cue value > 1.0 → clamped before weighting
  TC-06  Cue value < 0.0 → clamped to 0.0 before weighting

decide — happy-path cases
  TC-07  Single strong candidate IS the current target → keep
  TC-08  Single strong candidate, no current target → switch (acquire)
  TC-09  Target lost, one clear candidate (conf ≥ min, big margin) → switch (recover)

decide — hysteresis / anti-ID-switch cases
  TC-10  Two near-identical candidates (look-alikes), margin < ambiguous_margin
         → ambiguous (headline anti-ID-switch guard, NO switch)
  TC-11  Distractor slightly better than current but gap < switch_margin → keep
  TC-12  Distractor clearly better (gap ≥ switch_margin, margin ≥ ambiguous_margin)
         → switch
  TC-13  Everyone below min_confidence → ambiguous
  TC-14  Ambiguous on recovery (no current, small margin) → ambiguous

decide — edge / robustness cases
  TC-15  Identity-only candidate (motion/appearance/pose unavailable) → decides sensibly
  TC-16  Motion-only candidate → decides sensibly
  TC-17  Empty candidate list → ambiguous, track_id = current
  TC-18  Current target disappears while a challenger passes all thresholds → switch
  TC-19  Current target disappears while scene is ambiguous → ambiguous, track_id=None
  TC-20  Multi-candidate crowd: best IS current, runners-up are close → keep
  TC-21  challenger gap exactly at switch_margin boundary → switch (≥ is inclusive)
  TC-22  challenger gap just below switch_margin boundary → keep
  TC-23  margin exactly at ambiguous_margin boundary → allowed (≥ is inclusive)
  TC-24  margin just below ambiguous_margin boundary → ambiguous
"""

from __future__ import annotations

import pytest

from autoptz.engine.pipeline.associator import (
    CandidateCues,
    Cue,
    TargetAssociator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ABSENT = Cue(value=0.0, available=False)


def _assoc(
    track_id: int = 1,
    *,
    motion: float = 0.0,
    appearance: float = 0.0,
    pose: float = 0.0,
    identity: float = 0.0,
    motion_avail: bool = True,
    appearance_avail: bool = True,
    pose_avail: bool = True,
    identity_avail: bool = True,
) -> CandidateCues:
    return CandidateCues(
        track_id=track_id,
        motion=Cue(motion, motion_avail),
        appearance=Cue(appearance, appearance_avail),
        pose=Cue(pose, pose_avail),
        identity=Cue(identity, identity_avail),
    )


# Default weights from TargetAssociator:
#   w_motion=0.35, w_appearance=0.30, w_identity=0.20, w_pose=0.15
_W_M = 0.35
_W_A = 0.30
_W_I = 0.20
_W_P = 0.15
_W_TOTAL = _W_M + _W_A + _W_I + _W_P  # 1.00


@pytest.fixture()
def ta() -> TargetAssociator:
    """Default-parameter associator."""
    return TargetAssociator()


# ---------------------------------------------------------------------------
# association_confidence
# ---------------------------------------------------------------------------


class TestAssociationConfidence:
    def test_tc01_all_cues_weighted_average(self, ta: TargetAssociator) -> None:
        """TC-01: all cues present → exact weighted average."""
        cands = _assoc(motion=0.8, appearance=0.6, pose=0.5, identity=0.9)
        expected = (_W_M * 0.8 + _W_A * 0.6 + _W_I * 0.9 + _W_P * 0.5) / _W_TOTAL
        assert abs(ta.association_confidence(cands) - expected) < 1e-9

    def test_tc02_missing_cue_excluded_not_penalised(self, ta: TargetAssociator) -> None:
        """TC-02: unavailable cue excluded from denominator — higher than if zeroed."""
        # motion=0.8, appearance unavailable — should NOT pull score toward 0.
        cands_missing = _assoc(
            motion=0.8,
            appearance=0.0,
            appearance_avail=False,
            pose=0.0,
            pose_avail=False,
            identity=0.0,
            identity_avail=False,
        )
        conf_missing = ta.association_confidence(cands_missing)

        # If the missing cues were zeroed-in the score would be pulled down.
        cands_zero = _assoc(motion=0.8, appearance=0.0, pose=0.0, identity=0.0)
        conf_zero = ta.association_confidence(cands_zero)

        assert conf_missing > conf_zero
        # With only motion available the answer should equal the motion value.
        assert abs(conf_missing - 0.8) < 1e-9

    def test_tc03_only_motion_returns_motion_value(self, ta: TargetAssociator) -> None:
        """TC-03: only motion available → conf == motion.value."""
        cands = _assoc(motion=0.72, appearance_avail=False, pose_avail=False, identity_avail=False)
        assert abs(ta.association_confidence(cands) - 0.72) < 1e-9

    def test_tc04_no_cues_returns_zero(self, ta: TargetAssociator) -> None:
        """TC-04: no cues available → 0.0."""
        cands = _assoc(
            motion_avail=False, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        assert ta.association_confidence(cands) == 0.0

    def test_tc05_cue_value_above_one_clamped(self, ta: TargetAssociator) -> None:
        """TC-05: cue.value > 1.0 is clamped to 1.0 before weighting."""
        cands_clamped = _assoc(
            motion=1.5, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        assert ta.association_confidence(cands_clamped) == pytest.approx(1.0)

    def test_tc06_cue_value_below_zero_clamped(self, ta: TargetAssociator) -> None:
        """TC-06: cue.value < 0.0 is clamped to 0.0 before weighting."""
        cands = _assoc(motion=-0.5, appearance_avail=False, pose_avail=False, identity_avail=False)
        assert ta.association_confidence(cands) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# decide — happy-path
# ---------------------------------------------------------------------------


class TestDecideHappyPath:
    def test_tc07_single_strong_candidate_is_current_keep(self, ta: TargetAssociator) -> None:
        """TC-07: strong candidate IS current target → keep."""
        cands = [_assoc(1, motion=0.9, appearance=0.85, pose=0.8, identity=0.95)]
        d = ta.decide(cands, current_target_id=1)
        assert d.action == "keep"
        assert d.track_id == 1

    def test_tc08_no_current_strong_candidate_acquires(self, ta: TargetAssociator) -> None:
        """TC-08: no current target, one clear candidate → switch (first acquisition)."""
        cands = [_assoc(1, motion=0.9, appearance=0.8, pose=0.7, identity=0.9)]
        d = ta.decide(cands, current_target_id=None)
        assert d.action == "switch"
        assert d.track_id == 1
        assert d.confidence >= ta.min_confidence

    def test_tc09_target_lost_clear_candidate_recovers(self, ta: TargetAssociator) -> None:
        """TC-09: target lost (id=2 not in list), one clear candidate → switch (recover)."""
        cands = [_assoc(3, motion=0.85, appearance=0.80, pose=0.75, identity=0.90)]
        d = ta.decide(cands, current_target_id=2)
        # track_id=2 is absent from candidates, so treated as lost.
        assert d.action == "switch"
        assert d.track_id == 3


# ---------------------------------------------------------------------------
# decide — hysteresis / anti-ID-switch
# ---------------------------------------------------------------------------


class TestDecideHysteresis:
    def test_tc10_look_alikes_ambiguous_no_id_switch(self, ta: TargetAssociator) -> None:
        """TC-10: two near-identical candidates within ambiguous_margin → ambiguous.

        This is the headline anti-ID-switch case: when two people look alike the
        associator must NOT commit to a switch — it holds.
        """
        # Current target=1 with conf ~0.70; candidate 2 with conf ~0.71 (tiny margin).
        cand1 = _assoc(1, motion=0.75, appearance=0.70, pose=0.60, identity=0.70)
        cand2 = _assoc(2, motion=0.76, appearance=0.71, pose=0.62, identity=0.70)
        conf1 = ta.association_confidence(cand1)
        conf2 = ta.association_confidence(cand2)
        assert abs(conf2 - conf1) < ta.ambiguous_margin, "Precondition: margin must be tiny"

        d = ta.decide([cand1, cand2], current_target_id=1)
        assert d.action == "ambiguous"
        # Must NOT have switched to track 2.
        assert d.track_id == 1  # hold existing lock

    def test_tc11_distractor_slightly_better_keep(self, ta: TargetAssociator) -> None:
        """TC-11: distractor leads current by < switch_margin → keep (no flicker)."""
        # current=1 conf≈0.72, challenger=2 conf≈0.80 — gap≈0.08, switch_margin=0.15
        cand_current = _assoc(1, motion=0.70, appearance=0.70, pose=0.75, identity=0.75)
        cand_challenger = _assoc(2, motion=0.80, appearance=0.80, pose=0.82, identity=0.78)
        conf_current = ta.association_confidence(cand_current)
        conf_challenger = ta.association_confidence(cand_challenger)
        gap = conf_challenger - conf_current
        assert 0 < gap < ta.switch_margin, (
            f"Precondition: gap {gap:.3f} must be < switch_margin {ta.switch_margin}"
        )

        d = ta.decide([cand_current, cand_challenger], current_target_id=1)
        assert d.action == "keep"
        assert d.track_id == 1

    def test_tc12_distractor_clearly_better_switch(self, ta: TargetAssociator) -> None:
        """TC-12: distractor dominates (gap ≥ switch_margin, unambiguous) → switch."""
        # current=1 low confidence (e.g. occluded), challenger=2 carries identity.
        cand_current = _assoc(1, motion=0.40, appearance=0.35, pose=0.30, identity=0.20)
        cand_challenger = _assoc(2, motion=0.80, appearance=0.85, pose=0.75, identity=0.95)
        conf_current = ta.association_confidence(cand_current)
        conf_challenger = ta.association_confidence(cand_challenger)
        gap = conf_challenger - conf_current
        assert gap >= ta.switch_margin, f"Precondition: gap {gap:.3f} must be ≥ switch_margin"

        d = ta.decide([cand_current, cand_challenger], current_target_id=1)
        assert d.action == "switch"
        assert d.track_id == 2

    def test_tc13_everyone_below_min_confidence_ambiguous(self, ta: TargetAssociator) -> None:
        """TC-13: all candidates below min_confidence → ambiguous."""
        cand1 = _assoc(1, motion=0.2, appearance=0.15, pose=0.1, identity=0.1)
        cand2 = _assoc(2, motion=0.25, appearance=0.20, pose=0.15, identity=0.1)
        assert ta.association_confidence(cand1) < ta.min_confidence
        assert ta.association_confidence(cand2) < ta.min_confidence

        d = ta.decide([cand1, cand2], current_target_id=1)
        assert d.action == "ambiguous"

    def test_tc14_no_current_small_margin_ambiguous(self, ta: TargetAssociator) -> None:
        """TC-14: no current target, best confident but margin tiny → ambiguous."""
        cand1 = _assoc(1, motion=0.80, appearance=0.75, pose=0.70, identity=0.75)
        # cand2 nearly identical
        cand2 = _assoc(2, motion=0.81, appearance=0.75, pose=0.71, identity=0.75)
        conf1 = ta.association_confidence(cand1)
        conf2 = ta.association_confidence(cand2)
        margin = abs(conf2 - conf1)
        assert margin < ta.ambiguous_margin, f"Precondition: margin {margin:.4f} must be tiny"

        d = ta.decide([cand1, cand2], current_target_id=None)
        assert d.action == "ambiguous"


# ---------------------------------------------------------------------------
# decide — edge / robustness
# ---------------------------------------------------------------------------


class TestDecideEdgeCases:
    def test_tc15_identity_only_candidate_sensible(self, ta: TargetAssociator) -> None:
        """TC-15: only identity cue available, but high → can still acquire."""
        cand = _assoc(5, identity=1.0, motion_avail=False, appearance_avail=False, pose_avail=False)
        conf = ta.association_confidence(cand)
        assert conf == pytest.approx(1.0)
        # Single candidate, no current → should switch if conf ≥ min.
        d = ta.decide([cand], current_target_id=None)
        assert d.action == "switch"
        assert d.track_id == 5

    def test_tc16_motion_only_candidate_sensible(self, ta: TargetAssociator) -> None:
        """TC-16: only motion cue available → decides sensibly."""
        cand = _assoc(
            6, motion=0.85, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        conf = ta.association_confidence(cand)
        assert conf == pytest.approx(0.85)
        d = ta.decide([cand], current_target_id=None)
        assert d.action == "switch"
        assert d.track_id == 6

    def test_tc17_empty_candidate_list_ambiguous(self, ta: TargetAssociator) -> None:
        """TC-17: empty list → ambiguous, track_id held at current."""
        d = ta.decide([], current_target_id=7)
        assert d.action == "ambiguous"
        assert d.track_id == 7
        assert d.confidence == 0.0

    def test_tc18_current_disappeared_challenger_switches(self, ta: TargetAssociator) -> None:
        """TC-18: current disappears, unambiguous challenger → switch."""
        cand = _assoc(10, motion=0.90, appearance=0.85, pose=0.80, identity=0.92)
        # current_target_id=9 is NOT in the list (disappeared).
        d = ta.decide([cand], current_target_id=9)
        assert d.action == "switch"
        assert d.track_id == 10

    def test_tc19_current_disappeared_ambiguous_scene(self, ta: TargetAssociator) -> None:
        """TC-19: current disappears while scene is ambiguous → ambiguous, track_id=None."""
        cand1 = _assoc(11, motion=0.76, appearance=0.70, pose=0.65, identity=0.72)
        cand2 = _assoc(12, motion=0.77, appearance=0.71, pose=0.65, identity=0.72)
        margin = abs(ta.association_confidence(cand1) - ta.association_confidence(cand2))
        assert margin < ta.ambiguous_margin, "Precondition: must be ambiguous"

        d = ta.decide([cand1, cand2], current_target_id=None)
        assert d.action == "ambiguous"
        assert d.track_id is None

    def test_tc20_crowd_best_is_current_keep(self, ta: TargetAssociator) -> None:
        """TC-20: multi-candidate crowd, current leads → keep."""
        cand_current = _assoc(1, motion=0.90, appearance=0.85, pose=0.80, identity=0.92)
        cand_b = _assoc(2, motion=0.60, appearance=0.55, pose=0.50, identity=0.40)
        cand_c = _assoc(3, motion=0.50, appearance=0.45, pose=0.42, identity=0.35)
        d = ta.decide([cand_current, cand_b, cand_c], current_target_id=1)
        assert d.action == "keep"
        assert d.track_id == 1

    def test_tc21_challenger_gap_exactly_switch_margin_switches(self) -> None:
        """TC-21: gap exactly == switch_margin → switch (boundary is inclusive).

        To avoid float-subtraction rounding, compute the gap first and use it
        directly as the switch_margin threshold — this guarantees gap == threshold
        exactly (same IEEE-754 value) so the >= check in the associator is exercised
        at the true boundary.
        """
        # Compute the gap that results from the chosen motion values.
        _ta_probe = TargetAssociator()
        cand_current = _assoc(
            1, motion=0.55, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        cand_challenger = _assoc(
            2, motion=0.75, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        exact_gap = _ta_probe.association_confidence(
            cand_challenger
        ) - _ta_probe.association_confidence(cand_current)
        # Set switch_margin to that exact value so gap == switch_margin in float64.
        ta = TargetAssociator(switch_margin=exact_gap, ambiguous_margin=0.05, min_confidence=0.30)

        d = ta.decide([cand_current, cand_challenger], current_target_id=1)
        assert d.action == "switch", (
            f"gap={exact_gap} exactly equals switch_margin={ta.switch_margin}; "
            "'>=' boundary must trigger a switch"
        )
        assert d.track_id == 2

    def test_tc22_challenger_gap_just_below_switch_margin_keeps(self) -> None:
        """TC-22: gap just below switch_margin → keep (no switch)."""
        ta = TargetAssociator(switch_margin=0.20, ambiguous_margin=0.05, min_confidence=0.30)
        cand_current = _assoc(
            1, motion=0.51, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        cand_challenger = _assoc(
            2, motion=0.70, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        conf_current = ta.association_confidence(cand_current)  # 0.51
        conf_challenger = ta.association_confidence(cand_challenger)  # 0.70
        gap = conf_challenger - conf_current  # 0.19 < switch_margin=0.20
        assert gap < ta.switch_margin, f"gap={gap:.4f}"

        d = ta.decide([cand_current, cand_challenger], current_target_id=1)
        assert d.action == "keep"
        assert d.track_id == 1

    def test_tc23_margin_exactly_ambiguous_margin_allows_action(self) -> None:
        """TC-23: margin exactly == ambiguous_margin → unambiguous (≥ is inclusive).

        To avoid float-subtraction rounding, compute the margin first and use it
        directly as ambiguous_margin — same IEEE-754 value so >= is at the boundary.
        """
        _ta_probe = TargetAssociator()
        cand_best = _assoc(
            1, motion=0.75, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        cand_runup = _assoc(
            2, motion=0.55, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        exact_margin = _ta_probe.association_confidence(
            cand_best
        ) - _ta_probe.association_confidence(cand_runup)
        # Set ambiguous_margin to that exact value.
        ta = TargetAssociator(
            switch_margin=0.05, ambiguous_margin=exact_margin, min_confidence=0.30
        )

        d = ta.decide([cand_best, cand_runup], current_target_id=None)
        assert d.action == "switch", (
            f"margin={exact_margin} exactly equals ambiguous_margin={ta.ambiguous_margin}; "
            "'>=' boundary must allow a switch"
        )
        assert d.track_id == 1

    def test_tc24_margin_just_below_ambiguous_margin_stays_ambiguous(self) -> None:
        """TC-24: margin just below ambiguous_margin → ambiguous."""
        ta = TargetAssociator(switch_margin=0.10, ambiguous_margin=0.10, min_confidence=0.30)
        # best=0.70, runner-up=0.61 → margin=0.09 < 0.10
        cand_best = _assoc(
            1, motion=0.70, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        cand_runup = _assoc(
            2, motion=0.61, appearance_avail=False, pose_avail=False, identity_avail=False
        )
        margin = ta.association_confidence(cand_best) - ta.association_confidence(cand_runup)
        assert margin < ta.ambiguous_margin, f"margin={margin:.4f}"

        d = ta.decide([cand_best, cand_runup], current_target_id=None)
        assert d.action == "ambiguous"
