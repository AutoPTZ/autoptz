"""Target-association decision module — pure, fused, deterministic.

Why
---
The "is this still my person" decision was previously scattered across
``_apply_target_lock``, ``_mark_target_ambiguous``, ``_maybe_reid_recover``,
and pose-ownership checks, with ~a dozen magic thresholds that interacted
unpredictably and broke per detector tier.

This module replaces them with ONE principled decision:

1. Each candidate track is scored by a *weighted average over AVAILABLE cues*
   (motion, appearance, pose, identity).  Missing cues are excluded — never
   penalised — so the formula degrades gracefully when e.g. ReID is offline.

2. Explicit hysteresis prevents ID-switches and flicker: a different candidate
   must lead by at least ``switch_margin`` before a switch is allowed, and the
   lead must be unambiguous (gap to runner-up ≥ ``ambiguous_margin``).

Pure — no I/O, no NumPy, no external deps.  R5-B will wire this into the
worker hot-path; this PR only adds the module + its tests.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cue:
    """One association cue in [0, 1] and whether it was measurable this tick.

    ``available=False`` means the signal could not be produced this frame
    (e.g. ReID offline, pose not detected).  Such cues are *excluded* from
    the weighted average rather than treated as zero — no penalty for missing
    sensors.
    """

    value: float
    available: bool = True


@dataclass(frozen=True)
class CandidateCues:
    """All cues for one candidate track."""

    track_id: int
    motion: Cue  # IOU / proximity to predicted bbox
    appearance: Cue  # ReID cosine similarity to appearance template
    pose: Cue  # fraction of target keypoints inside candidate bbox
    identity: Cue  # face-identity match score


@dataclass(frozen=True)
class Decision:
    """Result of one ``TargetAssociator.decide`` call."""

    action: str  # "keep" | "switch" | "ambiguous"
    track_id: int | None  # chosen target (None when ambiguous with no current)
    confidence: float  # fused confidence of the chosen candidate
    margin: float  # best - runner-up confidence


# ---------------------------------------------------------------------------
# Associator
# ---------------------------------------------------------------------------


class TargetAssociator:
    """Fused, hysteretic target-association decision.

    Parameters
    ----------
    w_motion, w_appearance, w_identity, w_pose
        Weights for the four cue types (need not sum to 1; they are
        normalised over *available* cues at scoring time).
    min_confidence
        Below this the best candidate is not confident enough to act on;
        the decision is "ambiguous" regardless of everything else.
    switch_margin
        The challenger must exceed the current target's confidence by at
        least this much before a switch is allowed (hysteresis, prevents
        flicker).
    ambiguous_margin
        The best candidate must lead the runner-up by at least this much
        before the scene is considered unambiguous (anti-look-alike guard).
    """

    def __init__(
        self,
        *,
        w_motion: float = 0.35,
        w_appearance: float = 0.30,
        w_identity: float = 0.20,
        w_pose: float = 0.15,
        min_confidence: float = 0.40,
        switch_margin: float = 0.15,
        ambiguous_margin: float = 0.08,
    ) -> None:
        self._weights = {
            "motion": w_motion,
            "appearance": w_appearance,
            "identity": w_identity,
            "pose": w_pose,
        }
        self.min_confidence = min_confidence
        self.switch_margin = switch_margin
        self.ambiguous_margin = ambiguous_margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def association_confidence(self, cues: CandidateCues) -> float:
        """Weighted average over available cues only.

        Missing cues (``Cue.available=False``) are excluded — the denominator
        shrinks rather than pulling the score toward zero.  Returns 0.0 when
        no cue is available.  Output is clamped to [0, 1].
        """
        pairs = [
            (self._weights["motion"], cues.motion),
            (self._weights["appearance"], cues.appearance),
            (self._weights["identity"], cues.identity),
            (self._weights["pose"], cues.pose),
        ]

        numerator = 0.0
        denominator = 0.0
        for w, cue in pairs:
            if cue.available:
                v = max(0.0, min(1.0, cue.value))  # clamp input
                numerator += w * v
                denominator += w

        if denominator == 0.0:
            return 0.0

        return max(0.0, min(1.0, numerator / denominator))  # clamp output

    def decide(
        self,
        candidates: list[CandidateCues],
        current_target_id: int | None,
    ) -> Decision:
        """Choose an action given the current set of candidate tracks.

        Hysteresis rules (in priority order)
        -------------------------------------
        1. If best confidence < ``min_confidence`` → *ambiguous* (hold).
        2. If current target is present:
           a. Best IS the current target → *keep*.
           b. Best is a challenger:
              - If gap to current ≥ ``switch_margin`` AND margin ≥
                ``ambiguous_margin`` → *switch* (challenger clearly dominates).
              - If margin < ``ambiguous_margin`` → *ambiguous* (too close,
                look-alike risk — the headline anti-ID-switch guard).
              - Otherwise (gap < switch_margin) → *keep* (not decisive enough).
        3. No current target (target lost / first acquisition):
           - If best ≥ ``min_confidence`` AND margin ≥ ``ambiguous_margin``
             → *switch* (acquire / recover).
           - Else → *ambiguous*.

        Returns a ``Decision`` with the appropriate action and track_id.
        When action is "ambiguous" and there is a current target, track_id
        is set to ``current_target_id`` (hold the existing lock) — prefer
        "keep" over thrashing on uncertain evidence.
        """
        if not candidates:
            return Decision(
                action="ambiguous",
                track_id=current_target_id,
                confidence=0.0,
                margin=0.0,
            )

        # Score every candidate.
        scored = sorted(
            [(self.association_confidence(c), c) for c in candidates],
            key=lambda t: t[0],
            reverse=True,
        )

        best_conf, best_cand = scored[0]
        runner_up_conf = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_conf - runner_up_conf

        # Rule 1 — nobody confident enough.
        if best_conf < self.min_confidence:
            return Decision(
                action="ambiguous",
                track_id=current_target_id,
                confidence=best_conf,
                margin=margin,
            )

        # Rule 2 — current target is present in the candidate list.
        if current_target_id is not None:
            current_conf = next(
                (
                    self.association_confidence(c)
                    for c in candidates
                    if c.track_id == current_target_id
                ),
                None,
            )
            if current_conf is not None:
                # 2a — current target is already the best.
                if best_cand.track_id == current_target_id:
                    return Decision(
                        action="keep",
                        track_id=current_target_id,
                        confidence=best_conf,
                        margin=margin,
                    )

                # 2b — a challenger leads.
                gap = best_conf - current_conf

                if margin < self.ambiguous_margin:
                    # Scene is ambiguous (look-alike risk) — hold.
                    return Decision(
                        action="ambiguous",
                        track_id=current_target_id,
                        confidence=best_conf,
                        margin=margin,
                    )

                if gap >= self.switch_margin:
                    # Challenger clearly dominates — switch.
                    return Decision(
                        action="switch",
                        track_id=best_cand.track_id,
                        confidence=best_conf,
                        margin=margin,
                    )

                # Gap not decisive — keep current.
                return Decision(
                    action="keep",
                    track_id=current_target_id,
                    confidence=best_conf,
                    margin=margin,
                )

        # Rule 3 — target lost / no current target.
        if margin >= self.ambiguous_margin:
            return Decision(
                action="switch",
                track_id=best_cand.track_id,
                confidence=best_conf,
                margin=margin,
            )

        return Decision(
            action="ambiguous",
            track_id=current_target_id,
            confidence=best_conf,
            margin=margin,
        )
