"""Tests for reid.adaptive_threshold_hi (R6 — scene-adaptive ReID threshold).

TDD: these tests are written before the implementation so that all cases
are captured by specification, not by implementation behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autoptz.engine.pipeline.reid import _cosine, adaptive_threshold_hi

# ---------------------------------------------------------------------------
# Helpers to build unit vectors with a known pairwise cosine
# ---------------------------------------------------------------------------


def _unit(c: float) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors whose dot-product equals *c* (0 ≤ c ≤ 1).

    Strategy: a = [1, 0, ...], b = [c, sqrt(1-c²), 0, ...] in R^4.
    """
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([c, math.sqrt(max(0.0, 1.0 - c * c)), 0.0, 0.0], dtype=np.float32)
    return a, b


# ---------------------------------------------------------------------------
# Edge cases: <2 candidates → always base_hi
# ---------------------------------------------------------------------------


class TestFewCandidates:
    def test_empty_list_returns_base(self) -> None:
        assert adaptive_threshold_hi([], base_hi=0.70) == pytest.approx(0.70)

    def test_single_candidate_returns_base(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert adaptive_threshold_hi([v], base_hi=0.70) == pytest.approx(0.70)

    def test_single_candidate_does_not_go_below_base(self) -> None:
        v = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        base = 0.80
        assert adaptive_threshold_hi([v], base_hi=base) == pytest.approx(base)


# ---------------------------------------------------------------------------
# Two orthogonal vectors (cosine ≈ 0): max(base, 0 + margin) = base when base > margin
# ---------------------------------------------------------------------------


class TestOrthogonalVectors:
    def test_orthogonal_returns_base(self) -> None:
        a, b = _unit(0.0)
        # max_cosine ≈ 0 → max(0.70, 0 + 0.05) = 0.70
        result = adaptive_threshold_hi([a, b], base_hi=0.70, margin=0.05)
        assert result == pytest.approx(0.70)

    def test_orthogonal_verify_cosine(self) -> None:
        a, b = _unit(0.0)
        assert _cosine(a, b) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Two near-identical vectors (cosine ≈ 1): result should be clamped to cap
# ---------------------------------------------------------------------------


class TestNearIdenticalVectors:
    def test_near_identical_returns_cap(self) -> None:
        a, b = _unit(0.999)
        result = adaptive_threshold_hi([a, b], base_hi=0.70, margin=0.05, cap=0.95)
        assert result == pytest.approx(0.95)

    def test_exact_identical_returns_cap(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = adaptive_threshold_hi([v, v.copy()], base_hi=0.70, margin=0.05, cap=0.95)
        assert result == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Moderately similar pair: cosine ≈ 0.8, base 0.70, margin 0.05 → 0.85
# ---------------------------------------------------------------------------


class TestModeratelySimilar:
    def test_moderate_similarity(self) -> None:
        a, b = _unit(0.8)
        # verify the cosine first
        assert _cosine(a, b) == pytest.approx(0.8, abs=1e-5)
        # expected: max(0.70, 0.8 + 0.05) = 0.85, clamped to [0.70, 0.95]
        result = adaptive_threshold_hi([a, b], base_hi=0.70, margin=0.05, cap=0.95)
        assert result == pytest.approx(0.85, abs=1e-5)


# ---------------------------------------------------------------------------
# Invariants: result always in [base_hi, cap]
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.parametrize("cosine_val", [0.0, 0.3, 0.6, 0.8, 0.95, 1.0])
    def test_result_never_below_base(self, cosine_val: float) -> None:
        a, b = _unit(min(cosine_val, 0.9999))
        base = 0.70
        result = adaptive_threshold_hi([a, b], base_hi=base, margin=0.05, cap=0.95)
        assert result >= base - 1e-9

    @pytest.mark.parametrize("cosine_val", [0.0, 0.3, 0.6, 0.8, 0.95, 1.0])
    def test_result_never_above_cap(self, cosine_val: float) -> None:
        a, b = _unit(min(cosine_val, 0.9999))
        cap = 0.95
        result = adaptive_threshold_hi([a, b], base_hi=0.70, margin=0.05, cap=cap)
        assert result <= cap + 1e-9

    def test_base_higher_than_margined_cosine_returns_base(self) -> None:
        # cosine = 0.5, margin = 0.05 → candidate = 0.55 < base 0.70 → return 0.70
        a, b = _unit(0.5)
        result = adaptive_threshold_hi([a, b], base_hi=0.70, margin=0.05)
        assert result == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Three candidates: max pairwise among all pairs is used
# ---------------------------------------------------------------------------


class TestThreeCandidates:
    def test_three_orthogonal_returns_base(self) -> None:
        """Three fully orthogonal candidates → max pairwise = 0 → returns base."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        c = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        result = adaptive_threshold_hi([a, b, c], base_hi=0.70, margin=0.05, cap=0.95)
        assert result == pytest.approx(0.70)

    def test_most_similar_pair_dominates(self) -> None:
        """Most similar pair should drive the threshold even when others are distinct.

        Vectors (all unit):
          p = [1, 0, 0]
          q = [cos80, sin80, 0]  cos(p,q)≈0.8
          r = [0, 0, 1]          orthogonal to both p and q
        max_pairwise = cos(p,q) ≈ 0.8 → expected = max(0.70, 0.8+0.05)=0.85
        """
        p = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cos80 = 0.8
        sin80 = float(math.sqrt(1.0 - cos80**2))
        q = np.array([cos80, sin80, 0.0], dtype=np.float32)
        r = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # orthogonal to p and q
        assert _cosine(p, q) == pytest.approx(cos80, abs=1e-5)
        assert _cosine(p, r) == pytest.approx(0.0, abs=1e-5)
        assert _cosine(q, r) == pytest.approx(0.0, abs=1e-5)
        result = adaptive_threshold_hi([p, q, r], base_hi=0.70, margin=0.05, cap=0.95)
        assert result == pytest.approx(0.85, abs=1e-4)


# ---------------------------------------------------------------------------
# Degenerate (zero) vectors are skipped
# ---------------------------------------------------------------------------


class TestDegenerateVectors:
    def test_zero_vector_skipped(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        zero = np.array([0.0, 0.0], dtype=np.float32)
        # Only one valid candidate after skipping zero → should return base
        result = adaptive_threshold_hi([a, zero], base_hi=0.70)
        assert result == pytest.approx(0.70)

    def test_all_zero_vectors_returns_base(self) -> None:
        z1 = np.zeros(4, dtype=np.float32)
        z2 = np.zeros(4, dtype=np.float32)
        result = adaptive_threshold_hi([z1, z2], base_hi=0.70)
        assert result == pytest.approx(0.70)
