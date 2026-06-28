"""score_rating / score_reason / score_reason_full (pure, no Qt).

The rating module turns an AutoPTZ Mark score into a human word and a transparent
one-line (and expanded) math reason whose numbers visibly equal the displayed
score.  These are pure functions over a ``BenchmarkResult`` — no Qt, no I/O — so
they test plainly without a QApplication.
"""

from __future__ import annotations

import pytest

from autoptz.benchmark.rating import score_rating, score_reason, score_reason_full
from autoptz.benchmark.runner import BenchmarkResult


def _result(**kw) -> BenchmarkResult:
    base = {
        "profile": "full",
        "weight": 1.0,
        "floor_fps": 24.0,
        "max_cameras": 3,
        "sustained_cameras": 2,
        "min_fps_at_sustained": 28.0,
        "score": 1.87,
        "steps": [],
    }
    base.update(kw)
    return BenchmarkResult(**base)


@pytest.mark.parametrize(
    ("score", "word"),
    [
        (0.0, "Needs work"),
        (0.99, "Needs work"),
        (1.0, "Fair"),
        (1.99, "Fair"),
        (2.0, "Good"),
        (2.5, "Good"),
        (3.0, "Great"),
        (3.99, "Great"),
        (4.0, "Excellent"),
        (5.5, "Excellent"),
    ],
)
def test_band_boundaries(score: float, word: str) -> None:
    assert score_rating(score) == word


def test_reason_matches_formula() -> None:
    """2 cams @ 28 fps, full weight → score 1.87; the compact reason carries the
    rating word and every number that makes up the score."""
    r = _result(sustained_cameras=2, min_fps_at_sustained=28.0, weight=1.0, score=1.87)
    reason = score_reason(r)
    assert "Fair" in reason  # 1.87 is in the Fair band
    assert "2 cam" in reason
    assert "28" in reason
    assert "30" in reason
    assert "1.0" in reason
    assert "1.87" in reason


def test_reason_streams_weight() -> None:
    """4 cams @ 30 fps, streams weight 0.8 → score 3.20 → Great; the reason shows
    the streams weight and the recomputed score."""
    r = _result(
        profile="streams",
        weight=0.8,
        sustained_cameras=4,
        min_fps_at_sustained=30.0,
        score=3.20,
    )
    reason = score_reason(r)
    assert score_rating(r.score) == "Great"
    assert "Great" in reason
    assert "4 cam" in reason
    assert "30/30 fps" in reason
    assert "0.8" in reason
    assert "3.20" in reason


def test_reason_full_shows_formula_and_substitution() -> None:
    """The expanded reason states the named formula AND the numeric substitution that
    yields the score (transparency: configured → effective)."""
    r = _result(sustained_cameras=2, min_fps_at_sustained=28.0, weight=1.0, score=1.87)
    full = score_reason_full(r)
    assert "cameras" in full
    assert "fps" in full
    assert "profile weight" in full
    assert "2 × (28.0 ÷ 30) × 1.0 = 1.87" in full
