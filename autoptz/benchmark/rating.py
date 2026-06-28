"""AutoPTZ Mark score rating — a human word + a transparent score-math reason.

Turns the raw ramp :class:`~autoptz.benchmark.runner.BenchmarkResult` score into a
plain-English rating word and a one-line (or expanded) explanation whose numbers
visibly add up to the displayed score.  Pure functions, no Qt, no I/O — the verdict
label, the control panel highlight, and the completion modal all consume them.

The score itself is computed by the runner (``score = sustained_cameras × (min_fps ÷
30 target) × profile weight``); this module never re-derives it — it surfaces the
runner's ``result.score`` alongside the inputs that produced it so the reason stays
honest (transparency: show configured → effective).

Bands (inclusive lower bound; 1.0 ≈ one camera held at the full 30 fps target, each
whole point ≈ one more camera at target):

==================  ===========
Score range         Rating word
==================  ===========
``score < 1.0``     Needs work
``1.0 ≤ score``     Fair
``2.0 ≤ score``     Good
``3.0 ≤ score``     Great
``score ≥ 4.0``     Excellent
==================  ===========
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoptz.benchmark.runner import BenchmarkResult

# 30 fps target the score normalises against (mirrors runner._NOMINAL_FPS) — the
# divisor shown in the reason math so the printed numbers match the score formula.
_TARGET_FPS = 30

# Highest band first so the first matching threshold wins (inclusive lower bound).
_BANDS: list[tuple[float, str]] = [
    (4.0, "Excellent"),
    (3.0, "Great"),
    (2.0, "Good"),
    (1.0, "Fair"),
    (0.0, "Needs work"),
]


def score_rating(score: float) -> str:
    """The rating word for *score* (``Needs work`` / ``Fair`` / ``Good`` / ``Great`` /
    ``Excellent``)."""
    for threshold, word in _BANDS:
        if score >= threshold:
            return word
    return _BANDS[-1][1]  # score below 0 → "Needs work" (defensive; scores are ≥ 0)


def score_reason(result: BenchmarkResult) -> str:
    """A compact one-line reason for the verdict label: the rating word + the score math.

    Example: ``Good — 4 cam × 30/30 fps × 0.8 weight = 3.20``.
    """
    return (
        f"{score_rating(result.score)} — {result.sustained_cameras} cam × "
        f"{result.min_fps_at_sustained:.0f}/{_TARGET_FPS} fps × "
        f"{result.weight:.1f} weight = {result.score:.2f}"
    )


def score_reason_full(result: BenchmarkResult) -> str:
    """The expanded two-line reason for the completion modal: the named formula then the
    numeric substitution that yields the score.

    Example::

        score = cameras × (fps ÷ 30 target) × profile weight
              = 2 × (28.0 ÷ 30) × 1.0 = 1.87
    """
    return (
        f"score = cameras × (fps ÷ {_TARGET_FPS} target) × profile weight\n"
        f"      = {result.sustained_cameras} × "
        f"({result.min_fps_at_sustained:.1f} ÷ {_TARGET_FPS}) × "
        f"{result.weight:.1f} = {result.score:.2f}"
    )
