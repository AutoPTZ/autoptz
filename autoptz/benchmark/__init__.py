"""AutoPTZ Mark — a headless throughput benchmark for the engine pipeline."""

from __future__ import annotations

from autoptz.benchmark.profiles import PROFILES, BenchmarkProfile, get_profile
from autoptz.benchmark.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    StepResult,
    run_benchmark,
)

__all__ = [
    "PROFILES",
    "BenchmarkProfile",
    "BenchmarkResult",
    "BenchmarkRunner",
    "StepResult",
    "get_profile",
    "run_benchmark",
]
