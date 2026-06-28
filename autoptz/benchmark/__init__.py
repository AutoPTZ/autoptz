"""AutoPTZ Mark — a headless throughput benchmark for the engine pipeline."""

from __future__ import annotations

from autoptz.benchmark.profiles import PROFILES, BenchmarkProfile, get_profile
from autoptz.benchmark.results import (
    MarkResultBundle,
    collect_machine_info,
    save_mark_result,
)
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
    "MarkResultBundle",
    "StepResult",
    "collect_machine_info",
    "get_profile",
    "run_benchmark",
    "save_mark_result",
]
