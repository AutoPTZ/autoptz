"""AutoPTZ Mark — a headless throughput benchmark for the engine pipeline."""

from __future__ import annotations

from autoptz.benchmark.profiles import PROFILES, BenchmarkProfile, get_profile

__all__ = [
    "PROFILES",
    "BenchmarkProfile",
    "get_profile",
]
