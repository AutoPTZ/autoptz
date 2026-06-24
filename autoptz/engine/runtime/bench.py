"""Measure real inference performance and whether acceleration is helping.

Times ONNX Runtime sessions (warmup + N timed runs → median/p95/fps) and
compares the auto-selected execution provider against a CPU-forced baseline so a
user can see whether their GPU is actually doing work — the truth that the EP
*label* alone does not tell you (e.g. CoreML on an Intel Mac can report itself
active while silently running every op on the CPU).
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from autoptz.engine.runtime.inference import (
    EP,
    HardwarePrefs,
    effective_precision,
    get_best_ep,
    make_session,
)
from autoptz.engine.runtime.models import default_manager

log = logging.getLogger(__name__)

# ORT input type string ("tensor(float)") → numpy dtype.
_ORT_TO_NUMPY: dict[str, type[np.generic]] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
}


@dataclass(frozen=True)
class LatencyStats:
    """Timing summary for a series of inference runs."""

    runs: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    fps: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "runs": self.runs,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "mean_ms": self.mean_ms,
            "fps": self.fps,
        }


def zeros_for_session(session: ort.InferenceSession) -> dict[str, NDArray[np.generic]]:
    """All-zero feeds matching every input's shape/dtype (symbolic dims → 1)."""
    feeds: dict[str, NDArray[np.generic]] = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dtype = _ORT_TO_NUMPY.get(inp.type, np.float32)
        feeds[inp.name] = np.zeros(tuple(shape), dtype=dtype)
    return feeds


def _percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in [0,100]); pure-python, no numpy dep."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def time_session(
    session: ort.InferenceSession,
    feeds: dict[str, NDArray[np.generic]] | None = None,
    *,
    warmup: int = 3,
    runs: int = 20,
) -> LatencyStats:
    """Run *warmup* untimed then *runs* timed inferences; return latency stats."""
    if feeds is None:
        feeds = zeros_for_session(session)
    for _ in range(max(0, warmup)):
        session.run(None, feeds)
    samples_ms: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        session.run(None, feeds)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    median_ms = statistics.median(samples_ms)
    return LatencyStats(
        runs=len(samples_ms),
        median_ms=median_ms,
        p95_ms=_percentile(samples_ms, 95.0),
        mean_ms=statistics.fmean(samples_ms),
        fps=(1000.0 / median_ms) if median_ms > 0 else 0.0,
    )


#: Minimum chosen-EP-vs-CPU speedup to call an accelerator a real win.
ACCEL_MIN_SPEEDUP: float = 1.15

_VERDICT_BLURB: dict[str, str] = {
    "accelerated": "GPU/accelerator is helping",
    "no-benefit": "accelerator selected but no faster than CPU",
    "cpu-only": "running on CPU",
}


def _ep_label(ep_value: str) -> str:
    """`CoreMLExecutionProvider` → `CoreML` for human-facing output."""
    return ep_value.replace("ExecutionProvider", "")


def verdict(actual_ep: str, speedup: float) -> str:
    """Classify an acceleration result. CPU-in-use always wins over speedup."""
    if actual_ep == EP.CPU.value:
        return "cpu-only"
    return "accelerated" if speedup >= ACCEL_MIN_SPEEDUP else "no-benefit"


@dataclass(frozen=True)
class AccelReport:
    """Auto-selected EP vs CPU-forced baseline for one model."""

    model: str
    requested_ep: str
    actual_ep: str
    precision: str
    speedup: float
    verdict: str
    accel: LatencyStats
    cpu: LatencyStats

    def summary(self) -> str:
        actual = _ep_label(self.actual_ep)
        if self.actual_ep != self.requested_ep:
            device = f"{actual} (requested: {_ep_label(self.requested_ep)})"
        else:
            device = actual
        return (
            f"{device} · {self.precision} · {self.speedup:.2f}× CPU "
            f"({self.verdict}: {_VERDICT_BLURB.get(self.verdict, self.verdict)})"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "requested_ep": self.requested_ep,
            "actual_ep": self.actual_ep,
            "precision": self.precision,
            "speedup": self.speedup,
            "verdict": self.verdict,
            "accel": self.accel.to_dict(),
            "cpu": self.cpu.to_dict(),
        }


def measure_acceleration(
    model_path: str | Path,
    *,
    prefs: HardwarePrefs | None = None,
    warmup: int = 3,
    runs: int = 20,
) -> AccelReport:
    """Time the auto-selected EP against a CPU-forced baseline for *model_path*."""
    requested = get_best_ep(prefs)
    accel_session = make_session(model_path, prefs=prefs)
    actual_ep = accel_session.get_providers()[0]
    accel_stats = time_session(accel_session, warmup=warmup, runs=runs)

    cpu_session = make_session(model_path, prefs=HardwarePrefs(force_ep=EP.CPU))
    cpu_stats = time_session(cpu_session, warmup=warmup, runs=runs)

    # Neutral (1.0) fallback for a degenerate zero-latency reading, so it can't
    # be misclassified as "no-benefit"; never fires for a real model.
    speedup = (cpu_stats.median_ms / accel_stats.median_ms) if accel_stats.median_ms > 0 else 1.0
    return AccelReport(
        model=Path(model_path).name,
        requested_ep=requested.value,
        actual_ep=actual_ep,
        precision=effective_precision(actual_ep, prefs),
        speedup=speedup,
        verdict=verdict(actual_ep, speedup),
        accel=accel_stats,
        cpu=cpu_stats,
    )


def run_acceleration_bench(
    tier: str = "auto",
    json_path: str | None = None,
    *,
    warmup: int = 5,
    runs: int = 30,
) -> int:
    """Resolve the detector model, measure acceleration, print/save a report.

    Returns a process exit code: 0 on success, 1 when no model is available.
    Prints to stdout deliberately — this is the CLI face of the bench.
    """
    manager = default_manager()
    model_path = manager.ensure_detector(tier=tier, allow_download=False)
    if not model_path:
        reason = getattr(manager, "last_error", "") or "model not found"
        print(f"No detector model available for tier {tier!r}: {reason}")
        return 1

    report = measure_acceleration(model_path, warmup=warmup, runs=runs)
    print("AutoPTZ inference acceleration bench")
    print(f"  model:        {report.model}")
    print(f"  requested EP: {_ep_label(report.requested_ep)}")
    print(f"  actual EP:    {_ep_label(report.actual_ep)}  ({report.precision})")
    print(f"  accel:        {report.accel.median_ms:.2f} ms  ({report.accel.fps:.1f} fps)")
    print(f"  cpu baseline: {report.cpu.median_ms:.2f} ms  ({report.cpu.fps:.1f} fps)")
    print(f"  speedup:      {report.speedup:.2f}× CPU")
    print(f"  verdict:      {report.verdict} — {report.summary()}")
    if report.verdict == "no-benefit":
        print(
            "  ⚠ The selected accelerator is not faster than CPU — the GPU is "
            "likely not engaged (common on Intel Macs with AMD GPUs via CoreML)."
        )

    if json_path:
        import json as _json

        Path(json_path).write_text(_json.dumps(report.to_dict(), indent=2))
        print(f"  wrote: {json_path}")
    return 0
