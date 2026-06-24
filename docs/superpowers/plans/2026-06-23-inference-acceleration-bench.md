# Inference Acceleration Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone benchmark that measures the detector's real inference latency under the auto-selected execution provider versus a CPU-forced baseline, and reports a plain verdict ("accelerated" / "no-benefit" / "cpu-only") so a user can tell whether their GPU is actually doing work.

**Architecture:** A new pure module `autoptz/engine/runtime/bench.py` times ORT `InferenceSession.run` calls (warmup + N timed runs → median/p95/fps), builds the chosen-EP and a CPU-forced session for the same model via the existing `make_session`, compares their medians into an `AccelReport`, and renders it. A `--bench` flag in `autoptz/__main__.py` resolves the detector model and prints/saves the report. No changes to the camera worker, supervisor, or live telemetry — those are deferred (see "Out of scope").

**Tech Stack:** Python 3.12, onnxruntime, numpy 2.x, argparse. No new dependencies.

**Context for the implementer:** This is the first, self-contained slice of the unified plan's "K1 measurement engine" / "P1 honest acceleration telemetry" items (see `~/.claude/plans/determine-a-way-to-polymorphic-prism.md`). The motivating real-world bug: on an Intel Mac with an AMD GPU, the CoreML EP can report itself as active while silently running every op on the CPU, so the GPU is never used and nobody knows. This tool measures the truth.

## Global Constraints

- Python **3.12+**; all code in `autoptz/engine/runtime/` is type-checked under **mypy strict** in CI — annotate fully. ONNX Runtime's type stubs are loose (`NodeArg.shape` is `list[int | str]`, `NodeArg.type` is `str`); annotate locals explicitly and use a narrow `# type: ignore[...]` only where an ORT stub genuinely lacks a type.
- Lint/format: **ruff** (`ruff check`, `ruff format --check`) must pass. Broad excepts use the existing convention `except Exception:  # noqa: BLE001`.
- Logging: module-level `log = logging.getLogger(__name__)`; never `print` from library code (the CLI runner is the only place that prints, and it prints to stdout deliberately).
- Tests run headless in CI with `QT_QPA_PLATFORM=offscreen`; this feature touches no Qt. Global pytest timeout is 60s — keep `warmup`/`runs` small in tests.
- Do **not** download models in tests. Build models in-process (minimal ONNX, as `tests/test_inference.py::test_make_session_cpu` does) or use `autoptz.engine.pipeline.detect.make_synthetic_detector_session`.
- Reuse existing inference primitives — do **not** re-implement EP selection. Use `make_session`, `get_best_ep`, `HardwarePrefs`, `EP`, `effective_precision` from `autoptz/engine/runtime/inference.py`.

## Out of scope (explicit — do NOT build here)

- Live wiring of the verdict into `camera_worker._runtime_services()` / `RuntimeServiceInfo` (that is Plan A-2).
- Tracking-quality replay scoring / clip corpus (that is Plan B).
- Per-stage (detect/track/face/pose) timing during a replay run.

## File Structure

- Create: `autoptz/engine/runtime/bench.py` — timing core, acceleration measurement, report dataclasses, CLI runner. One responsibility: measure and report inference performance. Pure (no Qt, no worker).
- Create: `tests/test_bench.py` — unit tests for all of the above using in-process ONNX models.
- Modify: `autoptz/__main__.py` — add the `--bench` / `--bench-tier` / `--bench-json` CLI flags that call into `bench.py`.

---

### Task 1: Latency timing core

**Files:**
- Create: `autoptz/engine/runtime/bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `onnxruntime` (`ort.InferenceSession`), `numpy`.
- Produces:
  - `@dataclass(frozen=True) class LatencyStats` with fields `runs: int`, `median_ms: float`, `p95_ms: float`, `mean_ms: float`, `fps: float`, and method `to_dict(self) -> dict[str, float | int]`.
  - `zeros_for_session(session: ort.InferenceSession) -> dict[str, np.ndarray]` — a feeds dict of correctly-typed all-zero inputs (symbolic/`<=0` dims become `1`).
  - `time_session(session: ort.InferenceSession, feeds: dict[str, np.ndarray] | None = None, *, warmup: int = 3, runs: int = 20) -> LatencyStats` — runs `warmup` untimed then `runs` timed `session.run(None, feeds)` calls; `feeds=None` means use `zeros_for_session`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench.py
"""Unit tests for autoptz.engine.runtime.bench."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from autoptz.engine.runtime.bench import (
    LatencyStats,
    time_session,
    zeros_for_session,
)


def _identity_session() -> ort.InferenceSession:
    """A trivial [1,4]->[1,4] float Identity model as an ORT CPU session."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "id", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def test_zeros_for_session_matches_input() -> None:
    sess = _identity_session()
    feeds = zeros_for_session(sess)
    assert set(feeds) == {"X"}
    assert feeds["X"].shape == (1, 4)
    assert feeds["X"].dtype == np.float32


def test_time_session_returns_positive_stats() -> None:
    sess = _identity_session()
    stats = time_session(sess, warmup=1, runs=5)
    assert isinstance(stats, LatencyStats)
    assert stats.runs == 5
    assert stats.median_ms > 0.0
    assert stats.p95_ms >= stats.median_ms
    assert stats.fps > 0.0


def test_latency_stats_to_dict_roundtrips() -> None:
    stats = LatencyStats(runs=5, median_ms=2.0, p95_ms=3.0, mean_ms=2.5, fps=500.0)
    d = stats.to_dict()
    assert d["runs"] == 5
    assert d["median_ms"] == 2.0
    assert d["fps"] == 500.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoptz.engine.runtime.bench'`

- [ ] **Step 3: Write minimal implementation**

```python
# autoptz/engine/runtime/bench.py
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

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

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
    feed_arg: dict[str, NDArray[np.generic]] = feeds
    for _ in range(max(0, warmup)):
        session.run(None, feed_arg)
    samples_ms: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        session.run(None, feed_arg)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    median_ms = statistics.median(samples_ms)
    return LatencyStats(
        runs=len(samples_ms),
        median_ms=median_ms,
        p95_ms=_percentile(samples_ms, 95.0),
        mean_ms=statistics.fmean(samples_ms),
        fps=(1000.0 / median_ms) if median_ms > 0 else 0.0,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint/type-check the new module**

Run: `ruff check autoptz/engine/runtime/bench.py && ruff format --check autoptz/engine/runtime/bench.py && mypy autoptz/engine/runtime/bench.py`
Expected: no errors. (If mypy flags an ORT `NodeArg` attribute, add a narrow `# type: ignore[attr-defined]` on that line only.)

- [ ] **Step 6: Commit**

```bash
git add autoptz/engine/runtime/bench.py tests/test_bench.py
git commit -m "feat(bench): inference latency timing core"
```

---

### Task 2: Acceleration measurement + verdict

**Files:**
- Modify: `autoptz/engine/runtime/bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: Task 1's `time_session`, `LatencyStats`; from `autoptz.engine.runtime.inference`: `EP`, `HardwarePrefs`, `get_best_ep`, `make_session`, `effective_precision`.
- Produces:
  - module constant `ACCEL_MIN_SPEEDUP: float = 1.15`.
  - `verdict(actual_ep: str, speedup: float) -> str` → `"cpu-only" | "accelerated" | "no-benefit"`.
  - `@dataclass(frozen=True) class AccelReport` with fields `model: str`, `requested_ep: str`, `actual_ep: str`, `precision: str`, `speedup: float`, `verdict: str`, `accel: LatencyStats`, `cpu: LatencyStats`; methods `summary(self) -> str` and `to_dict(self) -> dict[str, object]`.
  - `measure_acceleration(model_path: str | Path, *, prefs: HardwarePrefs | None = None, warmup: int = 3, runs: int = 20) -> AccelReport`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bench.py
from pathlib import Path

from autoptz.engine.runtime.bench import (
    ACCEL_MIN_SPEEDUP,
    AccelReport,
    LatencyStats,
    measure_acceleration,
    verdict,
)
from autoptz.engine.runtime.inference import EP


def _save_identity_model(tmp_path: Path) -> Path:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "id", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "identity.onnx"
    onnx.save(model, str(path))
    return path


def test_verdict_cpu_only_when_actual_is_cpu() -> None:
    assert verdict(EP.CPU.value, 1.0) == "cpu-only"
    # Even a high "speedup" is cpu-only if the EP actually in use is CPU.
    assert verdict(EP.CPU.value, 9.9) == "cpu-only"


def test_verdict_accelerated_above_threshold() -> None:
    assert verdict(EP.COREML.value, ACCEL_MIN_SPEEDUP) == "accelerated"
    assert verdict(EP.COREML.value, ACCEL_MIN_SPEEDUP + 1.0) == "accelerated"


def test_verdict_no_benefit_below_threshold() -> None:
    # Accelerator EP selected but not actually faster than CPU.
    assert verdict(EP.COREML.value, 1.0) == "no-benefit"
    assert verdict(EP.DIRECTML.value, 1.05) == "no-benefit"


def test_measure_acceleration_cpu_machine(tmp_path: Path) -> None:
    """On a CPU-only host (CI) the chosen EP is CPU → verdict 'cpu-only'."""
    model_path = _save_identity_model(tmp_path)
    report = measure_acceleration(model_path, warmup=1, runs=3)
    assert isinstance(report, AccelReport)
    assert report.actual_ep == EP.CPU.value
    assert report.verdict == "cpu-only"
    assert report.accel.runs == 3
    assert report.cpu.runs == 3
    assert report.speedup > 0.0


def test_accel_report_summary_and_dict() -> None:
    stats = LatencyStats(runs=3, median_ms=2.0, p95_ms=3.0, mean_ms=2.5, fps=500.0)
    report = AccelReport(
        model="m.onnx",
        requested_ep=EP.COREML.value,
        actual_ep=EP.CPU.value,
        precision="fp32",
        speedup=1.0,
        verdict="cpu-only",
        accel=stats,
        cpu=stats,
    )
    assert "cpu-only" in report.summary().lower()
    assert "CoreML" in report.summary()  # label is stripped of "ExecutionProvider"
    d = report.to_dict()
    assert d["verdict"] == "cpu-only"
    assert isinstance(d["accel"], dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench.py -v`
Expected: FAIL — `ImportError: cannot import name 'measure_acceleration' from 'autoptz.engine.runtime.bench'`

- [ ] **Step 3: Write minimal implementation**

Add these imports to the top of `autoptz/engine/runtime/bench.py` (merge with existing):

```python
from pathlib import Path

from autoptz.engine.runtime.inference import (
    EP,
    HardwarePrefs,
    effective_precision,
    get_best_ep,
    make_session,
)
```

Append to `autoptz/engine/runtime/bench.py`:

```python
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
        return (
            f"{_ep_label(self.actual_ep)} · {self.precision} · "
            f"{self.speedup:.2f}× CPU ({_VERDICT_BLURB.get(self.verdict, self.verdict)})"
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

    speedup = (cpu_stats.median_ms / accel_stats.median_ms) if accel_stats.median_ms > 0 else 0.0
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bench.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Lint/type-check**

Run: `ruff check autoptz/engine/runtime/bench.py && ruff format --check autoptz/engine/runtime/bench.py && mypy autoptz/engine/runtime/bench.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add autoptz/engine/runtime/bench.py tests/test_bench.py
git commit -m "feat(bench): acceleration measurement + verdict"
```

---

### Task 3: `--bench` CLI entry

**Files:**
- Modify: `autoptz/engine/runtime/bench.py` (add `run_acceleration_bench`)
- Modify: `autoptz/__main__.py` (add flags + dispatch)
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: Task 2's `measure_acceleration`, `AccelReport`; `autoptz.engine.runtime.models.default_manager`.
- Produces: `run_acceleration_bench(tier: str = "auto", json_path: str | None = None, *, warmup: int = 5, runs: int = 30) -> int` — resolves the detector model (no download), measures, prints a report to stdout, optionally writes JSON, returns a process exit code (0 ok, 1 no model).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bench.py
import json
import types

from autoptz.engine.runtime import bench as bench_mod
from autoptz.engine.runtime.bench import run_acceleration_bench


def test_run_acceleration_bench_no_model(monkeypatch, capsys) -> None:
    fake_mgr = types.SimpleNamespace(
        ensure_detector=lambda *a, **k: None, last_error="no model"
    )
    monkeypatch.setattr(bench_mod, "default_manager", lambda: fake_mgr)
    code = run_acceleration_bench(tier="auto")
    assert code == 1
    assert "no detector model" in capsys.readouterr().out.lower()


def test_run_acceleration_bench_reports_and_writes_json(
    tmp_path, monkeypatch, capsys
) -> None:
    model_path = _save_identity_model(tmp_path)
    fake_mgr = types.SimpleNamespace(
        ensure_detector=lambda *a, **k: str(model_path), last_error=""
    )
    monkeypatch.setattr(bench_mod, "default_manager", lambda: fake_mgr)
    json_out = tmp_path / "bench.json"
    code = run_acceleration_bench(tier="auto", json_path=str(json_out), warmup=1, runs=3)
    assert code == 0
    out = capsys.readouterr().out
    assert "cpu-only" in out.lower()  # CI host has no accelerator
    data = json.loads(json_out.read_text())
    assert data["verdict"] == "cpu-only"
    assert data["actual_ep"] == "CPUExecutionProvider"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bench.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_acceleration_bench'` (and `default_manager` not yet referenced in `bench`).

- [ ] **Step 3: Write the runner in `bench.py`**

Add the import near the other `autoptz` imports in `autoptz/engine/runtime/bench.py`:

```python
from autoptz.engine.runtime.models import default_manager
```

Append to `autoptz/engine/runtime/bench.py`:

```python
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
```

- [ ] **Step 4: Wire the CLI flags in `autoptz/__main__.py`**

Two exact edits in `main()`. First, register the flags — replace the existing `--log-level` argument block (find this exact text):

```python
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
```

with:

```python
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark detector inference (auto EP vs CPU) and exit.",
    )
    parser.add_argument(
        "--bench-tier",
        default="auto",
        help="Detector tier to benchmark (default: auto).",
    )
    parser.add_argument(
        "--bench-json",
        default=None,
        help="Write the benchmark report as JSON to this path.",
    )
    args = parser.parse_args()
```

Second, dispatch it — replace the existing selftest/UI-launch tail (find this exact text):

```python
    if args.selftest:
        selftest()
        return

    # Default: launch the UI
    from autoptz.ui.app import run

    sys.exit(run())
```

with:

```python
    if args.selftest:
        selftest()
        return

    if args.bench:
        from autoptz.engine.runtime.bench import run_acceleration_bench

        raise SystemExit(
            run_acceleration_bench(tier=args.bench_tier, json_path=args.bench_json)
        )

    # Default: launch the UI
    from autoptz.ui.app import run

    sys.exit(run())
```

- [ ] **Step 5: Run the new tests**

Run: `python -m pytest tests/test_bench.py -v`
Expected: PASS (10 tests total)

- [ ] **Step 6: Smoke-test the CLI end-to-end (real machine)**

Run: `python -m autoptz --bench --log-level WARNING`
Expected: prints the report block; on this Apple Silicon / Intel Mac it reveals the actual EP and whether CoreML beats CPU. (Acceptance: the command exits 0 and prints a `verdict:` line. On a CPU-only CI host it would print `cpu-only`.)

- [ ] **Step 7: Lint/type-check both files**

Run: `ruff check autoptz/engine/runtime/bench.py autoptz/__main__.py && ruff format --check autoptz/engine/runtime/bench.py autoptz/__main__.py && mypy autoptz/engine/runtime/bench.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add autoptz/engine/runtime/bench.py autoptz/__main__.py tests/test_bench.py
git commit -m "feat(bench): --bench CLI to report GPU acceleration verdict"
```

---

## Final verification

- [ ] Run the full bench test file: `python -m pytest tests/test_bench.py -v` — all pass.
- [ ] Run the broader runtime suite to confirm no regressions: `python -m pytest tests/test_inference.py tests/test_diagnostics.py -v`.
- [ ] `ruff check autoptz/ && mypy autoptz/engine/runtime/bench.py` — clean.
- [ ] Manual: `python -m autoptz --bench` on the dev Mac prints a believable EP + verdict; on the user's Intel-Mac+AMD machine, confirm whether CoreML reports `accelerated` or `no-benefit` (the whole point).

## Follow-ups (separate plans)

- **Plan A-2:** surface `AccelReport.verdict` live in `camera_worker._runtime_services()` (extend the detector `RuntimeServiceInfo` row's `detail`/`confidence`) so the verdict shows in the Services panel, not just the CLI.
- **Plan B:** tracking-quality replay harness + labeled clip corpus (ID-switches, time-on-target, oscillation).
