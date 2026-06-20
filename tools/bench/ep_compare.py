#!/usr/bin/env python3
"""Compare ONNX Runtime execution providers for the detector on THIS machine.

Times each available EP through the real :func:`make_session` factory (so the
tuned SessionOptions + per-EP options under test are exactly what the app uses),
and prints a latency table. Use it to confirm the GPU/accelerator path is faster
than CPU on your hardware, and that TensorRT's engine cache makes the 2nd run
fast.

Usage::

    python tools/bench/ep_compare.py                 # auto-resolve cached model
    python tools/bench/ep_compare.py --model m.onnx  # explicit model
    python tools/bench/ep_compare.py --runs 100 --precision fp16

Torch-free: needs only onnxruntime + numpy.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from autoptz.engine.runtime.inference import EP, make_session


def _resolve_model(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).is_file() else None
    env = os.environ.get("AUTOPTZ_MODEL_PATH")
    if env and Path(env).is_file():
        return env
    try:
        from autoptz.engine.runtime.models import default_manager

        return default_manager().ensure_detector()
    except Exception:  # noqa: BLE001
        return None


def _bench_ep(model: str, ep: EP, runs: int, precision: str) -> dict[str, object] | None:
    os.environ["AUTOPTZ_FORCE_EP"] = ep.value
    os.environ["AUTOPTZ_PRECISION"] = precision
    try:
        session = make_session(model)
    except Exception as exc:  # noqa: BLE001
        return {"ep": ep.value, "error": str(exc)[:60]}

    actual = session.get_providers()[0]
    inp = session.get_inputs()[0]
    shape = [
        d if isinstance(d, int) and d > 0 else 1 if i == 0 else 640 for i, d in enumerate(inp.shape)
    ]
    x = np.random.rand(*shape).astype(np.float32)
    out_names = [o.name for o in session.get_outputs()]

    for _ in range(5):  # warmup (also builds the TRT engine on first ever run)
        session.run(out_names, {inp.name: x})

    times: list[float] = []
    for _ in range(runs):
        t = time.perf_counter()
        session.run(out_names, {inp.name: x})
        times.append((time.perf_counter() - t) * 1000.0)

    mean = statistics.mean(times)
    return {
        "ep": actual,
        "requested": ep.value,
        "mean_ms": mean,
        "p50_ms": statistics.median(times),
        "fps": 1000.0 / mean if mean else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare ORT EPs for the AutoPTZ detector.")
    ap.add_argument("--model", default=None, help="ONNX model path (default: cached detector)")
    ap.add_argument("--runs", type=int, default=50, help="timed runs per EP (default 50)")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16"])
    args = ap.parse_args()

    model = _resolve_model(args.model)
    if model is None:
        print(
            "ERROR: no model. Pass --model, set AUTOPTZ_MODEL_PATH, or pre-fetch with "
            "`python -m tools.fetch_models`."
        )
        return 1

    avail = set(ort.get_available_providers())
    candidates = [ep for ep in EP if ep.value in avail]
    print(f"model: {Path(model).name}   runs: {args.runs}   precision: {args.precision}")
    print(f"available EPs: {sorted(avail)}\n")
    print(f"{'requested EP':<26}{'actual EP':<26}{'mean ms':>10}{'p50 ms':>10}{'fps':>9}")
    print("-" * 81)

    for ep in candidates:
        r = _bench_ep(model, ep, args.runs, args.precision)
        if r is None:
            continue
        if "error" in r:
            print(f"{ep.value:<26}{'FAILED':<26}{r['error']}")
            continue
        print(
            f"{r['requested']:<26}{r['ep']:<26}"
            f"{r['mean_ms']:>10.2f}{r['p50_ms']:>10.2f}{r['fps']:>9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
