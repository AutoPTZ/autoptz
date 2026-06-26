"""python -m autoptz entry point.

Usage::

    python -m autoptz              # launch the UI (default)
    python -m autoptz --selftest   # verify foundations and exit
    python -m autoptz --help
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from typing import Any

from autoptz.logsetup import install_console_logging

# Coloured console logging (level + per-camera tint; TTY-gated, no-op when piped).
install_console_logging(level=logging.WARNING)
logger = logging.getLogger("autoptz")


def _ensure_utf8_console() -> None:
    """Prefer UTF-8 console output, but never fail startup if unavailable."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _status_mark() -> str:
    """Return a console-safe pass marker for selftest output."""
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "✓" if "utf" in enc else "OK"


# ── selftest ──────────────────────────────────────────────────────────────────


def selftest() -> None:
    import numpy as np

    from autoptz._selftest import shm_reader_proc
    from autoptz.engine.runtime.inference import get_best_ep
    from autoptz.engine.runtime.messages import (
        AddCameraCmd,
        BaseCommand,
        CmdKind,
        TelemetryMsg,
    )
    from autoptz.engine.runtime.shm import ShmWriter

    ok = _status_mark()

    print("AutoPTZ v2 - selftest")
    print("=" * 50)

    ep = get_best_ep()
    print(f"[1] EP selection:         {ep.value}")

    H, W, C = 360, 640, 3
    SHM_NAME = f"autoptz_st_{os.getpid()}"

    ctx = multiprocessing.get_context("spawn")
    result_q: multiprocessing.Queue[Any] = ctx.Queue()

    with ShmWriter(SHM_NAME, H, W, C) as writer:
        proc = ctx.Process(
            target=shm_reader_proc,
            args=(SHM_NAME, H, W, C, result_q),
            daemon=True,
        )
        proc.start()

        frame = np.full((H, W, C), 42, dtype=np.uint8)
        seq = writer.push(frame)
        expected_sum = int(frame.sum())

        try:
            result = result_q.get(timeout=12.0)
        except Exception:
            proc.kill()
            proc.join(timeout=2.0)
            print("[2] SHM ring buffer:      FAIL (reader timed out)")
            sys.exit(1)
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()

    if not result["ok"]:
        print(f"[2] SHM ring buffer:      FAIL ({result.get('error')})")
        sys.exit(1)

    assert result["seq"] == seq
    assert result["sum"] == expected_sum
    print(f"[2] SHM ring buffer:      OK  seq={seq} shape={list(frame.shape)} {ok}")

    tel = TelemetryMsg(camera_id="test-cam", seq=42, fps=30.0, ep=ep.value)
    packed = tel.to_msgpack()
    tel2 = TelemetryMsg.from_msgpack(packed)
    assert tel2.camera_id == tel.camera_id
    print(f"[3] Telemetry round-trip: OK  {len(packed)} bytes {ok}")

    cmd = AddCameraCmd(
        camera_id="cam-uuid-00000001",
        source_uri="rtsp://192.168.1.100/stream1",
        display_name="Stage Left",
    )
    packed_cmd = cmd.to_msgpack()
    cmd2 = BaseCommand.from_msgpack(packed_cmd)
    assert cmd2.kind == CmdKind.ADD_CAMERA
    print(f"[4] Command round-trip:   OK  {len(packed_cmd)} bytes {ok}")

    print("=" * 50)
    print("All selftest checks passed.")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    _ensure_utf8_console()

    parser = argparse.ArgumentParser(prog="autoptz", description="AutoPTZ v2")
    parser.add_argument("--selftest", action="store_true", help="Run foundation selftest and exit")
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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the AutoPTZ Mark throughput benchmark (headless) and exit.",
    )
    parser.add_argument(
        "--benchmark-profile",
        default="full",
        choices=["full", "streams"],
        help="Benchmark profile: 'full' (all inference) or 'streams' (capture only).",
    )
    parser.add_argument(
        "--benchmark-duration",
        type=float,
        default=20.0,
        help="Per-step dwell in seconds (default 20).",
    )
    parser.add_argument(
        "--benchmark-max-cameras",
        type=int,
        default=16,
        help="Maximum synthetic cameras to ramp to (default 16).",
    )
    parser.add_argument(
        "--benchmark-floor",
        type=float,
        default=24.0,
        help="Minimum sustained fps floor for a camera count to count (default 24).",
    )
    parser.add_argument(
        "--benchmark-json",
        default=None,
        help="Write the AutoPTZ Mark report as JSON to this path.",
    )
    parser.add_argument(
        "--mark",
        action="store_true",
        help="Open the AutoPTZ Mark GUI benchmark window (relaunched from the Help menu).",
    )
    args = parser.parse_args()

    install_console_logging(level=getattr(logging, args.log_level))

    if args.selftest:
        selftest()
        return

    if args.bench:
        from autoptz.engine.runtime.bench import run_acceleration_bench

        raise SystemExit(run_acceleration_bench(tier=args.bench_tier, json_path=args.bench_json))

    if args.benchmark:
        from autoptz.benchmark import run_benchmark

        raise SystemExit(
            run_benchmark(
                profile=args.benchmark_profile,
                floor_fps=args.benchmark_floor,
                max_cameras=args.benchmark_max_cameras,
                dwell_s=args.benchmark_duration,
                json_path=args.benchmark_json,
            )
        )

    # Default: launch the UI
    from autoptz.ui.app import run

    sys.exit(run())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
