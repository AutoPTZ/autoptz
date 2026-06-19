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

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("autoptz")


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

    print("AutoPTZ v2 — selftest")
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
    print(f"[2] SHM ring buffer:      OK  seq={seq} shape={list(frame.shape)} ✓")

    tel = TelemetryMsg(camera_id="test-cam", seq=42, fps=30.0, ep=ep.value)
    packed = tel.to_msgpack()
    tel2 = TelemetryMsg.from_msgpack(packed)
    assert tel2.camera_id == tel.camera_id
    print(f"[3] Telemetry round-trip: OK  {len(packed)} bytes ✓")

    cmd = AddCameraCmd(
        camera_id="cam-uuid-00000001",
        source_uri="rtsp://192.168.1.100/stream1",
        display_name="Stage Left",
    )
    packed_cmd = cmd.to_msgpack()
    cmd2 = BaseCommand.from_msgpack(packed_cmd)
    assert cmd2.kind == CmdKind.ADD_CAMERA
    print(f"[4] Command round-trip:   OK  {len(packed_cmd)} bytes ✓")

    print("=" * 50)
    print("All selftest checks passed.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="autoptz", description="AutoPTZ v2")
    parser.add_argument("--selftest", action="store_true",
                        help="Run foundation selftest and exit")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.selftest:
        selftest()
        return

    # Default: launch the UI
    from autoptz.ui.app import run
    sys.exit(run())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
