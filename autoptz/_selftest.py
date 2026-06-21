"""Helper module for --selftest subprocess workers.

Must be in a named module (not __main__) so multiprocessing 'spawn'
can pickle and locate the target function in the child process.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any


def shm_reader_proc(
    shm_name: str,
    height: int,
    width: int,
    channels: int,
    result_q: Any,
) -> None:
    """Runs in a spawned OS process; reads a frame from shm and reports via queue."""
    # Ensure autoptz is importable in the fresh child interpreter
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from autoptz.engine.runtime.shm import ShmReader

    reader = ShmReader(shm_name, height, width, channels)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        got = reader.latest()
        if got is not None:
            hdr, frame = got
            result_q.put(
                {
                    "ok": True,
                    "seq": hdr.seq,
                    "sum": int(frame.sum()),
                    "shape": list(frame.shape),
                }
            )
            reader.close()
            return
        time.sleep(0.005)
    reader.close()
    result_q.put({"ok": False, "error": "timeout waiting for frame"})
