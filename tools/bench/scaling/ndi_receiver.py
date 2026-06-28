"""NDI receiver benchmark — drives the REAL AutoPTZ Supervisor over N ndi:// cameras
and reports the new Phase-0 telemetry (delivered fps, drop estimate, end-to-end
latency) + App CPU. Senders run in a SEPARATE process (bench_sender.py) so this
process's CPU reflects the app's real receive+inference load.

Usage: python bench_receiver.py <N> <profile full|streams> <duration_s> [warmup_s]
Prints a JSON summary line prefixed RESULT_JSON.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication

from autoptz.benchmark.ndi_sim import _add_ndi_camera
from autoptz.benchmark.profiles import get_profile
from autoptz.config.store import ConfigStore
from autoptz.engine.runtime.diagnostics import system_metrics
from autoptz.engine.supervisor import Supervisor
from autoptz.ui.engine_client import EngineClient

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
PROFILE = sys.argv[2] if len(sys.argv) > 2 else "full"
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
WARMUP = float(sys.argv[4]) if len(sys.argv) > 4 else (12.0 if PROFILE == "full" else 5.0)


def discover(n: int, timeout_s: float = 20.0) -> list[str]:
    from cyndilib.finder import Finder

    f = Finder()
    f.open()
    want = {f"AutoPTZ Bench Cam {i + 1}" for i in range(n)}
    resolved: dict[str, str] = {}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and len(resolved) < n:
        try:
            f.wait_for_sources(0.3)
        except Exception:
            pass
        for full in (str(s) for s in f.iter_sources()):
            for short in want:
                if full.endswith(f"({short})") or full == short:
                    resolved[short] = full
        time.sleep(0.05)
    f.close()
    return [
        resolved[f"AutoPTZ Bench Cam {i + 1}"]
        for i in range(n)
        if f"AutoPTZ Bench Cam {i + 1}" in resolved
    ]


def main() -> None:
    app = QCoreApplication(sys.argv[:1])
    names = discover(N)
    if len(names) < N:
        print(f"DISCOVER_INCOMPLETE found {len(names)}/{N}", flush=True)
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    client = EngineClient(store=ConfigStore(db_path=Path(db), debounce_s=0))
    sup = Supervisor(client, store=client._store)
    sup.prime_features(dict(get_profile(PROFILE).features))
    cids = [_add_ndi_camera(client, name, i) for i, name in enumerate(names)]
    sup.start(run_pump=False)
    import os as _os

    print(
        f"SCHED_ENGAGED {getattr(sup, '_inference_scheduler', None) is not None} "
        f"flag={_os.environ.get('AUTOPTZ_INFERENCE_SCHEDULER')} "
        f"MS_ENGAGED {getattr(sup, '_model_server_proc', None) is not None} "
        f"ms_flag={_os.environ.get('AUTOPTZ_MODEL_SERVER')}",
        flush=True,
    )

    def pump(seconds: float) -> None:
        end = time.monotonic() + seconds
        last_tick = 0.0
        while time.monotonic() < end:
            app.processEvents()
            if time.monotonic() - last_tick > 0.5:
                try:
                    sup.tick()
                except Exception:
                    pass
                last_tick = time.monotonic()
            time.sleep(0.01)

    system_metrics()  # prime
    pump(WARMUP)

    samples: list[dict] = []
    t_end = time.monotonic() + DURATION
    while time.monotonic() < t_end:
        pump(1.0)
        m = system_metrics()
        per = []
        for cid in cids:
            rec = client.cameraModel.get_record(cid)
            tm = getattr(rec, "telemetry", None) if rec else None
            if tm is None:
                continue
            per.append(
                {
                    "fps": float(getattr(tm, "fps", 0.0)),
                    "delivered_fps": float(getattr(tm, "delivered_fps", 0.0)),
                    "source_fps": float(getattr(tm, "source_fps", 0.0)),
                    "drops": int(getattr(tm, "frames_dropped_est", 0)),
                    "e2e_ms": float(getattr(tm, "end_to_end_ms", 0.0)),
                    "capture_age_ms": float(getattr(tm, "capture_age_ms", 0.0)),
                    # detect_ms is the alive/dead proof for the model-server: a dead
                    # server makes detect() block the full IPC timeout (~5000ms); a live
                    # one returns in ANE+IPC time (tens of ms). Stall age confirms
                    # inference is completing regularly, not wedged.
                    "detect_ms": float(getattr(tm, "detect_ms", 0.0)),
                    "stall_s": float(getattr(tm, "inference_stall_age_s", 0.0)),
                    "qd": int(getattr(tm, "ndi_queue_depth", -1)),
                }
            )
        samples.append(
            {
                "app_cpu": m.get("app_cpu_percent", 0.0),
                "sys_cpu": m.get("cpu_percent", 0.0),
                "app_mem_mb": m.get("app_rss_mb", 0.0),
                "per": per,
            }
        )

    # Steady-state aggregation (all collected samples are post-warmup).
    def med(xs):
        return round(statistics.median(xs), 1) if xs else 0.0

    app_cpu = [s["app_cpu"] for s in samples]
    sys_cpu = [s["sys_cpu"] for s in samples]
    # per-camera averages over the run
    fps_all = [c["fps"] for s in samples for c in s["per"]]
    deliv_all = [c["delivered_fps"] for s in samples for c in s["per"]]
    first = samples[0]["per"] if samples else []
    last = samples[-1]["per"] if samples else []
    total_drops = sum(c["drops"] for c in last)
    # Steady-state drops: delta across the post-warmup window (excludes the
    # warmup/model-load period that inflates the cumulative count).
    drops_start = sum(c["drops"] for c in first)
    span_s = max(1e-6, (len(samples) - 1)) if len(samples) > 1 else 1.0
    drops_delta = max(0, total_drops - drops_start)
    e2e_all = [c["e2e_ms"] for s in samples for c in s["per"] if c["e2e_ms"] > 0]
    detect_all = [c["detect_ms"] for s in samples for c in s["per"] if c["detect_ms"] > 0]
    stall_all = [c["stall_s"] for s in samples for c in s["per"]]

    result = {
        "N": N,
        "profile": PROFILE,
        "discovered": len(names),
        "app_cpu_median": med(app_cpu),
        "app_cpu_max": round(max(app_cpu), 1) if app_cpu else 0.0,
        "sys_cpu_median": med(sys_cpu),
        "app_mem_mb": round(samples[-1]["app_mem_mb"], 0) if samples else 0,
        "fps_median_per_cam": med(fps_all),
        "delivered_fps_median": med(deliv_all),
        "drops_steady_window": drops_delta,
        "drops_per_s_steady": round(drops_delta / span_s, 1),
        "e2e_ms_median": med(e2e_all),
        # Detection-health: detect_ms_median proves inference actually ran (and how
        # fast); detect_active_pct is the share of camera-samples where detection
        # executed at all (0 => detection dead). stall_s_max flags a wedged pipeline.
        "detect_ms_median": med(detect_all),
        "detect_active_pct": round(
            100.0 * len(detect_all) / max(1, sum(len(s["per"]) for s in samples)), 1
        ),
        "stall_s_max": round(max(stall_all), 1) if stall_all else 0.0,
        "cams_reporting": len(last),
    }
    print("RESULT_JSON " + json.dumps(result), flush=True)
    try:
        sup.stop()
    except Exception:
        pass


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
