"""Prototype: N camera PROCESSES + ONE shared model-server process.

Goal — prove the scalable architecture's properties on real NDI + real detector:
  * camera processes escape the GIL → capture/track run truly in parallel,
  * exactly ONE model set lives in the server process → no per-process RAM cliff,
  * detection is a fairly-shared ANE-bound resource.

Each camera process: a capture thread pulls its NDI source at full rate into a
single-slot shared-memory frame buffer (latest-wins) and counts fps; a delegation
loop asks the server to detect its latest frame and counts detections. The server
process holds the one detector and serves requests round-robin-ish (FIFO with one
outstanding request per camera). The orchestrator measures aggregate fps, detection
rate, total CPU (whole process tree), and RAM.

Usage: python proto_mpserver.py <N> <width> <height> <fps> <duration_s>
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from multiprocessing.shared_memory import SharedMemory


def _camera_proc(cam_id, ndi_name, shm_name, req_q, resp_q, w, h, fps, dur, result_q):  # noqa: ANN001
    import threading

    import numpy as np

    shm = SharedMemory(name=shm_name)
    arr = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf)

    from autoptz.engine.pipeline.ingest import NDIAdapter

    adapter = NDIAdapter(cam_id, ndi_name, target_fps=fps)
    if not adapter._open():
        result_q.put((cam_id, 0.0, 0.0, "open-failed"))
        return

    stop = threading.Event()
    cap = {"n": 0}

    def _capture():
        import cv2

        period = 1.0 / max(1.0, fps)
        next_t = time.monotonic()
        while not stop.is_set():
            frame = adapter._read_frame()
            if frame is not None:
                if frame.shape != (h, w, 3):
                    frame = cv2.resize(frame, (w, h))
                arr[:] = frame  # latest-wins single slot
                cap["n"] += 1
            next_t += period  # pace to the source rate (don't spin on FrameSync re-reads)
            slp = next_t - time.monotonic()
            if slp > 0:
                time.sleep(slp)
            elif slp < -period:
                next_t = time.monotonic()

    t = threading.Thread(target=_capture, daemon=True)
    t.start()

    # Let capture prime, then delegate detection in a loop (one outstanding per camera).
    time.sleep(0.5)
    det = {"n": 0}
    end = time.monotonic() + dur
    t0 = time.monotonic()
    while time.monotonic() < end:
        try:
            req_q.put(cam_id)
            resp_q.get(timeout=3.0)
            det["n"] += 1
        except Exception:
            break
    elapsed = max(1e-6, time.monotonic() - t0)
    stop.set()
    t.join(timeout=1.0)
    cap_dur = max(1e-6, dur)
    result_q.put((cam_id, cap["n"] / cap_dur, det["n"] / elapsed, "ok"))
    try:
        adapter._close()
    except Exception:
        pass


def _server_proc(shm_names, req_q, resp_qs, w, h, ready_ev, stop_ev, served_q):  # noqa: ANN001
    import numpy as np

    from autoptz.engine.pipeline.pool import build_inference_pool

    pool = build_inference_pool(detector_tier="auto", unified_pose=False, allow_model_download=False)
    detector = pool.detector() if pool is not None else None
    views = {}
    shms = {}
    for cid, name in shm_names.items():
        s = SharedMemory(name=name)
        shms[cid] = s
        views[cid] = np.ndarray((h, w, 3), dtype=np.uint8, buffer=s.buf)
    ready_ev.set()
    served = 0
    while not stop_ev.is_set():
        try:
            cid = req_q.get(timeout=0.2)
        except Exception:
            continue
        try:
            dets = detector.detect(views[cid]) if detector is not None else []
            resp_qs[cid].put(len(dets))
            served += 1
        except Exception:
            try:
                resp_qs[cid].put(0)
            except Exception:
                pass
    served_q.put(served)


def discover(n, timeout_s=20.0):
    from cyndilib.finder import Finder

    f = Finder()
    f.open()
    want = {f"AutoPTZ Bench Cam {i + 1}" for i in range(n)}
    resolved = {}
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
    return [resolved[f"AutoPTZ Bench Cam {i + 1}"] for i in range(n) if f"AutoPTZ Bench Cam {i + 1}" in resolved]


def main():
    import json

    import psutil

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
    FPS = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    DUR = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0

    names = discover(N)
    if len(names) < N:
        print(f"DISCOVER_INCOMPLETE {len(names)}/{N}", flush=True)
        N = len(names)
    if N == 0:
        print('RESULT_JSON {"error":"no sources"}', flush=True)
        return

    ctx = mp.get_context("spawn")
    cam_ids = [f"cam{i}" for i in range(N)]
    frame_bytes = W * H * 3
    shms = {cid: SharedMemory(create=True, size=frame_bytes) for cid in cam_ids}
    shm_names = {cid: s.name for cid, s in shms.items()}
    req_q = ctx.Queue()
    resp_qs = {cid: ctx.Queue() for cid in cam_ids}
    result_q = ctx.Queue()
    served_q = ctx.Queue()
    ready_ev = ctx.Event()
    stop_ev = ctx.Event()

    server = ctx.Process(target=_server_proc, args=(shm_names, req_q, resp_qs, W, H, ready_ev, stop_ev, served_q), daemon=True)
    server.start()
    ready_ev.wait(timeout=60)  # wait for the model to load

    cams = []
    for cid, ndi in zip(cam_ids, names, strict=True):
        p = ctx.Process(target=_camera_proc, args=(cid, ndi, shm_names[cid], req_q, resp_qs[cid], W, H, FPS, DUR, result_q), daemon=True)
        p.start()
        cams.append(p)

    # Sample CPU/RAM over the whole tree while it runs. Prime + measure the SAME
    # Process objects (cpu_percent state lives on the object; a fresh child object
    # reads 0 — the classic priming bug).
    ncpu = psutil.cpu_count(logical=True) or 1
    parent = psutil.Process()
    time.sleep(1.0)  # let camera procs settle
    tree = [parent]
    for c in parent.children(recursive=True):
        tree.append(c)
    for p in tree:
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    time.sleep(DUR)
    cpu = 0.0
    rss = 0
    for p in tree:
        try:
            cpu += p.cpu_percent(None)
            rss += p.memory_info().rss
        except Exception:
            pass

    stop_ev.set()
    results = []
    for _ in cams:
        try:
            results.append(result_q.get(timeout=5))
        except Exception:
            pass
    try:
        served = served_q.get(timeout=5)
    except Exception:
        served = -1
    for p in cams:
        p.join(timeout=3)
    server.join(timeout=3)
    for s in shms.values():
        try:
            s.close()
            s.unlink()
        except Exception:
            pass

    cap_fps = sorted(r[1] for r in results)
    det_fps = [r[2] for r in results]
    out = {
        "N": N,
        "arch": "mp-model-server",
        "cams_ok": sum(1 for r in results if r[3] == "ok"),
        "capture_fps_median": round(cap_fps[len(cap_fps) // 2], 1) if cap_fps else 0.0,
        "capture_fps_min": round(min(cap_fps), 1) if cap_fps else 0.0,
        "detect_per_s_per_cam_median": round(sorted(det_fps)[len(det_fps) // 2], 2) if det_fps else 0.0,
        "server_detections_per_s": round(served / DUR, 1) if served >= 0 else -1,
        "total_cpu_pct_of_machine": round(cpu / ncpu, 1),
        "total_rss_gb": round(rss / (1 << 30), 2),
    }
    print("RESULT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
