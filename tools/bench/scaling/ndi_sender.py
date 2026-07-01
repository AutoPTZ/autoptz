"""NDI sender fleet for the AutoPTZ streaming benchmark.

Broadcasts N independent NDI sources at a target resolution/fps using a
pre-built frame per sender (cheap, so the sender process can actually sustain
the rate and not confound the receiver's drop measurement). Each sender runs in
its own thread, paced to the target fps. Logs the achieved per-sender fps so we
can confirm the SOURCE truly delivered ~target before trusting receiver drops.

Usage: python bench_sender.py <N> <width> <height> <fps> <seconds>
Prints "SENDER_READY <names...>" once all are broadcasting.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
from cyndilib.sender import Sender
from cyndilib.video_frame import VideoSendFrame
from cyndilib.wrapper import FourCC

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
H = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
FPS = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
SECONDS = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0

# A structured static frame (not flat black) — a moving gradient per sender so the
# stream has real bytes to push, without per-frame generation cost.
_frames = []
for i in range(N):
    rgba = np.empty((H, W, 4), dtype=np.uint8)
    yy = np.linspace(0, 255, H).astype(np.int32)[:, None]
    xx = np.linspace(0, 255, W).astype(np.int32)[None, :]
    rgba[..., 0] = ((xx + i * 16) % 256).astype(np.uint8)
    rgba[..., 1] = yy.astype(np.uint8)
    rgba[..., 2] = (((xx + yy) // 2 + i * 8) % 256).astype(np.uint8)
    rgba[..., 3] = 255
    _frames.append(np.ascontiguousarray(rgba).ravel())

_senders = []
_achieved = [0.0] * N
_stop = threading.Event()


def _run(i: int) -> None:
    s = Sender(ndi_name=f"AutoPTZ Bench Cam {i + 1}", clock_video=False)
    vf = VideoSendFrame()
    vf.set_resolution(W, H)
    vf.set_fourcc(FourCC.RGBA)
    vf.set_frame_rate(int(round(FPS)))
    s.set_video_frame(vf)
    s.open()
    _senders.append(s)
    period = 1.0 / FPS
    flat = _frames[i]
    n = 0
    t_start = time.monotonic()
    next_t = t_start
    while not _stop.is_set():
        s.write_video(flat)
        n += 1
        next_t += period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        elif sleep < -period:  # fell badly behind → resync (don't spiral)
            next_t = time.monotonic()
    _achieved[i] = n / max(1e-6, time.monotonic() - t_start)
    s.close()


threads = [threading.Thread(target=_run, args=(i,), daemon=True) for i in range(N)]
for t in threads:
    t.start()
time.sleep(1.0)  # let senders open
print(f"SENDER_READY {N} senders @ {W}x{H} {FPS}fps", flush=True)

try:
    time.sleep(SECONDS)
except KeyboardInterrupt:
    pass
finally:
    _stop.set()
    for t in threads:
        t.join(timeout=2.0)
    print("SENDER_ACHIEVED_FPS " + " ".join(f"{x:.1f}" for x in _achieved), flush=True)
