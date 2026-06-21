"""Latest-wins triple-buffered shared-memory frame ring buffer.

Writer fills one slot while the reader consumes the latest committed slot.
The "committed slot index" lives in a separate tiny shared-memory region
(one int64) so readers and writers coordinate without locks on the hot path.

Torn-read protection: the reader checks the frame sequence number before
and after copying; if it changed mid-copy it returns None and retries next
tick.  This is safe because there is always exactly one writer.

Usage::

    # Writer process
    with ShmWriter("cam_0_preview", 360, 640) as w:
        while running:
            w.push(frame)

    # Reader process (separate OS process)
    with ShmReader("cam_0_preview", 360, 640) as r:
        result = r.latest()     # None if no new frame
        if result:
            header, frame = result
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray

# Header layout: seq(u64) | ts_ns(u64) | height(u32) | width(u32)
_HDR_FMT = "=QQII"
_HDR_SIZE = struct.calcsize(_HDR_FMT)  # 24 bytes
_N_SLOTS = 3
_IDX_FMT = "=q"  # signed int64 (-1 = no frame yet)
_IDX_SIZE = struct.calcsize(_IDX_FMT)


def frame_region_size(height: int, width: int, channels: int = 3) -> int:
    """Return the byte size of the frame ring for this shape."""
    frame_bytes = height * width * channels
    return (_HDR_SIZE + frame_bytes) * _N_SLOTS


def _unlink_one(name: str) -> None:
    """Close and unlink one shared-memory segment if it exists."""
    try:
        shm = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    except Exception:
        return
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def unlink_shared_memory_pair(name: str) -> None:
    """Best-effort cleanup for a frame ring and its commit-index segment."""
    for segment in (name, f"{name}__idx"):
        _unlink_one(segment)


@dataclass(frozen=True)
class FrameHeader:
    seq: int
    ts_ns: int
    height: int
    width: int


class ShmWriter:
    """Owns the shared-memory region; call push() from the capture/decode thread."""

    def __init__(self, name: str, height: int, width: int, channels: int = 3) -> None:
        self.name = name
        self.height = height
        self.width = width
        self.channels = channels
        self._closed = False

        self._frame_bytes = height * width * channels
        self._slot_size = _HDR_SIZE + self._frame_bytes
        total = frame_region_size(height, width, channels)

        created: SharedMemory | None = None
        try:
            created = SharedMemory(name=name, create=True, size=total)
            self._idx_shm = SharedMemory(name=f"{name}__idx", create=True, size=_IDX_SIZE)
            self._shm = created
        except FileExistsError:
            if created is not None:
                try:
                    created.close()
                    created.unlink()
                except Exception:
                    pass
            unlink_shared_memory_pair(name)
            self._shm = SharedMemory(name=name, create=True, size=total)
            self._idx_shm = SharedMemory(name=f"{name}__idx", create=True, size=_IDX_SIZE)
        except Exception:
            if created is not None:
                try:
                    created.close()
                    created.unlink()
                except Exception:
                    pass
            raise

        # Numpy view over the data region for zero-copy writes
        self._buf = np.frombuffer(self._shm.buf, dtype=np.uint8)

        # Initialise commit index to -1 (no frame yet)
        struct.pack_into(_IDX_FMT, self._idx_shm.buf, 0, -1)

        self._write_slot = 0
        self._seq = 0

    def push(self, frame: NDArray[np.uint8], ts_ns: int | None = None) -> int:
        """Write *frame* into the ring and return the sequence number."""
        if frame.nbytes != self._frame_bytes:
            raise ValueError(
                f"Frame size mismatch: expected {self._frame_bytes} B, got {frame.nbytes} B"
            )
        if ts_ns is None:
            ts_ns = time.monotonic_ns()

        slot = self._write_slot
        offset = slot * self._slot_size

        # 1. Write header
        header = struct.pack(_HDR_FMT, self._seq, ts_ns, self.height, self.width)
        self._buf[offset : offset + _HDR_SIZE] = np.frombuffer(header, dtype=np.uint8)

        # 2. Write frame pixels
        frame_start = offset + _HDR_SIZE
        self._buf[frame_start : frame_start + self._frame_bytes] = frame.ravel()

        # 3. Atomically commit this slot as the latest readable frame
        struct.pack_into(_IDX_FMT, self._idx_shm.buf, 0, slot)

        seq = self._seq
        self._seq += 1
        self._write_slot = (self._write_slot + 1) % _N_SLOTS
        return seq

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Release the numpy array's exported C-pointer before closing the mmap.
        # Python's mmap raises BufferError if any exported pointers still exist.
        # CPython drops the refcount to 0 immediately (no reference cycles here).
        if hasattr(self, "_buf"):
            del self._buf
        self._shm.close()
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass
        self._idx_shm.close()
        try:
            self._idx_shm.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> ShmWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ShmReader:
    """Attaches read-only to an existing shm region created by ShmWriter."""

    def __init__(self, name: str, height: int, width: int, channels: int = 3) -> None:
        self.name = name
        self.height = height
        self.width = width
        self.channels = channels
        self._closed = False

        self._frame_bytes = height * width * channels
        self._slot_size = _HDR_SIZE + self._frame_bytes

        self._shm = SharedMemory(name=name, create=False)
        self._idx_shm = SharedMemory(name=f"{name}__idx", create=False)

        self._buf = np.frombuffer(self._shm.buf, dtype=np.uint8)
        self._last_seq: int = -1

    def latest(self) -> tuple[FrameHeader, NDArray[np.uint8]] | None:
        """Return *(header, frame)* if a new frame is available, else *None*.

        Returns *None* on a torn read (writer mid-commit); the caller should
        retry on the next tick.
        """
        slot = struct.unpack_from(_IDX_FMT, self._idx_shm.buf, 0)[0]
        if slot < 0:
            return None

        offset = slot * self._slot_size

        # Read seq before the frame copy
        seq_pre = struct.unpack_from("=Q", self._shm.buf, offset)[0]

        if seq_pre == self._last_seq:
            return None

        # Copy the frame (prevents stale data if writer overwrites mid-read)
        frame_start = offset + _HDR_SIZE
        raw = self._buf[frame_start : frame_start + self._frame_bytes].copy()

        # Read seq after copy – if it changed, the writer stomped us; skip
        seq_post = struct.unpack_from("=Q", self._shm.buf, offset)[0]
        if seq_pre != seq_post:
            return None  # torn read; try next tick

        _, ts_ns, height, width = struct.unpack_from(_HDR_FMT, self._shm.buf, offset)
        self._last_seq = seq_pre

        return (
            FrameHeader(seq=seq_pre, ts_ns=ts_ns, height=height, width=width),
            raw.reshape((self.height, self.width, self.channels)),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "_buf"):
            del self._buf  # release exported mmap pointer before closing
        self._shm.close()
        self._idx_shm.close()

    def __enter__(self) -> ShmReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
