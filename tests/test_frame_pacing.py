"""Deadline-accumulator pacing for ``_AdapterFrameSource``.

Regression guard for the double-pacing bug where the worker slept a full
``1/target`` period *and* the blocking read waited for the next hardware frame,
roughly halving the delivered rate (slider 30 → ~15, 15 → ~10).  These tests
drive the pacer with a fake clock so the timing is deterministic.
"""
from __future__ import annotations

import numpy as np

import autoptz.engine.camera_worker as cw


class _FakeClock:
    """Deterministic monotonic clock; ``sleep`` just advances it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def monotonic(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        if dt > 0:
            self.t += dt


class _InstantAdapter:
    """Non-blocking source (AVF-style newest-frame): read returns at once."""

    def __init__(self, clock: _FakeClock, target_fps: float, read_cost: float = 0.003) -> None:
        self._target_fps = target_fps
        self._clock = clock
        self._read_cost = read_cost
        self._frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def _open(self) -> bool:
        return True

    def _read_frame(self) -> np.ndarray:
        self._clock.sleep(self._read_cost)  # per-read processing cost
        return self._frame


class _BlockingAdapter:
    """Hardware-paced source (OpenCV-style): read blocks until the next frame."""

    def __init__(self, clock: _FakeClock, target_fps: float, hw_period: float) -> None:
        self._target_fps = target_fps
        self._clock = clock
        self._hw_period = hw_period
        self._next_hw = clock.t
        self._frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def _open(self) -> bool:
        return True

    def _read_frame(self) -> np.ndarray:
        if self._clock.t < self._next_hw:
            self._clock.sleep(self._next_hw - self._clock.t)
        self._next_hw = self._clock.t + self._hw_period
        return self._frame


def _count_reads(monkeypatch, clock: _FakeClock, adapter: object, window: float = 1.0) -> int:
    monkeypatch.setattr(cw.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(cw.time, "sleep", clock.sleep)
    src = cw._AdapterFrameSource(adapter)
    assert src.open()
    start = clock.t
    reads = 0
    while clock.t - start < window and reads < 10_000:
        assert src.read() is not None
        reads += 1
    return reads


def test_non_blocking_source_paces_to_target(monkeypatch) -> None:
    clock = _FakeClock()
    reads = _count_reads(monkeypatch, clock, _InstantAdapter(clock, target_fps=10))
    # ~10 frames in 1 s (plus the free anchor frame), never ~5 (the old 2x bug).
    assert 9 <= reads <= 12


def test_blocking_source_is_not_double_paced(monkeypatch) -> None:
    clock = _FakeClock()
    # Request 30 fps but the hardware blocks at 15 fps. The old code stacked the
    # sleep on top of the block → ~10 fps; the deadline pacer must not sleep here.
    adapter = _BlockingAdapter(clock, target_fps=30, hw_period=1.0 / 15.0)
    reads = _count_reads(monkeypatch, clock, adapter)
    assert 13 <= reads <= 17


def test_zero_target_does_not_sleep(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(cw.time, "monotonic", clock.monotonic)
    sleeps: list[float] = []
    monkeypatch.setattr(cw.time, "sleep", lambda dt: sleeps.append(dt))
    adapter = _InstantAdapter(clock, target_fps=0.0, read_cost=0.0)
    src = cw._AdapterFrameSource(adapter)
    src.open()
    for _ in range(5):
        src.read()
    assert sleeps == []
    assert src._next_deadline == 0.0
