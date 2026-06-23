"""Unit tests for autoptz.engine.runtime.shm."""

from __future__ import annotations

import time

import numpy as np
import pytest

from autoptz.engine.runtime.shm import FrameHeader, ShmReader, ShmWriter


def _unique_name(suffix: str = "") -> str:
    return f"autoptz_test_{id(object())}{suffix}"


class TestShmWriterReader:
    def test_basic_round_trip(self) -> None:
        H, W, C = 120, 160, 3
        name = _unique_name()
        frame = np.full((H, W, C), 77, dtype=np.uint8)

        with ShmWriter(name, H, W, C) as w:
            seq = w.push(frame)
            assert seq == 0

            with ShmReader(name, H, W, C) as r:
                result = r.latest()

        assert result is not None
        hdr, got = result
        assert isinstance(hdr, FrameHeader)
        assert hdr.seq == 0
        assert hdr.height == H
        assert hdr.width == W
        np.testing.assert_array_equal(got, frame)

    def test_latest_wins(self) -> None:
        """Reader should always get the most recently pushed frame."""
        H, W, C = 60, 80, 3
        name = _unique_name()

        with ShmWriter(name, H, W, C) as w:
            for val in range(10):
                frame = np.full((H, W, C), val, dtype=np.uint8)
                w.push(frame)

            with ShmReader(name, H, W, C) as r:
                result = r.latest()

        assert result is not None
        _, got = result
        # Reader sees last written value (9)
        assert int(got[0, 0, 0]) == 9

    def test_no_new_frame_returns_none(self) -> None:
        H, W, C = 60, 80, 3
        name = _unique_name()
        frame = np.zeros((H, W, C), dtype=np.uint8)

        with ShmWriter(name, H, W, C) as w:
            w.push(frame)
            with ShmReader(name, H, W, C) as r:
                r.latest()  # consume seq 0
                result = r.latest()  # same seq → None

        assert result is None

    def test_empty_reader_returns_none(self) -> None:
        H, W, C = 60, 80, 3
        name = _unique_name()

        with ShmWriter(name, H, W, C), ShmReader(name, H, W, C) as r:
            result = r.latest()  # nothing pushed yet

        assert result is None

    def test_peek_seq_and_has_new_do_not_consume(self) -> None:
        # The repaint-throttle relies on peek_seq/has_new reporting a new frame
        # WITHOUT consuming it (so latest() still returns the frame afterwards).
        H, W, C = 60, 80, 3
        name = _unique_name()
        frame = np.zeros((H, W, C), dtype=np.uint8)

        with ShmWriter(name, H, W, C) as w, ShmReader(name, H, W, C) as r:
            assert r.peek_seq() == -1  # nothing pushed yet
            assert r.has_new() is False
            w.push(frame)
            # A new frame is visible to peek, repeatedly, without consuming it.
            assert r.peek_seq() == 0
            assert r.has_new() is True
            assert r.has_new() is True
            assert r.latest() is not None  # peeking didn't consume it
            assert r.has_new() is False  # now consumed
            assert r.peek_seq() == 0  # seq still readable, just not "new"

    def test_sequence_numbers_increment(self) -> None:
        H, W, C = 60, 80, 3
        name = _unique_name()
        frame = np.zeros((H, W, C), dtype=np.uint8)

        with ShmWriter(name, H, W, C) as w:
            seqs = [w.push(frame) for _ in range(5)]

        assert seqs == list(range(5))

    def test_frame_size_mismatch_raises(self) -> None:
        H, W, C = 60, 80, 3
        name = _unique_name()
        bad_frame = np.zeros((H + 1, W, C), dtype=np.uint8)

        with ShmWriter(name, H, W, C) as w:
            with pytest.raises(ValueError, match="Frame size mismatch"):
                w.push(bad_frame)

    def test_timestamp_auto_filled(self) -> None:
        H, W, C = 60, 80, 3
        name = _unique_name()
        frame = np.zeros((H, W, C), dtype=np.uint8)
        before = time.monotonic_ns()

        with ShmWriter(name, H, W, C) as w:
            w.push(frame)
            with ShmReader(name, H, W, C) as r:
                result = r.latest()

        after = time.monotonic_ns()
        assert result is not None
        hdr, _ = result
        assert before <= hdr.ts_ns <= after

    def test_explicit_timestamp(self) -> None:
        H, W, C = 60, 80, 3
        name = _unique_name()
        frame = np.zeros((H, W, C), dtype=np.uint8)
        ts = 123_456_789

        with ShmWriter(name, H, W, C) as w:
            w.push(frame, ts_ns=ts)
            with ShmReader(name, H, W, C) as r:
                result = r.latest()

        assert result is not None
        hdr, _ = result
        assert hdr.ts_ns == ts

    def test_triple_buffer_wraps(self) -> None:
        """Push more frames than slots; reader should still get the latest."""
        H, W, C = 60, 80, 3
        name = _unique_name()

        with ShmWriter(name, H, W, C) as w:
            for val in range(9):  # 3× the number of slots
                frame = np.full((H, W, C), val, dtype=np.uint8)
                w.push(frame)

            with ShmReader(name, H, W, C) as r:
                result = r.latest()

        assert result is not None
        _, got = result
        assert int(got[0, 0, 0]) == 8
