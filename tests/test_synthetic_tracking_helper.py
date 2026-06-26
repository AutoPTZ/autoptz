"""Self-test for the shared synthetic-tracking helper module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autoptz.engine.pipeline.detect import BBox, Detection
from tests.synthetic_tracking import (
    FRAME,
    MockBackend,
    box_at,
    constant_velocity_centres,
    detections_for_centres,
    make_cfg,
    make_mock_impl,
    sinusoid_centres,
    tracker_rows_for_centres,
)


def test_sinusoid_matches_formula() -> None:
    centres = sinusoid_centres(cx=320.0, amp=100.0, omega=0.2, frames=5, dt=1.0)
    assert len(centres) == 5
    for i, (x, y) in enumerate(centres):
        assert x == pytest.approx(320.0 + 100.0 * math.sin(0.2 * i))
        assert y == pytest.approx(240.0)


def test_constant_velocity_is_linear() -> None:
    centres = constant_velocity_centres(x0=100.0, vx=5.0, frames=4)
    xs = [x for x, _ in centres]
    assert xs == pytest.approx([100.0, 105.0, 110.0, 115.0])


def test_box_at_is_centred() -> None:
    b = box_at(320.0, 240.0, w=80.0, h=200.0)
    assert isinstance(b, BBox)
    assert b.cx == pytest.approx(320.0)
    assert b.cy == pytest.approx(240.0)
    assert b.w == pytest.approx(80.0)


def test_detections_one_per_frame_with_miss() -> None:
    centres = constant_velocity_centres(x0=100.0, vx=10.0, frames=3)
    dets = detections_for_centres(centres, misses={1})
    assert len(dets) == 3
    assert len(dets[0]) == 1 and isinstance(dets[0][0], Detection)
    assert dets[1] == []  # injected miss
    assert dets[2][0].bbox.cx == pytest.approx(120.0)


def test_tracker_rows_shape_and_miss() -> None:
    centres = constant_velocity_centres(x0=100.0, vx=10.0, frames=3)
    rows = tracker_rows_for_centres(centres, track_id=7, misses={1})
    assert rows[0].shape == (1, 7)
    assert int(rows[0][0, 4]) == 7
    assert rows[1].shape[0] == 0  # miss → no rows


def test_make_mock_impl_returns_rows_in_order() -> None:
    rows = [np.zeros((1, 7), np.float32), np.empty((0, 7), np.float32)]
    impl = make_mock_impl(rows)
    assert impl.update(None, None).shape == (1, 7)
    assert impl.update(None, None).shape[0] == 0


def test_make_cfg_defaults_are_deterministic() -> None:
    cfg = make_cfg()
    assert cfg.max_accel == 0.0
    assert cfg.deadzone_x == 0.0
    assert cfg.auto_zoom is False


def test_mock_backend_records_calls() -> None:
    b = MockBackend()
    b.move_velocity(0.1, 0.2, 0.0)
    assert b.velocity_calls == [(0.1, 0.2, 0.0)]


def test_frame_shape() -> None:
    assert FRAME.shape == (480, 640, 3)
