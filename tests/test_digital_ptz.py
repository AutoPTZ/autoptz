from __future__ import annotations

from autoptz.engine.ptz.digital import DigitalPTZBackend


def test_home_is_full_frame_centered():
    b = DigitalPTZBackend()
    x, y, w, h = b.crop_rect(1920, 1080)
    assert (x, y, w, h) == (0, 0, 1920, 1080)


def test_zoom_in_shrinks_and_recenters():
    b = DigitalPTZBackend(min_crop_frac=0.5, max_step_per_s=10.0)
    b.move_velocity(0.0, 0.0, 1.0)  # zoom tele
    b.move_velocity(0.0, 0.0, 1.0)
    x, y, w, h = b.crop_rect(1000, 1000)
    assert w < 1000 and h < 1000
    assert abs((x + w / 2) - 500) <= 1  # still centered horizontally
    assert w >= 500  # never below min_crop_frac


def test_pan_right_shifts_crop_right():
    b = DigitalPTZBackend(min_crop_frac=0.5, max_step_per_s=10.0)
    b.move_velocity(0.0, 0.0, 1.0)  # zoom in so there's room to pan
    b.move_velocity(1.0, 0.0, 0.0)  # pan right
    x, _, w, _ = b.crop_rect(1000, 1000)
    assert x + w <= 1000 and x > 0  # shifted right, still inside the frame


def test_crop_never_leaves_frame():
    b = DigitalPTZBackend(min_crop_frac=0.4, max_step_per_s=100.0)
    for _ in range(50):
        b.move_velocity(1.0, 1.0, 1.0)
    x, y, w, h = b.crop_rect(1280, 720)
    assert 0 <= x and 0 <= y and x + w <= 1280 and y + h <= 720
