"""Center Stage auto-framer tests — the crop is computed directly from the target
bbox (not the velocity controller, which never zoomed)."""

from __future__ import annotations

from autoptz.engine.pipeline.digital_framer import DigitalFramer, desired_crop

ASPECT = 16.0 / 9.0


class TestDesiredCrop:
    def test_centers_and_zooms_on_a_small_subject(self):
        # A person occupying ~25% of a 1920x1080 frame, centred.
        bbox = (860, 400, 1060, 670)  # 200x270, centre (960, 535)
        x, y, w, h = desired_crop(
            bbox, 1920, 1080, out_aspect=ASPECT, fill=0.62, min_frac=0.34, max_frac=0.78
        )
        assert w < 1920 and h < 1080  # zoomed in
        assert abs(w / h - ASPECT) < 0.02  # output aspect preserved
        # crop horizontally centred on the subject
        assert abs((x + w / 2) - 960) <= 2

    def test_clamped_inside_frame_for_edge_subject(self):
        bbox = (20, 700, 180, 1050)  # bottom-left subject
        x, y, w, h = desired_crop(
            bbox, 1920, 1080, out_aspect=ASPECT, fill=0.62, min_frac=0.34, max_frac=0.78
        )
        assert x >= 0 and y >= 0
        assert x + w <= 1920 + 1 and y + h <= 1080 + 1

    def test_close_subject_is_still_cropped(self):
        # A subject filling most of the frame (close to the camera) must STILL be
        # cropped to a window — the max_frac cap is what makes Center Stage visibly
        # zoom/pan instead of falling back to the whole frame.
        bbox = (200, 50, 1700, 1040)  # fills most of the frame
        _, _, w, h = desired_crop(
            bbox, 1920, 1080, out_aspect=ASPECT, fill=0.70, min_frac=0.34, max_frac=0.78
        )
        assert h <= 0.78 * 1080 + 1  # capped to the window
        assert h < 1080 and w < 1920  # still a real crop (the fix)

    def test_min_frac_floor(self):
        # A tiny far-away box must not zoom below min_frac of the frame.
        bbox = (950, 520, 970, 560)  # 20x40
        _, _, w, h = desired_crop(
            bbox, 1920, 1080, out_aspect=ASPECT, fill=0.62, min_frac=0.34, max_frac=0.78
        )
        assert h >= 0.34 * 1080 - 1


class TestDigitalFramer:
    def test_smoothing_converges_toward_target(self):
        f = DigitalFramer(out_aspect=ASPECT, smooth=0.2)
        bbox = (860, 400, 1060, 670)
        first = f.frame_for(bbox, 1920, 1080)
        # First call snaps to the desired crop (no prior state).
        assert first[2] < 1920
        # Many calls on a steady bbox stay put (converged).
        last = first
        for _ in range(40):
            last = f.frame_for(bbox, 1920, 1080)
        assert abs(last[2] - first[2]) <= 2

    def test_full_frame_eases_back_to_whole_frame(self):
        f = DigitalFramer(out_aspect=ASPECT, smooth=1.0)  # no smoothing → immediate
        x, y, w, h = f.full_frame(1920, 1080)
        assert (x, y, w, h) == (0, 0, 1920, 1080)

    def test_tracks_a_moving_subject(self):
        f = DigitalFramer(out_aspect=ASPECT, smooth=1.0)
        left = f.frame_for((300, 400, 500, 670), 1920, 1080)
        right = f.frame_for((1400, 400, 1600, 670), 1920, 1080)
        assert right[0] > left[0]  # crop followed the subject to the right
