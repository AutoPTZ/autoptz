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


class TestPositionDeadZone:
    """B3 — a stationary subject with bbox noise must produce a still frame."""

    def test_small_jitter_holds_center(self):
        # Dead-zone on; tight position smoothing so any drift WOULD show without it.
        f = DigitalFramer(out_aspect=ASPECT, smooth=1.0, deadzone=0.04)
        bbox = (860, 400, 1060, 670)
        first = f.frame_for(bbox, 1920, 1080)
        # Tiny per-frame jitter (a few px) around the same centre — well within the
        # dead-zone band of the crop width.
        jitter = [(-3, 2), (4, -1), (-2, -3), (3, 1), (-1, 2), (2, -2)]
        last = first
        for dx, dy in jitter:
            jbox = (860 + dx, 400 + dy, 1060 + dx, 670 + dy)
            last = f.frame_for(jbox, 1920, 1080)
        # Crop centre held put despite the jitter.
        assert last[0] == first[0]
        assert last[1] == first[1]

    def test_sustained_move_follows(self):
        # A real, sustained move past the band must be followed.
        f = DigitalFramer(out_aspect=ASPECT, smooth=1.0, deadzone=0.04)
        start = f.frame_for((860, 400, 1060, 670), 1920, 1080)
        moved = start
        for _ in range(20):
            moved = f.frame_for((1300, 400, 1500, 670), 1920, 1080)
        assert moved[0] > start[0] + 50  # crop clearly followed the subject

    def test_deadzone_zero_reproduces_prior_behavior(self):
        # deadzone=0 must reproduce the old per-frame tracking exactly.
        old = DigitalFramer(out_aspect=ASPECT, smooth=1.0, deadzone=0.0, size_smooth=1.0)
        old.frame_for((860, 400, 1060, 670), 1920, 1080)
        # A 5px nudge moves the crop with deadzone=0 (no hold band).
        nudged = old.frame_for((870, 400, 1070, 670), 1920, 1080)
        ref = desired_crop(
            (870, 400, 1070, 670),
            1920,
            1080,
            out_aspect=ASPECT,
            fill=old.fill,
            min_frac=old.min_frac,
            max_frac=old.max_frac,
            headroom=old.headroom,
        )
        assert nudged[0] == int(round(ref[0]))


class TestSizeSmoothing:
    """B4 — split size smoothing from position smoothing."""

    def test_size_deadband_holds_w_h(self):
        # Position locked (deadzone), size flicker within the size dead-band → w/h
        # stays put.
        f = DigitalFramer(out_aspect=ASPECT, smooth=1.0, size_smooth=1.0, size_deadband=0.05)
        first = f.frame_for((860, 400, 1060, 670), 1920, 1080)
        # +2% subject height flicker — under the 5% size dead-band.
        flicker = f.frame_for((860, 400, 1060, 675), 1920, 1080)
        assert flicker[2] == first[2]
        assert flicker[3] == first[3]

    def test_sustained_size_change_eases_slowly(self):
        # A big sustained size change is followed, but at the slower size constant.
        slow = DigitalFramer(out_aspect=ASPECT, smooth=1.0, size_smooth=0.08, size_deadband=0.02)
        fast = DigitalFramer(out_aspect=ASPECT, smooth=1.0, size_smooth=0.5, size_deadband=0.02)
        slow.frame_for((860, 400, 1060, 670), 1920, 1080)
        fast.frame_for((860, 400, 1060, 670), 1920, 1080)
        # Subject grows a lot (steps closer).
        big = (760, 300, 1160, 900)
        s = slow.frame_for(big, 1920, 1080)
        fa = fast.frame_for(big, 1920, 1080)
        # Both grow toward the new size, but the slower constant lags behind.
        assert s[3] > 0 and fa[3] > s[3]

    def test_crop_stays_on_aspect_when_size_deltas_straddle_deadband(self):
        # B4 regression guard (point case). When w and h were eased through
        # INDEPENDENT size dead-bands, one axis could freeze while the other eased,
        # because cw != ch makes the same target produce different per-axis deltas
        # relative to the dead-band. Crop 800x450 (16:9) → target 818x470: w changes
        # 2.25% (under a 3% dead-band → frozen) but h changes 4.4% (over → eases),
        # so the OLD code returned 800x470 (aspect 1.70) and the fixed-size output
        # resize stretched the subject. The fix eases height then DERIVES width from
        # out_aspect, so the crop stays on aspect.
        f = DigitalFramer(
            out_aspect=ASPECT, smooth=1.0, size_smooth=1.0, deadzone=0.04, size_deadband=0.03
        )
        f._crop = (0.0, 0.0, 800.0, 450.0)
        _, _, w, h = f._step((0.0, 0.0, 818.0, 470.0))
        assert abs(w / h - ASPECT) < 0.01, f"crop went off-aspect: {w}x{h} = {w / h:.4f}"

    def test_crop_stays_on_aspect_through_slow_zoom(self):
        # B4 regression guard (sequence case): drive a slow zoom (subject grows over
        # many frames, small per-frame size steps near the dead-band) and assert the
        # crop aspect stays locked on out_aspect every frame.
        f = DigitalFramer(
            out_aspect=ASPECT,
            smooth=1.0,
            size_smooth=0.08,  # slow zoom → small per-frame size steps near the band
            deadzone=0.04,
            size_deadband=0.03,
        )
        for i in range(30):
            half_h = 135 + i * 2  # subject slowly steps closer
            half_w = 100
            bbox = (960 - half_w, 540 - half_h, 960 + half_w, 540 + half_h)
            _, _, w, h = f.frame_for(bbox, 1920, 1080)
            assert abs(w / h - ASPECT) < 0.01, f"frame {i}: aspect {w / h:.4f} drifted"

    def test_size_smooth_equal_reproduces_prior_behavior(self):
        # size_smooth == smooth and no dead-bands → old uniform EMA on all 4 params.
        f = DigitalFramer(
            out_aspect=ASPECT, smooth=0.18, size_smooth=0.18, deadzone=0.0, size_deadband=0.0
        )
        f._crop = (100.0, 100.0, 800.0, 450.0)
        ref = DigitalFramer(out_aspect=ASPECT, smooth=0.18)
        ref._crop = (100.0, 100.0, 800.0, 450.0)
        tgt = (200.0, 150.0, 900.0, 506.25)
        got = f._step(tgt)
        # Replicate the old uniform-EMA step for the reference.
        a = 0.18
        exp = tuple(c + a * (t - c) for c, t in zip((100.0, 100.0, 800.0, 450.0), tgt, strict=True))
        assert got == (
            int(round(exp[0])),
            int(round(exp[1])),
            max(1, int(round(exp[2]))),
            max(1, int(round(exp[3]))),
        )
