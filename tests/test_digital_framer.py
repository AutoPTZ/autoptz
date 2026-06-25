"""Center Stage auto-framer tests — the crop is computed directly from the target
bbox (not the velocity controller, which never zoomed)."""

from __future__ import annotations

from autoptz.engine.pipeline.digital_framer import (
    DigitalFramer,
    desired_crop,
    union_bbox,
)

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


class TestUnionBbox:
    """D1 — multi-person group framing: the union of several person boxes."""

    def test_union_of_one_box_is_itself(self):
        assert union_bbox([(10, 20, 30, 40)]) == (10.0, 20.0, 30.0, 40.0)

    def test_union_covers_all_boxes(self):
        boxes = [(100, 200, 300, 500), (700, 150, 900, 600), (400, 300, 500, 450)]
        u = union_bbox(boxes)
        # The union's corners are the extremes across all boxes.
        assert u == (100.0, 150.0, 900.0, 600.0)
        # Every input box is contained inside the union.
        for x1, y1, x2, y2 in boxes:
            assert u[0] <= x1 and u[1] <= y1
            assert u[2] >= x2 and u[3] >= y2

    def test_union_wider_than_any_single_box(self):
        boxes = [(300, 400, 500, 670), (1400, 400, 1600, 670)]
        u = union_bbox(boxes)
        u_w = u[2] - u[0]
        for x1, _, x2, _ in boxes:
            assert u_w > (x2 - x1)

    def test_empty_returns_none(self):
        assert union_bbox([]) is None

    def test_group_crop_frames_all_people(self):
        # Two people on opposite sides of the frame: the group crop (framing the
        # union) must be wider than either single-person crop, and cover both.
        left = (300, 400, 500, 670)
        right = (1400, 400, 1600, 670)
        u = union_bbox([left, right])
        assert u is not None
        group = desired_crop(
            u, 1920, 1080, out_aspect=ASPECT, fill=0.62, min_frac=0.34, max_frac=0.94
        )
        single = desired_crop(
            left, 1920, 1080, out_aspect=ASPECT, fill=0.62, min_frac=0.34, max_frac=0.94
        )
        assert group[2] > single[2]  # group crop is wider
        # Both subjects fall inside the group crop horizontally.
        gx, gw = group[0], group[2]
        assert gx <= left[0] and (gx + gw) >= right[2]


class TestHeadroomByPreset:
    """D2 — shot-size-aware headroom: the applied headroom comes from the preset."""

    def test_more_headroom_lifts_center_higher(self):
        bbox = (860, 400, 1060, 670)  # centre y = 535
        _, y_lo, _, h_lo = desired_crop(
            bbox,
            1920,
            1080,
            out_aspect=ASPECT,
            fill=0.62,
            min_frac=0.34,
            max_frac=0.78,
            headroom=0.0,
        )
        _, y_hi, _, h_hi = desired_crop(
            bbox,
            1920,
            1080,
            out_aspect=ASPECT,
            fill=0.62,
            min_frac=0.34,
            max_frac=0.78,
            headroom=0.2,
        )
        # More headroom raises the subject (smaller y, crop pulled up).
        assert y_hi < y_lo

    def test_framer_headroom_field_is_applied(self):
        # Two framers identical but for headroom must produce different crop y.
        low = DigitalFramer(out_aspect=ASPECT, smooth=1.0, headroom=0.0)
        high = DigitalFramer(out_aspect=ASPECT, smooth=1.0, headroom=0.25)
        bbox = (860, 400, 1060, 670)
        ylow = low.frame_for(bbox, 1920, 1080)[1]
        yhigh = high.frame_for(bbox, 1920, 1080)[1]
        assert yhigh < ylow


class TestLeadRoom:
    """D2 — subtle motion lead-room: bias the crop centre in the direction of
    motion so a walking subject sits back-of-centre."""

    def test_lead_zero_centers_as_before(self):
        # lead=0 must reproduce the prior centred behaviour for a moving subject.
        f = DigitalFramer(out_aspect=ASPECT, smooth=1.0, deadzone=0.0, size_smooth=1.0, lead=0.0)
        f.frame_for((300, 400, 500, 670), 1920, 1080)
        x, _, w, _ = f.frame_for((600, 400, 800, 670), 1920, 1080)
        ref = desired_crop(
            (600, 400, 800, 670),
            1920,
            1080,
            out_aspect=ASPECT,
            fill=f.fill,
            min_frac=f.min_frac,
            max_frac=f.max_frac,
            headroom=f.headroom,
        )
        assert x == int(round(ref[0]))

    def test_subject_moving_right_shifts_crop_right(self):
        # A subject moving steadily right: with lead-room the crop centre sits to
        # the RIGHT of the bbox centre (nose room ahead of the motion).
        lead = DigitalFramer(out_aspect=ASPECT, smooth=1.0, deadzone=0.0, size_smooth=1.0, lead=0.5)
        no_lead = DigitalFramer(
            out_aspect=ASPECT, smooth=1.0, deadzone=0.0, size_smooth=1.0, lead=0.0
        )
        # Walk the subject right across several frames so velocity builds up.
        bx = 300
        cx_lead = cx_no = 0.0
        for _ in range(12):
            box = (bx, 400, bx + 200, 670)
            cl = lead.frame_for(box, 1920, 1080)
            cn = no_lead.frame_for(box, 1920, 1080)
            cx_lead = cl[0] + cl[2] / 2
            cx_no = cn[0] + cn[2] / 2
            bx += 60
        # Lead-room pushes the crop centre ahead (to the right) of the no-lead one.
        assert cx_lead > cx_no

    def test_lead_default_is_subtle(self):
        # The default lead must be conservative (small): a moving subject is not
        # pushed more than a small fraction of the crop width off-centre.
        f = DigitalFramer(out_aspect=ASPECT, smooth=1.0, deadzone=0.0, size_smooth=1.0)
        bx = 300
        offset = 0.0
        for _ in range(12):
            box = (bx, 400, bx + 200, 670)
            x, _, w, _ = f.frame_for(box, 1920, 1080)
            bbox_cx = bx + 100
            crop_cx = x + w / 2
            offset = abs(crop_cx - bbox_cx)
            bx += 60
        assert offset <= 0.15 * w  # small bias, can't destabilise framing
