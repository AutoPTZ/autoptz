"""Shared framing-target selection (pure) — the single source of truth both
Center Stage and physical PTZ consume."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoptz.engine.framing_target import (
    FramingTarget,
    aim_error_for_box,
    confident_person_boxes,
    select_framing_target,
)


def _bbox(x1, y1, x2, y2):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)


def _track(track_id, box, *, lost=False):
    return SimpleNamespace(track_id=track_id, lost=lost, bbox=_bbox(*box))


def _select(tracks, *, tid=None, iid=None, trusted=None, group=False):
    return select_framing_target(
        tracks,
        target_track_id=tid,
        target_identity_id=iid,
        trusted_bbox=trusted,
        group_framing=group,
    )


class TestSharedCompositionTargets:
    """One preset table drives BOTH actuators, so Center Stage and physical PTZ
    compose the same shot for the same "Framing" setting."""

    def test_shared_table_values(self) -> None:
        from autoptz.engine.framing_target import SUBJECT_HEIGHT_TARGETS

        t = SUBJECT_HEIGHT_TARGETS
        # face/head_shoulders deliberately calmer than the old 0.80/0.60 —
        # user-reported as "intense" (over-zoomed, twitchy) on both actuators.
        assert t["face"] == pytest.approx(0.65)
        assert t["head_shoulders"] == pytest.approx(0.52)
        assert t["upper_body"] == pytest.approx(0.45)
        assert t["full_body"] == pytest.approx(0.30)
        assert t["face"] > t["head_shoulders"] > t["upper_body"] > t["full_body"]

    def test_ptz_zoom_targets_come_from_shared_table(self) -> None:
        from autoptz.engine.framing_target import SUBJECT_HEIGHT_TARGETS
        from autoptz.engine.ptz.controller import _ZOOM_FRAMING_TARGETS

        for key, value in SUBJECT_HEIGHT_TARGETS.items():
            assert _ZOOM_FRAMING_TARGETS[key] == pytest.approx(value), key

    def test_center_stage_fill_comes_from_shared_table(self) -> None:
        from autoptz.engine.camera_worker import _CENTERSTAGE_FRAMING
        from autoptz.engine.framing_target import SUBJECT_HEIGHT_TARGETS

        for key, value in SUBJECT_HEIGHT_TARGETS.items():
            assert _CENTERSTAGE_FRAMING[key][0] == pytest.approx(value), key


def test_explicit_track_lock_returns_that_box() -> None:
    tracks = [_track(1, (0, 0, 10, 20)), _track(2, (30, 0, 40, 20))]
    ft = _select(tracks, tid=2)
    assert ft == FramingTarget((30, 0, 40, 20), False, 2)


def test_explicit_lock_wins_over_group() -> None:
    """A locked person is followed even with group framing on and a crowd present."""
    tracks = [_track(1, (0, 0, 10, 20)), _track(2, (30, 0, 40, 20))]
    ft = _select(tracks, tid=1, group=True)
    assert ft.bbox == (0, 0, 10, 20)
    assert ft.is_group is False
    assert ft.primary_track_id == 1


def test_lock_holds_on_trusted_box_when_live_track_absent() -> None:
    ft = _select([_track(9, (0, 0, 1, 1))], tid=2, trusted=(5, 5, 15, 25), group=True)
    assert ft.bbox == (5, 5, 15, 25)
    assert ft.is_group is False


def test_lock_without_trusted_box_returns_none_not_group() -> None:
    ft = _select([_track(1, (0, 0, 10, 20)), _track(2, (30, 0, 40, 20))], tid=7, group=True)
    assert ft.bbox is None
    assert ft.primary_track_id == 7


def test_identity_lock_holds_on_trusted_box() -> None:
    ft = _select([], iid="alice", trusted=(2, 3, 4, 5), group=True)
    assert ft.bbox == (2, 3, 4, 5)
    assert ft.is_group is False


def test_no_lock_group_off_returns_none() -> None:
    ft = _select([_track(1, (0, 0, 10, 20)), _track(2, (30, 0, 40, 20))], group=False)
    assert ft.bbox is None


def test_group_single_confident_frames_that_person_not_group() -> None:
    ft = _select([_track(1, (10, 10, 20, 30))], group=True)
    assert ft.bbox == (10, 10, 20, 30)
    assert ft.is_group is False
    # The lone person's track id is exposed so the pose-stable ("Ignore arms")
    # framing applies to the group-single case exactly like an explicit lock.
    assert ft.primary_track_id == 1


def test_group_multiple_confident_frames_union() -> None:
    tracks = [_track(1, (0, 5, 10, 25)), _track(2, (30, 0, 40, 20))]
    ft = _select(tracks, group=True)
    assert ft.bbox == (0, 0, 40, 25)  # union
    assert ft.is_group is True
    assert ft.primary_track_id is None


def test_group_ignores_lost_people() -> None:
    tracks = [_track(1, (0, 0, 10, 20)), _track(2, (30, 0, 40, 20), lost=True)]
    ft = _select(tracks, group=True)
    assert ft.bbox == (0, 0, 10, 20)  # only the non-lost person
    assert ft.is_group is False


def test_confident_person_boxes_filters_lost_and_missing() -> None:
    tracks = [
        _track(1, (0, 0, 10, 20)),
        _track(2, (30, 0, 40, 20), lost=True),
        SimpleNamespace(track_id=3, lost=False, bbox=None),
    ]
    assert confident_person_boxes(tracks) == [(0, 0, 10, 20)]


def test_aim_error_centered_box_is_zero() -> None:
    (ex, ey), h = aim_error_for_box((45, 45, 55, 55), 100, 100, 0.5)
    assert abs(ex) < 1e-9 and abs(ey) < 1e-9
    assert abs(h - 0.10) < 1e-9


def test_aim_error_right_of_center_is_positive_x() -> None:
    (ex, ey), _ = aim_error_for_box((80, 45, 90, 55), 100, 100, 0.5)
    assert ex > 0


def test_aim_error_above_center_is_positive_y() -> None:
    # A box near the top → aim point above center → ey positive (tilt up).
    (_ex, ey), _ = aim_error_for_box((45, 0, 55, 10), 100, 100, 0.5)
    assert ey > 0


def test_aim_error_fraction_moves_aim_down_the_box() -> None:
    # aim_fraction=0.0 aims at the box top; 1.0 aims at the bottom → smaller ey.
    (_a, ey_top), _ = aim_error_for_box((45, 10, 55, 90), 100, 100, 0.0)
    (_b, ey_bot), _ = aim_error_for_box((45, 10, 55, 90), 100, 100, 1.0)
    assert ey_top > ey_bot


def test_aim_error_degenerate_frame_is_zero() -> None:
    assert aim_error_for_box((0, 0, 1, 1), 0, 0, 0.5) == ((0.0, 0.0), 0.0)
