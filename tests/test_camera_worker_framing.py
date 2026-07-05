"""CameraWorker Center-Stage crop-rect telemetry wiring.

Verifies _framed_output records the active digital crop into
_last_digital_crop_rect (and clears it when Center Stage is inactive), and that
the rect is carried on the emitted TelemetryMsg.
"""

from __future__ import annotations

import numpy as np


def _bare_worker():
    """A CameraWorker instance with only the attributes _framed_output touches,
    built without running the capture thread (we never call .start())."""
    from autoptz.config.models import CameraConfig
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(id="cam-fr", name="Cam FR")
    return CameraWorker("cam-fr", cfg, on_telemetry=lambda m: None)


def test_framed_output_records_none_without_digital_backend() -> None:
    w = _bare_worker()
    w._ptz_backend = None  # no Center Stage
    w._last_digital_crop_rect = (1, 2, 3, 4)  # stale value must be cleared
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = w._framed_output(frame)
    assert out is frame  # passthrough unchanged
    assert w._last_digital_crop_rect is None


def test_framed_output_records_crop_rect_when_center_stage_active() -> None:
    from autoptz.engine.ptz.digital import DigitalPTZBackend

    w = _bare_worker()
    w._ptz_backend = DigitalPTZBackend()  # digital crop active
    # No target locked → framer eases toward the full frame on the first tick,
    # but _framed_output STILL applies (crops/scales) and records a rect.
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    w._framed_output(frame)
    rect = w._last_digital_crop_rect
    assert rect is not None
    x, y, cw, ch = rect
    assert 0 <= x and 0 <= y
    assert 0 < cw <= 1920 and 0 < ch <= 1080


def _standing_kps():
    """COCO-17 keypoints for a standing person: shoulders y=100, hips y=300."""
    from autoptz.engine.pipeline.framing import (
        KP_LEFT_HIP,
        KP_LEFT_SHOULDER,
        KP_RIGHT_HIP,
        KP_RIGHT_SHOULDER,
        Keypoint,
    )

    kps = [Keypoint(0.0, 0.0, 0.0)] * 17
    kps[KP_LEFT_SHOULDER] = Keypoint(170.0, 100.0, 0.9)
    kps[KP_RIGHT_SHOULDER] = Keypoint(230.0, 100.0, 0.9)
    kps[KP_LEFT_HIP] = Keypoint(180.0, 300.0, 0.9)
    kps[KP_RIGHT_HIP] = Keypoint(220.0, 300.0, 0.9)
    return kps


def _locked_worker_with_pose(*, aim_body_mode: str = "torso", raised_arm_bbox=None):
    """A bare worker with track 1 locked, a big 'arms raised' bbox, and fresh
    cached torso keypoints for that track."""
    import time

    from autoptz.config.models import CameraConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker
    from autoptz.engine.runtime.messages import BBox, TrackInfo

    cfg = CameraConfig(
        id="cam-fr",
        name="Cam FR",
        tracking=TrackingConfig(aim_body_mode=aim_body_mode),
    )
    w = CameraWorker("cam-fr", cfg, on_telemetry=lambda m: None)
    w._tracking_enabled = True  # Center Stage only crops while tracking is on
    box = raised_arm_bbox or (60.0, 20.0, 340.0, 640.0)  # arms up: tall + wide
    w._last_tracks = [TrackInfo(track_id=1, bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))]
    w._target_track_id = 1
    # Mirror _pose_aim's success state: the strict per-tick cache AND the
    # sticky last-good cache that framing reads.
    w._pose_keypoints = _standing_kps()
    w._pose_kp_track_id = 1
    w._last_pose_t = time.monotonic()
    w._note_good_kps(_standing_kps(), 1, time.monotonic())
    return w, box


def test_center_stage_no_crop_when_tracking_disabled() -> None:
    """Center Stage must only crop while tracking is enabled: with the Track
    toggle off there is no framing target, so the crop eases to full frame."""
    w, _ = _locked_worker_with_pose()
    w._tracking_enabled = False
    assert w._current_digital_target() is None


def test_center_stage_no_crop_when_tracking_feature_off() -> None:
    """The global tracking feature switch gates the crop like the per-camera
    Track toggle does."""
    w, _ = _locked_worker_with_pose()
    w.set_features({"tracking": False})
    assert w._current_digital_target() is None


def test_center_stage_torso_box_when_ignore_arms() -> None:
    """aim_body_mode="torso" (Ignore arms): the Center Stage crop frames the
    pose-torso-derived box, NOT the raw (arms-inflated) detection bbox."""
    from autoptz.engine.pipeline.framing import torso_framing_box

    w, raw_box = _locked_worker_with_pose(aim_body_mode="torso")
    target = w._current_digital_target()
    assert target is not None
    assert target != raw_box
    assert target == torso_framing_box(_standing_kps())


def test_center_stage_torso_box_invariant_to_bbox_growth() -> None:
    """Raising arms grows the YOLO bbox — the framed target must not change."""
    w1, _ = _locked_worker_with_pose(raised_arm_bbox=(150.0, 60.0, 250.0, 640.0))
    w2, _ = _locked_worker_with_pose(raised_arm_bbox=(20.0, 5.0, 380.0, 640.0))
    assert w1._current_digital_target() == w2._current_digital_target()


def test_center_stage_raw_bbox_when_full_silhouette() -> None:
    """aim_body_mode="full_silhouette" (include arms) keeps the raw bbox."""
    w, raw_box = _locked_worker_with_pose(aim_body_mode="full_silhouette")
    assert w._current_digital_target() == raw_box


def test_framing_pose_survives_single_bad_estimate() -> None:
    """One failed/inconsistent pose estimate (production clears _pose_keypoints)
    must NOT snap framing back to the raw arms-inflated bbox: the last GOOD
    keypoints hold the torso box for _POSE_FRAMING_TTL_S."""
    from autoptz.engine.pipeline.framing import torso_framing_box

    w, raw_box = _locked_worker_with_pose()
    # Exactly what _pose_aim's failure branch does on one bad estimate:
    w._pose_keypoints = None
    w._pose_kp_track_id = None
    assert w._current_digital_target() == torso_framing_box(_standing_kps())
    assert w._current_digital_target() != raw_box


def test_center_stage_raw_bbox_without_pose() -> None:
    w, raw_box = _locked_worker_with_pose()
    w._reset_pose_aim()  # pose fully unavailable — no good keypoints ever held
    assert w._current_digital_target() == raw_box


def test_center_stage_raw_bbox_when_pose_is_other_track() -> None:
    w, raw_box = _locked_worker_with_pose()
    # Keypoints belong to someone else (both caches).
    w._pose_kp_track_id = 2
    w._last_good_kps_track_id = 2
    assert w._current_digital_target() == raw_box


def test_center_stage_raw_bbox_when_pose_stale() -> None:
    import time

    w, raw_box = _locked_worker_with_pose()
    # No successful estimate for a while (inference thread stalled).
    w._last_pose_good_t = time.monotonic() - 5.0
    assert w._current_digital_target() == raw_box


def test_pose_runs_for_group_single_person_without_lock() -> None:
    """Group framing + one confident person + NO explicit lock: pose must still
    be estimated for that person, so the torso-stable framing box exists."""
    import time

    from autoptz.config.models import CameraConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker
    from autoptz.engine.runtime.messages import BBox, TrackInfo

    cfg = CameraConfig(id="cam-fr", name="Cam FR", tracking=TrackingConfig(group_framing=True))
    w = CameraWorker("cam-fr", cfg, on_telemetry=lambda m: None)
    tracks = [TrackInfo(track_id=7, bbox=BBox(x1=0, y1=0, x2=100, y2=200))]
    seen: list[int] = []
    w._pose_aim = lambda t, *a, **k: (seen.append(t.track_id), (None, 0.0, 0.0))[1]
    w._maybe_estimate_pose_overlay(tracks, np.zeros((720, 1280, 3), np.uint8), time.monotonic())
    assert seen == [7]


def test_pose_not_run_for_group_of_many_without_lock() -> None:
    """A multi-person group union has no single subject — no pose focus."""
    import time

    from autoptz.config.models import CameraConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker
    from autoptz.engine.runtime.messages import BBox, TrackInfo

    cfg = CameraConfig(id="cam-fr", name="Cam FR", tracking=TrackingConfig(group_framing=True))
    w = CameraWorker("cam-fr", cfg, on_telemetry=lambda m: None)
    tracks = [
        TrackInfo(track_id=7, bbox=BBox(x1=0, y1=0, x2=100, y2=200)),
        TrackInfo(track_id=8, bbox=BBox(x1=300, y1=0, x2=400, y2=200)),
    ]
    seen: list[int] = []
    w._pose_aim = lambda t, *a, **k: (seen.append(t.track_id), (None, 0.0, 0.0))[1]
    w._maybe_estimate_pose_overlay(tracks, np.zeros((720, 1280, 3), np.uint8), time.monotonic())
    assert seen == []


def _ptz_error_for_box(box, *, aim_body_mode: str = "torso"):
    """(error, subject_height) the physical-PTZ path computes for *box* when the
    pose fusion is unavailable (worst case: previously pure raw-bbox aim)."""
    import time

    w, _ = _locked_worker_with_pose(aim_body_mode=aim_body_mode, raised_arm_bbox=box)
    w._pose_aim = lambda *a, **k: (None, 0.0, 0.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    return w._track_error(w._last_tracks[0], frame, time.monotonic(), tracks=w._last_tracks)


def test_ptz_track_error_arm_growth_invariant_in_torso_mode() -> None:
    """Physical PTZ: with fresh torso keypoints cached, an arm-inflated bbox must
    not move the aim error or the zoom subject height ("Ignore arms")."""
    err_a, h_a = _ptz_error_for_box((150.0, 60.0, 250.0, 640.0))  # arms down
    err_b, h_b = _ptz_error_for_box((20.0, 5.0, 380.0, 640.0))  # arms out+up
    assert err_a == err_b
    assert h_a == h_b


def test_ptz_track_error_follows_bbox_in_full_silhouette_mode() -> None:
    """Include-arms mode intentionally keeps the raw-box behaviour."""
    err_a, _ = _ptz_error_for_box((150.0, 60.0, 250.0, 640.0), aim_body_mode="full_silhouette")
    err_b, _ = _ptz_error_for_box((20.0, 5.0, 380.0, 640.0), aim_body_mode="full_silhouette")
    assert err_a != err_b


def test_torso_box_is_smoothed_not_stepped() -> None:
    """Pose estimates arrive in ~0.2 s steps; the framing box must EASE toward
    a new estimate, never jump onto it (the reported jitter)."""
    import time

    from autoptz.engine.pipeline.framing import Keypoint, torso_framing_box

    w, _ = _locked_worker_with_pose()
    first = w._current_digital_target()
    # Next pose estimate: torso shifted 80 px right (subject moved / kp noise).
    shifted = [Keypoint(kp.x + 80.0, kp.y, kp.conf) for kp in _standing_kps()]
    w._note_good_kps(shifted, 1, time.monotonic())
    second = w._current_digital_target()
    raw = torso_framing_box(shifted)
    assert second != raw  # no instant jump onto the new estimate
    assert abs(second[0] - first[0]) < 8.0  # microseconds later → barely moved


def test_head_recovery_has_hysteresis_against_flapping() -> None:
    """Once the tilt-up recovery is active, a BORDERLINE head landmark (conf
    just above the normal visibility floor) must not flap it off — exit needs a
    clearly-visible head."""
    import time

    from autoptz.engine.pipeline.framing import KP_NOSE, Keypoint

    w, _ = _locked_worker_with_pose(
        aim_body_mode="full_silhouette", raised_arm_bbox=(150.0, 0.0, 250.0, 720.0)
    )
    w._pose_aim = lambda *a, **k: (None, 0.0, 0.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Tick 1: head fully missing → recovery activates.
    (_, ey1), _ = w._track_error(w._last_tracks[0], frame, time.monotonic(), tracks=w._last_tracks)
    assert ey1 >= 0.30
    # Tick 2: nose flickers in at conf 0.40 (visible by the 0.35 floor, but not
    # CLEARLY visible) — recovery must hold, not flap off.
    kps = list(_standing_kps())
    kps[KP_NOSE] = Keypoint(200.0, 60.0, 0.40)
    w._note_good_kps(kps, 1, time.monotonic())
    (_, ey2), _ = w._track_error(w._last_tracks[0], frame, time.monotonic(), tracks=w._last_tracks)
    assert ey2 >= 0.30
    # Tick 3: nose clearly visible (0.60) → recovery releases.
    kps2 = list(_standing_kps())
    kps2[KP_NOSE] = Keypoint(200.0, 60.0, 0.60)
    w._note_good_kps(kps2, 1, time.monotonic())
    (_, ey3), _ = w._track_error(w._last_tracks[0], frame, time.monotonic(), tracks=w._last_tracks)
    assert ey3 < 0.30


def test_framing_source_flip_is_logged(caplog) -> None:
    """Transparency: when 'Ignore arms' framing degrades torso→bbox (or
    recovers), a log line says so — field runs must be diagnosable."""
    import logging

    w, _ = _locked_worker_with_pose()
    with caplog.at_level(logging.INFO, logger="autoptz.engine.camera_worker"):
        w._current_digital_target()  # torso-stable engaged
        w._last_pose_good_t -= 30.0  # pose expires mid-run
        w._current_digital_target()  # → raw bbox
    messages = [r.getMessage() for r in caplog.records]
    assert any("framing source" in m and "bbox" in m for m in messages), messages


def _head_assist_error(box, *, with_head: bool = False):
    """(ex, ey) for a worker whose cached pose has torso keypoints and — only
    when *with_head* — a confident nose. full_silhouette mode isolates the
    assist from the torso-anchor substitution."""
    import time

    from autoptz.engine.pipeline.framing import KP_NOSE, Keypoint

    w, _ = _locked_worker_with_pose(aim_body_mode="full_silhouette", raised_arm_bbox=box)
    if with_head:
        kps = list(w._pose_keypoints)
        kps[KP_NOSE] = Keypoint(200.0, 60.0, 0.9)
        w._pose_keypoints = kps
        w._note_good_kps(kps, 1, time.monotonic())
    w._pose_aim = lambda *a, **k: (None, 0.0, 0.0)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    (ex, ey), _ = w._track_error(w._last_tracks[0], frame, time.monotonic(), tracks=w._last_tracks)
    return ex, ey


def test_head_out_of_view_above_frame_tilts_up() -> None:
    """Body visible (torso keypoints), no head landmark, box clipped at the
    frame top → the aim error is biased upward so the PTZ recovers the head
    instead of parking on the body."""
    _, ey = _head_assist_error((150.0, 0.0, 250.0, 720.0))  # top-clipped, full height
    assert ey >= 0.30


def test_no_tilt_assist_when_head_visible() -> None:
    _, ey = _head_assist_error((150.0, 0.0, 250.0, 720.0), with_head=True)
    assert ey < 0.30


def test_no_tilt_assist_when_box_not_top_clipped() -> None:
    _, ey = _head_assist_error((150.0, 200.0, 250.0, 720.0))
    assert ey < 0.0  # aim sits below centre; no artificial up bias


def test_telemetry_carries_last_digital_crop_rect() -> None:
    from autoptz.engine.runtime.messages import HealthState, TelemetryMsg

    w = _bare_worker()
    w._last_digital_crop_rect = (100, 50, 600, 400)
    # The emit helper builds a TelemetryMsg; assert the field is wired through.
    captured: list[TelemetryMsg] = []
    w._on_telemetry = captured.append
    w._emit_telemetry(tracks=[], health=HealthState.OK, last_error=None)
    assert captured and captured[0].digital_crop_rect == (100, 50, 600, 400)
