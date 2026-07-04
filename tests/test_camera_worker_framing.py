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
    box = raised_arm_bbox or (60.0, 20.0, 340.0, 640.0)  # arms up: tall + wide
    w._last_tracks = [TrackInfo(track_id=1, bbox=BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))]
    w._target_track_id = 1
    w._pose_keypoints = _standing_kps()
    w._pose_kp_track_id = 1
    w._last_pose_t = time.monotonic()
    return w, box


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


def test_center_stage_raw_bbox_without_pose() -> None:
    w, raw_box = _locked_worker_with_pose()
    w._pose_keypoints = None
    assert w._current_digital_target() == raw_box


def test_center_stage_raw_bbox_when_pose_is_other_track() -> None:
    w, raw_box = _locked_worker_with_pose()
    w._pose_kp_track_id = 2  # keypoints belong to someone else
    assert w._current_digital_target() == raw_box


def test_center_stage_raw_bbox_when_pose_stale() -> None:
    import time

    w, raw_box = _locked_worker_with_pose()
    w._last_pose_t = time.monotonic() - 5.0  # inference thread stalled
    assert w._current_digital_target() == raw_box


def test_telemetry_carries_last_digital_crop_rect() -> None:
    from autoptz.engine.runtime.messages import HealthState, TelemetryMsg

    w = _bare_worker()
    w._last_digital_crop_rect = (100, 50, 600, 400)
    # The emit helper builds a TelemetryMsg; assert the field is wired through.
    captured: list[TelemetryMsg] = []
    w._on_telemetry = captured.append
    w._emit_telemetry(tracks=[], health=HealthState.OK, last_error=None)
    assert captured and captured[0].digital_crop_rect == (100, 50, 600, 400)
