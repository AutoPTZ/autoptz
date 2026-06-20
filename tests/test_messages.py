"""Unit tests for autoptz.engine.runtime.messages."""
from __future__ import annotations

import time

import pytest

from autoptz.engine.runtime.messages import (
    AddCameraCmd,
    BaseCommand,
    BBox,
    CmdKind,
    EnableTrackingCmd,
    EnrollIdentityCmd,
    FaceBox,
    HealthInfo,
    HealthState,
    PoseKeypoint,
    PtzGoToPresetCmd,
    PtzNudgeCmd,
    PtzSavePresetCmd,
    PTZState,
    RemoveCameraCmd,
    RuntimeEventInfo,
    RuntimeServiceInfo,
    SetLayoutCmd,
    SetTargetCmd,
    StageTimingInfo,
    SwitchStateInfo,
    TelemetryMsg,
    TrackInfo,
    QualityStateInfo,
)


class TestTelemetryMsg:
    def test_defaults(self) -> None:
        msg = TelemetryMsg(camera_id="cam-1", seq=0)
        assert msg.fps == 0.0
        assert msg.tracks == []
        assert msg.health.state == HealthState.OK

    def test_msgpack_round_trip(self) -> None:
        msg = TelemetryMsg(
            camera_id="cam-abc",
            seq=42,
            fps=29.97,
            ep="CoreMLExecutionProvider",
            tracks=[
                TrackInfo(
                    track_id=1,
                    bbox=BBox(x1=10, y1=20, x2=100, y2=200),
                    identity="Alice",
                    confidence=0.95,
                    is_target=True,
                )
            ],
            ptz=PTZState(pan=0.1, tilt=-0.2, zoom=1.5, moving=True),
            health=HealthInfo(state=HealthState.OK),
        )
        packed = msg.to_msgpack()
        restored = TelemetryMsg.from_msgpack(packed)

        assert restored.camera_id == msg.camera_id
        assert restored.seq == msg.seq
        assert abs(restored.fps - msg.fps) < 1e-9
        assert restored.ep == msg.ep
        assert len(restored.tracks) == 1
        assert restored.tracks[0].identity == "Alice"
        assert restored.tracks[0].is_target is True
        assert abs(restored.ptz.pan - 0.1) < 1e-9
        assert restored.health.state == HealthState.OK

    def test_msgpack_is_compact(self) -> None:
        msg = TelemetryMsg(camera_id="x", seq=0)
        packed = msg.to_msgpack()
        assert len(packed) < 1024, "Bare telemetry should stay reasonably small"

    def test_ts_auto_filled(self) -> None:
        before = time.time()
        msg = TelemetryMsg(camera_id="c", seq=0)
        after = time.time()
        assert before <= msg.ts <= after

    def test_health_error_state(self) -> None:
        msg = TelemetryMsg(
            camera_id="c",
            seq=0,
            health=HealthInfo(state=HealthState.ERROR, last_error="Source stalled"),
        )
        packed = msg.to_msgpack()
        restored = TelemetryMsg.from_msgpack(packed)
        assert restored.health.state == HealthState.ERROR
        assert restored.health.last_error == "Source stalled"

    def test_camera_info_fields_default_zero(self) -> None:
        msg = TelemetryMsg(camera_id="c", seq=0)
        assert msg.width == 0
        assert msg.height == 0
        assert msg.dropped_frames == 0

    def test_camera_info_fields_round_trip(self) -> None:
        msg = TelemetryMsg(
            camera_id="c", seq=1, width=1920, height=1080, dropped_frames=7,
        )
        restored = TelemetryMsg.from_msgpack(msg.to_msgpack())
        assert restored.width == 1920
        assert restored.height == 1080
        assert restored.dropped_frames == 7

    def test_stage_timings_default_zero(self) -> None:
        msg = TelemetryMsg(camera_id="c", seq=0)
        assert msg.face_ms == 0.0
        assert msg.pose_ms == 0.0

    def test_stage_timings_round_trip(self) -> None:
        """Per-stage costs (incl. the new face/pose) survive msgpack — they feed
        the tile "?" badge + Camera Info Performance section."""
        msg = TelemetryMsg(
            camera_id="c", seq=1, ingest_ms=2.0, detect_ms=8.5,
            track_ms=1.2, face_ms=13.0, pose_ms=4.5,
        )
        restored = TelemetryMsg.from_msgpack(msg.to_msgpack())
        assert restored.ingest_ms == 2.0
        assert restored.detect_ms == 8.5
        assert restored.face_ms == 13.0
        assert restored.pose_ms == 4.5

    def test_runtime_transparency_fields_round_trip(self) -> None:
        msg = TelemetryMsg(
            camera_id="c",
            seq=1,
            target_fps=30.0,
            frame_budget_ms=33.333,
            runtime_services=[
                RuntimeServiceInfo(
                    key="detector", name="Detector", scope="global",
                    state="active", model="yolo11s.onnx", tier="balanced",
                    ep="CoreMLExecutionProvider",
                ),
            ],
            stage_timings=[
                StageTimingInfo(
                    key="detect", name="Detector", status="active",
                    last_ms=12.0, avg_ms=10.0, p95_ms=15.0,
                    cadence="every frame", fresh=True, budget_pct=30.0,
                ),
            ],
            quality_state=QualityStateInfo(
                floor="auto", active="balanced",
                reason="latency headroom stable",
                detector_tier="balanced", detector_model="yolo11s.onnx",
                tracker="botsort", detect_interval=2,
            ),
            model_switch=SwitchStateInfo(
                kind="detector", state="active",
                from_value="fast", to_value="balanced",
                active_value="balanced", reason="ready",
            ),
            tracker_switch=SwitchStateInfo(
                kind="tracker", state="active",
                from_value="bytetrack", to_value="botsort",
                active_value="botsort",
            ),
            runtime_events=[RuntimeEventInfo(kind="detector", message="ready")],
        )
        restored = TelemetryMsg.from_msgpack(msg.to_msgpack())
        assert restored.target_fps == pytest.approx(30.0)
        assert restored.frame_budget_ms == pytest.approx(33.333)
        assert restored.runtime_services[0].model == "yolo11s.onnx"
        assert restored.stage_timings[0].avg_ms == pytest.approx(10.0)
        assert restored.quality_state.reason == "latency headroom stable"
        assert restored.model_switch is not None
        assert restored.model_switch.active_value == "balanced"
        assert restored.tracker_switch is not None
        assert restored.tracker_switch.to_value == "botsort"
        assert restored.runtime_events[0].message == "ready"

    def test_overlay_payloads_round_trip(self) -> None:
        """Face boxes + target pose keypoints survive msgpack for the overlays."""
        msg = TelemetryMsg(
            camera_id="c", seq=1, width=1280, height=720,
            faces=[FaceBox(bbox=BBox(x1=30, y1=25, x2=70, y2=65),
                           identity="Alice", score=0.82)],
            pose=[PoseKeypoint(x=50.0, y=60.0, conf=0.9) for _ in range(17)],
        )
        restored = TelemetryMsg.from_msgpack(msg.to_msgpack())
        assert len(restored.faces) == 1
        assert restored.faces[0].identity == "Alice"
        assert abs(restored.faces[0].score - 0.82) < 1e-9
        assert len(restored.pose) == 17
        assert restored.pose[0].conf == 0.9

    def test_overlay_payloads_default_empty(self) -> None:
        """Overlays default to empty so a worker that never ran them sends nothing."""
        msg = TelemetryMsg(camera_id="c", seq=0)
        assert msg.faces == []
        assert msg.pose == []

    def test_track_lost_and_velocity_round_trip(self) -> None:
        """The lost flag + velocity (for the prediction indicator) survive msgpack."""
        msg = TelemetryMsg(
            camera_id="c", seq=1,
            tracks=[TrackInfo(track_id=3, bbox=BBox(x1=1, y1=2, x2=3, y2=4),
                              lost=True, vx=5.5, vy=-2.0, is_target=True)],
        )
        restored = TelemetryMsg.from_msgpack(msg.to_msgpack())
        t = restored.tracks[0]
        assert t.lost is True
        assert abs(t.vx - 5.5) < 1e-9 and abs(t.vy + 2.0) < 1e-9

    def test_track_aim_round_trip(self) -> None:
        msg = TelemetryMsg(
            camera_id="c", seq=1, width=640, height=480,
            tracks=[TrackInfo(
                track_id=3, bbox=BBox(x1=10, y1=20, x2=110, y2=220),
                is_target=True, aim_x=70.0, aim_y=90.0, aim_source="pose",
            )],
        )
        restored = TelemetryMsg.from_msgpack(msg.to_msgpack())
        t = restored.tracks[0]
        assert t.aim_x == pytest.approx(70.0)
        assert t.aim_y == pytest.approx(90.0)
        assert t.aim_source == "pose"

    def test_track_defaults_not_lost(self) -> None:
        t = TrackInfo(track_id=1, bbox=BBox(x1=0, y1=0, x2=1, y2=1))
        assert t.lost is False and t.vx == 0.0 and t.vy == 0.0


class TestCommands:
    def _round_trip(self, cmd: BaseCommand) -> BaseCommand:
        packed = cmd.to_msgpack()
        return BaseCommand.from_msgpack(packed)

    def test_add_camera(self) -> None:
        cmd = AddCameraCmd(
            camera_id="cam-uuid-001",
            source_uri="rtsp://192.168.1.10/stream",
            display_name="Stage Left",
        )
        restored = self._round_trip(cmd)
        assert restored.kind == CmdKind.ADD_CAMERA
        assert restored.camera_id == "cam-uuid-001"

    def test_remove_camera(self) -> None:
        cmd = RemoveCameraCmd(camera_id="cam-uuid-001")
        restored = self._round_trip(cmd)
        assert restored.kind == CmdKind.REMOVE_CAMERA

    def test_set_target_by_track(self) -> None:
        cmd = SetTargetCmd(camera_id="cam-1", track_id=7)
        packed = cmd.to_msgpack()
        # Can also deserialise as SetTargetCmd directly
        restored = SetTargetCmd.from_msgpack(packed)
        assert restored.track_id == 7

    def test_set_target_by_identity(self) -> None:
        cmd = SetTargetCmd(camera_id="cam-1", identity="Alice")
        packed = cmd.to_msgpack()
        restored = SetTargetCmd.from_msgpack(packed)
        assert restored.identity == "Alice"

    def test_enable_tracking(self) -> None:
        cmd = EnableTrackingCmd(camera_id="cam-1", enabled=False)
        restored = self._round_trip(cmd)
        assert restored.kind == CmdKind.ENABLE_TRACKING

    def test_ptz_nudge(self) -> None:
        cmd = PtzNudgeCmd(camera_id="cam-1", pan_speed=0.5, tilt_speed=-0.3)
        packed = cmd.to_msgpack()
        restored = PtzNudgeCmd.from_msgpack(packed)
        assert abs(restored.pan_speed - 0.5) < 1e-9

    def test_ptz_go_to_preset(self) -> None:
        cmd = PtzGoToPresetCmd(camera_id="cam-1", preset_name="wide-shot")
        restored = self._round_trip(cmd)
        assert restored.kind == CmdKind.PTZ_GO_TO_PRESET

    def test_ptz_save_preset(self) -> None:
        cmd = PtzSavePresetCmd(camera_id="cam-1", preset_name="close-up")
        restored = self._round_trip(cmd)
        assert restored.kind == CmdKind.PTZ_SAVE_PRESET

    def test_enroll_identity(self) -> None:
        cmd = EnrollIdentityCmd(
            camera_id="cam-1", identity_name="Bob", track_id=3,
            click_x=0.25, click_y=0.75,
        )
        restored = EnrollIdentityCmd.from_msgpack(cmd.to_msgpack())
        assert restored.kind == CmdKind.ENROLL_IDENTITY
        assert restored.click_x == pytest.approx(0.25)
        assert restored.click_y == pytest.approx(0.75)

    def test_set_layout(self) -> None:
        cmd = SetLayoutCmd(layout_name="2x2")
        restored = self._round_trip(cmd)
        assert restored.kind == CmdKind.SET_LAYOUT

    def test_cmd_ids_are_unique(self) -> None:
        ids = {AddCameraCmd(camera_id="c").cmd_id for _ in range(20)}
        assert len(ids) == 20

    def test_ts_auto_filled(self) -> None:
        before = time.time()
        cmd = AddCameraCmd(camera_id="c")
        after = time.time()
        assert before <= cmd.ts <= after
