"""Unit tests for autoptz.engine.runtime.messages."""
from __future__ import annotations

import time

from autoptz.engine.runtime.messages import (
    AddCameraCmd,
    BaseCommand,
    BBox,
    CmdKind,
    EnableTrackingCmd,
    EnrollIdentityCmd,
    HealthInfo,
    HealthState,
    PtzGoToPresetCmd,
    PtzNudgeCmd,
    PtzSavePresetCmd,
    PTZState,
    RemoveCameraCmd,
    SetLayoutCmd,
    SetTargetCmd,
    TelemetryMsg,
    TrackInfo,
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
        assert len(packed) < 512, "Bare telemetry should be small"

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
        cmd = EnrollIdentityCmd(camera_id="cam-1", identity_name="Bob", track_id=3)
        restored = self._round_trip(cmd)
        assert restored.kind == CmdKind.ENROLL_IDENTITY

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
