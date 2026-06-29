from __future__ import annotations

import time

import numpy as np
import pytest

from autoptz.config.models import PTZConfig
from autoptz.engine.runtime.contracts import (
    FramePacket,
    RuntimeMode,
    SourceHealth,
    TargetState,
)
from autoptz.ui.widgets.dialogs import mark_preflight


def test_frame_packet_from_frame_records_shape_and_health() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    health = SourceHealth(source_fps=30.0, delivered_fps=30.0)
    pkt = FramePacket.from_frame(
        frame,
        sequence=7,
        source_ts=1.0,
        capture_ts=time.monotonic(),
        pixel_format="bgr",
        health=health,
    )
    assert pkt.width == 1280
    assert pkt.height == 720
    assert pkt.health.app_drop_free is True


def test_frame_packet_rejects_bad_dimensions() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="do not match"):
        FramePacket(
            frame=frame,
            sequence=0,
            source_ts=None,
            capture_ts=time.monotonic(),
            pixel_format="bgr",
            width=21,
            height=10,
        )


def test_runtime_mode_defaults_to_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_SERVER", raising=False)
    assert RuntimeMode.from_env() == RuntimeMode.PRODUCTION_SHARED_MODEL


def test_runtime_mode_model_server_is_labs_only_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOPTZ_MODEL_SERVER", "1")
    assert RuntimeMode.from_env() == RuntimeMode.LABS_MODEL_SERVER


def test_target_state_contract_values() -> None:
    assert [s.value for s in TargetState] == [
        "acquire",
        "track",
        "hold",
        "lost",
        "reacquire",
    ]


def test_fixed_zoom_is_default() -> None:
    assert PTZConfig().auto_zoom is False


def test_tracking_speed_preset_api_removed() -> None:
    import autoptz.config.models as models
    import autoptz.ui.widgets.properties_panel as properties_panel

    assert not hasattr(models, "TrackingSpeed")
    assert not hasattr(models, "SPEED_PROFILES")
    assert not hasattr(models, "apply_speed_profile")
    assert "tracking_speed" not in PTZConfig.model_fields
    assert not hasattr(properties_panel, "match_speed_preset")


def test_mark_camera_options_include_six() -> None:
    assert [value for _label, value in mark_preflight._MAX_CAMERA_OPTS] == [
        1,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
    ]
