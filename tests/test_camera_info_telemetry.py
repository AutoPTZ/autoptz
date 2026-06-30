"""Phase 0 per-camera telemetry: CameraRecord accessors + CameraInfoPanel rows.

These exercise (1) the read-only ``@property`` accessors on ``CameraRecord`` that
mirror the new ``TelemetryMsg`` Phase-0 fields, and (2) that ``CameraInfoPanel``
renders the new rows, conditionally hiding the ones that would otherwise show a
confusing sentinel (NDI queue depth ``-1``) or noise (zero/undriven values).

Run offscreen so they execute in CI without a display:
    QT_QPA_PLATFORM=offscreen python -m pytest tests/test_camera_info_telemetry.py -q
"""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _telemetry(camera_id: str = "cam0", **fields):
    from autoptz.engine.runtime.messages import TelemetryMsg

    return TelemetryMsg(camera_id=camera_id, seq=1, **fields)


# ── CameraRecord accessors ────────────────────────────────────────────────────


class TestCameraRecordPhase0:
    def test_defaults_without_telemetry(self, qtapp) -> None:
        from autoptz.ui.engine_client import CameraRecord

        rec = CameraRecord("id1", "ndi://x", "Cam")
        # ndi_queue_depth must be -1 (the "no queue" sentinel) when unknown.
        assert rec.ndi_queue_depth == -1
        assert rec.frames_delivered == 0
        assert rec.frames_dropped_est == 0
        assert rec.delivered_fps == 0.0
        assert rec.source_fps == 0.0
        assert rec.duplicate_frames == 0
        assert rec.stale_frames == 0
        assert rec.ndi_fourcc == ""
        assert rec.ndi_queue_audio == -1
        assert rec.ndi_queue_metadata == -1
        assert rec.ndi_total_video_frames == 0
        assert rec.ndi_dropped_video_frames == 0
        assert rec.ndi_connections == -1
        assert rec.ndi_conversion_ms == 0.0
        assert rec.ndi_buffer_ms == 0.0
        assert rec.ndi_copy_ms == 0.0
        assert rec.end_to_end_ms == 0.0
        assert rec.capture_age_ms == 0.0
        assert rec.command_send_ms == 0.0
        assert rec.actuation_estimate_ms == 0.0

    def test_values_from_telemetry(self, qtapp) -> None:
        from autoptz.ui.engine_client import CameraRecord

        rec = CameraRecord("id1", "ndi://x", "Cam")
        rec.telemetry = _telemetry(
            "id1",
            frames_delivered=900,
            frames_dropped_est=12,
            delivered_fps=29.5,
            source_fps=30.0,
            duplicate_frames=8,
            stale_frames=2,
            ndi_queue_depth=3,
            ndi_queue_audio=1,
            ndi_queue_metadata=0,
            ndi_total_video_frames=900,
            ndi_dropped_video_frames=5,
            ndi_connections=1,
            ndi_fourcc="UYVY",
            ndi_buffer_ms=0.5,
            ndi_conversion_ms=1.75,
            ndi_copy_ms=0.25,
            capture_age_ms=40.0,
            command_send_ms=5.0,
            actuation_estimate_ms=80.0,
            end_to_end_ms=125.0,
        )
        assert rec.frames_delivered == 900
        assert rec.frames_dropped_est == 12
        assert rec.delivered_fps == pytest.approx(29.5)
        assert rec.source_fps == pytest.approx(30.0)
        assert rec.duplicate_frames == 8
        assert rec.stale_frames == 2
        assert rec.ndi_queue_depth == 3
        assert rec.ndi_queue_audio == 1
        assert rec.ndi_queue_metadata == 0
        assert rec.ndi_total_video_frames == 900
        assert rec.ndi_dropped_video_frames == 5
        assert rec.ndi_connections == 1
        assert rec.ndi_fourcc == "UYVY"
        assert rec.ndi_buffer_ms == pytest.approx(0.5)
        assert rec.ndi_conversion_ms == pytest.approx(1.75)
        assert rec.ndi_copy_ms == pytest.approx(0.25)
        assert rec.capture_age_ms == pytest.approx(40.0)
        assert rec.command_send_ms == pytest.approx(5.0)
        assert rec.actuation_estimate_ms == pytest.approx(80.0)
        assert rec.end_to_end_ms == pytest.approx(125.0)


# ── CameraInfoPanel rows ──────────────────────────────────────────────────────


class _Signal:
    def connect(self, _slot) -> None:
        pass


class _FakeCameraModel:
    def __init__(self, rec) -> None:
        self._rec = rec

    def get_record(self, camera_id):  # noqa: ANN001
        return self._rec


class _FakeClient:
    """Minimal stand-in for EngineClient that CameraInfoPanel reads from."""

    def __init__(self, rec, config) -> None:
        self.engineRunning = True
        self.engineEp = "CPUExecutionProvider"
        self.cameraModel = _FakeCameraModel(rec)
        self._config = config
        # Signals the panel connects to in __init__.
        self.telemetryUpdated = _Signal()
        self.configChanged = _Signal()
        self.engineStateChanged = _Signal()
        # on_theme_changed() in widgets.common probes for a theme signal; a plain
        # object without it is fine (the helper swallows the AttributeError).

    def getCameraConfig(self, _camera_id):  # noqa: ANN001
        return self._config


def _panel_for(qtapp, telemetry):
    from autoptz.ui.engine_client import CameraRecord
    from autoptz.ui.widgets.camera_info_panel import CameraInfoPanel

    rec = CameraRecord("cam0", "ndi://x", "Cam")
    rec.telemetry = telemetry
    config = {
        "name": "Cam",
        "source": {"type": "ndi", "address": "ndi://x"},
        "tracking": {},
        "ptz": {},
    }
    client = _FakeClient(rec, config)
    panel = CameraInfoPanel(client)
    panel.set_camera("cam0")
    panel.refresh()
    return panel


def _text(panel, key: str) -> str:
    return panel._vals[key].text()


def _visible(panel, key: str) -> bool:
    # The panel is never .show()n in tests, so isVisible() is always False
    # (no top-level window). isHidden() reflects the *explicit* hide state set by
    # setVisible(), which is exactly the conditional-row behaviour under test.
    return not panel._vals[key].isHidden()


class TestCameraInfoPanelPhase0:
    def test_est_source_drops_row_shows_and_warns(self, qtapp) -> None:
        from autoptz.ui import theme as T

        panel = _panel_for(
            qtapp, _telemetry(frames_dropped_est=7, source_fps=30.0, delivered_fps=28.0)
        )
        assert _text(panel, "Est. source drops") == "7"
        # >0 drops paints WARNING.
        assert panel._val_colors["Est. source drops"] == T.WARNING

    def test_est_source_drops_no_warning_when_zero(self, qtapp) -> None:
        from autoptz.ui import theme as T

        panel = _panel_for(qtapp, _telemetry(frames_dropped_est=0))
        assert _text(panel, "Est. source drops") == "0"
        assert panel._val_colors["Est. source drops"] != T.WARNING

    def test_delivered_source_row_shown_when_source_fps_positive(self, qtapp) -> None:
        panel = _panel_for(qtapp, _telemetry(delivered_fps=28.0, source_fps=30.0))
        assert _visible(panel, "Delivered / source") is True
        assert _text(panel, "Delivered / source") == "28.0 / 30.0 fps"

    def test_delivered_source_row_hidden_when_source_fps_zero(self, qtapp) -> None:
        panel = _panel_for(qtapp, _telemetry(delivered_fps=0.0, source_fps=0.0))
        assert _visible(panel, "Delivered / source") is False

    def test_duplicate_stale_row_is_conditional(self, qtapp) -> None:
        panel = _panel_for(qtapp, _telemetry(duplicate_frames=0, stale_frames=0))
        assert _visible(panel, "Duplicate / stale") is False

        panel2 = _panel_for(qtapp, _telemetry(duplicate_frames=8, stale_frames=2))
        assert _visible(panel2, "Duplicate / stale") is True
        assert _text(panel2, "Duplicate / stale") == "8 / 2"

    def test_ndi_receiver_rows_are_conditional(self, qtapp) -> None:
        panel = _panel_for(qtapp, _telemetry())
        assert _visible(panel, "NDI recv drops") is False
        assert _visible(panel, "NDI connections") is False

        panel2 = _panel_for(
            qtapp,
            _telemetry(
                ndi_total_video_frames=900,
                ndi_dropped_video_frames=3,
                ndi_connections=1,
            ),
        )
        assert _visible(panel2, "NDI recv drops") is True
        assert _text(panel2, "NDI recv drops") == "3 / 900"
        assert _visible(panel2, "NDI connections") is True
        assert _text(panel2, "NDI connections") == "1"

    def test_ndi_queue_row_hidden_when_unavailable(self, qtapp) -> None:
        # -1 == no queue exposed; the row must NOT show a confusing "-1".
        panel = _panel_for(qtapp, _telemetry(ndi_queue_depth=-1))
        assert _visible(panel, "NDI queue depth") is False

    def test_ndi_queue_row_shown_when_available(self, qtapp) -> None:
        panel = _panel_for(qtapp, _telemetry(ndi_queue_depth=0))
        assert _visible(panel, "NDI queue depth") is True
        assert _text(panel, "NDI queue depth") == "0"

        panel2 = _panel_for(qtapp, _telemetry(ndi_queue_depth=4))
        assert _visible(panel2, "NDI queue depth") is True
        assert _text(panel2, "NDI queue depth") == "4"

    def test_ndi_format_and_conversion_rows_are_conditional(self, qtapp) -> None:
        panel = _panel_for(
            qtapp,
            _telemetry(ndi_fourcc="", ndi_buffer_ms=0.0, ndi_conversion_ms=0.0, ndi_copy_ms=0.0),
        )
        assert _visible(panel, "NDI format") is False
        assert _visible(panel, "NDI buffer") is False
        assert _visible(panel, "NDI conversion") is False
        assert _visible(panel, "NDI copy") is False

        panel2 = _panel_for(
            qtapp,
            _telemetry(
                ndi_fourcc="UYVY",
                ndi_buffer_ms=0.456,
                ndi_conversion_ms=1.234,
                ndi_copy_ms=0.178,
            ),
        )
        assert _visible(panel2, "NDI format") is True
        assert _text(panel2, "NDI format") == "UYVY"
        assert _visible(panel2, "NDI buffer") is True
        assert _text(panel2, "NDI buffer") == "0.46 ms"
        assert _visible(panel2, "NDI conversion") is True
        assert _text(panel2, "NDI conversion") == "1.23 ms"
        assert _visible(panel2, "NDI copy") is True
        assert _text(panel2, "NDI copy") == "0.18 ms"

    def test_end_to_end_latency_row_shown_when_probed(self, qtapp) -> None:
        panel = _panel_for(qtapp, _telemetry(end_to_end_ms=125.4))
        assert _visible(panel, "End-to-end latency") is True
        assert _text(panel, "End-to-end latency") == "125 ms"

    def test_end_to_end_latency_row_hidden_until_probed(self, qtapp) -> None:
        panel = _panel_for(qtapp, _telemetry(end_to_end_ms=0.0))
        assert _visible(panel, "End-to-end latency") is False
