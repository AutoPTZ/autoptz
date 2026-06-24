from __future__ import annotations

from autoptz.engine.ptz.digital import DigitalPTZBackend


def test_home_is_full_frame_centered():
    b = DigitalPTZBackend()
    x, y, w, h = b.crop_rect(1920, 1080)
    assert (x, y, w, h) == (0, 0, 1920, 1080)


def test_zoom_in_shrinks_and_recenters():
    b = DigitalPTZBackend(min_crop_frac=0.5, max_step_per_s=10.0)
    b.move_velocity(0.0, 0.0, 1.0)  # zoom tele
    b.move_velocity(0.0, 0.0, 1.0)
    x, y, w, h = b.crop_rect(1000, 1000)
    assert w < 1000 and h < 1000
    assert abs((x + w / 2) - 500) <= 1  # still centered horizontally
    assert w >= 500  # never below min_crop_frac


def test_pan_right_shifts_crop_right():
    b = DigitalPTZBackend(min_crop_frac=0.5, max_step_per_s=10.0)
    b.move_velocity(0.0, 0.0, 1.0)  # zoom in so there's room to pan
    b.move_velocity(1.0, 0.0, 0.0)  # pan right
    x, _, w, _ = b.crop_rect(1000, 1000)
    assert x + w <= 1000 and x > 0  # shifted right, still inside the frame


def test_crop_never_leaves_frame():
    b = DigitalPTZBackend(min_crop_frac=0.4, max_step_per_s=100.0)
    for _ in range(50):
        b.move_velocity(1.0, 1.0, 1.0)
    x, y, w, h = b.crop_rect(1280, 720)
    assert 0 <= x and 0 <= y and x + w <= 1280 and y + h <= 720


def test_framed_output_frames_the_target_when_digital_backend_active():
    import numpy as np

    from autoptz.config.models import CameraConfig, PTZConfig, SourceConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker
    from autoptz.engine.ptz.digital import DigitalPTZBackend
    from autoptz.engine.runtime.messages import BBox, TrackInfo

    cfg = CameraConfig(
        id="cam-dig-000001",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(),
        ptz=PTZConfig(backend="digital", digital_output_w=320, digital_output_h=240),
    )
    w = CameraWorker("cam-dig-000001", cfg, on_telemetry=lambda m: None)
    w._ptz_backend = DigitalPTZBackend()
    # A selected target occupying ~25% of a 1280x720 frame → the auto-framer crops
    # onto it (zoomed in) and the output is scaled to the configured size.
    w._target_track_id = 7
    w._last_tracks = [TrackInfo(track_id=7, bbox=BBox(x1=900, y1=300, x2=1020, y2=480))]
    out = w._framed_output(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert out.shape[1] == 320 and out.shape[0] == 240  # scaled to output size
    x, y, cw, ch = w._digital_framer._crop
    assert cw < 1280 and ch < 720  # cropped to less than the full frame (zoomed on target)


def test_toggling_center_stage_rebuilds_the_backend_live():
    from autoptz.config.models import CameraConfig, PTZConfig, SourceConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker
    from autoptz.engine.ptz.digital import DigitalPTZBackend

    cfg = CameraConfig(
        id="cam-cs-rebuild1",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(),
        ptz=PTZConfig(backend="auto"),
    )
    w = CameraWorker("cam-cs-rebuild1", cfg, on_telemetry=lambda m: None)
    # A webcam on "auto" has no PTZ hardware → no digital backend yet.
    assert not isinstance(w._ptz_backend, DigitalPTZBackend)

    # Toggle Center Stage on (backend → "digital") via a live config update.
    on = cfg.model_copy(update={"ptz": cfg.ptz.model_copy(update={"backend": "digital"})})
    w.update_config(on)
    w._drain_commands()
    assert isinstance(w._ptz_backend, DigitalPTZBackend)

    # Toggle off again → the digital backend is torn down.
    off = on.model_copy(update={"ptz": on.ptz.model_copy(update={"backend": "auto"})})
    w.update_config(off)
    w._drain_commands()
    assert not isinstance(w._ptz_backend, DigitalPTZBackend)


def test_framed_output_passthrough_without_digital_backend():
    import numpy as np

    from autoptz.config.models import CameraConfig, PTZConfig, SourceConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(
        id="cam-dig-000002",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(),
        ptz=PTZConfig(),
    )
    w = CameraWorker("cam-dig-000002", cfg, on_telemetry=lambda m: None)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert w._framed_output(frame) is frame  # untouched
