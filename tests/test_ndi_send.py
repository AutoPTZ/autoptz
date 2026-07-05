"""NDI output sink: pure BGR→RGBA conversion + graceful no-op when cyndilib is
absent (the repo .venv/CI case)."""

from __future__ import annotations

import numpy as np

from autoptz.engine.pipeline.ndi_send import (
    NDISendSink,
    bgr_to_rgba_flat,
    ndi_output_available,
)


def test_bgr_to_rgba_flat_swaps_channels_and_sets_alpha() -> None:
    # A single pixel BGR = (1, 2, 3) → RGBA = (3, 2, 1, 255).
    frame = np.array([[[1, 2, 3]]], dtype=np.uint8)
    out = bgr_to_rgba_flat(frame)
    assert out.shape == (4,)
    assert list(out) == [3, 2, 1, 255]


def test_bgr_to_rgba_flat_length_matches_pixels() -> None:
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    out = bgr_to_rgba_flat(frame)
    assert out.shape == (4 * 5 * 4,)
    assert out.dtype == np.uint8


def test_available_is_bool() -> None:
    assert isinstance(ndi_output_available(), bool)


def test_sink_is_graceful_noop_without_cyndilib() -> None:
    """With no cyndilib (the CI/.venv case) the sink builds, reports unavailable,
    and send/close never raise."""
    sink = NDISendSink(1280, 720, "AutoPTZ Test Cam")
    assert sink.ndi_name == "AutoPTZ Test Cam"
    if not ndi_output_available():
        assert sink.available is False
    # Regardless of availability, these must never raise.
    sink.send_bgr(np.zeros((720, 1280, 3), dtype=np.uint8))
    sink.close()
    sink.send_bgr(np.zeros((720, 1280, 3), dtype=np.uint8))  # safe after close


def test_send_bgr_uses_injected_sender_when_present() -> None:
    """Inject a fake sender to prove the send path converts + writes even when the
    real cyndilib runtime is absent."""
    writes: list[np.ndarray] = []
    sink = NDISendSink(2, 1, "Cam")
    sink._sender = type("S", (), {"write_video": lambda self, buf: writes.append(buf)})()
    frame = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)  # (1, 2, 3)
    sink.send_bgr(frame)
    assert len(writes) == 1
    # 2 px × RGBA = 8 bytes; first pixel BGR(10,20,30) → RGBA(30,20,10,255).
    assert list(writes[0][:4]) == [30, 20, 10, 255]


# ── worker wiring ────────────────────────────────────────────────────────────


def _worker(**ptz):
    from autoptz.config.models import CameraConfig, PTZConfig
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(id="c", name="Cam 1", ptz=PTZConfig(**ptz))
    w = CameraWorker("c", cfg, on_telemetry=lambda m: None)
    w._ptz_backend = None  # passthrough framing
    w._shm = type("Shm", (), {"push": lambda self, f: None, "height": 100, "width": 100})()
    return w


def test_ndi_output_name_default_and_configured() -> None:
    assert _worker()._ndi_output_name() == "AutoPTZ Cam 1"
    assert _worker(ndi_output_name="Studio Feed")._ndi_output_name() == "Studio Feed"


def test_push_frame_creates_ndi_sink_when_enabled(monkeypatch) -> None:
    import autoptz.engine.pipeline.ndi_send as ndi_mod

    created: list[str] = []

    class _FakeSink:
        def __init__(self, w, h, name, fps=30.0):
            created.append(name)

        def send_bgr(self, f):
            pass

        def close(self):
            pass

    monkeypatch.setattr(ndi_mod, "NDISendSink", _FakeSink)
    w = _worker(ndi_out=True)
    w._push_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    assert created == ["AutoPTZ Cam 1"]
    assert w._ndi is not None


def test_push_frame_closes_ndi_sink_when_disabled() -> None:
    w = _worker(ndi_out=False)
    closed = {"n": 0}
    w._ndi = type("S", (), {"close": lambda self: closed.__setitem__("n", 1)})()
    w._push_frame(np.zeros((100, 100, 3), dtype=np.uint8))
    assert closed["n"] == 1
    assert w._ndi is None
