"""Unit tests for autoptz.engine.pipeline.ingest.

All external I/O (cv2.VideoCapture, av, cyndilib) is mocked so tests run
without cameras or optional packages installed.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np

from autoptz.engine.pipeline.ingest import (
    AdapterState,
    NDIAdapter,
    RTSPAdapter,
    USBAdapter,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

_H, _W = 120, 160
_FRAME = np.zeros((_H, _W, 3), dtype=np.uint8)


def _make_cap(frames: list[np.ndarray | None]) -> MagicMock:
    """Build a mock cv2.VideoCapture that yields *frames* in order."""
    cap = MagicMock()
    cap.isOpened.return_value = True

    # read() returns (ok, frame) pairs
    read_returns = [(f is not None, f if f is not None else None) for f in frames]
    cap.read.side_effect = read_returns
    return cap


# ── USBAdapter ─────────────────────────────────────────────────────────────────


class TestUSBAdapterOpen:
    def test_open_success(self) -> None:
        cap = _make_cap([_FRAME])
        with patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap):
            adapter = USBAdapter("cam-1", source=0)
            ok = adapter._open()
        assert ok
        assert adapter._cap is cap

    def test_open_failure_returns_false(self) -> None:
        cap = MagicMock()
        cap.isOpened.return_value = False
        with patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap):
            adapter = USBAdapter("cam-1", source=0)
            ok = adapter._open()
        assert not ok
        assert adapter._cap is None

    def test_read_frame_returns_ndarray(self) -> None:
        frame = np.full((_H, _W, 3), 42, dtype=np.uint8)
        cap = _make_cap([frame])
        with patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap):
            adapter = USBAdapter("cam-1", source=0)
            adapter._open()
            result = adapter._read_frame()
        assert result is not None
        np.testing.assert_array_equal(result, frame)

    def test_read_frame_none_on_failure(self) -> None:
        cap = _make_cap([None])
        with patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap):
            adapter = USBAdapter("cam-1", source=0)
            adapter._open()
            result = adapter._read_frame()
        assert result is None

    def test_close_releases_cap(self) -> None:
        cap = _make_cap([_FRAME])
        with patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap):
            adapter = USBAdapter("cam-1", source=0)
            adapter._open()
            adapter._close()
        cap.release.assert_called_once()
        assert adapter._cap is None

    def test_low_current_fps_is_not_treated_as_max_cap(self) -> None:
        cap = _make_cap([_FRAME])
        cap.get.return_value = 15.0
        with patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap):
            adapter = USBAdapter("cam-low-cap", source=0, target_fps=60.0)
            assert adapter._open()

        assert adapter.status.source_fps_cap is None
        assert adapter._target_fps == 60.0

    def test_high_fps_probe_sets_trusted_max_cap(self) -> None:
        cap = _make_cap([_FRAME])
        state = {"fps": 30.0}

        def fake_set(prop: int, value: float) -> bool:
            if prop == 5:  # cv2.CAP_PROP_FPS
                state["fps"] = 60.0 if value <= 60.0 else 30.0
            return True

        def fake_get(prop: int) -> float:
            return state["fps"] if prop == 5 else 0.0

        cap.set.side_effect = fake_set
        cap.get.side_effect = fake_get
        with patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap):
            adapter = USBAdapter("cam-high-cap", source=0, target_fps=120.0)
            assert adapter._open()

        assert adapter.status.source_fps_cap == 60.0
        assert adapter._target_fps == 60.0


class TestUSBAdapterReconnect:
    """Verify the reconnect loop by running the adapter in a thread."""

    def test_stall_triggers_reconnect(self) -> None:
        """After stall_timeout without a frame, _open() should be called again."""
        open_calls: list[float] = []
        frame_count = 0

        def fake_open(self: USBAdapter) -> bool:
            open_calls.append(time.monotonic())
            # Succeed every time
            cap = MagicMock()
            cap.isOpened.return_value = True
            self._cap = cap
            return True

        def fake_read(self: USBAdapter) -> np.ndarray | None:
            nonlocal frame_count
            frame_count += 1
            if frame_count <= 3:
                return _FRAME
            return None  # simulate stall after 3 frames

        adapter = USBAdapter(
            "cam-stall",
            source=0,
            stall_timeout=0.2,  # very short stall timeout
            target_fps=50.0,
        )

        with (
            patch.object(USBAdapter, "_open", fake_open),
            patch.object(USBAdapter, "_read_frame", fake_read),
            patch.object(USBAdapter, "_close", lambda self: None),
        ):
            adapter.start()
            # Give enough time for stall detection + first reconnect
            time.sleep(1.0)
            adapter.stop()

        # Should have opened at least twice (initial + after stall)
        assert len(open_calls) >= 2, f"Expected ≥ 2 open calls, got {open_calls}"

    def test_status_transitions(self) -> None:
        """Status goes RUNNING after a successful open."""
        opened = threading.Event()

        def fake_open(self: USBAdapter) -> bool:
            opened.set()
            cap = MagicMock()
            cap.isOpened.return_value = True
            self._cap = cap
            return True

        def fake_read(self: USBAdapter) -> np.ndarray | None:
            time.sleep(0.05)
            return _FRAME

        adapter = USBAdapter("cam-status", source=0, target_fps=30.0)

        with (
            patch.object(USBAdapter, "_open", fake_open),
            patch.object(USBAdapter, "_read_frame", fake_read),
            patch.object(USBAdapter, "_close", lambda self: None),
        ):
            adapter.start()
            opened.wait(timeout=2.0)
            time.sleep(0.1)
            status = adapter.status
            adapter.stop()

        assert status.state == AdapterState.RUNNING
        assert status.frames_total > 0

    def test_status_includes_source_fps_cap(self) -> None:
        adapter = USBAdapter("cam-cap", source=0, target_fps=30.0)
        adapter._set_source_fps_cap(29.97)
        assert adapter.status.source_fps_cap == 29.97

    def test_stop_state_is_stopped(self) -> None:
        def fake_open(self: USBAdapter) -> bool:
            return False  # always fail

        adapter = USBAdapter("cam-fail", source=0, stall_timeout=5.0)
        with patch.object(USBAdapter, "_open", fake_open):
            adapter.start()
            time.sleep(0.05)
            adapter.stop()

        assert adapter.status.state == AdapterState.STOPPED


# ── RTSPAdapter ────────────────────────────────────────────────────────────────


class TestRTSPAdapterCV2Fallback:
    """RTSPAdapter falls back to cv2 when PyAV is unavailable."""

    def test_open_uses_cv2_fallback(self) -> None:
        cap = _make_cap([_FRAME])

        with (
            patch("autoptz.engine.pipeline.ingest._probe_av", return_value=False),
            patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap),
        ):
            adapter = RTSPAdapter("rtsp-1", url="rtsp://127.0.0.1/test")
            ok = adapter._open()

        assert ok
        assert adapter._cap is cap

    def test_cv2_stream_rate_is_not_treated_as_source_max(self) -> None:
        cap = _make_cap([_FRAME])
        cap.get.return_value = 15.0

        with (
            patch("autoptz.engine.pipeline.ingest._probe_av", return_value=False),
            patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap),
        ):
            adapter = RTSPAdapter("rtsp-low-rate", url="rtsp://127.0.0.1/test", target_fps=60.0)
            assert adapter._open()

        assert adapter.status.source_fps_cap is None
        assert adapter._target_fps == 60.0

    def test_read_frame_cv2(self) -> None:
        frame = np.full((_H, _W, 3), 99, dtype=np.uint8)
        cap = _make_cap([frame])

        with (
            patch("autoptz.engine.pipeline.ingest._probe_av", return_value=False),
            patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap),
        ):
            adapter = RTSPAdapter("rtsp-2", url="rtsp://127.0.0.1/test")
            adapter._open()
            result = adapter._read_frame()

        assert result is not None
        np.testing.assert_array_equal(result, frame)

    def test_close_cv2(self) -> None:
        cap = _make_cap([_FRAME])

        with (
            patch("autoptz.engine.pipeline.ingest._probe_av", return_value=False),
            patch("autoptz.engine.pipeline.ingest.cv2.VideoCapture", return_value=cap),
        ):
            adapter = RTSPAdapter("rtsp-3", url="rtsp://127.0.0.1/test")
            adapter._open()
            adapter._close()

        cap.release.assert_called_once()
        assert adapter._cap is None

    def test_stall_triggers_reconnect(self) -> None:
        open_count = 0

        def fake_open(self: RTSPAdapter) -> bool:
            nonlocal open_count
            open_count += 1
            cap = MagicMock()
            cap.isOpened.return_value = True
            self._cap = cap
            return True

        read_count = 0

        def fake_read(self: RTSPAdapter) -> np.ndarray | None:
            nonlocal read_count
            read_count += 1
            return _FRAME if read_count <= 2 else None

        adapter = RTSPAdapter(
            "rtsp-stall",
            url="rtsp://127.0.0.1/test",
            stall_timeout=0.2,
            target_fps=50.0,
        )

        with (
            patch.object(RTSPAdapter, "_open", fake_open),
            patch.object(RTSPAdapter, "_read_frame", fake_read),
            patch.object(RTSPAdapter, "_close", lambda self: None),
        ):
            adapter.start()
            time.sleep(1.0)
            adapter.stop()

        assert open_count >= 2, f"Expected ≥ 2 opens after stall, got {open_count}"


class TestRTSPAdapterPyAV:
    """RTSPAdapter with a mocked PyAV container."""

    def _make_mock_av_module(self, frames: list[np.ndarray]) -> types.ModuleType:
        av = types.ModuleType("av")

        mock_frame_objects = []
        for f in frames:
            mf = MagicMock()
            mf.to_ndarray.return_value = f
            mock_frame_objects.append(mf)

        mock_packet = MagicMock()
        mock_packet.size = 1
        mock_packet.decode.return_value = mock_frame_objects

        mock_container = MagicMock()
        mock_container.streams.video = [MagicMock()]
        mock_container.demux.return_value = iter([mock_packet])

        av.open = MagicMock(return_value=mock_container)
        av.FFmpegError = Exception
        av.Codec = MagicMock()

        sys.modules["av"] = av
        return av

    def test_open_and_read_with_pyav(self) -> None:
        frame = np.full((_H, _W, 3), 7, dtype=np.uint8)
        self._make_mock_av_module([frame])

        import autoptz.engine.pipeline.ingest as ingest_mod

        ingest_mod._AV_AVAILABLE = True  # force probe to True

        adapter = RTSPAdapter("rtsp-av", url="rtsp://127.0.0.1/cam")
        ok = adapter._open()
        assert ok
        assert adapter._container is not None

        result = adapter._read_frame()
        assert result is not None
        np.testing.assert_array_equal(result, frame)
        adapter._close()

        # Cleanup
        ingest_mod._AV_AVAILABLE = None
        sys.modules.pop("av", None)


# ── NDIAdapter ─────────────────────────────────────────────────────────────────


class TestNDIAdapter:
    """NDIAdapter with a fully mocked cyndilib."""

    def _install_mock_cyndilib(self, source_names: list[str], frame: np.ndarray) -> None:
        cyn = types.ModuleType("cyndilib")
        cyn.__path__ = []  # mark as a package so cyndilib.wrapper imports resolve
        finder_mod = types.ModuleType("cyndilib.finder")
        recv_mod = types.ModuleType("cyndilib.receiver")
        framesync_mod = types.ModuleType("cyndilib.framesync")
        vf_mod = types.ModuleType("cyndilib.video_frame")
        wrapper_mod = types.ModuleType("cyndilib.wrapper")
        wrapper_mod.__path__ = []
        ndi_recv_mod = types.ModuleType("cyndilib.wrapper.ndi_recv")

        # Mock source objects keyed by name (for get_source + iter_sources).
        sources_by_name: dict[str, MagicMock] = {}
        for name in source_names:
            src = MagicMock()
            src.__str__ = MagicMock(return_value=name)
            sources_by_name[name] = src

        # Finder — cyndilib ≥0.1 API: wait_for_sources + get_source + iter_sources.
        finder_instance = MagicMock()
        finder_instance.open.return_value = None
        finder_instance.close.return_value = None
        finder_instance.wait_for_sources.return_value = None
        finder_instance.get_source.side_effect = lambda n: sources_by_name.get(n)
        finder_instance.iter_sources.side_effect = lambda: iter(sources_by_name.values())
        finder_mod.Finder = MagicMock(return_value=finder_instance)

        # Receiver — exposes a ``frame_sync`` (FrameSync); capture_video() fills the
        # *registered* video frame (set via set_video_frame), taking no frame arg.
        registered: dict[str, MagicMock] = {}

        def fake_set_video_frame(vf: MagicMock) -> None:
            registered["vf"] = vf

        def fake_capture_video(*_a: object, **_k: object) -> None:
            vf = registered.get("vf")
            if vf is not None:
                vf.get_array.return_value = frame.reshape(-1)
                vf.yres = frame.shape[0]
                vf.xres = frame.shape[1]

        frame_sync = MagicMock()
        frame_sync.set_video_frame.side_effect = fake_set_video_frame
        frame_sync.capture_video.side_effect = fake_capture_video
        receiver_instance = MagicMock()
        receiver_instance.frame_sync = frame_sync
        recv_mod.Receiver = MagicMock(return_value=receiver_instance)

        # framesync module still exists (FrameSync type lives here).
        framesync_mod.FrameSync = MagicMock

        # wrapper.ndi_recv — color/bandwidth enums used by the adapter.
        ndi_recv_mod.RecvBandwidth = MagicMock(highest="highest")
        ndi_recv_mod.RecvColorFormat = MagicMock(BGRX_BGRA="bgra")
        wrapper_mod.ndi_recv = ndi_recv_mod

        # VideoFrameSync factory.
        vf_mod.VideoFrameSync = MagicMock(return_value=MagicMock())

        sys.modules["cyndilib"] = cyn
        sys.modules["cyndilib.finder"] = finder_mod
        sys.modules["cyndilib.receiver"] = recv_mod
        sys.modules["cyndilib.framesync"] = framesync_mod
        sys.modules["cyndilib.video_frame"] = vf_mod
        sys.modules["cyndilib.wrapper"] = wrapper_mod
        sys.modules["cyndilib.wrapper.ndi_recv"] = ndi_recv_mod

    def _remove_mock_cyndilib(self) -> None:
        for mod in [
            "cyndilib",
            "cyndilib.finder",
            "cyndilib.receiver",
            "cyndilib.framesync",
            "cyndilib.video_frame",
            "cyndilib.wrapper",
            "cyndilib.wrapper.ndi_recv",
        ]:
            sys.modules.pop(mod, None)

        import autoptz.engine.pipeline.ingest as ingest_mod

        ingest_mod._NDI_AVAILABLE = None

    def test_open_success_when_source_visible(self) -> None:
        frame = np.full((_H, _W, 3), 55, dtype=np.uint8)
        self._install_mock_cyndilib(["LAPTOP (NDI CAMERA)"], frame)

        import autoptz.engine.pipeline.ingest as ingest_mod

        ingest_mod._NDI_AVAILABLE = True

        try:
            adapter = NDIAdapter("ndi-1", ndi_name="LAPTOP (NDI CAMERA)")
            ok = adapter._open()
            assert ok
            assert adapter._receiver is not None
            assert adapter._video_frame is not None
        finally:
            adapter._close()
            self._remove_mock_cyndilib()

    def test_open_fails_when_source_not_visible(self) -> None:
        frame = np.zeros((_H, _W, 3), dtype=np.uint8)
        self._install_mock_cyndilib(["OTHER SOURCE"], frame)

        import autoptz.engine.pipeline.ingest as ingest_mod

        ingest_mod._NDI_AVAILABLE = True

        try:
            # Short discover timeout so the "not found" poll loop returns fast.
            adapter = NDIAdapter("ndi-2", ndi_name="MISSING SOURCE", discover_timeout=0.1)
            ok = adapter._open()
            assert not ok
        finally:
            self._remove_mock_cyndilib()

    def test_ndi_unavailable_returns_error(self) -> None:
        import autoptz.engine.pipeline.ingest as ingest_mod

        ingest_mod._NDI_AVAILABLE = False

        adapter = NDIAdapter("ndi-3", ndi_name="ANY")
        ok = adapter._open()
        assert not ok
        assert adapter.status.last_error is not None

        ingest_mod._NDI_AVAILABLE = None


# ── SourceAdapter deliver (shm write) ─────────────────────────────────────────


class TestDeliverToShm:
    def test_deliver_writes_to_shm(self) -> None:
        from autoptz.engine.runtime.shm import ShmWriter

        H, W = 60, 80
        name = f"test_deliver_{id(self)}"
        frame = np.full((H, W, 3), 123, dtype=np.uint8)

        with ShmWriter(name, H, W) as writer:
            adapter = USBAdapter("cam-shm", source=0, shm_writer=writer)
            adapter._deliver(frame)
            assert adapter.status.frames_total == 1

    def test_deliver_resizes_frame_to_shm_dims(self) -> None:
        from autoptz.engine.runtime.shm import ShmWriter

        SHM_H, SHM_W = 60, 80
        name = f"test_resize_{id(self)}"
        big_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        writes: list[np.ndarray] = []
        original_push = ShmWriter.push

        def capturing_push(self_shm: ShmWriter, frame: np.ndarray, ts_ns: int | None = None) -> int:
            writes.append(frame.copy())
            return original_push(self_shm, frame, ts_ns)

        with (
            ShmWriter(name, SHM_H, SHM_W) as writer,
            patch.object(ShmWriter, "push", capturing_push),
        ):
            adapter = USBAdapter("cam-shm-resize", source=0, shm_writer=writer)
            adapter._deliver(big_frame)

        assert len(writes) == 1
        assert writes[0].shape == (SHM_H, SHM_W, 3)

    def test_deliver_without_shm_counts_frames(self) -> None:
        adapter = USBAdapter("cam-no-shm", source=0, shm_writer=None)
        frame = np.zeros((_H, _W, 3), dtype=np.uint8)
        adapter._deliver(frame)
        adapter._deliver(frame)
        assert adapter.status.frames_total == 2
