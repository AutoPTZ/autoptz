"""Tests for the blank-navy-preview fix + macOS camera enumeration contract.

The core regression: the provider must be *self-healing* — an ``attach`` that
runs before the writer's segment exists must still serve real frames once the
writer appears, with no re-attach and no manual retry.

All tests are headless and mock PyObjC; the shm tests use real (small) shared
memory segments that are explicitly cleaned up.
"""

from __future__ import annotations

import sys
import time
import types
import uuid

import numpy as np
import PySide6  # noqa: F401


def _cleanup_shm(name: str) -> None:
    from multiprocessing.shared_memory import SharedMemory

    for n in (name, f"{name}__idx"):
        try:
            s = SharedMemory(name=n, create=False)
            s.close()
            s.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass


# ── bgr_to_qimage: color correctness (Format_BGR888, no reversal copy) ─────────


class TestBgrToQImage:
    """The preview converter must render BGR frames with correct colors — the
    Format_BGR888 fast path must not swap the red/blue channels."""

    def test_red_and_blue_channels_not_swapped(self) -> None:
        from autoptz.ui.frames import bgr_to_qimage

        # Pure red in BGR is (B=0, G=0, R=255) in the third channel.
        red_bgr = np.zeros((4, 6, 3), dtype=np.uint8)
        red_bgr[:, :, 2] = 255
        red_img = bgr_to_qimage(red_bgr)
        assert (red_img.width(), red_img.height()) == (6, 4)
        rc = red_img.pixelColor(0, 0)
        assert (rc.red(), rc.green(), rc.blue()) == (255, 0, 0)

        # Pure blue in BGR is the first channel.
        blue_bgr = np.zeros((4, 6, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255
        bc = bgr_to_qimage(blue_bgr).pixelColor(3, 2)
        assert (bc.red(), bc.green(), bc.blue()) == (0, 0, 255)

    def test_non_contiguous_input_is_handled(self) -> None:
        from autoptz.ui.frames import bgr_to_qimage

        # A non-contiguous (sliced) view must still render correctly.
        base = np.zeros((4, 12, 3), dtype=np.uint8)
        base[:, :, 1] = 255  # green
        view = base[:, ::2, :]  # non-contiguous
        assert not view.flags["C_CONTIGUOUS"]
        gc = bgr_to_qimage(view).pixelColor(0, 0)
        assert (gc.red(), gc.green(), gc.blue()) == (0, 255, 0)


# ── The self-healing provider regression ──────────────────────────────────────


class TestSelfHealingFrameSource:
    """The blank-preview regression, now guarding ``frames.ShmFrameSource``."""

    def test_serves_real_frame_after_writer_appears_post_attach(self) -> None:
        """attach() BEFORE the writer exists → real frame served once it does.

        This is the exact ordering that produced the navy-screen bug: the source
        attaches, the writer's segment does not exist yet, and a non-self-healing
        implementation would store no reader and never retry.  ShmFrameSource
        returns None until the writer appears, then serves the real frame.
        """
        from autoptz.engine.runtime.shm import ShmWriter
        from autoptz.ui.frames import ShmFrameSource

        h, w = 16, 24
        cid = "cam-" + uuid.uuid4().hex[:8]
        shm_name = f"pvtest_{uuid.uuid4().hex[:8]}"
        _cleanup_shm(shm_name)

        src = ShmFrameSource()
        # 1. Attach BEFORE any writer exists.  Must not raise; no frame yet.
        src.attach(cid, shm_name, h, w)
        assert src.latest_qimage(cid) is None

        writer = None
        try:
            # 2. NOW the writer appears (the worker creating its shm after the
            #    queued attach already ran).
            writer = ShmWriter(shm_name, h, w)
            frame = np.full((h, w, 3), 200, dtype=np.uint8)  # BGR solid colour
            writer.push(frame)

            # 3. The source lazily opens the reader and serves the real frame.
            real = None
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                img = src.latest_qimage(cid)
                if img is not None and img.width() == w and img.height() == h:
                    real = img
                    break
                time.sleep(0.02)
            assert real is not None, "frame source never served the real frame"
            px = real.pixelColor(w // 2, h // 2)
            assert px.red() == 200 and px.green() == 200 and px.blue() == 200
        finally:
            src.detach(cid)
            if writer is not None:
                writer.close()
            _cleanup_shm(shm_name)

    def test_attach_before_writer_does_not_raise(self) -> None:
        from autoptz.ui.frames import ShmFrameSource

        src = ShmFrameSource()
        # Intent recorded for a segment that will never exist; must be a no-op.
        src.attach("cam-x", "never_exists_segment", 8, 8)
        src.detach("cam-x")
        src.detach_all()  # also must not raise

    def test_detach_clears_intent_so_no_stale_open(self) -> None:
        from autoptz.engine.runtime.shm import ShmWriter
        from autoptz.ui.frames import ShmFrameSource

        h, w = 8, 8
        shm_name = f"pvdetach_{uuid.uuid4().hex[:8]}"
        _cleanup_shm(shm_name)
        src = ShmFrameSource()
        src.attach("cam-d", shm_name, h, w)
        src.detach("cam-d")  # intent cleared before any writer

        writer = None
        try:
            writer = ShmWriter(shm_name, h, w)
            writer.push(np.full((h, w, 3), 50, dtype=np.uint8))
            # Detached camera → no intent → None, never the real frame.
            assert src.latest_qimage("cam-d") is None
        finally:
            if writer is not None:
                writer.close()
            _cleanup_shm(shm_name)


# ── CameraWorker creates the shm writer eagerly in start() ─────────────────────


class TestWorkerEagerShm:
    def test_shm_segment_exists_right_after_start(self) -> None:
        """The writer's segment must be openable as soon as start() returns.

        This is what lets the supervisor emit the provider attach after start()
        and have the reader open on the first request.
        """
        from autoptz.engine.camera_worker import _PREVIEW_H, _PREVIEW_W, CameraWorker
        from autoptz.engine.runtime.shm import ShmReader

        cid = uuid.uuid4().hex[:12]
        shm_name = f"cam_{cid[:8]}_preview"
        _cleanup_shm(shm_name)

        class _Src:
            def open(self):
                return True

            def read(self):
                return np.full((_PREVIEW_H, _PREVIEW_W, 3), 7, dtype=np.uint8)

            def close(self):
                pass

        from autoptz.config.models import CameraConfig, SourceConfig

        cfg = CameraConfig(id=cid, name="C", source=SourceConfig(type="usb", address="usb://0"))
        worker = CameraWorker(cid, cfg, lambda m: None, frame_source=_Src())
        worker.start()
        try:
            # Immediately after start() — the segment must already exist.
            reader = ShmReader(shm_name, _PREVIEW_H, _PREVIEW_W)
            reader.close()
        finally:
            worker.stop()
        _cleanup_shm(shm_name)


# ── macOS camera enumeration (FROZEN contract) ─────────────────────────────────


class TestEnumerateCameras:
    def test_fallback_when_avfoundation_absent(self, monkeypatch) -> None:
        import autoptz.engine.discovery.usb as usb

        # Force the macOS branch but make AVFoundation import fail.
        monkeypatch.setattr(usb.platform, "system", lambda: "Darwin")
        monkeypatch.setitem(sys.modules, "AVFoundation", None)
        # The fallback now probes REAL openable indices; mock the probe so the
        # test is deterministic (and never touches the camera in the sandbox).
        monkeypatch.setattr(usb, "_probe_indices", lambda *a, **k: {0, 2})

        cams = usb.enumerate_cameras()
        assert isinstance(cams, list)
        # Only the probed-openable indices are returned (no phantom 0-3).
        assert [c["index"] for c in cams] == [0, 2]
        for cam in cams:
            assert set(cam.keys()) == {
                "name",
                "unique_id",
                "index",
                "is_continuity",
                "source_label",
            }
            assert cam["unique_id"] is None
            assert cam["is_continuity"] is False
            assert cam["source_label"] == "USB"
            assert isinstance(cam["index"], int)

    def test_fallback_returns_empty_when_no_openable_devices(self, monkeypatch) -> None:
        import autoptz.engine.discovery.usb as usb

        monkeypatch.setattr(usb.platform, "system", lambda: "Darwin")
        monkeypatch.setitem(sys.modules, "AVFoundation", None)
        # No openable indices → empty list, NOT phantom Camera 0-3.
        monkeypatch.setattr(usb, "_probe_indices", lambda *a, **k: set())
        assert usb.enumerate_cameras() == []

    def test_avfoundation_enumeration_maps_fields(self, monkeypatch) -> None:
        import autoptz.engine.discovery.usb as usb

        monkeypatch.setattr(usb.platform, "system", lambda: "Darwin")

        # Build a fake AVFoundation module with two devices.
        fake = types.ModuleType("AVFoundation")
        fake.AVCaptureDeviceTypeBuiltInWideAngleCamera = "builtin"
        fake.AVCaptureDeviceTypeExternal = "external"
        fake.AVCaptureDeviceTypeContinuityCamera = "continuity"
        fake.AVMediaTypeVideo = "vid"
        fake.AVCaptureDevicePositionUnspecified = 0

        class _Dev:
            def __init__(self, name, uid, dtype):
                self._name, self._uid, self._dtype = name, uid, dtype

            def localizedName(self):
                return self._name

            def uniqueID(self):
                return self._uid

            def deviceType(self):
                return self._dtype

        devices = [
            _Dev("FaceTime HD Camera", "0xABC", "builtin"),
            _Dev("Steven's iPhone", "0xDEF", "continuity"),
        ]

        class _Session:
            @staticmethod
            def discoverySessionWithDeviceTypes_mediaType_position_(types_, media, pos):
                return _Session()

            def devices(self):
                return devices

        fake.AVCaptureDeviceDiscoverySession = _Session
        monkeypatch.setitem(sys.modules, "AVFoundation", fake)

        cams = usb.enumerate_cameras()
        assert len(cams) == 2
        assert cams[0] == {
            "name": "FaceTime HD Camera",
            "unique_id": "0xABC",
            "index": 0,
            "is_continuity": False,
            "source_label": "Built-in",
        }
        assert cams[1]["name"] == "Steven's iPhone"
        assert cams[1]["unique_id"] == "0xDEF"
        assert cams[1]["index"] == 1
        assert cams[1]["is_continuity"] is True
        assert cams[1]["source_label"] == "Continuity Camera"

    def test_enumeration_never_raises(self, monkeypatch) -> None:
        import autoptz.engine.discovery.usb as usb

        monkeypatch.setattr(usb.platform, "system", lambda: "Darwin")
        # Deterministic probe fallback (don't touch real hardware in sandbox).
        monkeypatch.setattr(usb, "_probe_indices", lambda *a, **k: {1})

        # AVFoundation present but its discovery session throws.
        fake = types.ModuleType("AVFoundation")
        fake.AVCaptureDeviceTypeBuiltInWideAngleCamera = "builtin"
        fake.AVMediaTypeVideo = "vid"
        fake.AVCaptureDevicePositionUnspecified = 0

        class _Session:
            @staticmethod
            def discoverySessionWithDeviceTypes_mediaType_position_(*a):
                raise RuntimeError("boom")

        fake.AVCaptureDeviceDiscoverySession = _Session
        monkeypatch.setitem(sys.modules, "AVFoundation", fake)

        cams = usb.enumerate_cameras()  # must fall back, not raise
        assert isinstance(cams, list)
        assert [c["index"] for c in cams] == [1]


# ── USB device resolution honours unique_id when present ───────────────────────


class TestUsbDeviceResolution:
    def test_resolves_index_by_unique_id(self, monkeypatch) -> None:
        from autoptz.engine import camera_worker

        monkeypatch.setattr(
            "autoptz.engine.discovery.usb.enumerate_cameras",
            lambda: [
                {"name": "A", "unique_id": "uid-A", "index": 0, "is_continuity": False},
                {"name": "B", "unique_id": "uid-B", "index": 1, "is_continuity": False},
            ],
        )

        source = types.SimpleNamespace(unique_id="uid-B", address="usb://0")
        assert camera_worker._resolve_usb_device(source) == 1

    def test_falls_back_to_address_without_unique_id(self) -> None:
        from autoptz.engine import camera_worker

        source = types.SimpleNamespace(address="usb://3")  # no unique_id attr
        assert camera_worker._resolve_usb_device(source) == 3

    def test_falls_back_when_unique_id_not_found(self, monkeypatch) -> None:
        from autoptz.engine import camera_worker

        monkeypatch.setattr(
            "autoptz.engine.discovery.usb.enumerate_cameras",
            lambda: [
                {"name": "A", "unique_id": "uid-A", "index": 0, "is_continuity": False},
            ],
        )
        source = types.SimpleNamespace(unique_id="uid-MISSING", address="usb://2")
        assert camera_worker._resolve_usb_device(source) == 2

    def test_macos_opencv_fallback_matches_sorted_video_plus_muxed(self, monkeypatch) -> None:
        import autoptz.engine.pipeline.ingest as ingest

        monkeypatch.setattr(ingest.platform, "system", lambda: "Darwin")
        fake = types.ModuleType("AVFoundation")
        fake.AVMediaTypeVideo = "video"
        fake.AVMediaTypeMuxed = "muxed"

        class _Dev:
            def __init__(self, uid):
                self._uid = uid

            def uniqueID(self):
                return self._uid

        video = [_Dev("uid-z"), _Dev("uid-a")]
        muxed = [_Dev("uid-m")]

        class _CaptureDevice:
            @staticmethod
            def devicesWithMediaType_(media):
                return video if media == "video" else muxed

        fake.AVCaptureDevice = _CaptureDevice
        monkeypatch.setitem(sys.modules, "AVFoundation", fake)

        assert ingest._macos_index_for_unique_id("uid-a") == 0
        assert ingest._macos_index_for_unique_id("uid-m") == 1
        assert ingest._macos_index_for_unique_id("uid-z") == 2
