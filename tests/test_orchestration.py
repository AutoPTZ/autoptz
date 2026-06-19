"""P0 engine-orchestration tests: CameraWorker, Supervisor, EngineClient lifecycle.

All tests are headless (no display): they use ``QCoreApplication`` for the
EngineClient's Qt machinery and inject fakes for frame sources / workers so no
camera hardware, ML model, or GUI is required.
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

import PySide6  # noqa: F401


# ── one QCoreApplication for the whole module ─────────────────────────────────


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtCore import QCoreApplication
    existing = QCoreApplication.instance()
    if existing is not None:
        yield existing
        return
    app = QCoreApplication(sys.argv[:1])
    yield app


# ── helpers / fakes ───────────────────────────────────────────────────────────


def _cleanup_shm(name: str) -> None:
    """Unlink a possibly-leaked shm segment (and its ``__idx``) before a test."""
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


def _camera_config(camera_id: str = "cam-1234abcd5678", name: str = "Cam"):
    from autoptz.config.models import CameraConfig, SourceConfig
    return CameraConfig(
        id=camera_id,
        name=name,
        source=SourceConfig(type="usb", address="usb://0"),
    )


class FakeFrameSource:
    """Deterministic frame source that yields solid-colour BGR frames."""

    def __init__(self, h: int = 720, w: int = 1280, fail_open: bool = False) -> None:
        self.h = h
        self.w = w
        self._fail_open = fail_open
        self.opened = False
        self.closed = False
        self.reads = 0

    def open(self) -> bool:
        if self._fail_open:
            return False
        self.opened = True
        return True

    def read(self):
        self.reads += 1
        frame = np.full((self.h, self.w, 3), 123, dtype=np.uint8)
        return frame

    def close(self) -> None:
        self.closed = True


class TestAdapterFrameSourceFpsLimit:
    def test_direct_worker_source_paces_reads_to_target_fps(self, monkeypatch) -> None:
        from autoptz.engine import camera_worker
        from autoptz.engine.camera_worker import _AdapterFrameSource

        class Adapter:
            def __init__(self) -> None:
                self._target_fps = 30.0
                self.reads = 0

            def _read_frame(self):
                self.reads += 1
                return np.zeros((4, 4, 3), dtype=np.uint8)

            def set_target_fps(self, fps: float) -> None:
                self._target_fps = float(fps)

        now = [100.0]
        sleeps: list[float] = []

        def fake_monotonic() -> float:
            return now[0]

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        monkeypatch.setattr(camera_worker.time, "monotonic", fake_monotonic)
        monkeypatch.setattr(camera_worker.time, "sleep", fake_sleep)

        adapter = Adapter()
        source = _AdapterFrameSource(adapter)
        assert source.read() is not None

        source.set_target_fps(10.0)
        now[0] += 0.02
        assert source.read() is not None

        assert adapter.reads == 2
        assert sleeps == pytest.approx([0.08], abs=1e-6)


class FakeWorker:
    """Records lifecycle + command calls for supervisor routing tests."""

    def __init__(self, camera_id, config, on_telemetry):
        self.camera_id = camera_id
        self.config = config
        self.on_telemetry = on_telemetry
        self.shm_name = f"cam_{camera_id[:8]}_preview"
        self.started = False
        self.stopped = False
        self.tracking = None
        self.target = "unset"
        self.nudges = []
        self.configs = []
        self.inference_paused: list[bool] = []

    def start(self):
        self.started = True

    def stop(self, timeout: float = 5.0):
        self.stopped = True

    @property
    def is_running(self):
        return self.started and not self.stopped

    def enable_tracking(self, enabled):
        self.tracking = enabled

    def set_target(self, track_id):
        self.target = track_id

    def ptz_nudge(self, pan, tilt, zoom):
        self.nudges.append((pan, tilt, zoom))

    def update_config(self, config):
        self.configs.append(config)

    def set_inference_start_paused(self, paused: bool):
        self.inference_paused.append(bool(paused))


def _make_client(qapp):
    from autoptz.ui.engine_client import EngineClient
    return EngineClient()


def _make_supervisor(client, factory=None):
    from autoptz.engine.supervisor import Supervisor
    return Supervisor(client, store=None, worker_factory=factory)


class FakeDetector:
    """Detector that always returns one Detection regardless of frame."""

    ep = "FakeEP"

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        from autoptz.engine.pipeline.detect import BBox as DBBox
        from autoptz.engine.pipeline.detect import Detection
        self.calls += 1
        return [Detection(bbox=DBBox(10.0, 20.0, 110.0, 220.0), conf=0.9, class_id=0)]


class FakeTracker:
    """Tracker that echoes detections back as a single confirmed track."""

    def update(self, detections, frame, fps=30.0):
        from autoptz.engine.pipeline.track import Track, TrackState
        out = []
        for i, d in enumerate(detections):
            out.append(Track(
                track_id=i + 1, bbox=d.bbox, conf=d.conf,
                state=TrackState.CONFIRMED, age=1, hits=1, velocity=(0.0, 0.0),
            ))
        return out


def _install_fake_detect_stack(monkeypatch):
    """Patch _build_detect_stack to return a fake detector+tracker stack.

    Returns the FakeDetector so a test can assert detect() was called.  Also
    stubs the per-worker FACE build to None so the test never compiles the real
    (heavy, CoreML) insightface stack on the inference thread — production uses
    the shared inference pool for face, so this fallback build only happens in
    tests and would otherwise starve the inference loop within tight deadlines.
    """
    from autoptz.engine import camera_worker as cw
    det = FakeDetector()
    stack = cw._DetectStack(detector=det, tracker=FakeTracker(), ep=det.ep)
    monkeypatch.setattr(cw, "_build_detect_stack", lambda config: stack)
    monkeypatch.setattr(cw, "_build_face_stack", lambda *a, **k: None)
    return det


# ─────────────────────────────────────────────────────────────────────────────
# CameraWorker
# ─────────────────────────────────────────────────────────────────────────────


class TestCameraWorker:
    def test_writes_frames_to_shm_and_emits_telemetry(self, qapp) -> None:
        """Frames land in shm and telemetry reaches the callback.

        We inject a ShmWriter the *test* owns (and read it via a ShmReader
        created from the same region) so the segment's lifecycle is fully
        deterministic and immune to the pytest-session resource_tracker
        reclaiming a worker-owned segment mid-test.
        """
        import uuid

        from autoptz.engine.camera_worker import _PREVIEW_H, _PREVIEW_W, CameraWorker
        from autoptz.engine.runtime.shm import ShmReader, ShmWriter

        received = []
        lock = threading.Lock()

        def on_tel(msg):
            with lock:
                received.append(msg)

        cid = uuid.uuid4().hex[:12]
        shm_name = f"shmtest_{cid}"
        _cleanup_shm(shm_name)
        src = FakeFrameSource(h=_PREVIEW_H, w=_PREVIEW_W)

        writer = ShmWriter(shm_name, _PREVIEW_H, _PREVIEW_W)
        reader = ShmReader(shm_name, _PREVIEW_H, _PREVIEW_W)
        worker = CameraWorker(
            cid, _camera_config(cid), on_tel,
            frame_source=src, shm_writer=writer, telemetry_hz=50.0,
        )
        worker.start()
        try:
            frame = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                result = reader.latest()
                if result is not None:
                    _hdr, frame = result
                    break
                time.sleep(0.02)
            assert frame is not None, "no frame landed in shm"
            assert frame.shape == (_PREVIEW_H, _PREVIEW_W, 3)
            assert int(frame.mean()) == 123  # the fake's solid colour
        finally:
            worker.stop()
            reader.close()
            writer.close()  # test owns the segment → test unlinks it

        assert src.opened is True
        assert src.closed is True
        with lock:
            assert len(received) >= 1
            msg = received[0]
        assert msg.camera_id == cid
        # live-preview-only path: no model → empty tracks, still valid telemetry
        assert isinstance(msg.tracks, list)

    def test_fps_becomes_positive(self, qapp) -> None:
        from autoptz.engine.camera_worker import CameraWorker

        received = []
        lock = threading.Lock()

        def on_tel(m):
            with lock:
                received.append(m)

        worker = CameraWorker(
            "fpscam01abcd", _camera_config("fpscam01abcd"), on_tel,
            frame_source=FakeFrameSource(), telemetry_hz=50.0,
        )
        worker.start()
        try:
            deadline = time.monotonic() + 3.0
            best = 0.0
            while time.monotonic() < deadline:
                with lock:
                    if received:
                        best = max(best, received[-1].fps)
                if best > 0.0:
                    break
                time.sleep(0.05)
            assert best > 0.0, "fps never became positive with a live source"
        finally:
            worker.stop()

    def test_stop_is_idempotent_and_releases_source(self, qapp) -> None:
        from autoptz.engine.camera_worker import CameraWorker

        src = FakeFrameSource()
        worker = CameraWorker(
            "stopcam01abc", _camera_config("stopcam01abc"),
            lambda m: None, frame_source=src,
        )
        worker.start()
        time.sleep(0.1)
        worker.stop()
        worker.stop()  # idempotent — must not raise
        assert worker.is_running is False
        assert src.closed is True

    def test_failed_source_emits_error_telemetry_no_crash(self, qapp) -> None:
        from autoptz.engine.camera_worker import CameraWorker
        from autoptz.engine.runtime.messages import HealthState

        received = []
        lock = threading.Lock()

        def on_tel(m):
            with lock:
                received.append(m)

        worker = CameraWorker(
            "failcam01abc", _camera_config("failcam01abc"),
            on_tel, frame_source=FakeFrameSource(fail_open=True),
        )
        worker.start()
        try:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with lock:
                    if received:
                        break
                time.sleep(0.02)
            with lock:
                assert received, "no telemetry emitted on failed source"
                states = {m.health.state for m in received}
            assert HealthState.ERROR in states or HealthState.STOPPED in states
        finally:
            worker.stop()

    def test_commands_are_thread_safe_noops_before_start(self, qapp) -> None:
        from autoptz.engine.camera_worker import CameraWorker
        worker = CameraWorker("cmdcam01abcd", _camera_config("cmdcam01abcd"),
                              lambda m: None, frame_source=FakeFrameSource())
        # Queueing commands before start must not raise.
        worker.enable_tracking(True)
        worker.set_target(7)
        worker.ptz_nudge(0.5, -0.2, 0.1)
        worker.update_config(_camera_config("cmdcam01abcd"))
        worker.enroll_track(7, "id-1", "Alice")

    def test_enroll_track_sets_pending_and_immediate_label(self, qapp) -> None:
        """Click-to-assign queues a pending enrollment and labels the box at once."""
        from autoptz.engine.camera_worker import CameraWorker
        worker = CameraWorker("enrollcam01a", _camera_config("enrollcam01a"),
                              lambda m: None, frame_source=FakeFrameSource())
        worker._apply_command("enroll_track", (7, "id-123", "Alice", (0.4, 0.3)))
        # Awaiting a detected face to bind the embedding…
        assert worker._pending_enroll == {7: ("id-123", "Alice", (0.4, 0.3))}
        # …but the name shows on the box immediately (score 1.0 = manual).
        assert worker._track_identity[7] == ("id-123", "Alice", 1.0)

    def test_set_target_resets_reid_template(self, qapp) -> None:
        """Switching target drops the previous subject's appearance template."""
        from autoptz.engine.camera_worker import CameraWorker
        worker = CameraWorker("reidcam01abc", _camera_config("reidcam01abc"),
                              lambda m: None, frame_source=FakeFrameSource())

        class _FakeReid:
            def __init__(self) -> None:
                self.reset_calls = 0

            def reset(self) -> None:
                self.reset_calls += 1

        worker._reid = _FakeReid()
        worker._target_track_id = 1
        worker._apply_command("set_target", 2)
        assert worker._target_track_id == 2
        assert worker._reid.reset_calls == 1

    def test_detection_runs_with_engine_on_and_no_target(self, qapp, monkeypatch) -> None:
        """Detection + tracks must appear when the engine is on even with NO target.

        Regression: detection used to be gated on ``_tracking_enabled`` (which
        defaults from ``target.mode != "off"``), so with no per-camera target
        set, nothing ever detected and no boxes appeared.  Detection is now
        decoupled: a loaded detector produces tracks regardless of target.
        """
        from autoptz.engine.camera_worker import CameraWorker

        det = _install_fake_detect_stack(monkeypatch)

        received = []
        lock = threading.Lock()

        def on_tel(m):
            with lock:
                received.append(m)

        # Default config → target.mode == "off" → _tracking_enabled is False.
        worker = CameraWorker(
            "dettest01abc", _camera_config("dettest01abc"), on_tel,
            frame_source=FakeFrameSource(), telemetry_hz=50.0,
        )
        assert worker._tracking_enabled is False  # no target, tracking off
        worker.start()
        try:
            deadline = time.monotonic() + 3.0
            tracks = []
            while time.monotonic() < deadline:
                with lock:
                    for m in received:
                        if m.tracks:
                            tracks = m.tracks
                            break
                if tracks:
                    break
                time.sleep(0.02)
            assert det.calls > 0, "detector was never run despite engine being on"
            assert tracks, "no tracks emitted even though a detector is loaded"
            assert tracks[0].is_target is False  # detection without a follow-target
        finally:
            worker.stop()

    def test_telemetry_reports_resolution_and_dropped_frames(self, qapp) -> None:
        """Width/height come from the source frame; dropped counts read() misses."""
        from autoptz.engine.camera_worker import CameraWorker

        class FlakySource:
            """Alternates a real frame with a None (dropped) read."""

            def __init__(self):
                self.opened = self.closed = False
                self._n = 0

            def open(self):
                self.opened = True
                return True

            def read(self):
                self._n += 1
                if self._n % 2 == 0:
                    return None  # simulated decode miss / dropped frame
                return np.full((480, 640, 3), 50, dtype=np.uint8)

            def close(self):
                self.closed = True

        received = []
        lock = threading.Lock()

        def on_tel(m):
            with lock:
                received.append(m)

        worker = CameraWorker(
            "restest01abc", _camera_config("restest01abc"), on_tel,
            frame_source=FlakySource(), telemetry_hz=50.0,
        )
        worker.start()
        try:
            deadline = time.monotonic() + 3.0
            res_ok = drop_ok = False
            while time.monotonic() < deadline:
                with lock:
                    for m in received:
                        if m.width == 640 and m.height == 480:
                            res_ok = True
                        if m.dropped_frames > 0:
                            drop_ok = True
                if res_ok and drop_ok:
                    break
                time.sleep(0.02)
            assert res_ok, "telemetry never reported source resolution 640x480"
            assert drop_ok, "telemetry never counted a dropped frame"
        finally:
            worker.stop()

    def test_reclaims_leaked_shm_segment(self, qapp) -> None:
        """A stale segment from a crashed run is reclaimed, not fatal."""
        import uuid
        from multiprocessing.shared_memory import SharedMemory

        from autoptz.engine.camera_worker import _PREVIEW_H, _PREVIEW_W, CameraWorker
        from autoptz.engine.runtime.shm import ShmReader

        cid = uuid.uuid4().hex[:12]
        shm_name = f"cam_{cid[:8]}_preview"
        _cleanup_shm(shm_name)

        # Simulate a leaked main segment from a previous crashed process.
        leaked = SharedMemory(name=shm_name, create=True, size=64)
        leaked.close()  # leave it linked (orphaned)

        worker = CameraWorker(
            cid, _camera_config(cid), lambda m: None,
            frame_source=FakeFrameSource(h=_PREVIEW_H, w=_PREVIEW_W), telemetry_hz=50.0,
        )
        worker.start()
        try:
            # Worker should have reclaimed the orphan and produced a live region.
            reader = None
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    reader = ShmReader(shm_name, _PREVIEW_H, _PREVIEW_W)
                except Exception:
                    time.sleep(0.05)
                    continue
                if reader.latest() is not None:
                    break
                time.sleep(0.05)
            assert reader is not None, "worker did not reclaim leaked segment"
            reader.close()
        finally:
            worker.stop()
        _cleanup_shm(shm_name)

    def test_ptz_nudge_drives_injected_backend(self, qapp, monkeypatch) -> None:
        from autoptz.engine import camera_worker as cw
        from autoptz.engine.camera_worker import CameraWorker

        # Don't build real (heavy) models on the inference thread — this test is
        # about the command pump reaching the PTZ backend, not inference.  Stub
        # both per-worker builds so the inference loop starts (and drains
        # commands) immediately.  Production uses the shared pool for these.
        monkeypatch.setattr(cw, "_build_detect_stack", lambda config: None)
        monkeypatch.setattr(cw, "_build_face_stack", lambda *a, **k: None)

        class FakeBackend:
            def __init__(self):
                self.moves = []
                self.stopped = False

            def move_velocity(self, pan, tilt, zoom=0.0):
                self.moves.append((pan, tilt, zoom))

            def stop(self):
                self.stopped = True

        backend = FakeBackend()
        worker = CameraWorker(
            "ptzcam01abcd", _camera_config("ptzcam01abcd"), lambda m: None,
            frame_source=FakeFrameSource(), ptz_controller=backend,
        )
        worker.start()
        try:
            worker.ptz_nudge(0.7, 0.0, 0.0)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not backend.moves:
                time.sleep(0.02)
            assert backend.moves, "ptz nudge did not reach the backend"
            assert backend.moves[0][0] == pytest.approx(0.7)
        finally:
            worker.stop()
        assert backend.stopped is True


# ─────────────────────────────────────────────────────────────────────────────
# CameraWorker._track_error — region-aware aim point
# ─────────────────────────────────────────────────────────────────────────────


class TestTrackErrorAimRegion:
    """The vertical aim point must follow ``tracking.aim_region`` while the
    horizontal aim stays on the box centre."""

    @staticmethod
    def _worker(aim_region: str, aim_body_mode: str = "torso"):
        from autoptz.config.models import (
            CameraConfig,
            SourceConfig,
            TrackingConfig,
        )
        from autoptz.engine.camera_worker import CameraWorker
        cfg = CameraConfig(
            id="aimcam012345",
            name="Aim",
            source=SourceConfig(type="usb", address="usb://0"),
            tracking=TrackingConfig(aim_region=aim_region, aim_body_mode=aim_body_mode),
        )
        # No frame source / shm needed — we never start(); we call the pure method.
        return CameraWorker("aimcam012345", cfg, lambda _m: None)

    @staticmethod
    def _track(x1, y1, x2, y2):
        from autoptz.engine.runtime.messages import BBox, TrackInfo
        return TrackInfo(track_id=1, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))

    def test_full_body_aims_at_box_centre(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        w = self._worker("full_body")
        # Box fills the height, horizontally centred → error should be ~(0, 0).
        (ex, ey), _h = w._track_error(self._track(40, 0, 60, 100), frame)
        assert abs(ex) < 1e-6
        assert abs(ey) < 1e-6

    def test_tighter_region_aims_higher(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        trk = self._track(40, 0, 60, 100)
        # ey is up-positive, so a higher aim point (toward the face) → larger ey.
        ey_by_region = {}
        for region in ("face", "head_shoulders", "upper_body", "full_body"):
            (_ex, ey), _h = self._worker(region)._track_error(trk, frame)
            ey_by_region[region] = ey
        assert (ey_by_region["face"] > ey_by_region["head_shoulders"]
                > ey_by_region["upper_body"] > ey_by_region["full_body"])
        assert abs(ey_by_region["full_body"]) < 1e-6

    def test_horizontal_aim_is_box_centre_regardless_of_region(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Box shifted to the right half → ex must be > 0 (target right of centre)
        # and independent of the aim region.
        trk = self._track(60, 0, 100, 100)  # centre-x = 80 → ex = (80-50)/50 = 0.6
        for region in ("face", "upper_body", "full_body"):
            (ex, _ey), _h = self._worker(region)._track_error(trk, frame)
            assert abs(ex - 0.6) < 1e-6

    def test_arm_modes_share_framing_aim_without_pose(self) -> None:
        # Without pose (now=None) the aim CENTRE follows the framing region in
        # BOTH arm modes — "ignore arms" (torso) vs "include arms"
        # (full_silhouette) differ only in the zoom subject-height once pose is
        # available, never in the vertical aim of the bbox fallback.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        trk = self._track(40, 0, 60, 100)
        (_ex, torso_ey), torso_h = self._worker("face", "torso")._track_error(trk, frame)
        (_ex, sil_ey), sil_h = self._worker("face", "full_silhouette")._track_error(trk, frame)
        assert torso_ey > 0.0                     # "face" aims high in both modes
        assert sil_ey == pytest.approx(torso_ey)
        assert sil_h == pytest.approx(torso_h)    # same bbox height when pose absent

    def test_pose_aim_sits_on_torso_and_sets_source(self) -> None:
        # With a confident pose, the aim CENTRE is the pose torso anchor (not the
        # box) and ``aim_source == "pose"`` — this is what keeps the on-screen
        # circle on the skeleton and following the body, not the bounding box.
        from autoptz.engine.pipeline.framing import Keypoint

        class _FakePose:
            available = True

            def estimate(self, _frame, _bbox):
                kps = [Keypoint(0.0, 0.0, 0.0)] * 17
                kps[5] = Keypoint(40.0, 30.0, 0.9)   # left shoulder
                kps[6] = Keypoint(60.0, 30.0, 0.9)   # right shoulder
                kps[11] = Keypoint(42.0, 70.0, 0.9)  # left hip
                kps[12] = Keypoint(58.0, 70.0, 0.9)  # right hip
                return kps

        w = self._worker("upper_body", "torso")
        w._pose = _FakePose()
        w._pose_probed = True  # bypass the lazy model build
        trk = self._track(20, 0, 80, 100)        # box centre (50,50) ≠ torso anchor
        trk.is_target = True
        w._target_track_id = trk.track_id
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        (ex, ey), _h = w._track_error(trk, frame, now=123.0)
        assert trk.aim_source == "pose"
        # upper_body bias → shoulder midpoint (50, 30), not the box centre (50, 50).
        assert trk.aim_x == pytest.approx(50.0, abs=1.0)
        assert trk.aim_y == pytest.approx(30.0, abs=1.0)
        assert ey > 0.0  # torso anchor sits above frame centre

    def test_target_aim_fields_are_annotated(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        trk = self._track(40, 0, 60, 100)
        trk.is_target = True
        self._worker("upper_body")._track_error(trk, frame)
        assert trk.aim_x == pytest.approx(50.0)
        assert trk.aim_y == pytest.approx(38.0)
        assert trk.aim_source == "bbox"


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor — command routing & lifecycle
# ─────────────────────────────────────────────────────────────────────────────


class TestSupervisorRouting:
    def test_start_spawns_worker_per_existing_camera(self, qapp) -> None:
        client = _make_client(qapp)
        client.addCamera("usb://0", "A")
        client.addCamera("usb://1", "B")
        client.drain_commands()  # clear add-cmds so they don't double-spawn

        sup = _make_supervisor(client, factory=FakeWorker)
        sup.start()
        try:
            assert sup.worker_count == 2
            assert sup.is_running is True
        finally:
            sup.stop()
        assert sup.worker_count == 0

    def test_start_stop_idempotent(self, qapp) -> None:
        client = _make_client(qapp)
        sup = _make_supervisor(client, factory=FakeWorker)
        sup.start()
        sup.start()  # idempotent
        assert sup.is_running is True
        sup.stop()
        sup.stop()  # idempotent
        assert sup.is_running is False

    def test_add_camera_cmd_spawns_worker(self, qapp) -> None:
        client = _make_client(qapp)
        sup = _make_supervisor(client, factory=FakeWorker)
        sup.start()
        try:
            cid = client.addCamera("usb://0", "Late")
            sup.tick()  # drain + route the AddCameraCmd
            assert sup.has_worker(cid) is True
        finally:
            sup.stop()

    def test_remove_camera_cmd_stops_and_drops_worker(self, qapp) -> None:
        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()

        captured = {}

        def factory(camera_id, config, on_tel):
            w = FakeWorker(camera_id, config, on_tel)
            captured[camera_id] = w
            return w

        sup = _make_supervisor(client, factory=factory)
        sup.start()
        try:
            assert sup.has_worker(cid)
            client.removeCamera(cid)
            sup.tick()
            assert sup.has_worker(cid) is False
            assert captured[cid].stopped is True
        finally:
            sup.stop()

    def test_enable_tracking_routes_to_worker(self, qapp) -> None:
        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()
        captured = {}
        sup = _make_supervisor(
            client,
            factory=lambda c, cfg, t: captured.setdefault(c, FakeWorker(c, cfg, t)),
        )
        sup.start()
        try:
            client.enableTracking(cid, True)
            sup.tick()
            assert captured[cid].tracking is True
        finally:
            sup.stop()

    def test_set_target_routes_to_worker(self, qapp) -> None:
        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()
        captured = {}
        sup = _make_supervisor(
            client,
            factory=lambda c, cfg, t: captured.setdefault(c, FakeWorker(c, cfg, t)),
        )
        sup.start()
        try:
            client.setTarget(cid, 99)
            sup.tick()
            assert captured[cid].target == 99
            client.clearTarget(cid)
            sup.tick()
            assert captured[cid].target is None
        finally:
            sup.stop()

    def test_ptz_nudge_routes_to_worker(self, qapp) -> None:
        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()
        captured = {}
        sup = _make_supervisor(
            client,
            factory=lambda c, cfg, t: captured.setdefault(c, FakeWorker(c, cfg, t)),
        )
        sup.start()
        try:
            client.ptzNudge(cid, 0.5, -0.3, 0.2)
            sup.tick()
            assert captured[cid].nudges == [(0.5, -0.3, 0.2)]
        finally:
            sup.stop()

    def test_update_config_routes_to_worker(self, qapp) -> None:
        import json
        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()
        captured = {}
        sup = _make_supervisor(
            client,
            factory=lambda c, cfg, t: captured.setdefault(c, FakeWorker(c, cfg, t)),
        )
        sup.start()
        try:
            cfg = client.getCameraConfig(cid)
            cfg["name"] = "Renamed"
            client.updateCameraConfig(cid, json.dumps(cfg))
            sup.tick()
            assert len(captured[cid].configs) == 1
            assert captured[cid].configs[0].name == "Renamed"
        finally:
            sup.stop()

    def test_tick_when_stopped_is_noop(self, qapp) -> None:
        client = _make_client(qapp)
        client.addCamera("usb://0", "X")  # leaves an AddCameraCmd queued
        sup = _make_supervisor(client, factory=FakeWorker)
        sup.tick()  # not running → must not route or raise
        assert sup.worker_count == 0

    def test_telemetry_callback_reaches_push_telemetry(self, qapp) -> None:
        """A worker's on_telemetry callback is wired to client.push_telemetry."""
        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()

        captured = {}
        sup = _make_supervisor(
            client,
            factory=lambda c, cfg, t: captured.setdefault(c, FakeWorker(c, cfg, t)),
        )
        sup.start()
        try:
            from autoptz.engine.runtime.messages import TelemetryMsg
            worker = captured[cid]
            # Simulate the worker emitting telemetry from its own context.
            worker.on_telemetry(TelemetryMsg(camera_id=cid, seq=3, fps=24.0))
            rec = client.get_camera(cid)
            assert rec is not None
            assert rec.fps == pytest.approx(24.0)
        finally:
            sup.stop()

    def test_active_ep_is_nonempty_label(self, qapp) -> None:
        client = _make_client(qapp)
        sup = _make_supervisor(client, factory=FakeWorker)
        # EP label should be a short string like "CoreML" or "CPU".
        label = sup.active_ep
        assert isinstance(label, str)
        assert "ExecutionProvider" not in label

    def test_provider_attach_requested_on_spawn(self, qapp) -> None:
        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "X")
        client.drain_commands()

        attaches = []
        client.providerAttachRequested.connect(
            lambda c, shm, w, h: attaches.append((c, shm, w, h))
        )
        sup = _make_supervisor(client, factory=FakeWorker)
        sup.start()
        try:
            assert any(a[0] == cid for a in attaches)
            # shm name must match the UI provider's convention
            shm_for_cid = next(a[1] for a in attaches if a[0] == cid)
            assert shm_for_cid == f"cam_{cid[:8]}_preview"
        finally:
            sup.stop()

    def test_adaptive_startup_concurrency_rules(self, monkeypatch) -> None:
        from autoptz.engine import supervisor as sup_mod
        from autoptz.engine.runtime import diagnostics
        from autoptz.engine.supervisor import Supervisor

        monkeypatch.setattr(sup_mod.os, "cpu_count", lambda: 10)
        monkeypatch.setattr(
            diagnostics, "system_metrics",
            lambda: {"available": True, "cpu_percent": 20.0, "mem_percent": 40.0},
        )
        assert Supervisor._adaptive_startup_concurrency() == 4

        monkeypatch.setattr(sup_mod.os, "cpu_count", lambda: 6)
        monkeypatch.setattr(
            diagnostics, "system_metrics",
            lambda: {"available": True, "cpu_percent": 60.0, "mem_percent": 80.0},
        )
        assert Supervisor._adaptive_startup_concurrency() == 2

        monkeypatch.setattr(
            diagnostics, "system_metrics",
            lambda: {"available": False},
        )
        assert Supervisor._adaptive_startup_concurrency() == 1

    def test_staged_start_reports_progress_and_spawns(self, qapp, monkeypatch) -> None:
        client = _make_client(qapp)
        client.addCamera("usb://0", "A")
        client.addCamera("usb://1", "B")
        client.drain_commands()
        sup = _make_supervisor(client, factory=FakeWorker)
        monkeypatch.setattr(sup, "_ensure_inference_pool", lambda: None)
        monkeypatch.setattr(sup, "_adaptive_startup_concurrency", lambda: 2)
        monkeypatch.setattr(sup, "_warm_reid", lambda: None)
        events = []

        sup.start(staged=True, progress=lambda **kw: events.append(kw))
        try:
            deadline = time.monotonic() + 2.0
            while sup.worker_count < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert sup.worker_count == 2
            phases = [e.get("phase") for e in events]
            assert "Opening cameras" in phases
            assert "Ready" in phases
        finally:
            sup.stop()

    def test_staged_start_spawns_preview_before_detector_warmup(self, qapp, monkeypatch) -> None:
        client = _make_client(qapp)
        client.addCamera("usb://0", "A")
        client.addCamera("usb://1", "B")
        client.drain_commands()
        events: list[str] = []

        class OrderedWorker(FakeWorker):
            def start(self):
                events.append(f"start:{self.camera_id}")
                super().start()

        class WarmPool:
            detector_ep = "FakeEP"

            def detector(self):
                events.append("detector")
                return None

            def face(self):
                events.append("face")
                return None

            def pose(self):
                events.append("pose")
                return None

        sup = _make_supervisor(client, factory=OrderedWorker)
        monkeypatch.setattr(sup, "_ensure_inference_pool", lambda: WarmPool())
        monkeypatch.setattr(sup, "_adaptive_startup_concurrency", lambda: 2)
        monkeypatch.setattr(sup, "_warm_reid", lambda: events.append("reid"))
        monkeypatch.setattr("autoptz.engine.supervisor.time.sleep", lambda _s: None)

        sup.start(staged=True, progress=lambda **_kw: None)
        try:
            deadline = time.monotonic() + 2.0
            while "detector" not in events and time.monotonic() < deadline:
                time.sleep(0.01)
            assert events[:2] == [f"start:{client.cameraModel.camera_ids()[0]}",
                                  f"start:{client.cameraModel.camera_ids()[1]}"]
            assert events[2] == "detector"
            workers = list(sup._workers.values())
            assert workers and all(w.inference_paused[0] is True for w in workers)
            deadline = time.monotonic() + 2.0
            while (
                not all(w.inference_paused and w.inference_paused[-1] is False for w in workers)
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert all(w.inference_paused[-1] is False for w in workers)
        finally:
            sup.stop()


class TestTrackingStability:
    def test_pooled_detector_respects_worker_detect_interval(self, qapp) -> None:
        from autoptz.config.models import TrackingConfig
        from autoptz.engine import camera_worker as cw
        from autoptz.engine.camera_worker import CameraWorker

        cfg = _camera_config("pooled123456").model_copy(
            update={"tracking": TrackingConfig(detect_interval=3)},
        )
        worker = CameraWorker("pooled123456", cfg, lambda _m: None)
        det = FakeDetector()
        worker._detect = cw._DetectStack(detector=det, tracker=FakeTracker(), ep=det.ep)
        worker._pooled_detector = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(4):
            worker._maybe_track(frame)

        assert det.calls == 2  # frames 1 and 4; frames 2/3 skipped by worker

    def test_pose_overlay_expires_when_stale(self, qapp) -> None:
        from autoptz.engine.camera_worker import CameraWorker
        from autoptz.engine.pipeline.framing import Keypoint

        worker = CameraWorker("pose12345678", _camera_config("pose12345678"), lambda _m: None)
        worker._target_track_id = 1
        worker._pose_kp_track_id = 1
        worker._pose_keypoints = [Keypoint(1.0, 2.0, 0.9)] * 17
        worker._last_pose_overlay_t = 0.0

        assert worker._pose_overlay() == []

    def test_pose_overlay_publishes_once_per_inference_frame(self, qapp) -> None:
        from autoptz.engine.camera_worker import CameraWorker
        from autoptz.engine.pipeline.framing import Keypoint

        worker = CameraWorker("poseonce1234", _camera_config("poseonce1234"), lambda _m: None)
        worker._target_track_id = 1
        worker._pose_kp_track_id = 1
        worker._pose_keypoints = [Keypoint(1.0, 2.0, 0.9)] * 17
        worker._last_pose_overlay_t = time.monotonic()
        worker._last_pose_overlay_frame_id = 9
        worker._last_tracks_frame_id = 9

        assert len(worker._pose_overlay()) == 17
        assert worker._pose_overlay() == []

    def test_stable_mode_requires_repeated_reid_confirmation(self, qapp) -> None:
        from autoptz.config.models import TrackingConfig
        from autoptz.engine.camera_worker import CameraWorker

        stable_cfg = _camera_config("stable123456").model_copy(
            update={"tracking": TrackingConfig(tracking_mode="stable")},
        )
        stable = CameraWorker("stable123456", stable_cfg, lambda _m: None)
        assert stable._reid_recovery_confirmed(2) is False
        assert stable._reid_recovery_confirmed(2) is True

        resp_cfg = _camera_config("resp12345678").model_copy(
            update={"tracking": TrackingConfig(tracking_mode="responsive")},
        )
        responsive = CameraWorker("resp12345678", resp_cfg, lambda _m: None)
        assert responsive._reid_recovery_confirmed(2) is True


# ─────────────────────────────────────────────────────────────────────────────
# EngineClient — engine lifecycle (FROZEN contract)
# ─────────────────────────────────────────────────────────────────────────────


class TestEngineLifecycleApi:
    def test_default_stopped(self, qapp) -> None:
        c = _make_client(qapp)
        assert c.engineRunning is False
        assert c.engineEp == ""

    def test_frozen_api_surface_present(self, qapp) -> None:
        from PySide6.QtCore import QMetaMethod  # noqa: F401  (ensures Qt meta avail)
        c = _make_client(qapp)
        # Properties exist on the meta-object and read as the right types.
        mo = c.metaObject()
        prop_names = {mo.property(i).name() for i in range(mo.propertyCount())}
        assert "engineRunning" in prop_names
        assert "engineEp" in prop_names
        assert isinstance(c.engineRunning, bool)
        assert isinstance(c.engineEp, str)
        # Slots / signals
        assert hasattr(c, "startEngine")
        assert hasattr(c, "stopEngine")
        assert hasattr(c, "engineStateChanged")
        # Supervisor contract preserved
        assert hasattr(c, "push_telemetry")
        assert hasattr(c, "drain_commands")
        assert hasattr(c, "set_supervisor")

    def test_start_engine_idempotent_with_injected_supervisor(self, qapp) -> None:
        c = _make_client(qapp)
        sup = _make_supervisor(c, factory=FakeWorker)
        c.set_supervisor(sup)

        states = []
        c.engineStateChanged.connect(lambda: states.append(c.engineRunning))

        c.startEngine()
        assert c.engineRunning is True
        assert sup.is_running is True
        c.startEngine()  # idempotent — no second start
        assert states.count(True) == 1

    def test_stop_engine_idempotent(self, qapp) -> None:
        c = _make_client(qapp)
        sup = _make_supervisor(c, factory=FakeWorker)
        c.set_supervisor(sup)
        c.startEngine()
        c.stopEngine()
        assert c.engineRunning is False
        assert sup.is_running is False
        c.stopEngine()  # idempotent
        assert c.engineRunning is False

    def test_start_engine_uses_factory(self, qapp) -> None:
        c = _make_client(qapp)
        built = []

        def factory(client):
            sup = _make_supervisor(client, factory=FakeWorker)
            built.append(sup)
            return sup

        c.set_supervisor_factory(factory)
        c.startEngine()
        try:
            assert len(built) == 1
            assert c.engineRunning is True
        finally:
            c.stopEngine()

    def test_engine_ep_set_when_running(self, qapp) -> None:
        c = _make_client(qapp)
        c.set_supervisor(_make_supervisor(c, factory=FakeWorker))
        c.startEngine()
        try:
            # active_ep returns a short label; may be "CPU"/"CoreML"/etc.
            assert isinstance(c.engineEp, str)
            assert "ExecutionProvider" not in c.engineEp
        finally:
            c.stopEngine()
        assert c.engineEp == ""

    def test_engine_state_changed_emitted(self, qapp) -> None:
        c = _make_client(qapp)
        c.set_supervisor(_make_supervisor(c, factory=FakeWorker))
        fired = []
        c.engineStateChanged.connect(lambda: fired.append(True))
        c.startEngine()
        c.stopEngine()
        assert len(fired) == 2  # one on start, one on stop


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safety: off-thread telemetry is marshalled, not applied inline
# ─────────────────────────────────────────────────────────────────────────────


class TestTelemetryThreadSafety:
    def test_push_from_worker_thread_marshals_to_owning_thread(self, qapp) -> None:
        from autoptz.engine.runtime.messages import TelemetryMsg

        c = _make_client(qapp)
        cid = c.addCamera("usb://0", "X")
        c.drain_commands()

        def worker_push():
            c.push_telemetry(TelemetryMsg(camera_id=cid, seq=1, fps=42.0))

        t = threading.Thread(target=worker_push)
        t.start()
        t.join()

        # Off-thread push is queued: model not updated until the event loop runs.
        assert c.get_camera(cid).fps == pytest.approx(0.0)

        qapp.processEvents()
        # After processing the queued event, the model reflects the update.
        deadline = time.monotonic() + 2.0
        while c.get_camera(cid).fps == 0.0 and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert c.get_camera(cid).fps == pytest.approx(42.0)

    def test_push_from_owning_thread_applies_synchronously(self, qapp) -> None:
        from autoptz.engine.runtime.messages import TelemetryMsg
        c = _make_client(qapp)
        cid = c.addCamera("usb://0", "X")
        c.drain_commands()
        c.push_telemetry(TelemetryMsg(camera_id=cid, seq=1, fps=30.0))
        # Same-thread push applies immediately (no event-loop spin needed).
        assert c.get_camera(cid).fps == pytest.approx(30.0)
