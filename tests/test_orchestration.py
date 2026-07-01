"""Engine-orchestration tests: CameraWorker, Supervisor, EngineClient lifecycle.

All tests are headless (no display): they use ``QCoreApplication`` for the
EngineClient's Qt machinery and inject fakes for frame sources / workers so no
camera hardware, ML model, or GUI is required.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import PySide6  # noqa: F401
import pytest

# ── helpers / fakes ───────────────────────────────────────────────────────────


def _cleanup_shm(name: str) -> None:
    """Unlink a possibly-leaked shm segment (and its ``__idx``) before a test."""
    from autoptz.engine.runtime.shm import unlink_shared_memory_pair

    unlink_shared_memory_pair(name)


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
                self._target_fps = 10.0  # period = 0.1s
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

        # Deadline accumulator: the first read anchors the cadence (no sleep),
        # setting the next deadline at t0 + period (100.1).
        assert source.read() is not None
        assert sleeps == []

        # Arriving 0.04s later must sleep only the *remaining* 0.06s to the
        # deadline — not a full period stacked on top of the read (the old bug).
        now[0] += 0.04
        assert source.read() is not None
        assert sleeps == pytest.approx([0.06], abs=1e-6)

        # Arriving after the next deadline has already passed: no sleep at all,
        # so a blocking/slow source is never double-paced.
        now[0] += 0.12
        assert source.read() is not None
        assert len(sleeps) == 1
        assert adapter.reads == 3


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
            out.append(
                Track(
                    track_id=i + 1,
                    bbox=d.bbox,
                    conf=d.conf,
                    state=TrackState.CONFIRMED,
                    age=1,
                    hits=1,
                    velocity=(0.0, 0.0),
                )
            )
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
    def test_writes_frames_to_shm_and_emits_telemetry(self, qapp, wait_until) -> None:
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
            cid,
            _camera_config(cid),
            on_tel,
            frame_source=src,
            shm_writer=writer,
            telemetry_hz=50.0,
        )
        worker.start()
        try:

            def _latest_frame():
                result = reader.latest()
                if result is not None:
                    _hdr, frame = result
                    return frame
                return None

            frame = wait_until(
                _latest_frame,
                timeout=5.0,
                interval=0.02,
                message="no frame landed in shm",
            )
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

    def test_fps_becomes_positive(self, qapp, wait_until) -> None:
        from autoptz.engine.camera_worker import CameraWorker

        received = []
        lock = threading.Lock()

        def on_tel(m):
            with lock:
                received.append(m)

        worker = CameraWorker(
            "fpscam01abcd",
            _camera_config("fpscam01abcd"),
            on_tel,
            frame_source=FakeFrameSource(),
            telemetry_hz=50.0,
        )
        worker.start()
        try:

            def _best_fps() -> float:
                with lock:
                    if received:
                        return max(m.fps for m in received)
                return 0.0

            best = wait_until(
                _best_fps,
                timeout=3.0,
                interval=0.05,
                message="fps never became positive with a live source",
            )
            assert best > 0.0, "fps never became positive with a live source"
        finally:
            worker.stop()

    def test_stop_is_idempotent_and_releases_source(self, qapp, wait_until) -> None:
        from autoptz.engine.camera_worker import CameraWorker

        src = FakeFrameSource()
        worker = CameraWorker(
            "stopcam01abc",
            _camera_config("stopcam01abc"),
            lambda m: None,
            frame_source=src,
        )
        worker.start()
        wait_until(lambda: src.opened, timeout=2.0, message="worker did not open source")
        worker.stop()
        worker.stop()  # idempotent — must not raise
        assert worker.is_running is False
        assert src.closed is True

    def test_failed_source_emits_error_telemetry_no_crash(self, qapp, wait_until) -> None:
        from autoptz.engine.camera_worker import CameraWorker
        from autoptz.engine.runtime.messages import HealthState

        received = []
        lock = threading.Lock()

        def on_tel(m):
            with lock:
                received.append(m)

        worker = CameraWorker(
            "failcam01abc",
            _camera_config("failcam01abc"),
            on_tel,
            frame_source=FakeFrameSource(fail_open=True),
        )
        worker.start()
        try:
            wait_until(
                lambda: received,
                timeout=3.0,
                interval=0.02,
                message="no telemetry emitted on failed source",
            )
            with lock:
                assert received, "no telemetry emitted on failed source"
                states = {m.health.state for m in received}
            assert HealthState.ERROR in states or HealthState.STOPPED in states
        finally:
            worker.stop()

    def test_commands_are_thread_safe_noops_before_start(self, qapp) -> None:
        from autoptz.engine.camera_worker import CameraWorker

        worker = CameraWorker(
            "cmdcam01abcd",
            _camera_config("cmdcam01abcd"),
            lambda m: None,
            frame_source=FakeFrameSource(),
        )
        # Queueing commands before start must not raise.
        worker.enable_tracking(True)
        worker.set_target(7)
        worker.ptz_nudge(0.5, -0.2, 0.1)
        worker.update_config(_camera_config("cmdcam01abcd"))
        worker.enroll_track(7, "id-1", "Alice")

        # …and they must be true no-ops on lifecycle: none may start the worker or
        # flip it to running before start() is actually called.
        assert worker.is_running is False

    def test_enroll_track_sets_pending_and_immediate_label(self, qapp) -> None:
        """Click-to-assign queues a pending enrollment and labels the box at once."""
        from autoptz.engine.camera_worker import CameraWorker

        worker = CameraWorker(
            "enrollcam01a",
            _camera_config("enrollcam01a"),
            lambda m: None,
            frame_source=FakeFrameSource(),
        )
        worker._apply_command("enroll_track", (7, "id-123", "Alice", (0.4, 0.3)))
        # Awaiting a detected face to bind the embedding…
        assert worker._pending_enroll == {7: ("id-123", "Alice", (0.4, 0.3))}
        # …but the name shows on the box immediately (score 1.0 = manual).
        assert worker._track_identity[7] == ("id-123", "Alice", 1.0)

    def test_set_target_resets_reid_template(self, qapp) -> None:
        """Switching target drops the previous subject's appearance template."""
        from autoptz.engine.camera_worker import CameraWorker

        worker = CameraWorker(
            "reidcam01abc",
            _camera_config("reidcam01abc"),
            lambda m: None,
            frame_source=FakeFrameSource(),
        )

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

    def test_detection_runs_with_engine_on_and_no_target(
        self, qapp, monkeypatch, wait_until
    ) -> None:
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
            "dettest01abc",
            _camera_config("dettest01abc"),
            on_tel,
            frame_source=FakeFrameSource(),
            telemetry_hz=50.0,
        )
        assert worker._tracking_enabled is False  # no target, tracking off
        worker.start()
        try:

            def _latest_tracks():
                with lock:
                    for m in received:
                        if m.tracks:
                            return m.tracks
                return []

            tracks = wait_until(
                _latest_tracks,
                timeout=3.0,
                interval=0.02,
                message="no tracks emitted even though a detector is loaded",
            )
            assert det.calls > 0, "detector was never run despite engine being on"
            assert tracks, "no tracks emitted even though a detector is loaded"
            assert tracks[0].is_target is False  # detection without a follow-target
        finally:
            worker.stop()

    def test_disabling_detection_feature_releases_detector(
        self, qapp, monkeypatch, wait_until
    ) -> None:
        """Turning the detection feature off frees the detector; on rebuilds it.

        Regression for "models not unloaded": disabling a subsystem must drop the
        worker's model reference (so its memory is reclaimed once the pool drops
        its copy too), not merely skip running it.
        """
        from autoptz.engine.camera_worker import CameraWorker

        det = _install_fake_detect_stack(monkeypatch)
        worker = CameraWorker(
            "lifetest01abc",
            _camera_config("lifetest01abc"),
            lambda m: None,
            frame_source=FakeFrameSource(),
            telemetry_hz=50.0,
        )
        worker.start()
        try:
            wait_until(
                lambda: worker._detect is not None,
                timeout=3.0,
                interval=0.02,
                message="detector was not built",
            )
            wait_until(
                lambda: det.calls > 0,
                timeout=3.0,
                interval=0.02,
                message="detector was never called",
            )

            # Disable detection → lifecycle drops the worker's detector reference.
            worker.set_features({"detection": False})
            wait_until(
                lambda: worker._detect is None,
                timeout=3.0,
                interval=0.02,
                message="detector not released after disabling detection",
            )
            calls_when_off = det.calls
            time.sleep(0.2)
            assert det.calls == calls_when_off, "detector still running while disabled"

            # Re-enable → lifecycle rebuilds the detector stack.
            worker.set_features({"detection": True})
            wait_until(
                lambda: worker._detect is not None,
                timeout=3.0,
                interval=0.02,
                message="detector not rebuilt after re-enabling",
            )
        finally:
            worker.stop()

    def test_pool_authoritative_detector_no_per_worker_download(self, qapp, monkeypatch) -> None:
        """With a pool present, a missing model must NOT trigger the per-worker
        fallback (which resolves with allow_download=True and would silently
        download/export, ignoring the operator's auto-download setting)."""
        from autoptz.engine import camera_worker as cw
        from autoptz.engine.camera_worker import CameraWorker

        calls = {"n": 0}

        def _recording_build(config):
            calls["n"] += 1
            return None

        monkeypatch.setattr(cw, "_build_detect_stack", _recording_build)

        class _NoModelPool:
            def detector(self):  # pool has no model (e.g. all deleted)
                return None

        worker = CameraWorker(
            "poolauth01ab",
            _camera_config("poolauth01ab"),
            lambda m: None,
            frame_source=FakeFrameSource(),
        )
        worker.set_inference_pool(_NoModelPool())
        assert worker._resolve_detect_stack() is None
        assert calls["n"] == 0, "per-worker downloading build ran despite a pool being present"

    def test_telemetry_reports_resolution_and_dropped_frames(self, qapp, wait_until) -> None:
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
            "restest01abc",
            _camera_config("restest01abc"),
            on_tel,
            frame_source=FlakySource(),
            telemetry_hz=50.0,
        )
        worker.start()
        try:

            def _quality_seen() -> bool:
                res_ok = drop_ok = False
                with lock:
                    for m in received:
                        if m.width == 640 and m.height == 480:
                            res_ok = True
                        if m.dropped_frames > 0:
                            drop_ok = True
                return res_ok and drop_ok

            wait_until(
                _quality_seen,
                timeout=3.0,
                interval=0.02,
                message="telemetry never reported resolution and dropped frame count",
            )
            with lock:
                res_ok = any(m.width == 640 and m.height == 480 for m in received)
                drop_ok = any(m.dropped_frames > 0 for m in received)
            assert res_ok, "telemetry never reported source resolution 640x480"
            assert drop_ok, "telemetry never counted a dropped frame"
        finally:
            worker.stop()

    def test_reclaims_leaked_shm_segment(self, qapp, wait_until) -> None:
        """A stale segment from a crashed run is reclaimed, not fatal."""
        import uuid
        from multiprocessing import resource_tracker
        from multiprocessing.shared_memory import SharedMemory

        from autoptz.engine.camera_worker import _PREVIEW_H, _PREVIEW_W, CameraWorker
        from autoptz.engine.runtime.shm import ShmReader, frame_region_size

        cid = uuid.uuid4().hex[:12]
        shm_name = f"cam_{cid[:8]}_preview"
        _cleanup_shm(shm_name)

        # Simulate a leaked main segment from a previous crashed process.  Use
        # the real preview region size; tiny placeholders can corrupt native
        # mmap cleanup on Linux when the worker replaces the region.
        leaked = SharedMemory(
            name=shm_name,
            create=True,
            size=frame_region_size(_PREVIEW_H, _PREVIEW_W),
        )
        try:
            resource_tracker.unregister(leaked._name, "shared_memory")  # noqa: SLF001
        except Exception:
            pass
        leaked.close()  # leave it linked (orphaned)

        worker = CameraWorker(
            cid,
            _camera_config(cid),
            lambda m: None,
            frame_source=FakeFrameSource(h=_PREVIEW_H, w=_PREVIEW_W),
            telemetry_hz=50.0,
        )
        worker.start()
        try:
            # Worker should have reclaimed the orphan and produced a live region.
            def _open_reader_with_frame():
                try:
                    candidate = ShmReader(shm_name, _PREVIEW_H, _PREVIEW_W)
                except Exception:
                    return None
                if candidate.latest() is not None:
                    return candidate
                candidate.close()
                return None

            reader = wait_until(
                _open_reader_with_frame,
                timeout=5.0,
                interval=0.05,
                message="worker did not reclaim leaked segment",
            )
            reader.close()
        finally:
            worker.stop()
        _cleanup_shm(shm_name)

    def test_ptz_nudge_drives_injected_backend(self, qapp, monkeypatch, wait_until) -> None:
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
            "ptzcam01abcd",
            _camera_config("ptzcam01abcd"),
            lambda m: None,
            frame_source=FakeFrameSource(),
            ptz_controller=backend,
        )
        worker.start()
        try:
            worker.ptz_nudge(0.7, 0.0, 0.0)
            wait_until(
                lambda: backend.moves,
                timeout=2.0,
                interval=0.02,
                message="ptz nudge did not reach the backend",
            )
            assert backend.moves, "ptz nudge did not reach the backend"
            assert backend.moves[0][0] == pytest.approx(0.7)
        finally:
            worker.stop()
        assert backend.stopped is True

    def test_ptz_nudge_uses_low_latency_queue(self, qapp) -> None:
        # Manual nudges must ride the dedicated PTZ queue (drained on the capture
        # thread), NOT the inference command queue — otherwise a heavy detect+track
        # pass adds tens of ms of lag to every joystick move.
        from autoptz.engine.camera_worker import CameraWorker

        worker = CameraWorker(
            "ptzcam01abcd",
            _camera_config("ptzcam01abcd"),
            lambda m: None,
            frame_source=FakeFrameSource(),
        )
        worker.ptz_nudge(0.5, 0.0, 0.0)
        assert len(worker._ptz_cmd_queue) == 1
        assert worker._ptz_cmd_queue[0][0] == "ptz_nudge"
        assert len(worker._cmd_queue) == 0
        # The nudge must wake a backed-off capture sleep so it applies immediately
        # even when the video feed is stalled (not just at the next loop tick).
        assert worker._capture_wake.is_set()


# ─────────────────────────────────────────────────────────────────────────────
# CameraWorker — fixed-rate PTZ command pump flag (AUTOPTZ_PTZ_PUMP)
# ─────────────────────────────────────────────────────────────────────────────


class _SpyController:
    """Stand-in PTZController recording step()/update()/start()/stop() calls.

    Has ``step`` so the worker treats it as a controller (not a bare backend).
    ``step`` returns a fixed command so the legacy inline path can mirror it.
    """

    def __init__(self) -> None:
        self.steps: list[tuple] = []
        self.updates: list[tuple] = []
        self.started = 0
        self.stopped = 0
        self.latencies: list[float] = []
        self.manual_holds = 0

    def step(self, error, velocity, subject_height, track_active, t=None):
        self.steps.append((error, velocity, subject_height, track_active))
        return (0.1, 0.2, 0.0)

    def update(self, error, velocity, subject_height, track_active):
        self.updates.append((error, velocity, subject_height, track_active))

    def note_manual_hold(self) -> None:
        self.manual_holds += 1

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def set_loop_latency(self, seconds: float) -> None:
        self.latencies.append(seconds)


class TestPtzPumpFlag:
    """AUTOPTZ_PTZ_PUMP gates the off-thread pump; OFF == legacy inline step()."""

    def _worker(self, monkeypatch, *, pump: bool):
        from autoptz.engine.camera_worker import CameraWorker

        monkeypatch.setenv("AUTOPTZ_PTZ_PUMP", "1" if pump else "0")
        ctrl = _SpyController()
        worker = CameraWorker(
            "ptzpump12345",
            _camera_config("ptzpump12345"),
            lambda _m: None,
            frame_source=FakeFrameSource(),
            ptz_controller=ctrl,
        )
        return worker, ctrl

    def test_env_helper_parses(self, monkeypatch) -> None:
        from autoptz.engine.camera_worker import _ptz_pump_enabled

        for on in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv("AUTOPTZ_PTZ_PUMP", on)
            assert _ptz_pump_enabled() is True
        for off in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("AUTOPTZ_PTZ_PUMP", off)
            assert _ptz_pump_enabled() is False

    def test_default_off(self, monkeypatch) -> None:
        from autoptz.engine.camera_worker import _ptz_pump_enabled

        monkeypatch.delenv("AUTOPTZ_PTZ_PUMP", raising=False)
        assert _ptz_pump_enabled() is False

    def test_publish_routes_to_update_when_pump_on(self, monkeypatch) -> None:
        worker, ctrl = self._worker(monkeypatch, pump=True)
        worker._publish_ptz(
            ctrl, (0.3, 0.1), (0.0, 0.0), 0.45, track_active=True, now=0.0, log_label="auto"
        )
        assert ctrl.updates == [((0.3, 0.1), (0.0, 0.0), 0.45, True)]
        assert ctrl.steps == [], "pump ON must not call step()"

    def test_publish_routes_to_step_when_pump_off(self, monkeypatch) -> None:
        worker, ctrl = self._worker(monkeypatch, pump=False)
        worker._publish_ptz(
            ctrl, (0.3, 0.1), (0.0, 0.0), 0.45, track_active=True, now=0.0, log_label="auto"
        )
        assert ctrl.steps == [((0.3, 0.1), (0.0, 0.0), 0.45, True)]
        assert ctrl.updates == [], "pump OFF must not call update()"
        # Legacy inline path mirrors the returned command.
        assert worker._ptz_last_cmd == (0.1, 0.2, 0.0)

    def test_maybe_start_pump_starts_controller_when_on(self, monkeypatch) -> None:
        worker, ctrl = self._worker(monkeypatch, pump=True)
        worker._maybe_start_ptz_pump()
        assert ctrl.started == 1

    def test_maybe_start_pump_noop_when_off(self, monkeypatch) -> None:
        worker, ctrl = self._worker(monkeypatch, pump=False)
        worker._maybe_start_ptz_pump()
        assert ctrl.started == 0

    def test_manual_override_refreshes_heartbeat_when_pump_on(self, monkeypatch) -> None:
        """I1: during a manual-override window the auto loop early-returns without
        an update(); pump mode must refresh the heartbeat so the loop doesn't
        inject a stop() that fights the operator's nudge."""
        worker, ctrl = self._worker(monkeypatch, pump=True)
        now = time.monotonic()
        worker._manual_override_until = now + 10.0  # override is active
        worker._drive_ptz_auto([], None, now)
        assert ctrl.manual_holds == 1, "pump mode must refresh heartbeat during manual override"
        assert ctrl.updates == [], "manual override must not feed the controller"
        assert ctrl.steps == []

    def test_manual_override_no_heartbeat_refresh_when_pump_off(self, monkeypatch) -> None:
        """OFF mode has no heartbeat, so the override path must not touch it."""
        worker, ctrl = self._worker(monkeypatch, pump=False)
        now = time.monotonic()
        worker._manual_override_until = now + 10.0
        worker._drive_ptz_auto([], None, now)
        assert ctrl.manual_holds == 0, "OFF mode must not call note_manual_hold"

    def test_unusable_bbox_publishes_idle_not_tracking(self, monkeypatch) -> None:
        from autoptz.engine.runtime.messages import BBox, TrackInfo

        worker, ctrl = self._worker(monkeypatch, pump=False)
        worker._tracking_enabled = True
        worker._target_track_id = 7
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        valid = TrackInfo(
            track_id=7,
            bbox=BBox(x1=100, y1=100, x2=220, y2=360),
            confidence=0.9,
        )
        worker._apply_target_lock([valid], frame, now=1.0)
        ctrl.steps.clear()

        tiny = TrackInfo(
            track_id=7,
            bbox=BBox(x1=500, y1=20, x2=510, y2=30),
            confidence=0.9,
        )
        worker._apply_target_lock([tiny], frame, now=1.1)
        worker._drive_ptz_auto([tiny], frame, now=1.1)

        assert ctrl.steps
        assert ctrl.steps[-1][3] is False


class _ThreadedSpyController:
    """Spy controller with a real daemon loop, like the production controller.

    ``start()`` spins a daemon thread that idles until ``stop()`` joins it, so a
    test can assert the rebuild stops the OLD controller (no leaked/extra thread)
    before the new stack comes up.  Records call ordering so we can prove the old
    stop happens before the new pump starts.
    """

    _events: list[str] = []

    def __init__(self, name: str) -> None:
        self.name = name
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = 0
        self.stopped = 0

    # the worker only treats an object as a controller (not a bare backend) when
    # it has step(); never actually invoked in pump mode but required for routing.
    def step(self, *a, **k):  # pragma: no cover - not exercised in pump mode
        return (0.0, 0.0, 0.0)

    def update(self, *a, **k) -> None:  # pragma: no cover - not exercised here
        pass

    def set_loop_latency(self, seconds: float) -> None:  # pragma: no cover
        pass

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(0.01)

    def start(self) -> None:
        self.started += 1
        type(self)._events.append(f"start:{self.name}")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name=f"spy-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stopped += 1
        type(self)._events.append(f"stop:{self.name}")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class TestPtzRebuildLifecycle:
    """Rebuilding the PTZ backend in pump mode must stop the OLD controller's
    background loop (C1) before the new one starts — no leaked/extra thread."""

    def _worker(self, monkeypatch, *, pump: bool, old_ctrl):
        from autoptz.engine.camera_worker import CameraWorker

        monkeypatch.setenv("AUTOPTZ_PTZ_PUMP", "1" if pump else "0")
        worker = CameraWorker(
            "ptzrebuild123",
            _camera_config("ptzrebuild123"),
            lambda _m: None,
            frame_source=FakeFrameSource(),
            ptz_controller=old_ctrl,
        )
        return worker

    def test_rebuild_stops_old_controller_before_new_starts(self, monkeypatch, wait_until) -> None:
        _ThreadedSpyController._events = []
        old = _ThreadedSpyController("old")
        worker = self._worker(monkeypatch, pump=True, old_ctrl=old)
        # We own the controller for the test so the rebuild treats it as ours and
        # the stop path is exercised; mark it owned to mirror a built stack.
        worker._ptz_owned = True

        # The rebuild then constructs a fresh stack.  Stub it to install a second
        # threaded spy as the NEW controller and start its pump, so we can assert
        # ordering (old stop BEFORE new start) without a real backend/hardware.
        new = _ThreadedSpyController("new")

        def _build_new_stack() -> None:
            worker._ptz = new
            worker._ptz_owned = True

        monkeypatch.setattr(worker, "_build_ptz_stack", _build_new_stack)

        before = threading.active_count()
        old.start()  # the old controller's loop is running, like a live pump
        assert old._thread is not None and old._thread.is_alive()

        worker._rebuild_ptz_backend()

        # OLD controller's loop was stopped (its stop() called, thread joined)…
        assert old.stopped == 1, "old controller was not stopped on rebuild (C1 leak)"
        assert old._thread is None, "old controller's loop thread leaked"
        # …and the NEW controller's pump was started.
        assert new.started == 1, "new controller pump did not start after rebuild"
        # Ordering: old stop strictly precedes new start (no two-senders window).
        assert _ThreadedSpyController._events.index(
            "stop:old"
        ) < _ThreadedSpyController._events.index("start:new")
        # No thread leak: the old loop is gone; the new spy's loop is the only add.
        new.stop()
        # Allow the joined threads to fully retire before counting.
        wait_until(
            lambda: threading.active_count() <= before,
            timeout=2.0,
            message="rebuild leaked a controller loop thread",
        )
        assert threading.active_count() <= before, "rebuild leaked a controller loop thread"

    def test_rebuild_off_mode_does_not_stop_controller(self, monkeypatch) -> None:
        """OFF mode never runs a controller loop, so the rebuild must NOT call
        stop() on the controller (no loop to stop; backend close path unchanged)."""
        _ThreadedSpyController._events = []
        old = _ThreadedSpyController("old")
        worker = self._worker(monkeypatch, pump=False, old_ctrl=old)
        worker._ptz_owned = True
        monkeypatch.setattr(worker, "_build_ptz_stack", lambda: None)

        worker._rebuild_ptz_backend()

        assert old.stopped == 0, "OFF mode must not stop the controller on rebuild"


class TestInferenceWakeTimeout:
    """The inference loop's frame-wait ceiling must stay small so commands drain
    promptly when frames are sparse (C3: lower wake-latency ceiling)."""

    def test_wake_timeout_is_small(self) -> None:
        from autoptz.engine.camera_worker import _INFER_WAKE_TIMEOUT_S

        # Tightened from the old 50 ms ceiling; a value check guards a regression.
        assert _INFER_WAKE_TIMEOUT_S == pytest.approx(0.01)
        assert _INFER_WAKE_TIMEOUT_S < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# CameraWorker._track_error — region-aware aim point
# ─────────────────────────────────────────────────────────────────────────────


class TestTrackErrorAimRegion:
    """The vertical aim point must follow ``tracking.framing`` while the
    horizontal aim stays on the box centre."""

    @staticmethod
    def _worker(framing: str, aim_body_mode: str = "torso"):
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
            tracking=TrackingConfig(framing=framing, aim_body_mode=aim_body_mode),
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
        assert (
            ey_by_region["face"]
            > ey_by_region["head_shoulders"]
            > ey_by_region["upper_body"]
            > ey_by_region["full_body"]
        )
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
        assert torso_ey > 0.0  # "face" aims high in both modes
        assert sil_ey == pytest.approx(torso_ey)
        assert sil_h == pytest.approx(torso_h)  # same bbox height when pose absent

    def test_pose_aim_sits_on_torso_and_sets_source(self) -> None:
        # With a confident pose, the aim CENTRE is the landmark-precise torso
        # anchor fused with the box (pose-weighted by confidence) and
        # ``aim_source == "pose"`` — this keeps the on-screen circle on the body,
        # not the bounding box.
        from autoptz.engine.pipeline.framing import Keypoint

        class _FakePose:
            available = True

            def estimate(self, _frame, _bbox):
                kps = [Keypoint(0.0, 0.0, 0.0)] * 17
                kps[5] = Keypoint(40.0, 30.0, 0.9)  # left shoulder
                kps[6] = Keypoint(60.0, 30.0, 0.9)  # right shoulder
                kps[11] = Keypoint(42.0, 70.0, 0.9)  # left hip
                kps[12] = Keypoint(58.0, 70.0, 0.9)  # right hip
                return kps

        w = self._worker("upper_body", "torso")
        w._pose = _FakePose()
        w._pose_probed = True  # bypass the lazy model build
        trk = self._track(20, 0, 80, 100)  # box centre (50,50) ≠ torso anchor
        trk.is_target = True
        w._target_track_id = trk.track_id
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        (ex, ey), _h = w._track_error(trk, frame, now=123.0)
        assert trk.aim_source == "pose"
        # upper_body → chest anchor (shoulders y=30 nudged 20% toward hips y=70 →
        # y≈38), horizontally on the shoulder centre (x=50), NOT the box centre y.
        assert trk.aim_x == pytest.approx(50.0, abs=1.0)
        assert trk.aim_y == pytest.approx(38.0, abs=1.5)
        assert ey > 0.0  # torso anchor sits above frame centre

    def test_target_aim_fields_are_annotated(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        trk = self._track(40, 0, 60, 100)
        trk.is_target = True
        self._worker("upper_body")._track_error(trk, frame)
        assert trk.aim_x == pytest.approx(50.0)
        assert trk.aim_y == pytest.approx(38.0)
        assert trk.aim_source == "bbox"

    def test_bbox_shape_change_is_stabilized_for_ptz_framing(self) -> None:
        from autoptz.engine.runtime.messages import BBox

        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        w = self._worker("full_body")
        w._features["pose"] = False
        trk = self._track(350, 200, 550, 800)  # raw bbox aim x=450
        trk.is_target = True
        w._target_track_id = trk.track_id
        w._target_lock.previous_trusted_bbox = BBox(x1=450, y1=200, x2=550, y2=800)
        w._target_lock.trusted_aim = (500.0, 500.0)

        (ex, ey), height = w._track_error(trk, frame, now=10.0)

        assert trk.aim_source == "bbox_stable"
        assert trk.aim_x == pytest.approx(487.5)
        assert trk.aim_y == pytest.approx(500.0)
        assert ex == pytest.approx(-0.025)
        assert ey == pytest.approx(0.0)
        assert height == pytest.approx(0.6)

    def test_bbox_motion_without_overlap_is_not_shape_stabilized(self) -> None:
        from autoptz.engine.runtime.messages import BBox

        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        w = self._worker("full_body")
        w._features["pose"] = False
        trk = self._track(650, 200, 750, 800)  # real move to the right
        trk.is_target = True
        w._target_track_id = trk.track_id
        w._target_lock.previous_trusted_bbox = BBox(x1=450, y1=200, x2=550, y2=800)
        w._target_lock.trusted_aim = (500.0, 500.0)

        (ex, ey), height = w._track_error(trk, frame, now=10.0)

        assert trk.aim_source == "bbox"
        assert trk.aim_x == pytest.approx(700.0)
        assert ex == pytest.approx(0.4)
        assert ey == pytest.approx(0.0)
        assert height == pytest.approx(0.6)


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
        client.providerAttachRequested.connect(lambda c, shm, w, h: attaches.append((c, shm, w, h)))
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
            diagnostics,
            "system_metrics",
            lambda: {"available": True, "cpu_percent": 20.0, "mem_percent": 40.0},
        )
        assert Supervisor._adaptive_startup_concurrency() == 4

        monkeypatch.setattr(sup_mod.os, "cpu_count", lambda: 6)
        monkeypatch.setattr(
            diagnostics,
            "system_metrics",
            lambda: {"available": True, "cpu_percent": 60.0, "mem_percent": 80.0},
        )
        assert Supervisor._adaptive_startup_concurrency() == 2

        monkeypatch.setattr(
            diagnostics,
            "system_metrics",
            lambda: {"available": False},
        )
        assert Supervisor._adaptive_startup_concurrency() == 1

    def test_staged_start_reports_progress_and_spawns(self, qapp, monkeypatch, wait_until) -> None:
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
            wait_until(
                lambda: sup.worker_count == 2 and any(e.get("phase") == "Ready" for e in events),
                timeout=2.0,
                message="staged supervisor start did not spawn all workers and report Ready",
            )
            assert sup.worker_count == 2
            phases = [e.get("phase") for e in events]
            assert "Opening cameras" in phases
            assert "Ready" in phases
        finally:
            sup.stop()

    def test_staged_start_releases_inference_without_forced_warmup(
        self, qapp, monkeypatch, wait_until
    ) -> None:
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

            def _workers_released():
                workers = list(sup._workers.values())
                if workers and all(
                    w.inference_paused and w.inference_paused[-1] is False for w in workers
                ):
                    return workers
                return []

            workers = wait_until(
                _workers_released,
                timeout=2.0,
                message="staged supervisor start did not release inference",
            )
            assert events[:2] == [
                f"start:{client.cameraModel.camera_ids()[0]}",
                f"start:{client.cameraModel.camera_ids()[1]}",
            ]
            assert "detector" not in events
            assert "face" not in events
            assert "pose" not in events
            assert "reid" not in events
            assert workers and all(w.inference_paused[0] is True for w in workers)
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

    def test_engine_ep_updates_from_telemetry(self, qapp) -> None:
        # Regression: engineEp must reflect the worker-reported EP, not stay blank.
        from autoptz.engine.runtime.messages import TelemetryMsg

        c = _make_client(qapp)
        cid = c.addCamera("usb://0", "X")
        c.drain_commands()
        c.set_supervisor(_make_supervisor(c, factory=FakeWorker))
        c.startEngine()
        try:
            c.push_telemetry(TelemetryMsg(camera_id=cid, seq=1, ep="CoreMLExecutionProvider"))
            assert c.engineEp == "CoreML"
        finally:
            c.stopEngine()

    def test_model_task_releases_sessions_before_mutating(self, qapp, monkeypatch) -> None:
        # Regression: on Windows the model file can't be deleted/replaced while
        # onnxruntime holds it open, so sessions must be released BEFORE the
        # on-disk mutation and rebuilt after.
        import autoptz.engine.runtime.models as models_mod
        from autoptz.ui.widgets.dialogs.model_manager import _ModelTask

        order: list[str] = []

        class _SpyClient:
            def releaseModelSessions(self):
                order.append("release")

            def rebuildModelSessions(self):
                order.append("rebuild")

        class _FakeManager:
            def remove_app_models(self, *, keys=None):
                order.append("remove")
                return [{"name": "m", "state": "removed", "path": "", "size": "0", "error": ""}]

        monkeypatch.setattr(models_mod, "default_manager", lambda: _FakeManager())
        _ModelTask("remove", ["detector_fast"], client=_SpyClient()).run()
        assert order == ["release", "remove", "rebuild"]

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

        queued: list[TelemetryMsg] = []

        class _QueuedSignal:
            def emit(self, msg: TelemetryMsg) -> None:
                queued.append(msg)

        # Avoid spinning Qt's native dispatcher in the mixed suite. The production
        # signal is already connected as QueuedConnection; this test pins the
        # thread branch and verifies the owning-thread slot separately.
        c._telemetryArrived = _QueuedSignal()

        def worker_push():
            c.push_telemetry(TelemetryMsg(camera_id=cid, seq=1, fps=42.0))

        t = threading.Thread(target=worker_push)
        t.start()
        t.join()

        # Off-thread push is queued: model not updated inline.
        assert c.get_camera(cid).fps == pytest.approx(0.0)
        assert [msg.camera_id for msg in queued] == [cid]

        # The queued signal targets this slot on the owning Qt thread in production.
        c._on_telemetry_main(queued[0])
        assert c.get_camera(cid).fps == pytest.approx(42.0)

    def test_push_from_owning_thread_applies_synchronously(self, qapp) -> None:
        from autoptz.engine.runtime.messages import TelemetryMsg

        c = _make_client(qapp)
        cid = c.addCamera("usb://0", "X")
        c.drain_commands()
        c.push_telemetry(TelemetryMsg(camera_id=cid, seq=1, fps=30.0))
        # Same-thread push applies immediately (no event-loop spin needed).
        assert c.get_camera(cid).fps == pytest.approx(30.0)


class _FeatureFakeWorker(FakeWorker):
    """FakeWorker that records feature pushes + model-reload calls."""

    def __init__(self, camera_id, config, on_telemetry):
        super().__init__(camera_id, config, on_telemetry)
        self.features_calls: list[dict] = []
        self.reloads = 0

    def set_features(self, features):
        self.features_calls.append(dict(features))

    def reload_inference_models(self):
        self.reloads += 1


class _FakePool:
    """Records which shared models the supervisor asked to release."""

    def __init__(self):
        self.released: list[str] = []

    def release_detector(self):
        self.released.append("detector")

    def release_face(self):
        self.released.append("face")

    def release_pose(self):
        self.released.append("pose")


def _all_features(**overrides):
    feats = {
        "detection": True,
        "tracking": True,
        "face_recognition": True,
        "pose": True,
        "reid": True,
    }
    feats.update(overrides)
    return feats


class TestSupervisorModelLifecycle:
    def test_disabling_feature_releases_pool_model(self, qapp) -> None:
        from autoptz.engine.runtime.messages import SetFeaturesCmd

        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "A")
        client.drain_commands()
        captured: dict = {}
        sup = _make_supervisor(
            client,
            factory=lambda c, cfg, t: captured.setdefault(c, _FeatureFakeWorker(c, cfg, t)),
        )
        sup.start()
        try:
            pool = _FakePool()
            sup._inference_pool = pool  # inject fake shared pool post-start
            sup._route(SetFeaturesCmd(camera_id=None, features=_all_features()))
            sup._route(
                SetFeaturesCmd(
                    camera_id=None,
                    features=_all_features(detection=False, pose=False, reid=False),
                )
            )
            assert "detector" in pool.released
            assert "pose" in pool.released
            assert "face" not in pool.released  # face stayed on → not released
            assert captured[cid].features_calls[-1]["detection"] is False
        finally:
            sup.stop()

    def test_apply_model_cache_changed_reloads_workers(self, qapp) -> None:
        client = _make_client(qapp)
        cid = client.addCamera("usb://0", "A")
        client.drain_commands()
        captured: dict = {}
        sup = _make_supervisor(
            client,
            factory=lambda c, cfg, t: captured.setdefault(c, _FeatureFakeWorker(c, cfg, t)),
        )
        sup.start()
        try:
            pool = _FakePool()
            sup._inference_pool = pool
            sup.apply_model_cache_changed()
            assert "detector" in pool.released and "pose" in pool.released
            assert captured[cid].reloads == 1
        finally:
            sup.stop()

    def test_apply_model_cache_changed_noop_when_stopped(self, qapp) -> None:
        client = _make_client(qapp)
        sup = _make_supervisor(client, factory=_FeatureFakeWorker)
        sup.apply_model_cache_changed()  # not started → safe no-op
        assert sup.is_running is False
