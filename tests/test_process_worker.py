"""Tests for the experimental opt-in process-per-camera mode.

Covers the IPC/lifecycle plumbing that CAN be validated without a real camera or
models: WorkerSpec pickle-safety, the supervisor-side handle proxying method calls
to the command queue, and one end-to-end spawn of a child process driving a
synthetic frame source (frames must reach shared memory and telemetry must flow
back), then a clean stop.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import time
import uuid

from autoptz.config.models import CameraConfig, SourceConfig
from autoptz.engine.process_worker import (
    _STOP,
    ProcessWorkerHandle,
    WorkerSpec,
    process_per_camera_enabled,
)


def _config(camera_id: str) -> CameraConfig:
    return CameraConfig(
        id=camera_id,
        name="ProcCam",
        source=SourceConfig(type="usb", address="usb://0"),
    )


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


class TestWorkerSpec:
    def test_spec_is_picklable(self) -> None:
        # spawn (macOS default) pickles the spec to the child — it must round-trip.
        cid = "proc-" + uuid.uuid4().hex[:8]
        spec = WorkerSpec(
            camera_id=cid,
            config=_config(cid),
            db_path="/tmp/x.db",
            detector_tier="fast",
            features={"detection": False},
        )
        restored = pickle.loads(pickle.dumps(spec))
        assert restored.camera_id == cid
        assert restored.config.source.address == "usb://0"
        assert restored.features == {"detection": False}
        assert restored.detector_tier == "fast"


class TestEnabledFlag:
    def test_disabled_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("AUTOPTZ_PROCESS_PER_CAMERA", raising=False)
        assert process_per_camera_enabled() is False

    def test_enabled_by_env(self, monkeypatch) -> None:
        for val in ("1", "true", "on", "YES"):
            monkeypatch.setenv("AUTOPTZ_PROCESS_PER_CAMERA", val)
            assert process_per_camera_enabled() is True


class TestRelayIdentityLockFree:
    """_relay_identity_to_siblings must be a lock-free no-op in threaded (default) mode."""

    def _make_supervisor(self, qapp):  # noqa: ANN001
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient

        client = EngineClient()
        sup = Supervisor(client, store=None)
        sup._ensure_identity_service = lambda: None  # type: ignore[method-assign]
        sup._ensure_inference_pool = lambda: None  # type: ignore[method-assign]
        return sup

    def test_relay_is_noop_when_disabled(self, qapp, monkeypatch) -> None:
        """With AUTOPTZ_PROCESS_PER_CAMERA unset the relay returns immediately without
        touching any worker, confirming the lock-free early-return path."""
        from unittest.mock import MagicMock, patch

        monkeypatch.delenv("AUTOPTZ_PROCESS_PER_CAMERA", raising=False)
        sup = self._make_supervisor(qapp)

        # Add a fake process worker so we can confirm it is never called.
        fake_worker = MagicMock()
        fake_worker._is_process_worker = True
        sup._workers["cam-other"] = fake_worker

        sentinel = object()
        # Patch at the source module so the local import inside the method picks up the mock.
        with patch(
            "autoptz.engine.process_worker.process_per_camera_enabled",
            return_value=False,
        ) as mock_gate:
            sup._relay_identity_to_siblings("cam-source", sentinel)
            mock_gate.assert_called_once()

        # The fake worker must never have been touched.
        fake_worker.ingest_identity.assert_not_called()

    def test_relay_skips_threaded_workers(self, qapp, monkeypatch) -> None:
        """Even if process-per-camera is enabled, threaded CameraWorkers are skipped
        (they have no ``_is_process_worker`` flag) — relay is a structural no-op for them."""
        monkeypatch.setenv("AUTOPTZ_PROCESS_PER_CAMERA", "1")
        sup = self._make_supervisor(qapp)

        # Install a fake threaded worker (no _is_process_worker attr).
        class _FakeThreadedWorker:
            ingest_calls: list = []

            def ingest_identity(self, record):  # noqa: ANN001
                self.ingest_calls.append(record)

        fake = _FakeThreadedWorker()
        sup._workers["cam-other"] = fake

        sentinel = object()
        sup._relay_identity_to_siblings("cam-source", sentinel)
        # Threaded worker must NOT have received the relay.
        assert fake.ingest_calls == [], (
            "relay incorrectly forwarded identity to a threaded CameraWorker"
        )


class TestChildThreadCaps:
    """A spawned camera child must re-apply the per-camera thread budget itself.

    It inherits the supervisor's env, but ``torch.set_num_threads`` / OpenCV caps
    are *runtime* calls no env performs — without them each camera process imports
    torch/cv2 at cores-wide and several oversubscribe the machine (the "each new
    process eats a lot of CPU" headroom)."""

    def test_applies_inherited_budget_to_torch_and_opencv(self, monkeypatch) -> None:
        from autoptz.engine import process_worker as pw
        from autoptz.engine.runtime import flags

        seen: dict[str, object] = {}
        monkeypatch.setattr(
            flags, "apply_opencv_thread_cap", lambda *a, **k: seen.setdefault("cv2", True)
        )
        monkeypatch.setattr(flags, "apply_thread_caps", lambda n: seen.__setitem__("torch", n))
        monkeypatch.setenv("AUTOPTZ_ORT_INTRA_THREADS", "3")

        pw._apply_child_thread_caps()

        assert seen.get("cv2") is True
        assert seen.get("torch") == 3

    def test_skips_torch_cap_when_no_budget_published(self, monkeypatch) -> None:
        from autoptz.engine import process_worker as pw
        from autoptz.engine.runtime import flags

        seen: dict[str, object] = {}
        monkeypatch.setattr(
            flags, "apply_opencv_thread_cap", lambda *a, **k: seen.setdefault("cv2", True)
        )
        monkeypatch.setattr(flags, "apply_thread_caps", lambda n: seen.__setitem__("torch", n))
        monkeypatch.delenv("AUTOPTZ_ORT_INTRA_THREADS", raising=False)

        pw._apply_child_thread_caps()

        # OpenCV cap (reads its own env) still runs; the torch budget cap is skipped.
        assert seen.get("cv2") is True
        assert "torch" not in seen


class TestHandleProxy:
    """The handle must serialize each supervisor method call onto the command queue
    (post-start), and buffer pre-start feature/defer config into the spec."""

    def _handle(self) -> ProcessWorkerHandle:
        cid = "proc-" + uuid.uuid4().hex[:8]
        return ProcessWorkerHandle(cid, _config(cid), on_telemetry=lambda _m: None, db_path="")

    def test_pre_start_set_features_buffers_not_sends(self) -> None:
        h = self._handle()
        h.set_features({"detection": False, "pose": False})
        # Buffered (no queue yet), applied via the spec at start().
        assert h._features == {"detection": False, "pose": False}
        assert h._cmd_q is None

    def test_post_start_methods_enqueue_commands(self) -> None:
        h = self._handle()

        class _FakeQ:
            def __init__(self) -> None:
                self.items: list = []

            def put(self, item) -> None:
                self.items.append(item)

        q = _FakeQ()
        h._cmd_q = q
        h._started = True

        h.ptz_nudge(0.5, -0.2, 0.1)
        h.enable_tracking(True)
        h.set_features({"pose": False})
        h.set_target("track-7")

        assert ("ptz_nudge", (0.5, -0.2, 0.1), {}) in q.items
        assert ("enable_tracking", (True,), {}) in q.items
        assert ("set_features", ({"pose": False},), {}) in q.items
        assert ("set_target", ("track-7",), {}) in q.items

    def test_injection_setters_are_noops(self) -> None:
        # The shared pool/service can't cross to a child; these must not raise.
        h = self._handle()
        h.set_inference_pool(object())
        h.set_identity_service(object())
        h.set_identity_callback(lambda _r: None)
        assert h._on_identity is not None

    def test_shm_name_matches_camera_worker_format(self) -> None:
        cid = "abcd1234ef"
        h = ProcessWorkerHandle(cid, _config(cid), on_telemetry=lambda _m: None, db_path="")
        assert h.shm_name == f"cam_{cid[:8]}_preview"


class TestHandleIsAlive:
    """is_alive() must mirror the child process's liveness for the health scan."""

    def _handle(self) -> ProcessWorkerHandle:
        cid = "proc-" + uuid.uuid4().hex[:8]
        return ProcessWorkerHandle(cid, _config(cid), on_telemetry=lambda _m: None, db_path="")

    def test_false_before_start(self) -> None:
        h = self._handle()
        assert h._proc is None
        assert h.is_alive() is False

    def test_true_when_proc_alive(self) -> None:
        class _FakeProc:
            def is_alive(self) -> bool:
                return True

        h = self._handle()
        h._proc = _FakeProc()
        assert h.is_alive() is True

    def test_false_when_proc_dead(self) -> None:
        class _FakeProc:
            def is_alive(self) -> bool:
                return False

        h = self._handle()
        h._proc = _FakeProc()
        assert h.is_alive() is False

    def test_is_running_mirrors_is_alive(self) -> None:
        class _FakeProc:
            def is_alive(self) -> bool:
                return True

        h = self._handle()
        h._proc = _FakeProc()
        assert h.is_running is True


class TestHandleStop:
    """stop() must escalate join -> terminate -> join and log an unclean exit."""

    def _handle(self):  # noqa: ANN202
        cid = "proc-" + uuid.uuid4().hex[:8]
        return ProcessWorkerHandle(cid, _config(cid), on_telemetry=lambda _m: None, db_path="")

    def test_stop_terminates_when_join_times_out(self) -> None:
        h = self._handle()

        class _StuckProc:
            """Joins never finish; terminate() makes it die."""

            def __init__(self) -> None:
                self.alive = True
                self.terminated = False

            def is_alive(self) -> bool:
                return self.alive

            def join(self, timeout=None) -> None:  # noqa: ANN001
                return None  # never transitions on its own

            def terminate(self) -> None:
                self.terminated = True
                self.alive = False

        proc = _StuckProc()
        h._proc = proc  # type: ignore[assignment]
        h._cmd_q = None  # exercise the no-queue branch
        h._started = True

        h.stop(timeout=0.01)

        assert proc.terminated, "a child that won't join must be terminated"
        assert h._proc is None
        assert h._started is False

    def test_stop_warns_when_child_survives_terminate(self, caplog) -> None:  # noqa: ANN001
        h = self._handle()

        class _ZombieProc:
            def is_alive(self) -> bool:
                return True  # never dies, even after terminate

            def join(self, timeout=None) -> None:  # noqa: ANN001
                return None

            def terminate(self) -> None:
                return None

        h._proc = _ZombieProc()  # type: ignore[assignment]
        h._started = True
        import logging

        with caplog.at_level(logging.WARNING):
            h.stop(timeout=0.01)
        assert any("did not exit" in r.message.lower() for r in caplog.records), (
            "an unclean child exit must surface as a WARNING"
        )
        assert h._proc is None

    def test_stop_closes_queues_and_drops_refs(self) -> None:
        """stop() must release the three mp.Queues (close + cancel feeder join) and
        drop the references, so a respawned/flapping camera doesn't leak feeder
        threads + pipe FDs on every restart.
        """
        h = self._handle()

        class _FakeQ:
            def __init__(self) -> None:
                self.closed = False
                self.cancelled = False

            def put(self, _item) -> None:  # noqa: ANN001
                return None

            def cancel_join_thread(self) -> None:
                self.cancelled = True

            def close(self) -> None:
                self.closed = True

        cmd_q, tel_q, ident_q = _FakeQ(), _FakeQ(), _FakeQ()
        h._cmd_q = cmd_q  # type: ignore[assignment]
        h._telemetry_q = tel_q  # type: ignore[assignment]
        h._identity_q = ident_q  # type: ignore[assignment]
        h._proc = None  # no child to join
        h._started = True

        h.stop(timeout=0.01)

        for q in (cmd_q, tel_q, ident_q):
            assert q.cancelled, "each queue's feeder-join must be cancelled"
            assert q.closed, "each queue must be closed to release its pipe FDs"
        assert h._cmd_q is None
        assert h._telemetry_q is None
        assert h._identity_q is None


class TestIngestIdentityProxy:
    def test_ingest_identity_enqueues_command(self) -> None:
        cid = "proc-" + uuid.uuid4().hex[:8]
        h = ProcessWorkerHandle(cid, _config(cid), on_telemetry=lambda _m: None, db_path="")

        class _FakeQ:
            def __init__(self) -> None:
                self.items: list = []

            def put(self, item) -> None:  # noqa: ANN001
                self.items.append(item)

        q = _FakeQ()
        h._cmd_q = q
        h._started = True

        from autoptz.config.models import IdentityRecord

        rec = IdentityRecord(name="Person 3", enabled=False, labeled=False)
        h.ingest_identity(rec)
        assert any(name == "ingest_identity" for (name, _a, _k) in q.items)

    def test_ingest_identity_noop_before_start(self) -> None:
        cid = "proc-" + uuid.uuid4().hex[:8]
        h = ProcessWorkerHandle(cid, _config(cid), on_telemetry=lambda _m: None, db_path="")
        from autoptz.config.models import IdentityRecord

        # No queue yet -> must not raise.
        h.ingest_identity(IdentityRecord(name="Person 4", enabled=False, labeled=False))


def test_child_drain_routes_ingest_identity() -> None:
    import queue as _queue

    from autoptz.config.models import IdentityRecord
    from autoptz.engine.process_worker import _STOP, _drain_commands

    ingested: list = []

    class _FakeWorker:
        def ingest_identity(self, record) -> None:  # noqa: ANN001
            ingested.append(record.id)

    rec = IdentityRecord(name="Person 9", enabled=False, labeled=False)
    q: _queue.Queue = _queue.Queue()
    q.put(("ingest_identity", (rec,), {}))
    q.put((_STOP, (), {}))
    _drain_commands(_FakeWorker(), q)
    assert ingested == [rec.id]


class TestIdentityRelay:
    def test_harvested_identity_relays_to_sibling_not_source(self, qapp, monkeypatch) -> None:  # noqa: ANN001
        from autoptz.config.models import IdentityRecord
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient

        # Process-per-camera must be enabled so the relay's early-return guard passes.
        monkeypatch.setenv("AUTOPTZ_PROCESS_PER_CAMERA", "1")

        ingested: dict[str, list[str]] = {}

        class _FakeHandle:
            # Marks this as a process worker so the relay treats it like a real
            # ProcessWorkerHandle (cross-process sibling), not a threaded worker.
            _is_process_worker = True

            def __init__(self, camera_id, config, on_telemetry) -> None:  # noqa: ANN001
                self.camera_id = camera_id
                self.shm_name = f"cam_{camera_id[:8]}_preview"
                self._cb = None
                ingested[camera_id] = []

            def is_alive(self) -> bool:
                return True

            def set_identity_service(self, _s) -> None: ...  # noqa: ANN001
            def set_identity_callback(self, cb) -> None:  # noqa: ANN001
                self._cb = cb

            def set_inference_pool(self, _p) -> None: ...  # noqa: ANN001
            def set_features(self, _f) -> None: ...  # noqa: ANN001
            def ingest_identity(self, record) -> None:  # noqa: ANN001
                ingested[self.camera_id].append(record.id)

            def start(self) -> None: ...
            def stop(self) -> None: ...

        client = EngineClient()
        cid_a = client.addCamera("usb://0", "RelayA")
        cid_b = client.addCamera("usb://0", "RelayB")
        client.drain_commands()
        sup = Supervisor(client, store=None, worker_factory=_FakeHandle)
        sup.start()
        try:
            src = sup._workers[cid_a]
            rec = IdentityRecord(name="Person 1", enabled=False, labeled=False)
            src._cb(rec)  # child A harvested -> parent callback fires
            assert ingested[cid_b] == [rec.id], "sibling must receive the relay"
            assert ingested[cid_a] == [], "source must NOT receive its own relay"
        finally:
            sup.stop()

    def test_threaded_default_path_does_not_relay(self, qapp) -> None:  # noqa: ANN001
        """Threaded (in-process) workers share one gallery, so the relay must be a
        true no-op for them — even though ``CameraWorker`` now exposes
        ``ingest_identity``.  Relaying would needlessly take the supervisor lock and
        re-ingest a record every sibling already sees.
        """
        from autoptz.config.models import IdentityRecord
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient

        ingested: dict[str, list[str]] = {}

        class _FakeThreadedWorker:
            """Mimics the threaded ``CameraWorker`` surface (no process marker)."""

            def __init__(self, camera_id, config, on_telemetry) -> None:  # noqa: ANN001
                self.camera_id = camera_id
                self.shm_name = f"cam_{camera_id[:8]}_preview"
                self._cb = None
                ingested[camera_id] = []

            def is_alive(self) -> bool:
                return True

            def set_identity_service(self, _s) -> None: ...  # noqa: ANN001
            def set_identity_callback(self, cb) -> None:  # noqa: ANN001
                self._cb = cb

            def set_inference_pool(self, _p) -> None: ...  # noqa: ANN001
            def set_features(self, _f) -> None: ...  # noqa: ANN001
            def ingest_identity(self, record) -> None:  # noqa: ANN001
                ingested[self.camera_id].append(record.id)

            def start(self) -> None: ...
            def stop(self) -> None: ...

        client = EngineClient()
        cid_a = client.addCamera("usb://0", "ThreadA")
        cid_b = client.addCamera("usb://0", "ThreadB")
        client.drain_commands()
        sup = Supervisor(client, store=None, worker_factory=_FakeThreadedWorker)
        sup.start()
        try:
            src = sup._workers[cid_a]
            rec = IdentityRecord(name="Person 2", enabled=False, labeled=False)
            src._cb(rec)  # threaded worker harvested -> parent callback fires
            assert ingested[cid_b] == [], "threaded sibling must NOT be relayed to"
            assert ingested[cid_a] == [], "source must NOT receive its own relay"
        finally:
            sup.stop()


class TestMakeWorkerGuidance:
    def test_guidance_log_at_four_cameras(self, qapp, monkeypatch, caplog) -> None:  # noqa: ANN001
        import logging

        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient

        monkeypatch.setenv("AUTOPTZ_PROCESS_PER_CAMERA", "1")
        client = EngineClient()
        cids = [client.addCamera("usb://0", f"Guide{i}") for i in range(4)]
        client.drain_commands()
        sup = Supervisor(client, store=None)  # default factory -> process path is eligible

        with caplog.at_level(logging.INFO):
            sup._make_worker(cids[0], _config(cids[0]))
        assert any(
            "gil" in r.message.lower() or "process-per-camera" in r.message.lower()
            for r in caplog.records
        ), "expected a GIL-relief guidance log at >=4 cameras"


class TestChildLogging:
    def test_child_log_setup_installs_warning_handler(self) -> None:
        import logging

        from autoptz.engine.process_worker import _configure_child_logging

        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        try:
            root.handlers = []
            _configure_child_logging()
            assert root.handlers, "child must install at least one log handler"
            assert root.level <= logging.WARNING
        finally:
            root.handlers = saved_handlers
            root.setLevel(saved_level)


class TestEndToEndSpawn:
    """Spawn a real child process with a synthetic source (no camera, no models):
    frames must reach shared memory and telemetry must flow back, then stop cleanly."""

    def test_child_delivers_frames_and_telemetry_then_stops(self) -> None:
        from autoptz.engine.camera_worker import _PREVIEW_H, _PREVIEW_W
        from autoptz.engine.process_worker import run_camera_process
        from autoptz.engine.runtime.shm import ShmReader

        cid = "proc-" + uuid.uuid4().hex[:8]
        shm_name = f"cam_{cid[:8]}_preview"
        _cleanup_shm(shm_name)

        spec = WorkerSpec(
            camera_id=cid,
            config=_config(cid),
            db_path="",
            features={  # all off → pure capture/preview, no model loading in the child
                "detection": False,
                "tracking": False,
                "face_recognition": False,
                "pose": False,
                "reid": False,
            },
            synthetic=True,
        )
        ctx = mp.get_context("spawn")
        cmd_q = ctx.Queue()
        tel_q = ctx.Queue()
        idn_q = ctx.Queue()
        proc = ctx.Process(
            target=run_camera_process,
            args=(spec, cmd_q, tel_q, idn_q),
            daemon=True,
        )
        proc.start()
        try:
            # Telemetry should arrive once the child finishes its (heavy) imports;
            # generous timeout so a slow CI runner spawning + importing the full
            # stack doesn't flake.
            msg = tel_q.get(timeout=45.0)
            assert getattr(msg, "camera_id", None) == cid

            # A real frame should land in the child's shm preview ring.
            reader = None
            frame = None
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and frame is None:
                try:
                    if reader is None:
                        reader = ShmReader(shm_name, _PREVIEW_H, _PREVIEW_W)
                    result = reader.latest()
                    if result is not None:
                        frame = result[1]
                except Exception:
                    reader = None
                time.sleep(0.1)
            assert frame is not None, "no frame reached shared memory from the child process"
            assert frame.shape == (_PREVIEW_H, _PREVIEW_W, 3)
            assert int(frame.mean()) == 123  # the synthetic source's solid colour
            if reader is not None:
                reader.close()
        finally:
            cmd_q.put((_STOP, (), {}))
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=3.0)
            _cleanup_shm(shm_name)

        assert not proc.is_alive(), "child process did not stop on the _STOP sentinel"
