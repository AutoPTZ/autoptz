"""Late-camera attach: the old model-server must DRAIN, never die mid-read.

``_attach_camera_to_model_server`` restarts the shared server so a camera added
after startup gets an IPC slot.  The old code set the graceful stop event and
then called ``terminate()`` on the very next line — killing a process that is
parked inside ``req_q.get()``.  Terminating a ``multiprocessing.Queue`` consumer
poisons the shared queue (the dead reader holds the queue's reader lock / leaves
a partial length-prefixed message in the pipe), so the RESPAWNED server never
receives a single request: every camera loses detection/tracking/face until the
whole app restarts.  Reproduced 5/5 with a standalone two-consumer harness and
live in-app after a remove + re-add.

These tests pin the drain-first contract: stop event set, bounded wait, and
``terminate()`` only as an escalation when the old server does not exit.

The wait itself must be NON-BLOCKING (``_attach_camera_to_model_server`` runs on
the GUI-thread ``tick()``): it hands the old process off to
``_poll_model_server_drain`` (ticked from ``_scan_model_server_health``) instead
of joining synchronously, so these tests drive the drain forward explicitly via
``_poll_model_server_drain(now)`` calls with an advancing synthetic clock —
mirroring how ``TestModelServerRespawnNonBlocking`` in test_supervisor_health.py
already drives the (pre-existing) spawn side of the same non-blocking pattern.
"""

from __future__ import annotations

import time
import uuid

from autoptz.engine.supervisor import _MS_DRAIN_TIMEOUT_S


class _FakeEvent:
    def __init__(self) -> None:
        self.set_calls = 0

    def is_set(self) -> bool:
        return self.set_calls > 0

    def set(self) -> None:
        self.set_calls += 1


class _FakeProc:
    """Records join/terminate calls; 'exits' on its own (no terminate needed)
    only if drains=True. is_alive() is polled non-blockingly across simulated
    ticks by _poll_model_server_drain — it is never blocked on via join()."""

    def __init__(self, *, drains: bool) -> None:
        self._drains = drains
        self._terminated = False
        self.calls: list[tuple[str, float | None]] = []

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(("join", timeout))

    def terminate(self) -> None:
        self.calls.append(("terminate", None))
        self._terminated = True

    def is_alive(self) -> bool:
        if self._terminated:
            return False
        return not self._drains


def _forged_supervisor(monkeypatch, *, drains: bool):  # noqa: ANN201
    """A Supervisor with hand-forged model-server state (no processes spawned)."""
    from autoptz.engine.supervisor import Supervisor
    from autoptz.ui.engine_client import EngineClient

    sup = Supervisor(EngineClient(), store=None)
    proc = _FakeProc(drains=drains)
    stop_ev = _FakeEvent()
    down_ev = _FakeEvent()
    sup._infer_req_q = object()
    sup._infer_resp_qs = {}
    sup._model_server_camera_ids = []
    sup._model_server_proc = proc
    sup._model_server_stop = stop_ev
    sup._model_server_down = down_ev

    respawned: list[float] = []
    monkeypatch.setattr(sup, "_respawn_model_server", lambda now: respawned.append(now))
    return sup, proc, stop_ev, respawned


class TestLateAttachDrainsOldServer:
    def test_drained_server_is_never_terminated(self, qapp, monkeypatch) -> None:  # noqa: ANN001
        sup, proc, stop_ev, respawned = _forged_supervisor(monkeypatch, drains=True)

        q = sup._attach_camera_to_model_server("cam-" + uuid.uuid4().hex[:8])

        assert q is not None, "attach must mint a response queue"
        assert stop_ev.set_calls >= 1, "graceful stop event must be set"
        # The attach call itself must be non-blocking: it hands the old process
        # off instead of joining it here (tick() runs on the GUI thread).
        assert proc.calls == []
        assert sup._ms_drain_proc is proc
        assert not respawned, "must not respawn before the drain is even polled"

        # One non-blocking poll tick: the server has already exited on its own
        # (drains=True) — respawn now, and it must NEVER have been terminated.
        sup._poll_model_server_drain(time.monotonic())

        assert ("terminate", None) not in proc.calls, (
            "terminating a server parked in req_q.get() poisons the shared request "
            "queue — a server that drains on its own must NEVER be terminated"
        )
        assert sup._ms_drain_proc is None
        assert respawned, "a fresh server must be spawned after the drain"

    def test_stuck_server_is_terminated_after_drain_window(self, qapp, monkeypatch) -> None:  # noqa: ANN001
        sup, proc, stop_ev, respawned = _forged_supervisor(monkeypatch, drains=False)

        q = sup._attach_camera_to_model_server("cam-" + uuid.uuid4().hex[:8])
        assert q is not None
        now = time.monotonic()

        # Well within the drain window: the wedged server must get a REAL chance
        # to exit before being killed — no terminate yet, no respawn yet.
        sup._poll_model_server_drain(now + 0.1)
        assert ("terminate", None) not in proc.calls
        assert sup._ms_drain_proc is proc
        assert not respawned

        # Past the drain deadline, still alive: escalate.
        sup._poll_model_server_drain(now + _MS_DRAIN_TIMEOUT_S + 0.1)
        names = [name for (name, _t) in proc.calls]
        assert "terminate" in names, "a wedged server must still be escalated to terminate"
        assert sup._ms_drain_proc is None
        assert respawned


class TestRemovePrunesServerSlots:
    """Removing a camera must drop its response queue + slot id, or every
    remove/re-add cycle leaks an mp.Queue (fds + feeder thread) and the next
    server respawn re-pickles queues for cameras that no longer exist."""

    def test_remove_camera_prunes_resp_queue_and_slot(self, qapp, monkeypatch) -> None:  # noqa: ANN001
        from autoptz.engine.runtime.messages import RemoveCameraCmd
        from autoptz.engine.supervisor import Supervisor
        from autoptz.ui.engine_client import EngineClient

        sup = Supervisor(EngineClient(), store=None)
        cid = "cam-" + uuid.uuid4().hex[:8]
        sup._infer_resp_qs = {cid: object(), "other": object()}
        sup._model_server_camera_ids = [cid, "other"]

        sup._on_remove_camera(RemoveCameraCmd(camera_id=cid))

        assert cid not in sup._infer_resp_qs
        assert cid not in sup._model_server_camera_ids
        assert "other" in sup._infer_resp_qs and "other" in sup._model_server_camera_ids
