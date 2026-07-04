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

These tests pin the drain-first contract: stop event set, bounded join, and
``terminate()`` only as an escalation when the old server does not exit.
"""

from __future__ import annotations

import uuid


class _FakeEvent:
    def __init__(self) -> None:
        self.set_calls = 0

    def is_set(self) -> bool:
        return self.set_calls > 0

    def set(self) -> None:
        self.set_calls += 1


class _FakeProc:
    """Records the join/terminate order; 'exits' on join only if drained_ok."""

    def __init__(self, *, drains: bool) -> None:
        self._drains = drains
        self._alive = True
        self.calls: list[tuple[str, float | None]] = []

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(("join", timeout))
        if self._drains:
            self._alive = False

    def terminate(self) -> None:
        self.calls.append(("terminate", None))
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


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
        assert ("terminate", None) not in proc.calls, (
            "terminating a server parked in req_q.get() poisons the shared request "
            "queue — a server that drains on its own must NEVER be terminated"
        )
        joins = [t for (name, t) in proc.calls if name == "join"]
        assert joins and joins[0] and joins[0] >= 1.0, (
            f"the drain join must give the serve loop a real window to exit (got timeouts {joins})"
        )
        assert respawned, "a fresh server must be spawned after the drain"

    def test_stuck_server_is_terminated_after_drain_window(self, qapp, monkeypatch) -> None:  # noqa: ANN001
        sup, proc, stop_ev, respawned = _forged_supervisor(monkeypatch, drains=False)

        q = sup._attach_camera_to_model_server("cam-" + uuid.uuid4().hex[:8])

        assert q is not None
        names = [name for (name, _t) in proc.calls]
        assert "terminate" in names, "a wedged server must still be escalated to terminate"
        assert names.index("join") < names.index("terminate"), (
            "the drain join must come BEFORE terminate — kill only after the "
            "graceful window expires"
        )
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
