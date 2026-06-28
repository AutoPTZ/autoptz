"""Supervisor publishes experimental_features into os.environ at start."""

from __future__ import annotations

import os
from typing import Any

from autoptz.config.store import ConfigStore
from autoptz.engine.supervisor import Supervisor


class _StubClient:
    """Minimal stand-in for EngineClient (Supervisor only touches cameraModel)."""

    class _Model:
        @staticmethod
        def camera_ids() -> list[str]:
            return []

    def __init__(self) -> None:
        self.cameraModel = self._Model()

    def push_telemetry(self, *_a: Any, **_k: Any) -> None:  # pragma: no cover
        pass


def _sup(tmp_path: Any) -> tuple[Supervisor, ConfigStore]:
    store = ConfigStore(db_path=tmp_path / "exp.db", debounce_s=0)
    return Supervisor(_StubClient(), store=store), store


def test_enabled_nondefault_flag_is_set(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.delenv("AUTOPTZ_PTZ_PUMP", raising=False)
    sup, store = _sup(tmp_path)
    store.set_setting("experimental_features", {"AUTOPTZ_PTZ_PUMP": "1"})
    sup._apply_experimental_env()
    assert os.environ.get("AUTOPTZ_PTZ_PUMP") == "1"


def test_model_server_init_failure_clears_infer_queues(tmp_path: Any, monkeypatch: Any) -> None:
    """If the model-server fails to spawn AFTER its queues were created, the supervisor
    must clear the infer queues — otherwise camera children see non-None handles and
    delegate detection to a server that doesn't exist (every detect times out) instead
    of falling back to their own local detector.
    """
    import multiprocessing as mp

    monkeypatch.setenv("AUTOPTZ_MODEL_SERVER", "1")
    sup, _store = _sup(tmp_path)
    real_ctx = mp.get_context("spawn")

    class _FakeProc:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("cannot spawn")  # fail AFTER queues are built

    class _FakeCtx:
        Queue = staticmethod(real_ctx.Queue)
        Event = staticmethod(real_ctx.Event)

        def Process(self, *_a: Any, **_k: Any) -> _FakeProc:  # noqa: N802
            return _FakeProc()

    monkeypatch.setattr(mp, "get_context", lambda *_a, **_k: _FakeCtx())
    sup._ensure_model_server(["camA", "camB"])

    assert sup._model_server_proc is None
    assert sup._infer_req_q is None  # cleared so children build a local detector
    assert sup._infer_resp_qs == {}


def test_choice_flag_value_is_set(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.delenv("AUTOPTZ_REID_DEVICE", raising=False)
    sup, store = _sup(tmp_path)
    store.set_setting("experimental_features", {"AUTOPTZ_REID_DEVICE": "cpu"})
    sup._apply_experimental_env()
    assert os.environ.get("AUTOPTZ_REID_DEVICE") == "cpu"


def test_default_value_pops_existing_env(tmp_path: Any, monkeypatch: Any) -> None:
    # A stale env var left from a prior run is cleared when the saved value is
    # the engine default (so the in-code fallback runs).
    monkeypatch.setenv("AUTOPTZ_PTZ_PUMP", "1")
    sup, store = _sup(tmp_path)
    store.set_setting("experimental_features", {"AUTOPTZ_PTZ_PUMP": "0"})
    sup._apply_experimental_env()
    assert "AUTOPTZ_PTZ_PUMP" not in os.environ


def test_absent_dict_leaves_unmanaged_env_untouched(tmp_path: Any, monkeypatch: Any) -> None:
    # Feature-inactive baseline: with nothing persisted, a directly-set env var
    # (exported by the operator, or by another test) is NOT clobbered.  Only keys
    # the user actually persists in experimental_features are managed.
    monkeypatch.setenv("AUTOPTZ_PROCESS_PER_CAMERA", "1")
    sup, _store = _sup(tmp_path)  # nothing persisted
    sup._apply_experimental_env()
    assert os.environ.get("AUTOPTZ_PROCESS_PER_CAMERA") == "1"


def test_persisted_default_pops_stale_managed_key(tmp_path: Any, monkeypatch: Any) -> None:
    # When the user HAS persisted a key at its engine default, a stale env var
    # from a prior selection is cleared so the in-code fallback runs.
    monkeypatch.setenv("AUTOPTZ_PROCESS_PER_CAMERA", "1")
    sup, store = _sup(tmp_path)
    store.set_setting("experimental_features", {"AUTOPTZ_PROCESS_PER_CAMERA": "0"})
    sup._apply_experimental_env()
    assert "AUTOPTZ_PROCESS_PER_CAMERA" not in os.environ


def test_tracking_keys_in_dict_are_ignored_for_env(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.delenv("stage_spread", raising=False)
    sup, store = _sup(tmp_path)
    store.set_setting("experimental_features", {"stage_spread": False})
    sup._apply_experimental_env()
    # Non-env-flag keys never reach os.environ.
    assert "stage_spread" not in os.environ
