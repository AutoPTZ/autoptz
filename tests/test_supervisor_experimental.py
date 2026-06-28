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
