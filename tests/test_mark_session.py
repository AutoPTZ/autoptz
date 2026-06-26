from __future__ import annotations

import sys

from autoptz.ui.mark_session import (
    MARK_SESSION_KEY,
    MarkSession,
    clear_mark_session,
    clear_window_geometry,
    load_mark_session,
    relaunch_argv,
    store_mark_session,
)


class _Store:
    def __init__(self, kv=None) -> None:
        self.kv = dict(kv or {})

    def get_setting(self, key, default=None):
        return self.kv.get(key, default)

    def set_setting(self, key, value):
        self.kv[key] = value

    def delete_setting(self, key):
        self.kv.pop(key, None)


class TestRoundTrip:
    def test_store_load_clear(self) -> None:
        store = _Store()
        assert load_mark_session(store) is None
        s = MarkSession(
            profile="streams", source="ndi", floor_fps=24.0, max_cameras=8, dwell_s=20.0
        )
        store_mark_session(store, s)
        assert store.kv[MARK_SESSION_KEY]["profile"] == "streams"
        loaded = load_mark_session(store)
        assert loaded == s
        clear_mark_session(store)
        assert load_mark_session(store) is None

    def test_from_dict_defaults(self) -> None:
        s = MarkSession.from_dict({})
        assert s.profile == "full" and s.source == "synthetic"


class TestGeometryClear:
    def test_clears_both_keys(self) -> None:
        store = _Store({"win_geometry": "abc", "win_state": "def", "other": 1})
        clear_window_geometry(store)
        assert "win_geometry" not in store.kv and "win_state" not in store.kv
        assert store.kv["other"] == 1


class TestRelaunchArgv:
    def test_dev_argv(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "frozen", False, raising=False)
        argv = relaunch_argv(mark=True)
        assert argv[1:] == ["-m", "autoptz", "--mark"]
        assert relaunch_argv(mark=False)[1:] == ["-m", "autoptz"]

    def test_frozen_argv(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert relaunch_argv(mark=True) == [sys.executable, "--mark"]
        assert relaunch_argv(mark=False) == [sys.executable]
