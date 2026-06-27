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
        assert s.profile == "full" and s.source == "clip"
        # Resolution + model + ramp defaults favour a realistic-looking run.
        assert s.resolution == "1080p" and s.model == "small"
        assert s.floor_fps == 30.0 and s.max_cameras == 4 and s.dwell_s == 10.0

    def test_dataclass_defaults(self) -> None:
        s = MarkSession()
        assert s.profile == "full"
        assert s.source == "clip"
        assert s.floor_fps == 30.0
        assert s.max_cameras == 4
        assert s.dwell_s == 10.0
        assert s.resolution == "1080p"
        assert s.model == "small"


class TestResolutionAndModel:
    def test_resolution_and_model_round_trip(self) -> None:
        store = _Store()
        s = MarkSession(resolution="1080p", model="nano")
        store_mark_session(store, s)
        # The dict carries the new fields...
        assert store.kv[MARK_SESSION_KEY]["resolution"] == "1080p"
        assert store.kv[MARK_SESSION_KEY]["model"] == "nano"
        # ...and they survive a load round-trip.
        loaded = load_mark_session(store)
        assert loaded == s
        assert loaded.resolution == "1080p" and loaded.model == "nano"

    def test_resolution_size_maps_to_wh(self) -> None:
        assert MarkSession(resolution="720p").resolution_size() == (1280, 720)
        assert MarkSession(resolution="1080p").resolution_size() == (1920, 1080)
        assert MarkSession(resolution="4k").resolution_size() == (3840, 2160)
        # Unknown / malformed falls back to 720p so the engine never gets a bad size.
        assert MarkSession(resolution="garbage").resolution_size() == (1280, 720)

    def test_detector_tier_maps_model(self) -> None:
        # "auto" → default tier; "nano"/"small"/"medium" → the engine's tiers.
        assert MarkSession(model="auto").detector_tier() == "auto"
        assert MarkSession(model="nano").detector_tier() == "fast"
        assert MarkSession(model="small").detector_tier() == "balanced"
        assert MarkSession(model="medium").detector_tier() == "medium"
        # Unknown falls back to auto.
        assert MarkSession(model="weird").detector_tier() == "auto"


class TestClipSource:
    def test_default_source_is_clip(self) -> None:
        assert MarkSession().source == "clip"
        assert MarkSession().is_clip() is True
        assert MarkSession(source="synthetic").is_clip() is False
        assert MarkSession(source="ndi").is_clip() is False

    def test_clip_path_points_at_bundled_mp4(self) -> None:
        from pathlib import Path

        path = MarkSession().clip_path()
        assert isinstance(path, str)
        p = Path(path)
        assert p.is_absolute()
        assert p.name == "mark_people_1080p.mp4"
        # NOTE: we intentionally do NOT assert ``p.is_file()`` here — the clip is a
        # bundled asset that may be absent in a fresh clone / CI checkout.  The path
        # must still resolve correctly; presence is covered by clip_available().

    def test_clip_available_reflects_disk_state(self, monkeypatch) -> None:
        from pathlib import Path

        import autoptz.ui.mark_session as ms

        # Unpatched: clip_available() reflects the real resolved path's state.
        assert MarkSession().clip_available() == ms._clip_path().is_file()
        # Present on disk → available.
        monkeypatch.setattr(Path, "is_file", lambda self: True)
        assert MarkSession().clip_available() is True
        # Absent on disk → not available (no exception, just False).
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        assert MarkSession().clip_available() is False

    def test_clip_source_round_trips(self) -> None:
        store = _Store()
        s = MarkSession(source="clip")
        store_mark_session(store, s)
        assert store.kv[MARK_SESSION_KEY]["source"] == "clip"
        loaded = load_mark_session(store)
        assert loaded == s
        assert loaded.source == "clip"


class TestGeometryClear:
    def test_clears_both_keys(self) -> None:
        store = _Store({"win_geometry": "abc", "win_state": "def", "other": 1})
        clear_window_geometry(store)
        assert "win_geometry" not in store.kv and "win_state" not in store.kv
        assert store.kv["other"] == 1


class TestRelaunchArgv:
    """``relaunch``/``relaunch_argv`` are a DEPRECATED shim now.

    AutoPTZ Mark is an in-process swap (Help → Run AutoPTZ Mark…), so nothing in
    the app calls these anymore — but the helpers stay for backward compatibility
    and are still exercised here so the argv shape doesn't silently rot.
    """

    def test_dev_argv(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "frozen", False, raising=False)
        argv = relaunch_argv(mark=True)
        assert argv[1:] == ["-m", "autoptz", "--mark"]
        assert relaunch_argv(mark=False)[1:] == ["-m", "autoptz"]

    def test_frozen_argv(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert relaunch_argv(mark=True) == [sys.executable, "--mark"]
        assert relaunch_argv(mark=False) == [sys.executable]
