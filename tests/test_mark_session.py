from __future__ import annotations

import sys

from autoptz.ui.mark_session import (
    CLIP_LIBRARY,
    DEFAULT_CLIP_ID,
    MARK_SESSION_KEY,
    ClipMetadata,
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
        # The default clip is the HD crowd-crossing clip (DEFAULT_CLIP_ID == "crowd").
        assert p.name == CLIP_LIBRARY[DEFAULT_CLIP_ID].filename == "mark_crowd_30.mp4"
        # NOTE: we intentionally do NOT assert ``p.is_file()`` here — the clip is a
        # bundled asset that may be absent in a fresh clone / CI checkout.  The path
        # must still resolve correctly; presence is covered by clip_available().

    def test_clip_available_reflects_disk_state(self, monkeypatch) -> None:
        from pathlib import Path

        import autoptz.ui.mark_session as ms

        # Unpatched: clip_available() reflects the real resolved path's state.  The
        # default session resolves to the default clip (crowd), so compare against
        # that clip's path — NOT the bare _clip_path() (which is the original asset).
        default_path = ms._clip_path(ms.CLIP_LIBRARY[ms.DEFAULT_CLIP_ID].filename)
        assert MarkSession().clip_available() == default_path.is_file()
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


class TestClipLibrary:
    def test_clip_library_registry(self) -> None:
        # Every registry entry is a ClipMetadata with all fields populated.
        assert isinstance(CLIP_LIBRARY, dict)
        assert CLIP_LIBRARY  # non-empty
        for clip_id, meta in CLIP_LIBRARY.items():
            assert isinstance(meta, ClipMetadata)
            assert meta.id == clip_id
            assert isinstance(meta.id, str) and meta.id
            assert isinstance(meta.filename, str) and meta.filename
            assert isinstance(meta.label, str) and meta.label
            assert isinstance(meta.native_fps, float) and meta.native_fps > 0
            assert isinstance(meta.purpose, str) and meta.purpose
            # v3: every entry carries a native resolution (w, h both > 0) and a
            # non-empty tuple of capability tags.
            assert isinstance(meta.native_resolution, tuple)
            assert len(meta.native_resolution) == 2
            w, h = meta.native_resolution
            assert isinstance(w, int) and isinstance(h, int)
            assert w > 0 and h > 0
            assert isinstance(meta.capability_tags, tuple)
            assert meta.capability_tags  # non-empty
            assert all(isinstance(t, str) and t for t in meta.capability_tags)
        # The default clip id is present and is the HD crowd clip.
        assert DEFAULT_CLIP_ID == "crowd"
        assert DEFAULT_CLIP_ID in CLIP_LIBRARY
        # The bundled capability scenes (24/30/60fps + a dedicated faces scene)
        # are all registered.
        assert {
            "crowd",
            "pedestrians",
            "cinematic_24",
            "cinematic_60",
            "faces",
        } == set(CLIP_LIBRARY)
        # The library spans the three requested native cadences.
        assert {m.native_fps for m in CLIP_LIBRARY.values()} == {24.0, 30.0, 60.0}
        # Every requested capability is covered by at least one scene.
        all_tags = {t for m in CLIP_LIBRARY.values() for t in m.capability_tags}
        assert {"tracking", "reid", "center-stage", "face"} <= all_tags

    def test_clip_id_round_trip(self) -> None:
        store = _Store()
        s = MarkSession(source="clip", clip_id="cinematic_60")
        store_mark_session(store, s)
        # The dict carries clip_id...
        assert store.kv[MARK_SESSION_KEY]["clip_id"] == "cinematic_60"
        # ...and it survives a load round-trip.
        loaded = load_mark_session(store)
        assert loaded == s
        assert loaded.clip_id == "cinematic_60"
        # A back-compat dict missing clip_id defaults to "".
        assert MarkSession.from_dict({}).clip_id == ""

    def test_clip_info_lookup(self, caplog) -> None:
        import logging

        # A known id resolves to its registry metadata.
        meta = MarkSession(clip_id="cinematic_60").clip_info()
        assert meta is CLIP_LIBRARY["cinematic_60"]
        assert meta.native_fps == 60.0
        # Empty id falls back to the default clip (no warning).
        empty = MarkSession(clip_id="").clip_info()
        assert empty is CLIP_LIBRARY[DEFAULT_CLIP_ID]
        # An unknown id falls back to the default AND logs a warning.
        with caplog.at_level(logging.WARNING):
            unknown = MarkSession(clip_id="does_not_exist").clip_info()
        assert unknown is CLIP_LIBRARY[DEFAULT_CLIP_ID]
        assert any("does_not_exist" in r.getMessage() for r in caplog.records)

    def test_clip_path_from_id(self) -> None:
        from pathlib import Path

        path = MarkSession(clip_id="cinematic_60").clip_path()
        assert isinstance(path, str)
        p = Path(path)
        assert p.is_absolute()
        assert p.name == CLIP_LIBRARY["cinematic_60"].filename
        # An empty id resolves to the default clip's filename.
        dflt = Path(MarkSession(clip_id="").clip_path())
        assert dflt.name == CLIP_LIBRARY[DEFAULT_CLIP_ID].filename
        # NOTE: we intentionally do NOT assert ``p.is_file()`` here — the new clips
        # are bundled assets produced in parallel that may be absent in a fresh
        # clone / CI checkout.  Presence is covered by clip_available().

    def test_target_fps_follows_clip_native_cadence(self) -> None:
        # For a CLIP source the benchmark target is the clip's native fps (a 24fps
        # clip can't sustain 30; a 60fps clip is graded against 60) — NOT floor_fps.
        assert (
            MarkSession(source="clip", clip_id="cinematic_24", floor_fps=30.0).target_fps() == 24.0
        )
        assert (
            MarkSession(source="clip", clip_id="cinematic_60", floor_fps=30.0).target_fps() == 60.0
        )
        assert MarkSession(source="clip", clip_id="pedestrians").target_fps() == 30.0
        # Empty id → default clip's native fps.
        assert MarkSession(source="clip", clip_id="").target_fps() == float(
            CLIP_LIBRARY[DEFAULT_CLIP_ID].native_fps
        )
        # For synthetic / NDI (render-any-rate) sources the user's floor_fps wins.
        assert MarkSession(source="synthetic", floor_fps=45.0).target_fps() == 45.0
        assert MarkSession(source="ndi", floor_fps=24.0).target_fps() == 24.0


class TestClipCapabilitiesV3:
    """Slice 1 of Mark v3: capability tags + native resolution + variants table."""

    def test_native_resolutions_match_spec(self) -> None:
        assert CLIP_LIBRARY["crowd"].native_resolution == (1280, 720)
        assert CLIP_LIBRARY["pedestrians"].native_resolution == (1920, 1080)
        assert CLIP_LIBRARY["cinematic_24"].native_resolution == (1920, 1080)
        assert CLIP_LIBRARY["cinematic_60"].native_resolution == (1920, 1080)

    def test_capability_tags_match_spec(self) -> None:
        assert CLIP_LIBRARY["crowd"].capability_tags == ("tracking", "reid")
        assert CLIP_LIBRARY["pedestrians"].capability_tags == ("tracking",)
        assert CLIP_LIBRARY["cinematic_24"].capability_tags == ("center-stage",)
        assert CLIP_LIBRARY["cinematic_60"].capability_tags == ("center-stage",)
        assert CLIP_LIBRARY["faces"].capability_tags == ("face",)

    def test_capability_tags_method_returns_clip_tags(self) -> None:
        assert MarkSession(clip_id="crowd").capability_tags() == ("tracking", "reid")
        assert MarkSession(clip_id="faces").capability_tags() == ("face",)
        # Empty id resolves to the default clip's tags.
        assert (
            MarkSession(clip_id="").capability_tags()
            == CLIP_LIBRARY[DEFAULT_CLIP_ID].capability_tags
        )

    def test_available_variants_shape(self) -> None:
        variants = MarkSession(clip_id="crowd").available_variants()
        assert isinstance(variants, list)
        assert variants
        for v in variants:
            assert {"res", "fps", "res_tag", "fps_tag", "synthetic"} <= set(v)

    def test_available_variants_crowd_synthetic_tagging(self) -> None:
        # crowd master is 720p30: 1080p/4k targets are upscaled-synthetic, 60fps
        # targets are interpolated-synthetic, and 720p/30 is native (not synthetic).
        variants = MarkSession(clip_id="crowd").available_variants()
        by_key = {(v["res"], v["fps"]): v for v in variants}

        native = by_key[((1280, 720), 30.0)]
        assert native["res_tag"] == "native"
        assert native["fps_tag"] == "native"
        assert native["synthetic"] is False

        upscaled_1080 = by_key[((1920, 1080), 30.0)]
        assert upscaled_1080["res_tag"] == "upscaled"
        assert upscaled_1080["synthetic"] is True

        upscaled_4k = by_key[((3840, 2160), 30.0)]
        assert upscaled_4k["res_tag"] == "upscaled"
        assert upscaled_4k["synthetic"] is True

        interpolated_60 = by_key[((1280, 720), 60.0)]
        assert interpolated_60["fps_tag"] == "interpolated"
        assert interpolated_60["synthetic"] is True

    def test_available_variants_cinematic_60_has_native_60(self) -> None:
        # cinematic_60 master is 1080p60: the 1080p/60 combo is fully native.
        variants = MarkSession(clip_id="cinematic_60").available_variants()
        by_key = {(v["res"], v["fps"]): v for v in variants}
        native_60 = by_key[((1920, 1080), 60.0)]
        assert native_60["res_tag"] == "native"
        assert native_60["fps_tag"] == "native"
        assert native_60["synthetic"] is False


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
