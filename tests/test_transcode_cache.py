from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from autoptz.engine.pipeline.transcode_cache import CACHE_VERSION, TranscodeCache


def _make_source_clip(path, *, res=(640, 480), fps=30.0, frames=15):
    """Write a tiny real .mp4 with cv2 so build can be exercised for real."""
    w, h = res
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), float(fps), (w, h))
    assert writer.isOpened(), "could not open source VideoWriter"
    for i in range(frames):
        frame = np.full((h, w, 3), (i * 7) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


def _probe(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cannot decode {path}"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return {"res": (w, h), "fps": fps, "count": count}


class TestVariantPath:
    def test_format(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        p = tc._variant_path("clipA", (1920, 1080), 30.0)
        assert p == tmp_path / "clipA" / f"1920x1080_30fps_v{CACHE_VERSION}.mp4"

    def test_fps_floored_in_name(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        p = tc._variant_path("clipA", (1280, 720), 59.94)
        assert p.name == f"1280x720_59fps_v{CACHE_VERSION}.mp4"


class TestGetCachedVariant:
    def test_none_when_absent(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        assert tc.get_cached_variant("nope", (1280, 720), 30.0) is None

    def test_none_when_empty(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        p = tc._variant_path("clipA", (1280, 720), 30.0)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()  # zero-byte → not a valid cached variant
        assert tc.get_cached_variant("clipA", (1280, 720), 30.0) is None

    def test_path_when_present(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        p = tc._variant_path("clipA", (1280, 720), 30.0)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 10)
        assert tc.get_cached_variant("clipA", (1280, 720), 30.0) == p


class TestValidCombos:
    def test_720_master_tags(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        combos = tc.valid_combos((1280, 720), 30.0)
        by_key = {(c["res"], c["fps"]): c for c in combos}

        # 720 native, 1080 + 4k are upscaled (synthetic)
        assert by_key[((1280, 720), 30.0)]["res_tag"] == "native"
        assert by_key[((1920, 1080), 30.0)]["res_tag"] == "upscaled"
        assert by_key[((1920, 1080), 30.0)]["synthetic"] is True
        assert by_key[((3840, 2160), 30.0)]["res_tag"] == "upscaled"

        # fps tagging at native res
        assert by_key[((1280, 720), 24.0)]["fps_tag"] == "resampled"
        assert by_key[((1280, 720), 30.0)]["fps_tag"] == "native"
        assert by_key[((1280, 720), 60.0)]["fps_tag"] == "interpolated"
        assert by_key[((1280, 720), 60.0)]["synthetic"] is True

    def test_fully_native_not_synthetic(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        combos = tc.valid_combos((1280, 720), 30.0)
        native = next(c for c in combos if c["res"] == (1280, 720) and c["fps"] == 30.0)
        assert native["res_tag"] == "native"
        assert native["fps_tag"] == "native"
        assert native["synthetic"] is False

    def test_downscaled(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        combos = tc.valid_combos((3840, 2160), 30.0)
        by_key = {(c["res"], c["fps"]): c for c in combos}
        assert by_key[((1280, 720), 30.0)]["res_tag"] == "downscaled"
        # downscale alone is not synthetic
        assert by_key[((1280, 720), 30.0)]["synthetic"] is False


class TestBuildCachedVariant:
    def test_builds_decodable_at_target_res(self, tmp_path) -> None:
        src = _make_source_clip(tmp_path / "src.mp4", res=(640, 480), fps=30.0, frames=15)
        tc = TranscodeCache(cache_dir=tmp_path / "cache")
        out = tc.build_cached_variant(
            "clipA",
            src,
            master_res=(640, 480),
            master_fps=30.0,
            target_res=(320, 240),
            target_fps=30.0,
        )
        assert out.exists() and out.stat().st_size > 0
        info = _probe(out)
        assert info["res"] == (320, 240)
        assert abs(info["fps"] - 30.0) < 1.0
        # looped to a minimum ~8s of output @30fps
        assert info["count"] >= 8 * 30 - 5

    def test_interpolated_has_more_frames(self, tmp_path) -> None:
        src = _make_source_clip(tmp_path / "src.mp4", res=(320, 240), fps=30.0, frames=30)
        tc = TranscodeCache(cache_dir=tmp_path / "cache")
        out60 = tc.build_cached_variant(
            "clipA",
            src,
            master_res=(320, 240),
            master_fps=30.0,
            target_res=(320, 240),
            target_fps=60.0,
        )
        info = _probe(out60)
        assert abs(info["fps"] - 60.0) < 1.0
        # ~8s @ 60fps via frame duplication
        assert info["count"] >= 8 * 60 - 10

    def test_atomic_no_leftover_tmp(self, tmp_path) -> None:
        src = _make_source_clip(tmp_path / "src.mp4", res=(320, 240), fps=30.0, frames=15)
        cache = tmp_path / "cache"
        tc = TranscodeCache(cache_dir=cache)
        out = tc.build_cached_variant(
            "clipA",
            src,
            master_res=(320, 240),
            master_fps=30.0,
            target_res=(320, 240),
            target_fps=30.0,
        )
        leftovers = list(out.parent.glob("*.tmp.mp4"))
        assert leftovers == []

    def test_raises_and_cleans_tmp_on_bad_source(self, tmp_path) -> None:
        bad = tmp_path / "missing.mp4"  # does not exist → cv2 cannot open
        cache = tmp_path / "cache"
        tc = TranscodeCache(cache_dir=cache)
        with pytest.raises(Exception):  # noqa: B017,PT011
            tc.build_cached_variant(
                "clipA",
                bad,
                master_res=(320, 240),
                master_fps=30.0,
                target_res=(320, 240),
                target_fps=30.0,
            )
        # no half-written tmp/final left behind under the clip dir
        clip_dir = cache / "clipA"
        if clip_dir.exists():
            assert list(clip_dir.glob("*.tmp.mp4")) == []
            assert list(clip_dir.glob("*.mp4")) == []


class TestCleanup:
    def test_evicts_oldest_over_budget(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        # three files of 400 bytes each → 1200 total; budget 1000 → evict oldest
        paths = []
        for i, clip in enumerate(("a", "b", "c")):
            p = tc._variant_path(clip, (1280, 720), 30.0)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * 400)
            # stagger mtimes so "a" is oldest, "c" newest
            ts = 1_000_000 + i * 100
            os.utime(p, (ts, ts))
            paths.append(p)

        tc.cleanup(max_total_bytes=1000)

        assert not paths[0].exists()  # oldest evicted
        assert paths[1].exists()
        assert paths[2].exists()

    def test_noop_under_budget(self, tmp_path) -> None:
        tc = TranscodeCache(cache_dir=tmp_path)
        p = tc._variant_path("a", (1280, 720), 30.0)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 100)
        tc.cleanup(max_total_bytes=1000)
        assert p.exists()
