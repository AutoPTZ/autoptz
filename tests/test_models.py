"""Tests for the model bootstrap (autoptz.engine.runtime.models.ModelManager).

All network / ultralytics export is mocked so these run offline and fast.
The contract under test: ``ensure_detector()`` resolves the env override, reuses
a cached ONNX, **prefers a prebuilt torch-free ONNX download**, falls back to
the ultralytics export, and NEVER raises — it returns ``None``
(live-preview-only) when neither acquisition path is reachable.
"""

from __future__ import annotations

import hashlib
import sys
import types
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoptz.engine.runtime.models import ModelManager, app_model_specs, detector_model_for_tier


@pytest.fixture(autouse=True)
def _disable_prebuilt_by_default(monkeypatch, tmp_path) -> None:
    """Disable the prebuilt-download path by default for the export-focused tests.

    Tests that specifically exercise the prebuilt path re-enable it locally by
    setting ``AUTOPTZ_MODEL_URL`` and mocking ``urllib.request.urlopen``.  With
    no URL set, :meth:`ModelManager._download_prebuilt` returns ``None`` without
    touching the network, so the existing ultralytics-export assertions hold.

    Also points the bundled-models lookup at an empty dir so a local dev build
    that populated ``autoptz/models/`` can't shadow the download/export paths.
    """
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", "")
    monkeypatch.delenv("AUTOPTZ_NO_MODEL_EXPORT", raising=False)
    empty = tmp_path / "no-bundled-models"
    empty.mkdir()
    monkeypatch.setattr("autoptz.engine.runtime.models.bundled_models_dir", lambda: empty)


# ── env override ──────────────────────────────────────────────────────────────


def test_env_override_returns_path_when_file_exists(tmp_path, monkeypatch) -> None:
    model = tmp_path / "mymodel.onnx"
    model.write_bytes(b"fake-onnx")
    monkeypatch.setenv("AUTOPTZ_MODEL_PATH", str(model))

    mgr = ModelManager(cache_dir=tmp_path / "cache")
    assert mgr.ensure_detector() == str(model)


def test_env_override_ignored_when_file_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOPTZ_MODEL_PATH", str(tmp_path / "nope.onnx"))
    # No ultralytics installed → falls through to None, never raises.
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    mgr = ModelManager(cache_dir=tmp_path / "cache")
    assert mgr.ensure_detector() is None


def test_cached_only_detector_does_not_download_or_export(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    captured = _install_fake_ultralytics(monkeypatch)

    mgr = ModelManager(cache_dir=tmp_path / "cache")
    assert mgr.ensure_detector(tier="balanced", allow_download=False) is None
    assert "not cached" in mgr.last_error
    assert "weights" not in captured


# ── bundled (shipped-inside-the-app) models ───────────────────────────────────


def test_bundled_model_resolves_without_cache_download_or_export(tmp_path, monkeypatch) -> None:
    """A model shipped in autoptz/models resolves offline — no cache, net, or torch."""
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "yolo11n.onnx").write_bytes(b"x" * (1 << 19))
    monkeypatch.setattr("autoptz.engine.runtime.models.bundled_models_dir", lambda: bundled)
    # Neither ultralytics nor a download URL is available.
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    mgr = ModelManager(cache_dir=tmp_path / "cache")
    assert mgr.ensure_detector(tier="fast") == str(bundled / "yolo11n.onnx")
    # Even with allow_download=False (the in-app default), a bundled model is fine.
    assert mgr.ensure_detector(tier="fast", allow_download=False) == str(bundled / "yolo11n.onnx")


def test_bundled_model_reported_included_and_not_removable(tmp_path, monkeypatch) -> None:
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "yolo11n.onnx").write_bytes(b"x" * (1 << 19))
    monkeypatch.setattr("autoptz.engine.runtime.models.bundled_models_dir", lambda: bundled)

    mgr = ModelManager(cache_dir=tmp_path / "cache")
    rows = {r["key"]: r for r in mgr.app_model_statuses()}
    assert rows["detector_fast"]["state"] == "ok"
    assert rows["detector_fast"]["cached"] is True  # available
    assert rows["detector_fast"]["bundled"] is True
    assert rows["detector_fast"]["removable"] is False
    # A non-bundled tier is still missing (would need a download).
    assert rows["detector_accurate"]["state"] == "missing"


# ── cached ONNX reuse ─────────────────────────────────────────────────────────


def test_cached_onnx_is_reused_without_export(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    onnx = cache / "yolo11n.onnx"
    onnx.write_bytes(b"cached")

    # Make ultralytics import *fail* — if reuse works, export is never attempted.
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    mgr = ModelManager(cache_dir=cache)
    assert mgr.ensure_detector() == str(onnx)


def test_detector_tier_maps_to_expected_weights() -> None:
    assert detector_model_for_tier("auto") == "yolo11n.pt"
    assert detector_model_for_tier("fast") == "yolo11n.pt"
    assert detector_model_for_tier("balanced") == "yolo11s.pt"
    assert detector_model_for_tier("medium") == "yolo11m.pt"
    assert detector_model_for_tier("bogus") == "yolo11n.pt"


def test_detector_tier_includes_rtdetr() -> None:
    assert detector_model_for_tier("rtdetr") == "rtdetr-l.pt"
    assert detector_model_for_tier("rtdetr-x") == "rtdetr-x.pt"


def test_ensure_detector_int8_missing_file_returns_none(tmp_path) -> None:
    mgr = ModelManager(cache_dir=tmp_path / "cache")
    assert mgr.ensure_detector_int8(tmp_path / "nope.onnx") is None


def test_ensure_detector_int8_reuses_cached(tmp_path) -> None:
    """An existing ``*.int8.onnx`` is returned without re-quantizing."""
    fp32 = tmp_path / "yolo11n.onnx"
    fp32.write_bytes(b"\x00" * 1024)
    int8 = tmp_path / "yolo11n.int8.onnx"
    int8.write_bytes(b"\x01" * 1024)
    mgr = ModelManager(cache_dir=tmp_path)
    out = mgr.ensure_detector_int8(fp32)
    assert out == str(int8)
    assert int8.read_bytes() == b"\x01" * 1024  # untouched (no re-quantization)


def test_maybe_quantize_int8_is_noop_without_env(monkeypatch) -> None:
    from autoptz.engine.worker.stacks import _maybe_quantize_int8

    monkeypatch.delenv("AUTOPTZ_PRECISION", raising=False)
    assert _maybe_quantize_int8("/models/yolo11n.onnx") == "/models/yolo11n.onnx"


def test_maybe_quantize_int8_uses_manager_when_enabled(monkeypatch) -> None:
    from autoptz.engine.runtime import models as models_mod
    from autoptz.engine.worker import stacks

    monkeypatch.setenv("AUTOPTZ_PRECISION", "int8")

    class _FakeMgr:
        def ensure_detector_int8(self, p):
            return "/models/yolo11n.int8.onnx"

    monkeypatch.setattr(models_mod, "default_manager", lambda: _FakeMgr())
    assert stacks._maybe_quantize_int8("/models/yolo11n.onnx") == "/models/yolo11n.int8.onnx"


# ── download + export path (mocked ultralytics) ───────────────────────────────


def _install_fake_ultralytics(
    monkeypatch, *, export_writes: bool = True, raise_on_export: bool = False
) -> dict:
    """Install a fake ``ultralytics`` module exposing ``YOLO``.

    Returns a dict capturing the kwargs the test asserts on.
    """
    captured: dict = {}

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            captured["weights"] = weights

        def export(self, **kwargs):
            captured["export_kwargs"] = kwargs
            if raise_on_export:
                raise RuntimeError("export blew up")
            # ultralytics exports next to the .pt (cwd is the cache dir here).
            out = Path.cwd() / (Path(captured["weights"]).stem + ".onnx")
            if export_writes:
                out.write_bytes(b"exported-onnx")
            return str(out)

    fake_mod = types.ModuleType("ultralytics")
    fake_mod.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)
    return captured


def test_download_export_produces_cached_onnx(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    cache = tmp_path / "cache"
    captured = _install_fake_ultralytics(monkeypatch)

    mgr = ModelManager(cache_dir=cache)
    result = mgr.ensure_detector()

    onnx = cache / "yolo11n.onnx"
    assert result == str(onnx)
    assert onnx.is_file()
    # Export must use the NMS-free settings detect.py expects.
    assert captured["export_kwargs"]["format"] == "onnx"
    assert captured["export_kwargs"]["nms"] is False
    assert captured["export_kwargs"]["dynamic"] is False
    assert captured["export_kwargs"]["opset"] == 12
    assert captured["weights"] == "yolo11n.pt"


def test_export_failure_returns_none_not_raise(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    _install_fake_ultralytics(monkeypatch, raise_on_export=True)
    mgr = ModelManager(cache_dir=tmp_path / "cache")
    assert mgr.ensure_detector() is None  # logged, not raised


def test_missing_ultralytics_returns_none(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # import → ImportError
    mgr = ModelManager(cache_dir=tmp_path / "cache")
    assert mgr.ensure_detector() is None


def test_export_disabled_env_skips_ultralytics(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.setenv("AUTOPTZ_NO_MODEL_EXPORT", "1")
    captured = _install_fake_ultralytics(monkeypatch)

    mgr = ModelManager(cache_dir=tmp_path / "cache")
    assert mgr.ensure_detector() is None
    assert "disabled" in mgr.last_error
    assert "export_kwargs" not in captured


def test_export_disabled_env_applies_to_pose(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_POSE_MODEL_PATH", raising=False)
    monkeypatch.setenv("AUTOPTZ_NO_MODEL_EXPORT", "true")
    captured = _install_fake_ultralytics(monkeypatch)

    mgr = ModelManager(cache_dir=tmp_path / "cache")
    assert mgr.ensure_pose() is None
    assert "disabled" in mgr.last_error
    assert "export_kwargs" not in captured


def test_ensure_app_models_fetches_detector_tiers_and_pose(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.delenv("AUTOPTZ_POSE_MODEL_PATH", raising=False)
    captured = _install_fake_ultralytics(monkeypatch)
    progress = []

    mgr = ModelManager(cache_dir=tmp_path / "cache")
    rows = mgr.ensure_app_models(
        progress=lambda label, value, total: progress.append((label, value, total))
    )

    assert [r["state"] for r in rows] == ["ok", "ok", "ok", "ok"]
    assert {Path(r["path"]).name for r in rows} == {
        "yolo11n.onnx",
        "yolo11s.onnx",
        "yolo11m.onnx",
        "yolo11n-pose.onnx",
    }
    assert progress[0] == ("Fast detector", 0, 4)
    assert progress[-1] == ("Pose model", 4, 4)
    assert captured["weights"] == "yolo11n-pose.pt"


def test_ensure_app_models_can_skip_pose(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    captured = _install_fake_ultralytics(monkeypatch)

    mgr = ModelManager(cache_dir=tmp_path / "cache")
    rows = mgr.ensure_app_models(include_pose=False)

    assert [r["name"] for r in rows] == [
        "Fast detector",
        "Balanced detector",
        "Accurate detector",
    ]
    assert captured["weights"] == "yolo11m.pt"


def test_app_model_statuses_describe_catalog_and_cache(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "yolo11n.onnx").write_bytes(b"model")

    mgr = ModelManager(cache_dir=cache)
    rows = mgr.app_model_statuses()

    assert {row["key"] for row in rows} == {spec["key"] for spec in app_model_specs()}
    fast = next(row for row in rows if row["key"] == "detector_fast")
    pose = next(row for row in rows if row["key"] == "pose")
    assert fast["cached"] is True
    assert fast["state"] == "ok"
    assert fast["size_bytes"] > 0
    assert pose["cached"] is False
    assert pose["state"] == "missing"


def test_ensure_app_models_can_fetch_selected_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.delenv("AUTOPTZ_POSE_MODEL_PATH", raising=False)
    _install_fake_ultralytics(monkeypatch)

    mgr = ModelManager(cache_dir=tmp_path / "cache")
    rows = mgr.ensure_app_models(keys=["detector_balanced", "pose"])

    assert [row["key"] for row in rows] == ["detector_balanced", "pose"]
    assert {Path(row["path"]).name for row in rows} == {
        "yolo11s.onnx",
        "yolo11n-pose.onnx",
    }


def test_remove_app_models_deletes_only_managed_files(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    managed = [
        cache / "yolo11n.onnx",
        cache / "yolo11s.pt",
        cache / "yolo11m.int8.onnx",
        cache / "yolo11n-pose.onnx",
    ]
    for path in managed:
        path.write_bytes(b"model")
    custom = cache / "custom.onnx"
    custom.write_bytes(b"keep")

    mgr = ModelManager(cache_dir=cache)
    rows = mgr.remove_app_models()

    assert {row["name"] for row in rows} == {path.name for path in managed}
    assert all(row["state"] == "removed" for row in rows)
    assert not any(path.exists() for path in managed)
    assert custom.read_bytes() == b"keep"


def test_remove_app_models_can_target_selected_keys(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    fast = cache / "yolo11n.onnx"
    pose = cache / "yolo11n-pose.onnx"
    fast.write_bytes(b"model")
    pose.write_bytes(b"model")

    mgr = ModelManager(cache_dir=cache)
    rows = mgr.remove_app_models(keys=["pose"])

    assert {row["name"] for row in rows} == {"yolo11n-pose.onnx"}
    assert fast.exists()
    assert not pose.exists()


def test_export_does_not_change_cwd(tmp_path, monkeypatch) -> None:
    """The exporter chdir's into the cache dir but must restore cwd."""
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    _install_fake_ultralytics(monkeypatch)
    before = Path.cwd()
    mgr = ModelManager(cache_dir=tmp_path / "cache")
    mgr.ensure_detector()
    assert Path.cwd() == before


# ── prebuilt ONNX download (preferred, torch-free) ────────────────────────────


class _FakeHTTPResponse:
    """Minimal context-manager response with a chunked ``read()``."""

    def __init__(self, payload: bytes, chunk: int = 1 << 16) -> None:
        self._payload = payload
        self._chunk = chunk
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = len(self._payload) - self._pos
        out = self._payload[self._pos : self._pos + n]
        self._pos += len(out)
        return out

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def test_prebuilt_download_is_preferred_over_export(tmp_path, monkeypatch) -> None:
    """ensure_detector downloads the prebuilt ONNX and never touches ultralytics."""
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", "https://example.test/yolo11n.onnx")
    # ultralytics import would fail (None) — proves export is NOT used.
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    payload = b"\x00" * (300 * 1024)  # > _MIN_ONNX_BYTES (256 KiB)

    def fake_urlopen(url, *a, **k):
        if url.endswith("/SHA256SUMS"):
            # No manifest published for this release — legacy path, proceeds.
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        assert url == "https://example.test/yolo11n.onnx"
        return _FakeHTTPResponse(payload)

    cache = tmp_path / "cache"
    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", fake_urlopen):
        mgr = ModelManager(cache_dir=cache)
        result = mgr.ensure_detector()

    onnx = cache / "yolo11n.onnx"
    assert result == str(onnx)
    assert onnx.is_file()
    assert onnx.read_bytes() == payload


def test_prebuilt_truncated_download_falls_back_to_export(tmp_path, monkeypatch) -> None:
    """A too-small download (e.g. an HTML error page) is rejected → export runs."""
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", "https://example.test/yolo11n.onnx")
    captured = _install_fake_ultralytics(monkeypatch)

    def fake_urlopen(url, *a, **k):
        return _FakeHTTPResponse(b"<html>not a model</html>")

    cache = tmp_path / "cache"
    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", fake_urlopen):
        mgr = ModelManager(cache_dir=cache)
        result = mgr.ensure_detector()

    onnx = cache / "yolo11n.onnx"
    assert result == str(onnx)
    # The export fallback (fake ultralytics) actually produced the file.
    assert captured["weights"] == "yolo11n.pt"


def test_prebuilt_network_error_falls_back_to_export(tmp_path, monkeypatch) -> None:
    """A network failure on the prebuilt path falls back to the export, no raise."""
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", "https://example.test/yolo11n.onnx")
    captured = _install_fake_ultralytics(monkeypatch)

    def boom_urlopen(url, *a, **k):
        raise OSError("network down")

    cache = tmp_path / "cache"
    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", boom_urlopen):
        mgr = ModelManager(cache_dir=cache)
        result = mgr.ensure_detector()

    assert result == str(cache / "yolo11n.onnx")
    assert captured["weights"] == "yolo11n.pt"


def test_prebuilt_failure_and_no_ultralytics_returns_none(tmp_path, monkeypatch) -> None:
    """Both acquisition paths unavailable → None, never raises."""
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", "https://example.test/yolo11n.onnx")
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # export unavailable

    def boom_urlopen(url, *a, **k):
        raise OSError("network down")

    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", boom_urlopen):
        mgr = ModelManager(cache_dir=tmp_path / "cache")
        assert mgr.ensure_detector() is None


def test_prebuilt_download_loadable_by_person_detector(tmp_path, monkeypatch) -> None:
    """A 'downloaded' synthetic ONNX loads in PersonDetector and detects.

    Mirrors the real bootstrap: ensure_detector returns a cached path, then
    PersonDetector(model_path=...) opens it via onnxruntime.  We serialise a
    synthetic NMS-free model and serve its bytes through the prebuilt path so
    the end-to-end "model present → boxes" wiring is exercised offline.
    """
    import io

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Build a tiny [1, 1, 6] NMS-free model (one person box) and serialise it.
    data = np.array([[[120.0, 90.0, 240.0, 380.0, 0.9, 0.0]]], dtype=np.float32)
    const = numpy_helper.from_array(data, name="out_const")
    node = helper.make_node("Constant", [], ["output0"], value=const)
    images_in = helper.make_tensor_value_info(
        "images",
        TensorProto.FLOAT,
        [1, 3, 640, 640],
    )
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1, 6])
    graph = helper.make_graph([node], "synthetic", [images_in], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    buf = io.BytesIO()
    onnx.save(model, buf)
    payload = buf.getvalue()  # small synthetic model; size guard lowered below

    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", "https://example.test/yolo11n.onnx")
    # Lower the size guard so the small synthetic model passes.
    monkeypatch.setattr("autoptz.engine.runtime.models._MIN_ONNX_BYTES", 1)

    def fake_urlopen(url, *a, **k):
        return _FakeHTTPResponse(payload)

    cache = tmp_path / "cache"
    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", fake_urlopen):
        mgr = ModelManager(cache_dir=cache)
        path = mgr.ensure_detector()

    assert path is not None

    from autoptz.engine.pipeline.detect import PersonDetector

    det = PersonDetector(model_path=path)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dets = det.detect(frame)
    assert len(dets) == 1
    assert dets[0].class_id == 0


# ── cache dir resolution ──────────────────────────────────────────────────────


# ── SHA-256 manifest verification (prebuilt downloads) ────────────────────────


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_urlopen(*, model_url: str, model_payload: bytes, manifest: str | None):
    """Return a fake ``urlopen`` serving *model_url* and a sibling ``SHA256SUMS``.

    ``manifest=None`` simulates a 404 (no checksum manifest published for this
    release yet); any other string is served verbatim for the ``SHA256SUMS``
    URL derived from *model_url*'s directory.
    """
    manifest_url = model_url.rsplit("/", 1)[0] + "/SHA256SUMS"

    def fake_urlopen(url, *a, **k):
        if url == model_url:
            return _FakeHTTPResponse(model_payload)
        if url == manifest_url:
            if manifest is None:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            return _FakeHTTPResponse(manifest.encode("utf-8"))
        raise AssertionError(f"unexpected URL: {url}")

    return fake_urlopen


def test_prebuilt_download_good_checksum_is_kept(tmp_path, monkeypatch, caplog) -> None:
    """(a) Manifest present + entry matches → file is verified and kept."""
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    model_url = "https://example.test/yolo11n.onnx"
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", model_url)
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    payload = b"\x00" * (300 * 1024)
    manifest = f"{_sha256_hex(payload)}  yolo11n.onnx\n"
    fake_urlopen = _make_urlopen(model_url=model_url, model_payload=payload, manifest=manifest)

    cache = tmp_path / "cache"
    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", fake_urlopen):
        with caplog.at_level("DEBUG"):
            mgr = ModelManager(cache_dir=cache)
            result = mgr.ensure_detector()

    onnx = cache / "yolo11n.onnx"
    assert result == str(onnx)
    assert onnx.is_file()
    assert onnx.read_bytes() == payload


def test_prebuilt_download_bad_checksum_is_deleted_and_errors(
    tmp_path, monkeypatch, caplog
) -> None:
    """(b) Manifest present + mismatch → file deleted, ERROR logged, no fallback machinery."""
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    model_url = "https://example.test/yolo11n.onnx"
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", model_url)
    # No ultralytics → if the code fell back to export, it would return None too,
    # so also assert the tmp/final file never lingers to prove the delete path ran.
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    payload = b"\x00" * (300 * 1024)
    wrong_hash = "0" * 64
    manifest = f"{wrong_hash}  yolo11n.onnx\n"
    fake_urlopen = _make_urlopen(model_url=model_url, model_payload=payload, manifest=manifest)

    cache = tmp_path / "cache"
    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", fake_urlopen):
        with caplog.at_level("ERROR"):
            mgr = ModelManager(cache_dir=cache)
            result = mgr.ensure_detector()

    onnx = cache / "yolo11n.onnx"
    assert result is None
    assert not onnx.exists()
    assert not any(p.suffix == ".part" for p in cache.glob("*.part"))
    assert any(
        "SHA-256" in rec.message or "checksum" in rec.message.lower() for rec in caplog.records
    )


def test_prebuilt_download_missing_manifest_warns_and_keeps_file(
    tmp_path, monkeypatch, caplog
) -> None:
    """(c) Manifest missing (404) → WARNING logged, file kept (legacy release)."""
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    model_url = "https://example.test/yolo11n.onnx"
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", model_url)
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    payload = b"\x00" * (300 * 1024)
    fake_urlopen = _make_urlopen(model_url=model_url, model_payload=payload, manifest=None)

    cache = tmp_path / "cache"
    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", fake_urlopen):
        with caplog.at_level("WARNING"):
            mgr = ModelManager(cache_dir=cache)
            result = mgr.ensure_detector()

    onnx = cache / "yolo11n.onnx"
    assert result == str(onnx)
    assert onnx.is_file()
    assert onnx.read_bytes() == payload
    assert any(
        "legacy release" in rec.message.lower() or "no checksum" in rec.message.lower()
        for rec in caplog.records
        if rec.levelname == "WARNING"
    )


def test_prebuilt_download_entry_absent_warns_and_keeps_file(tmp_path, monkeypatch, caplog) -> None:
    """(d) Manifest present but has no entry for this file → WARNING, file kept."""
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    model_url = "https://example.test/yolo11n.onnx"
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", model_url)
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    payload = b"\x00" * (300 * 1024)
    # Manifest published, but only lists an unrelated file.
    manifest = f"{_sha256_hex(b'other')}  yolo11s.onnx\n"
    fake_urlopen = _make_urlopen(model_url=model_url, model_payload=payload, manifest=manifest)

    cache = tmp_path / "cache"
    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", fake_urlopen):
        with caplog.at_level("WARNING"):
            mgr = ModelManager(cache_dir=cache)
            result = mgr.ensure_detector()

    onnx = cache / "yolo11n.onnx"
    assert result == str(onnx)
    assert onnx.is_file()
    assert onnx.read_bytes() == payload
    assert any(rec.levelname == "WARNING" for rec in caplog.records)


def test_fetch_models_tool_path_verifies_checksum(tmp_path, monkeypatch) -> None:
    """(e) tools/fetch_models.py's ModelManager.ensure_app_models exercises the
    same checksum helper — a bad checksum must not leave a bad file cached even
    when driven through the CLI's entry point (ensure_app_models -> ensure_detector
    -> ensure_pose, all routed through ModelManager._download_prebuilt).
    """
    monkeypatch.delenv("AUTOPTZ_MODEL_PATH", raising=False)
    monkeypatch.delenv("AUTOPTZ_POSE_MODEL_PATH", raising=False)
    monkeypatch.setenv("AUTOPTZ_MODEL_URL", "https://example.test/{stem}.onnx")
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    payload = b"\x00" * (300 * 1024)
    good_hash = _sha256_hex(payload)

    def fake_urlopen(url, *a, **k):
        if url.endswith("/SHA256SUMS"):
            # Only yolo11n (fast tier) gets a correct entry; everything else is
            # either wrong or absent, to prove each is verified independently.
            manifest = f"{good_hash}  yolo11n.onnx\n{'0' * 64}  yolo11s.onnx\n"
            return _FakeHTTPResponse(manifest.encode("utf-8"))
        return _FakeHTTPResponse(payload)

    cache = tmp_path / "cache"
    with patch("autoptz.engine.runtime.models.urllib.request.urlopen", fake_urlopen):
        mgr = ModelManager(cache_dir=cache)
        rows = mgr.ensure_app_models(
            keys=["detector_fast", "detector_balanced"], include_pose=False
        )

    by_key = {r["key"]: r for r in rows}
    assert by_key["detector_fast"]["state"] == "ok"
    assert (cache / "yolo11n.onnx").is_file()
    # yolo11s.onnx had a checksum mismatch and no ultralytics fallback → failed.
    assert by_key["detector_balanced"]["state"] == "failed"
    assert not (cache / "yolo11s.onnx").exists()


def test_default_cache_dir_is_under_appdata_models() -> None:
    mgr = ModelManager()
    # Lives under the platform AutoPTZ dir, in a "models" subfolder.
    assert mgr.cache_dir.name == "models"
    assert mgr.cache_dir.parent.name == "AutoPTZ"


# ── camera_worker wiring ──────────────────────────────────────────────────────


def test_camera_worker_resolve_model_path_uses_manager(monkeypatch) -> None:
    from autoptz.config.models import CameraConfig, SourceConfig
    from autoptz.engine import camera_worker

    sentinel = "/tmp/some/model.onnx"
    fake_mgr = MagicMock()
    fake_mgr.ensure_detector.return_value = sentinel
    monkeypatch.setattr(
        "autoptz.engine.runtime.models.default_manager",
        lambda: fake_mgr,
    )

    cfg = CameraConfig(
        id="cam-abcd1234", name="C", source=SourceConfig(type="usb", address="usb://0")
    )
    assert camera_worker._resolve_model_path(cfg) == sentinel
    fake_mgr.ensure_detector.assert_called_once()


def test_camera_worker_resolve_model_path_never_raises(monkeypatch) -> None:
    from autoptz.config.models import CameraConfig, SourceConfig
    from autoptz.engine import camera_worker

    def boom():
        raise RuntimeError("manager broke")

    monkeypatch.setattr(
        "autoptz.engine.runtime.models.default_manager",
        boom,
    )
    cfg = CameraConfig(
        id="cam-abcd1234", name="C", source=SourceConfig(type="usb", address="usb://0")
    )
    assert camera_worker._resolve_model_path(cfg) is None
