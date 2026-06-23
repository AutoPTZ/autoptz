"""Tests for the model bootstrap (autoptz.engine.runtime.models.ModelManager).

All network / ultralytics export is mocked so these run offline and fast.
The contract under test: ``ensure_detector()`` resolves the env override, reuses
a cached ONNX, **prefers a prebuilt torch-free ONNX download**, falls back to
the ultralytics export, and NEVER raises — it returns ``None``
(live-preview-only) when neither acquisition path is reachable.
"""

from __future__ import annotations

import sys
import types
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
