"""Unit tests for the synthetic test source + ORT idle-threading tuning.

These cover the test/scaling infrastructure added for multi-camera CPU work:
the procedural :class:`SyntheticAdapter`, the ``AUTOPTZ_DB_PATH`` override, and
the low-idle ORT session tuning that stops worker threads busy-spinning.
"""

from __future__ import annotations

import numpy as np

from autoptz.engine.pipeline.ingest import SyntheticAdapter


class TestSyntheticAdapter:
    def test_procedural_open_and_read(self) -> None:
        a = SyntheticAdapter("cam-proc", address="anim", width=320, height=240, target_fps=30.0)
        assert a._open() is True
        f = a._read_frame()
        assert f is not None
        assert f.shape == (240, 320, 3)
        assert f.dtype == np.uint8
        a._close()

    def test_frames_change_over_time(self) -> None:
        # The scene must animate (pan) so motion-dependent stages get real work.
        a = SyntheticAdapter("cam-move", address="anim", width=320, height=240)
        a._open()
        f1 = a._read_frame()
        for _ in range(10):
            f2 = a._read_frame()
        assert f1 is not None and f2 is not None
        assert not np.array_equal(f1, f2)
        a._close()

    def test_unknown_path_falls_back_without_raising(self) -> None:
        a = SyntheticAdapter("cam-bad", address="/no/such/file.mp4", width=160, height=120)
        assert a._open() is True  # graceful fallback to a procedural/sample scene
        f = a._read_frame()
        assert f is not None and f.shape == (120, 160, 3)
        a._close()

    def test_factory_builds_synthetic_adapter(self) -> None:
        from autoptz.config.models import CameraConfig, SourceConfig
        from autoptz.engine.worker.frame_source import build_frame_source

        cfg = CameraConfig(
            name="Synthetic 1",
            source=SourceConfig(type="synthetic", address="anim", fps=30.0),
        )
        fs = build_frame_source("cam-x", cfg)
        assert fs.open() is True
        frame = fs.read()
        assert frame is not None and frame.ndim == 3
        fs.close()


class TestDbPathOverride:
    def test_env_override(self, monkeypatch, tmp_path) -> None:
        from autoptz.config.store import default_db_path

        target = tmp_path / "alt" / "profile.db"
        monkeypatch.setenv("AUTOPTZ_DB_PATH", str(target))
        assert default_db_path() == target

    def test_default_when_unset(self, monkeypatch) -> None:
        from autoptz.config.store import default_config_dir, default_db_path

        monkeypatch.delenv("AUTOPTZ_DB_PATH", raising=False)
        assert default_db_path() == default_config_dir() / "autoptz.db"


class TestLowIdleThreading:
    def test_session_options_disable_spinning(self) -> None:
        import onnxruntime as ort

        from autoptz.engine.runtime.inference import _apply_low_idle_threading

        so = ort.SessionOptions()
        _apply_low_idle_threading(so)
        assert so.inter_op_num_threads == 1
        assert so.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
        assert so.get_session_config_entry("session.intra_op.allow_spinning") == "0"
        assert so.get_session_config_entry("session.inter_op.allow_spinning") == "0"

    def test_build_session_options_applies_idle_tuning(self) -> None:
        import onnxruntime as ort

        from autoptz.engine.runtime.inference import _build_session_options

        so = _build_session_options(None, None)
        assert so.get_session_config_entry("session.intra_op.allow_spinning") == "0"
        assert so.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class TestInsightfaceCap:
    def test_injects_capped_options_and_restores(self) -> None:
        import onnxruntime as ort

        from autoptz.engine.pipeline.identify import _capped_insightface_sessions

        orig = ort.InferenceSession.__init__
        captured: dict[str, object] = {}

        # Replace InferenceSession.__init__ so we can observe what kwargs reach it
        # (mirrors insightface's get_model passing only providers, no sess_options).
        def fake_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            captured.update(kwargs)

        ort.InferenceSession.__init__ = fake_init  # type: ignore[method-assign]
        try:
            with _capped_insightface_sessions(3):
                ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])  # type: ignore[call-arg]
            so = captured.get("sess_options")
            assert so is not None
            assert so.intra_op_num_threads == 3
            assert so.get_session_config_entry("session.intra_op.allow_spinning") == "0"
            # Restored to the pre-context implementation (our fake), not left patched.
            assert ort.InferenceSession.__init__ is fake_init
        finally:
            ort.InferenceSession.__init__ = orig  # type: ignore[method-assign]
