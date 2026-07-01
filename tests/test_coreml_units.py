"""CoreML compute-unit selection (AUTOPTZ_COREML_UNITS).

Lets an Intel + AMD Mac (e.g. iMac Pro Xeon + Vega) verify/force whether CoreML
uses the discrete GPU or silently falls back to the CPU.
"""

from __future__ import annotations

from autoptz.engine.runtime.inference import EP, _coreml_compute_units, _provider_options


class TestCoreMLCacheUnderModelServerProcess:
    """The CoreML on-disk MLProgram cache (ModelCacheDirectory) makes the CoreML EP
    fail in a SPAWNED child ("Failed to create model URL from path"), which forced a
    fallback/recompile. It must be omitted in model-server camera child processes
    (the ANE/GPU path is kept) and present otherwise."""

    def test_cache_dir_present_for_shared_in_process(self, monkeypatch):
        monkeypatch.delenv("AUTOPTZ_PROCESS_PER_CAMERA", raising=False)
        monkeypatch.delenv("AUTOPTZ_MODEL_SERVER", raising=False)
        opts = _provider_options(EP.COREML, None)
        assert "ModelCacheDirectory" in opts
        assert opts["ModelFormat"] == "MLProgram"

    def test_cache_dir_omitted_under_model_server_process(self, monkeypatch):
        monkeypatch.setenv("AUTOPTZ_MODEL_SERVER", "1")
        opts = _provider_options(EP.COREML, None)
        assert "ModelCacheDirectory" not in opts
        # Still routes to the Neural Engine / GPU — only the on-disk cache is dropped.
        assert opts["ModelFormat"] == "MLProgram"


class TestCoreMLComputeUnits:
    def test_default_is_all(self, monkeypatch):
        monkeypatch.delenv("AUTOPTZ_COREML_UNITS", raising=False)
        assert _coreml_compute_units() == "ALL"

    def test_explicit_cpu_only(self, monkeypatch):
        monkeypatch.setenv("AUTOPTZ_COREML_UNITS", "CPUOnly")
        assert _coreml_compute_units() == "CPUOnly"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("AUTOPTZ_COREML_UNITS", "cpuandgpu")
        assert _coreml_compute_units() == "CPUAndGPU"

    def test_invalid_falls_back_to_all(self, monkeypatch):
        monkeypatch.setenv("AUTOPTZ_COREML_UNITS", "turbo")
        assert _coreml_compute_units() == "ALL"
