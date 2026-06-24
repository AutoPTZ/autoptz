"""CoreML compute-unit selection (AUTOPTZ_COREML_UNITS).

Lets an Intel + AMD Mac (e.g. iMac Pro Xeon + Vega) verify/force whether CoreML
uses the discrete GPU or silently falls back to the CPU.
"""

from __future__ import annotations

from autoptz.engine.runtime.inference import _coreml_compute_units


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
