"""Unit tests for autoptz.engine.runtime.inference."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from autoptz.engine.runtime.inference import (
    EP,
    HardwarePrefs,
    _candidate_order,
    _ep_can_run_fp16,
    effective_precision,
    get_best_ep,
    make_session,
)


def test_ep_enum_values() -> None:
    assert EP.COREML.value == "CoreMLExecutionProvider"
    assert EP.CPU.value == "CPUExecutionProvider"


def test_candidate_order_no_duplicates() -> None:
    order = _candidate_order()
    assert len(order) == len(set(order)), "Duplicate EPs in candidate order"
    assert EP.CPU in order, "CPU must always be in candidate list"


def test_candidate_order_cpu_last_by_default() -> None:
    """CPU should be the last resort."""
    order = _candidate_order()
    assert order[-1] == EP.CPU


def test_get_best_ep_returns_ep() -> None:
    ep = get_best_ep()
    assert isinstance(ep, EP)


def test_get_best_ep_cpu_always_available() -> None:
    """Even with no GPU, CPU must be selectable."""
    with patch(
        "autoptz.engine.runtime.inference._available_providers",
        return_value=frozenset(["CPUExecutionProvider"]),
    ):
        ep = get_best_ep()
    assert ep == EP.CPU


def test_get_best_ep_prefers_coreml_on_macos() -> None:
    with (
        patch.object(sys, "platform", "darwin"),
        patch(
            "autoptz.engine.runtime.inference._available_providers",
            return_value=frozenset(["CoreMLExecutionProvider", "CPUExecutionProvider"]),
        ),
    ):
        ep = get_best_ep()
    assert ep == EP.COREML


def test_get_best_ep_prefers_cuda_over_cpu_on_windows() -> None:
    with (
        patch.object(sys, "platform", "win32"),
        patch(
            "autoptz.engine.runtime.inference._available_providers",
            return_value=frozenset(["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ),
    ):
        ep = get_best_ep()
    assert ep == EP.CUDA


def test_get_best_ep_force_ep_honoured() -> None:
    prefs = HardwarePrefs(force_ep=EP.CPU)
    with patch(
        "autoptz.engine.runtime.inference._available_providers",
        return_value=frozenset(["CoreMLExecutionProvider", "CPUExecutionProvider"]),
    ):
        ep = get_best_ep(prefs)
    assert ep == EP.CPU


def test_get_best_ep_force_ep_unavailable_falls_back() -> None:
    """If forced EP is unavailable, auto-select kicks in."""
    prefs = HardwarePrefs(force_ep=EP.TENSORRT)
    with patch(
        "autoptz.engine.runtime.inference._available_providers",
        return_value=frozenset(["CPUExecutionProvider"]),
    ):
        ep = get_best_ep(prefs)
    assert ep == EP.CPU  # best available after forced EP fails


def test_cache_dir_creates_directory() -> None:
    import os

    from autoptz.engine.runtime.inference import _cache_dir

    d = _cache_dir("pytest_coreml_cache_probe")
    assert os.path.isdir(d)


def test_coreml_provider_options_include_model_cache() -> None:
    import os

    from autoptz.engine.runtime.inference import EP, _provider_options

    opts = _provider_options(EP.COREML, None)
    assert opts["ModelFormat"] == "MLProgram"
    assert opts["MLComputeUnits"] == "ALL"
    assert "ModelCacheDirectory" in opts
    assert os.path.isdir(str(opts["ModelCacheDirectory"]))


def test_make_session_cpu(tmp_path: Path) -> None:
    """Create a minimal ONNX model and verify make_session returns a session."""
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "test_identity", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    model_path = tmp_path / "identity.onnx"
    onnx.save(model, str(model_path))

    prefs = HardwarePrefs(force_ep=EP.CPU)
    session = make_session(str(model_path), prefs=prefs)
    assert session is not None
    assert "CPUExecutionProvider" in session.get_providers()


# ---------------------------------------------------------------------------
# effective_precision / _ep_can_run_fp16 — honest per-EP labelling
# ---------------------------------------------------------------------------


def test_cpu_ep_always_fp32() -> None:
    """CPU EP is always FP32 regardless of prefs."""
    assert effective_precision(EP.CPU) == "fp32"
    assert effective_precision(EP.CPU, HardwarePrefs(precision="auto")) == "fp32"
    assert effective_precision(EP.CPU, HardwarePrefs(precision="fp16")) == "fp32"


def test_tensorrt_fp16_when_auto() -> None:
    assert effective_precision(EP.TENSORRT) == "fp16"
    assert effective_precision(EP.TENSORRT, HardwarePrefs(precision="auto")) == "fp16"


def test_tensorrt_fp32_when_forced() -> None:
    assert effective_precision(EP.TENSORRT, HardwarePrefs(precision="fp32")) == "fp32"


def test_cuda_fp16_when_auto() -> None:
    assert effective_precision(EP.CUDA) == "fp16"


def test_cuda_fp32_when_forced() -> None:
    assert effective_precision(EP.CUDA, HardwarePrefs(precision="fp32")) == "fp32"


def test_directml_always_fp32() -> None:
    """DirectML passes the FP32 ONNX model through — never auto-converts to FP16."""
    assert effective_precision(EP.DIRECTML) == "fp32"
    assert effective_precision(EP.DIRECTML, HardwarePrefs(precision="auto")) == "fp32"
    assert effective_precision(EP.DIRECTML, HardwarePrefs(precision="fp16")) == "fp32"


def test_openvino_always_fp32_conservative() -> None:
    """OpenVINO is reported FP32 conservatively (device at runtime unknown)."""
    assert effective_precision(EP.OPENVINO) == "fp32"
    assert effective_precision(EP.OPENVINO, HardwarePrefs(precision="auto")) == "fp32"


def test_coreml_fp16_on_apple_silicon() -> None:
    """CoreML genuinely runs FP16 via MLProgram on arm64."""
    with patch("platform.machine", return_value="arm64"):
        assert _ep_can_run_fp16(EP.COREML) is True
        assert effective_precision(EP.COREML) == "fp16"
        assert effective_precision(EP.COREML, HardwarePrefs(precision="auto")) == "fp16"


def test_coreml_fp16_on_aarch64() -> None:
    """aarch64 (Linux ARM) is also genuinely FP16 capable."""
    with patch("platform.machine", return_value="aarch64"):
        assert _ep_can_run_fp16(EP.COREML) is True
        assert effective_precision(EP.COREML) == "fp16"


def test_coreml_fp32_on_intel_mac() -> None:
    """CoreML on x86_64 (Intel Mac) has no ANE; effective precision is FP32."""
    with patch("platform.machine", return_value="x86_64"):
        assert _ep_can_run_fp16(EP.COREML) is False
        assert effective_precision(EP.COREML) == "fp32"
        assert effective_precision(EP.COREML, HardwarePrefs(precision="auto")) == "fp32"
        # Even if user requests fp16, the EP can't honour it — we still report fp32
        assert effective_precision(EP.COREML, HardwarePrefs(precision="fp16")) == "fp32"


def test_coreml_fp32_on_intel_mac_with_fp32_prefs() -> None:
    """Explicit fp32 prefs + Intel Mac CoreML → fp32."""
    with patch("platform.machine", return_value="x86_64"):
        assert effective_precision(EP.COREML, HardwarePrefs(precision="fp32")) == "fp32"


def test_int8_prefs_overrides_all_eps() -> None:
    """int8 precision is returned verbatim for every EP."""
    prefs = HardwarePrefs(precision="int8")
    for ep in EP:
        assert effective_precision(ep, prefs) == "int8", f"Expected int8 for {ep}"


def test_fp32_prefs_overrides_accelerated_eps() -> None:
    """Explicit fp32 user prefs beats any accelerated EP capability."""
    prefs = HardwarePrefs(precision="fp32")
    with patch("platform.machine", return_value="arm64"):
        for ep in EP:
            assert effective_precision(ep, prefs) == "fp32", f"Expected fp32 for {ep}"


def test_effective_precision_unknown_string_ep() -> None:
    """Unknown EP string falls back to fp32."""
    assert effective_precision("SomeUnknownEP") == "fp32"


def test_effective_precision_known_string_ep() -> None:
    """String EP values are resolved correctly."""
    with patch("platform.machine", return_value="arm64"):
        assert effective_precision("CoreMLExecutionProvider") == "fp16"
    with patch("platform.machine", return_value="x86_64"):
        assert effective_precision("CoreMLExecutionProvider") == "fp32"
    assert effective_precision("CPUExecutionProvider") == "fp32"
    assert effective_precision("DmlExecutionProvider") == "fp32"
