"""Unit tests for autoptz.engine.runtime.inference."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from autoptz.engine.runtime.inference import (
    EP,
    HardwarePrefs,
    _candidate_order,
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
