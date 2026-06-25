from __future__ import annotations

from unittest.mock import patch

import pytest

from tools.install import SystemInfo, _auto_accelerator, _resolve_accelerator, plan_install


def _step_args(system: SystemInfo, **kwargs: object) -> list[tuple[str, ...]]:
    return [step.args for step in plan_install(system, **kwargs).steps]


# ---------------------------------------------------------------------------
# CI safety
# ---------------------------------------------------------------------------


def test_ci_install_is_hardware_independent() -> None:
    system = SystemInfo("Linux", "x86_64", ("NVIDIA RTX 4090",))

    plan = plan_install(system, ci=True, dev=True, editable=True)

    assert plan.profiles == ("base", "dev", "editable")
    assert (
        "uninstall",
        "-y",
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "onnxruntime-openvino",
    ) not in [step.args for step in plan.steps]
    assert any("hardware-independent" in note for note in plan.notes)


# ---------------------------------------------------------------------------
# Windows auto-select
# ---------------------------------------------------------------------------


def test_windows_auto_uses_directml_swap() -> None:
    system = SystemInfo("Windows", "AMD64", ("AMD Radeon",))

    args = _step_args(system)

    assert ("install", "-r", "requirements/base.txt") in args
    assert (
        "uninstall",
        "-y",
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "onnxruntime-openvino",
    ) in args
    assert ("install", "-r", "requirements/gpu-directml.txt") in args


def test_windows_intel_arc_auto_uses_openvino() -> None:
    """Arc GPU on Windows should get OpenVINO, not DirectML."""
    system = SystemInfo("Windows", "AMD64", ("Intel Arc A770",))

    args = _step_args(system)

    assert ("install", "-r", "requirements/openvino.txt") in args
    assert ("install", "-r", "requirements/gpu-directml.txt") not in args


def test_windows_intel_iris_auto_uses_openvino() -> None:
    system = SystemInfo("Windows", "AMD64", ("Intel Iris Xe Graphics",))

    assert _auto_accelerator(system) == "openvino"


def test_windows_intel_uhd_auto_uses_openvino() -> None:
    system = SystemInfo("Windows", "AMD64", ("Intel UHD Graphics 770",))

    assert _auto_accelerator(system) == "openvino"


def test_windows_nvidia_beats_intel_igpu() -> None:
    """NVIDIA dGPU + Intel iGPU on Windows → nvidia (not openvino, not directml)."""
    system = SystemInfo("Windows", "AMD64", ("Intel UHD Graphics 770", "NVIDIA RTX 4090"))

    assert _auto_accelerator(system) == "nvidia"


# ---------------------------------------------------------------------------
# Linux auto-select
# ---------------------------------------------------------------------------


def test_linux_nvidia_auto_uses_nvidia_swap() -> None:
    system = SystemInfo("Linux", "x86_64", ("NVIDIA RTX 4090",))

    args = _step_args(system)

    assert args.index(
        (
            "uninstall",
            "-y",
            "onnxruntime",
            "onnxruntime-gpu",
            "onnxruntime-directml",
            "onnxruntime-openvino",
        )
    ) > args.index(("install", "-r", "requirements/base.txt"))
    assert ("install", "-r", "requirements/gpu-nvidia.txt") in args


def test_linux_intel_uhd_auto_uses_openvino_swap() -> None:
    # Intel CPU/iGPU on Linux with no NVIDIA → OpenVINO (faster than the stock CPU EP).
    system = SystemInfo("Linux", "x86_64", ("Intel Corporation UHD Graphics 770",))

    args = _step_args(system)

    assert ("install", "-r", "requirements/openvino.txt") in args


def test_linux_intel_arc_auto_uses_openvino() -> None:
    """Arc GPU reported by lspci matches on 'arc' token."""
    system = SystemInfo("Linux", "x86_64", ("Intel Corporation DG2 [Arc A770]",))

    assert _auto_accelerator(system) == "openvino"


def test_linux_pci_8086_matches_intel_gpu() -> None:
    """lspci output containing '8086' vendor prefix is treated as Intel GPU."""
    system = SystemInfo("Linux", "x86_64", ("00:02.0 VGA compatible controller [0300]: 8086:a780",))

    assert system.has_intel_gpu is True


def test_linux_intel_cpu_only_auto_uses_openvino() -> None:
    """Intel CPU with no GPU listed → OpenVINO via /proc/cpuinfo probe."""
    system = SystemInfo("Linux", "x86_64", ())  # no GPU name detected

    with patch(
        "tools.install._run_command_lines",
        return_value=("model name : Intel(R) Core(TM) i7-1165G7 @ 2.80GHz",),
    ):
        result = _auto_accelerator(system)

    assert result == "openvino"


def test_linux_amd_cpu_no_gpu_falls_through_to_none() -> None:
    """AMD/unknown CPU with no GPU → None (base CPU EP)."""
    system = SystemInfo("Linux", "x86_64", ())

    with patch(
        "tools.install._run_command_lines",
        return_value=("model name : AMD Ryzen 9 5950X",),
    ):
        result = _auto_accelerator(system)

    assert result is None


def test_linux_nvidia_beats_intel_when_both_present() -> None:
    # A box with both an NVIDIA dGPU and an Intel iGPU should pick NVIDIA, not OpenVINO.
    system = SystemInfo("Linux", "x86_64", ("Intel UHD Graphics", "NVIDIA RTX 4090"))

    args = _step_args(system)

    assert ("install", "-r", "requirements/gpu-nvidia.txt") in args
    assert ("install", "-r", "requirements/openvino.txt") not in args


# ---------------------------------------------------------------------------
# macOS auto-select and openvino gate
# ---------------------------------------------------------------------------


def test_macos_auto_stays_on_base_coreml_wheel() -> None:
    system = SystemInfo("Darwin", "arm64", ("Apple M3",))

    plan = plan_install(system)

    assert plan.profiles == ("base",)
    assert [step.args for step in plan.steps] == [("install", "-r", "requirements/base.txt")]
    assert any("CoreML" in note for note in plan.notes)


def test_macos_openvino_explicit_raises_clear_error() -> None:
    """--accelerator openvino on macOS must fail with an actionable message."""
    system = SystemInfo("Darwin", "arm64", ("Apple M3",))

    with pytest.raises(ValueError, match="no macOS wheel"):
        _resolve_accelerator(system, "openvino")


def test_macos_openvino_via_plan_install_raises() -> None:
    """plan_install propagates the ValueError so main() can surface it."""
    system = SystemInfo("Darwin", "arm64", ("Apple M3",))

    with pytest.raises(ValueError, match="no macOS wheel"):
        plan_install(system, accelerator="openvino")


def test_macos_intel_igpu_auto_stays_base() -> None:
    """Even if a macOS machine has an Intel GPU label, no openvino wheel exists."""
    system = SystemInfo("Darwin", "x86_64", ("Intel Iris Plus Graphics",))

    assert _auto_accelerator(system) is None


# ---------------------------------------------------------------------------
# has_intel_gpu detection coverage
# ---------------------------------------------------------------------------


def test_has_intel_gpu_matches_arc() -> None:
    assert SystemInfo("Linux", "x86_64", ("Arc A770",)).has_intel_gpu is True


def test_has_intel_gpu_matches_iris() -> None:
    assert SystemInfo("Linux", "x86_64", ("Iris Xe Graphics",)).has_intel_gpu is True


def test_has_intel_gpu_matches_uhd() -> None:
    assert SystemInfo("Linux", "x86_64", ("UHD Graphics 770",)).has_intel_gpu is True


def test_has_intel_gpu_matches_intel_substring() -> None:
    assert SystemInfo("Linux", "x86_64", ("Intel Corporation HD Graphics",)).has_intel_gpu is True


def test_has_intel_gpu_rejects_amd() -> None:
    assert SystemInfo("Linux", "x86_64", ("AMD Radeon RX 7900 XTX",)).has_intel_gpu is False


def test_has_intel_gpu_rejects_nvidia() -> None:
    assert SystemInfo("Linux", "x86_64", ("NVIDIA GeForce RTX 4090",)).has_intel_gpu is False


def test_has_intel_gpu_rejects_empty() -> None:
    assert SystemInfo("Linux", "x86_64", ()).has_intel_gpu is False


# ---------------------------------------------------------------------------
# Other platform guards
# ---------------------------------------------------------------------------


def test_ui_only_skips_accelerator_profiles() -> None:
    system = SystemInfo("Windows", "AMD64", ("NVIDIA RTX 4090",))

    plan = plan_install(system, ui_only=True, accelerator="nvidia")

    assert plan.profiles == ("ui",)
    assert [step.args for step in plan.steps] == [("install", "-r", "requirements/ui.txt")]


def test_directml_is_windows_only() -> None:
    system = SystemInfo("Linux", "x86_64")

    with pytest.raises(ValueError, match="DirectML"):
        plan_install(system, accelerator="directml")


def test_nvidia_is_linux_or_windows_only() -> None:
    system = SystemInfo("Darwin", "arm64")

    with pytest.raises(ValueError, match="NVIDIA"):
        plan_install(system, accelerator="nvidia")


def test_openvino_explicit_on_linux_works() -> None:
    system = SystemInfo("Linux", "x86_64", ())

    args = _step_args(system, accelerator="openvino")

    assert ("install", "-r", "requirements/openvino.txt") in args


def test_openvino_explicit_on_windows_works() -> None:
    system = SystemInfo("Windows", "AMD64", ())

    args = _step_args(system, accelerator="openvino")

    assert ("install", "-r", "requirements/openvino.txt") in args
