from __future__ import annotations

import pytest

from tools.install import SystemInfo, plan_install


def _step_args(system: SystemInfo, **kwargs: object) -> list[tuple[str, ...]]:
    return [step.args for step in plan_install(system, **kwargs).steps]


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


def test_linux_intel_auto_uses_openvino_swap() -> None:
    # Intel CPU/iGPU on Linux with no NVIDIA → OpenVINO (faster than the stock CPU EP).
    system = SystemInfo("Linux", "x86_64", ("Intel Corporation UHD Graphics 770",))

    args = _step_args(system)

    assert ("install", "-r", "requirements/openvino.txt") in args


def test_linux_nvidia_beats_intel_when_both_present() -> None:
    # A box with both an NVIDIA dGPU and an Intel iGPU should pick NVIDIA, not OpenVINO.
    system = SystemInfo("Linux", "x86_64", ("Intel UHD Graphics", "NVIDIA RTX 4090"))

    args = _step_args(system)

    assert ("install", "-r", "requirements/gpu-nvidia.txt") in args
    assert ("install", "-r", "requirements/openvino.txt") not in args


def test_macos_auto_stays_on_base_coreml_wheel() -> None:
    system = SystemInfo("Darwin", "arm64", ("Apple M3",))

    plan = plan_install(system)

    assert plan.profiles == ("base",)
    assert [step.args for step in plan.steps] == [("install", "-r", "requirements/base.txt")]
    assert any("CoreML" in note for note in plan.notes)


def test_ui_only_skips_accelerator_profiles() -> None:
    system = SystemInfo("Windows", "AMD64", ("NVIDIA RTX 4090",))

    plan = plan_install(system, ui_only=True, accelerator="nvidia")

    assert plan.profiles == ("ui",)
    assert [step.args for step in plan.steps] == [("install", "-r", "requirements/ui.txt")]


def test_directml_is_windows_only() -> None:
    system = SystemInfo("Linux", "x86_64")

    with pytest.raises(ValueError, match="DirectML"):
        plan_install(system, accelerator="directml")
