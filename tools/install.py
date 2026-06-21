"""Install AutoPTZ dependencies with one readable, hardware-aware entry point.

Static requirements files can detect OS with environment markers, but they
cannot inspect the actual GPU or CUDA/TensorRT runtime on a machine. This script
keeps the requirements files simple and makes ONNX Runtime swaps explicit so one
and only one ``onnxruntime*`` wheel remains installed.
"""

from __future__ import annotations

import argparse
import logging
import platform
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("autoptz.install")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORT_PACKAGES = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-openvino",
)

Accelerator = str


@dataclass(frozen=True)
class SystemInfo:
    """The small slice of host state needed to choose dependency profiles."""

    os_name: str
    machine: str
    gpu_names: tuple[str, ...] = ()

    @property
    def is_macos(self) -> bool:
        return self.os_name == "Darwin"

    @property
    def is_windows(self) -> bool:
        return self.os_name == "Windows"

    @property
    def is_linux(self) -> bool:
        return self.os_name == "Linux"

    @property
    def has_nvidia_gpu(self) -> bool:
        return any("nvidia" in name.lower() for name in self.gpu_names)

    def label(self) -> str:
        gpu_label = ", ".join(self.gpu_names) if self.gpu_names else "no GPU name detected"
        return f"{self.os_name} {self.machine} ({gpu_label})"


@dataclass(frozen=True)
class PipStep:
    """One pip command plus a human reason for logs and dry runs."""

    description: str
    args: tuple[str, ...]

    def command(self, python_executable: str = sys.executable) -> tuple[str, ...]:
        return (python_executable, "-m", "pip", *self.args)


@dataclass(frozen=True)
class InstallPlan:
    """Fully resolved install work for a host and selected options."""

    system: SystemInfo
    profiles: tuple[str, ...]
    steps: tuple[PipStep, ...]
    notes: tuple[str, ...]


def _requirement_path(name: str) -> str:
    return str(Path("requirements") / f"{name}.txt")


def _requirement_step(name: str) -> PipStep:
    return PipStep(f"Install requirements/{name}.txt", ("install", "-r", _requirement_path(name)))


def _remove_ort_step() -> PipStep:
    return PipStep(
        "Remove conflicting ONNX Runtime wheels",
        ("uninstall", "-y", *ORT_PACKAGES),
    )


def _run_command_lines(command: Sequence[str], timeout: float = 2.0) -> tuple[str, ...]:
    """Run a best-effort probe command and return non-empty stdout lines."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        log.debug("probe failed: %s", shlex.join(command), exc_info=True)
        return ()
    if result.returncode != 0:
        log.debug(
            "probe returned %s: %s",
            result.returncode,
            result.stderr.strip() or shlex.join(command),
        )
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _dedupe(items: Sequence[str]) -> tuple[str, ...]:
    """Preserve order while removing duplicate probe output."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)


def _detect_gpu_names(os_name: str) -> tuple[str, ...]:
    """Return display/GPU names from lightweight platform probes."""
    names: list[str] = []
    names.extend(_run_command_lines(("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")))

    if os_name == "Windows":
        names.extend(
            _run_command_lines(
                (
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
                ),
                timeout=4.0,
            )
        )
    elif os_name == "Linux":
        for line in _run_command_lines(("lspci",), timeout=2.0):
            lowered = line.lower()
            if "vga" in lowered or "3d controller" in lowered or "display controller" in lowered:
                names.append(line)
    elif os_name == "Darwin":
        for line in _run_command_lines(("system_profiler", "SPDisplaysDataType"), timeout=4.0):
            if "Chipset Model:" in line:
                names.append(line.split(":", 1)[1].strip())

    return _dedupe(names)


def detect_system() -> SystemInfo:
    """Inspect OS, architecture, and best-effort GPU names for install planning."""
    os_name = platform.system()
    return SystemInfo(
        os_name=os_name,
        machine=platform.machine(),
        gpu_names=_detect_gpu_names(os_name),
    )


def _auto_accelerator(system: SystemInfo) -> Accelerator | None:
    """Choose the safest useful accelerator for a normal local install."""
    if system.is_windows:
        return "directml"
    if system.is_linux and system.has_nvidia_gpu:
        return "nvidia"
    return None


def _resolve_accelerator(system: SystemInfo, requested: Accelerator) -> Accelerator | None:
    """Normalize user intent into an accelerator profile, or None for base CPU/CoreML."""
    if requested == "auto":
        return _auto_accelerator(system)
    if requested == "cpu":
        return None
    if requested == "directml":
        if not system.is_windows:
            raise ValueError("DirectML is only available on Windows.")
        return "directml"
    if requested == "nvidia":
        if not (system.is_windows or system.is_linux):
            raise ValueError("The NVIDIA ONNX Runtime wheel is only for Windows/Linux.")
        return "nvidia"
    if requested == "openvino":
        return "openvino"
    raise ValueError(f"Unknown accelerator: {requested}")


def plan_install(
    system: SystemInfo,
    *,
    accelerator: Accelerator = "auto",
    dev: bool = False,
    packaging: bool = False,
    editable: bool = False,
    ui_only: bool = False,
    ci: bool = False,
    upgrade_pip: bool = False,
) -> InstallPlan:
    """Build a deterministic pip plan without running it."""
    if ui_only and packaging:
        raise ValueError("--ui-only and --packaging target different environments.")

    requested = "cpu" if ci and accelerator == "auto" else accelerator
    selected = None if ui_only else _resolve_accelerator(system, requested)

    steps: list[PipStep] = []
    profiles: list[str] = []
    notes: list[str] = []

    if upgrade_pip:
        steps.append(PipStep("Upgrade pip", ("install", "--upgrade", "pip")))

    if ui_only:
        profiles.append("ui")
        steps.append(_requirement_step("ui"))
        notes.append("UI-only mode skips the ML/video stack and all accelerator wheels.")
    else:
        profiles.append("base")
        steps.append(_requirement_step("base"))
        if selected:
            profiles.append(selected)
            steps.append(_remove_ort_step())
            if selected == "directml":
                steps.append(_requirement_step("gpu-directml"))
                notes.append("Windows default: DirectML works on DX12 GPUs and keeps CPU fallback.")
            elif selected == "nvidia":
                steps.append(_requirement_step("gpu-nvidia"))
                notes.append(
                    "NVIDIA installs TensorRT/CUDA EPs; CUDA/cuDNN/TensorRT are system deps."
                )
            elif selected == "openvino":
                steps.append(_requirement_step("openvino"))
                notes.append("OpenVINO is best for Intel CPU/iGPU/Arc systems.")
        elif system.is_macos:
            notes.append(
                "macOS uses the base ONNX Runtime wheel; CoreML appears when the wheel exposes it."
            )

    if dev:
        profiles.append("dev")
        steps.append(_requirement_step("dev"))
    if packaging:
        profiles.append("packaging")
        steps.append(_requirement_step("packaging"))
    if editable:
        profiles.append("editable")
        steps.append(
            PipStep(
                "Install AutoPTZ editable without re-resolving deps",
                ("install", "-e", ".", "--no-deps"),
            )
        )
    if ci:
        notes.append("CI mode keeps dependency selection hardware-independent.")

    return InstallPlan(
        system=system,
        profiles=tuple(profiles),
        steps=tuple(steps),
        notes=tuple(notes),
    )


def print_plan(plan: InstallPlan, python_executable: str = sys.executable) -> None:
    """Print the exact commands so installs are easy to debug and review."""
    print(f"Detected: {plan.system.label()}")
    print(f"Profiles: {', '.join(plan.profiles)}")
    if plan.notes:
        print("Notes:")
        for note in plan.notes:
            print(f"  - {note}")
    print("Commands:")
    for step in plan.steps:
        print(f"  # {step.description}")
        print(f"  {shlex.join(step.command(python_executable))}")


def run_plan(plan: InstallPlan, *, dry_run: bool = False) -> None:
    """Execute the resolved pip plan, stopping on the first failed command."""
    print_plan(plan)
    if dry_run:
        return
    for step in plan.steps:
        log.info(step.description)
        subprocess.run(step.command(), cwd=PROJECT_ROOT, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "directml", "nvidia", "openvino"),
        default="auto",
        help="ONNX Runtime accelerator profile. Default: auto.",
    )
    parser.add_argument("--dev", action="store_true", help="Install test/lint/type-check tools.")
    parser.add_argument(
        "--packaging", action="store_true", help="Install PyInstaller build tooling."
    )
    parser.add_argument(
        "--editable", action="store_true", help="Install AutoPTZ with pip install -e ."
    )
    parser.add_argument(
        "--ui-only", action="store_true", help="Install only lightweight UI dependencies."
    )
    parser.add_argument(
        "--ci", action="store_true", help="Use hardware-independent CI dependency choices."
    )
    parser.add_argument(
        "--upgrade-pip", action="store_true", help="Upgrade pip before installing profiles."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without installing anything."
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Show debug probe logs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    try:
        plan = plan_install(
            detect_system(),
            accelerator=args.accelerator,
            dev=args.dev,
            packaging=args.packaging,
            editable=args.editable,
            ui_only=args.ui_only,
            ci=args.ci,
            upgrade_pip=args.upgrade_pip,
        )
    except ValueError as exc:
        parser.error(str(exc))
    run_plan(plan, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
