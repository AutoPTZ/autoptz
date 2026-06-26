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
    cpu_brand: str = ""

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

    @property
    def has_intel_gpu(self) -> bool:
        """True when any detected GPU is clearly Intel.

        Matches canonical Intel GPU brand strings (Arc, Iris, UHD) *and* the
        PCI vendor string "intel" so that lspci lines like "Intel Corporation
        DG2 [Arc A770]" or "8086:" prefix entries are also caught.  A bare
        substring "intel" in the adapter name is sufficient; vendor ID ``8086``
        or the word "arc"/"iris"/"uhd" are additional signals.
        """
        _INTEL_TOKENS = ("intel", "arc", "iris", "uhd", "8086")
        return any(any(tok in name.lower() for tok in _INTEL_TOKENS) for name in self.gpu_names)

    @property
    def has_intel_cpu(self) -> bool:
        """True when the detected CPU brand string is Intel."""
        return "intel" in self.cpu_brand.lower()

    @property
    def has_discrete_non_intel_gpu(self) -> bool:
        """True when any detected GPU is a discrete NVIDIA or AMD card."""
        return self.has_nvidia_gpu or any(
            "amd" in name.lower() or "radeon" in name.lower()
            for name in self.gpu_names
            if "intel" not in name.lower()
        )

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
    # Pip accepts forward slashes on every supported OS, and keeping this stable
    # makes the install plan easy to test and compare on Windows.
    return f"requirements/{name}.txt"


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


def _detect_cpu_brand(os_name: str) -> str:
    """Best-effort CPU brand string (for Intel-CPU accelerator selection)."""
    if os_name == "Linux":
        for line in _run_command_lines(("grep", "-m1", "model name", "/proc/cpuinfo")):
            return line.split(":", 1)[1].strip() if ":" in line else line
        return ""
    if os_name == "Darwin":
        out = _run_command_lines(("sysctl", "-n", "machdep.cpu.brand_string"))
        return out[0] if out else ""
    if os_name == "Windows":
        # platform.processor() returns the brand on Windows (e.g. "Intel64 Family ...").
        return platform.processor() or ""
    return ""


def detect_system() -> SystemInfo:
    """Inspect OS, architecture, and best-effort GPU/CPU names for install planning."""
    os_name = platform.system()
    return SystemInfo(
        os_name=os_name,
        machine=platform.machine(),
        gpu_names=_detect_gpu_names(os_name),
        cpu_brand=_detect_cpu_brand(os_name),
    )


def _auto_accelerator(system: SystemInfo) -> Accelerator | None:
    """Choose the safest useful accelerator for a normal local install.

    Selection priority (first match wins):
      1. macOS          → None  (base wheel; CoreML picked up automatically)
      2. NVIDIA present → nvidia  (Windows or Linux)
      3. Intel GPU      → openvino  (Arc / Iris / UHD on Windows or Linux)
      4. Intel CPU only → openvino  (beats the stock CPU EP on all Intel boxes)
      5. Windows AMD    → directml  (DX12 fallback; covers AMD/other dGPUs)
      6. Everything else → None (base CPU EP)

    OpenVINO is never auto-selected on macOS because there is no macOS wheel.
    The ``--ci`` path forces ``accelerator="cpu"`` before this function is
    called, so CI always lands on the None/base branch.
    """
    if system.is_macos:
        return None
    if system.has_nvidia_gpu:
        return "nvidia"
    # Intel GPU (Arc / Iris / UHD) on Windows or Linux → OpenVINO.
    if system.has_intel_gpu:
        return "openvino"
    # Linux Intel CPU with no discrete GPU → OpenVINO beats the stock CPU EP.
    # (Windows CPU-only boxes already get DirectML below, which is the existing behaviour.)
    # Reads the CPU brand captured in SystemInfo (probed once at detect_system) so
    # plan_install stays deterministic from its inputs instead of re-probing the host.
    if system.is_linux and not system.has_discrete_non_intel_gpu and system.has_intel_cpu:
        return "openvino"
    if system.is_windows:
        return "directml"
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
        if system.is_macos:
            raise ValueError(
                "OpenVINO has no macOS wheel; use the default CoreML/CPU build"
                " (omit --accelerator or pass --accelerator cpu)."
            )
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
    with_tracking: bool = False,
    with_export: bool = False,
    full: bool = False,
) -> InstallPlan:
    """Build a deterministic pip plan without running it.

    The default install is torch-free: the heavy boxmot (tracking) and
    ultralytics (export) extras are opt-in via ``--with-tracking`` /
    ``--with-export`` / ``--full``. ``--dev`` pulls them in through dev.txt
    (the test suite needs them), so they are not added again here when dev is
    selected.
    """
    if ui_only and packaging:
        raise ValueError("--ui-only and --packaging target different environments.")
    # dev.txt already references tracking.txt + export.txt, so contributors and
    # CI (--dev) always get them. The explicit extras only matter for non-dev
    # installs that opt in.
    want_tracking = with_tracking or full
    want_export = with_export or full

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
    # The torch-heavy extras. dev.txt already references both, so only add them
    # explicitly when opted in without --dev (avoids a redundant pip step). They
    # are skipped in --ui-only mode, which deliberately omits the ML stack.
    if not ui_only and not dev:
        if want_tracking:
            profiles.append("tracking")
            steps.append(_requirement_step("tracking"))
            notes.append(
                "tracking: boxmot adds BoT-SORT/DeepOCSORT/ByteTrack + OSNet ReID (pulls in torch)."
            )
        if want_export:
            profiles.append("export")
            steps.append(_requirement_step("export"))
            notes.append(
                "export: ultralytics adds the YOLO11 .pt → ONNX export fallback (pulls in torch)."
            )
    if not ui_only and not dev and not (want_tracking or want_export):
        notes.append(
            "Lean torch-free default: detection + IoU tracking only. Add --full"
            " (or --with-tracking / --with-export) for boxmot/ultralytics."
        )
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
        "--with-tracking",
        action="store_true",
        help="Add the boxmot tracking extra (BoT-SORT/DeepOCSORT/ReID; pulls in torch).",
    )
    parser.add_argument(
        "--with-export",
        action="store_true",
        help="Add the ultralytics export extra (.pt → ONNX fallback; pulls in torch).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Add both torch-heavy extras (tracking + export). Default install is torch-free.",
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
            with_tracking=args.with_tracking,
            with_export=args.with_export,
            full=args.full,
        )
    except ValueError as exc:
        parser.error(str(exc))
    run_plan(plan, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
