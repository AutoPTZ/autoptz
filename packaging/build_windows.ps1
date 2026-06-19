# build_windows.ps1 — build AutoPTZ for Windows x64 with PyInstaller.
#
# Produces dist\AutoPTZ\AutoPTZ.exe (onedir).  Set -OneFile for a single exe.
# DirectML is the DEFAULT inference EP on Windows (works on any DX12 GPU with no
# CUDA install); see the CUDA/TensorRT note at the bottom for the NVIDIA variant.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -OneFile
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -SkipInstall

param(
    [switch]$OneFile,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

# Resolve repo root (this script lives in packaging\).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

$Venv = if ($env:VENV) { $env:VENV } else { ".venv" }
$Py = Join-Path $Venv "Scripts\python.exe"

Write-Host "==> AutoPTZ Windows build"
Write-Host "    repo: $Root"
Write-Host "    venv: $Venv"

# ── 1. venv ──────────────────────────────────────────────────────────────────
if (-not (Test-Path $Py)) {
    Write-Host "==> Creating venv at $Venv"
    py -3.12 -m venv $Venv
}

# ── 2. dependencies ──────────────────────────────────────────────────────────
# DirectML EP for onnxruntime: install onnxruntime-directml in place of the CPU
# onnxruntime from base.txt (DirectML accelerates on any DX12 GPU).
if (-not $SkipInstall) {
    Write-Host "==> Installing dependencies (DirectML default)"
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -r requirements\base.txt -r requirements\packaging.txt
    # Swap the CPU onnxruntime for the DirectML build:
    & $Py -m pip uninstall -y onnxruntime
    & $Py -m pip install onnxruntime-directml
    # Friendly Windows camera names (parity with macOS AVFoundation enum):
    & $Py -m pip install pygrabber
    & $Py -m pip install -e .
} else {
    Write-Host "==> -SkipInstall — using existing venv as-is"
}

# ── 3. (optional) pre-fetch the detector model ───────────────────────────────
# Uncomment to bake the YOLO11 ONNX into autoptz\models before building:
#   & $Py -m tools.fetch_models --cache-dir autoptz\models

# ── 4. NDI runtime (optional) ────────────────────────────────────────────────
# Processing.NDI.Lib.x64.dll is a system install, not a pip wheel.  To bundle
# it, drop the DLL in packaging\ndi\ (or set $env:NDI_RUNTIME) before building;
# the spec picks it up.  Otherwise NDI ingest degrades gracefully.

# ── 5. build ─────────────────────────────────────────────────────────────────
Write-Host "==> Cleaning previous build"
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist)  { Remove-Item -Recurse -Force dist }

if ($OneFile) { $env:ONEFILE = "1" } else { $env:ONEFILE = "0" }

Write-Host "==> Running PyInstaller (ONEFILE=$($env:ONEFILE))"
& $Py -m PyInstaller packaging\autoptz.spec --noconfirm

if ($OneFile) {
    $Out = "dist\AutoPTZ.exe"
} else {
    $Out = "dist\AutoPTZ\AutoPTZ.exe"
}

if (-not (Test-Path $Out)) {
    Write-Error "Build did not produce $Out"
    exit 1
}
Write-Host "==> Built $Out"

@'

============================================================================
 NEXT STEPS — code-sign ON YOUR MACHINE (needs YOUR code-signing certificate)
============================================================================
 Not run here: signing needs your OV/EV code-signing certificate + signtool.exe
 (from the Windows SDK).

 1) Sign the exe (and, for onedir, the folder) with a timestamp:
      signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 ^
        /a dist\AutoPTZ\AutoPTZ.exe
      signtool verify /pa /v dist\AutoPTZ\AutoPTZ.exe

 2) (Optional) Build an installer with Inno Setup or WiX, then sign the
    installer the same way.

 CUDA / TensorRT variant (NVIDIA GPUs, instead of the DirectML default):
      pip uninstall -y onnxruntime onnxruntime-directml
      pip install -r requirements\gpu-nvidia.txt   # onnxruntime-gpu (CUDA+TensorRT EPs)
    then rebuild.  Requires CUDA 12.x + cuDNN 9.x (CUDA EP) and TensorRT 10.x
    (TensorRT EP) installed on the build + target machines.
============================================================================
'@ | Write-Host
