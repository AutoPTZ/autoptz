# build_windows.ps1 - build AutoPTZ for Windows x64 with PyInstaller.
#
# Produces dist\AutoPTZ\AutoPTZ.exe (onedir).  Set -OneFile for a single exe.
# DirectML is the DEFAULT inference EP on Windows (works on any DX12 GPU with no
# CUDA install); see the CUDA/TensorRT note at the bottom for the NVIDIA variant.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -OneFile
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -Accelerator nvidia
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -SkipInstall

param(
    [switch]$OneFile,
    [switch]$SkipInstall,
    [string]$Accelerator = ""
)

$ErrorActionPreference = "Stop"

# Resolve repo root (this script lives in packaging\).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
Set-Location $Root

if ($env:VENV) { $Venv = $env:VENV } else { $Venv = ".venv" }
$Py = Join-Path $Venv "Scripts\python.exe"

Write-Host "==> AutoPTZ Windows build"
Write-Host "    repo: $Root"
Write-Host "    venv: $Venv"

# -- 1. venv ------------------------------------------------------------------
if (-not (Test-Path $Py)) {
    Write-Host "==> Creating venv at $Venv"
    py -3.12 -m venv $Venv
}

# -- 2. dependencies ----------------------------------------------------------
# DirectML EP for onnxruntime: install onnxruntime-directml in place of the CPU
# onnxruntime from base.txt (DirectML accelerates on any DX12 GPU).
if (-not $SkipInstall) {
    if (-not $Accelerator) {
        if ($env:ACCELERATOR) { $Accelerator = $env:ACCELERATOR } else { $Accelerator = "directml" }
    }
    Write-Host "==> Installing dependencies (accelerator=$Accelerator)"
    & $Py tools\install.py --upgrade-pip --packaging --editable --accelerator $Accelerator
} else {
    Write-Host "==> -SkipInstall - using existing venv as-is"
}

# -- 3. (optional) pre-fetch the detector model -------------------------------
# Uncomment to bake the YOLO11 ONNX into autoptz\models before building:
#   & $Py -m tools.fetch_models --cache-dir autoptz\models

# -- 4. NDI runtime (optional) ------------------------------------------------
# cyndilib is installed from requirements/base.txt and collected by the spec when
# present. If you need to add an external NDI runtime DLL, drop it in
# packaging\ndi\ (or set $env:NDI_RUNTIME) before building.

# -- 5. build -----------------------------------------------------------------
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

# -- 6. (optional) Inno Setup installer ---------------------------------------
# MakeInstaller (or $env:MAKE_INSTALLER=1) compiles the onedir bundle into
# dist\AutoPTZ-<version>-windows-x64-setup.exe when iscc.exe (Inno Setup 6) is on
# PATH. The release workflow installs Inno Setup and sets this. Skipped for
# -OneFile (the installer packages the onedir tree).
$MakeInstaller = $env:MAKE_INSTALLER -eq "1"
if (-not $OneFile -and $MakeInstaller) {
    $Iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue)
    if ($null -eq $Iscc) {
        Write-Warning "iscc.exe (Inno Setup) not found on PATH; skipping installer."
    } else {
        $Ver = (& $Py -c "import autoptz; print(autoptz.__version__)").Trim()
        Write-Host "==> Compiling Inno Setup installer (v$Ver)"
        & $Iscc.Source "/DMyAppVersion=$Ver" "packaging\autoptz.iss"
        $Installer = "dist\AutoPTZ-$Ver-windows-x64-setup.exe"
        if (Test-Path $Installer) { Write-Host "==> Built $Installer" }
        else { Write-Error "Inno Setup did not produce $Installer"; exit 1 }
    }
}

@'

============================================================================
 NEXT STEPS - code-sign ON YOUR MACHINE (needs YOUR code-signing certificate)
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
      powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -Accelerator nvidia
    then rebuild.  Requires CUDA 12.x + cuDNN 9.x (CUDA EP) and TensorRT 10.x
    (TensorRT EP) installed on the build + target machines.
============================================================================
'@ | Write-Host
