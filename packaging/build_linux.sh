#!/usr/bin/env bash
#
# build_linux.sh — build AutoPTZ on Linux x86_64 with PyInstaller, and
# (optionally) package a self-contained AppImage.
#
# Usage:
#   bash packaging/build_linux.sh                 # onedir in dist/AutoPTZ/
#   MAKE_APPIMAGE=1 bash packaging/build_linux.sh # also dist/AutoPTZ-<ver>-linux-x86_64.AppImage
#   SKIP_INSTALL=1 MAKE_APPIMAGE=1 bash packaging/build_linux.sh
#
# System libs needed at build/run time (Debian/Ubuntu):
#   sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libdbus-1-3 \
#        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
#        libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

VENV="${VENV:-.venv}"
PY="${VENV}/bin/python"

echo "==> AutoPTZ Linux build"
echo "    repo: ${ROOT}"
echo "    venv: ${VENV}"

# ── 1. venv ─────────────────────────────────────────────────────────────────
if [[ ! -x "${PY}" ]]; then
    echo "==> Creating venv at ${VENV}"
    python3.12 -m venv "${VENV}"
fi

# ── 2. dependencies ─────────────────────────────────────────────────────────
# Portable CPU build by default. Set ACCELERATOR=nvidia or ACCELERATOR=openvino
# for a hardware-specific local build.
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo "==> Installing dependencies"
    ACCELERATOR="${ACCELERATOR:-cpu}"

    # Force CPU-only torch for the portable CPU build. On Linux, PyPI's default
    # torch wheel bundles CUDA (~2.5 GB of nvidia libs), which both bloats the
    # AppImage and pushes it past GitHub's 2 GiB release-asset limit. Installing
    # the CPU wheels *first* (from PyTorch's CPU index) means the ultralytics /
    # boxmot requirements see torch already satisfied and never pull the CUDA
    # build. Skipped for the nvidia accelerator, which wants the CUDA wheels.
    if [[ "${ACCELERATOR}" == "cpu" ]]; then
        echo "==> Pre-installing CPU-only torch (keeps CUDA libs out of the build)"
        "${PY}" -m pip install --upgrade pip
        "${PY}" -m pip install --index-url https://download.pytorch.org/whl/cpu \
            torch torchvision
    fi

    "${PY}" tools/install.py \
        --upgrade-pip \
        --packaging \
        --editable \
        --accelerator "${ACCELERATOR}"
else
    echo "==> SKIP_INSTALL=1 — using existing venv as-is"
fi

# ── 3. build ────────────────────────────────────────────────────────────────
echo "==> Cleaning previous build"
rm -rf build dist

echo "==> Running PyInstaller"
"${PY}" -m PyInstaller packaging/autoptz.spec --noconfirm

ONEDIR="dist/AutoPTZ"
if [[ ! -x "${ONEDIR}/AutoPTZ" ]]; then
    echo "!! Build did not produce ${ONEDIR}/AutoPTZ" >&2
    exit 1
fi
echo "==> Built ${ONEDIR}/AutoPTZ"

# ── 4. (optional) AppImage ────────────────────────────────────────────────────
if [[ "${MAKE_APPIMAGE:-0}" == "1" ]]; then
    VER="$("${PY}" -c 'import autoptz; print(autoptz.__version__)')"
    APPDIR="dist/AutoPTZ.AppDir"
    echo "==> Assembling ${APPDIR}"
    rm -rf "${APPDIR}"
    mkdir -p "${APPDIR}/usr/bin"
    cp -r "${ONEDIR}" "${APPDIR}/usr/bin/AutoPTZ"
    cp packaging/AutoPTZ-256.png "${APPDIR}/AutoPTZ.png"

    cat > "${APPDIR}/AutoPTZ.desktop" <<'DESK'
[Desktop Entry]
Type=Application
Name=AutoPTZ
Comment=AI-driven PTZ camera tracking
Exec=AutoPTZ
Icon=AutoPTZ
Categories=AudioVideo;Video;
Terminal=false
DESK

    cat > "${APPDIR}/AppRun" <<'RUN'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "${HERE}/usr/bin/AutoPTZ/AutoPTZ" "$@"
RUN
    chmod +x "${APPDIR}/AppRun"

    # Locate (or fetch) appimagetool. It is itself an AppImage; run it with
    # extract-and-run so it works on CI runners without FUSE.
    TOOL="${APPIMAGETOOL:-$(command -v appimagetool || true)}"
    if [[ -z "${TOOL}" ]]; then
        echo "==> Downloading appimagetool"
        TOOL="$(mktemp -d)/appimagetool"
        curl -fsSL -o "${TOOL}" \
            "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
        chmod +x "${TOOL}"
    fi

    OUT="dist/AutoPTZ-${VER}-linux-x86_64.AppImage"
    echo "==> Building ${OUT}"
    APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 "${TOOL}" "${APPDIR}" "${OUT}"
    echo "==> Built ${OUT}"
fi
