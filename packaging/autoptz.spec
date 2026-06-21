# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AutoPTZ — builds a native macOS .app and a Windows exe.

Build (from the repo root, with PyInstaller installed in the venv):

    pyinstaller packaging/autoptz.spec --noconfirm

Outputs:
    macOS    -> dist/AutoPTZ.app   (proper bundle; menu reads "AutoPTZ")
    Windows  -> dist/AutoPTZ/AutoPTZ.exe  (onedir; flip ONEFILE=1 for onefile)

The entry point is autoptz/__main__.py so the frozen app behaves exactly like
`python -m autoptz` (it calls multiprocessing.freeze_support() — REQUIRED for
the spawn-based engine workers to start under a frozen build).

What gets bundled
-----------------
* the whole `autoptz` package (collected as data + submodules so dynamic
  imports like the engine pipeline / ptz backends are present)
* autoptz/assets (the logo) and autoptz/models (if non-empty) so a pre-fetched
  detector ONNX can ship inside the app
* PySide6 Qt plugins needed for a Qt Widgets app: platforms, styles,
  imageformats, iconengines

Notes
-----
* macOS: the BUNDLE step writes packaging/Info.plist into
  AutoPTZ.app/Contents/Info.plist.  CFBundleName=AutoPTZ there is the real fix
  for the app menu showing "Python".
* NDI: the NDI runtime (libndi) is a system install, not a pip wheel.  To ship
  it inside the app, drop the dylib/dll next to this spec (or point NDI_RUNTIME
  at it) and it will be added to the bundle's Frameworks/binaries — see below.
* Models: the YOLO11 detector ONNX auto-downloads on first run via
  ModelManager, OR pre-fetch it with `python -m tools.fetch_models` and copy it
  into autoptz/models/ before building to ship it inside the app.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# ── paths ──────────────────────────────────────────────────────────────────────
# PyInstaller execs the spec with SPECPATH set to the spec's directory.
SPEC_DIR = Path(globals().get("SPECPATH", os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = SPEC_DIR.parent
PKG_DIR = PROJECT_ROOT / "autoptz"

IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

# Toggle a single-file Windows exe with:  ONEFILE=1 pyinstaller packaging/autoptz.spec
ONEFILE = os.environ.get("ONEFILE", "0") == "1"

APP_NAME = "AutoPTZ"
ENTRY = str(PKG_DIR / "__main__.py")

# Optional icons (provide your own; build still works without them).
ICON_ICNS = SPEC_DIR / "AutoPTZ.icns"   # macOS
ICON_ICO = SPEC_DIR / "AutoPTZ.ico"     # Windows


# ── data files ─────────────────────────────────────────────────────────────────
def _tree(src: Path, dest: str) -> list[tuple[str, str]]:
    """Collect every file under `src` mapped into bundle `dest/<relpath>`."""
    out: list[tuple[str, str]] = []
    if not src.exists():
        return out
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix in {".pyc", ".pyo"} or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(src).parent
        out.append((str(f), str(Path(dest) / rel)))
    return out


datas: list[tuple[str, str]] = []

# Ship assets (logo) and any pre-fetched model weights (skips empty /
# .gitkeep-only dirs gracefully — _tree yields nothing for empty trees).
datas += _tree(PKG_DIR / "assets", "autoptz/assets")
datas += _tree(PKG_DIR / "models", "autoptz/models")

# PySide6 / Qt Widgets: the bundled hook collects the bulk; make sure the Qt
# plugins a Widgets app needs come along (no QtQuick/QML — the UI is Widgets).
datas += collect_data_files("PySide6", includes=[
    "Qt/plugins/platforms/**",
    "Qt/plugins/styles/**",
    "Qt/plugins/imageformats/**",
    "Qt/plugins/iconengines/**",
])

# onnxruntime / cv2 ship data + native libs that the contrib hooks normally
# handle; collect defensively so EP providers and codecs are present.
binaries: list[tuple[str, str]] = []
for mod in ("onnxruntime", "cv2"):
    try:
        binaries += collect_dynamic_libs(mod)
    except Exception:
        pass

# Optional: bundle an NDI runtime if present beside the spec (NDI_RUNTIME env or
# packaging/ndi/).  libndi is NOT a pip wheel; see docs/building.md.
_ndi = os.environ.get("NDI_RUNTIME")
_ndi_dir = Path(_ndi) if _ndi else (SPEC_DIR / "ndi")
if _ndi_dir.exists():
    for lib in _ndi_dir.glob("*"):
        if lib.is_file() and lib.suffix in {".dylib", ".so", ".dll"}:
            binaries.append((str(lib), "."))


# ── hidden imports ─────────────────────────────────────────────────────────────
# Collect the full autoptz package so dynamically-imported engine modules
# (pipeline.*, ptz.* backends, discovery.*, identity.*) are not pruned.
hiddenimports: list[str] = []
hiddenimports += collect_submodules("autoptz")
hiddenimports += [
    "PySide6.QtWidgets",
    "PySide6.QtGui",
    "PySide6.QtCore",
]
# ML / engine deps that are imported lazily and may be missed by static analysis.
for opt in (
    "onnxruntime", "cv2", "numpy", "scipy", "PIL",
    "msgpack", "pydantic", "serial",
):
    hiddenimports.append(opt)

# Heavy optional deps: include their submodules ONLY if installed in the build
# env (keeps a UI-only build from failing the analysis).
for opt_pkg in ("ultralytics", "boxmot", "insightface", "av", "onnx"):
    try:
        __import__(opt_pkg)
    except Exception:
        continue
    try:
        hiddenimports += collect_submodules(opt_pkg)
        datas += collect_data_files(opt_pkg)
    except Exception:
        pass

# Trim obvious bloat that PySide6 pulls in but AutoPTZ never uses.
excludes = [
    "tkinter",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtMultimedia",
    "matplotlib",
    "pytest",
]


# ── analysis / build graph ─────────────────────────────────────────────────────
block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)


# ── executable ─────────────────────────────────────────────────────────────────
_exe_icon = None
if IS_WINDOWS and ICON_ICO.exists():
    _exe_icon = str(ICON_ICO)
elif IS_MACOS and ICON_ICNS.exists():
    _exe_icon = str(ICON_ICNS)

if ONEFILE:
    # Single-file build (Windows convenience; not used for the macOS .app).
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,      # set True only if you need macOS file-open events
        target_arch=None,
        codesign_identity=None,    # sign post-build with codesign (see build script)
        entitlements_file=None,
        icon=_exe_icon,
    )
else:
    # onedir: a thin exe + a COLLECT folder of libs/data (preferred — faster
    # startup, and required for the macOS BUNDLE step).
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=_exe_icon,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=APP_NAME,
    )


# ── macOS .app bundle (the app-name fix lives here) ─────────────────────────────
if IS_MACOS and not ONEFILE:
    import plistlib

    _plist_path = SPEC_DIR / "Info.plist"
    with open(_plist_path, "rb") as fh:
        info_plist = plistlib.load(fh)

    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=str(ICON_ICNS) if ICON_ICNS.exists() else None,
        bundle_identifier=info_plist.get("CFBundleIdentifier", "com.autoptz.app"),
        version=info_plist.get("CFBundleShortVersionString", "2.0.0"),
        # info_plist wins over PyInstaller's auto-generated keys — this is where
        # CFBundleName=AutoPTZ + NSCameraUsageDescription get baked in.
        info_plist=info_plist,
    )
