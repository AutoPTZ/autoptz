# Building installers

AutoPTZ ships as a PyInstaller bundle per OS, packaged into a `.dmg` (macOS),
an Inno Setup `.exe` installer (Windows), and an `AppImage` (Linux). The
[release workflow](../.github/workflows/release.yml) does all three on a `v*`
tag; the scripts below build them locally.

The app icon is generated from `autoptz/assets/AutoPTZLogo.png` into
`packaging/AutoPTZ.icns` / `AutoPTZ.ico` / `AutoPTZ-256.png` by
`python packaging/make_icons.py` (re-run if the logo changes).

## macOS → `.app` + `.dmg`

```bash
bash packaging/build_macos.sh                 # dist/AutoPTZ.app
MAKE_DMG=1 bash packaging/build_macos.sh      # + dist/AutoPTZ-<ver>-macos-arm64.dmg
```

Produces an unsigned, correctly-named bundle (`CFBundleName=AutoPTZ`). Signing +
notarization need your Apple Developer ID — the exact `codesign` / `notarytool` /
`stapler` commands are printed at the end of the script.

## Windows → `.exe` + installer

Needs [Inno Setup 6](https://jrsoftware.org/isdl.php) on `PATH` (`iscc.exe`).

```powershell
# onedir bundle + installer:
$env:MAKE_INSTALLER = "1"
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
# -> dist\AutoPTZ\AutoPTZ.exe and dist\AutoPTZ-<ver>-windows-x64-setup.exe
```

DirectML is the default EP on Windows; pass nothing extra. For NVIDIA, install
with `-Accelerator nvidia` before building. Signing uses `signtool.exe` (see the
notes printed by the script).

## Linux → AppImage

Install the Qt system libs first (Debian/Ubuntu):

```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libdbus-1-3 libfuse2 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0
MAKE_APPIMAGE=1 bash packaging/build_linux.sh
# -> dist/AutoPTZ/AutoPTZ and dist/AutoPTZ-<ver>-linux-x86_64.AppImage
```

`appimagetool` is downloaded automatically if it isn't on `PATH`.

Packaging scripts install dependencies through `tools/install.py`. Set
`ACCELERATOR=nvidia` or `ACCELERATOR=openvino` for a hardware-specific Linux
build; the default release build stays CPU for portability.

## CI release

Push a tag to build + publish all three:

```bash
git tag v2.0.0 && git push origin v2.0.0      # v2.0.0-rc1 → marked pre-release
```

The workflow builds on macOS/Windows/Linux runners and attaches the artifacts to
a GitHub Release (auto-generated notes). Keep the asset suffixes stable
(`.dmg`, `windows-x64-setup.exe`, `.AppImage`): the in-app updater uses them to
choose the right download for the current OS, launch it, and close AutoPTZ.

## The spec

`packaging/autoptz.spec` is the shared PyInstaller spec: it bundles the package,
assets (logo), optional pre-fetched models, and PySide6 plugins; sets the
app/exe/bundle icon; and runs `multiprocessing.freeze_support()` for spawned
workers. Set `ONEFILE=1` for a single-file Windows exe (not used by the
installer, which packages the onedir tree).
