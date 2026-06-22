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

Produces a correctly-named bundle (`CFBundleName=AutoPTZ`). By default it is
**unsigned** and the script prints the manual `codesign` / `notarytool` / `stapler`
commands at the end.

### Signing + notarization (opt-in)

Set `MACOS_SIGN_IDENTITY` to your Developer ID and the script signs the `.app` (and,
with `MAKE_DMG=1`, the `.dmg`). Add notary credentials and it also notarizes +
staples:

```bash
export MACOS_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
# notarize too (either a stored profile, or Apple-ID creds):
export MACOS_NOTARY_KEYCHAIN_PROFILE="AUTOPTZ_NOTARY"     # from `notarytool store-credentials`
#   …or…
export MACOS_NOTARY_APPLE_ID="you@example.com"
export MACOS_NOTARY_TEAM_ID="TEAMID"
export MACOS_NOTARY_PASSWORD="app-specific-password"
MAKE_DMG=1 bash packaging/build_macos.sh                 # signed + notarized + stapled .dmg
```

`security find-identity -v -p codesigning` lists your identity + Team ID. Entitlements
come from `packaging/entitlements.plist` (hardened runtime, required for notarization).

> **You need a "Developer ID Application" certificate.** An **"Apple Development"**
> cert (the default Xcode one) signs locally but **cannot be notarized** — Apple
> returns `status: Invalid`. Create a Developer ID cert (needs the paid Apple
> Developer Program): Xcode → Settings → Accounts → your team → Manage Certificates →
> **+** → *Developer ID Application* (or via developer.apple.com → Certificates).
> Then export it from Keychain Access as a `.p12`.

If signing succeeds but notarization can't (e.g. wrong cert type), the script ships a
**signed-but-not-notarized** build by default rather than failing — so a signing
problem never blocks the Windows/Linux release. Set **`MACOS_SIGN_REQUIRED=1`** to make
a non-notarized result a hard error once your Developer ID cert is in place.

### Signed releases in CI

The [release workflow](../.github/workflows/release.yml) signs + notarizes the published
`.dmg` automatically once these repository secrets are set (Settings → Secrets and
variables → Actions); without them it still builds an unsigned `.dmg`:

| Secret | What it is |
| --- | --- |
| `MACOS_CERTIFICATE_P12_BASE64` | Your "Developer ID Application" cert exported as `.p12`, base64-encoded (`base64 -i cert.p12 \| pbcopy`). |
| `MACOS_CERTIFICATE_PASSWORD` | The password you set when exporting the `.p12`. |
| `MACOS_SIGN_IDENTITY` | `Developer ID Application: Your Name (TEAMID)`. |
| `MACOS_NOTARY_APPLE_ID` | Apple ID email for notarization. |
| `MACOS_NOTARY_TEAM_ID` | Your 10-character Team ID. |
| `MACOS_NOTARY_PASSWORD` | An app-specific password for that Apple ID. |

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
app/exe/bundle icon; and runs `multiprocessing.freeze_support()` so the
spawn-based multiprocessing the app uses (e.g. the selftest's shared-memory probe)
works under a frozen build. Set `ONEFILE=1` for a single-file Windows exe (not used
by the installer, which packages the onedir tree).
