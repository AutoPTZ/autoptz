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
MAKE_DMG=1 bash packaging/build_macos.sh      # + dist/AutoPTZ-<ver>-macos-<arch>.dmg
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

If signing succeeds but notarization can't, the script ships a
**signed-but-not-notarized** local build by default. Set
**`MACOS_SIGN_REQUIRED=1`** for release validation; CI sets this, so a bad
certificate, missing notary credential, or rejected notarization fails the macOS
release job instead of publishing a Gatekeeper-blocked `.dmg`.

### Signed releases in CI

The [release workflow](../.github/workflows/release.yml) signs + notarizes the published
`.dmg` automatically once these repository secrets are set (Settings → Secrets and
variables → Actions). Release CI requires all of them:

| Secret | What it is |
| --- | --- |
| `MACOS_CERTIFICATE_P12_BASE64` | Your "Developer ID Application" cert exported as `.p12`, base64-encoded (`base64 -i cert.p12 \| pbcopy`). |
| `MACOS_CERTIFICATE_PASSWORD` | The password you set when exporting the `.p12`. |
| `MACOS_SIGN_IDENTITY` | Prefer the SHA-1 hash from `security find-identity -v -p codesigning`; the full `Developer ID Application: Your Name (TEAMID)` string also works. |
| `MACOS_NOTARY_APPLE_ID` | Apple ID email for notarization. |
| `MACOS_NOTARY_TEAM_ID` | Your 10-character Team ID. |
| `MACOS_NOTARY_PASSWORD` | An app-specific password for that Apple ID, not the Apple ID login password. |

#### Exact macOS signing setup

1. Join the paid Apple Developer Program for the Team ID you will ship under.
2. Create a **Developer ID Application** certificate. Do not use **Apple
   Development**; it can sign locally but Apple notarization rejects it.
3. In Keychain Access, find the `Developer ID Application: ... (TEAMID)`
   certificate, expand it, confirm it has a private key, then export the
   certificate **and private key** as `cert.p12`. Set a strong export password.
4. Verify the exported certificate locally:

   ```bash
   security import cert.p12 -k ~/Library/Keychains/login.keychain-db -P '<p12 export password>' -T /usr/bin/codesign
   security find-identity -v -p codesigning | grep "Developer ID Application"
   ```

   Copy the SHA-1 hash at the start of the matching line; that is the safest value
   for `MACOS_SIGN_IDENTITY`.

5. Create an Apple app-specific password for notarization, then set repository
   secrets:

   ```bash
   gh secret set MACOS_CERTIFICATE_P12_BASE64 --body "$(base64 -i cert.p12)"
   gh secret set MACOS_CERTIFICATE_PASSWORD --body '<p12 export password>'
   gh secret set MACOS_SIGN_IDENTITY --body '<SHA-1 from security find-identity>'
   gh secret set MACOS_NOTARY_APPLE_ID --body 'you@example.com'
   gh secret set MACOS_NOTARY_TEAM_ID --body '<TEAMID>'
   gh secret set MACOS_NOTARY_PASSWORD --body '<app-specific password>'
   ```

6. Test before tagging:

   ```bash
   export MACOS_SIGN_IDENTITY='<SHA-1 from security find-identity>'
   export MACOS_NOTARY_APPLE_ID='you@example.com'
   export MACOS_NOTARY_TEAM_ID='<TEAMID>'
   export MACOS_NOTARY_PASSWORD='<app-specific password>'
   MACOS_SIGN_REQUIRED=1 MAKE_DMG=1 bash packaging/build_macos.sh
   spctl -a -vvv -t open --context context:primary-signature dist/AutoPTZ-*-macos-*.dmg
   ```

   A valid release build prints `status: Accepted`, staples the `.dmg`, and exits
   successfully. If Apple reports `not signed with a valid Developer ID
   certificate`, the `.p12` or `MACOS_SIGN_IDENTITY` is wrong.

#### macOS architectures in CI

The release workflow builds arm64 on `macos-14` and x86_64 on `macos-26-intel`.
They intentionally stay separate because a universal2 PyInstaller bundle would
require every bundled native dependency (`onnxruntime`, OpenCV, PySide/Qt, Python
extensions, etc.) to be universal too. Both macOS artifacts gate the main
release. Publishing waits for arm64, x86_64, Windows, and Linux to succeed before
generating `SHA256SUMS`, so every uploaded installer has a checksum entry in the
same manifest.

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
