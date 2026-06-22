#!/usr/bin/env bash
#
# build_macos.sh — build AutoPTZ.app for the host macOS architecture
# (Apple Silicon arm64 or Intel x86_64; PyInstaller freezes host-native).
#
# Produces dist/AutoPTZ.app, an unsigned bundle whose Info.plist declares
# CFBundleName=AutoPTZ — that is what makes the macOS app menu read "AutoPTZ"
# instead of "Python".  Signing + notarization are NOT run here (they need YOUR
# Apple Developer ID + Team ID); the exact commands are printed at the end as
# commented TODOs.
#
# Usage:
#   bash packaging/build_macos.sh
#   SKIP_INSTALL=1 bash packaging/build_macos.sh   # reuse the existing venv
#
set -euo pipefail

# Resolve repo root (this script lives in packaging/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

VENV="${VENV:-.venv}"
PY="${VENV}/bin/python"

echo "==> AutoPTZ macOS build"
echo "    repo:  ${ROOT}"
echo "    venv:  ${VENV}"

# ── 1. venv ─────────────────────────────────────────────────────────────────
if [[ ! -x "${PY}" ]]; then
    echo "==> Creating venv at ${VENV}"
    python3.12 -m venv "${VENV}"
fi

# ── 2. dependencies ─────────────────────────────────────────────────────────
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo "==> Installing dependencies"
    "${PY}" tools/install.py --upgrade-pip --packaging --editable --accelerator cpu
else
    echo "==> SKIP_INSTALL=1 — using existing venv as-is"
fi

# ── 3. (optional) pre-fetch the detector model so it ships inside the app ─────
# The app also auto-downloads on first run, but bundling makes it work offline.
# Uncomment to bake the YOLO11 ONNX into autoptz/models before building:
#   echo "==> Pre-fetching detector model"
#   "${PY}" -m tools.fetch_models --cache-dir autoptz/models

# ── 4. NDI runtime (optional) ────────────────────────────────────────────────
# libndi is a system install, not a pip wheel.  To bundle it, drop the dylib in
# packaging/ndi/ (or export NDI_RUNTIME=/path/to/dir) before building; the spec
# picks it up automatically.  Otherwise NDI ingest degrades gracefully.

# ── 5. build ────────────────────────────────────────────────────────────────
echo "==> Cleaning previous build"
rm -rf build dist

echo "==> Running PyInstaller"
"${PY}" -m PyInstaller packaging/autoptz.spec --noconfirm

APP="dist/AutoPTZ.app"
if [[ ! -d "${APP}" ]]; then
    echo "!! Build did not produce ${APP}" >&2
    exit 1
fi

echo "==> Built ${APP}"
echo "==> Verifying CFBundleName (the app-name fix):"
/usr/libexec/PlistBuddy -c 'Print :CFBundleName' "${APP}/Contents/Info.plist"
plutil -lint "${APP}/Contents/Info.plist"

# ── 6. (optional) DMG ─────────────────────────────────────────────────────────
# MAKE_DMG=1 produces a compressed, versioned dmg with an /Applications symlink
# (drag-to-install).  Dependency-free (uses macOS hdiutil).  The release workflow
# sets MAKE_DMG=1; local builds skip it by default.
if [[ "${MAKE_DMG:-0}" == "1" ]]; then
    VER="$("${PY}" -c 'import autoptz; print(autoptz.__version__)')"
    # Native host architecture: "arm64" on Apple Silicon, "x86_64" on Intel.
    # PyInstaller (target_arch=None) freezes for the host, so the dmg name must
    # follow suit instead of always claiming arm64.
    ARCH="$(uname -m)"
    DMG="dist/AutoPTZ-${VER}-macos-${ARCH}.dmg"
    echo "==> Building ${DMG}"
    STAGE="$(mktemp -d)"
    cp -R "${APP}" "${STAGE}/"
    ln -s /Applications "${STAGE}/Applications"
    rm -f "${DMG}"
    hdiutil create -volname "AutoPTZ ${VER}" -srcfolder "${STAGE}" \
        -ov -format UDZO "${DMG}"
    rm -rf "${STAGE}"
    echo "==> Built ${DMG}"
fi

cat <<'EOF'

============================================================================
 NEXT STEPS — sign & notarize ON YOUR MAC (needs YOUR Apple Developer ID)
============================================================================
 These are intentionally NOT run by this script: they require your private
 "Developer ID Application" certificate (in your login Keychain) and Team ID.

 0) Find your signing identity + Team ID:
      security find-identity -v -p codesigning

 1) Deep-sign the bundle with the hardened runtime + entitlements:
      codesign --deep --force --options runtime --timestamp \
        --entitlements packaging/entitlements.plist \
        --sign "Developer ID Application: <YOUR NAME> (<TEAM_ID>)" \
        dist/AutoPTZ.app
      # verify:
      codesign --verify --deep --strict --verbose=2 dist/AutoPTZ.app
      spctl -a -vvv -t exec dist/AutoPTZ.app   # may say "rejected" until notarized

 2) Zip and submit to Apple's notary service:
      ditto -c -k --keepParent dist/AutoPTZ.app dist/AutoPTZ.zip
      # one-time: store creds in a keychain profile named AUTOPTZ_NOTARY
      xcrun notarytool store-credentials AUTOPTZ_NOTARY \
        --apple-id "<you@example.com>" --team-id "<TEAM_ID>" \
        --password "<app-specific-password>"
      xcrun notarytool submit dist/AutoPTZ.zip \
        --keychain-profile AUTOPTZ_NOTARY --wait

 3) Staple the ticket so it validates offline, then re-zip for distribution:
      xcrun stapler staple dist/AutoPTZ.app
      ditto -c -k --keepParent dist/AutoPTZ.app dist/AutoPTZ-notarized.zip

 (Optional) Build a DMG instead of a zip:
      hdiutil create -volname AutoPTZ -srcfolder dist/AutoPTZ.app \
        -ov -format UDZO dist/AutoPTZ.dmg
      # sign + notarize the .dmg the same way (codesign the .dmg, notarytool
      # submit the .dmg, stapler staple the .dmg).
============================================================================
EOF
