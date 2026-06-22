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

# ── signing config (opt-in) ──────────────────────────────────────────────────
# Sign + notarize only when MACOS_SIGN_IDENTITY is set (CI sets it from repo
# secrets; locally, export it to sign with your Developer ID, e.g.
#   export MACOS_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
# Unset → unsigned build, and the manual sign/notarize commands are printed at
# the end instead.
SIGN_IDENTITY="${MACOS_SIGN_IDENTITY:-}"

# Notarize + staple a signed artifact (.app or .dmg) with whichever credentials
# are available; a no-op (with a note) when none are set. On a non-Accepted
# submission it prints Apple's detailed log, then by default warns and continues
# (the build ships signed-but-not-notarized) so a signing problem can't brick the
# whole cross-platform release. Set MACOS_SIGN_REQUIRED=1 to make it fatal once a
# valid Developer ID Application cert is in place.
notarize_and_staple() {
    local target="$1"
    local creds=()
    if [[ -n "${MACOS_NOTARY_KEYCHAIN_PROFILE:-}" ]]; then
        creds=(--keychain-profile "${MACOS_NOTARY_KEYCHAIN_PROFILE}")
    elif [[ -n "${MACOS_NOTARY_APPLE_ID:-}" && -n "${MACOS_NOTARY_TEAM_ID:-}" \
            && -n "${MACOS_NOTARY_PASSWORD:-}" ]]; then
        creds=(--apple-id "${MACOS_NOTARY_APPLE_ID}" --team-id "${MACOS_NOTARY_TEAM_ID}"
               --password "${MACOS_NOTARY_PASSWORD}")
    else
        echo "==> Signed but NOT notarized (no notary credentials): ${target}"
        echo "    Set MACOS_NOTARY_APPLE_ID/TEAM_ID/PASSWORD or MACOS_NOTARY_KEYCHAIN_PROFILE."
        return 0
    fi
    echo "==> Submitting ${target} to Apple's notary service…"
    local out submission_id
    # `|| true`: notarytool exits non-zero on a rejected submission; we want to
    # inspect the status and fetch the log ourselves rather than abort here.
    out="$(xcrun notarytool submit "${target}" "${creds[@]}" --wait 2>&1)" || true
    echo "${out}"
    if ! printf '%s\n' "${out}" | grep -q "status: Accepted"; then
        submission_id="$(printf '%s\n' "${out}" | awk -F'[: ]+' '/^[[:space:]]*id:/{print $3; exit}')"
        echo "!! Notarization was NOT Accepted for ${target}."
        if [[ -n "${submission_id}" ]]; then
            echo "==> Apple notary log for ${submission_id}:"
            xcrun notarytool log "${submission_id}" "${creds[@]}" || true
        fi
        echo "   Notarization requires a 'Developer ID Application' certificate; an"
        echo "   'Apple Development' cert cannot notarize. See docs/building.md."
        if [[ "${MACOS_SIGN_REQUIRED:-0}" == "1" ]]; then
            echo "!! MACOS_SIGN_REQUIRED=1 — failing the build."
            return 1
        fi
        echo "==> Continuing with a signed-but-NOT-notarized ${target}. Users will"
        echo "    see a Gatekeeper warning until a Developer ID cert is configured."
        return 0
    fi
    echo "==> Stapling ${target}"
    xcrun stapler staple "${target}"
}

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

# ── 5b. sign the .app (opt-in) ───────────────────────────────────────────────
# Deep-sign with the hardened runtime + entitlements so the bundle is notarizable.
# Must happen BEFORE the .dmg is built so the dmg ships the signed app.
if [[ -n "${SIGN_IDENTITY}" ]]; then
    echo "==> Codesigning ${APP} with: ${SIGN_IDENTITY}"
    # Sign every nested Mach-O (dylib/.so) first, inside-out. `codesign --deep` is
    # unreliable for notarization — Apple returns "Invalid" when any nested binary
    # is unsigned or carries an old signature — so sign them explicitly with the
    # hardened runtime + a secure timestamp before sealing the bundle.
    while IFS= read -r -d '' lib; do
        codesign --force --options runtime --timestamp --sign "${SIGN_IDENTITY}" "${lib}" \
            || { echo "!! codesign failed for ${lib}"; exit 1; }
    done < <(find "${APP}/Contents" -type f \( -name "*.dylib" -o -name "*.so" \))
    # Then the app bundle itself, carrying the entitlements.
    codesign --force --options runtime --timestamp \
        --entitlements packaging/entitlements.plist \
        --sign "${SIGN_IDENTITY}" "${APP}"
    codesign --verify --deep --strict --verbose=2 "${APP}"
    echo "==> ${APP} signed"
fi

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

    # Sign + notarize + staple the distributed .dmg (opt-in).
    if [[ -n "${SIGN_IDENTITY}" ]]; then
        echo "==> Codesigning ${DMG}"
        codesign --force --timestamp --sign "${SIGN_IDENTITY}" "${DMG}"
        notarize_and_staple "${DMG}"
    fi
fi

if [[ -n "${SIGN_IDENTITY}" ]]; then
    echo
    echo "==> Signed build complete (notarized if notary credentials were provided)."
    exit 0
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
