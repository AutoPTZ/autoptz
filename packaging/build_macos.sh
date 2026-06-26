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
SIGN_REQUIRED="${MACOS_SIGN_REQUIRED:-0}"

if [[ "${SIGN_REQUIRED}" == "1" && -z "${SIGN_IDENTITY}" ]]; then
    echo "!! MACOS_SIGN_REQUIRED=1 but MACOS_SIGN_IDENTITY is empty." >&2
    exit 1
fi

require_developer_id_signature() {
    local target="$1"
    local authorities
    authorities="$(codesign -dvv "${target}" 2>&1 | grep -i "^Authority=" || true)"
    printf '%s\n' "${authorities}"
    if ! printf '%s\n' "${authorities}" | grep -q "^Authority=Developer ID Application:"; then
        echo "!! ${target} is not signed with a Developer ID Application certificate." >&2
        echo "   Export a 'Developer ID Application' certificate as .p12 and set" >&2
        echo "   MACOS_SIGN_IDENTITY to the matching identity or SHA-1 hash." >&2
        return 1
    fi
}

sign_macho_files() {
    local app="$1"
    local signed=0
    local candidate
    while IFS= read -r -d '' candidate; do
        if file -b "${candidate}" | grep -q "Mach-O"; then
            codesign --force --options runtime --timestamp --sign "${SIGN_IDENTITY}" "${candidate}" \
                || { echo "!! codesign failed for ${candidate}"; exit 1; }
            signed=$((signed + 1))
        fi
    done < <(find "${app}/Contents" -type f -print0)
    echo "==> Signed ${signed} nested Mach-O file(s)"
}

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
        if [[ "${SIGN_REQUIRED}" == "1" ]]; then
            echo "!! MACOS_SIGN_REQUIRED=1 — failing because notarization credentials are missing."
            return 1
        fi
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
        if [[ "${SIGN_REQUIRED}" == "1" ]]; then
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

# Build a compressed DMG, retrying a few times on transient failures. hdiutil's
# image creation intermittently dies on CI with "hdiutil: create failed -
# Resource busy" — a disk-arbitration race (a prior volume not fully detached,
# Spotlight indexing the scratch dir, etc.), not a problem with the bundle, which
# is already built and signed by this point. Without a retry, that flake sinks an
# otherwise-good build — including the release-gating arm64 dmg, where it would
# block the whole cross-platform release.
create_dmg() {
    local volname="$1" srcfolder="$2" dmg="$3"
    local attempt
    for (( attempt = 1; attempt <= 5; attempt++ )); do
        rm -f "${dmg}"
        if hdiutil create -volname "${volname}" -srcfolder "${srcfolder}" \
                -ov -format UDZO "${dmg}"; then
            return 0
        fi
        if (( attempt < 5 )); then
            echo "!! hdiutil create failed (attempt ${attempt}/5) — retrying in $(( attempt * 5 ))s…" >&2
            sleep "$(( attempt * 5 ))"
        fi
    done
    echo "!! hdiutil create failed after 5 attempts for ${dmg}" >&2
    return 1
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

# ── 3. pre-fetch the detector + pose models so they ship inside the app ───────
# Bundling the YOLO11 ONNX into autoptz/models makes detection work on first
# launch with no download / network / setup.  Best-effort: if the fetch fails the
# app still downloads the prebuilt ONNX on demand (see engine/runtime/models.py).
echo "==> Pre-fetching models into autoptz/models (bundled, zero-setup)"
"${PY}" -m tools.fetch_models --cache-dir autoptz/models \
    || echo "!! model pre-fetch failed; the app will download them on first run"

# ── 4. NDI runtime (optional) ────────────────────────────────────────────────
# cyndilib is installed from requirements/base.txt and collected by the spec when
# present. If you need to add an external NDI runtime dylib, drop it in
# packaging/ndi/ (or export NDI_RUNTIME=/path/to/dir) before building.

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
    # Sign every nested Mach-O first. `codesign --deep` and extension-only scans can
    # miss Python/PyInstaller binaries with no .dylib/.so suffix, which notarization
    # reports later as "not signed with a valid Developer ID certificate".
    sign_macho_files "${APP}"
    # Then the app bundle itself, carrying the entitlements.
    codesign --force --options runtime --timestamp \
        --entitlements packaging/entitlements.plist \
        --sign "${SIGN_IDENTITY}" "${APP}"
    codesign --verify --deep --strict --verbose=2 "${APP}"
    # Diagnostic: which certificate actually signed the bundle? Notarization needs
    # "Authority=Developer ID Application: …"; "Apple Development: …" is rejected.
    echo "==> Signing authority on ${APP}:"
    require_developer_id_signature "${APP}"
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
    create_dmg "AutoPTZ ${VER}" "${STAGE}" "${DMG}"
    rm -rf "${STAGE}"
    echo "==> Built ${DMG}"

    # Sign + notarize + staple the distributed .dmg (opt-in).
    if [[ -n "${SIGN_IDENTITY}" ]]; then
        echo "==> Codesigning ${DMG}"
        codesign --force --timestamp --sign "${SIGN_IDENTITY}" "${DMG}"
        echo "==> Signing authority on ${DMG}:"
        require_developer_id_signature "${DMG}"
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
