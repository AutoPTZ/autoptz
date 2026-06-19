# 11 — Packaging & Distribution (Phase 10)

> Turns `python -m autoptz` into shippable, correctly-named native apps. The
> headline fix here is the **macOS app menu reading "AutoPTZ" instead of
> "Python"** — which only a real `.app` bundle with `CFBundleName=AutoPTZ` can
> deliver. Signing and notarization need *your* Apple Developer ID and Team ID,
> so those steps are documented (and printed by the build scripts) but never run
> automatically.

## Assets (all under `packaging/`)

| File | Purpose |
|------|---------|
| `autoptz.spec` | PyInstaller spec — builds `dist/AutoPTZ.app` (macOS) and `dist/AutoPTZ/AutoPTZ.exe` (Windows). Bundles the `autoptz` package, the QML tree (to `autoptz/ui/qml` so `app.py`'s `__file__`-relative lookup works frozen), `assets/`, `models/`, and PySide6 Qt plugins (platforms, quick, qml, labsplatform, styles, imageformats). |
| `Info.plist` | macOS bundle metadata. **`CFBundleName=AutoPTZ`** is the app-name fix. Also carries `CFBundleDisplayName`, `CFBundleIdentifier=com.autoptz.app`, `CFBundleShortVersionString=2.0.0`, `LSMinimumSystemVersion=12.0`, and the **required** `NSCameraUsageDescription` (no camera access without it). |
| `entitlements.plist` | Hardened-runtime entitlements: `com.apple.security.device.camera`, plus the JIT / unsigned-executable-memory / library-validation entitlements PySide6 + CPython + onnxruntime commonly need. Network client/server for NDI/ONVIF/RTSP. Signing-identity / Team-ID placeholders documented in comments. |
| `build_macos.sh` | Create venv → install `base + macos + packaging` reqs + `-e .` → PyInstaller → verify `CFBundleName` + `plutil -lint`. Prints the codesign / notarytool / stapler commands as TODOs. |
| `build_windows.ps1` | Same shape for Windows. DirectML EP by default; documents the CUDA/TensorRT variant and `signtool` signing. |
| `requirements/packaging.txt` | `pyinstaller`, `pyinstaller-hooks-contrib`. (Apple/Windows signing tools are not pip-installable.) |

## Why the macOS menu said "Python"

When AutoPTZ runs unbundled (`python -m autoptz`), AppKit has no bundle
`Info.plist` to read, so it falls back to the **executable's process name** —
`Python` — for the application menu and Dock label. `app.py` already sets
`QGuiApplication.setApplicationDisplayName("AutoPTZ")` and best-effort pokes
`NSBundle` via PyObjC, but neither overrides the menu reliably. The **definitive
fix** is shipping a real bundle: `AutoPTZ.app/Contents/Info.plist` with
`CFBundleName=AutoPTZ`. PyInstaller's `BUNDLE` step writes our `packaging/Info.plist`
into the app, so the built bundle is named correctly out of the box.

## Build — macOS (Apple Silicon)

```bash
bash packaging/build_macos.sh
# → dist/AutoPTZ.app  (unsigned; menu already reads "AutoPTZ")
```

Then sign + notarize **on your Mac** (needs your Developer ID + Team ID — find
them with `security find-identity -v -p codesigning`):

```bash
# 1. Sign with the hardened runtime + entitlements
codesign --deep --force --options runtime --timestamp \
  --entitlements packaging/entitlements.plist \
  --sign "Developer ID Application: <YOUR NAME> (<TEAM_ID>)" \
  dist/AutoPTZ.app
codesign --verify --deep --strict --verbose=2 dist/AutoPTZ.app

# 2. Notarize (store creds once in a keychain profile)
ditto -c -k --keepParent dist/AutoPTZ.app dist/AutoPTZ.zip
xcrun notarytool store-credentials AUTOPTZ_NOTARY \
  --apple-id "<you@example.com>" --team-id "<TEAM_ID>" \
  --password "<app-specific-password>"
xcrun notarytool submit dist/AutoPTZ.zip --keychain-profile AUTOPTZ_NOTARY --wait

# 3. Staple the ticket (validates offline)
xcrun stapler staple dist/AutoPTZ.app
```

Optional DMG: `hdiutil create -volname AutoPTZ -srcfolder dist/AutoPTZ.app -ov -format UDZO dist/AutoPTZ.dmg` (sign + notarize the `.dmg` the same way).

## Build — Windows (x64)

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
# → dist\AutoPTZ\AutoPTZ.exe   (onedir; add -OneFile for a single exe)
```

- **DirectML is the default EP** (`onnxruntime-directml`) — runs on any DX12 GPU
  with no CUDA install.
- **NVIDIA variant:** uninstall the DirectML/CPU onnxruntime and
  `pip install -r requirements\gpu-nvidia.txt` (onnxruntime-gpu → TensorRT/CUDA
  EPs), then rebuild. Needs CUDA 12.x + cuDNN 9.x (CUDA EP) and TensorRT 10.x.
- Sign with `signtool` (Windows SDK) + your code-signing cert; optionally wrap in
  an Inno Setup / WiX installer and sign that too.

## Models & NDI runtime

- **Detector model:** the YOLO11 ONNX **auto-downloads on first run** via
  `ModelManager`. To ship it inside the app (offline first run), pre-fetch it
  before building: `python -m tools.fetch_models --cache-dir autoptz/models`
  (the build scripts have a commented step for this). Face (InsightFace) and
  ReID (boxmot) weights auto-download into their own caches on first use.
- **NDI runtime:** `libndi` (macOS `.dylib` / Windows `Processing.NDI.Lib.x64.dll`)
  is a **system install, not a pip wheel**. To bundle it, drop the lib in
  `packaging/ndi/` (or set `NDI_RUNTIME=/path/to/dir`) before building; the spec
  adds it automatically. Without it, NDI ingest degrades gracefully.

## Icons (optional)

Drop `packaging/AutoPTZ.icns` (macOS) and/or `packaging/AutoPTZ.ico` (Windows)
beside the spec; both the EXE and the BUNDLE pick them up automatically. The
build succeeds without them (generic icon).

## What can't be done in this environment

- **No signing / notarization** — there are no Apple credentials here. The
  scripts print the exact commands instead of running them.
- **No full GUI build executed** — the assets are validated for correctness
  (spec parses, `Info.plist` / `entitlements.plist` pass `plutil -lint` and a
  `plistlib` round-trip, `import PyInstaller` works), but producing the final
  `.app` requires running PyInstaller on the target machine.

## CI (suggested)

A GitHub Actions matrix can produce both artifacts:

- `macos-14` (arm64): run `packaging/build_macos.sh` (skip signing unless the
  Developer ID cert + notary creds are provided as encrypted secrets), upload
  `dist/AutoPTZ.app` (zipped).
- `windows-latest` (x64): run `packaging/build_windows.ps1`, upload `dist/AutoPTZ`.
