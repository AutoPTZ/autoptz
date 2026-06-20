"""Generate platform icon files from the master logo.

Run from the repo root (idempotent; regenerate whenever the logo changes)::

    python packaging/make_icons.py

Inputs:
    autoptz/assets/AutoPTZLogo.png   (master, square RGBA)

Outputs (committed, consumed by the PyInstaller spec + installers):
    packaging/AutoPTZ.ico            Windows (multi-size)
    packaging/AutoPTZ.icns           macOS
    packaging/AutoPTZ-256.png        Linux (.desktop / AppImage)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "autoptz" / "assets" / "AutoPTZLogo.png"
OUT = ROOT / "packaging"

ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
ICNS_SIZES = [16, 32, 64, 128, 256, 512, 1024]


def _load_square() -> Image.Image:
    im = Image.open(SRC).convert("RGBA")
    if im.width != im.height:
        side = max(im.size)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        im = canvas
    return im


def make_ico(im: Image.Image) -> None:
    dst = OUT / "AutoPTZ.ico"
    im.save(dst, format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    print(f"wrote {dst.relative_to(ROOT)}")


def make_png(im: Image.Image) -> None:
    dst = OUT / "AutoPTZ-256.png"
    im.resize((256, 256), Image.LANCZOS).save(dst, format="PNG")
    print(f"wrote {dst.relative_to(ROOT)}")


def make_icns(im: Image.Image) -> None:
    dst = OUT / "AutoPTZ.icns"
    # Prefer macOS `iconutil` (reference-quality); fall back to Pillow elsewhere.
    if sys.platform == "darwin":
        with tempfile.TemporaryDirectory() as td:
            iconset = Path(td) / "AutoPTZ.iconset"
            iconset.mkdir()
            for s in (16, 32, 128, 256, 512):
                im.resize((s, s), Image.LANCZOS).save(iconset / f"icon_{s}x{s}.png")
                im.resize((s * 2, s * 2), Image.LANCZOS).save(iconset / f"icon_{s}x{s}@2x.png")
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dst)], check=True)
        print(f"wrote {dst.relative_to(ROOT)} (iconutil)")
        return
    try:
        im.resize((1024, 1024), Image.LANCZOS).save(
            dst, format="ICNS", sizes=[(s, s) for s in ICNS_SIZES]
        )
        print(f"wrote {dst.relative_to(ROOT)} (Pillow)")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not write {dst.name} on this platform: {exc}")


def main() -> int:
    if not SRC.is_file():
        print(f"ERROR: master logo missing at {SRC}", file=sys.stderr)
        return 1
    im = _load_square()
    make_ico(im)
    make_png(im)
    make_icns(im)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
