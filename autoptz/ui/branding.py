"""Shared app branding: resolve the logo asset and build the window QIcon.

Works in source/editable runs and in frozen PyInstaller bundles (where data files
are extracted under ``sys._MEIPASS``).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtGui import QIcon, QPixmap

LOGO_FILENAME = "AutoPTZLogo.png"


def logo_path() -> Path:
    """Absolute path to the master logo PNG (source and frozen runs)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "autoptz" / "assets" / LOGO_FILENAME
        if bundled.is_file():
            return bundled
    # Source / editable: this module is autoptz/ui/branding.py → ../assets/<file>.
    return Path(__file__).resolve().parent.parent / "assets" / LOGO_FILENAME


@lru_cache(maxsize=1)
def app_icon() -> QIcon:
    """Return the application QIcon (empty icon if the asset is missing)."""
    from PySide6.QtGui import QIcon

    path = logo_path()
    return QIcon(str(path)) if path.is_file() else QIcon()


def logo_pixmap(size: int) -> QPixmap:
    """Return the logo scaled to *size*×*size* (smooth), or an empty pixmap."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap

    path = logo_path()
    if not path.is_file():
        return QPixmap()
    pm = QPixmap(str(path))
    return pm.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
