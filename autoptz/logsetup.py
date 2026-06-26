"""Console logging setup with ANSI colour (level + stable per-camera tint).

Pure stdlib (no Qt / no third-party dep) so it can be installed from
``python -m autoptz`` before the UI or any heavy import.  Colour is emitted only
when the stream is a TTY and the environment doesn't opt out (``NO_COLOR`` /
``TERM=dumb`` / ``AUTOPTZ_NO_COLOR``), so piped/redirected logs and CI stay clean.

Two colour dimensions:

* **Level** — DEBUG dim, INFO green, WARNING yellow, ERROR/CRITICAL bold red.
* **Camera** — log lines carry ``camera_id=<uuid>``; each camera gets a stable
  colour (hash → 256-colour palette) so multi-camera output is scannable at a
  glance.  The ``camera_id=…`` token is recoloured in place.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import TextIO

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

# Per-level SGR colour (foreground).
_LEVEL_SGR: dict[int, str] = {
    logging.DEBUG: _DIM + "\033[37m",  # dim grey
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: _BOLD + "\033[31m",  # bold red
    logging.CRITICAL: _BOLD + "\033[97;41m",  # bold white on red
}

# Distinct 256-colour foreground codes for per-camera tint (avoids near-black /
# near-white so it reads on both dark and light terminals).
_CAMERA_PALETTE: tuple[int, ...] = (
    39,
    208,
    213,
    154,
    220,
    45,
    201,
    118,
    214,
    51,
    199,
    190,
    81,
    165,
    226,
    123,
)

_CAMERA_RE = re.compile(r"(camera_id=)([0-9a-fA-F-]{6,})")


def _supports_color(stream: TextIO) -> bool:
    if os.environ.get("AUTOPTZ_NO_COLOR") or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("AUTOPTZ_FORCE_COLOR"):
        return True
    if (os.environ.get("TERM") or "").lower() == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001 — odd stream → assume no colour
        return False


def camera_ansi(camera_id: str) -> str:
    """Return the stable 256-colour ANSI prefix for *camera_id* (empty-safe)."""
    if not camera_id:
        return ""
    code = _CAMERA_PALETTE[hash(camera_id) % len(_CAMERA_PALETTE)]
    return f"\033[38;5;{code}m"


class ColorFormatter(logging.Formatter):
    """Formatter that wraps the level name and any ``camera_id=`` token in colour."""

    def __init__(self, fmt: str, *, use_color: bool) -> None:
        super().__init__(fmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_color:
            return super().format(record)
        # Colour the level name in place so the base format string is untouched.
        level_sgr = _LEVEL_SGR.get(record.levelno, "")
        orig_levelname = record.levelname
        if level_sgr:
            record.levelname = f"{level_sgr}{orig_levelname}{_RESET}"
        try:
            line = super().format(record)
        finally:
            record.levelname = orig_levelname
        # Recolour every camera_id token (the message may name several).
        line = _CAMERA_RE.sub(
            lambda m: f"{m.group(1)}{camera_ansi(m.group(2))}{m.group(2)}{_RESET}",
            line,
        )
        return line


def install_console_logging(
    level: int = logging.WARNING,
    *,
    stream: TextIO | None = None,
    fmt: str = "%(levelname)s  %(name)s  %(message)s",
) -> None:
    """Install a single coloured stderr handler on the root logger.

    Idempotent-ish: replaces any handler previously installed by this function so
    re-calling (e.g. after ``--log-level``) doesn't stack handlers.  Safe to call
    before the UI is imported.
    """
    stream = stream or sys.stderr
    root = logging.getLogger()
    # Drop a prior handler we installed so level changes don't duplicate output.
    for h in list(root.handlers):
        if getattr(h, "_autoptz_console", False):
            root.removeHandler(h)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ColorFormatter(fmt, use_color=_supports_color(stream)))
    handler._autoptz_console = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
