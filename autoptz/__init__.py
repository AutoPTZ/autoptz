"""AutoPTZ — AI-driven PTZ camera tracking."""

from __future__ import annotations

#: Canonical version string. This is the single source of truth; ``pyproject.toml``
#: reads it via ``[tool.setuptools.dynamic]`` and the UI imports it at runtime.
__version__ = "2.1.0-rc2"


def version() -> str:
    """Return the running app version.

    :data:`__version__` is the single source of truth — ``pyproject.toml`` derives
    the packaged version from it, so this value is always correct for source,
    editable, and frozen builds alike.
    """
    return __version__
