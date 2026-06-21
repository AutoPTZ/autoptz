"""Native Qt Widgets UI for AutoPTZ.

The shell is a :class:`~autoptz.ui.widgets.main_window.MainWindow` (``QMainWindow``)
with dockable panels around a central camera wall.  All panels bind to the same
framework-agnostic :class:`~autoptz.ui.engine_client.EngineClient`.
"""

from __future__ import annotations

from autoptz.ui.widgets.main_window import MainWindow

__all__ = ["MainWindow"]
