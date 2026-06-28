"""Theme stylesheet (pure): the built stylesheet carries the Mark object-name rules.

The Mark HUD widgets (chart card, control panel, details header) read their
surface/border/text from the GLOBAL stylesheet via object-name selectors so they
track light/dark for free; these are pure string asserts on the built sheet.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

from autoptz.ui.theme import DARK, build_stylesheet


def _sheet() -> str:
    accent = QColor("#2563eb")
    return build_stylesheet(DARK, accent, accent)


def test_stylesheet_has_mark_chart_rule() -> None:
    assert "markChart" in _sheet()


def test_stylesheet_has_mark_control_panel_rule() -> None:
    assert "markControlPanel" in _sheet()


def test_stylesheet_has_details_header_rule() -> None:
    assert "detailsHeader" in _sheet()
