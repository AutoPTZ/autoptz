"""Pure UI-factory helpers for the Properties panel.

Small widget factories + value formatters extracted from ``properties_panel`` so
the panel class focuses on wiring sections together. ``properties_panel``
re-exports these.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QWidget

from autoptz.ui import theme as T

if TYPE_CHECKING:
    from autoptz.ui.widgets.common import CostChip

log = logging.getLogger(__name__)


def _form() -> QFormLayout:
    f = QFormLayout()
    f.setContentsMargins(0, 0, 0, 0)
    f.setHorizontalSpacing(14)
    f.setVerticalSpacing(8)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return f


def _wrap(layout: QFormLayout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


def _with_chip(widget: QWidget, *trailing: QWidget) -> QWidget:
    """Pack ``widget`` (stretched) with one or more trailing badges/chips.

    Accepts any number of trailing widgets so a control can carry BOTH a cost
    chip and a "?" HelpBadge in the same row.
    """
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    row.addWidget(widget, 1)
    for w in trailing:
        row.addWidget(w, 0)
    return holder


def _ro_value() -> QLabel:
    lab = QLabel("—")
    lab.setStyleSheet(f"color: {T.CURRENT.text};")
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def _restyle_chip(chip: CostChip, cost: str) -> None:
    color = {"light": T.TRACKING, "medium": T.WARNING, "heavy": T.ERROR}.get(cost, T.TRACKING)
    chip.setText(cost.upper())
    chip.setStyleSheet(
        f"color: {color}; border: 1px solid {color}; border-radius: 7px;"
        f"padding: 1px 6px; font-size: 9px; font-weight: 700;"
    )


def _set_combo(combo: QComboBox, value: str) -> None:
    i = combo.findText(str(value))
    if i >= 0:
        combo.setCurrentIndex(i)


def _set_combo_data(combo: QComboBox, value: str) -> None:
    """Select the item whose userData == *value* (for caption≠value combos)."""
    i = combo.findData(str(value))
    if i >= 0:
        combo.setCurrentIndex(i)


def _short(addr: str) -> str:
    import re

    return re.sub(r"(\w+://)[^@/]*@", r"\1", str(addr or "—")) or "—"


def _signed_pct(value: int) -> str:
    return f"{value:+d}%"


def _source_supports_substream(src: dict[str, Any]) -> bool:
    """True only when config carries a concrete alternate stream reference."""
    if not isinstance(src, dict):
        return False
    for key in ("substream_url", "substream_address", "secondary_address", "lowres_address"):
        if src.get(key):
            return True
    profiles = src.get("profiles")
    if isinstance(profiles, list) and len(profiles) > 1:
        return True
    return False


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _connect(obj: Any, name: str, slot: Any) -> None:
    try:
        getattr(obj, name).connect(slot)
    except Exception:  # noqa: BLE001
        log.debug("connect %s failed", name, exc_info=True)
