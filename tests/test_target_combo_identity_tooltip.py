"""Regression: the 'Track person' combo has no elide+tooltip recovery path.

``PropertiesPanel._constrain_field_widths`` (properties_panel.py) forces every
``QComboBox``/``QLineEdit`` descendant to an ``Ignored`` horizontal size policy
plus a 48px floor, so the form never widens a narrow dock — including
``self._target_combo``, which is populated with user-entered identity names of
unbounded length (see ``_reload_targets``).

Unlike the panel's own ``_set_caption`` helper (elides a QLabel to its current
width and always mirrors the FULL text on the tooltip — used for the Address
and tracking-state captions), a ``QComboBox`` has no built-in elide-on-paint
for its closed-box text and no tooltip that tracks the current selection. At a
panel resized to its own floor width, a long identity name can render
hard-clipped with nothing to recover the full value.

``tests/test_panel_min_width_no_clip.py``'s ``_overflows()`` helper only
inspects ``QLabel | QCheckBox | QPushButton`` widgets, so it never catches
this combo-box gap.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


LONG_IDENTITY_NAME = "Bartholomew Alexander Montgomery-Fitzgerald the Third of Whitmoreshire"


def _expand_all_sections(panel, app) -> None:
    """Force-expand every CollapsibleGroup, skipping the animation.

    ``_target_combo`` lives in the "Tracking" section, which is collapsed by
    default (``expanded=False``) — expand it so the combo has real, laid-out
    geometry to measure (same helper as tests/test_panel_min_width_no_clip.py).
    """
    from autoptz.ui.widgets.common import CollapsibleGroup

    for g in panel.findChildren(CollapsibleGroup):
        g._toggle.setChecked(True)
        g._on_toggle(True)
        g._height_anim.stop()
        g._content.setMaximumHeight(16777215)
        g._content.setVisible(True)
        g._expanded = True
    app.processEvents()
    app.processEvents()


def _panel_targeting(name: str):
    """Build a panel whose one camera's tracking target is a labeled identity
    named *name*, registered BEFORE ``set_camera`` so ``_reload_targets`` picks
    it up and auto-selects it (mirrors ``_load`` reading ``target.identity_id``)."""
    from autoptz.config.models import IdentityRecord
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.properties_panel import PropertiesPanel

    client = EngineClient()
    cid = client.addCamera("usb://0", "Cam")
    ident = IdentityRecord(name=name)
    client.push_identity(ident)
    client.setTargetIdentity(cid, ident.id)
    panel = PropertiesPanel(client, frame_source=ShmFrameSource())
    return client, cid, panel


def test_target_combo_tooltip_mirrors_full_identity_name_at_floor_width(qtapp) -> None:
    """Repro: long identity name + panel shrunk to its own floor width.

    The combo may still hard-clip its VISIBLE text (Qt gives QComboBox no
    built-in elide-on-paint for the closed box), but the full untruncated name
    must be one hover away — the tooltip must mirror it, exactly like
    ``_set_caption``'s label contract elsewhere in this panel.
    """
    _client, cid, panel = _panel_targeting(LONG_IDENTITY_NAME)
    try:
        width = max(panel.minimumWidth(), panel.minimumSizeHint().width())
        panel.resize(width, 820)
        panel.show()
        qtapp.processEvents()
        panel.set_camera(cid)
        _expand_all_sections(panel, qtapp)

        combo = panel._target_combo
        assert combo.currentText() == LONG_IDENTITY_NAME  # sanity: selection loaded

        # Sanity: this repro only proves something if the combo really is too
        # narrow to fit the whole name at the panel's own floor width (the
        # reported scenario) — _constrain_field_widths forces Ignored + 48px.
        avail = combo.width()
        needed = combo.fontMetrics().horizontalAdvance(LONG_IDENTITY_NAME)
        assert avail < needed, (
            f"test setup didn't reproduce overflow at floor width: avail={avail} needed={needed}"
        )

        assert combo.toolTip() == LONG_IDENTITY_NAME, (
            "long identity name is not recoverable when the combo can't show it "
            f"in full: expected the tooltip to mirror the full name, got {combo.toolTip()!r}"
        )
    finally:
        panel.deleteLater()


def test_target_combo_tooltip_keeps_help_text_for_anyone(qtapp) -> None:
    """No identity is selected ('— Anyone —') -> keep the descriptive help
    tooltip instead of mirroring a (non-existent) identity name."""
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.properties_panel import PropertiesPanel

    client = EngineClient()
    cid = client.addCamera("usb://0", "Cam")
    panel = PropertiesPanel(client, frame_source=ShmFrameSource())
    try:
        panel.set_camera(cid)
        qtapp.processEvents()
        combo = panel._target_combo
        assert combo.currentText() == "— Anyone —"
        assert "registered person" in combo.toolTip()
    finally:
        panel.deleteLater()
