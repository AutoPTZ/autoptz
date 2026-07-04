"""INVARIANT: at a panel's own minimum width, nothing may overflow horizontally.

"I never want horizontal scroll for a min width" — every side panel and dialog,
with every collapsible section EXPANDED and worst-case dynamic content injected
(a long NDI address, the "source isn't reaching" fps caption, a degradation
reason), must lay out with zero widgets crossing the scroll viewport's right
edge and zero hard-clipped (non-wrapping, non-elided) label text.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


LONG_ADDR = "ndi://PRINCES-MBP (AutoPTZ PRINCES-MBP (AutoPTZ MacBook Pro Camera Long))"
LONG_NAME = "PRINCES-MBP (AutoPTZ Very Long Loopback Camera Name)"


def _expand_all_sections(panel, app) -> None:
    """Force-expand every CollapsibleGroup, skipping the animation."""
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


def _overflows(scroll, app) -> list[str]:
    """Every visible leaf widget crossing the viewport's right edge, and every
    hard-clipped label (text wider than widget, no wrap, no ellipsis)."""
    from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton, QWidget

    app.processEvents()
    vp = scroll.viewport()
    vp_right = vp.mapToGlobal(vp.rect().topRight()).x()
    problems: list[str] = []
    for w in vp.findChildren(QWidget):
        if not w.isVisible() or w.width() <= 0:
            continue
        over = w.mapToGlobal(w.rect().topRight()).x() - vp_right
        if over > 1 and not w.findChildren(QWidget):
            problems.append(f"+{over}px {type(w).__name__}({w.objectName() or ''})")
        if isinstance(w, QLabel | QCheckBox | QPushButton):
            import re

            raw = str(w.text())
            txt = re.sub(r"<[^>]+>", "", raw)  # metrics on VISIBLE text, not HTML tags
            wraps = bool(getattr(w, "wordWrap", lambda: False)())
            if txt and "…" not in txt and not wraps:
                need = w.fontMetrics().horizontalAdvance(txt)
                # Small tolerance for rich-text labels: plain metrics slightly
                # misestimate bold rendering.
                slack = 8 if raw != txt else 2
                if need > w.width() + slack:
                    problems.append(
                        f"CLIPTEXT {type(w).__name__}({txt[:30]!r}) needs {need} has {w.width()}"
                    )
    return problems


def test_properties_panel_min_width_never_clips(qtapp) -> None:
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.properties_panel import PropertiesPanel

    client = EngineClient()
    cid = client.addCamera(LONG_ADDR, LONG_NAME)
    panel = PropertiesPanel(client, frame_source=ShmFrameSource())
    try:
        width = max(panel.minimumWidth(), panel.minimumSizeHint().width())
        panel.resize(width, 820)
        panel.show()
        qtapp.processEvents()
        panel.set_camera(cid)
        _expand_all_sections(panel, qtapp)
        # Worst-case dynamic texts (the ones that appear under load).
        panel._fps_measured.setText("measured: 10.2 fps — source isn't reaching 30 fps")
        panel._set_caption(
            panel._track_reason,
            "Degraded ×4: Auto quality: over frame budget; detector cadence relaxed",
        )
        panel._set_caption(panel._track_state, "State: Standing by for reacquire (standby)")
        qtapp.processEvents()
        qtapp.processEvents()  # deferred re-elide
        problems = _overflows(panel._scroll, qtapp)
        assert not problems, f"overflow at min width {width}: " + "; ".join(problems[:8])
    finally:
        panel.deleteLater()


def test_services_panel_min_width_never_clips(qtapp) -> None:
    from PySide6.QtWidgets import QScrollArea

    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.widgets.services_panel import ServicesPanel

    panel = ServicesPanel(EngineClient())
    try:
        width = max(panel.minimumWidth(), panel.minimumSizeHint().width())
        panel.resize(width, 820)
        panel.show()
        qtapp.processEvents()
        scroll = panel.findChildren(QScrollArea)[0]
        problems = _overflows(scroll, qtapp)
        assert not problems, f"overflow at min width {width}: " + "; ".join(problems[:8])
    finally:
        panel.deleteLater()


def test_camera_info_panel_min_width_never_clips(qtapp) -> None:
    from PySide6.QtWidgets import QScrollArea

    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.widgets.camera_info_panel import CameraInfoPanel

    client = EngineClient()
    cid = client.addCamera(LONG_ADDR, LONG_NAME)
    panel = CameraInfoPanel(client)
    try:
        width = max(280, panel.minimumWidth(), panel.minimumSizeHint().width())
        panel.resize(width, 820)
        panel.show()
        qtapp.processEvents()
        panel.set_camera(cid)
        qtapp.processEvents()
        scrolls = panel.findChildren(QScrollArea)
        if scrolls:
            problems = _overflows(scrolls[0], qtapp)
            assert not problems, f"overflow at min width {width}: " + "; ".join(problems[:8])
    finally:
        panel.deleteLater()


def test_experimental_dialog_min_width_never_clips(qtapp) -> None:
    from types import SimpleNamespace

    from PySide6.QtWidgets import QScrollArea

    from autoptz.ui.widgets.dialogs.experimental import ExperimentalFeaturesDialog

    store: dict = {}
    client = SimpleNamespace(
        getSetting=lambda k, d=None: store.get(k, d),
        setSetting=lambda k, v: store.__setitem__(k, v),
    )
    dlg = ExperimentalFeaturesDialog(client)
    try:
        dlg.resize(dlg.minimumWidth(), 700)
        dlg.show()
        qtapp.processEvents()
        scroll = dlg.findChildren(QScrollArea)[0]
        problems = _overflows(scroll, qtapp)
        assert not problems, "; ".join(problems[:8])
    finally:
        dlg.close()
