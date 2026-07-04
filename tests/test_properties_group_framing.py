"""Per-camera Group framing checkbox (Properties → PTZ), offscreen.

Group framing used to be a dead "New-camera tracking defaults" entry in the
Experimental dialog (nothing consumed it for a new camera). It is a live
TrackingConfig field, so it belongs on the per-camera panel next to Center Stage,
where the debounced write-back applies it to the running camera.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _panel_with_camera(tmp_path: Path):
    from autoptz.config.store import ConfigStore
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.properties_panel import PropertiesPanel

    client = EngineClient(store=ConfigStore(db_path=tmp_path / "cfg.db", debounce_s=0))
    cid = client.addCamera("usb://0", "Cam")
    panel = PropertiesPanel(client, frame_source=ShmFrameSource())
    return client, cid, panel


def test_checkbox_defaults_unchecked_for_new_camera(qtapp, tmp_path) -> None:
    client, cid, panel = _panel_with_camera(tmp_path)
    try:
        panel.set_camera(cid)
        assert panel._group_framing.isChecked() is False
    finally:
        panel.deleteLater()


def test_checkbox_loads_saved_value(qtapp, tmp_path) -> None:
    client, cid, panel = _panel_with_camera(tmp_path)
    try:
        cfg = client.getCameraConfig(cid)
        cfg["tracking"]["group_framing"] = True
        client.updateCameraConfig(cid, json.dumps(cfg))
        panel.set_camera(cid)
        assert panel._group_framing.isChecked() is True
    finally:
        panel.deleteLater()


def test_on_config_changed_does_not_raise(qtapp, tmp_path) -> None:
    """Regression: _on_config_changed called a removed _sync_framing_sliders()
    method, so any external config change on the selected camera (e.g. a toggle
    round-tripping through the client's configChanged signal) raised
    AttributeError. It must complete cleanly."""
    client, cid, panel = _panel_with_camera(tmp_path)
    try:
        panel.set_camera(cid)
        panel._on_config_changed(cid)  # must not raise
    finally:
        panel.deleteLater()


def test_toggle_persists_via_push(qtapp, tmp_path) -> None:
    client, cid, panel = _panel_with_camera(tmp_path)
    try:
        panel.set_camera(cid)
        framing_before = client.getCameraConfig(cid)["tracking"].get("framing")
        panel._group_framing.setChecked(True)
        panel._push()
        tracking = client.getCameraConfig(cid)["tracking"]
        assert tracking["group_framing"] is True
        # Unrelated tracking fields are untouched by the toggle.
        assert tracking.get("framing") == framing_before
    finally:
        panel.deleteLater()


def test_uncheck_persists_false(qtapp, tmp_path) -> None:
    client, cid, panel = _panel_with_camera(tmp_path)
    try:
        cfg = client.getCameraConfig(cid)
        cfg["tracking"]["group_framing"] = True
        client.updateCameraConfig(cid, json.dumps(cfg))
        panel.set_camera(cid)
        panel._group_framing.setChecked(False)
        panel._push()
        assert client.getCameraConfig(cid)["tracking"]["group_framing"] is False
    finally:
        panel.deleteLater()
