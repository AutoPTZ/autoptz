"""Per-camera output toggles (Properties → PTZ): NDI output + virtual camera."""

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


def test_ndi_and_vcam_toggles_default_off(qtapp, tmp_path) -> None:
    client, cid, panel = _panel_with_camera(tmp_path)
    try:
        panel.set_camera(cid)
        assert panel._ndi_out.isChecked() is False
        assert panel._vcam_out.isChecked() is False
    finally:
        panel.deleteLater()


def test_ndi_out_persists_via_push(qtapp, tmp_path) -> None:
    client, cid, panel = _panel_with_camera(tmp_path)
    try:
        panel.set_camera(cid)
        panel._ndi_out.setChecked(True)
        panel._push()
        ptz = client.getCameraConfig(cid)["ptz"]
        assert ptz["ndi_out"] is True
        assert ptz["vcam_out"] is False  # independent of NDI
    finally:
        panel.deleteLater()


def test_ndi_out_loads_saved_value(qtapp, tmp_path) -> None:
    client, cid, panel = _panel_with_camera(tmp_path)
    try:
        cfg = client.getCameraConfig(cid)
        cfg["ptz"]["ndi_out"] = True
        client.updateCameraConfig(cid, json.dumps(cfg))
        panel.set_camera(cid)
        assert panel._ndi_out.isChecked() is True
    finally:
        panel.deleteLater()
