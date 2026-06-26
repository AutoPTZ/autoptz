"""CameraWall / CameraTile context-menu gating (offscreen).

The normal app keeps the right-click tile context menu; AutoPTZ Mark disables it
(``context_menu_enabled=False``) so a demo viewer can't remove cameras or change
targets via the menu.  These tests construct the widgets offscreen and assert the
flag both threads through the wall to its tiles AND suppresses the menu.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


def _client_with_camera(qtapp):
    from autoptz.benchmark.runner import _add_synthetic_camera
    from autoptz.ui.engine_client import EngineClient

    client = EngineClient()
    cid = _add_synthetic_camera(client, 0)
    return client, cid


def _tile(qtapp, *, context_menu_enabled: bool):
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.camera_tile import CameraTile

    client, cid = _client_with_camera(qtapp)
    return CameraTile(cid, client, ShmFrameSource(), context_menu_enabled=context_menu_enabled)


def test_tile_context_menu_enabled_by_default(qtapp) -> None:
    tile = _tile(qtapp, context_menu_enabled=True)
    assert tile._context_menu_enabled is True
    tile.deleteLater()


def test_tile_context_menu_disabled_suppresses_menu(qtapp, monkeypatch) -> None:
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QMenu

    tile = _tile(qtapp, context_menu_enabled=False)
    assert tile._context_menu_enabled is False

    # A disabled menu must never be exec'd — patch QMenu.exec to record any attempt.
    execs: list[object] = []
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: execs.append(self))

    from PySide6.QtCore import QPoint

    event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(5, 5), QPoint(5, 5))
    tile.contextMenuEvent(event)
    assert execs == []  # no menu shown when disabled
    tile.deleteLater()


def test_wall_threads_flag_to_tiles(qtapp) -> None:
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.camera_wall import CameraWall

    client, _cid = _client_with_camera(qtapp)
    wall = CameraWall(client, ShmFrameSource(), context_menu_enabled=False)
    assert wall._context_menu_enabled is False
    # Every constructed tile inherits the wall's flag.
    assert wall._tiles
    assert all(t._context_menu_enabled is False for t in wall._tiles.values())
    wall.deleteLater()


def test_wall_context_menu_enabled_by_default(qtapp) -> None:
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.camera_wall import CameraWall

    client, _cid = _client_with_camera(qtapp)
    wall = CameraWall(client, ShmFrameSource())
    assert wall._context_menu_enabled is True
    assert all(t._context_menu_enabled is True for t in wall._tiles.values())
    wall.deleteLater()
