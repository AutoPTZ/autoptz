"""ModelManagerDialog face-pack row (offscreen).

The face pack is a first-class row (Download / Remove) driven by ModelManager's
app-data cache; ReID stays a read-only upstream-managed row.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


class _FakeMgr:
    def __init__(self, face_status: dict[str, Any]) -> None:
        self._face_status = face_status
        self.calls: list[str] = []

    def app_model_statuses(self) -> list[dict[str, Any]]:
        return []

    def face_pack_status(self) -> dict[str, Any]:
        return dict(self._face_status)

    def ensure_face_pack(self) -> list[dict[str, str]]:
        self.calls.append("ensure_face_pack")
        return [
            {"name": "Face pack", "state": "downloaded", "path": "/x", "size": "1", "error": ""}
        ]

    def remove_face_pack(self) -> list[dict[str, str]]:
        self.calls.append("remove_face_pack")
        return [
            {"name": "w.onnx", "state": "removed", "path": "/x/w.onnx", "size": "1", "error": ""}
        ]


def _client() -> SimpleNamespace:
    store: dict[str, Any] = {}
    return SimpleNamespace(
        getDetectorModelTier=lambda: "auto",
        autoDownloadModels=lambda: False,
        optionalComponents=lambda: [
            {
                "key": "reid",
                "name": "ReID (re-acquire)",
                "state": "off",
                "detail": "OSNet",
                "managed": "upstream",
                "why": "stable tracking",
                "path": "/reid",
            },
            {
                "key": "face",
                "name": "Face recognition",
                "state": "warn",
                "detail": "insightface",
                "managed": "downloadable",
                "why": "identity",
                "path": "/face",
            },
        ],
        getSetting=lambda k, d=None: store.get(k, d),
        setSetting=lambda k, v: store.__setitem__(k, v),
        releaseModelSessions=lambda include_face=False: store.__setitem__(
            "rel", store.get("rel", 0) + 1
        )
        or store.__setitem__("rel_face", include_face),
        rebuildModelSessions=lambda include_face=False: store.__setitem__(
            "reb", store.get("reb", 0) + 1
        )
        or store.__setitem__("reb_face", include_face),
        _store=store,
    )


def _dialog(qtapp, monkeypatch, face_status: dict[str, Any]):
    import autoptz.engine.runtime.models as models
    from autoptz.ui.widgets.dialogs.model_manager import ModelManagerDialog

    mgr = _FakeMgr(face_status)
    monkeypatch.setattr(models, "default_manager", lambda: mgr)
    dlg = ModelManagerDialog(_client())
    return dlg, mgr


_MISSING = {
    "model": "buffalo_l",
    "location": "missing",
    "path": "/c",
    "present": False,
    "removable": False,
    "size_bytes": 0,
}
_APPDATA = {
    "model": "buffalo_l",
    "location": "app-data",
    "path": "/c",
    "present": True,
    "removable": True,
    "size_bytes": 300_000_000,
}
_BUNDLED = {
    "model": "buffalo_l",
    "location": "bundled",
    "path": "/b",
    "present": True,
    "removable": False,
    "size_bytes": 300_000_000,
}
_HOME = {
    "model": "buffalo_l",
    "location": "home",
    "path": "/h",
    "present": True,
    "removable": True,
    "size_bytes": 300_000_000,
}


def test_face_row_download_enabled_when_missing(qtapp, monkeypatch) -> None:
    dlg, _ = _dialog(qtapp, monkeypatch, _MISSING)
    try:
        assert dlg._face_row is not None
        assert dlg._face_row.download_btn.isEnabled() is True
        assert dlg._face_row.remove_btn.isEnabled() is False
    finally:
        dlg.close()


def test_face_row_remove_enabled_only_for_appdata(qtapp, monkeypatch) -> None:
    dlg, _ = _dialog(qtapp, monkeypatch, _APPDATA)
    try:
        assert dlg._face_row.remove_btn.isEnabled() is True
        assert dlg._face_row.download_btn.isEnabled() is False  # already present
    finally:
        dlg.close()


def test_face_row_bundled_not_removable(qtapp, monkeypatch) -> None:
    dlg, _ = _dialog(qtapp, monkeypatch, _BUNDLED)
    try:
        assert dlg._face_row.removable is False
        assert dlg._face_row.remove_btn.isEnabled() is False
        assert dlg._face_row.download_btn.isEnabled() is False
    finally:
        dlg.close()


def test_face_row_home_is_removable(qtapp, monkeypatch) -> None:
    """A pack in the user's own ~/.insightface can be removed."""
    dlg, _ = _dialog(qtapp, monkeypatch, _HOME)
    try:
        assert dlg._face_row.removable is True
        assert dlg._face_row.remove_btn.isEnabled() is True
    finally:
        dlg.close()


def test_face_not_in_upstream_external_rows(qtapp, monkeypatch) -> None:
    """The upstream-managed section shows only ReID — face has its own row now."""
    from autoptz.ui.widgets.dialogs.model_manager import _ExternalRow

    dlg, _ = _dialog(qtapp, monkeypatch, _MISSING)
    try:
        external = dlg.findChildren(_ExternalRow)
        texts = [w.findChild(type(w)) for w in external]  # touch to ensure built
        assert texts is not None
        # No _ExternalRow should be a face row; ReID is the only upstream row.
        labels = " ".join(
            lbl.text() for row in external for lbl in row.findChildren(type(dlg._status))
        )
        assert "Face recognition" not in labels
        assert "ReID" in labels
    finally:
        dlg.close()


def test_face_download_runs_manager_and_brackets_sessions(qtapp, monkeypatch) -> None:
    """Download drives the background task, which releases + rebuilds ORT sessions
    around the manager call (the Windows file-lock bracket)."""
    from autoptz.ui.widgets.dialogs.model_manager import _ModelTask

    dlg, mgr = _dialog(qtapp, monkeypatch, _MISSING)
    try:
        # Drive the task body synchronously (no thread) for a deterministic assert.
        task = _ModelTask("download_face", [], client=dlg._client)
        task.run()
        assert "ensure_face_pack" in mgr.calls
        assert dlg._client._store.get("rel") == 1  # released before
        assert dlg._client._store.get("reb") == 1  # rebuilt after
        # A face op releases/rebuilds the shared face session too.
        assert dlg._client._store.get("rel_face") is True
        assert dlg._client._store.get("reb_face") is True
    finally:
        dlg.close()


def test_detector_op_does_not_touch_face_session(qtapp, monkeypatch) -> None:
    """A detector download/remove must pass include_face=False so it never drops
    and reloads the ~1.3 GB face pack."""
    from autoptz.ui.widgets.dialogs.model_manager import _ModelTask

    dlg, mgr = _dialog(qtapp, monkeypatch, _MISSING)
    try:
        # Give the fake manager a remove_app_models so the detector 'remove' works.
        mgr.remove_app_models = lambda *, keys=None: [
            {"name": "m", "state": "removed", "path": "", "size": "0", "error": ""}
        ]
        _ModelTask("remove", ["detector_fast"], client=dlg._client).run()
        assert dlg._client._store.get("rel_face") is False
        assert dlg._client._store.get("reb_face") is False
    finally:
        dlg.close()
