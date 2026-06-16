"""EngineClient: connects the QML UI to the Engine over typed command/telemetry channels.

In-process implementation: commands are queued in a deque so the engine supervisor
(Phase 6) can drain them via ``drain_commands()``.  The same interface is designed
to be swappable for a WebSocket transport in a later phase without touching QML.

Thread-safety: ``push_telemetry()`` may be called from any thread (engine thread);
all other methods must be called from the Qt main thread.
"""
from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import (
    Property,
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
    Signal,
    Slot,
)

from autoptz.engine.runtime.messages import (
    AddCameraCmd,
    BaseCommand,
    EnableTrackingCmd,
    PtzGoToPresetCmd,
    PtzNudgeCmd,
    RemoveCameraCmd,
    SetTargetCmd,
    TelemetryMsg,
)

# ── per-camera state ──────────────────────────────────────────────────────────


@dataclass
class CameraRecord:
    """Mutable per-camera state owned by CameraListModel."""

    camera_id: str
    source_uri: str
    display_name: str
    tracking_enabled: bool = False
    target_track_id: int | None = None
    telemetry: TelemetryMsg | None = None
    # SHM name written by the engine worker; derived from camera_id by default
    shm_name: str = field(init=False)
    shm_width: int = 1280
    shm_height: int = 720

    def __post_init__(self) -> None:
        self.shm_name = f"cam_{self.camera_id[:8]}_preview"

    @property
    def fps(self) -> float:
        return self.telemetry.fps if self.telemetry else 0.0

    @property
    def health(self) -> str:
        return self.telemetry.health.state.value if self.telemetry else "ok"

    def tracks_as_list(self) -> list[dict[str, Any]]:
        """Return tracks as plain dicts suitable for QML Repeater models."""
        if not self.telemetry:
            return []
        result = []
        for t in self.telemetry.tracks:
            result.append({
                "track_id": t.track_id,
                "bbox": {
                    "x1": t.bbox.x1,
                    "y1": t.bbox.y1,
                    "x2": t.bbox.x2,
                    "y2": t.bbox.y2,
                },
                "identity": t.identity or "",
                "confidence": t.confidence,
                "is_target": t.is_target,
            })
        return result

    def ptz_as_dict(self) -> dict[str, Any]:
        if not self.telemetry:
            return {"pan": 0.0, "tilt": 0.0, "zoom": 0.0, "moving": False, "state": "idle"}
        p = self.telemetry.ptz
        return {"pan": p.pan, "tilt": p.tilt, "zoom": p.zoom, "moving": p.moving, "state": p.state}


# ── Qt list model ─────────────────────────────────────────────────────────────


class CameraListModel(QAbstractListModel):
    """Ordered list of CameraRecords exposed to QML as a list model.

    Roles are accessible from QML delegate expressions as ``model.cameraId``, etc.
    """

    CameraIdRole = Qt.ItemDataRole.UserRole + 1
    DisplayNameRole = Qt.ItemDataRole.UserRole + 2
    TrackingEnabledRole = Qt.ItemDataRole.UserRole + 3
    TargetTrackIdRole = Qt.ItemDataRole.UserRole + 4
    FpsRole = Qt.ItemDataRole.UserRole + 5
    TracksRole = Qt.ItemDataRole.UserRole + 6
    PtzStateRole = Qt.ItemDataRole.UserRole + 7
    HealthRole = Qt.ItemDataRole.UserRole + 8
    ShmNameRole = Qt.ItemDataRole.UserRole + 9
    ShmWidthRole = Qt.ItemDataRole.UserRole + 10
    ShmHeightRole = Qt.ItemDataRole.UserRole + 11

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._order: list[str] = []         # camera_id insertion order
        self._records: dict[str, CameraRecord] = {}

    # ── QAbstractListModel interface ──────────────────────────────────────────

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self._order)

    def roleNames(self) -> dict[int, QByteArray]:
        return {
            self.CameraIdRole: QByteArray(b"cameraId"),
            self.DisplayNameRole: QByteArray(b"displayName"),
            self.TrackingEnabledRole: QByteArray(b"trackingEnabled"),
            self.TargetTrackIdRole: QByteArray(b"targetTrackId"),
            self.FpsRole: QByteArray(b"fps"),
            self.TracksRole: QByteArray(b"tracks"),
            self.PtzStateRole: QByteArray(b"ptzState"),
            self.HealthRole: QByteArray(b"health"),
            self.ShmNameRole: QByteArray(b"shmName"),
            self.ShmWidthRole: QByteArray(b"shmWidth"),
            self.ShmHeightRole: QByteArray(b"shmHeight"),
        }

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._order):
            return None
        rec = self._records.get(self._order[index.row()])
        if rec is None:
            return None
        match role:
            case self.CameraIdRole:
                return rec.camera_id
            case self.DisplayNameRole:
                return rec.display_name
            case self.TrackingEnabledRole:
                return rec.tracking_enabled
            case self.TargetTrackIdRole:
                return rec.target_track_id
            case self.FpsRole:
                return rec.fps
            case self.TracksRole:
                return rec.tracks_as_list()
            case self.PtzStateRole:
                return rec.ptz_as_dict()
            case self.HealthRole:
                return rec.health
            case self.ShmNameRole:
                return rec.shm_name
            case self.ShmWidthRole:
                return rec.shm_width
            case self.ShmHeightRole:
                return rec.shm_height
        return None

    def setData(self, index: QModelIndex | QPersistentModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or index.row() >= len(self._order):
            return False
        rec = self._records.get(self._order[index.row()])
        if rec is None:
            return False
        if role == self.TrackingEnabledRole:
            rec.tracking_enabled = bool(value)
            self.dataChanged.emit(index, index, [role])
            return True
        if role == self.TargetTrackIdRole:
            rec.target_track_id = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    # ── mutation helpers ──────────────────────────────────────────────────────

    def add_camera(self, rec: CameraRecord) -> None:
        if rec.camera_id in self._records:
            return
        row = len(self._order)
        self.beginInsertRows(QModelIndex(), row, row)
        self._order.append(rec.camera_id)
        self._records[rec.camera_id] = rec
        self.endInsertRows()

    def remove_camera(self, camera_id: str) -> bool:
        if camera_id not in self._records:
            return False
        row = self._order.index(camera_id)
        self.beginRemoveRows(QModelIndex(), row, row)
        self._order.pop(row)
        del self._records[camera_id]
        self.endRemoveRows()
        return True

    def update_telemetry(self, msg: TelemetryMsg) -> None:
        rec = self._records.get(msg.camera_id)
        if rec is None:
            return
        rec.telemetry = msg
        row = self._order.index(msg.camera_id)
        idx = self.index(row)
        self.dataChanged.emit(idx, idx, [
            self.FpsRole, self.TracksRole, self.PtzStateRole, self.HealthRole,
        ])

    def get_record(self, camera_id: str) -> CameraRecord | None:
        return self._records.get(camera_id)

    def camera_ids(self) -> list[str]:
        return list(self._order)

    @Slot(str, str)
    def swapCameras(self, id_a: str, id_b: str) -> None:
        """Swap two cameras by ID (called from QML drag-reorder)."""
        if id_a not in self._records or id_b not in self._records or id_a == id_b:
            return
        i, j = self._order.index(id_a), self._order.index(id_b)
        self.layoutAboutToBeChanged.emit()
        self._order[i], self._order[j] = self._order[j], self._order[i]
        self.layoutChanged.emit()

    @Slot(str, int)
    def moveCamera(self, camera_id: str, new_index: int) -> None:
        """Move camera to a specific position (called from QML drag-reorder)."""
        if camera_id not in self._records:
            return
        old_index = self._order.index(camera_id)
        new_index = max(0, min(new_index, len(self._order) - 1))
        if old_index == new_index:
            return
        self.layoutAboutToBeChanged.emit()
        self._order.pop(old_index)
        self._order.insert(new_index, camera_id)
        self.layoutChanged.emit()


# ── engine client ─────────────────────────────────────────────────────────────


class EngineClient(QObject):
    """Typed wrapper over the Engine command/telemetry contract.

    QML usage::

        engineClient.addCamera("rtsp://...", "Camera 1")
        engineClient.enableTracking(cameraId, true)
        engineClient.setTarget(cameraId, trackId)
        engineClient.ptzNudge(cameraId, 0.5, 0.0, 0.0)

    Engine integration (Phase 6)::

        # Called by the supervisor/worker on the engine thread:
        client.push_telemetry(msg)

        # Drain pending commands for the engine to execute:
        cmds = client.drain_commands()
    """

    # ── signals → QML ────────────────────────────────────────────────────────
    cameraAdded = Signal(str)        # camera_id
    cameraRemoved = Signal(str)      # camera_id
    telemetryUpdated = Signal(str)   # camera_id
    errorOccurred = Signal(str)      # human-readable message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._model = CameraListModel(self)
        self._cmd_queue: deque[BaseCommand] = deque()
        self._lock = threading.Lock()

    # ── Q_PROPERTY: cameraModel ───────────────────────────────────────────────

    @Property(QObject, constant=True)  # type: ignore[call-arg]
    def cameraModel(self) -> CameraListModel:
        return self._model

    # ── camera management ─────────────────────────────────────────────────────

    @Slot(str, str, result=str)
    def addCamera(self, source_uri: str, display_name: str) -> str:
        """Add a camera and return its stable UUID camera_id."""
        camera_id = str(uuid.uuid4())
        rec = CameraRecord(
            camera_id=camera_id,
            source_uri=source_uri,
            display_name=display_name or source_uri,
        )
        self._model.add_camera(rec)
        self._enqueue(AddCameraCmd(
            camera_id=camera_id,
            source_uri=source_uri,
            display_name=display_name,
        ))
        self.cameraAdded.emit(camera_id)
        return camera_id

    @Slot(str)
    def removeCamera(self, camera_id: str) -> None:
        if self._model.remove_camera(camera_id):
            self._enqueue(RemoveCameraCmd(camera_id=camera_id))
            self.cameraRemoved.emit(camera_id)

    # ── tracking control ──────────────────────────────────────────────────────

    @Slot(str, bool)
    def enableTracking(self, camera_id: str, enabled: bool) -> None:
        rec = self._model.get_record(camera_id)
        if rec is not None:
            rec.tracking_enabled = enabled
            row = self._model.camera_ids().index(camera_id)
            idx = self._model.index(row)
            self._model.dataChanged.emit(idx, idx, [CameraListModel.TrackingEnabledRole])
        self._enqueue(EnableTrackingCmd(camera_id=camera_id, enabled=enabled))

    @Slot(str, int)
    def setTarget(self, camera_id: str, track_id: int) -> None:
        rec = self._model.get_record(camera_id)
        if rec is not None:
            rec.target_track_id = track_id
            row = self._model.camera_ids().index(camera_id)
            idx = self._model.index(row)
            self._model.dataChanged.emit(idx, idx, [CameraListModel.TargetTrackIdRole])
        self._enqueue(SetTargetCmd(camera_id=camera_id, track_id=track_id))

    @Slot(str)
    def clearTarget(self, camera_id: str) -> None:
        rec = self._model.get_record(camera_id)
        if rec is not None:
            rec.target_track_id = None
            row = self._model.camera_ids().index(camera_id)
            idx = self._model.index(row)
            self._model.dataChanged.emit(idx, idx, [CameraListModel.TargetTrackIdRole])
        self._enqueue(SetTargetCmd(camera_id=camera_id, track_id=None))

    # ── PTZ nudge ─────────────────────────────────────────────────────────────

    @Slot(str, float, float, float)
    def ptzNudge(self, camera_id: str, pan: float, tilt: float, zoom: float) -> None:
        self._enqueue(PtzNudgeCmd(
            camera_id=camera_id,
            pan_speed=pan,
            tilt_speed=tilt,
            zoom_speed=zoom,
        ))

    @Slot(str, str)
    def ptzGoToPreset(self, camera_id: str, preset_name: str) -> None:
        self._enqueue(PtzGoToPresetCmd(camera_id=camera_id, preset_name=preset_name))

    # ── telemetry ingest (called from engine thread) ──────────────────────────

    def push_telemetry(self, msg: TelemetryMsg) -> None:
        """Deliver a telemetry snapshot from the engine worker.  Thread-safe."""
        self._model.update_telemetry(msg)
        self.telemetryUpdated.emit(msg.camera_id)

    # ── command drain (called from engine supervisor) ─────────────────────────

    def drain_commands(self) -> list[BaseCommand]:
        """Return and clear all pending commands.  Thread-safe."""
        with self._lock:
            cmds = list(self._cmd_queue)
            self._cmd_queue.clear()
        return cmds

    # ── read-only accessors for tests ─────────────────────────────────────────

    @property
    def camera_count(self) -> int:
        return self._model.rowCount()

    def get_camera(self, camera_id: str) -> CameraRecord | None:
        return self._model.get_record(camera_id)

    # ── private ───────────────────────────────────────────────────────────────

    def _enqueue(self, cmd: BaseCommand) -> None:
        with self._lock:
            self._cmd_queue.append(cmd)
