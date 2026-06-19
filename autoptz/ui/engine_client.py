"""EngineClient: connects the QML UI to the Engine over typed command/telemetry channels.

Phase 8 additions
-----------------
- ConfigStore integration: cameras/layouts/identities persist across restarts.
- updateCameraConfig   — debounced write + UpdateCameraConfigCmd to engine.
- ptzSavePreset / deletePreset — preset CRUD with model + store sync.
- saveCurrentLayout / loadLayout / deleteLayout — named layout round-trip.
- enrollIdentity / deleteIdentity / renameIdentity — identity management.
- getCameraConfig      — returns CameraConfig as a QVariant dict for QML.
- IdentityListModel / LayoutListModel — new list models for QML.
- getTheme / setTheme  — persist theme preference via ConfigStore.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import (
    Property,
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
    QThread,
    Signal,
    Slot,
)

from autoptz.engine.runtime.messages import (
    AddCameraCmd,
    BaseCommand,
    DeleteIdentityCmd,
    DeleteLayoutCmd,
    EnableTrackingCmd,
    EnrollIdentityCmd,
    PtzGoToPresetCmd,
    PtzHomeCmd,
    PtzMenuCmd,
    PtzNudgeCmd,
    PtzSavePresetCmd,
    RecallPtzPresetCmd,
    RemoveCameraCmd,
    RenameIdentityCmd,
    SaveLayoutCmd,
    SavePtzPresetCmd,
    SetFeaturesCmd,
    SetLayoutCmd,
    SetTargetCmd,
    SetTargetFpsCmd,
    SetTargetIdentityCmd,
    TelemetryMsg,
    UpdateCameraConfigCmd,
)

if TYPE_CHECKING:
    from autoptz.config.store import ConfigStore

log = logging.getLogger(__name__)

# Auto-harvested ("Person N") identities not seen for this long are forgotten, so
# their track-ID numbers don't accumulate; swept on this cadence.
_UNLABELED_MAX_AGE_S = 90.0
_UNLABELED_SWEEP_MS = 15000

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
    camera_config: Any | None = None   # autoptz.config.models.CameraConfig
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

    @property
    def resolution(self) -> str:
        """Source resolution as ``"WxH"`` from telemetry, or ``""`` until known."""
        if not self.telemetry:
            return ""
        w = getattr(self.telemetry, "width", 0)
        h = getattr(self.telemetry, "height", 0)
        if w and h:
            return f"{int(w)}x{int(h)}"
        return ""

    @property
    def dropped_frames(self) -> int:
        """Cumulative dropped/missed frames reported in telemetry."""
        if not self.telemetry:
            return 0
        return int(getattr(self.telemetry, "dropped_frames", 0))

    @property
    def latency_ms(self) -> int:
        """Per-frame processing latency in whole milliseconds (0 until known)."""
        if not self.telemetry:
            return 0
        return int(round(float(getattr(self.telemetry, "latency_ms", 0.0))))

    @property
    def streaming(self) -> bool:
        """True once the worker has read + pushed at least one real frame."""
        if not self.telemetry:
            return False
        return bool(getattr(self.telemetry, "streaming", False))

    @property
    def source_fps_cap(self) -> float:
        """The source's detected hardware fps ceiling (0.0 until known).

        The UI's fps slider caps itself at this real value instead of a hardcoded
        60/120 when it is non-zero.
        """
        if not self.telemetry:
            return 0.0
        return float(getattr(self.telemetry, "source_fps_cap", 0.0))

    def tracks_as_list(self) -> list[dict[str, Any]]:
        if not self.telemetry:
            return []
        # Detector produces pixel-space coords; QML overlays expect normalized 0–1.
        w = max(1.0, float(self.telemetry.width or 1))
        h = max(1.0, float(self.telemetry.height or 1))
        result = []
        for t in self.telemetry.tracks:
            result.append({
                "track_id": t.track_id,
                "bbox": {
                    "x1": t.bbox.x1 / w,
                    "y1": t.bbox.y1 / h,
                    "x2": t.bbox.x2 / w,
                    "y2": t.bbox.y2 / h,
                },
                "identity": t.identity or "",
                "identity_id": t.identity_id or "",
                "confidence": t.confidence,
                "is_target": t.is_target,
                "lost": bool(getattr(t, "lost", False)),
                # Velocity normalized to frame fractions/frame for overlay drawing.
                "vx": getattr(t, "vx", 0.0) / w,
                "vy": getattr(t, "vy", 0.0) / h,
                "aim": (
                    {
                        "x": max(0.0, min(1.0, float(t.aim_x) / w)),
                        "y": max(0.0, min(1.0, float(t.aim_y) / h)),
                        "source": t.aim_source or "",
                    }
                    if getattr(t, "aim_x", None) is not None
                    and getattr(t, "aim_y", None) is not None
                    else None
                ),
            })
        return result

    def faces_as_list(self) -> list[dict[str, Any]]:
        """Detected faces with normalized (0–1) bboxes for the face overlay."""
        if not self.telemetry:
            return []
        w = max(1.0, float(self.telemetry.width or 1))
        h = max(1.0, float(self.telemetry.height or 1))
        return [
            {
                "bbox": {
                    "x1": f.bbox.x1 / w, "y1": f.bbox.y1 / h,
                    "x2": f.bbox.x2 / w, "y2": f.bbox.y2 / h,
                },
                "identity": f.identity or "",
                "score": f.score,
            }
            for f in getattr(self.telemetry, "faces", []) or []
        ]

    def pose_as_list(self) -> list[dict[str, float]]:
        """Target pose keypoints with normalized (0–1) coords for the pose overlay."""
        if not self.telemetry:
            return []
        w = max(1.0, float(self.telemetry.width or 1))
        h = max(1.0, float(self.telemetry.height or 1))
        return [
            {"x": k.x / w, "y": k.y / h, "conf": k.conf}
            for k in getattr(self.telemetry, "pose", []) or []
        ]

    def ptz_as_dict(self) -> dict[str, Any]:
        if not self.telemetry:
            return {"pan": 0.0, "tilt": 0.0, "zoom": 0.0, "moving": False, "state": "idle"}
        p = self.telemetry.ptz
        return {"pan": p.pan, "tilt": p.tilt, "zoom": p.zoom, "moving": p.moving, "state": p.state}

    def presets_as_list(self) -> list[dict[str, Any]]:
        if not self.camera_config:
            return []
        return [
            {
                "id": pr.id,
                "idx": pr.idx,
                "name": pr.name,
                "pan": pr.pan,
                "tilt": pr.tilt,
                "zoom": pr.zoom,
            }
            for pr in self.camera_config.presets
        ]


# ── camera list model ─────────────────────────────────────────────────────────


class CameraListModel(QAbstractListModel):
    """Ordered list of CameraRecords exposed to QML."""

    CameraIdRole        = Qt.ItemDataRole.UserRole + 1
    DisplayNameRole     = Qt.ItemDataRole.UserRole + 2
    TrackingEnabledRole = Qt.ItemDataRole.UserRole + 3
    TargetTrackIdRole   = Qt.ItemDataRole.UserRole + 4
    FpsRole             = Qt.ItemDataRole.UserRole + 5
    TracksRole          = Qt.ItemDataRole.UserRole + 6
    PtzStateRole        = Qt.ItemDataRole.UserRole + 7
    HealthRole          = Qt.ItemDataRole.UserRole + 8
    ShmNameRole         = Qt.ItemDataRole.UserRole + 9
    ShmWidthRole        = Qt.ItemDataRole.UserRole + 10
    ShmHeightRole       = Qt.ItemDataRole.UserRole + 11
    PresetsRole         = Qt.ItemDataRole.UserRole + 12
    ResolutionRole      = Qt.ItemDataRole.UserRole + 13
    DroppedFramesRole   = Qt.ItemDataRole.UserRole + 14
    LatencyMsRole       = Qt.ItemDataRole.UserRole + 15
    StreamingRole       = Qt.ItemDataRole.UserRole + 16

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._order: list[str] = []
        self._records: dict[str, CameraRecord] = {}

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self._order)

    def roleNames(self) -> dict[int, QByteArray]:
        return {
            self.CameraIdRole:        QByteArray(b"cameraId"),
            self.DisplayNameRole:     QByteArray(b"displayName"),
            self.TrackingEnabledRole: QByteArray(b"trackingEnabled"),
            self.TargetTrackIdRole:   QByteArray(b"targetTrackId"),
            self.FpsRole:             QByteArray(b"fps"),
            self.TracksRole:          QByteArray(b"tracks"),
            self.PtzStateRole:        QByteArray(b"ptzState"),
            self.HealthRole:          QByteArray(b"health"),
            self.ShmNameRole:         QByteArray(b"shmName"),
            self.ShmWidthRole:        QByteArray(b"shmWidth"),
            self.ShmHeightRole:       QByteArray(b"shmHeight"),
            self.PresetsRole:         QByteArray(b"presets"),
            self.ResolutionRole:      QByteArray(b"resolution"),
            self.DroppedFramesRole:   QByteArray(b"droppedFrames"),
            self.LatencyMsRole:       QByteArray(b"latencyMs"),
            self.StreamingRole:       QByteArray(b"streaming"),
        }

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._order):
            return None
        rec = self._records.get(self._order[index.row()])
        if rec is None:
            return None
        match role:
            case self.CameraIdRole:        return rec.camera_id
            case self.DisplayNameRole:     return rec.display_name
            case self.TrackingEnabledRole: return rec.tracking_enabled
            case self.TargetTrackIdRole:   return rec.target_track_id
            case self.FpsRole:             return rec.fps
            case self.TracksRole:          return rec.tracks_as_list()
            case self.PtzStateRole:        return rec.ptz_as_dict()
            case self.HealthRole:          return rec.health
            case self.ShmNameRole:         return rec.shm_name
            case self.ShmWidthRole:        return rec.shm_width
            case self.ShmHeightRole:       return rec.shm_height
            case self.PresetsRole:         return rec.presets_as_list()
            case self.ResolutionRole:      return rec.resolution
            case self.DroppedFramesRole:   return rec.dropped_frames
            case self.LatencyMsRole:       return rec.latency_ms
            case self.StreamingRole:       return rec.streaming
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
            self.ResolutionRole, self.DroppedFramesRole, self.LatencyMsRole,
            self.StreamingRole,
        ])

    def get_record(self, camera_id: str) -> CameraRecord | None:
        return self._records.get(camera_id)

    def camera_ids(self) -> list[str]:
        return list(self._order)

    def _notify_camera(self, camera_id: str, roles: list[int]) -> None:
        if camera_id not in self._records:
            return
        row = self._order.index(camera_id)
        idx = self.index(row)
        self.dataChanged.emit(idx, idx, roles)

    @Slot(str, str)
    def swapCameras(self, id_a: str, id_b: str) -> None:
        if id_a not in self._records or id_b not in self._records or id_a == id_b:
            return
        i, j = self._order.index(id_a), self._order.index(id_b)
        self.layoutAboutToBeChanged.emit()
        self._order[i], self._order[j] = self._order[j], self._order[i]
        self.layoutChanged.emit()

    @Slot(str, int)
    def moveCamera(self, camera_id: str, new_index: int) -> None:
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


# ── identity list model ───────────────────────────────────────────────────────


def _thumbnail_data_uri(thumbnail: bytes | None) -> str:
    """Encode raw PNG/JPEG bytes as a ``data:image/png;base64,…`` URI.

    Returns ``""`` when there is no thumbnail.  The bytes are emitted verbatim
    under the ``image/png`` media type (auto-harvest crops are PNG-encoded; the
    UI sniffs the real format regardless of the declared type).
    """
    if not thumbnail:
        return ""
    import base64
    return "data:image/png;base64," + base64.b64encode(bytes(thumbnail)).decode("ascii")


def _thumbnails_data_uris(thumbnails: list[bytes] | None) -> list[str]:
    """Encode each candidate profile photo as a data URI (skips empties)."""
    if not thumbnails:
        return []
    return [_thumbnail_data_uri(t) for t in thumbnails if t]


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``patch`` into ``base`` in place and return ``base``.

    Nested dicts are merged key-by-key; any non-dict value in ``patch`` replaces
    the value in ``base``.  Used by :meth:`EngineClient.updateCameraConfigPatch`.
    """
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class IdentityListModel(QAbstractListModel):
    """Flat list of IdentityRecords for the People view / IdentityManager.

    FROZEN roles (QML binds to these exact names):
      ``identityId`` (str), ``identityName`` (str),
      ``thumbnail`` (str — a ``data:image/png;base64,…`` URI or ``""``),
      ``enabled`` (bool), ``labeled`` (bool; false = auto-harvested/unlabeled).
    """

    IdRole         = Qt.ItemDataRole.UserRole + 1
    NameRole       = Qt.ItemDataRole.UserRole + 2
    ThumbnailRole  = Qt.ItemDataRole.UserRole + 3
    EnabledRole    = Qt.ItemDataRole.UserRole + 4
    LabeledRole    = Qt.ItemDataRole.UserRole + 5
    ThumbnailsRole = Qt.ItemDataRole.UserRole + 6   # list of candidate photo URIs

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._records: list[Any] = []   # list[IdentityRecord]

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self._records)

    def roleNames(self) -> dict[int, QByteArray]:
        return {
            self.IdRole:         QByteArray(b"identityId"),
            self.NameRole:       QByteArray(b"identityName"),
            self.ThumbnailRole:  QByteArray(b"thumbnail"),
            self.EnabledRole:    QByteArray(b"enabled"),
            self.LabeledRole:    QByteArray(b"labeled"),
            self.ThumbnailsRole: QByteArray(b"thumbnails"),
        }

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._records):
            return None
        rec = self._records[index.row()]
        match role:
            case self.IdRole:        return rec.id
            case self.NameRole:      return rec.name
            case self.ThumbnailRole: return _thumbnail_data_uri(getattr(rec, "thumbnail", None))
            case self.EnabledRole:   return bool(getattr(rec, "enabled", True))
            case self.LabeledRole:   return bool(getattr(rec, "labeled", True))
            case self.ThumbnailsRole: return _thumbnails_data_uris(getattr(rec, "thumbnails", None))
        return None

    def _index_of(self, identity_id: str) -> int:
        for i, rec in enumerate(self._records):
            if rec.id == identity_id:
                return i
        return -1

    def add_identity(self, rec: Any) -> None:
        # Upsert: if an identity with this id already exists, update it in place
        # (auto-harvest re-pushes the same id when its template/thumbnail changes).
        i = self._index_of(rec.id)
        if i >= 0:
            self._records[i] = rec
            idx = self.index(i)
            self.dataChanged.emit(idx, idx, [
                self.NameRole, self.ThumbnailRole, self.EnabledRole, self.LabeledRole,
            ])
            return
        row = len(self._records)
        self.beginInsertRows(QModelIndex(), row, row)
        self._records.append(rec)
        self.endInsertRows()

    def update_identity(self, rec: Any) -> None:
        """Replace an existing record (no-op if absent)."""
        i = self._index_of(rec.id)
        if i < 0:
            return
        self._records[i] = rec
        idx = self.index(i)
        self.dataChanged.emit(idx, idx, [
            self.NameRole, self.ThumbnailRole, self.EnabledRole, self.LabeledRole,
            self.ThumbnailsRole,
        ])

    def remove_identity(self, identity_id: str) -> bool:
        for i, rec in enumerate(self._records):
            if rec.id == identity_id:
                self.beginRemoveRows(QModelIndex(), i, i)
                self._records.pop(i)
                self.endRemoveRows()
                return True
        return False

    def rename_identity(self, identity_id: str, new_name: str) -> bool:
        for i, rec in enumerate(self._records):
            if rec.id == identity_id:
                self._records[i] = rec.model_copy(update={"name": new_name})
                idx = self.index(i)
                self.dataChanged.emit(idx, idx, [self.NameRole])
                return True
        return False

    def get(self, identity_id: str) -> Any | None:
        i = self._index_of(identity_id)
        return self._records[i] if i >= 0 else None

    def get_all(self) -> list[Any]:
        return list(self._records)


# ── layout list model ─────────────────────────────────────────────────────────


class LayoutListModel(QAbstractListModel):
    """Flat list of Layout records for the Layout manager panel."""

    IdRole   = Qt.ItemDataRole.UserRole + 1
    NameRole = Qt.ItemDataRole.UserRole + 2

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._layouts: list[Any] = []  # list[Layout]

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self._layouts)

    def roleNames(self) -> dict[int, QByteArray]:
        return {
            self.IdRole:   QByteArray(b"layoutId"),
            self.NameRole: QByteArray(b"layoutName"),
        }

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or index.row() >= len(self._layouts):
            return None
        lo = self._layouts[index.row()]
        match role:
            case self.IdRole:   return lo.id
            case self.NameRole: return lo.name
        return None

    def add_layout(self, layout: Any) -> None:
        # Skip duplicates
        if any(lo.id == layout.id for lo in self._layouts):
            return
        row = len(self._layouts)
        self.beginInsertRows(QModelIndex(), row, row)
        self._layouts.append(layout)
        self.endInsertRows()

    def remove_layout(self, layout_id: str) -> bool:
        for i, lo in enumerate(self._layouts):
            if lo.id == layout_id:
                self.beginRemoveRows(QModelIndex(), i, i)
                self._layouts.pop(i)
                self.endRemoveRows()
                return True
        return False

    def get_layout(self, layout_id: str) -> Any | None:
        for lo in self._layouts:
            if lo.id == layout_id:
                return lo
        return None

    def get_all(self) -> list[Any]:
        return list(self._layouts)


# ── engine client ─────────────────────────────────────────────────────────────


class EngineClient(QObject):
    """Typed wrapper over the Engine command/telemetry contract.

    QML usage::

        engineClient.addCamera("rtsp://...", "Camera 1")
        engineClient.enableTracking(cameraId, true)
        engineClient.updateCameraConfig(cameraId, JSON.stringify(cfg))
        engineClient.ptzSavePreset(cameraId, "Home")
        engineClient.ptzGoToPreset(cameraId, "Home")
        engineClient.enrollIdentity(cameraId, "Alice", trackId)
        engineClient.saveCurrentLayout("Stage Left")
        engineClient.loadLayout(layoutId)
    """

    # ── signals → QML ────────────────────────────────────────────────────────
    cameraAdded      = Signal(str)     # camera_id
    cameraRemoved    = Signal(str)     # camera_id
    telemetryUpdated = Signal(str)     # camera_id
    configChanged    = Signal(str)     # camera_id
    identitiesChanged = Signal()
    layoutsChanged   = Signal()
    themeChanged     = Signal(str)     # "dark"|"light"|"system"
    featuresChanged  = Signal()        # global subsystem switches changed
    detectorModelTierChanged = Signal() # persisted detector model tier changed
    overlaysChanged  = Signal()        # which on-video overlays are shown changed
    startupProgressChanged = Signal()  # startup active/phase/counts changed
    optionalComponentsChanged = Signal()  # optional model/dependency prompt state
    targetChanged    = Signal(str)     # camera_id — the tracked target changed
    trackingChanged  = Signal(str)     # camera_id — tracking on/off changed
    errorOccurred    = Signal(str)     # human-readable message

    # ── engine lifecycle (FROZEN — other agents bind to these exact names) ─────
    engineStateChanged = Signal()      # engineRunning / engineEp changed

    # ── frame-source bridge (GUI thread connects these to ShmFrameSource) ──
    # Emitted when a worker's preview shm becomes available / goes away so the
    # app can attach/detach the camera tiles' ShmFrameSource on the GUI thread.
    providerAttachRequested = Signal(str, str, int, int)  # camera_id, shm_name, w, h
    providerDetachRequested = Signal(str)                  # camera_id

    # ── internal: marshals worker-thread telemetry onto the GUI thread ─────────
    _telemetryArrived = Signal(object)  # TelemetryMsg
    # ── internal: marshals a worker-thread harvested identity onto the GUI thread
    _identityArrived = Signal(object)   # IdentityRecord
    _startupProgressArrived = Signal(object)  # dict payload

    def __init__(
        self,
        store: "ConfigStore | None" = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._model = CameraListModel(self)
        self._identity_model = IdentityListModel(self)
        self._layout_model = LayoutListModel(self)
        self._cmd_queue: deque[BaseCommand] = deque()
        self._lock = threading.Lock()
        self._theme_mode: str = "dark"
        # uri → stable device unique_id, populated by scanUSBCameras() so
        # addCamera() can persist a stable id for USB sources.
        self._usb_unique_ids: dict[str, str] = {}
        # uri → friendly source kind ("Built-in"/"Continuity Camera"/…), also
        # populated by scanUSBCameras() so addCamera() can persist it for display.
        self._usb_source_labels: dict[str, str] = {}

        # ── engine lifecycle state ─────────────────────────────────────────────
        self._supervisor: Any | None = None
        self._supervisor_factory: Callable[[EngineClient], Any] | None = None
        self._engine_running: bool = False
        self._engine_ep: str = ""
        self._startup_active: bool = False
        self._startup_phase: str = ""
        self._startup_started_cameras: int = 0
        self._startup_total_cameras: int = 0
        self._startup_missing_components: list[str] = []
        self._features: dict[str, bool] = {
            "detection": True,
            "tracking": True,
            "face_recognition": True,
            "pose": True,
        }
        self._detector_model_tier: str = "auto"

        # Shared identity gallery (set by the supervisor when the engine starts).
        # When present, identity CRUD delegates to it so the running engine
        # matches the same records the UI shows; otherwise CRUD is store-only.
        self._identity_service: Any | None = None

        # In-app log bridge (the QtLogHandler + LogListModel built in app.py).
        # When wired, setLogLevel / copyLogsToClipboard / exportLogs operate on
        # them; the slots degrade to safe no-ops when absent (e.g. headless tests).
        self._log_model: Any | None = None
        self._log_handler: Any | None = None

        # Marshal worker-thread telemetry onto the GUI/owning thread.  A queued
        # connection guarantees the slot (which mutates the Qt model) runs on the
        # thread that owns this QObject, never on the worker thread.
        self._telemetryArrived.connect(
            self._on_telemetry_main, Qt.ConnectionType.QueuedConnection,
        )
        # Same pattern for worker-thread identity harvests.
        self._identityArrived.connect(
            self._on_identity_main, Qt.ConnectionType.QueuedConnection,
        )
        self._startupProgressArrived.connect(
            self._on_startup_progress_main, Qt.ConnectionType.QueuedConnection,
        )

        # Periodically forget auto-harvested "Person N" identities that haven't
        # been seen for a while, so their track-ID numbers don't pile up forever.
        from PySide6.QtCore import QTimer
        self._expire_timer = QTimer(self)
        self._expire_timer.timeout.connect(self._expire_unlabeled_identities)
        self._expire_timer.start(_UNLABELED_SWEEP_MS)

        if store:
            self._load_from_store()

    def _expire_unlabeled_identities(self) -> None:
        """Drop stale unlabeled identities from the gallery + UI model."""
        svc = self._identity_service
        if svc is None:
            return
        try:
            removed = svc.expire_unlabeled(_UNLABELED_MAX_AGE_S)
        except Exception:  # noqa: BLE001
            log.debug("expire_unlabeled failed", exc_info=True)
            return
        if not removed:
            return
        for iid in removed:
            self._identity_model.remove_identity(iid)
        self.identitiesChanged.emit()

    # ── startup load ──────────────────────────────────────────────────────────

    def _load_from_store(self) -> None:
        assert self._store is not None
        try:
            cameras = self._store.load_cameras()
            for cam in cameras:
                if not cam.enabled:
                    continue
                rec = CameraRecord(
                    camera_id=cam.id,
                    source_uri=cam.source.address,
                    display_name=cam.name,
                    camera_config=cam,
                )
                self._model.add_camera(rec)
            self._apply_persisted_camera_order()

            identities = self._store.load_identities()
            for identity in identities:
                self._identity_model.add_identity(identity)

            layouts = self._store.load_layouts()
            for layout in layouts:
                self._layout_model.add_layout(layout)

            theme_data = self._store.get_setting("theme", {})
            if isinstance(theme_data, dict):
                self._theme_mode = theme_data.get("name", "dark")
            tier = self._store.get_setting("detector_model_tier", "auto")
            self._detector_model_tier = self._normalize_detector_model_tier(tier)
        except Exception:
            log.exception("Failed to load config from store")

    # ── Q_PROPERTYs ───────────────────────────────────────────────────────────

    @Property(QObject, constant=True)  # type: ignore[call-arg]
    def cameraModel(self) -> CameraListModel:
        return self._model

    @Property(QObject, constant=True)  # type: ignore[call-arg]
    def identityModel(self) -> IdentityListModel:
        return self._identity_model

    @Property(QObject, constant=True)  # type: ignore[call-arg]
    def layoutModel(self) -> LayoutListModel:
        return self._layout_model

    @Property(str, notify=themeChanged)
    def themeMode(self) -> str:
        return self._theme_mode

    # ── engine lifecycle (FROZEN contract) ─────────────────────────────────────

    @Property(bool, notify=engineStateChanged)
    def engineRunning(self) -> bool:
        """True while the engine/supervisor is running."""
        return self._engine_running

    @Property(str, notify=engineStateChanged)
    def engineEp(self) -> str:
        """Active inference EP label (e.g. ``"CoreML"``, ``"CPU"``, or ``""``)."""
        return self._engine_ep

    @Property(bool, notify=startupProgressChanged)
    def startupActive(self) -> bool:
        """True while staged engine/camera/model startup is in progress."""
        return self._startup_active

    @Property(str, notify=startupProgressChanged)
    def startupPhase(self) -> str:
        """Short user-facing startup phase shown by the top loading bar."""
        return self._startup_phase

    @Property(int, notify=startupProgressChanged)
    def startupStartedCameras(self) -> int:
        return self._startup_started_cameras

    @Property(int, notify=startupProgressChanged)
    def startupTotalCameras(self) -> int:
        return self._startup_total_cameras

    @Property("QVariantList", notify=startupProgressChanged)
    def startupMissingComponents(self) -> list[str]:
        return list(self._startup_missing_components)

    def set_supervisor(self, supervisor: Any | None) -> None:
        """Inject the supervisor instance the lifecycle slots will drive."""
        self._supervisor = supervisor

    def set_identity_service(self, service: Any | None) -> None:
        """Share the engine's identity gallery so UI CRUD reaches the matcher.

        Called by the supervisor when the engine starts.  When set, the identity
        slots mutate this gallery (and let it own persistence) instead of writing
        to the store directly, keeping the UI model and the live matcher in sync.
        """
        self._identity_service = service

    def set_supervisor_factory(
        self, factory: Callable[[EngineClient], Any] | None,
    ) -> None:
        """Inject a factory ``(engine_client) -> supervisor`` used by startEngine.

        Lets the UI defer supervisor construction (and its heavy imports) until
        the user actually starts the engine.
        """
        self._supervisor_factory = factory

    @Slot()
    def startEngine(self) -> None:
        """Create (if needed) and start the supervisor.  Idempotent."""
        if self._engine_running:
            return
        if self._supervisor is None and self._supervisor_factory is not None:
            try:
                self._supervisor = self._supervisor_factory(self)
            except Exception as exc:  # noqa: BLE001
                log.exception("Failed to create supervisor")
                self.errorOccurred.emit(f"Engine failed to start: {exc}")
                return
        if self._supervisor is None:
            log.warning("startEngine: no supervisor injected")
            self.errorOccurred.emit("Engine not available (no supervisor configured)")
            return
        self._set_startup_progress(
            active=True,
            phase="Starting engine",
            started=0,
            total=len(self._model.camera_ids()),
            missing=self._missing_optional_components(promptable_only=True),
        )
        self.resetFeatureOverrides()
        try:
            try:
                self._supervisor.start(
                    staged=True,
                    progress=self._set_startup_progress,
                )
            except TypeError:
                # Test fakes and older supervisors keep the no-arg contract.
                self._supervisor.start()
                self._set_startup_progress(
                    active=False,
                    phase="Ready",
                    started=len(self._model.camera_ids()),
                    total=len(self._model.camera_ids()),
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Supervisor start failed")
            self._set_startup_progress(active=False, phase="Startup failed")
            self.errorOccurred.emit(f"Engine failed to start: {exc}")
            return

        self._engine_running = True
        try:
            self._engine_ep = self._supervisor.active_ep
        except Exception:  # noqa: BLE001
            self._engine_ep = ""
        # Push the session-only global feature switches. They intentionally reset
        # to all-on at launch; Services-panel disables are testing overrides only.
        try:
            self._enqueue(SetFeaturesCmd(camera_id=None, features=self.features()))
        except Exception:  # noqa: BLE001
            log.debug("initial SetFeaturesCmd enqueue failed", exc_info=True)
        self.engineStateChanged.emit()
        log.info("Engine started (ep=%s)", self._engine_ep)

    @Slot()
    def stopEngine(self) -> None:
        """Stop the supervisor.  Idempotent."""
        if not self._engine_running:
            return
        if self._supervisor is not None:
            try:
                self._supervisor.stop()
            except Exception:  # noqa: BLE001
                log.exception("Supervisor stop failed")
        self._engine_running = False
        self._engine_ep = ""
        self._set_startup_progress(active=False, phase="")
        # Detach the (now-stopped) engine's gallery; CRUD reverts to store-only.
        self._identity_service = None
        self.engineStateChanged.emit()
        log.info("Engine stopped")

    def _set_startup_progress(
        self,
        *,
        active: bool | None = None,
        phase: str | None = None,
        started: int | None = None,
        total: int | None = None,
        missing: list[str] | None = None,
    ) -> None:
        """Update startup progress from the GUI thread or supervisor thread.

        The supervisor calls this from its staged startup thread; marshal back to
        the QObject owner thread before mutating Qt-facing properties.
        """
        payload = {
            "active": active,
            "phase": phase,
            "started": started,
            "total": total,
            "missing": missing,
        }
        if self.thread() is not QThread.currentThread():
            self._startupProgressArrived.emit(payload)
            return
        self._apply_startup_progress(payload)

    @Slot(object)
    def _on_startup_progress_main(self, payload: dict[str, Any]) -> None:
        self._apply_startup_progress(payload)

    def _apply_startup_progress(self, payload: dict[str, Any]) -> None:
        active = payload.get("active")
        phase = payload.get("phase")
        started = payload.get("started")
        total = payload.get("total")
        missing = payload.get("missing")
        if active is not None:
            self._startup_active = bool(active)
        if phase is not None:
            self._startup_phase = str(phase)
        if started is not None:
            self._startup_started_cameras = max(0, int(started))
        if total is not None:
            self._startup_total_cameras = max(0, int(total))
        if missing is not None:
            self._startup_missing_components = list(missing)
        self.startupProgressChanged.emit()

    @Slot()
    def restartEngine(self) -> None:
        """Stop (if running) then start the engine.  Safe regardless of state."""
        log.info("Restarting engine")
        self.stopEngine()
        self.startEngine()

    # ── app settings (thin wrappers over ConfigStore — used by QML to persist
    #    window geometry, columns, theme, selection, engine on/off, …) ─────────

    @Slot(str, "QVariant", result="QVariant")
    def getSetting(self, key: str, default: Any = None) -> Any:
        """Return a persisted setting (JSON value) or *default* when absent."""
        if self._store is None:
            return default
        try:
            return self._store.get_setting(key, default)
        except Exception:  # noqa: BLE001
            log.exception("getSetting(%r) failed", key)
            return default

    @Slot(str, "QVariant")
    def setSetting(self, key: str, value: Any) -> None:
        """Persist a setting as a JSON value."""
        if self._store is None:
            return
        try:
            self._store.set_setting(key, value)
        except Exception:  # noqa: BLE001
            log.exception("setSetting(%r) failed", key)

    def _persist_camera_order(self) -> None:
        """Persist the current model order as a simple camera-id list."""
        if self._store is None:
            return
        try:
            self._store.set_setting("camera_order", self._model.camera_ids())
        except Exception:  # noqa: BLE001
            log.exception("persist camera_order failed")

    def _apply_persisted_camera_order(self) -> None:
        """Apply the persisted camera order after camera configs are loaded."""
        if self._store is None:
            return
        raw = self._store.get_setting("camera_order", [])
        if not isinstance(raw, list):
            return
        existing = set(self._model.camera_ids())
        ordered = [str(cid) for cid in raw if str(cid) in existing]
        for i, cid in enumerate(ordered):
            self._model.moveCamera(cid, i)

    @Slot(str, int)
    def moveCameraPersisted(self, camera_id: str, new_index: int) -> None:
        """Move a camera in the wall order and persist the new order."""
        before = self._model.camera_ids()
        self._model.moveCamera(camera_id, int(new_index))
        if self._model.camera_ids() != before:
            self._persist_camera_order()

    # ── on-video overlays (operator toggles) ────────────────────────────────────

    # Default visibility for each overlay layer; detection boxes on, the heavier
    # diagnostic layers off until the operator asks for them.
    _OVERLAY_DEFAULTS = {"detection": True, "faces": False, "pose": False}

    def overlays(self) -> dict[str, bool]:
        """Return which on-video overlays are enabled (persisted, with defaults)."""
        stored = self.getSetting("overlays", {}) or {}
        return {k: bool(stored.get(k, d)) for k, d in self._OVERLAY_DEFAULTS.items()}

    @Slot(str, bool)
    def setOverlay(self, key: str, enabled: bool) -> None:
        """Toggle one overlay layer (``detection`` / ``faces`` / ``pose``)."""
        if key not in self._OVERLAY_DEFAULTS:
            return
        cur = self.overlays()
        cur[key] = bool(enabled)
        self.setSetting("overlays", cur)
        self.overlaysChanged.emit()

    # ── frame-provider bridge (called from the supervisor / worker side) ───────

    def request_provider_attach(
        self, camera_id: str, shm_name: str, width: int, height: int,
    ) -> None:
        """Ask the GUI thread to attach the frame provider for *camera_id*.

        Safe to call from any thread — the connected slot in ``app.py`` uses a
        queued connection so the attach runs on the GUI thread.
        """
        self.providerAttachRequested.emit(camera_id, shm_name, width, height)

    def request_provider_detach(self, camera_id: str) -> None:
        """Ask the GUI thread to detach the frame provider for *camera_id*."""
        self.providerDetachRequested.emit(camera_id)

    # ── camera management ─────────────────────────────────────────────────────

    @Slot(str, str, result=str)
    @Slot(str, str, str, result=str)
    @Slot(str, str, str, str, result=str)
    def addCamera(
        self, source_uri: str, display_name: str,
        unique_id: str = "", source_label: str = "",
    ) -> str:
        from autoptz.config.models import CameraConfig, SourceConfig

        camera_id = str(uuid.uuid4())
        name = display_name.strip() or source_uri

        source_type: str = "usb"
        if source_uri.startswith("rtsp://") or source_uri.startswith("rtsps://"):
            source_type = "rtsp"
        elif source_uri.startswith("onvif://"):
            source_type = "onvif"
        elif source_uri.startswith("ndi://"):
            source_type = "ndi"

        # For USB sources, persist the stable device id so the camera re-binds to
        # the EXACT physical device by uniqueID (not a fragile index) across
        # restarts / index shuffles (Continuity Camera coming and going).  Prefer
        # the id passed straight from the camera menu (which knows it for the
        # exact row clicked); only fall back to the scan cache when absent — so a
        # stale/rebuilt cache can never mis-associate the wrong device.
        uid: str | None = None
        if source_type == "usb":
            uid = unique_id or self._usb_unique_ids.get(source_uri) or None
            source_label = source_label or self._usb_source_labels.get(source_uri, "")

        cam_config = CameraConfig(
            id=camera_id,
            name=name,
            source=SourceConfig(
                type=source_type, address=source_uri, unique_id=uid,
                source_label=source_label,
            ),
        )

        rec = CameraRecord(
            camera_id=camera_id,
            source_uri=source_uri,
            display_name=name,
            camera_config=cam_config,
        )
        self._model.add_camera(rec)

        if self._store:
            self._store.save_camera(cam_config)

        self._enqueue(AddCameraCmd(
            camera_id=camera_id,
            source_uri=source_uri,
            display_name=name,
        ))
        self.cameraAdded.emit(camera_id)
        self._persist_camera_order()
        return camera_id

    @Slot(str)
    def removeCamera(self, camera_id: str) -> None:
        if self._model.remove_camera(camera_id):
            if self._store:
                self._store.delete_camera(camera_id)
                self._persist_camera_order()
            self._enqueue(RemoveCameraCmd(camera_id=camera_id))
            self.cameraRemoved.emit(camera_id)

    # ── camera config ─────────────────────────────────────────────────────────

    @Slot(str, result="QVariant")
    def getCameraConfig(self, camera_id: str) -> dict[str, Any]:
        """Return the current CameraConfig as a plain dict for QML."""
        rec = self._model.get_record(camera_id)
        if rec and rec.camera_config:
            return json.loads(rec.camera_config.model_dump_json())
        return {}

    def updateCameraConfigPatch(self, camera_id: str, patch: dict[str, Any]) -> None:
        """Deep-merge *patch* into the camera's current config, then apply it.

        A convenience over :meth:`updateCameraConfig` for callers that only touch
        a few nested keys (e.g. the tile dragging the framing box updates just
        ``{"ptz": {"safe_zone_w": ..., "safe_zone_h": ...}}``) without having to
        round-trip and rebuild the whole config dict themselves.
        """
        cfg = self.getCameraConfig(camera_id)
        if not cfg:
            return
        _deep_merge(cfg, patch)
        self.updateCameraConfig(camera_id, json.dumps(cfg))

    @Slot(str, str)
    def updateCameraConfig(self, camera_id: str, config_json: str) -> None:
        """Apply a full CameraConfig update from QML (debounced write to store)."""
        from autoptz.config.models import CameraConfig

        rec = self._model.get_record(camera_id)
        if rec is None:
            return
        try:
            config_data = json.loads(config_json)
            # Preserve the stable ID
            config_data["id"] = camera_id
            new_cfg = CameraConfig.model_validate(config_data)
        except Exception as exc:
            log.warning("updateCameraConfig: invalid config for %s: %s", camera_id, exc)
            self.errorOccurred.emit(f"Config validation error: {exc}")
            return

        rec.camera_config = new_cfg
        rec.display_name = new_cfg.name

        self._model._notify_camera(camera_id, [
            CameraListModel.DisplayNameRole, CameraListModel.PresetsRole,
        ])

        if self._store:
            self._store.save_camera_debounced(new_cfg)

        self._enqueue(UpdateCameraConfigCmd(
            camera_id=camera_id,
            config=json.loads(new_cfg.model_dump_json()),
        ))
        self.configChanged.emit(camera_id)

    # ── tracking control ──────────────────────────────────────────────────────

    @Slot(str, bool)
    def enableTracking(self, camera_id: str, enabled: bool) -> None:
        rec = self._model.get_record(camera_id)
        if rec is not None:
            rec.tracking_enabled = enabled
            self._model._notify_camera(camera_id, [CameraListModel.TrackingEnabledRole])
            self.trackingChanged.emit(camera_id)
        self._enqueue(EnableTrackingCmd(camera_id=camera_id, enabled=enabled))

    @Slot(str, int)
    def setTarget(self, camera_id: str, track_id: int) -> None:
        rec = self._model.get_record(camera_id)
        if rec is not None:
            rec.target_track_id = track_id
            # An explicit box-click supersedes identity targeting; clear the stored
            # identity target so the pickers don't show a stale name.
            if rec.camera_config is not None:
                new_target = rec.camera_config.target.model_copy(update={
                    "mode": "manual", "identity_id": None,
                })
                rec.camera_config = rec.camera_config.model_copy(
                    update={"target": new_target})
                if self._store:
                    self._store.save_camera_debounced(rec.camera_config)
            self._model._notify_camera(camera_id, [CameraListModel.TargetTrackIdRole])
        self._enqueue(SetTargetCmd(camera_id=camera_id, track_id=track_id))
        self.targetChanged.emit(camera_id)

    @Slot(str)
    def clearTarget(self, camera_id: str) -> None:
        rec = self._model.get_record(camera_id)
        if rec is not None:
            rec.target_track_id = None
            if rec.camera_config is not None:
                new_target = rec.camera_config.target.model_copy(update={
                    "mode": "off", "identity_id": None,
                })
                rec.camera_config = rec.camera_config.model_copy(
                    update={"target": new_target})
                if self._store:
                    self._store.save_camera_debounced(rec.camera_config)
            self._model._notify_camera(camera_id, [CameraListModel.TargetTrackIdRole])
        self._enqueue(SetTargetCmd(camera_id=camera_id, track_id=None))
        self._enqueue(SetTargetIdentityCmd(camera_id=camera_id, identity_id=None))
        self.targetChanged.emit(camera_id)

    @Slot(str)
    def clearTargetAndStop(self, camera_id: str) -> None:
        """Stop tracking and clear any manual/identity target for one camera."""
        self.enableTracking(camera_id, False)
        self.clearTarget(camera_id)

    # ── fps control ───────────────────────────────────────────────────────────

    @Slot(str, float)
    def setTargetFps(self, camera_id: str, fps: float) -> None:
        """Change a camera's capture/detection fps **live** (no engine restart).

        Persists the new fps into the camera's ``SourceConfig`` (so it survives
        restart) and enqueues a :class:`SetTargetFpsCmd` the supervisor routes to
        the running worker, which re-paces its ingest adapter immediately.  When
        the source's hardware cap is known the value is clamped to it.
        """
        rec = self._model.get_record(camera_id)
        if rec is not None and rec.camera_config is not None:
            cap = rec.source_fps_cap
            applied = float(fps)
            if cap and cap > 0:
                applied = min(applied, cap)
            applied = max(1.0, applied)
            new_source = rec.camera_config.source.model_copy(update={"fps": applied})
            new_cfg = rec.camera_config.model_copy(update={"source": new_source})
            rec.camera_config = new_cfg
            if self._store:
                self._store.save_camera_debounced(new_cfg)
            self._enqueue(SetTargetFpsCmd(camera_id=camera_id, fps=applied))
            # Also push the new config so the running worker re-paces live even
            # before the supervisor routes the dedicated SetTargetFpsCmd: the
            # worker's update_config handler applies an fps change immediately.
            self._enqueue(UpdateCameraConfigCmd(
                camera_id=camera_id,
                config=json.loads(new_cfg.model_dump_json()),
            ))
        else:
            self._enqueue(SetTargetFpsCmd(camera_id=camera_id, fps=max(1.0, float(fps))))

    @Slot(str, result=float)
    def sourceFpsCap(self, camera_id: str) -> float:
        """Return the camera's detected hardware fps ceiling (0.0 until known).

        The properties panel reads this to cap its fps slider at the source's
        real maximum instead of a hardcoded 60/120.
        """
        rec = self._model.get_record(camera_id)
        return rec.source_fps_cap if rec is not None else 0.0

    # ── PTZ control ───────────────────────────────────────────────────────────

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

    @Slot(str, str)
    def ptzSavePreset(self, camera_id: str, preset_name: str) -> None:
        """Capture current PTZ position (from telemetry) as a named preset."""
        from autoptz.config.models import PTZPreset

        rec = self._model.get_record(camera_id)
        if rec is None or rec.camera_config is None:
            return

        ptz = rec.ptz_as_dict()
        existing_idxes = [p.idx for p in rec.camera_config.presets]
        next_idx = max(existing_idxes, default=0) + 1

        new_preset = PTZPreset(
            camera_id=camera_id,
            idx=next_idx,
            name=preset_name,
            pan=ptz.get("pan", 0.0),
            tilt=ptz.get("tilt", 0.0),
            zoom=ptz.get("zoom", 0.0),
        )
        new_presets = list(rec.camera_config.presets) + [new_preset]
        new_cfg = rec.camera_config.model_copy(update={"presets": new_presets})
        rec.camera_config = new_cfg

        if self._store:
            self._store.save_camera_debounced(new_cfg)

        self._model._notify_camera(camera_id, [CameraListModel.PresetsRole])
        self._enqueue(PtzSavePresetCmd(camera_id=camera_id, preset_name=preset_name))

    @Slot(str, int)
    def deletePreset(self, camera_id: str, preset_idx: int) -> None:
        rec = self._model.get_record(camera_id)
        if rec is None or rec.camera_config is None:
            return
        new_presets = [p for p in rec.camera_config.presets if p.idx != preset_idx]
        new_cfg = rec.camera_config.model_copy(update={"presets": new_presets})
        rec.camera_config = new_cfg

        if self._store:
            self._store.save_camera_debounced(new_cfg)

        self._model._notify_camera(camera_id, [CameraListModel.PresetsRole])

    # ── PTZ quick-recall preset slots (the 6-button row on the camera tile) ─────

    @Slot(str, int)
    @Slot(str, int, str)
    @Slot(str, int, str, str)
    def savePtzPreset(
        self, camera_id: str, slot: int, label: str = "", thumbnail: str = "",
    ) -> None:
        """Store the camera's current PTZ position into hardware preset *slot*.

        Sends :class:`SavePtzPresetCmd` (the supervisor routes it to the worker's
        ``save_preset(slot)``) and records the slot's UI metadata — a ``label``
        and a ``thumbnail`` (a base64 / data-URI snapshot of the current view) —
        in the camera's :class:`PTZConfig.preset_slots` so the PTZ section can show
        a labelled snapshot tile that survives a restart.  ``label`` defaults to
        ``"Preset N"`` when blank; a blank ``thumbnail`` keeps any existing one.
        """
        from autoptz.config.models import PtzPresetSlot

        self._enqueue(SavePtzPresetCmd(camera_id=camera_id, slot=int(slot)))

        rec = self._model.get_record(camera_id)
        if rec is None or rec.camera_config is None:
            return
        text = (label or "").strip() or f"Preset {int(slot) + 1}"
        slots = dict(rec.camera_config.ptz.preset_slots)
        prev = slots.get(int(slot))
        thumb = (thumbnail or "").strip() or (prev.thumbnail if prev else None)
        slots[int(slot)] = PtzPresetSlot(label=text, thumbnail=thumb)
        new_ptz = rec.camera_config.ptz.model_copy(update={"preset_slots": slots})
        new_cfg = rec.camera_config.model_copy(update={"ptz": new_ptz})
        rec.camera_config = new_cfg

        if self._store:
            self._store.save_camera_debounced(new_cfg)

        self._model._notify_camera(camera_id, [CameraListModel.PresetsRole])
        # Mirror the new config to the running worker so its PTZ config stays in
        # sync (matches the path other config setters use).
        self._enqueue(UpdateCameraConfigCmd(
            camera_id=camera_id,
            config=json.loads(new_cfg.model_dump_json()),
        ))
        self.configChanged.emit(camera_id)

    @Slot(str, int)
    def recallPtzPreset(self, camera_id: str, slot: int) -> None:
        """Recall hardware PTZ preset *slot* on the camera (a safe no-op when the
        backend can't, handled worker-side)."""
        self._enqueue(RecallPtzPresetCmd(camera_id=camera_id, slot=int(slot)))

    @Slot(str)
    def ptzHome(self, camera_id: str) -> None:
        """Send the camera to its home position (safe no-op when unsupported)."""
        self._enqueue(PtzHomeCmd(camera_id=camera_id))

    @Slot(str)
    def ptzMenu(self, camera_id: str) -> None:
        """Toggle the camera's on-screen (OSD) menu (safe no-op when unsupported)."""
        self._enqueue(PtzMenuCmd(camera_id=camera_id))

    # ── session-only feature switches + detector model tier ────────────────────

    _FEATURE_KEYS = ("detection", "tracking", "face_recognition", "pose")

    @Slot(result="QVariant")
    def features(self) -> dict[str, bool]:
        """Return the session-only ML-subsystem switches (all default True).

        Keys: ``detection``, ``tracking``, ``face_recognition``, ``pose``.
        These do not persist; they are testing overrides that reset each launch.
        """
        return {k: bool(self._features.get(k, True)) for k in self._FEATURE_KEYS}

    @Slot(str, bool)
    def setFeatureEnabled(self, name: str, enabled: bool) -> None:
        """Enable/disable a global subsystem for this session.

        ``name`` is one of ``detection``/``tracking``/``face_recognition``/``pose``.
        Broadcasts a :class:`SetFeaturesCmd` so every worker turns the subsystem
        on/off without a restart, but does not persist across app launches.
        """
        if name not in self._FEATURE_KEYS:
            return
        feats = self.features()
        feats[name] = bool(enabled)
        self._features = feats
        self._enqueue(SetFeaturesCmd(camera_id=None, features=feats))
        self.featuresChanged.emit()

    @Slot()
    def resetFeatureOverrides(self) -> None:
        """Reset all session-only module testing overrides to enabled."""
        feats = {k: True for k in self._FEATURE_KEYS}
        changed = feats != self._features
        self._features = feats
        if self._engine_running:
            self._enqueue(SetFeaturesCmd(camera_id=None, features=feats))
        if changed:
            self.featuresChanged.emit()

    @Property(str, notify=detectorModelTierChanged)
    def detectorModelTier(self) -> str:
        return self._detector_model_tier

    @Slot(result=str)
    def getDetectorModelTier(self) -> str:
        return self._detector_model_tier

    @Slot(str)
    def setDetectorModelTier(self, tier: str) -> None:
        normalized = self._normalize_detector_model_tier(tier)
        if normalized == self._detector_model_tier:
            return
        self._detector_model_tier = normalized
        self.setSetting("detector_model_tier", normalized)
        self.detectorModelTierChanged.emit()

    @staticmethod
    def _normalize_detector_model_tier(tier: Any) -> str:
        value = str(tier or "auto").strip().lower()
        return value if value in {"auto", "fast", "balanced"} else "auto"

    # ── optional component setup / ignore state ─────────────────────────────

    _OPTIONAL_COMPONENTS = ("reid", "pose", "face")

    @Slot(result="QVariantList")
    def optionalComponents(self) -> list[dict[str, Any]]:
        """Return optional model/dependency setup rows for the Services panel.

        ``ignored`` is persisted per component so "ignore forever" hides future
        launch prompts without hiding health rows or the manual Retry action.
        """
        ignored = self.getSetting("optional_components_ignored", {}) or {}
        out = []
        try:
            from autoptz.engine.runtime.diagnostics import optional_components

            rows = optional_components()
        except Exception:  # noqa: BLE001
            rows = []
        for row in rows:
            key = str(row.get("key", ""))
            if key not in self._OPTIONAL_COMPONENTS:
                continue
            item = dict(row)
            item["ignored"] = bool(ignored.get(key, False))
            item["prompt"] = item.get("state") != "ok" and not item["ignored"]
            out.append(item)
        return out

    @Slot(str, bool)
    def setOptionalComponentIgnored(self, key: str, ignored: bool) -> None:
        if key not in self._OPTIONAL_COMPONENTS:
            return
        cur = self.getSetting("optional_components_ignored", {}) or {}
        if not isinstance(cur, dict):
            cur = {}
        cur[key] = bool(ignored)
        self.setSetting("optional_components_ignored", cur)
        self._startup_missing_components = self._missing_optional_components(
            promptable_only=True,
        )
        self.optionalComponentsChanged.emit()
        self.startupProgressChanged.emit()

    @Slot(str)
    def retryOptionalComponent(self, key: str) -> None:
        """Offer a retry hook without mutating the Python environment.

        In-app setup intentionally downloads/prepares model assets only; Python
        packages still come from requirements/installer.  The concrete download
        command stays in ``tools/fetch_models.py`` until packaged model bundles
        are wired.
        """
        if key not in self._OPTIONAL_COMPONENTS:
            return
        self.setOptionalComponentIgnored(key, False)
        self.errorOccurred.emit(
            "Optional setup is available via tools/fetch_models.py; "
            "packaged model-bundle download will use this retry hook."
        )

    def _missing_optional_components(self, *, promptable_only: bool) -> list[str]:
        missing: list[str] = []
        for row in self.optionalComponents():
            if row.get("state") == "ok":
                continue
            if promptable_only and row.get("ignored"):
                continue
            missing.append(str(row.get("key", "")))
        return [m for m in missing if m]

    # ── layout management ─────────────────────────────────────────────────────

    @Slot(str)
    def saveCurrentLayout(self, layout_name: str) -> None:
        """Persist the current camera order as a named layout."""
        from autoptz.config.models import Layout, TilePlacement

        if not layout_name.strip():
            self.errorOccurred.emit("Layout name must not be blank")
            return

        tiles = [
            TilePlacement(camera_id=cid, x=0, y=0, w=1, h=1, z=0, visible=True)
            for cid in self._model.camera_ids()
        ]
        layout = Layout(name=layout_name.strip(), tiles=tiles)

        if self._store:
            self._store.save_layout(layout)

        self._layout_model.add_layout(layout)
        self.layoutsChanged.emit()

        self._enqueue(SaveLayoutCmd(
            layout_name=layout_name,
            tiles=[json.loads(t.model_dump_json()) for t in tiles],
        ))

    @Slot(str)
    def loadLayout(self, layout_id: str) -> None:
        """Restore camera order from a saved layout."""
        layout = self._layout_model.get_layout(layout_id)
        if layout is None:
            return

        ordered = [t.camera_id for t in layout.tiles if t.visible]
        existing = set(self._model.camera_ids())
        valid = [cid for cid in ordered if cid in existing]

        for i, cid in enumerate(valid):
            self._model.moveCamera(cid, i)

        if self._store:
            self._store.set_setting("active_layout_id", layout_id)
            self._persist_camera_order()

        self._enqueue(SetLayoutCmd(layout_name=layout.name))

    @Slot(str)
    def deleteLayout(self, layout_id: str) -> None:
        if self._store:
            self._store.delete_layout(layout_id)
        self._layout_model.remove_layout(layout_id)
        self.layoutsChanged.emit()
        self._enqueue(DeleteLayoutCmd(camera_id=None, layout_id=layout_id))

    # ── identity management ───────────────────────────────────────────────────

    @Slot(str, str, int)
    @Slot(str, str, int, float, float)
    def enrollIdentity(
        self, camera_id: str, identity_name: str, track_id: int,
        click_x: float | None = None, click_y: float | None = None,
    ) -> None:
        """Register a new identity and send enrollment command to engine."""
        from autoptz.config.models import IdentityRecord

        if not identity_name.strip():
            self.errorOccurred.emit("Identity name must not be blank")
            return

        identity_id = str(uuid.uuid4())
        identity = IdentityRecord(id=identity_id, name=identity_name.strip())

        # Prefer the live gallery (it owns persistence); else store directly.
        if self._identity_service is not None:
            try:
                self._identity_service.enroll(
                    identity_name.strip(), None, identity_id=identity_id,
                )
            except Exception:  # noqa: BLE001
                log.debug("identity_service.enroll failed", exc_info=True)
                if self._store:
                    self._store.save_identity(identity)
        elif self._store:
            self._store.save_identity(identity)

        self._identity_model.add_identity(identity)
        self.identitiesChanged.emit()

        self._enqueue(EnrollIdentityCmd(
            camera_id=camera_id,
            identity_name=identity_name.strip(),
            identity_id=identity_id,
            track_id=track_id,
            click_x=click_x,
            click_y=click_y,
        ))

    @Slot(str, str, int)
    @Slot(str, str, int, float, float)
    def assignTrackToIdentity(
        self, camera_id: str, identity_id: str, track_id: int,
        click_x: float | None = None, click_y: float | None = None,
    ) -> None:
        """Bind a clicked track's face to an EXISTING identity (click-to-assign).

        The worker captures the track's current face embedding on the next face
        tick and appends it to ``identity_id`` so the person is recognised later.
        """
        if not identity_id:
            return
        name = ""
        for ident in self.registeredIdentities():
            if ident.get("id") == identity_id:
                name = ident.get("name", "")
                break
        self._enqueue(EnrollIdentityCmd(
            camera_id=camera_id,
            identity_name=name,
            identity_id=identity_id,
            track_id=track_id,
            click_x=click_x,
            click_y=click_y,
        ))

    @Slot(str)
    def deleteIdentity(self, identity_id: str) -> None:
        if self._identity_service is not None:
            try:
                self._identity_service.delete(identity_id)
            except Exception:  # noqa: BLE001
                log.debug("identity_service.delete failed", exc_info=True)
        elif self._store:
            self._store.delete_identity(identity_id)
        self._identity_model.remove_identity(identity_id)
        self.identitiesChanged.emit()
        self._enqueue(DeleteIdentityCmd(camera_id=None, identity_id=identity_id))

    @Slot(str, str)
    def renameIdentity(self, identity_id: str, new_name: str) -> None:
        if not new_name.strip():
            self.errorOccurred.emit("Identity name must not be blank")
            return

        if self._store:
            from datetime import UTC, datetime
            for identity in self._store.load_identities():
                if identity.id == identity_id:
                    updated = identity.model_copy(update={
                        "name": new_name.strip(),
                        "updated_at": datetime.now(UTC).replace(tzinfo=None),
                    })
                    self._store.save_identity(updated)
                    break

        self._identity_model.rename_identity(identity_id, new_name.strip())

        # Keep the live matcher's gallery in sync when the engine is running.
        if self._identity_service is not None:
            try:
                self._identity_service.rename(identity_id, new_name.strip())
            except Exception:  # noqa: BLE001
                log.debug("identity_service.rename failed", exc_info=True)

        self.identitiesChanged.emit()
        self._enqueue(RenameIdentityCmd(
            camera_id=None, identity_id=identity_id, new_name=new_name.strip(),
        ))

    @Slot(str, str)
    def labelIdentity(self, identity_id: str, name: str) -> None:
        """Promote an unlabeled (auto-harvested) identity → named + enabled +
        persisted.  This is the single point that flips an in-memory "Person N"
        into a real, saved identity that the engine will follow.
        """
        clean = name.strip()
        if not clean:
            self.errorOccurred.emit("Identity name must not be blank")
            return

        updated: Any | None = None
        if self._identity_service is not None:
            try:
                updated = self._identity_service.label(identity_id, clean)
            except Exception:  # noqa: BLE001
                log.debug("identity_service.label failed", exc_info=True)

        if updated is None:
            # Engine not running (or service rejected) — promote the model record
            # and persist it directly so labeling works without a live engine.
            rec = self._identity_model.get(identity_id)
            if rec is None:
                return
            updated = rec.model_copy(update={
                "name": clean, "labeled": True, "enabled": True,
            })
            if self._store:
                self._store.save_identity(updated)

        self._identity_model.update_identity(updated)
        self.identitiesChanged.emit()
        # A label is a rename in the persisted-command sense for the engine side.
        self._enqueue(RenameIdentityCmd(
            camera_id=None, identity_id=identity_id, new_name=clean,
        ))

    @Slot(str, str, int)
    def registerIdentity(self, identity_id: str, name: str, thumbnail_index: int) -> None:
        """Register a recognized face: pick a profile photo + name → labeled.

        ``thumbnail_index`` selects the profile photo from the identity's
        candidate ``thumbnails`` (``-1`` keeps the current/default).  This is the
        UI's "Register" action on a recognized "Person N".
        """
        clean = name.strip()
        if not clean:
            self.errorOccurred.emit("Identity name must not be blank")
            return
        idx = thumbnail_index if thumbnail_index is not None and thumbnail_index >= 0 else None

        updated: Any | None = None
        if self._identity_service is not None:
            try:
                updated = self._identity_service.label(identity_id, clean, idx)
            except Exception:  # noqa: BLE001
                log.debug("identity_service.label(register) failed", exc_info=True)

        if updated is None:
            rec = self._identity_model.get(identity_id)
            if rec is None:
                return
            update: dict[str, Any] = {"name": clean, "labeled": True, "enabled": True}
            if idx is not None and 0 <= idx < len(getattr(rec, "thumbnails", []) or []):
                update["thumbnail"] = rec.thumbnails[idx]
            updated = rec.model_copy(update=update)
            if self._store:
                self._store.save_identity(updated)

        self._identity_model.update_identity(updated)
        self.identitiesChanged.emit()
        self._enqueue(RenameIdentityCmd(
            camera_id=None, identity_id=identity_id, new_name=clean,
        ))

    @Slot(str, int)
    def setProfileThumbnail(self, identity_id: str, index: int) -> None:
        """Choose which candidate photo is the identity's profile picture."""
        updated: Any | None = None
        if self._identity_service is not None:
            try:
                if self._identity_service.set_profile_thumbnail(identity_id, index):
                    updated = self._identity_service.get(identity_id)
            except Exception:  # noqa: BLE001
                log.debug("identity_service.set_profile_thumbnail failed", exc_info=True)

        if updated is None:
            rec = self._identity_model.get(identity_id)
            thumbs = getattr(rec, "thumbnails", []) or [] if rec else []
            if rec is None or not (0 <= index < len(thumbs)):
                return
            updated = rec.model_copy(update={"thumbnail": thumbs[index]})
            if self._store and getattr(updated, "labeled", True):
                self._store.save_identity(updated)

        self._identity_model.update_identity(updated)
        self.identitiesChanged.emit()

    @Slot(str, int)
    def removeIdentityPhoto(self, identity_id: str, index: int) -> None:
        """Delete a SINGLE captured photo (``thumbnails[index]``) from a person.

        Lets the user prune bad/odd-angle shots without deleting the whole person.
        Prefers the identity service's ``remove_thumbnail`` (which also drops the
        aligned embedding); otherwise falls back to editing the model record +
        store directly.  If the removed photo was the profile thumbnail, the next
        remaining photo becomes the profile (or ``None`` when none remain).
        """
        updated: Any | None = None
        if self._identity_service is not None:
            try:
                if self._identity_service.remove_thumbnail(identity_id, index):
                    updated = self._identity_service.get(identity_id)
            except Exception:  # noqa: BLE001
                log.debug("identity_service.remove_thumbnail failed", exc_info=True)

        if updated is None:
            rec = self._identity_model.get(identity_id)
            thumbs = list(getattr(rec, "thumbnails", []) or []) if rec else []
            if rec is None or not (0 <= index < len(thumbs)):
                return
            removed = thumbs.pop(index)
            update: dict[str, Any] = {"thumbnails": thumbs}
            if getattr(rec, "thumbnail", None) == removed:
                update["thumbnail"] = thumbs[0] if thumbs else None
            updated = rec.model_copy(update=update)
            if self._store and getattr(updated, "labeled", True):
                self._store.save_identity(updated)

        self._identity_model.update_identity(updated)
        self.identitiesChanged.emit()

    @Slot(str, str)
    def addIdentityPhoto(self, identity_id: str, data_uri: str) -> None:
        """Add a user-imported profile/gallery photo to a person (no embedding).

        Decodes a ``data:image/…;base64,…`` URI to raw PNG bytes and appends it
        to the identity's photo set via the service (falling back to the model +
        store).  Recognition still relies on the auto-gathered embeddings — this
        only curates how the person looks in the gallery / what the profile is.
        """
        import base64  # noqa: PLC0415

        try:
            b64 = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
            photo = base64.b64decode(b64)
        except Exception:  # noqa: BLE001
            log.debug("addIdentityPhoto: bad data URI", exc_info=True)
            return
        if not photo:
            return

        updated: Any | None = None
        if self._identity_service is not None:
            try:
                if self._identity_service.add_photo(identity_id, photo):
                    updated = self._identity_service.get(identity_id)
            except Exception:  # noqa: BLE001
                log.debug("identity_service.add_photo failed", exc_info=True)

        if updated is None:
            rec = self._identity_model.get(identity_id)
            if rec is None:
                return
            thumbs = (list(getattr(rec, "thumbnails", []) or []) + [photo])[-8:]
            update: dict[str, Any] = {"thumbnails": thumbs}
            if not getattr(rec, "thumbnail", None):
                update["thumbnail"] = photo
            updated = rec.model_copy(update=update)
            if self._store and getattr(updated, "labeled", True):
                self._store.save_identity(updated)

        self._identity_model.update_identity(updated)
        self.identitiesChanged.emit()

    @Slot(str, bool)
    def setIdentityEnabled(self, identity_id: str, enabled: bool) -> None:
        """Enable/disable whether the engine actively matches/follows an identity."""
        updated: Any | None = None
        if self._identity_service is not None:
            try:
                if self._identity_service.set_enabled(identity_id, bool(enabled)):
                    updated = self._identity_service.get(identity_id)
            except Exception:  # noqa: BLE001
                log.debug("identity_service.set_enabled failed", exc_info=True)

        if updated is None:
            rec = self._identity_model.get(identity_id)
            if rec is None:
                return
            updated = rec.model_copy(update={"enabled": bool(enabled)})
            # Persist only labeled identities (retention policy).
            if self._store and getattr(updated, "labeled", True):
                self._store.save_identity(updated)

        self._identity_model.update_identity(updated)
        self.identitiesChanged.emit()

    @Slot(str, str)
    def mergeIdentities(self, keep_id: str, drop_id: str) -> None:
        """Fold ``drop_id``'s embeddings + thumbnail into ``keep_id`` and drop it."""
        if keep_id == drop_id:
            return

        merged: Any | None = None
        if self._identity_service is not None:
            try:
                merged = self._identity_service.merge(keep_id, drop_id)
            except Exception:  # noqa: BLE001
                log.debug("identity_service.merge failed", exc_info=True)

        if merged is None:
            # Engine not running — merge in the model + store directly.
            keep = self._identity_model.get(keep_id)
            drop = self._identity_model.get(drop_id)
            if keep is None or drop is None:
                return
            blobs = list(keep.embeddings) + list(drop.embeddings)
            merged = keep.model_copy(update={
                "embeddings": blobs,
                "thumbnail": keep.thumbnail or drop.thumbnail,
            })
            if self._store and getattr(merged, "labeled", True):
                self._store.save_identity(merged)
                self._store.delete_identity(drop_id)

        self._identity_model.update_identity(merged)
        self._identity_model.remove_identity(drop_id)
        self.identitiesChanged.emit()

    @Slot(str, str)
    def setTargetIdentity(self, camera_id: str, identity_id: str) -> None:
        """Tell *camera_id* to follow a named identity ("track when found").

        An empty ``identity_id`` clears identity targeting on that camera.  Only
        one target per camera; this supersedes any explicit click-a-box target
        once the identity is detected on a live track.
        """
        ident = identity_id or None
        rec = self._model.get_record(camera_id)
        if rec is not None and rec.camera_config is not None:
            rec.target_track_id = None
            new_target = rec.camera_config.target.model_copy(update={
                "mode": "identity" if ident else "off",
                "identity_id": ident,
            })
            new_cfg = rec.camera_config.model_copy(update={"target": new_target})
            rec.camera_config = new_cfg
            if self._store:
                self._store.save_camera_debounced(new_cfg)
            self._model._notify_camera(camera_id, [CameraListModel.TargetTrackIdRole])
        self._enqueue(SetTargetIdentityCmd(camera_id=camera_id, identity_id=ident))
        self.targetChanged.emit(camera_id)

    @Slot(result="QVariantList")
    def registeredIdentities(self) -> list[dict[str, str]]:
        """Return the named (labeled) identities as ``[{id, name}]``.

        Used by the per-camera "track this person" pickers (Properties inspector
        and camera tile).  Auto-harvested (unlabeled) faces are excluded.
        """
        out: list[dict[str, str]] = []
        try:
            for rec in self._identity_model.get_all():
                if getattr(rec, "labeled", True):
                    out.append({"id": str(rec.id), "name": str(rec.name or "")})
        except Exception:  # noqa: BLE001
            log.debug("registeredIdentities failed", exc_info=True)
        return out

    # ── source discovery ──────────────────────────────────────────────────────

    @Slot(result="QVariantList")
    def scanUSBCameras(self) -> list[dict[str, Any]]:
        """Return USB cameras as ``[{name, uri, unique_id, in_use, source_label}]``.

        Uses the platform discovery backend
        (:func:`autoptz.engine.discovery.usb.enumerate_cameras`, which keys on
        stable device ids and labels Continuity Camera, and otherwise probes
        actually-openable VideoCapture indices).  Only **real, openable** devices
        are returned — no phantom "Camera 0-3".  Names are always plain strings.

        ``uri`` is ``usb://<index>``; ``unique_id`` is a plain ``str`` (``""``
        when none); ``in_use`` is True when a camera with the same ``unique_id``
        (or ``uri``) is already in the camera model.  The ``uri → unique_id`` map
        is cached so :meth:`addCamera` can persist the stable id.

        Degrades gracefully: if the discovery backend cannot even be imported the
        method returns ``[]`` rather than inventing phantom indices.
        """
        results: list[dict[str, Any]] = []
        try:
            from autoptz.engine.discovery.usb import enumerate_cameras  # type: ignore
            devices = enumerate_cameras()
        except Exception:  # noqa: BLE001 — import or runtime failure → fallback
            log.debug("scanUSBCameras: enumerate_cameras unavailable; using fallback")
            devices = None

        # Build the set of already-bound USB devices for de-dup (by unique_id
        # and by uri).
        active_uids: set[str] = set()
        active_uris: set[str] = set()
        for cid in self._model.camera_ids():
            rec = self._model.get_record(cid)
            if rec is None or rec.camera_config is None:
                continue
            src = rec.camera_config.source
            if getattr(src, "type", None) != "usb":
                continue
            if getattr(src, "unique_id", None):
                active_uids.add(src.unique_id)  # type: ignore[arg-type]
            if getattr(src, "address", None):
                active_uris.add(src.address)

        if devices is not None:
            self._usb_unique_ids.clear()
            self._usb_source_labels.clear()
            for dev in devices:
                index = dev.get("index", 0)
                uri = f"usb://{index}"
                unique_id = str(dev.get("unique_id") or "")
                source_label = str(dev.get("source_label") or "USB")
                # Coerce to a plain string: enumeration may hand back any object
                # if a backend misbehaves; QML must only ever see a str name.
                name = str(dev.get("name") or f"Camera {index}")
                if dev.get("is_continuity"):
                    name = f"{name} (Continuity Camera)"
                if unique_id:
                    self._usb_unique_ids[uri] = unique_id
                self._usb_source_labels[uri] = source_label
                in_use = (
                    (unique_id and unique_id in active_uids)
                    or uri in active_uris
                )
                results.append({
                    "name": name,
                    "uri": uri,
                    "unique_id": unique_id,
                    "in_use": bool(in_use),
                    "is_continuity": bool(dev.get("is_continuity")),
                    # Friendly source kind ("Built-in" / "Continuity Camera" /
                    # "External" / "USB") for the picker to show instead of the
                    # opaque usb://N uri.
                    "source_label": str(dev.get("source_label") or "USB"),
                })
            return results

        # Fallback: enumeration backend could not even be imported.  Return an
        # empty list rather than inventing phantom indices 0-3.
        return results

    @Slot(result="QVariantList")
    def scanNDISources(self) -> list[dict[str, str]]:
        """Return discovered NDI sources. Requires cyndilib; returns [] if unavailable."""
        try:
            from cyndilib.finder import Finder  # type: ignore[import]
            import time
            finder = Finder()
            finder.open()
            time.sleep(2.0)
            names = list(finder.get_source_names())
            finder.close()
            return [{"name": n, "uri": f"ndi://{n}"} for n in names]
        except Exception:
            return []

    # ── theme ─────────────────────────────────────────────────────────────────

    @Slot(str)
    def setTheme(self, mode: str) -> None:
        if mode not in ("dark", "light", "system"):
            return
        self._theme_mode = mode
        if self._store:
            self._store.set_setting("theme", {"name": mode, "accent": "#2563eb"})
        self.themeChanged.emit(mode)

    # ── diagnostics: service status + live system metrics ──────────────────────

    @Slot(result="QVariantList")
    def serviceStatus(self) -> list[dict[str, str]]:
        """Return service-availability rows for the Services panel.

        Each row is ``{key, name, state, detail}`` where ``state`` is one of
        ``ok`` / ``warn`` / ``off`` / ``running`` / ``stopped``.  Never raises.
        """
        try:
            from autoptz.engine.runtime.diagnostics import collect_services

            return collect_services(
                engine_running=self._engine_running, engine_ep=self._engine_ep,
            )
        except Exception:  # noqa: BLE001
            log.debug("serviceStatus failed", exc_info=True)
            return []

    @Slot(result="QVariant")
    def systemMetrics(self) -> dict[str, Any]:
        """Return live CPU / memory metrics (system-wide + this process).

        ``{available, cpu_percent, mem_percent, app_cpu_percent, app_rss_mb}``.
        ``available`` is False (placeholders) when psutil is not installed.
        """
        try:
            from autoptz.engine.runtime.diagnostics import system_metrics

            return system_metrics()
        except Exception:  # noqa: BLE001
            log.debug("systemMetrics failed", exc_info=True)
            return {"available": False}

    # ── logging control + export (drives the in-app log bridge) ─────────────────

    def set_log_bridge(self, model: Any | None, handler: Any | None) -> None:
        """Wire the in-app log model + handler so the log slots can drive them.

        Called from ``app.py`` after the :class:`LogListModel` and
        :class:`QtLogHandler` are constructed.  Optional — when unset the log
        slots are safe no-ops (headless tests / CLI use).
        """
        self._log_model = model
        self._log_handler = handler

    @Slot(str)
    def setLogLevel(self, level: str) -> None:
        """Adjust the root logger + handler threshold (e.g. ``"INFO"``/``"DEBUG"``).

        Accepts any stdlib level name (case-insensitive).  Raises nothing on a
        bad name — it logs a warning and leaves the level unchanged so the UI
        dropdown can never wedge the app.
        """
        name = (level or "").strip().upper()
        numeric = logging.getLevelName(name)
        if not isinstance(numeric, int):
            log.warning("setLogLevel: unknown level %r", level)
            return
        # The root logger gates which records reach handlers at all; the in-app
        # handler gates which of those land in the console model.
        logging.getLogger().setLevel(numeric)
        if self._log_handler is not None:
            try:
                self._log_handler.setLevel(numeric)
            except Exception:  # noqa: BLE001
                log.debug("setLogLevel: handler.setLevel failed", exc_info=True)
        log.info("Log level set to %s", name)

    @Slot(result=str)
    def copyLogsToClipboard(self) -> str:
        """Copy the full buffered log to the clipboard; return the copied text.

        Returns the text regardless of clipboard availability so QML can also
        use the return value (and tests can assert on it without a GUI app).
        """
        text = self._dump_logs()
        try:
            from PySide6.QtGui import QGuiApplication

            # The clipboard only exists under a QGuiApplication; touching it
            # under a bare QCoreApplication (headless / tests) crashes, so guard
            # on the running instance being a GUI app before accessing it.
            app = QGuiApplication.instance()
            if isinstance(app, QGuiApplication):
                cb = app.clipboard()
                if cb is not None:
                    cb.setText(text)
        except Exception:  # noqa: BLE001 — no GUI clipboard (headless) is fine
            log.debug("copyLogsToClipboard: clipboard unavailable", exc_info=True)
        return text

    @Slot(str, result=bool)
    def exportLogs(self, path: str) -> bool:
        """Write the full buffered log to *path*.  Returns True on success.

        A ``file://`` URL (as QML FileDialog yields) is accepted and normalised.
        """
        text = self._dump_logs()
        target = path
        if target.startswith("file://"):
            from PySide6.QtCore import QUrl

            target = QUrl(target).toLocalFile() or target[len("file://"):]
        try:
            from pathlib import Path

            Path(target).expanduser().write_text(text, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning("exportLogs: could not write %s: %s", target, exc)
            self.errorOccurred.emit(f"Could not export logs: {exc}")
            return False
        log.info("Exported %d bytes of logs to %s", len(text), target)
        return True

    def _dump_logs(self) -> str:
        """Return the full buffered log text (``""`` when no bridge is wired)."""
        if self._log_model is not None and hasattr(self._log_model, "dump_text"):
            try:
                return str(self._log_model.dump_text())
            except Exception:  # noqa: BLE001
                log.debug("_dump_logs failed", exc_info=True)
        return ""

    # ── telemetry ingest (called from engine thread) ──────────────────────────

    def push_telemetry(self, msg: TelemetryMsg) -> None:
        """Deliver a telemetry snapshot.  Thread-safe.

        Qt models are not safe to mutate off-thread, so when called from a
        worker thread the model mutation is marshalled onto the owning (GUI)
        thread via a queued signal.  When called from the owning thread (tests,
        and any direct same-thread use) it applies synchronously so callers can
        observe the update immediately without spinning the event loop.
        """
        from PySide6.QtCore import QThread

        if self.thread() is QThread.currentThread():
            self._on_telemetry_main(msg)
        else:
            self._telemetryArrived.emit(msg)

    @Slot(object)
    def _on_telemetry_main(self, msg: TelemetryMsg) -> None:
        """Apply telemetry on the owning (GUI) thread."""
        self._model.update_telemetry(msg)
        self.telemetryUpdated.emit(msg.camera_id)

    # ── identity ingest (called from engine/worker thread) ─────────────────────

    def push_identity(self, record: Any) -> None:
        """Surface a harvested/updated identity (an ``IdentityRecord``).  Thread-safe.

        Mirrors :meth:`push_telemetry`: when called from a worker thread the Qt
        model mutation is marshalled onto the owning (GUI) thread via a queued
        signal; when called from the owning thread it applies synchronously so
        tests observe the update without spinning the event loop.
        """
        from PySide6.QtCore import QThread

        if self.thread() is QThread.currentThread():
            self._on_identity_main(record)
        else:
            self._identityArrived.emit(record)

    @Slot(object)
    def _on_identity_main(self, record: Any) -> None:
        """Apply a harvested/updated identity on the owning (GUI) thread."""
        self._identity_model.add_identity(record)   # upserts by id
        self.identitiesChanged.emit()

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
