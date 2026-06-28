"""Qt list models + per-camera record for the UI.

``CameraRecord`` holds one camera's live UI state (telemetry, config snapshot,
target, events). ``CameraListModel`` / ``IdentityListModel`` / ``LayoutListModel``
are the ``QAbstractListModel``s the camera wall, People view, and Layout manager
bind to. Extracted from ``engine_client`` so the bridge module stays focused on
command/telemetry routing; ``engine_client`` re-exports these names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import (
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
    Slot,
)

from autoptz.engine.runtime.messages import TelemetryMsg

log = logging.getLogger(__name__)


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
    camera_config: Any | None = None  # autoptz.config.models.CameraConfig
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

    # ── Phase 0 per-source delivery telemetry ─────────────────────────────────

    @property
    def frames_delivered(self) -> int:
        """Cumulative frames the receiver actually delivered (0 until known)."""
        if not self.telemetry:
            return 0
        return int(getattr(self.telemetry, "frames_delivered", 0))

    @property
    def frames_dropped_est(self) -> int:
        """Estimated frames the source produced but the receiver missed."""
        if not self.telemetry:
            return 0
        return int(getattr(self.telemetry, "frames_dropped_est", 0))

    @property
    def delivered_fps(self) -> float:
        """Receiver-side delivered frame rate (0.0 until known)."""
        if not self.telemetry:
            return 0.0
        return float(getattr(self.telemetry, "delivered_fps", 0.0))

    @property
    def source_fps(self) -> float:
        """Source-advertised frame rate (0.0 until known)."""
        if not self.telemetry:
            return 0.0
        return float(getattr(self.telemetry, "source_fps", 0.0))

    @property
    def ndi_queue_depth(self) -> int:
        """Pending receive-queue depth; -1 when the source exposes no queue."""
        if not self.telemetry:
            return -1
        return int(getattr(self.telemetry, "ndi_queue_depth", -1))

    # ── Phase 0 end-to-end latency decomposition ──────────────────────────────

    @property
    def end_to_end_ms(self) -> float:
        """Capture→actuation control dead time (0.0 until the probe runs)."""
        if not self.telemetry:
            return 0.0
        return float(getattr(self.telemetry, "end_to_end_ms", 0.0))

    @property
    def capture_age_ms(self) -> float:
        """Age of the driving frame at command time (0.0 until known)."""
        if not self.telemetry:
            return 0.0
        return float(getattr(self.telemetry, "capture_age_ms", 0.0))

    @property
    def command_send_ms(self) -> float:
        """PTZ transport send wall time (0.0 until known)."""
        if not self.telemetry:
            return 0.0
        return float(getattr(self.telemetry, "command_send_ms", 0.0))

    @property
    def actuation_estimate_ms(self) -> float:
        """Configured camera-motor actuation lag estimate (0.0 until known)."""
        if not self.telemetry:
            return 0.0
        return float(getattr(self.telemetry, "actuation_estimate_ms", 0.0))

    def _crop_norm(self) -> tuple[float, float, float, float] | None:
        """Active digital-crop mapping params ``(cx, cy, cw, ch)`` in full-frame
        pixels when Center Stage is driving the preview, else None.

        When present, overlay coords normalized to the FULL frame must be
        re-normalized into the CROPPED preview: a full-frame pixel ``px`` maps to
        ``(px - c0) / clen`` along each axis. None ⇒ keep full-frame normalization
        (legacy behavior, byte-identical).
        """
        if not self.telemetry:
            return None
        rect = getattr(self.telemetry, "digital_crop_rect", None)
        if not rect:
            return None
        cx, cy, cw, ch = rect
        if cw <= 0 or ch <= 0:
            return None
        return (float(cx), float(cy), float(cw), float(ch))

    def tracks_as_list(self) -> list[dict[str, Any]]:
        if not self.telemetry:
            return []
        # Detector produces pixel-space coords; the UI overlays expect normalized 0–1.
        w = max(1.0, float(self.telemetry.width or 1))
        h = max(1.0, float(self.telemetry.height or 1))
        crop = self._crop_norm()
        if crop is not None:
            cx, cy, cw, ch = crop

            def nx(px: float) -> float:
                return max(0.0, min(1.0, (px - cx) / cw))

            def ny(px: float) -> float:
                return max(0.0, min(1.0, (px - cy) / ch))

            def sx(dpx: float) -> float:  # delta (velocity): scale, no clamp
                return dpx / cw

            def sy(dpx: float) -> float:
                return dpx / ch
        else:

            def nx(px: float) -> float:
                return px / w

            def ny(px: float) -> float:
                return px / h

            def sx(dpx: float) -> float:
                return dpx / w

            def sy(dpx: float) -> float:
                return dpx / h

        result = []
        for t in self.telemetry.tracks:
            result.append(
                {
                    "track_id": t.track_id,
                    "bbox": {
                        "x1": nx(t.bbox.x1),
                        "y1": ny(t.bbox.y1),
                        "x2": nx(t.bbox.x2),
                        "y2": ny(t.bbox.y2),
                    },
                    "identity": t.identity or "",
                    "identity_id": t.identity_id or "",
                    "confidence": t.confidence,
                    "is_target": t.is_target,
                    "lost": bool(getattr(t, "lost", False)),
                    # Velocity normalized to frame fractions/frame for overlay drawing.
                    "vx": sx(getattr(t, "vx", 0.0)),
                    "vy": sy(getattr(t, "vy", 0.0)),
                    "aim": (
                        {
                            # Aim is clamped to [0,1] in BOTH branches (legacy clamped
                            # aim but not bbox); the crop nx/ny already clamp.
                            "x": (
                                nx(float(t.aim_x))
                                if crop is not None
                                else max(0.0, min(1.0, float(t.aim_x) / w))
                            ),
                            "y": (
                                ny(float(t.aim_y))
                                if crop is not None
                                else max(0.0, min(1.0, float(t.aim_y) / h))
                            ),
                            "source": t.aim_source or "",
                        }
                        if getattr(t, "aim_x", None) is not None
                        and getattr(t, "aim_y", None) is not None
                        else None
                    ),
                }
            )
        return result

    def faces_as_list(self) -> list[dict[str, Any]]:
        """Detected faces with normalized (0–1) bboxes for the face overlay."""
        if not self.telemetry:
            return []
        w = max(1.0, float(self.telemetry.width or 1))
        h = max(1.0, float(self.telemetry.height or 1))
        crop = self._crop_norm()
        if crop is not None:
            cx, cy, cw, ch = crop

            def nx(px: float) -> float:
                return max(0.0, min(1.0, (px - cx) / cw))

            def ny(px: float) -> float:
                return max(0.0, min(1.0, (px - cy) / ch))
        else:

            def nx(px: float) -> float:
                return px / w

            def ny(px: float) -> float:
                return px / h

        return [
            {
                "bbox": {
                    "x1": nx(f.bbox.x1),
                    "y1": ny(f.bbox.y1),
                    "x2": nx(f.bbox.x2),
                    "y2": ny(f.bbox.y2),
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
        crop = self._crop_norm()
        if crop is not None:
            cx, cy, cw, ch = crop
            return [
                {
                    "x": max(0.0, min(1.0, (k.x - cx) / cw)),
                    "y": max(0.0, min(1.0, (k.y - cy) / ch)),
                    "conf": k.conf,
                }
                for k in getattr(self.telemetry, "pose", []) or []
            ]
        return [
            {"x": k.x / w, "y": k.y / h, "conf": k.conf}
            for k in getattr(self.telemetry, "pose", []) or []
        ]

    def ptz_as_dict(self) -> dict[str, Any]:
        if not self.telemetry:
            return {"pan": 0.0, "tilt": 0.0, "zoom": 0.0, "moving": False, "state": "idle"}
        p = self.telemetry.ptz
        return {"pan": p.pan, "tilt": p.tilt, "zoom": p.zoom, "moving": p.moving, "state": p.state}

    def tracking_status_as_dict(self) -> dict[str, Any]:
        if not self.telemetry:
            return {
                "state": "idle",
                "headline": "",
                "detail": "",
                "action": "",
                "remaining_s": 0.0,
                "severity": "info",
            }
        s = self.telemetry.tracking_status
        return {
            "state": s.state,
            "headline": s.headline,
            "detail": s.detail,
            "action": s.action,
            "remaining_s": s.remaining_s,
            "severity": s.severity,
        }

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
    """Ordered list of CameraRecords exposed to the UI."""

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
    PresetsRole = Qt.ItemDataRole.UserRole + 12
    ResolutionRole = Qt.ItemDataRole.UserRole + 13
    DroppedFramesRole = Qt.ItemDataRole.UserRole + 14
    LatencyMsRole = Qt.ItemDataRole.UserRole + 15
    StreamingRole = Qt.ItemDataRole.UserRole + 16
    TrackingStatusRole = Qt.ItemDataRole.UserRole + 17

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
            self.PresetsRole: QByteArray(b"presets"),
            self.ResolutionRole: QByteArray(b"resolution"),
            self.DroppedFramesRole: QByteArray(b"droppedFrames"),
            self.LatencyMsRole: QByteArray(b"latencyMs"),
            self.StreamingRole: QByteArray(b"streaming"),
            self.TrackingStatusRole: QByteArray(b"trackingStatus"),
        }

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
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
            case self.PresetsRole:
                return rec.presets_as_list()
            case self.ResolutionRole:
                return rec.resolution
            case self.DroppedFramesRole:
                return rec.dropped_frames
            case self.LatencyMsRole:
                return rec.latency_ms
            case self.StreamingRole:
                return rec.streaming
            case self.TrackingStatusRole:
                return rec.tracking_status_as_dict()
        return None

    def setData(
        self,
        index: QModelIndex | QPersistentModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
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
        self.dataChanged.emit(
            idx,
            idx,
            [
                self.FpsRole,
                self.TracksRole,
                self.PtzStateRole,
                self.HealthRole,
                self.ResolutionRole,
                self.DroppedFramesRole,
                self.LatencyMsRole,
                self.StreamingRole,
                self.TrackingStatusRole,
            ],
        )

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

    FROZEN roles (the UI binds to these exact names):
      ``identityId`` (str), ``identityName`` (str),
      ``thumbnail`` (str — a ``data:image/png;base64,…`` URI or ``""``),
      ``enabled`` (bool), ``labeled`` (bool; false = auto-harvested/unlabeled).
    """

    IdRole = Qt.ItemDataRole.UserRole + 1
    NameRole = Qt.ItemDataRole.UserRole + 2
    ThumbnailRole = Qt.ItemDataRole.UserRole + 3
    EnabledRole = Qt.ItemDataRole.UserRole + 4
    LabeledRole = Qt.ItemDataRole.UserRole + 5
    ThumbnailsRole = Qt.ItemDataRole.UserRole + 6  # list of candidate photo URIs

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._records: list[Any] = []  # list[IdentityRecord]

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self._records)

    def roleNames(self) -> dict[int, QByteArray]:
        return {
            self.IdRole: QByteArray(b"identityId"),
            self.NameRole: QByteArray(b"identityName"),
            self.ThumbnailRole: QByteArray(b"thumbnail"),
            self.EnabledRole: QByteArray(b"enabled"),
            self.LabeledRole: QByteArray(b"labeled"),
            self.ThumbnailsRole: QByteArray(b"thumbnails"),
        }

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if not index.isValid() or index.row() >= len(self._records):
            return None
        rec = self._records[index.row()]
        match role:
            case self.IdRole:
                return rec.id
            case self.NameRole:
                return rec.name
            case self.ThumbnailRole:
                return _thumbnail_data_uri(getattr(rec, "thumbnail", None))
            case self.EnabledRole:
                return bool(getattr(rec, "enabled", True))
            case self.LabeledRole:
                return bool(getattr(rec, "labeled", True))
            case self.ThumbnailsRole:
                return _thumbnails_data_uris(getattr(rec, "thumbnails", None))
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
            self.dataChanged.emit(
                idx,
                idx,
                [
                    self.NameRole,
                    self.ThumbnailRole,
                    self.EnabledRole,
                    self.LabeledRole,
                ],
            )
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
        self.dataChanged.emit(
            idx,
            idx,
            [
                self.NameRole,
                self.ThumbnailRole,
                self.EnabledRole,
                self.LabeledRole,
                self.ThumbnailsRole,
            ],
        )

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

    IdRole = Qt.ItemDataRole.UserRole + 1
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
            self.IdRole: QByteArray(b"layoutId"),
            self.NameRole: QByteArray(b"layoutName"),
        }

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if not index.isValid() or index.row() >= len(self._layouts):
            return None
        lo = self._layouts[index.row()]
        match role:
            case self.IdRole:
                return lo.id
            case self.NameRole:
                return lo.name
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
