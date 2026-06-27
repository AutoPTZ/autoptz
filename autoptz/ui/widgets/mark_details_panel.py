"""MarkDetailsPanel — per-stream resolution/fps/source/people/stage-ms for a tile.

Shows live stats for the camera the operator clicked on the Mark wall:
resolution, fps, source type, tracked-people count, and per-stage milliseconds.
It reuses :func:`CameraInfoPanel`'s ``_stage_text`` helper so the stage row reads
exactly like the main app (handling the ``StageTimingInfo`` list shape, idle /
warming / stale states).  Robust to telemetry being absent (warm-up): the panel
shows an empty hint until a camera is set and never raises on an unknown id.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from autoptz.ui import theme as T
from autoptz.ui.widgets.camera_info_panel import _stage_text


class MarkDetailsPanel(QWidget):
    """Per-stream stats for the selected Mark tile."""

    _EMPTY = "No camera selected — click a tile to view live stats."
    _STAGES = (
        ("ingest", "Ingest", "ingest_ms"),
        ("detect", "Detect", "detect_ms"),
        ("track", "Track", "track_ms"),
        ("face", "Face", "face_ms"),
    )

    def __init__(self, client: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("markDetailsPanel")
        self._client = client
        self._camera_id: str = ""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # An always-visible section header (global #detailsHeader rule styles it).
        self._header = QLabel("CAMERA DETAILS")
        self._header.setObjectName("detailsHeader")
        outer.addWidget(self._header)
        form = QFormLayout()
        outer.addLayout(form)
        self._empty = QLabel(self._EMPTY)
        self._empty.setWordWrap(True)
        form.addRow(self._empty)
        self._vals: dict[str, QLabel] = {}
        rows: list[tuple[str, str]] = [
            ("resolution", "Resolution"),
            ("fps", "FPS"),
            ("source", "Source"),
            ("people", "People"),
        ]
        rows += [(key, label) for key, label, _attr in self._STAGES]
        for key, label in rows:
            v = QLabel("—")
            self._vals[key] = v
            form.addRow(label, v)
        self._set_rows_visible(False)
        # 1 Hz fallback refresh; also refresh on telemetry where available.
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        connect = getattr(client, "telemetryUpdated", None)
        if connect is not None:
            try:
                client.telemetryUpdated.connect(self._on_telemetry)
            except Exception:  # noqa: BLE001
                pass
        # Per-widget restyle fallback so a Light/Dark flip repaints the idle hint
        # (re-run on theme change via common.on_theme_changed in MarkWindow).
        self._restyle()

    def _restyle(self) -> None:
        """Refresh the idle hint color from the active palette (re-run on theme flip)."""
        self._empty.setStyleSheet(f"color: {T.CURRENT.subtext};")

    def _set_rows_visible(self, visible: bool) -> None:
        self._empty.setVisible(not visible)
        for v in self._vals.values():
            v.setVisible(visible)

    def set_camera(self, camera_id: str) -> None:
        self._camera_id = camera_id or ""
        self.refresh()

    def _on_telemetry(self, camera_id: str) -> None:
        if camera_id == self._camera_id:
            self.refresh()

    def refresh(self) -> None:
        cid = self._camera_id
        if not cid:
            self._set_rows_visible(False)
            return
        rec = None
        try:
            rec = self._client.cameraModel.get_record(cid)
        except Exception:  # noqa: BLE001
            rec = None
        if rec is None:
            self._set_rows_visible(False)
            return
        self._set_rows_visible(True)
        tel = getattr(rec, "telemetry", None)
        src = getattr(getattr(rec, "camera_config", None), "source", None)
        self._vals["source"].setText(getattr(src, "type", "—") or "—")
        if tel is not None:
            w = getattr(tel, "width", 0)
            h = getattr(tel, "height", 0)
            self._vals["resolution"].setText(f"{w}×{h}" if w and h else "—")
            self._vals["fps"].setText(f"{float(getattr(tel, 'fps', 0.0) or 0.0):.1f}")
            self._vals["people"].setText(str(len(getattr(tel, "tracks", []) or [])))
            for key, _label, attr in self._STAGES:
                text, _color = _stage_text(tel, key, attr)
                self._vals[key].setText(text)
        else:
            for key in ("resolution", "fps", "people"):
                self._vals[key].setText("—")
            for key, _label, _attr in self._STAGES:
                self._vals[key].setText("—")
