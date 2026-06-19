"""CameraInfoPanel — live per-camera stats, reactive off telemetry.

The old QML panel froze because it rendered a computed-array snapshot.  This one
holds named value labels and updates them imperatively whenever
``telemetryUpdated`` fires for the shown camera (plus a slow fallback timer and
on ``configChanged``/``engineStateChanged``), so it actually shows live data.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.common import HelpBadge, on_theme_changed, section_label

log = logging.getLogger(__name__)


class CameraInfoPanel(QWidget):
    """Spec-sheet + live stats for the selected camera."""

    def __init__(self, client: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._camera_id = ""
        self._vals: dict[str, QLabel] = {}
        self._keys: list[QLabel] = []
        # Remember the semantic color requested per value so a theme flip can
        # re-resolve palette-driven defaults without losing status colors.
        self._val_colors: dict[str, str | None] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        root.addWidget(scroll)

        body = QWidget()
        self._col = QVBoxLayout(body)
        self._col.setContentsMargins(12, 12, 12, 12)
        self._col.setSpacing(6)
        scroll.setWidget(body)

        self._empty = QLabel("Select a camera to view its info")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._col.addWidget(self._empty)

        self._groups = QWidget()
        gcol = QVBoxLayout(self._groups)
        gcol.setContentsMargins(0, 0, 0, 0)
        gcol.setSpacing(10)
        self._build_groups(gcol)
        self._col.addWidget(self._groups)
        self._col.addStretch(1)
        self._groups.setVisible(False)

        self._restyle()
        on_theme_changed(client, self._restyle)
        _connect(client, "telemetryUpdated", self._on_telemetry)
        _connect(client, "configChanged", self._on_config_changed)
        _connect(client, "engineStateChanged", self.refresh)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(1000)

    def _restyle(self) -> None:
        """Re-apply literal-color styling (construction + theme change).

        Field *keys* are secondary text and field *values* default to primary
        text — both read from the active palette so they stay legible in light
        and dark (no washed-out hardcoded gray).
        """
        self._empty.setStyleSheet(f"color: {T.CURRENT.muted};")
        for k in self._keys:
            k.setStyleSheet(f"color: {T.CURRENT.subtext};")
        for key, lab in self._vals.items():
            self._paint(lab, self._val_colors.get(key))
        # Re-resolve palette-derived value colors (subtext/muted) for the live
        # camera so they don't linger from the previous appearance.
        if self._camera_id:
            self.refresh()

    # ── construction ─────────────────────────────────────────────────────────────

    _GROUPS: list[tuple[str, list[str]]] = [
        ("Tracking", ["Following", "Match", "Lock", "People in view"]),
        ("Identity", ["Display name", "Source", "Address"]),
        ("Stream", ["Resolution", "Live fps", "Health", "Dropped frames"]),
        ("Performance", ["Ingest", "Detect", "Track", "Face", "Latency", "Load"]),
        ("PTZ", ["Backend"]),
        ("Detection", ["Quality floor", "Tracker", "Detect interval"]),
        ("Engine", ["Inference EP"]),
    ]

    # Plain-English help for each group, shown via a "?" badge on its header so
    # the live numbers are self-explanatory.
    _GROUP_HELP: dict[str, str] = {
        "Tracking": "Who the camera is following right now: the locked person (or "
                    "“no target”), the match confidence, lock state, and how many "
                    "people are currently in view.",
        "Identity": "The camera's display name and where its video comes from "
                    "(source type and address; credentials are hidden).",
        "Stream": "Live video health: current resolution, measured frames per "
                  "second, an overall health flag, and how many frames have been "
                  "dropped.",
        "Performance": "Per-stage processing time — ingest, detection, and "
                       "tracking — plus end-to-end latency and an overall Load "
                       "estimate (Light / Medium / Heavy vs the frame budget).",
        "PTZ": "The pan/tilt/zoom backend in use for this camera (NDI, ONVIF, "
               "VISCA-IP/USB, or auto).",
        "Detection": "The detection settings in effect: quality floor, the "
                     "tracking algorithm, and how often the detector runs (every "
                     "N frames).",
        "Engine": "The ONNX Runtime execution provider doing inference (e.g. "
                  "CoreML / CUDA / CPU).",
    }

    def _build_groups(self, col: QVBoxLayout) -> None:
        for title, keys in self._GROUPS:
            head = QHBoxLayout()
            head.setContentsMargins(0, 0, 0, 0)
            head.setSpacing(6)
            head.addWidget(section_label(title))
            tip = self._GROUP_HELP.get(title)
            if tip:
                head.addWidget(HelpBadge(tip))
            head.addStretch(1)
            col.addLayout(head)
            form = QFormLayout()
            form.setContentsMargins(6, 0, 6, 0)
            form.setHorizontalSpacing(16)
            form.setVerticalSpacing(4)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            for key in keys:
                k = QLabel(key)
                self._keys.append(k)
                v = QLabel("—")
                v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                v.setWordWrap(True)
                self._vals[key] = v
                self._val_colors[key] = None
                form.addRow(k, v)
            holder = QWidget()
            holder.setLayout(form)
            col.addWidget(holder)

    # ── public ─────────────────────────────────────────────────────────────────

    def set_camera(self, camera_id: str) -> None:
        self._camera_id = camera_id or ""
        self._empty.setVisible(not self._camera_id)
        self._groups.setVisible(bool(self._camera_id))
        self.refresh()

    # ── reactive hooks ───────────────────────────────────────────────────────────

    def _on_telemetry(self, camera_id: str) -> None:
        if camera_id == self._camera_id:
            self.refresh()

    def _on_config_changed(self, camera_id: str) -> None:
        if camera_id == self._camera_id:
            self.refresh()

    # ── refresh ──────────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        if not self._camera_id:
            return
        running = bool(_safe(lambda: self._client.engineRunning, False))
        cfg = _safe(lambda: self._client.getCameraConfig(self._camera_id), {}) or {}
        rec = _safe(lambda: self._client.cameraModel.get_record(self._camera_id), None)

        src = cfg.get("source", {}) or {}
        track_cfg = cfg.get("tracking", {}) or {}
        ptz_cfg = cfg.get("ptz", {}) or {}

        # Identity / config
        self._set("Display name", cfg.get("name", "—"))
        # Prefer the friendly source kind ("Built-in"/"Continuity Camera"/…) over
        # the bare "usb" type; fall back to the type when no label was captured.
        self._set("Source", src.get("source_label") or src.get("type", "—"))
        # For USB the raw address is the opaque usb://N index — show the stable
        # device id instead when we have one; otherwise the sanitized address.
        usb_with_uid = src.get("type") == "usb" and src.get("unique_id")
        self._set(
            "Address",
            str(src.get("unique_id")) if usb_with_uid else _sanitize(src.get("address", "")),
        )
        self._set("Quality floor", track_cfg.get("quality_floor", "—"))
        self._set("Tracker", track_cfg.get("tracker", "—"))
        self._set("Detect interval", str(track_cfg.get("detect_interval", "—")))
        self._set("Backend", ptz_cfg.get("backend", "auto"))

        if not running or rec is None:
            for k in ("Following", "Match", "Lock", "People in view", "Resolution",
                      "Live fps", "Health", "Dropped frames", "Ingest", "Detect",
                      "Track", "Face", "Latency", "Load", "Inference EP"):
                self._set(k, "engine stopped" if not running else "—",
                          color=T.CURRENT.muted)
            return

        tel = getattr(rec, "telemetry", None)
        streaming = bool(getattr(rec, "streaming", False))
        tracks = _tracks(rec)
        target = next((t for t in tracks if t.get("is_target")), None)

        self._set("Following",
                  (target.get("identity") or f"ID {target.get('track_id')}") if target else "no target",
                  color=T.TRACKING if target else T.CURRENT.subtext)
        conf = target.get("confidence") if target else None
        self._set("Match", f"{round(conf * 100)}%" if isinstance(conf, (int, float)) and conf > 0 else "—")
        self._set("Lock", "Locked" if target else "Searching",
                  color=T.TRACKING if target else T.CURRENT.subtext)
        self._set("People in view", str(len(tracks)))

        self._set("Resolution", getattr(rec, "resolution", "") or ("waiting for video…" if not streaming else "—"))
        fps = float(getattr(rec, "fps", 0.0) or 0.0)
        self._set("Live fps", f"{fps:.1f} fps" if fps > 0 else ("waiting…" if not streaming else "0 fps"))
        health = str(getattr(rec, "health", "ok"))
        self._set("Health", health,
                  color=T.TRACKING if health == "ok" else T.ERROR if health == "error" else T.WARNING)
        self._set("Dropped frames", str(getattr(rec, "dropped_frames", 0)))

        # Performance — per-stage latency (Phase 4 telemetry); falls back to "—".
        self._set("Ingest", _ms(tel, "ingest_ms"))
        self._set("Detect", _ms(tel, "detect_ms"))
        self._set("Track", _ms(tel, "track_ms"))
        self._set("Face", _ms(tel, "face_ms"))
        lat = int(getattr(rec, "latency_ms", 0) or 0)
        self._set("Latency", f"{lat} ms" if lat > 0 else "—")
        self._set("Load", *_load_estimate(lat, float(src.get("fps", 30.0) or 30.0)))

        ep = (_safe(lambda: self._client.engineEp, "") or "—").replace("ExecutionProvider", "")
        self._set("Inference EP", ep)

    def _set(self, key: str, value: str, color: str | None = None) -> None:
        lab = self._vals.get(key)
        if lab is None:
            return
        lab.setText(str(value))
        self._val_colors[key] = color
        self._paint(lab, color)

    @staticmethod
    def _paint(lab: QLabel, color: str | None) -> None:
        """Color a value label, defaulting to the active palette's primary text."""
        lab.setStyleSheet(f"color: {color or T.CURRENT.text};")


# ── helpers ─────────────────────────────────────────────────────────────────────


def _ms(tel: Any, attr: str) -> str:
    v = getattr(tel, attr, None) if tel is not None else None
    if isinstance(v, (int, float)) and v >= 0:
        return f"{v:.1f} ms"
    return "—"


def _load_estimate(latency_ms: int, target_fps: float) -> tuple[str, str]:
    if latency_ms <= 0 or target_fps <= 0:
        return "—", T.CURRENT.muted
    budget = 1000.0 / target_fps
    ratio = latency_ms / budget
    if ratio < 0.5:
        return "Light", T.TRACKING
    if ratio < 0.85:
        return "Medium", T.WARNING
    return "Heavy", T.ERROR


def _sanitize(addr: str) -> str:
    if not addr:
        return "—"
    return re.sub(r"(\w+://)[^@/]*@", r"\1", str(addr))


def _tracks(rec: Any) -> list[dict[str, Any]]:
    try:
        return rec.tracks_as_list()
    except Exception:  # noqa: BLE001
        return []


def _connect(obj: Any, name: str, slot: Any) -> None:
    try:
        getattr(obj, name).connect(slot)
    except Exception:  # noqa: BLE001
        log.debug("connect %s failed", name, exc_info=True)


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default
