"""CameraInfoPanel — live per-camera stats, reactive off telemetry.

An earlier reactive panel froze because it rendered a computed-array snapshot.  This one
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
        self.setObjectName("cameraInfoPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._client = client
        self._camera_id = ""
        self._vals: dict[str, QLabel] = {}
        self._keys: list[QLabel] = []
        # key → its caption QLabel, so conditional rows can be hidden as a pair.
        self._key_labels: dict[str, QLabel] = {}
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
        (
            "Stream",
            [
                "Resolution",
                "Live fps",
                "Health",
                "Dropped frames",
                "Est. source drops",
                "Delivered / source",
                "NDI queue depth",
            ],
        ),
        (
            "Runtime",
            [
                "Target fps",
                "Frame budget",
                "Runtime load",
                "Setting cost",
                "Quality floor",
                "Quality active",
                "Quality reason",
            ],
        ),
        (
            "Models",
            [
                "Detector model",
                "Detector tier",
                "Model switch",
                "Tracker backend",
                "Tracker switch",
            ],
        ),
        (
            "Services",
            [
                "Detector service",
                "Tracker service",
                "ReID",
                "Face",
                "Pose",
                "Framing",
            ],
        ),
        (
            "Performance",
            [
                "Ingest",
                "Detector stage",
                "Tracker stage",
                "Face stage",
                "Pose stage",
                "Latency",
                "End-to-end latency",
            ],
        ),
        ("PTZ", ["Backend"]),
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
        "Runtime": "Effective runtime choices: FPS budget, load against that "
        "budget, active quality level, and the reason for the current "
        "auto/manual setting.",
        "Models": "The shared detector model and per-camera tracker backend in "
        "effect, including any current or recent switch.",
        "Services": "Enabled/active service state for this selected camera. "
        "Disabled or stale services keep their last-known timing "
        "ghosted instead of pretending they are 0 ms.",
        "Performance": "Per-stage processing time. Rows show last + rolling "
        "average by default, with Disabled/Stale labels when a "
        "service is off or no longer fresh.",
        "PTZ": "The pan/tilt/zoom backend in use for this camera (NDI, ONVIF, "
        "VISCA-IP/USB, or auto).",
        "Detection": "The detection settings in effect: quality floor, the "
        "tracking algorithm, and how often the detector runs (every "
        "N frames).",
        "Engine": "The ONNX Runtime execution provider doing inference (e.g. CoreML / CUDA / CPU).",
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
                self._key_labels[key] = k
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
        self._set("Backend", ptz_cfg.get("backend", "auto"))
        self._set("Tracker backend", track_cfg.get("tracker", "—"))

        if not running or rec is None:
            for k in self._vals:
                if k in {"Display name", "Source", "Address", "Backend", "Tracker backend"}:
                    continue
                self._set(k, "engine stopped" if not running else "—", color=T.CURRENT.muted)
            return

        tel = getattr(rec, "telemetry", None)
        streaming = bool(getattr(rec, "streaming", False))
        tracks = _tracks(rec)
        target = next((t for t in tracks if t.get("is_target")), None)

        self._set(
            "Following",
            (target.get("identity") or f"ID {target.get('track_id')}") if target else "no target",
            color=T.TRACKING if target else T.CURRENT.subtext,
        )
        conf = target.get("confidence") if target else None
        self._set(
            "Match", f"{round(conf * 100)}%" if isinstance(conf, int | float) and conf > 0 else "—"
        )
        status_headline = str(getattr(getattr(tel, "tracking_status", None), "headline", "") or "")
        self._set(
            "Lock",
            status_headline or ("Locked" if target else "Searching"),
            color=T.TRACKING if target else T.WARNING if status_headline else T.CURRENT.subtext,
        )
        self._set("People in view", str(len(tracks)))

        self._set(
            "Resolution",
            getattr(rec, "resolution", "") or ("waiting for video…" if not streaming else "—"),
        )
        fps = float(getattr(rec, "fps", 0.0) or 0.0)
        self._set(
            "Live fps", f"{fps:.1f} fps" if fps > 0 else ("waiting…" if not streaming else "0 fps")
        )
        health = str(getattr(rec, "health", "ok"))
        self._set(
            "Health",
            health,
            color=T.TRACKING if health == "ok" else T.ERROR if health == "error" else T.WARNING,
        )
        self._set("Dropped frames", str(getattr(rec, "dropped_frames", 0)))

        # Phase 0a per-source delivery telemetry.
        drops_est = int(getattr(rec, "frames_dropped_est", 0) or 0)
        self._set(
            "Est. source drops",
            str(drops_est),
            color=T.WARNING if drops_est > 0 else None,
        )
        src_fps = float(getattr(rec, "source_fps", 0.0) or 0.0)
        deliv_fps = float(getattr(rec, "delivered_fps", 0.0) or 0.0)
        show_rates = src_fps > 0
        self._set_row_visible("Delivered / source", show_rates)
        if show_rates:
            self._set("Delivered / source", f"{deliv_fps:.1f} / {src_fps:.1f} fps")
        # -1 == the source exposes no queue; omit the row entirely so non-NDI /
        # no-queue sources don't show a confusing "-1".
        queue_depth = int(getattr(rec, "ndi_queue_depth", -1))
        show_queue = queue_depth >= 0
        self._set_row_visible("NDI queue depth", show_queue)
        if show_queue:
            self._set("NDI queue depth", str(queue_depth))

        target_fps = float(getattr(tel, "target_fps", 0.0) or src.get("fps", 30.0) or 30.0)
        budget_ms = float(getattr(tel, "frame_budget_ms", 0.0) or (1000.0 / max(1.0, target_fps)))
        self._set("Target fps", f"{target_fps:.0f} fps")
        self._set("Frame budget", f"{budget_ms:.1f} ms/frame")
        self._set("Runtime load", *_runtime_load(tel, budget_ms))
        self._set("Setting cost", _setting_cost(tel), color=T.CURRENT.subtext)

        qs = getattr(tel, "quality_state", None)
        floor = _field(qs, "floor", track_cfg.get("quality_floor", "—"))
        active = _field(qs, "active", "—")
        self._set("Quality floor", floor)
        # Surface the effective detector cadence alongside the active level, and
        # flag (amber) when "auto" is actively scaling away from the floor — so
        # the operator can SEE the system adapting rather than guessing.
        active_txt = str(active)
        interval = _field(qs, "detect_interval", None)
        try:
            n = int(interval)
            active_txt += f" · detect every {n} frame{'s' if n != 1 else ''}"
        except (TypeError, ValueError):
            pass
        diverged = str(floor) == "auto" and str(active) not in ("auto", "—", "")
        self._set("Quality active", active_txt, color=T.WARNING if diverged else None)
        self._set("Quality reason", _field(qs, "reason", "—"), color=T.CURRENT.subtext)
        self._set("Detector model", _field(qs, "detector_model", "—") or "—")
        self._set("Detector tier", _field(qs, "detector_tier", "—") or "—")
        self._set(
            "Model switch",
            _switch_text(getattr(tel, "model_switch", None)),
            color=_switch_color(getattr(tel, "model_switch", None)),
        )
        self._set(
            "Tracker switch",
            _switch_text(getattr(tel, "tracker_switch", None)),
            color=_switch_color(getattr(tel, "tracker_switch", None)),
        )

        for key, label in (
            ("detector", "Detector service"),
            ("tracker", "Tracker service"),
            ("reid", "ReID"),
            ("face", "Face"),
            ("pose", "Pose"),
            ("framing", "Framing"),
        ):
            text, color = _service_text(tel, key)
            self._set(label, text, color=color)

        self._set("Ingest", *_stage_text(tel, "ingest", "ingest_ms"))
        self._set("Detector stage", *_stage_text(tel, "detect", "detect_ms"))
        self._set("Tracker stage", *_stage_text(tel, "track", "track_ms"))
        self._set("Face stage", *_stage_text(tel, "face", "face_ms"))
        self._set("Pose stage", *_stage_text(tel, "pose", "pose_ms"))
        lat = int(getattr(rec, "latency_ms", 0) or 0)
        self._set("Latency", f"{lat} ms" if lat > 0 else "—")
        # Phase 0b end-to-end control dead time — 0.0 until the probe runs, so
        # hide the row entirely until there's a real measurement.
        e2e = float(getattr(rec, "end_to_end_ms", 0.0) or 0.0)
        show_e2e = e2e > 0
        self._set_row_visible("End-to-end latency", show_e2e)
        if show_e2e:
            self._set("End-to-end latency", f"{e2e:.0f} ms")

        ep = (_safe(lambda: self._client.engineEp, "") or "—").replace("ExecutionProvider", "")
        self._set("Inference EP", ep)

    def _set(self, key: str, value: str, color: str | None = None) -> None:
        lab = self._vals.get(key)
        if lab is None:
            return
        lab.setText(str(value))
        self._val_colors[key] = color
        self._paint(lab, color)

    def _set_row_visible(self, key: str, visible: bool) -> None:
        """Show/hide a whole form row (its caption + value) as a pair.

        Used for rows that would otherwise show a confusing sentinel (e.g. the
        ``-1`` "no queue" NDI depth) or pure noise (zero/undriven values).
        """
        val = self._vals.get(key)
        cap = self._key_labels.get(key)
        if val is not None:
            val.setVisible(visible)
        if cap is not None:
            cap.setVisible(visible)

    @staticmethod
    def _paint(lab: QLabel, color: str | None) -> None:
        """Color a value label, defaulting to the active palette's primary text."""
        lab.setStyleSheet(f"color: {color or T.CURRENT.text};")


# ── helpers ─────────────────────────────────────────────────────────────────────


def _ms(tel: Any, attr: str) -> str:
    v = getattr(tel, attr, None) if tel is not None else None
    if isinstance(v, int | float) and v >= 0:
        return f"{v:.1f} ms"
    return "—"


def _runtime_load(tel: Any, budget_ms: float) -> tuple[str, str]:
    latency_ms = float(getattr(tel, "latency_ms", 0.0) or 0.0)
    if latency_ms <= 0 or budget_ms <= 0:
        return "—", T.CURRENT.muted
    ratio = latency_ms / budget_ms
    text = f"{ratio * 100:.0f}% of frame budget"
    if ratio < 0.65:
        return text, T.TRACKING
    if ratio < 0.95:
        return text, T.WARNING
    return text, T.ERROR


def _setting_cost(tel: Any) -> str:
    rows = list(getattr(tel, "stage_timings", []) or [])
    total = 0.0
    for row in rows:
        if _field(row, "key", "") in {"detect", "track", "face", "pose"}:
            total += float(_field(row, "avg_ms", 0.0) or 0.0)
    return f"{total:.1f} ms avg ML/tracking" if total > 0 else "—"


def _stage_text(tel: Any, key: str, fallback_attr: str) -> tuple[str, str | None]:
    row = _find_by_key(getattr(tel, "stage_timings", []) or [], key)
    if row is None:
        return _ms(tel, fallback_attr), None
    status = str(_field(row, "status", "idle"))
    last = float(_field(row, "last_ms", 0.0) or 0.0)
    avg = float(_field(row, "avg_ms", 0.0) or 0.0)
    cadence = str(_field(row, "cadence", "") or "")
    text = f"{last:.1f} ms  avg {avg:.1f}"
    if cadence:
        text += f"  {cadence}"
    if status in {"disabled", "stale", "warming", "idle"}:
        text = f"{status.title()} · {text}"
    color = T.CURRENT.muted if status in {"disabled", "stale", "idle"} else None
    if status == "warming":
        color = T.WARNING
    return text, color


def _service_text(tel: Any, key: str) -> tuple[str, str | None]:
    row = _find_by_key(getattr(tel, "runtime_services", []) or [], key)
    if row is None:
        return "—", T.CURRENT.muted
    state = str(_field(row, "state", "idle"))
    detail = str(_field(row, "detail", "") or "")
    model = str(_field(row, "model", "") or _field(row, "backend", "") or "")
    text = state.title()
    if model:
        text += f" · {model}"
    if detail:
        text += f" · {detail}"
    color = (
        T.ERROR
        if state == "failed"
        else T.TRACKING
        if state == "active"
        else T.WARNING
        if state == "warming"
        else T.CURRENT.muted
    )
    return text, color


def _switch_text(sw: Any) -> str:
    if sw is None:
        return "—"
    state = str(_field(sw, "state", "idle"))
    to_value = str(_field(sw, "to_value", "") or _field(sw, "active_value", "") or "")
    reason = str(_field(sw, "reason", "") or "")
    text = state.title()
    if to_value:
        text += f" · {to_value}"
    if reason:
        text += f" · {reason}"
    return text


def _switch_color(sw: Any) -> str | None:
    state = str(_field(sw, "state", "idle")) if sw is not None else "idle"
    if state == "failed":
        return T.ERROR
    if state == "warming":
        return T.WARNING
    if state == "active":
        return T.TRACKING
    return T.CURRENT.muted


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _find_by_key(rows: list[Any], key: str) -> Any | None:
    for row in rows:
        if _field(row, "key", "") == key:
            return row
    return None


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
