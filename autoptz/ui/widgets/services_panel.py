"""ServicesPanel — engine services + status, with enabled-aware controls.

Header reads "Services and Status" (no ``&`` mnemonic, so no apparent double
space).  Start is disabled while running; Stop/Restart while stopped.  Rows come
from ``serviceStatus()``.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.common import (
    HelpBadge,
    hline,
    on_theme_changed,
    section_label,
)

log = logging.getLogger(__name__)

_STATE_LABEL = {
    "ok": "OK", "running": "RUNNING", "warn": "FALLBACK",
    "stopped": "STOPPED", "off": "OFF",
}

# Global ML-subsystem switches: (feature key, label, help text).  The keys match
# ``EngineClient.features()`` / ``setFeatureEnabled`` exactly.
_FEATURE_TOGGLES = (
    ("detection", "Person detection",
     "Find people in each frame. Turning this off stops all detection, "
     "tracking, and aiming — and saves the most CPU."),
    ("tracking", "Tracking",
     "Follow detected people across frames so PTZ aiming stays on the same "
     "person. Off saves CPU but disables auto-follow."),
    ("face_recognition", "Face recognition",
     "Match faces to enrolled identities so tracking can prefer a named "
     "person. Off saves CPU; tracking still works without names."),
    ("pose", "Pose",
     "Estimate body keypoints for smarter aim/framing. Off saves CPU and "
     "falls back to bounding-box aiming."),
)


def _state_color(state: str) -> str:
    """Resolve a service state to a color at call time (theme-aware)."""
    if state in ("ok", "running"):
        return T.TRACKING
    if state == "warn":
        return T.WARNING
    if state == "stopped":
        return T.CURRENT.subtext
    return T.ERROR  # off / unknown


class ServicesPanel(QWidget):
    """Live service status + engine lifecycle controls."""

    def __init__(self, client: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client = client
        self._rows: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        self._feature_boxes: dict[str, QCheckBox] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # header + controls
        head = QHBoxLayout()
        head.setSpacing(6)
        title = QLabel("Services and Status")
        title.setStyleSheet("font-weight: 700;")
        head.addWidget(title)
        head.addWidget(HelpBadge(
            "Live state of each engine service. Use Start / Stop / Restart to "
            "control the tracking engine."
        ))
        head.addStretch(1)
        self._start = QPushButton("Start"); self._start.clicked.connect(client.startEngine)
        self._stop = QPushButton("Stop"); self._stop.clicked.connect(client.stopEngine)
        self._restart = QPushButton("Restart"); self._restart.clicked.connect(client.restartEngine)
        for b in (self._start, self._stop, self._restart):
            # min-height (not fixed) so vertical padding/descenders aren't clipped.
            b.setMinimumHeight(26)
            head.addWidget(b)
        root.addLayout(head)

        # ── global subsystem toggles ──────────────────────────────────────────
        # Live on/off switches for the heavy ML stages; each saves CPU when off.
        root.addWidget(hline())
        root.addWidget(section_label("Subsystems"))
        perf = QPushButton("Performance Mode")
        perf.setToolTip(
            "Disable Face recognition and Pose while keeping preview, detection, "
            "and tracking on."
        )
        perf.clicked.connect(self._enable_performance_mode)
        root.addWidget(perf)
        for key, label, tip in _FEATURE_TOGGLES:
            row = QHBoxLayout()
            row.setSpacing(6)
            box = QCheckBox(label)
            box.setToolTip(tip)
            box.toggled.connect(lambda checked, k=key: self._on_feature_toggled(k, checked))
            self._feature_boxes[key] = box
            row.addWidget(box)
            row.addWidget(HelpBadge(tip))
            row.addStretch(1)
            root.addLayout(row)
        _connect(client, "featuresChanged", self._refresh_features)

        root.addWidget(hline())
        root.addWidget(section_label("Optional Setup"))
        self._setup_list = QVBoxLayout()
        self._setup_list.setSpacing(6)
        root.addLayout(self._setup_list)
        _connect(client, "optionalComponentsChanged", self._refresh_optional_components)

        root.addWidget(hline())
        self._list = QVBoxLayout()
        self._list.setSpacing(0)
        root.addLayout(self._list)
        root.addStretch(1)

        self._restyle()
        on_theme_changed(client, self._restyle)
        _connect(client, "engineStateChanged", self.refresh)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(1500)
        self._refresh_features()
        self._refresh_optional_components()
        self.refresh()

    def _restyle(self) -> None:
        """Re-apply literal-color styling (construction + theme change)."""
        # Rows bake theme colors at build time; refresh repaints them now.
        self.refresh()

    def refresh(self) -> None:
        running = bool(_safe(lambda: self._client.engineRunning, False))
        self._start.setEnabled(not running)
        self._stop.setEnabled(running)
        self._restart.setEnabled(running)

        rows = _safe(lambda: self._client.serviceStatus(), []) or []
        seen = set()
        for r in rows:
            key = str(r.get("key", r.get("name", "")))
            seen.add(key)
            state = str(r.get("state", "off"))
            self._ensure_row(key)
            dot, name, pill = self._rows[key]
            color = _state_color(state)
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            name.setText(f"<b>{r.get('name', key)}</b><br>"
                         f"<span style='color:{T.CURRENT.subtext}'>{r.get('detail', '')}</span>")
            name.setWordWrap(True)
            pill.setText(_STATE_LABEL.get(state, state.upper()))
            pill.setStyleSheet(
                f"color: {color}; border: 1px solid {color}; border-radius: 9px;"
                f"padding: 1px 8px; font-size: 9px; font-weight: 700;"
            )
        # remove stale rows
        for key in list(self._rows):
            if key not in seen:
                dot, name, pill = self._rows.pop(key)
                for w in (dot, name, pill):
                    w.deleteLater()

    # ── subsystem toggles ────────────────────────────────────────────────────

    def _on_feature_toggled(self, key: str, checked: bool) -> None:
        """Apply a subsystem switch live via the client."""
        _safe(lambda: self._client.setFeatureEnabled(key, checked), None)

    def _refresh_features(self) -> None:
        """Mirror the client's feature flags into the checkboxes.

        Blocks signals while setting so reflecting an external change (the
        ``featuresChanged`` signal) doesn't loop back into ``setFeatureEnabled``.
        """
        feats = _safe(lambda: self._client.features(), {}) or {}
        for key, box in self._feature_boxes.items():
            on = bool(feats.get(key, True))
            if box.isChecked() != on:
                box.blockSignals(True)
                box.setChecked(on)
                box.blockSignals(False)

    def _enable_performance_mode(self) -> None:
        _safe(lambda: self._client.setFeatureEnabled("face_recognition", False), None)
        _safe(lambda: self._client.setFeatureEnabled("pose", False), None)
        _safe(lambda: self._client.setFeatureEnabled("detection", True), None)
        _safe(lambda: self._client.setFeatureEnabled("tracking", True), None)
        self._refresh_features()

    def _refresh_optional_components(self) -> None:
        while self._setup_list.count():
            item = self._setup_list.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        rows = _safe(lambda: self._client.optionalComponents(), []) or []
        visible = [r for r in rows if r.get("state") != "ok"]
        if not visible:
            lab = QLabel("All optional components are available.")
            lab.setStyleSheet(f"color: {T.CURRENT.subtext};")
            self._setup_list.addWidget(lab)
            return
        for row in visible:
            self._setup_list.addWidget(self._setup_row(row))

    def _setup_row(self, row: dict[str, Any]) -> QWidget:
        key = str(row.get("key", ""))
        box = QFrame()
        box.setFrameShape(QFrame.Shape.NoFrame)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(5)
        box.setStyleSheet(
            f"QFrame {{ background: {T.CURRENT.surface_hov};"
            f" border: 1px solid {T.CURRENT.border}; border-radius: {T.RADIUS}px; }}"
        )
        title = QLabel(f"<b>{row.get('name', key)}</b>")
        title.setTextFormat(Qt.TextFormat.RichText)
        detail = QLabel(
            f"{row.get('detail', '')}<br>"
            f"<span style='color:{T.CURRENT.subtext}'>"
            f"Source: {row.get('source', '—')} · Size: {row.get('size', '—')}<br>"
            f"Path: {row.get('path', '—')}<br>{row.get('network', '')}</span>"
        )
        detail.setTextFormat(Qt.TextFormat.RichText)
        detail.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(detail)
        actions = QHBoxLayout()
        retry = QPushButton("Retry setup")
        retry.clicked.connect(lambda _=False, k=key: self._client.retryOptionalComponent(k))
        ignore = QPushButton("Ignore forever")
        ignore.setEnabled(not bool(row.get("ignored")))
        ignore.clicked.connect(lambda _=False, k=key: self._client.setOptionalComponentIgnored(k, True))
        actions.addWidget(retry)
        actions.addWidget(ignore)
        actions.addStretch(1)
        lay.addLayout(actions)
        return box

    def _ensure_row(self, key: str) -> None:
        if key in self._rows:
            return
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 6, 0, 6)
        h.setSpacing(10)
        dot = QLabel("●")
        name = QLabel(); name.setTextFormat(Qt.TextFormat.RichText)
        pill = QLabel(); pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(name, 1)
        h.addWidget(pill, 0, Qt.AlignmentFlag.AlignVCenter)
        self._list.addWidget(row)
        self._rows[key] = (dot, name, pill)


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
