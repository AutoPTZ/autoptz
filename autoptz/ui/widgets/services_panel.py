"""ServicesPanel — engine services + status, with enabled-aware controls.

Header reads "Services and Status" (no ``&`` mnemonic, so no apparent double
space).  Start is disabled while running; Stop/Restart while stopped.  Rows come
from ``serviceStatus()``.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
    "ok": "OK",
    "running": "RUNNING",
    "warn": "FALLBACK",
    "stopped": "STOPPED",
    "off": "OFF",
}

# Session-only ML-subsystem testing overrides: (feature key, label, help text). The keys match
# ``EngineClient.features()`` / ``setFeatureEnabled`` exactly.
_FEATURE_TOGGLES = (
    (
        "detection",
        "Person detection",
        "Testing override: disable person detection for this session. It resets on launch.",
    ),
    (
        "tracking",
        "Tracking",
        "Testing override: disable track association/PTZ follow for this session.",
    ),
    (
        "face_recognition",
        "Face recognition",
        "Testing override: disable face matching for this session.",
    ),
    ("pose", "Pose", "Testing override: disable pose keypoints for this session."),
    (
        "reid",
        "ReID (stable tracking)",
        "Master switch for appearance ReID. A camera only uses it when this is on "
        "AND its tracking Mode is “Stable”. Turn it off to disable Stable mode "
        "everywhere. Resets to on each launch.",
    ),
)

# Detector tier value → display label.  The combo that lets you *change* the tier
# now lives in the Model Manager dialog; the Services panel only shows the active
# tier read-only, so it just needs the label lookup.
_DETECTOR_TIER_LABELS = {
    "auto": "Auto",
    "fast": "Fast",
    "balanced": "Balanced",
    "medium": "Accurate",
}


_FEATURE_COMPONENTS = {
    "detection": "detector",
    "face_recognition": "face",
    "pose": "pose",
    "reid": "reid",
}


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
        self.setObjectName("servicesPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._client = client
        self._rows: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        self._feature_boxes: dict[str, QCheckBox] = {}
        self._feature_tips: dict[str, str] = {}
        self._component_states: dict[str, str] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(scroll, 1)

        body = QWidget()
        body.setMinimumSize(0, 0)
        scroll.setWidget(body)

        root = QVBoxLayout(body)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # header + controls
        head = QHBoxLayout()
        head.setSpacing(6)
        title = QLabel("Services and Status")
        title.setStyleSheet("font-weight: 700;")
        head.addWidget(title)
        head.addWidget(
            HelpBadge(
                "Live state of each engine service. Use Start / Stop / Restart to "
                "control the tracking engine."
            )
        )
        head.addStretch(1)
        self._start = QPushButton("Start")
        self._start.clicked.connect(client.startEngine)
        self._stop = QPushButton("Stop")
        self._stop.clicked.connect(client.stopEngine)
        self._restart = QPushButton("Restart")
        self._restart.clicked.connect(client.restartEngine)
        for b in (self._start, self._stop, self._restart):
            # min-height (not fixed) so vertical padding/descenders aren't clipped.
            b.setMinimumHeight(26)
            head.addWidget(b)
        root.addLayout(head)

        # ── testing controls ──────────────────────────────────────────────────
        root.addWidget(hline())
        root.addWidget(section_label("Testing Overrides"))
        self._hint = QLabel(
            "Services start enabled every launch. Disable modules here only for testing."
        )
        self._hint.setWordWrap(True)
        root.addWidget(self._hint)

        for key, label, tip in _FEATURE_TOGGLES:
            row = QHBoxLayout()
            row.setSpacing(6)
            box = QCheckBox(label)
            box.setToolTip(tip)
            box.toggled.connect(lambda checked, k=key: self._on_feature_toggled(k, checked))
            self._feature_boxes[key] = box
            self._feature_tips[key] = tip
            row.addWidget(box)
            row.addWidget(HelpBadge(tip))
            row.addStretch(1)
            root.addLayout(row)
        _connect(client, "featuresChanged", self._refresh_features)

        # ── models (compact summary; full controls live in Manage Models) ────────
        root.addWidget(hline())
        root.addWidget(section_label("Models"))
        self._tier_label = QLabel()
        self._tier_label.setWordWrap(True)
        root.addWidget(self._tier_label)
        self._model_summary = QLabel()
        self._model_summary.setTextFormat(Qt.TextFormat.RichText)
        self._model_summary.setWordWrap(True)
        root.addWidget(self._model_summary)
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        self._manage_models = QPushButton("Manage Models...")
        self._manage_models.setToolTip(
            "Open the model setup window: pick the detector tier, see cache status, "
            "and download or remove the detector/pose models AutoPTZ manages."
        )
        self._manage_models.clicked.connect(self._open_model_manager)
        model_row.addWidget(self._manage_models)
        model_row.addWidget(HelpBadge(self._manage_models.toolTip()))
        model_row.addStretch(1)
        root.addLayout(model_row)
        _connect(client, "optionalComponentsChanged", self._refresh_optional_components)
        _connect(client, "detectorModelTierChanged", self._refresh_detector_tier)

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
        self._refresh_detector_tier()
        self._refresh_optional_components()
        self.refresh()

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(260, 220)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(360, 520)

    def _restyle(self) -> None:
        """Re-apply literal-color styling (construction + theme change)."""
        # Rows bake theme colors at build time; refresh repaints them now.
        self._style_emphasis_labels()
        self._refresh_optional_components()
        self.refresh()

    def _style_emphasis_labels(self) -> None:
        """Colour the secondary text at full strength so it reads clearly.

        ``subtext`` looked washed-out for these; the active-tier line and the
        testing-overrides hint are real information, so they use ``text``.
        """
        hint = getattr(self, "_hint", None)
        tier = getattr(self, "_tier_label", None)
        for label in (hint, tier):
            if label is not None:
                label.setStyleSheet(f"color: {T.CURRENT.text};")

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
            name.setText(
                f"<b>{r.get('name', key)}</b><br>"
                f"<span style='color:{T.CURRENT.subtext}'>{r.get('detail', '')}</span>"
            )
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

    # ── testing overrides ────────────────────────────────────────────────────

    def _on_feature_toggled(self, key: str, checked: bool) -> None:
        """Apply a session-only subsystem switch live via the client."""
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
        self._apply_component_gating()

    def _refresh_detector_tier(self) -> None:
        label = getattr(self, "_tier_label", None)
        if label is None:
            return
        tier = str(_safe(lambda: self._client.getDetectorModelTier(), "auto") or "auto")
        label.setText(f"Active detector model: {_DETECTOR_TIER_LABELS.get(tier, tier.title())}")

    def _refresh_optional_components(self) -> None:
        summary = getattr(self, "_model_summary", None)
        rows = _safe(lambda: self._client.optionalComponents(), []) or []
        self._component_states = {
            str(row.get("key", "")): str(row.get("state", "off")) for row in rows
        }
        self._apply_component_gating()
        if summary is None:
            return
        states = self._component_states
        missing = [
            row.get("name", row.get("key", ""))
            for row in rows
            if row.get("key") in {"detector", "pose"} and row.get("state") != "ok"
        ]
        ok_color = T.TRACKING
        warn_color = T.WARNING

        def chip(key: str, text: str) -> str:
            color = ok_color if states.get(key) == "ok" else warn_color
            mark = "✓" if states.get(key) == "ok" else "•"
            return f"<span style='color:{color}'>{mark} {text}</span>"

        line = (
            f"AutoPTZ-managed: {chip('detector', 'Detector')} · {chip('pose', 'Pose')}"
            f"<br><span style='color:{T.CURRENT.subtext}'>"
            "Face &amp; ReID weights are managed by their upstream packages.</span>"
        )
        if missing:
            line += (
                f"<br><span style='color:{warn_color}'>"
                f"{', '.join(str(m) for m in missing)} not downloaded — open Manage Models."
                "</span>"
            )
        summary.setText(line)

    def _apply_component_gating(self) -> None:
        """Grey out toggles whose required model/component is missing.

        Purely **visual** — the engine's feature flags are left untouched.  The
        detector is pool-authoritative now: a missing model means nothing runs
        and nothing is drawn regardless of the flag, so there is no UI/engine
        desync to "enforce".  Just as importantly, NOT forcing the flag off means
        detection (and the downstream features) resume automatically once the
        model is downloaded again, instead of sticking off — which previously
        left the engine quiet after a delete→re-download.  The checked state
        mirrors the engine via :meth:`_refresh_features`.
        """
        states = self._component_states
        detector_ok = states.get("detector", "ok") == "ok"
        # Everything visible flows from the body detector: face/pose/ReID only
        # label or stabilise *already-detected* bodies, and tracking follows them —
        # so without the detector model nothing is drawn no matter the individual
        # switches.  Grey the downstream toggles too, with a tooltip that explains
        # why, instead of leaving the operator wondering why "Face recognition" is
        # on but nothing happens.
        for feature in ("detection", "tracking", "face_recognition", "pose", "reid"):
            box = self._feature_boxes.get(feature)
            if box is None:
                continue
            component = _FEATURE_COMPONENTS.get(feature)
            own_ok = component is None or states.get(component, "ok") == "ok"
            needs_detector = feature != "detection"
            available = own_ok and (detector_ok or not needs_detector)
            box.setEnabled(available)
            if available:
                box.setToolTip(self._feature_tips.get(feature, ""))
                continue
            if needs_detector and not detector_ok:
                reason = "Requires the detector model — open Manage Models to download it."
            elif component is not None:
                reason = f"Disabled until the {component} component is available."
            else:
                reason = "Unavailable."
            box.setToolTip(f"{self._feature_tips.get(feature, '')}\n\n{reason}")

    def _open_model_manager(self) -> None:
        from autoptz.ui.widgets.dialogs.model_manager import ModelManagerDialog

        ModelManagerDialog(self._client, parent=self).exec()
        self._refresh_optional_components()
        self.refresh()

    def _ensure_row(self, key: str) -> None:
        if key in self._rows:
            return
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 6, 0, 6)
        h.setSpacing(10)
        dot = QLabel("●")
        name = QLabel()
        name.setTextFormat(Qt.TextFormat.RichText)
        pill = QLabel()
        pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
