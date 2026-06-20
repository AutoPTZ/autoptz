"""StatusBar — the bottom summary strip.

Replaces the meaningless "total fps" with a meaningful summary: engine state +
EP, an add-camera affordance, and an aggregate ``N cams · X fps avg · ● Health``
chip.  Recomputed on telemetry/engine/camera signals.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from autoptz.ui import theme as T
from autoptz.ui.widgets.common import on_theme_changed

log = logging.getLogger(__name__)


class StatusBar(QWidget):
    """Bottom status strip bound to the EngineClient."""

    def __init__(
        self,
        client: Any,
        logs_toggle: Callable[[bool], None] | None = None,
        cameras_popup: Callable[..., None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._logs_toggle = logs_toggle
        self._seps: list[QLabel] = []

        row = QHBoxLayout(self)
        # Roomier margins so descenders aren't clipped by the status-bar frame
        # and buttons size to their (padded) content rather than a rigid height.
        row.setContentsMargins(12, 7, 12, 7)
        row.setSpacing(12)

        # Passive engine-state indicator (running/stopped + execution provider).
        # Start/Stop lives in the Engine menu, so there's no button here.
        self._engine_dot = QLabel("●")
        self._engine_label = QLabel("Engine stopped")
        row.addWidget(self._engine_dot)
        row.addWidget(self._engine_label)

        if cameras_popup is not None:
            row.addWidget(self._make_sep())
            cam_btn = QPushButton("＋ Camera  ▾")
            cam_btn.setToolTip("Add a camera, or enable/disable discovered ones.")
            # Pass the button so the menu anchors under it (and stays put on Rescan).
            cam_btn.clicked.connect(lambda: cameras_popup(cam_btn))
            row.addWidget(cam_btn)

        row.addStretch(1)

        self._metrics = QLabel("")
        row.addWidget(self._metrics)

        row.addWidget(self._make_sep())
        self._summary = QLabel("")
        row.addWidget(self._summary)

        self._logs_btn: QPushButton | None = None
        if logs_toggle is not None:
            row.addWidget(self._make_sep())
            self._logs_btn = QPushButton("Logs  ▾")
            self._logs_btn.setCheckable(True)
            self._logs_btn.setToolTip("Show or hide the log console.")
            self._logs_btn.toggled.connect(self._on_logs_toggled)
            row.addWidget(self._logs_btn)

        self._restyle()
        on_theme_changed(client, self._restyle)
        _connect(client, "engineStateChanged", self.refresh)
        _connect(client, "telemetryUpdated", lambda *_: self._refresh_summary())
        _connect(client, "cameraAdded", lambda *_: self._refresh_summary())
        _connect(client, "cameraRemoved", lambda *_: self._refresh_summary())

        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._refresh_metrics)
        self._metrics_timer.start(1500)

        self.refresh()

    # ── styling ────────────────────────────────────────────────────────────────

    def _make_sep(self) -> QLabel:
        """A thin vertical divider tracked for theme restyling."""
        s = QLabel("│")
        self._seps.append(s)
        return s

    def _restyle(self) -> None:
        """Re-apply literal-color styling (construction + theme change)."""
        for s in self._seps:
            s.setStyleSheet(f"color: {T.CURRENT.border};")
        self.refresh()

    # ── actions ────────────────────────────────────────────────────────────────

    def set_logs_visible(self, shown: bool) -> None:
        # Chevron points up when the console is open, down when it's hidden.
        if self._logs_btn is not None:
            was_blocked = self._logs_btn.blockSignals(True)
            self._logs_btn.setChecked(shown)
            self._logs_btn.blockSignals(was_blocked)
            self._logs_btn.setText("Logs  ▴" if shown else "Logs  ▾")

    def _on_logs_toggled(self, shown: bool) -> None:
        self.set_logs_visible(shown)
        if self._logs_toggle is not None:
            self._logs_toggle(shown)

    # ── refreshers ───────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        running = bool(_safe(lambda: self._client.engineRunning, False))
        ep = (_safe(lambda: self._client.engineEp, "") or "").replace("ExecutionProvider", "")
        self._engine_dot.setStyleSheet(
            f"color: {T.TRACKING if running else T.CURRENT.muted};"
        )
        self._engine_label.setText(
            f"Engine running{('  ·  ' + ep) if ep else ''}" if running else "Engine stopped"
        )
        self._refresh_metrics()
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        n, avg, state = self._aggregate()
        color = {"healthy": T.TRACKING, "degraded": T.WARNING,
                 "error": T.ERROR}.get(state, T.CURRENT.muted)
        cams = f"{n} cam" if n == 1 else f"{n} cams"
        if not bool(_safe(lambda: self._client.engineRunning, False)):
            self._summary.setText(f"{cams}")
            return
        label = {"healthy": "Healthy", "degraded": "Degraded",
                 "error": "Error", "idle": "Idle"}.get(state, "Idle")
        self._summary.setText(
            f'{cams} · {avg:.0f} fps avg · <span style="color:{color}">●</span> {label}'
        )
        self._summary.setTextFormat(Qt.TextFormat.RichText)

    def _refresh_metrics(self) -> None:
        m = _safe(lambda: self._client.systemMetrics(), {}) or {}
        if not bool(m.get("available", False)):
            self._metrics.setText("CPU -- · System Mem -- · App -- · App Mem --")
            return
        self._metrics.setText(
            f"CPU {float(m.get('cpu_percent', 0.0)):.0f}% · "
            f"System Mem {float(m.get('mem_percent', 0.0)):.0f}% · "
            f"App {float(m.get('app_cpu_percent', 0.0)):.1f}% · "
            f"App Mem {float(m.get('app_mem_percent', 0.0)):.1f}%"
        )

    def _aggregate(self) -> tuple[int, float, str]:
        model = _safe(lambda: self._client.cameraModel, None)
        if model is None:
            return 0, 0.0, "idle"
        try:
            ids = list(model.camera_ids())
        except Exception:  # noqa: BLE001
            return 0, 0.0, "idle"
        n = len(ids)
        if n == 0:
            return 0, 0.0, "idle"
        total = 0.0
        any_streaming = False
        worst = "healthy"
        for cid in ids:
            rec = model.get_record(cid)
            if rec is None:
                continue
            total += float(getattr(rec, "fps", 0.0) or 0.0)
            health = str(getattr(rec, "health", "ok"))
            streaming = bool(getattr(rec, "streaming", False))
            any_streaming = any_streaming or streaming
            if health == "error":
                worst = "error"
            elif health in ("reconnecting", "stalled") and worst != "error":
                worst = "degraded"
        state = worst if any_streaming or worst != "healthy" else "idle"
        return n, total / n, state


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
