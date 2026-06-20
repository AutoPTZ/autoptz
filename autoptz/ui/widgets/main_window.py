"""MainWindow — the native Qt Widgets shell (QMainWindow).

A native menu bar + dockable panels around a central camera wall, with engine
lifecycle controls and a summary status bar.  The Cameras menu is a checkable,
rescan-able list of discovered USB devices (keyed on stable ``unique_id`` so the
Continuity-Camera index shuffle never confuses the naming/checks) plus NDI and
"Add Network Camera…".

Panel object names are stable so a later phase can persist/restore the dock
layout with ``saveState()``/``restoreState()``.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QMenu,
    QProgressBar,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui.widgets.camera_info_panel import CameraInfoPanel
from autoptz.ui.widgets.camera_wall import CameraWall
from autoptz.ui.widgets.common import on_theme_changed
from autoptz.ui.widgets.dialogs import AboutDialog, NetworkCameraDialog
from autoptz.ui.widgets.logs_panel import LogsPanel
from autoptz.ui.widgets.people_panel import PeoplePanel
from autoptz.ui.widgets.properties_panel import PropertiesPanel
from autoptz.ui.widgets.services_panel import ServicesPanel
from autoptz.ui.widgets.status_bar import StatusBar

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Top-level window: menus, docks, status bar, engine controls."""

    # The right-side docks are tabified into one OBS-style section group; their
    # QDockWidget title bars are redundant with the section tab labels, so they
    # are hidden (see ``_build_docks``) and lock/close lives on the tab strip.
    _SECTION_KEYS = ("camera_info", "people", "services")

    def __init__(
        self,
        client: Any,
        log_model: Any | None = None,
        frame_source: Any | None = None,
        theme: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._log_model = log_model
        self._frames = frame_source
        self._theme = theme
        self._docks: dict[str, QDockWidget] = {}
        self._selected_camera: str = ""
        self._shown_optional_setup_prompt = False

        self.setWindowTitle("AutoPTZ")
        self.setDockNestingEnabled(True)
        self.resize(1320, 820)

        self._build_central()
        self._build_docks()
        self._build_menus()
        self._build_status_bar()

        _connect(client, "engineStateChanged", self._refresh_engine_state)
        _connect(client, "startupProgressChanged", self._refresh_startup_progress)
        _connect(client, "errorOccurred", self._on_error)
        # Segmented section tabs bake in literal palette colors, so restyle them
        # whenever the appearance flips.
        on_theme_changed(client, self._style_section_tabs)

        self._restore_geometry()
        self._refresh_engine_state()

    # ── section title bars ──────────────────────────────────────────────────────

    def _install_section_title_bars(self) -> None:
        """Suppress the redundant native title bar on the tabified section docks.

        Camera Info / People / Services take their name from the section tab
        strip, so a separate QDockWidget title bar above the tabs would just
        repeat it; replace it with a slim empty strip.  Properties and Logs keep
        their default title bar (which shows their name), and every panel stays
        fully movable — there is no locking.
        """
        for key in self._SECTION_KEYS:
            dock = self._docks.get(key)
            if dock is None:
                continue
            bar = QWidget(self)
            bar.setObjectName("dockTitleBar")
            bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            bar.setFixedHeight(3)
            dock.setTitleBarWidget(bar)

    # ── central ──────────────────────────────────────────────────────────────────

    def _build_central(self) -> None:
        holder = QWidget(self)
        col = QVBoxLayout(holder)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        self._startup_bar = QProgressBar(holder)
        self._startup_bar.setTextVisible(True)
        self._startup_bar.setFixedHeight(5 + self.fontMetrics().height())
        self._startup_bar.setRange(0, 0)
        self._startup_bar.setVisible(False)
        col.addWidget(self._startup_bar)
        self._wall = CameraWall(self._client, self._frames, holder)
        self._wall.cameraSelected.connect(self._on_camera_selected)
        self._wall.cameraInfoRequested.connect(self._on_camera_info_requested)
        self._wall.addCameraRequested.connect(self._open_cameras_menu)
        col.addWidget(self._wall, 1)
        self.setCentralWidget(holder)

    # ── docks ──────────────────────────────────────────────────────────────────

    def _build_docks(self) -> None:
        self._properties = PropertiesPanel(self._client, frame_source=self._frames)
        self._camera_info = CameraInfoPanel(self._client)
        self._people = PeoplePanel(self._client)
        self._services = ServicesPanel(self._client)
        if self._log_model is not None:
            self._logs: QWidget = LogsPanel(self._client, self._log_model)
        else:
            self._logs = QLabel("Logs unavailable")

        specs = [
            ("properties", "Properties", self._properties, Qt.DockWidgetArea.LeftDockWidgetArea),
            ("camera_info", "Camera Info", self._camera_info, Qt.DockWidgetArea.RightDockWidgetArea),
            ("people", "People", self._people, Qt.DockWidgetArea.RightDockWidgetArea),
            ("services", "Services", self._services, Qt.DockWidgetArea.RightDockWidgetArea),
            ("logs", "Logs", self._logs, Qt.DockWidgetArea.BottomDockWidgetArea),
        ]
        for key, title, widget, area in specs:
            dock = QDockWidget(title, self)
            dock.setObjectName(f"dock_{key}")
            dock.setWidget(widget)
            self.addDockWidget(area, dock)
            self._docks[key] = dock

        self.tabifyDockWidget(self._docks["camera_info"], self._docks["people"])
        self.tabifyDockWidget(self._docks["people"], self._docks["services"])
        self._docks["camera_info"].raise_()

        # Suppress the redundant native title bars on the tabified section docks
        # (their name comes from the section tab strip).
        self._install_section_title_bars()

        # Section tabs ride the TOP of the right-side dock group (OBS-style),
        # not the default bottom plain-text strip.  A targeted objectName lets
        # us give that QTabBar a clean segmented-button look in _style_section_tabs.
        self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea,
                            QTabWidget.TabPosition.North)
        self._style_section_tabs()

        # Sensible first-run proportions (a saved layout overrides these later).
        self.resizeDocks(
            [self._docks["properties"], self._docks["camera_info"]],
            [330, 360], Qt.Orientation.Horizontal,
        )
        self.resizeDocks([self._docks["logs"]], [200], Qt.Orientation.Vertical)

    def _style_section_tabs(self) -> None:
        """Give the dock-area QTabBars a clean segmented look at the top.

        The global stylesheet already draws rounded/underlined ``QTabBar::tab``;
        here we tag the dock tab bars with a stable objectName and layer a
        targeted, palette-driven rule so the right-side group reads as a
        contiguous segmented section bar (Camera Info · People · Services).
        Called on construction and on every theme flip via on_theme_changed.
        """
        from autoptz.ui import theme as T

        pal = T.CURRENT
        r = T.RADIUS
        for bar in self.findChildren(QTabBar):
            # Only the docked-panel tab bars belong to the QMainWindow itself;
            # tab bars owned by panel widgets have a parent widget of their own.
            if bar.parent() is not self:
                continue
            bar.setObjectName("sectionTabs")
            bar.setExpanding(False)
            bar.setDrawBase(False)
            bar.setStyleSheet(
                f"QTabBar#sectionTabs {{ background: {pal.background};"
                f" qproperty-drawBase: 0; }}"
                f"QTabBar#sectionTabs::tab {{ background: {pal.surface_alt};"
                f" color: {pal.subtext}; border: 1px solid {pal.border};"
                f" padding: 6px 16px; margin: 0; min-width: 56px; }}"
                f"QTabBar#sectionTabs::tab:first {{"
                f" border-top-left-radius: {r}px; border-bottom-left-radius: {r}px; }}"
                f"QTabBar#sectionTabs::tab:last {{"
                f" border-top-right-radius: {r}px; border-bottom-right-radius: {r}px; }}"
                f"QTabBar#sectionTabs::tab:!first {{ border-left: none; }}"
                f"QTabBar#sectionTabs::tab:selected {{ background: {pal.surface};"
                f" color: {pal.text}; }}"
                f"QTabBar#sectionTabs::tab:hover:!selected {{ background: {pal.surface_hov};"
                f" color: {pal.text}; }}"
            )

    # ── menus ──────────────────────────────────────────────────────────────────

    def _build_menus(self) -> None:
        bar = self.menuBar()

        engine = bar.addMenu("&Engine")
        engine.setToolTipsVisible(True)
        # Keep the tracking-stop action's enabled state honest as cameras come
        # and go, not only on engine start/stop.
        engine.aboutToShow.connect(self._refresh_engine_state)
        self._act_start = _action(
            self, "Start", self._client.startEngine, "Ctrl+E",
            "Start the detection/tracking engine and open all enabled cameras.",
        )
        self._act_stop = _action(
            self, "Stop", self._client.stopEngine, "Ctrl+Shift+E",
            "Stop the engine and release all cameras.",
        )
        self._act_restart = _action(
            self, "Restart", self._client.restartEngine, "Ctrl+Shift+R",
            "Restart the engine — re-applies model and execution-provider settings.",
        )
        engine.addAction(self._act_start)
        engine.addAction(self._act_stop)
        engine.addAction(self._act_restart)
        engine.addSeparator()
        self._act_stop_tracking = _action(
            self, "Stop All Tracking", self._stop_all_tracking, "Ctrl+Shift+T",
            "Turn off person tracking on every camera at once. Cameras keep "
            "streaming; only the auto-follow is disabled.",
        )
        engine.addAction(self._act_stop_tracking)

        # Cameras — rebuilt on open so device checks stay live.  Populated once
        # now so the native (macOS) menu bar shows it instead of hiding an empty
        # menu.
        self._cameras_menu = bar.addMenu("&Cameras")
        self._cameras_menu.aboutToShow.connect(
            lambda: self._populate_cameras_menu(self._cameras_menu)
        )
        self._populate_cameras_menu(self._cameras_menu)

        view = bar.addMenu("&View")
        appearance = view.addMenu("Appearance")
        appearance.setToolTipsVisible(True)
        group = QActionGroup(self); group.setExclusive(True)
        current = _safe(lambda: str(self._client.themeMode), "dark")
        tips = {"system": "Follow the OS light/dark setting.",
                "dark": "Dark broadcast palette (easiest on the eyes).",
                "light": "Light palette for bright rooms."}
        for mode, label in (("system", "System"), ("dark", "Dark"), ("light", "Light")):
            act = QAction(label, self, checkable=True)
            act.setChecked(mode == current)
            act.setToolTip(tips[mode])
            act.triggered.connect(lambda _c, m=mode: self._client.setTheme(m))
            group.addAction(act)
            appearance.addAction(act)

        self._build_scale_menu(view)

        view.addSeparator()
        overlays = view.addMenu("Overlays")
        overlays.setToolTipsVisible(True)
        cur = _safe(lambda: self._client.overlays(), {}) or {}
        for key, label, tip in (
            ("detection", "Detection boxes",
             "Show a box around every detected person."),
            ("faces", "Face boxes",
             "Show face-recognition boxes with the matched name."),
            ("pose", "Pose skeleton",
             "Draw the tracked person's body skeleton."),
            ("prediction", "Motion prediction",
             "Debug overlay: draw the target prediction ghost/lead indicator."),
        ):
            act = QAction(label, self, checkable=True)
            act.setChecked(bool(cur.get(key, key == "detection")))
            act.setToolTip(tip)
            act.toggled.connect(lambda on, k=key: self._client.setOverlay(k, on))
            overlays.addAction(act)

        panels = bar.addMenu("&Panels")
        for key in ("properties", "camera_info", "people", "services", "logs"):
            panels.addAction(self._docks[key].toggleViewAction())

        self._layouts_menu = bar.addMenu("&Layouts")
        self._layouts_menu.setToolTipsVisible(True)
        self._layouts_menu.aboutToShow.connect(self._populate_layouts_menu)

        helpm = bar.addMenu("&Help")
        helpm.addAction(_action(self, "About AutoPTZ", self._show_about))

    def _build_scale_menu(self, view: QMenu) -> None:
        """View → UI Scale: discrete steps + Zoom In/Out/Reset shortcuts."""
        scale = view.addMenu("UI Scale")
        scale.setToolTipsVisible(True)
        self._scale_group = QActionGroup(self); self._scale_group.setExclusive(True)
        steps = getattr(self._theme, "SCALE_STEPS", (0.9, 1.0, 1.1, 1.25, 1.5)) \
            if self._theme is not None else (1.0,)
        for s in steps:
            act = QAction(f"{round(s * 100)}%", self, checkable=True)
            act.setToolTip("Scale all text and controls by this amount.")
            act.triggered.connect(lambda _c, f=s: self._set_scale(f))
            self._scale_group.addAction(act)
            scale.addAction(act)
        scale.aboutToShow.connect(self._refresh_scale_checks)
        self._refresh_scale_checks()

        view.addSeparator()
        zin = _action(self, "Zoom In", lambda: self._nudge_scale(1), "Ctrl+=")
        zin.setToolTip("Make text and controls larger.")
        zout = _action(self, "Zoom Out", lambda: self._nudge_scale(-1), "Ctrl+-")
        zout.setToolTip("Make text and controls smaller.")
        zreset = _action(self, "Reset Zoom", lambda: self._nudge_scale(0), "Ctrl+0")
        zreset.setToolTip("Return to 100%.")
        for act in (zin, zout, zreset):
            act.setEnabled(self._theme is not None)
            view.addAction(act)

    def _set_scale(self, factor: float) -> None:
        if self._theme is not None:
            self._theme.set_scale(factor)
            self._refresh_scale_checks()

    def _nudge_scale(self, direction: int) -> None:
        if self._theme is not None:
            self._theme.nudge_scale(direction)
            self._refresh_scale_checks()

    def _refresh_scale_checks(self) -> None:
        if self._theme is None:
            return
        cur = float(getattr(self._theme, "scale", 1.0))
        for act in self._scale_group.actions():
            try:
                pct = int(act.text().rstrip("%"))
            except ValueError:
                continue
            act.setChecked(abs(pct / 100.0 - cur) < 0.001)

    def _open_cameras_menu(self, anchor: QWidget | None = None) -> None:
        """Pop the Cameras menu, anchored under its triggering button.

        Re-opening after Rescan reuses the same anchor (rather than the live
        cursor position) so the menu stays put instead of hopping to wherever the
        pointer landed on the Rescan item.
        """
        from PySide6.QtGui import QCursor
        if isinstance(anchor, QWidget):
            self._cameras_menu_anchor = anchor.mapToGlobal(anchor.rect().bottomLeft())
        elif getattr(self, "_cameras_menu_anchor", None) is None:
            self._cameras_menu_anchor = QCursor.pos()
        menu = QMenu(self)
        self._populate_cameras_menu(menu)
        menu.exec(self._cameras_menu_anchor)

    def _populate_cameras_menu(self, menu: QMenu | None = None) -> None:
        menu = menu if menu is not None else self._cameras_menu
        menu.clear()
        menu.setToolTipsVisible(True)

        # USB / built-in / Continuity cameras live in their own submenu so the
        # source kinds read clearly; check to add, uncheck to remove.
        usb = menu.addMenu("USB Cameras")
        usb.setToolTipsVisible(True)
        devices = _safe(lambda: self._client.scanUSBCameras(), []) or []
        if not devices:
            none = usb.addAction("No USB cameras found")
            none.setEnabled(False)
        for dev in devices:
            # Show the friendly source kind ("— Built-in" / "— External") rather
            # than the opaque usb://N uri; Continuity cameras already carry their
            # tag in the name, so don't double it.
            label = str(dev.get("name", dev.get("uri", "?")))
            src_label = str(dev.get("source_label") or "")
            if src_label and not dev.get("is_continuity") and src_label not in label:
                label = f"{label} — {src_label}"
            act = QAction(label, self, checkable=True)
            act.setChecked(bool(dev.get("in_use")))
            if dev.get("is_continuity"):
                act.setToolTip(
                    "iPhone Continuity Camera. Matched by device id so reconnects "
                    "stay correct, but it disappears when the phone sleeps/locks."
                )
            else:
                act.setToolTip(
                    "Local USB / built-in webcam. Lowest latency; CPU cost scales "
                    "with resolution and frame rate."
                )
            act.toggled.connect(
                lambda checked, d=dev: self._toggle_usb_camera(d, checked)
            )
            usb.addAction(act)
        usb.addSeparator()
        usb.addAction(_action(
            self, "Rescan", self._rescan_usb, tip="Re-probe connected USB cameras.",
        ))

        menu.addSeparator()
        ndi = menu.addMenu("NDI Sources")
        ndi.setToolTipsVisible(True)
        ndi.addAction(_action(
            self, "Scan for NDI Sources…", self._scan_ndi,
            tip="Discover NDI network video sources on the LAN. Network-dependent "
                "and heavier to decode than USB.",
        ))

        menu.addSeparator()
        menu.addAction(_action(
            self, "Add Network Camera…", self._add_network_camera,
            tip="Add an RTSP / ONVIF IP camera by URL. Decoding a network stream "
                "uses more CPU than a local USB camera.",
        ))

    # ── camera actions ───────────────────────────────────────────────────────────

    def _toggle_usb_camera(self, dev: dict, checked: bool) -> None:
        uri = dev.get("uri", "")
        unique_id = dev.get("unique_id", "")
        if checked:
            # Pass the uniqueID straight through (the menu knows it for this exact
            # device) so selection never depends on a scan-cache lookup by uri.
            self._client.addCamera(
                uri, dev.get("name", ""), unique_id, dev.get("source_label", ""),
            )
        else:
            cid = self._find_camera(uri, unique_id)
            if cid:
                self._client.removeCamera(cid)

    def _find_camera(self, uri: str, unique_id: str) -> str | None:
        model = self._client.cameraModel
        for cid in _safe(lambda: model.camera_ids(), []) or []:
            rec = model.get_record(cid)
            cfg = getattr(rec, "camera_config", None)
            src = getattr(cfg, "source", None)
            if src is None:
                continue
            if unique_id and getattr(src, "unique_id", None) == unique_id:
                return cid
            if getattr(src, "address", None) == uri:
                return cid
        return None

    def _rescan_usb(self) -> None:
        # scanUSBCameras() re-probes devices; re-open the refreshed list.
        self._open_cameras_menu()

    def _scan_ndi(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        sources = _safe(lambda: self._client.scanNDISources(), []) or []
        if not sources:
            self.statusBar().showMessage("No NDI sources found (cyndilib not installed?)", 5000)
            return
        labels = [s.get("name", s.get("uri", "?")) for s in sources]
        label, ok = QInputDialog.getItem(self, "NDI Sources", "Source:", labels, 0, False)
        if ok:
            src = sources[labels.index(label)]
            self._client.addCamera(src.get("uri", ""), src.get("name", ""))

    def _add_network_camera(self) -> None:
        NetworkCameraDialog(self._client, self).exec()

    def _show_about(self) -> None:
        AboutDialog(self._client, self).exec()

    # ── saved dock layouts ───────────────────────────────────────────────────────

    def _populate_layouts_menu(self) -> None:
        menu = self._layouts_menu
        menu.clear()
        save = _action(self, "Save Current Layout…", self._save_layout)
        save.setToolTip(
            "A layout captures which panels are open, their docked positions, "
            "sizes, and the active section tab."
        )
        save.setStatusTip(save.toolTip())
        menu.addAction(save)
        layouts = _safe(lambda: self._client.getSetting("dock_layouts", {}), {}) or {}
        if layouts:
            menu.addSeparator()
            recall = menu.addMenu("Recall")
            delete = menu.addMenu("Delete")
            for name in sorted(layouts):
                recall.addAction(_action(self, name, lambda n=name: self._recall_layout(n)))
                delete.addAction(_action(self, name, lambda n=name: self._delete_layout(n)))

    def _save_layout(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Layout", "Layout name:")
        if not ok or not name.strip():
            return
        layouts = dict(_safe(lambda: self._client.getSetting("dock_layouts", {}), {}) or {})
        # Entries are dicts carrying the encoded dock state; legacy plain-string
        # and ``{"state","locked"}`` entries still recall fine below.
        layouts[name.strip()] = {"state": self._encode_state()}
        self._client.setSetting("dock_layouts", layouts)

    def _recall_layout(self, name: str) -> None:
        layouts = _safe(lambda: self._client.getSetting("dock_layouts", {}), {}) or {}
        entry = layouts.get(name)
        if not entry:
            return
        # Tolerate both the legacy "base64 string" shape and the dict shape
        # (the ``locked`` key from older saves is simply ignored now).
        blob = entry.get("state", "") if isinstance(entry, dict) else entry
        if blob:
            try:
                self.restoreState(QByteArray(base64.b64decode(blob)))
            except Exception:  # noqa: BLE001
                log.debug("recall layout failed", exc_info=True)
        # restoreState can rebuild the dock tab bar; keep tabs at the top with
        # the segmented look, and re-suppress the redundant section title bars.
        self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea,
                            QTabWidget.TabPosition.North)
        self._style_section_tabs()
        self._install_section_title_bars()

    def _delete_layout(self, name: str) -> None:
        layouts = dict(_safe(lambda: self._client.getSetting("dock_layouts", {}), {}) or {})
        if layouts.pop(name, None) is not None:
            self._client.setSetting("dock_layouts", layouts)

    def _encode_state(self) -> str:
        return bytes(self.saveState().toBase64()).decode("ascii")

    # ── status bar ───────────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        self._status = StatusBar(
            self._client, logs_toggle=self._toggle_logs,
            cameras_popup=self._open_cameras_menu,
        )
        self.statusBar().addPermanentWidget(self._status, 1)
        self.statusBar().setSizeGripEnabled(False)

    def _toggle_logs(self) -> None:
        dock = self._docks.get("logs")
        if dock is None:
            return
        if dock.isVisible():
            dock.hide()
        else:
            dock.show()
            dock.raise_()

    # ── selection routing ──────────────────────────────────────────────────────

    def _on_camera_selected(self, camera_id: str) -> None:
        self._selected_camera = camera_id
        self._properties.set_camera(camera_id)
        self._camera_info.set_camera(camera_id)
        rec = _safe(lambda: self._client.cameraModel.get_record(camera_id), None)
        name = getattr(rec, "display_name", "") if rec else ""
        self._people.set_target_camera(camera_id, name)

    def _on_camera_info_requested(self, camera_id: str) -> None:
        self._on_camera_selected(camera_id)
        info = self._docks.get("camera_info")
        if info is not None:
            info.show()
            info.raise_()

    # ── tracking control ─────────────────────────────────────────────────────────

    def _stop_all_tracking(self) -> None:
        """Disable auto-follow on every camera (cameras keep streaming)."""
        ids = _safe(lambda: list(self._client.cameraModel.camera_ids()), []) or []
        stopped = 0
        for cid in ids:
            try:
                self._client.enableTracking(cid, False)
                stopped += 1
            except Exception:  # noqa: BLE001
                log.debug("stop tracking failed for %s", cid, exc_info=True)
        if stopped:
            self.statusBar().showMessage(
                f"Stopped tracking on {stopped} camera(s).", 4000
            )

    # ── engine state ───────────────────────────────────────────────────────────

    def _refresh_engine_state(self) -> None:
        running = bool(_safe(lambda: self._client.engineRunning, False))
        self._act_start.setEnabled(not running)
        self._act_stop.setEnabled(running)
        self._act_restart.setEnabled(running)
        # "Stop All Tracking" only makes sense with a live engine and ≥1 camera.
        cams = _safe(lambda: len(self._client.cameraModel.camera_ids()), 0) or 0
        self._act_stop_tracking.setEnabled(running and cams > 0)
        self._refresh_startup_progress()

    def _refresh_startup_progress(self) -> None:
        bar = getattr(self, "_startup_bar", None)
        if bar is None:
            return
        active = bool(_safe(lambda: self._client.startupActive, False))
        phase = str(_safe(lambda: self._client.startupPhase, "") or "")
        started = int(_safe(lambda: self._client.startupStartedCameras, 0) or 0)
        total = int(_safe(lambda: self._client.startupTotalCameras, 0) or 0)
        missing = list(_safe(lambda: self._client.startupMissingComponents, []) or [])
        if active:
            if total > 0:
                bar.setRange(0, total)
                bar.setValue(max(0, min(total, started)))
                bar.setFormat(f"{phase} · {started}/{total}")
            else:
                bar.setRange(0, 0)
                bar.setFormat(phase or "Starting engine")
            bar.setVisible(True)
            if missing and not self._shown_optional_setup_prompt:
                self._shown_optional_setup_prompt = True
                dock = self._docks.get("services")
                if dock is not None:
                    dock.show()
                    dock.raise_()
                self.statusBar().showMessage(
                    "Optional tracking components are missing. Review Services setup.",
                    8000,
                )
        else:
            bar.setVisible(False)

    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 6000)

    # ── geometry persistence (dock-layout persistence lands in Phase 7) ────────

    def _restore_geometry(self) -> None:
        geo = _safe(lambda: self._client.getSetting("win_geometry", ""), "")
        if isinstance(geo, str) and geo:
            try:
                self.restoreGeometry(QByteArray(base64.b64decode(geo)))
            except Exception:  # noqa: BLE001
                log.debug("restoreGeometry failed", exc_info=True)
        state = _safe(lambda: self._client.getSetting("win_state", ""), "")
        if isinstance(state, str) and state:
            try:
                self.restoreState(QByteArray(base64.b64decode(state)))
            except Exception:  # noqa: BLE001
                log.debug("restoreState failed", exc_info=True)
            # restoreState can rebuild the right-side tab bar; re-assert the
            # top segmented look.
            self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea,
                                QTabWidget.TabPosition.North)
            self._style_section_tabs()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        # Qt can defer creating the dock tab bar until first show; restyle it
        # once it exists so the segmented look is guaranteed on screen.
        super().showEvent(event)
        self._style_section_tabs()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        try:
            self._client.setSetting(
                "win_geometry", bytes(self.saveGeometry().toBase64()).decode("ascii"),
            )
            self._client.setSetting("win_state", self._encode_state())
        except Exception:  # noqa: BLE001
            log.debug("save geometry/state failed", exc_info=True)
        super().closeEvent(event)


# ── helpers ──────────────────────────────────────────────────────────────────


def _action(
    parent: QWidget, text: str, slot: Any,
    shortcut: str | None = None, tip: str | None = None,
) -> QAction:
    act = QAction(text, parent)
    act.triggered.connect(lambda _checked=False: slot())
    if shortcut:
        act.setShortcut(shortcut)
    if tip:
        act.setToolTip(tip)
        act.setStatusTip(tip)
    return act


def _connect(obj: Any, signal_name: str, slot: Any) -> None:
    try:
        getattr(obj, signal_name).connect(slot)
    except Exception:  # noqa: BLE001
        log.debug("could not connect %s", signal_name, exc_info=True)


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default
