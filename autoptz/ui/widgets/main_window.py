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
import threading
from typing import Any

from PySide6.QtCore import QByteArray, QObject, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QGuiApplication
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QProgressBar,
    QProgressDialog,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.camera_info_panel import CameraInfoPanel
from autoptz.ui.widgets.camera_wall import CameraWall
from autoptz.ui.widgets.common import on_theme_changed
from autoptz.ui.widgets.dialogs import (
    AboutDialog,
    ExperimentalFeaturesDialog,
    ModelManagerDialog,
    NetworkCameraDialog,
)
from autoptz.ui.widgets.dialogs.model_manager import (
    model_setup_reminder_suppressed,
    startup_missing_model_keys,
)
from autoptz.ui.widgets.dialogs.update_dialog import UpdateDialog
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
        # Discovery state: a running modal scan (USB rescan / NDI rescan), plus
        # USB and NDI source lists kept populated in the background so opening
        # the Cameras menu never probes devices on the GUI thread.
        self._scan_ctx: tuple | None = None
        self._usb_devices_cache: list = []
        self._usb_scanning = False
        self._usb_task: _ScanTask | None = None
        self._ndi_sources_cache: list = []
        self._ndi_scanning = False
        self._ndi_task: _ScanTask | None = None

        self.setWindowTitle("AutoPTZ")
        self.setDockNestingEnabled(True)
        self.resize(1320, 820)

        # Update checker (notify-only): created before menus reference it; fired
        # once on first show.
        from autoptz import version as _app_version
        from autoptz.ui.update_manager import UpdateManager

        self._updates = UpdateManager(client, _app_version(), self)
        self._updates.checkStarted.connect(self._on_update_check_started)
        self._updates.updateAvailable.connect(self._on_update_available)
        self._updates.upToDate.connect(self._on_up_to_date)
        self._updates.checkFailed.connect(self._on_update_check_failed)
        self._updates.downloadStarted.connect(self._on_update_download_started)
        self._updates.downloadProgress.connect(self._on_update_download_progress)
        self._updates.downloadFinished.connect(self._on_update_download_finished)
        self._updates.downloadFailed.connect(self._on_update_download_failed)
        self._update_progress: Any | None = None
        self._update_check_busy: Any | None = None
        self._startup_update_checked = False

        self._build_central()
        self._build_docks()
        self._build_menus()
        self._build_status_bar()

        _connect(client, "engineStateChanged", self._refresh_engine_state)
        _connect(client, "startupProgressChanged", self._refresh_startup_progress)
        _connect(client, "modelDownloadStarted", self._on_model_download_started)
        _connect(client, "modelDownloadProgress", self._on_model_download_progress)
        _connect(client, "modelDownloadFinished", self._on_model_download_finished)
        _connect(client, "errorOccurred", self._on_error)
        # Segmented section tabs bake in literal palette colors, so restyle them
        # whenever the appearance flips.
        on_theme_changed(client, self._style_section_tabs)
        on_theme_changed(client, self._style_startup_banner)

        self._ensure_desktop_window_chrome()
        self._restore_geometry()
        self._refresh_engine_state()

        # Start discovery in the background so the Cameras menu is already useful
        # when first opened (deferred to the event loop).
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, self._refresh_usb_async)
        QTimer.singleShot(0, self._refresh_ndi_async)

        # Keep the USB cache hot in the background so plugging/unplugging a camera
        # is reflected the FIRST time the menu is opened — previously the menu showed
        # the stale cache and only kicked off a scan on open, so a hotplug took a few
        # reopens to appear.  Only when enumeration is cheap (macOS/AVFoundation lists
        # devices without opening them); the cross-platform fallback opens each device,
        # which is too costly to poll, so there we keep the on-open refresh.
        self._usb_poll_timer: QTimer | None = None
        if _safe(lambda: self._client.usbEnumerationCheap(), False):
            self._usb_poll_timer = QTimer(self)
            self._usb_poll_timer.setInterval(3000)
            self._usb_poll_timer.timeout.connect(self._refresh_usb_async)
            self._usb_poll_timer.start()

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
        holder.setObjectName("mainContent")
        holder.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        col = QVBoxLayout(holder)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        self._startup_bar = QFrame(holder)
        self._startup_bar.setObjectName("startupBar")
        self._startup_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._startup_bar.setVisible(False)
        bar_col = QVBoxLayout(self._startup_bar)
        bar_col.setContentsMargins(14, 8, 14, 8)
        bar_col.setSpacing(6)
        text_row = QHBoxLayout()
        text_row.setSpacing(8)
        self._startup_dot = QLabel("●")
        self._startup_dot.setObjectName("startupDot")
        self._startup_banner = QLabel()
        self._startup_banner.setObjectName("startupBanner")
        self._startup_banner.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        text_row.addWidget(self._startup_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        text_row.addWidget(self._startup_banner, 1)
        bar_col.addLayout(text_row)
        self._startup_progress_bar = QProgressBar()
        self._startup_progress_bar.setObjectName("startupProgress")
        self._startup_progress_bar.setTextVisible(False)
        self._startup_progress_bar.setFixedHeight(4)
        bar_col.addWidget(self._startup_progress_bar)
        self._style_startup_banner()
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
            (
                "camera_info",
                "Camera Info",
                self._camera_info,
                Qt.DockWidgetArea.RightDockWidgetArea,
            ),
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
        self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea, QTabWidget.TabPosition.North)
        self._style_section_tabs()

        # Sensible first-run proportions (a saved layout overrides these later).
        self.resizeDocks(
            [self._docks["properties"], self._docks["camera_info"]],
            [330, 360],
            Qt.Orientation.Horizontal,
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

    def _style_startup_banner(self) -> None:
        bar = getattr(self, "_startup_bar", None)
        if bar is None:
            return
        accent = T.ACCENT.name()
        bar.setStyleSheet(
            f"QFrame#startupBar {{ background: {T.CURRENT.surface_alt};"
            f" border-left: 3px solid {accent};"
            f" border-bottom: 1px solid {T.CURRENT.border}; }}"
            f"QLabel#startupBanner {{ background: transparent; color: {T.CURRENT.text};"
            " font-weight: 700; }"
            f"QLabel#startupDot {{ background: transparent; color: {accent};"
            f" font-size: {T.fs(13)}px; }}"
            f"QProgressBar#startupProgress {{ background: {T.CURRENT.surface_hov};"
            f" border: none; border-radius: 2px; }}"
            f"QProgressBar#startupProgress::chunk {{ background: {accent};"
            " border-radius: 2px; }"
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
            self,
            "Start",
            self._client.startEngine,
            "Ctrl+E",
            "Start the detection/tracking engine and open all enabled cameras.",
        )
        self._act_stop = _action(
            self,
            "Stop",
            self._client.stopEngine,
            "Ctrl+Shift+E",
            "Stop the engine and release all cameras.",
        )
        self._act_restart = _action(
            self,
            "Restart",
            self._client.restartEngine,
            "Ctrl+Shift+R",
            "Restart the engine — re-applies model and execution-provider settings.",
        )
        engine.addAction(self._act_start)
        engine.addAction(self._act_stop)
        engine.addAction(self._act_restart)
        engine.addSeparator()
        engine.addAction(
            _action(
                self,
                "Models...",
                self._open_model_manager,
                tip=(
                    "Open model setup with selectable detector/pose models, "
                    "cache status, download/remove actions, and reminder settings."
                ),
            )
        )
        engine.addSeparator()
        self._act_stop_tracking = _action(
            self,
            "Stop All Tracking",
            self._stop_all_tracking,
            "Ctrl+Shift+T",
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
        group = QActionGroup(self)
        group.setExclusive(True)
        current = _safe(lambda: str(self._client.themeMode), "dark")
        tips = {
            "system": "Follow the OS light/dark setting.",
            "dark": "Dark broadcast palette (easiest on the eyes).",
            "light": "Light palette for bright rooms.",
        }
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
            ("detection", "Detection boxes", "Show a box around every detected person."),
            ("faces", "Face boxes", "Show face-recognition boxes with the matched name."),
            ("pose", "Pose skeleton", "Draw the tracked person's body skeleton."),
            (
                "prediction",
                "Motion prediction",
                "Debug overlay: draw the target prediction ghost/lead indicator.",
            ),
        ):
            act = QAction(label, self, checkable=True)
            act.setChecked(bool(cur.get(key, key == "detection")))
            act.setToolTip(tip)
            act.toggled.connect(lambda on, k=key: self._client.setOverlay(k, on))
            overlays.addAction(act)

        # Quick collapse/expand of the two side panels (mirrored on the status bar).
        view.addSeparator()
        self._act_toggle_left = QAction("Show Left Panel", self, checkable=True)
        self._act_toggle_left.setShortcut("Ctrl+Alt+[")
        self._act_toggle_left.setToolTip("Show or hide the left Properties panel.")
        self._act_toggle_left.setStatusTip(self._act_toggle_left.toolTip())
        self._act_toggle_left.toggled.connect(self._set_left_panel_visible)
        view.addAction(self._act_toggle_left)
        self._act_toggle_right = QAction("Show Right Panel", self, checkable=True)
        self._act_toggle_right.setShortcut("Ctrl+Alt+]")
        self._act_toggle_right.setToolTip(
            "Show or hide the right Camera Info / People / Services panel."
        )
        self._act_toggle_right.setStatusTip(self._act_toggle_right.toolTip())
        self._act_toggle_right.toggled.connect(self._set_right_panel_visible)
        view.addAction(self._act_toggle_right)

        # Panels (dock toggles) and Layouts are view concerns, so they live under
        # View rather than as separate top-level menus.
        view.addSeparator()
        panels = view.addMenu("Panels")
        for key in ("properties", "camera_info", "people", "services", "logs"):
            panels.addAction(self._docks[key].toggleViewAction())

        self._layouts_menu = view.addMenu("Layouts")
        self._layouts_menu.setToolTipsVisible(True)
        self._layouts_menu.aboutToShow.connect(self._populate_layouts_menu)

        helpm = bar.addMenu("&Help")
        updates = helpm.addMenu("Updates")
        updates.setToolTipsVisible(True)
        updates.addAction(
            _action(
                self,
                "Check Now…",
                self._updates.check_now,
                tip="Check GitHub for a newer AutoPTZ release.",
            )
        )
        updates.addSeparator()
        self._act_auto_check = QAction("Check Automatically on Startup", self, checkable=True)
        self._act_auto_check.setChecked(self._updates.auto_check_enabled)
        self._act_auto_check.setToolTip(
            "Check GitHub for a newer release once a day when AutoPTZ starts."
        )
        self._act_auto_check.setStatusTip(self._act_auto_check.toolTip())
        self._act_auto_check.toggled.connect(self._updates.set_auto_check)
        updates.addAction(self._act_auto_check)
        self._act_prereleases = QAction("Include Pre-release (Beta) Updates", self, checkable=True)
        self._act_prereleases.setChecked(self._updates.include_prereleases)
        self._act_prereleases.setToolTip(
            "Also offer beta / release-candidate builds when checking for updates."
        )
        self._act_prereleases.setStatusTip(self._act_prereleases.toolTip())
        self._act_prereleases.toggled.connect(self._updates.set_include_prereleases)
        updates.addAction(self._act_prereleases)
        helpm.addSeparator()
        helpm.addAction(
            _action(
                self,
                "Experimental Features…",
                self._show_experimental,
                tip="Toggle experimental engine flags and new-camera tracking defaults.",
            )
        )
        helpm.addAction(_action(self, "About AutoPTZ", self._show_about))

    def _build_scale_menu(self, view: QMenu) -> None:
        """View → UI Scale: discrete steps + Zoom In/Out/Reset shortcuts."""
        scale = view.addMenu("UI Scale")
        scale.setToolTipsVisible(True)
        self._scale_group = QActionGroup(self)
        self._scale_group.setExclusive(True)
        steps = (
            getattr(self._theme, "SCALE_STEPS", (0.9, 1.0, 1.1, 1.25, 1.5))
            if self._theme is not None
            else (1.0,)
        )
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
        devices = list(getattr(self, "_usb_devices_cache", []) or [])
        if not devices:
            label = "Scanning for USB cameras…" if self._usb_scanning else "No USB cameras found"
            none = usb.addAction(label)
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
            # Check state from the LIVE camera model (like the NDI rows below), not
            # the scan cache's ``in_use`` — that field is only recomputed when a
            # scan finishes, so right after toggling a camera the checkmark lagged
            # by a scan cycle (you had to reopen the menu a few times).  add/remove
            # update the model synchronously, so a live lookup is always current.
            act.setChecked(
                self._find_camera(str(dev.get("uri", "")), str(dev.get("unique_id", "") or ""))
                is not None
            )
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
            act.toggled.connect(lambda checked, d=dev: self._toggle_usb_camera(d, checked))
            usb.addAction(act)
        usb.addSeparator()
        usb.addAction(
            _action(
                self,
                "Rescan",
                self._rescan_usb,
                tip="Re-probe connected USB cameras.",
            )
        )
        self._refresh_usb_async()

        menu.addSeparator()
        ndi = menu.addMenu("NDI Sources")
        ndi.setToolTipsVisible(True)
        if not _safe(lambda: self._client.ndiAvailable(), False):
            # Honest, persistent reason instead of a fleeting "none found" message.
            unavail = ndi.addAction("NDI unavailable — install cyndilib")
            unavail.setEnabled(False)
            unavail.setToolTip(
                "NDI discovery needs the 'cyndilib' package. Install it, then reopen this menu."
            )
        else:
            # List discovered sources directly (check to add, uncheck to remove),
            # exactly like USB — populated from the background discovery cache.
            sources = list(self._ndi_sources_cache)
            if not sources:
                msg = "Scanning for NDI sources…" if self._ndi_scanning else "No NDI sources found"
                placeholder = ndi.addAction(msg)
                placeholder.setEnabled(False)
            for src in sources:
                name = str(src.get("name", src.get("uri", "?")))
                uri = str(src.get("uri", ""))
                act = QAction(name, self, checkable=True)
                act.setChecked(self._find_camera(uri, "") is not None)
                act.setToolTip(
                    "NDI network video source. Network-dependent and heavier to "
                    "decode than a local USB camera."
                )
                act.toggled.connect(lambda checked, s=src: self._toggle_ndi_camera(s, checked))
                ndi.addAction(act)
            ndi.addSeparator()
            ndi.addAction(
                _action(
                    self,
                    "Rescan",
                    self._rescan_ndi,
                    tip="Re-scan the network for NDI sources.",
                )
            )
            # Refresh the cache in the background so the next open is current.
            self._refresh_ndi_async()

        menu.addSeparator()
        menu.addAction(
            _action(
                self,
                "Add Network Camera…",
                self._add_network_camera,
                tip="Add an RTSP / ONVIF IP camera by URL. Decoding a network stream "
                "uses more CPU than a local USB camera.",
            )
        )

    # ── camera actions ───────────────────────────────────────────────────────────

    def _toggle_usb_camera(self, dev: dict, checked: bool) -> None:
        uri = dev.get("uri", "")
        unique_id = dev.get("unique_id", "")
        if checked:
            # Pass the uniqueID straight through (the menu knows it for this exact
            # device) so selection never depends on a scan-cache lookup by uri.
            self._client.addCamera(
                uri,
                dev.get("name", ""),
                unique_id,
                dev.get("source_label", ""),
            )
        else:
            cid = self._find_camera(uri, unique_id)
            if cid:
                self._client.removeCamera(cid)

    def _find_camera(self, uri: str, unique_id: str) -> str | None:
        model = self._client.cameraModel
        cameras: list[tuple[str, str, str, str]] = []
        for cid in _safe(lambda: model.camera_ids(), []) or []:
            rec = model.get_record(cid)
            cfg = getattr(rec, "camera_config", None)
            src = getattr(cfg, "source", None)
            if src is None:
                continue
            cameras.append(
                (
                    cid,
                    str(getattr(src, "type", "") or ""),
                    str(getattr(src, "unique_id", "") or ""),
                    str(getattr(src, "address", "") or ""),
                )
            )
        return _match_camera_id(cameras, uri, unique_id)

    def _run_scan(self, label: str, work: Any, on_done: Any) -> None:
        """Run a blocking discovery ``work()`` off the GUI thread with a busy dialog.

        Shows an indeterminate "searching…" progress dialog so the user can see it
        is doing something (the scans block for ~1–2 s), runs ``work`` on a worker
        thread, and calls ``on_done(result)`` on the GUI thread when it finishes.
        Single-flight: a second request while one is running is ignored.
        """
        if getattr(self, "_scan_ctx", None) is not None:
            return  # a scan is already running
        dlg = QProgressDialog(label, "", 0, 0, self)  # range 0,0 → busy spinner
        dlg.setWindowTitle("Searching…")
        dlg.setCancelButton(None)  # no cancel: the scans are short and uncancellable
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()

        task = _ScanTask(work)
        self._scan_ctx = (dlg, on_done, task)  # keep refs alive until done
        task.done.connect(self._on_scan_finished)
        threading.Thread(target=task.run, name="ui-scan", daemon=True).start()

    def _on_scan_finished(self, result: Any) -> None:
        ctx = getattr(self, "_scan_ctx", None)
        if ctx is None:
            return
        dlg, on_done, _task = ctx
        self._scan_ctx = None
        dlg.close()
        try:
            on_done(result or [])
        except Exception:  # noqa: BLE001
            log.debug("scan completion handler failed", exc_info=True)

    def _rescan_usb(self) -> None:
        # Re-probe USB devices off the GUI thread (it opens VideoCapture handles
        # and can take ~1 s) with a busy indicator, then reopen the refreshed menu.
        def _done(devices: list) -> None:
            self._usb_devices_cache = devices or []
            self._open_cameras_menu()

        self._run_scan(
            "Re-probing USB cameras…",
            lambda: _safe(lambda: self._client.scanUSBCameras(), []) or [],
            _done,
        )

    def _refresh_usb_async(self) -> None:
        """Refresh the USB camera cache in the background without a dialog."""
        if self._usb_scanning:
            return
        self._usb_scanning = True
        task = _ScanTask(lambda: _safe(lambda: self._client.scanUSBCameras(), []) or [])
        self._usb_task = task
        task.done.connect(self._on_usb_refresh)
        threading.Thread(target=task.run, name="ui-usb-scan", daemon=True).start()

    def _on_usb_refresh(self, devices: list) -> None:
        self._usb_scanning = False
        self._usb_task = None
        self._usb_devices_cache = devices or []

    def _toggle_ndi_camera(self, src: dict, checked: bool) -> None:
        uri = str(src.get("uri", ""))
        name = str(src.get("name", ""))
        if checked:
            self._client.addCamera(uri, name)
        else:
            cid = self._find_camera(uri, "")
            if cid:
                self._client.removeCamera(cid)

    def _refresh_ndi_async(self) -> None:
        """Refresh the NDI source cache in the background (no dialog).

        Keeps the Cameras → NDI Sources list populated without blocking the GUI.
        Single-flight via ``_ndi_scanning``; a no-op when NDI is unavailable.
        """
        if self._ndi_scanning or not _safe(lambda: self._client.ndiAvailable(), False):
            return
        self._ndi_scanning = True
        task = _ScanTask(lambda: _safe(lambda: self._client.scanNDISources(), []) or [])
        self._ndi_task = task  # keep a ref so the QObject isn't GC'd mid-flight
        task.done.connect(self._on_ndi_refresh)
        threading.Thread(target=task.run, name="ui-ndi-scan", daemon=True).start()

    def _on_ndi_refresh(self, sources: list) -> None:
        self._ndi_scanning = False
        self._ndi_task = None
        self._ndi_sources_cache = sources or []

    def _rescan_ndi(self) -> None:
        """User-triggered NDI rescan with a visible busy indicator."""
        if not _safe(lambda: self._client.ndiAvailable(), False):
            self.statusBar().showMessage("NDI unavailable — install cyndilib.", 8000)
            return

        def _done(sources: list) -> None:
            self._ndi_sources_cache = sources
            if not sources:
                self.statusBar().showMessage(
                    "No NDI sources found on the network (is an NDI source running?).", 6000
                )
            self._open_cameras_menu()  # reopen with the refreshed NDI list

        self._run_scan(
            "Searching the network for NDI sources…",
            lambda: _safe(lambda: self._client.scanNDISources(), []) or [],
            _done,
        )

    def _add_network_camera(self) -> None:
        NetworkCameraDialog(self._client, self).exec()

    def _show_about(self) -> None:
        AboutDialog(self._client, self).exec()

    def _show_experimental(self) -> None:
        ExperimentalFeaturesDialog(self._client, self).exec()

    def _open_model_manager(
        self,
        *,
        startup_prompt: bool = False,
        selected_keys: list[str] | None = None,
    ) -> None:
        ModelManagerDialog(
            self._client,
            startup_prompt=startup_prompt,
            selected_keys=selected_keys,
            parent=self,
        ).exec()
        try:
            self._services._refresh_optional_components()  # noqa: SLF001
            self._services.refresh()
        except Exception:  # noqa: BLE001
            log.debug("refresh services after model dialog failed", exc_info=True)

    def _maybe_show_model_setup_on_startup(self) -> None:
        if self._shown_optional_setup_prompt:
            return
        if model_setup_reminder_suppressed(self._client):
            return
        missing = startup_missing_model_keys()
        if not missing:
            return
        self._shown_optional_setup_prompt = True
        self._open_model_manager(startup_prompt=True, selected_keys=missing)

    def _on_update_available(self, info: Any) -> None:
        self._set_update_check_busy(False)
        from autoptz import version as _app_version

        UpdateDialog(
            info,
            _app_version(),
            on_skip=self._updates.skip_version,
            on_install=self._updates.download,
            parent=self,
        ).exec()

    def _on_update_download_started(self, info: Any) -> None:
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtWidgets import QProgressDialog

        version = str(getattr(info, "version", "") or "new version")
        self._close_update_progress()
        dlg = QProgressDialog(f"Downloading AutoPTZ {version}…", "", 0, 100, self)
        dlg.setWindowTitle("Updating AutoPTZ")
        dlg.setWindowModality(_Qt.WindowModality.WindowModal)
        dlg.setCancelButton(None)  # the download runs to completion in the background
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)
        dlg.setRange(0, 0)  # indeterminate until the first progress tick
        dlg.setValue(0)
        self._update_progress = dlg
        dlg.show()
        self.statusBar().showMessage(f"Downloading AutoPTZ {version}…")

    def _on_update_download_progress(self, done: int, total: int) -> None:
        dlg = self._update_progress
        if dlg is None:
            return
        if total > 0:
            pct = max(0, min(100, int(done * 100 / total)))
            dlg.setRange(0, 100)
            dlg.setValue(pct)
            dlg.setLabelText(f"Downloading AutoPTZ update…  {done >> 20} / {total >> 20} MB")
        else:
            dlg.setRange(0, 0)  # unknown size → keep the busy indicator
            dlg.setLabelText(f"Downloading AutoPTZ update…  {done >> 20} MB")

    def _close_update_progress(self) -> None:
        dlg = self._update_progress
        self._update_progress = None
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()

    def _on_update_download_finished(self, result: Any) -> None:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from autoptz.update.installer import launch_update

        path = getattr(result, "path", None)
        if path is None:
            self._on_update_download_failed(
                "The update downloaded, but no installer path was returned."
            )
            return
        dlg = self._update_progress
        if dlg is not None:
            dlg.setRange(0, 100)
            dlg.setValue(100)
            dlg.setLabelText("Installing update… AutoPTZ will restart.")
        try:
            launch_update(path)
        except Exception as exc:  # noqa: BLE001
            self._on_update_download_failed(str(exc))
            return
        self.statusBar().showMessage("Update started. AutoPTZ will close now.", 5000)
        QTimer.singleShot(700, self._close_update_progress)
        QTimer.singleShot(900, QApplication.quit)

    def _on_update_download_failed(self, message: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        self._close_update_progress()
        self.statusBar().clearMessage()
        QMessageBox.warning(
            self,
            "Update Failed",
            f"{message}\n\nYou can still download the installer from the Releases page.",
        )

    def _on_update_check_started(self, manual: bool) -> None:
        """Show a 'Checking for updates…' indicator while the check runs."""
        self._set_update_check_busy(True)

    def _set_update_check_busy(self, on: bool) -> None:
        """Add/remove an indeterminate progress bar + status message for a check."""
        if on:
            if self._update_check_busy is None:
                bar = QProgressBar()
                bar.setRange(0, 0)  # indeterminate "working" animation
                bar.setMaximumWidth(T.fs(120))
                bar.setMaximumHeight(T.fs(14))
                bar.setTextVisible(False)
                self.statusBar().addWidget(bar)
                self._update_check_busy = bar
            self.statusBar().showMessage("Checking for updates…")
        else:
            bar = self._update_check_busy
            self._update_check_busy = None
            if bar is not None:
                self.statusBar().removeWidget(bar)
                bar.deleteLater()

    def _on_up_to_date(self, manual: bool) -> None:
        self._set_update_check_busy(False)
        from autoptz import version as _app_version

        if manual:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Check for Updates",
                f"You're on the latest version.\n\nAutoPTZ {_app_version()} is up to date.",
            )
        else:
            # Startup check: visible (so it's clearly working) but unobtrusive.
            self.statusBar().showMessage(f"AutoPTZ {_app_version()} is up to date.", 6000)

    def _on_update_check_failed(self, reason: str, manual: bool) -> None:
        self._set_update_check_busy(False)
        if manual:
            from PySide6.QtWidgets import QMessageBox

            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Couldn't Check for Updates")
            box.setText(reason)
            box.setInformativeText(
                "AutoPTZ couldn't reach the update server, so it can't tell whether "
                "you're on the latest version. Check your internet connection and try again."
            )
            retry = box.addButton("Retry", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is retry:
                self._updates.check_now()
        else:
            # Startup check failure: don't interrupt; surface briefly + log.
            self.statusBar().showMessage("Couldn't check for updates (offline?).", 6000)

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
        self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea, QTabWidget.TabPosition.North)
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
            self._client,
            logs_toggle=self._toggle_logs,
            cameras_popup=self._open_cameras_menu,
            left_toggle=self._set_left_panel_visible,
            right_toggle=self._set_right_panel_visible,
        )
        self.statusBar().addPermanentWidget(self._status, 1)
        self.statusBar().setSizeGripEnabled(False)
        logs = self._docks.get("logs")
        if logs is not None:
            logs.visibilityChanged.connect(self._status.set_logs_visible)
            self._status.set_logs_visible(logs.isVisible())
        # Keep both the menu checkmarks and the status-bar buttons in sync with the
        # side panels however they're toggled (menu, button, shortcut, or close box).
        for key in ("properties", *self._RIGHT_PANEL_KEYS):
            dock = self._docks.get(key)
            if dock is not None:
                dock.visibilityChanged.connect(lambda _v: self._sync_panel_toggles())
        self._sync_panel_toggles()

    def _toggle_logs(self, shown: bool | None = None) -> None:
        dock = self._docks.get("logs")
        if dock is None:
            return
        target = not dock.isVisible() if shown is None else shown
        if target:
            dock.show()
            dock.raise_()
        else:
            dock.hide()

    # ── side-panel collapse ──────────────────────────────────────────────────────

    #: The right-hand dock group is three tabbed panels toggled as one unit.
    _RIGHT_PANEL_KEYS = ("camera_info", "people", "services")

    def _set_left_panel_visible(self, shown: bool) -> None:
        dock = self._docks.get("properties")
        if dock is None:
            return
        if shown:
            dock.show()
            dock.raise_()
        else:
            dock.hide()

    def _set_right_panel_visible(self, shown: bool) -> None:
        docks = [d for k in self._RIGHT_PANEL_KEYS if (d := self._docks.get(k)) is not None]
        for dock in docks:
            dock.setVisible(shown)
        if shown and docks:
            docks[0].raise_()

    def _sync_panel_toggles(self) -> None:
        """Keep the menu checkmarks and status-bar buttons in step with the docks."""
        left = self._docks.get("properties")
        left_on = bool(left is not None and left.isVisible())
        right_on = any(
            (d := self._docks.get(k)) is not None and d.isVisible() for k in self._RIGHT_PANEL_KEYS
        )
        for act, on in (
            (getattr(self, "_act_toggle_left", None), left_on),
            (getattr(self, "_act_toggle_right", None), right_on),
        ):
            if act is not None:
                blocked = act.blockSignals(True)
                act.setChecked(on)
                act.blockSignals(blocked)
        status = getattr(self, "_status", None)
        if status is not None:
            status.set_left_visible(left_on)
            status.set_right_visible(right_on)

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
            self.statusBar().showMessage(f"Stopped tracking on {stopped} camera(s).", 4000)

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
        banner = getattr(self, "_startup_banner", None)
        progress = getattr(self, "_startup_progress_bar", None)
        if bar is None or banner is None or progress is None:
            return
        active = bool(_safe(lambda: self._client.startupActive, False))
        phase = str(_safe(lambda: self._client.startupPhase, "") or "")
        started = int(_safe(lambda: self._client.startupStartedCameras, 0) or 0)
        total = int(_safe(lambda: self._client.startupTotalCameras, 0) or 0)
        if not active:
            bar.setVisible(False)
            return
        if total > 0:
            banner.setText(f"{phase or 'Starting engine'} ({started}/{total})")
            # Determinate while cameras open; full once they're all up (warmup).
            progress.setRange(0, total)
            progress.setValue(min(started, total))
        else:
            banner.setText(phase or "Starting engine")
            # No camera count yet → indeterminate "working" sweep.
            progress.setRange(0, 0)
        bar.setVisible(True)

    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 6000)

    def _on_model_download_started(self, label: str) -> None:
        self.statusBar().showMessage(label)

    def _on_model_download_progress(self, label: str, value: int, total: int) -> None:
        total = max(1, int(total))
        value = max(0, min(total, int(value)))
        self.statusBar().showMessage(f"{label} ({value}/{total})")

    def _on_model_download_finished(self, ok: bool, message: str) -> None:
        self.statusBar().showMessage(message, 7000 if ok else 10000)

    # ── geometry persistence ────────────────────────────────────────────────

    def _ensure_desktop_window_chrome(self) -> None:
        """Keep the main shell as a normal native desktop window."""
        flags = self.windowFlags()
        flags &= ~Qt.WindowType.FramelessWindowHint
        flags &= ~Qt.WindowType.NoDropShadowWindowHint
        flags |= Qt.WindowType.Window
        flags |= Qt.WindowType.WindowTitleHint
        flags |= Qt.WindowType.WindowSystemMenuHint
        flags |= Qt.WindowType.WindowMinimizeButtonHint
        flags |= Qt.WindowType.WindowMaximizeButtonHint
        flags |= Qt.WindowType.WindowCloseButtonHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    def _restore_geometry(self) -> None:
        restored_geometry = False
        geo = _safe(lambda: self._client.getSetting("win_geometry", ""), "")
        if isinstance(geo, str) and geo:
            try:
                restored_geometry = bool(self.restoreGeometry(QByteArray(base64.b64decode(geo))))
            except Exception:  # noqa: BLE001
                log.debug("restoreGeometry failed", exc_info=True)
        self._clear_minimized_state()
        if not restored_geometry or not self._has_visible_window_frame():
            self.resize(1320, 820)
            self._center_on_primary_screen()
        state = _safe(lambda: self._client.getSetting("win_state", ""), "")
        if isinstance(state, str) and state:
            try:
                self.restoreState(QByteArray(base64.b64decode(state)))
            except Exception:  # noqa: BLE001
                log.debug("restoreState failed", exc_info=True)
            # restoreState can rebuild the right-side tab bar; re-assert the
            # top segmented look.
            self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea, QTabWidget.TabPosition.North)
            self._style_section_tabs()
        self._clear_minimized_state()
        if not self._has_visible_window_frame():
            self.resize(1320, 820)
            self._center_on_primary_screen()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        # Qt can defer creating the dock tab bar until first show; restyle it
        # once it exists so the segmented look is guaranteed on screen.
        super().showEvent(event)
        self._style_section_tabs()
        self._clear_minimized_state()
        if not self._has_visible_window_frame():
            self.resize(1320, 820)
            self._center_on_primary_screen()
        # Fire the throttled update check once, deferred so first paint isn't blocked.
        if not self._startup_update_checked:
            self._startup_update_checked = True
            from PySide6.QtCore import QTimer

            QTimer.singleShot(900, self._maybe_show_model_setup_on_startup)
            QTimer.singleShot(2500, self._updates.maybe_check_on_startup)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        try:
            self._client.setSetting(
                "win_geometry",
                bytes(self.saveGeometry().toBase64()).decode("ascii"),
            )
            self._client.setSetting("win_state", self._encode_state())
        except Exception:  # noqa: BLE001
            log.debug("save geometry/state failed", exc_info=True)
        super().closeEvent(event)

    def _clear_minimized_state(self) -> None:
        state = self.windowState()
        if state & Qt.WindowState.WindowMinimized:
            self.setWindowState(state & ~Qt.WindowState.WindowMinimized)

    def _has_visible_window_frame(self) -> bool:
        frame = self.frameGeometry()
        if frame.width() < 320 or frame.height() < 240:
            return False
        screens = QGuiApplication.screens()
        if not screens:
            return True
        for screen in screens:
            visible = frame.intersected(screen.availableGeometry())
            if visible.width() >= 320 and visible.height() >= 200:
                return True
        return False

    def _center_on_primary_screen(self) -> None:
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())


# ── helpers ──────────────────────────────────────────────────────────────────


class _ScanTask(QObject):
    """Runs a blocking discovery callable off the GUI thread, emitting its result.

    ``done`` is emitted from the worker thread; connecting it to a bound method of
    a GUI-thread QObject (e.g. the MainWindow) gives a queued connection, so the
    completion handler runs safely on the GUI thread.
    """

    done = Signal(object)

    def __init__(self, work: Any) -> None:
        super().__init__()
        self._work = work

    def run(self) -> None:
        try:
            result = self._work()
        except Exception:  # noqa: BLE001 — a scan failure must surface as "none found"
            log.debug("scan task failed", exc_info=True)
            result = []
        self.done.emit(result)


def _action(
    parent: QWidget,
    text: str,
    slot: Any,
    shortcut: str | None = None,
    tip: str | None = None,
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


def _match_camera_id(
    cameras: list[tuple[str, str, str, str]],
    uri: str,
    unique_id: str,
) -> str | None:
    """Resolve a scanned device → the matching camera id, or None.

    ``cameras`` is ``(camera_id, source_type, source_unique_id, source_address)``.

    USB cameras are identified **only by their stable ``unique_id``**: the
    ``usb://<index>`` address is volatile (it shifts when a device is replugged or
    another camera is added), so matching USB by address could resolve to — and
    then disable/delete — the *wrong* camera after enumeration changes (the
    reported bug).  As a last resort, an id-less USB device matches another id-less
    USB camera at the same address, but never one that carries a different known id.

    Network sources (rtsp/onvif/ndi) keep address matching — their address (URL /
    NDI name) *is* their stable identity.
    """
    # unique_id match first (works for any source type).
    if unique_id:
        for cid, _stype, suid, _addr in cameras:
            if suid and suid == unique_id:
                return cid
    # Address fallback, scoped by source type.
    for cid, stype, suid, addr in cameras:
        if addr != uri:
            continue
        if stype == "usb":
            # Only when BOTH sides lack a stable id (never override a known id).
            if not unique_id and not suid:
                return cid
        else:
            return cid
    return None
