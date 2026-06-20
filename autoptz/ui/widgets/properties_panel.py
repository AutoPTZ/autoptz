"""PropertiesPanel — per-camera inspector (native, collapsible, cost-aware).

Loads the selected camera's config (``getCameraConfig``), edits it through native
widgets grouped in collapsible sections, and writes back debounced via
``updateCameraConfig``.  Intensive settings carry a Light/Medium/Heavy cost chip
with a tooltip so the user can see what's expensive.  A compact manual PTZ pad
nudges the camera; Remove deletes it (with confirmation).
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.common import (
    CollapsibleGroup,
    CostChip,
    DangerButton,
    HelpBadge,
    data_uri_to_pixmap,
    on_theme_changed,
)
from autoptz.ui.widgets.joystick import JoystickPad

log = logging.getLogger(__name__)

_NUDGE = 0.6
_PRESET_COUNT = 6           # quick-recall slots shown as tiles in the PTZ section
_PRESET_THUMB = 84          # preset tile thumbnail edge (px)

# Friendly labels for the unified Framing control (value → caption shown to user).
_FRAMING_CHOICES = [
    ("face", "Face"),
    ("head_shoulders", "Head & Shoulders"),
    ("upper_body", "Upper body"),
    ("full_body", "Full body"),
]

_TRACKING_MODE_CHOICES = [
    ("stable", "Stable target"),
    ("responsive", "Responsive"),
]

_TRACKER_HELP = {
    "botsort": "BoT-SORT: best default for people. Uses motion plus optional appearance cues; steady but medium cost.",
    "deepocsort": "DeepOCSORT: stronger through occlusion and crossing people; usually heavier than BoT-SORT.",
    "bytetrack": "ByteTrack: fastest/lightest. Good boxes, less robust when people overlap or disappear.",
}

# Plain-English cost notes shown as tooltips next to the chips.
_COST_HELP = {
    "detect_interval": "How often the detector runs. Every frame (1) is heaviest; "
                       "higher values skip frames and cost far less CPU.",
    "tracker": "\n".join(_TRACKER_HELP.values()),
    "reid": "Appearance re-identification recovers a target after occlusion — "
            "accurate but the most expensive option (runs an extra model).",
    "face_confirm": "Confirms identity with face recognition — moderate extra cost.",
}


class _PresetTile(QWidget):
    """One PTZ preset slot rendered as a snapshot thumbnail + label tile.

    * Empty slots read as a dashed "+ Save" placeholder.
    * Occupied slots show the thumbnail (or a numbered placeholder when no
      snapshot was captured) with the label below.
    * Left-click an occupied tile recalls it; left-click an empty tile saves
      into it.  Right-click (or the ⋯ corner button) opens a Save / Overwrite /
      Clear menu.
    """

    recallRequested = Signal(int)   # slot index
    saveRequested = Signal(int)     # slot index
    clearRequested = Signal(int)    # slot index

    def __init__(self, slot: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.slot = slot
        self._occupied = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        self._thumb = QLabel()
        self._thumb.setObjectName("presetThumb")
        self._thumb.setFixedSize(_PRESET_THUMB, int(_PRESET_THUMB * 0.62))
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        col.addWidget(self._thumb)

        self._caption = QLabel("")
        self._caption.setObjectName("presetLabel")
        self._caption.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._caption.setFixedWidth(_PRESET_THUMB)
        col.addWidget(self._caption)

        # ⋯ overflow menu pinned to the thumbnail's top-right corner.
        self._menu_btn = QPushButton("⋯", self._thumb)
        self._menu_btn.setObjectName("presetMenuBtn")
        _menu_sz = T.fs(18)
        self._menu_btn.setFixedSize(_menu_sz, _menu_sz)
        self._menu_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._menu_btn.setCursor(Qt.CursorShape.ArrowCursor)
        self._menu_btn.move(_PRESET_THUMB - _menu_sz - 2, 2)
        self._menu_btn.clicked.connect(self._open_menu)

        self.restyle()
        self.set_state(occupied=False, label="", thumbnail=None)

    # ── state ────────────────────────────────────────────────────────────────────

    def set_state(self, *, occupied: bool, label: str, thumbnail: str | None) -> None:
        self._occupied = bool(occupied)
        pm = data_uri_to_pixmap(thumbnail, size=_PRESET_THUMB, circular=False) if thumbnail else None
        if pm is not None and not pm.isNull():
            self._thumb.setPixmap(
                pm.scaled(
                    self._thumb.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._thumb.setText("")
        else:
            self._thumb.setPixmap(QPixmap())
            self._thumb.setText(f"{self.slot + 1}" if occupied else "＋")
        self._caption.setText(label if (occupied and label) else
                              ("Preset" if occupied else "Empty"))
        self.setToolTip(
            (f"Preset {self.slot + 1}" + (f" — “{label}”" if label else "") +
             ": click to recall, ⋯ to overwrite/clear.")
            if occupied else
            f"Preset {self.slot + 1} (empty): click to save the current view."
        )
        self.setProperty("occupied", self._occupied)
        self.style().unpolish(self); self.style().polish(self)
        self.restyle()

    def restyle(self) -> None:
        pal = T.CURRENT
        occ = self._occupied
        border = T.TRACKING if occ else pal.border
        style = "solid" if occ else "dashed"
        self._thumb.setStyleSheet(
            f"QLabel#presetThumb {{ background: {pal.surface_hov};"
            f" border: 1px {style} {border}; border-radius: {T.RADIUS}px;"
            f" color: {pal.muted}; font-size: {T.fs(18)}px; font-weight: 700; }}"
        )
        self._caption.setStyleSheet(
            f"color: {pal.subtext if occ else pal.muted}; font-size: {T.fs(10)}px;"
        )
        self._menu_btn.setStyleSheet(
            f"QPushButton#presetMenuBtn {{ background: rgba(0,0,0,140); color: #ffffff;"
            f" border: none; border-radius: 9px; font-weight: 700; }}"
            f"QPushButton#presetMenuBtn:hover {{ background: {T.ACCENT.name()}; }}"
        )

    # ── interaction ──────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if self._occupied:
                self.recallRequested.emit(self.slot)
            else:
                self.saveRequested.emit(self.slot)
        elif event.button() == Qt.MouseButton.RightButton:
            self._open_menu()
        super().mousePressEvent(event)

    def _open_menu(self) -> None:
        menu = QMenu(self)
        if self._occupied:
            menu.addAction("Overwrite (save current view)",
                           lambda: self.saveRequested.emit(self.slot))
            menu.addAction("Recall", lambda: self.recallRequested.emit(self.slot))
            menu.addSeparator()
            menu.addAction("Clear", lambda: self.clearRequested.emit(self.slot))
        else:
            menu.addAction("Save current view here",
                           lambda: self.saveRequested.emit(self.slot))
        menu.exec(self.mapToGlobal(self._menu_btn.geometry().bottomLeft()))


class PropertiesPanel(QWidget):
    """Inspector for the selected camera."""

    def __init__(
        self,
        client: Any,
        parent: QWidget | None = None,
        *,
        frame_source: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("propertiesPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._client = client
        # Optional live-frame handle (a ``ShmFrameSource`` like the camera tiles
        # use).  When supplied, "Save preset" grabs the current frame as a JPEG
        # data-URI thumbnail; when ``None`` the preset still saves, just without a
        # snapshot.  Wire it from the host with ``PropertiesPanel(client,
        # frame_source=frames)``.
        self._frames = frame_source
        self._camera_id = ""
        self._cfg: dict[str, Any] = {}
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Per-widget styled elements that bake T.CURRENT/T.ERROR colors at build
        # time are collected here so :meth:`_restyle_all` can re-apply them on a
        # Light/Dark switch (otherwise they keep the previous palette's colors).
        self._muted_captions: list[QLabel] = []   # muted "caption" labels
        self._ro_values: list[QLabel] = []        # read-only primary-text values

        self._empty = QLabel("Select a camera to edit its settings")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._empty)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        root.addWidget(self._scroll)

        body = QWidget()
        self._col = QVBoxLayout(body)
        self._col.setContentsMargins(10, 10, 10, 10)
        self._col.setSpacing(6)
        self._build_controls()
        self._col.addStretch(1)
        self._scroll.setWidget(body)
        self._scroll.setVisible(False)

        # Debounced write-back.
        self._push_timer = QTimer(self)
        self._push_timer.setSingleShot(True)
        self._push_timer.setInterval(400)
        self._push_timer.timeout.connect(self._push)
        # Fast, near-live push JUST for the framing box (w/h/roundness) so the
        # oval on the tile tracks the slider as you drag it (the 400 ms general
        # debounce felt unresponsive for a direct-manipulation control).
        self._framing_timer = QTimer(self)
        self._framing_timer.setSingleShot(True)
        self._framing_timer.setInterval(40)
        self._framing_timer.timeout.connect(self._push_framing)

        # Apply the literal-color styling for every per-widget styled element once,
        # and re-run it when the user flips Light/Dark (these widgets bake
        # T.CURRENT.*/T.ERROR/T.ACCENT colors that go stale on a theme switch).
        self._apply_track_label(self._track_btn.isChecked())
        self._restyle_all()
        on_theme_changed(self._client, self._restyle_all)

        # Keep the target dropdown and the live "measured fps" readout fresh, and
        # mirror tracking on/off when it's flipped elsewhere (tile overlay, Stop
        # All) so the Start/Stop Tracking button never goes stale.
        _connect(self._client, "identitiesChanged", self._reload_targets)
        # Reflect target changes made elsewhere (e.g. clicking a recognized person
        # on the video) in the "Track person" dropdown.
        _connect(self._client, "targetChanged", self._on_external_target_changed)
        _connect(self._client, "trackingChanged", lambda cid: self._sync_track_state()
                 if cid == self._camera_id else None)
        _connect(self._client, "telemetryUpdated", lambda *_: self._on_telemetry())
        # Re-grey Mode▸Stable live when the global ReID feature is toggled.
        _connect(self._client, "featuresChanged", self._refresh_reid_gating)
        # Re-render preset tiles when this camera's config changes (e.g. after a
        # save/clear writes the new ``preset_slots`` back asynchronously).
        _connect(self._client, "configChanged", self._on_config_changed)

    def _restyle_all(self) -> None:
        """Re-apply EVERY per-widget literal-color style from the LIVE palette.

        Routes all elements that bake ``T.CURRENT``/``T.ERROR``/``T.ACCENT`` into
        a ``setStyleSheet`` (track button, measured-fps caption, PTZ pad, preset
        tiles, cost chips, section captions, read-only values, the empty-state
        label, and the Remove button) through one method invoked on construction
        and on every ``themeChanged`` — so nothing keeps a previous appearance's
        colors after a Light↔Dark switch.
        """
        pal = T.CURRENT
        self._empty.setStyleSheet(f"color: {pal.muted};")
        for cap in getattr(self, "_muted_captions", []):
            cap.setStyleSheet(f"color: {pal.muted}; font-size: {T.fs(11)}px;")
        for lab in getattr(self, "_ro_values", []):
            lab.setStyleSheet(f"color: {pal.text};")
        # _remove_btn is a DangerButton — styled by the global stylesheet, so it
        # tracks the theme with no per-widget restyle here.
        self._restyle_track_btn(self._track_btn.isChecked())
        self._restyle_fps_measured()
        self._restyle_ptz_pad()
        for tile in getattr(self, "_preset_tiles", []):
            tile.restyle()
        # Cost chips re-resolve their semantic colors from the active palette.
        self._refresh_cost_chips()

    # ── construction ─────────────────────────────────────────────────────────────

    def _build_controls(self) -> None:
        # General
        g = CollapsibleGroup("General")
        gf = _form()
        self._name = QLineEdit()
        self._name.editingFinished.connect(self._schedule)
        gf.addRow("Name", self._name)
        self._source_type = _ro_value()
        self._ro_values.append(self._source_type)
        gf.addRow("Source", self._source_type)
        self._address = _ro_value()
        self._ro_values.append(self._address)
        gf.addRow("Address", self._address)
        self._substream = QCheckBox("Use camera sub-stream")
        self._substream.setToolTip(
            "Pull the camera's lower-resolution sub-stream (IP cameras). Much "
            "cheaper to decode; slightly less detail for detection."
        )
        self._substream.toggled.connect(self._schedule)
        self._substream_holder = _with_chip(self._substream, HelpBadge(
            "Pulls the camera's lower-resolution sub-stream (IP cameras only). "
            "Much cheaper to decode and keeps the UI responsive; detection loses "
            "a little detail on small/distant people."
        ))
        gf.addRow("", self._substream_holder)
        gf.addRow("Frame rate", _with_chip(self._build_fps_row(), HelpBadge(
            "Frames per second requested from the camera. Higher is smoother for "
            "tracking but costs more CPU/GPU; the slider's max is the source's "
            "detected hardware ceiling, and changes apply live."
        )))
        g.add_widget(_wrap(gf))
        self._col.addWidget(g)

        # Detection
        d = CollapsibleGroup("Detection", expanded=False)
        df = _form()
        self._quality = QComboBox(); self._quality.addItems(["auto", "high", "balanced", "low"])
        self._quality.setToolTip(
            "Minimum detection quality. Higher quality finds smaller/partly-hidden "
            "people but costs more; “auto” adapts to load."
        )
        self._quality.currentTextChanged.connect(self._schedule)
        df.addRow("Quality floor", _with_chip(self._quality, HelpBadge(
            "Minimum detection quality. Higher levels find smaller or partly "
            "hidden people but cost more CPU; “auto” adapts the floor to the "
            "current load."
        )))
        # Live readout of what "auto" is actually resolving to right now (and why).
        self._quality_effective = QLabel("")
        self._quality_effective.setWordWrap(True)
        self._muted_captions.append(self._quality_effective)
        df.addRow("", self._quality_effective)
        self._detect_interval = QSpinBox(); self._detect_interval.setRange(1, 30)
        self._detect_interval.valueChanged.connect(self._schedule)
        self._detect_chip = CostChip("heavy")
        self._detect_chip.setToolTip(_COST_HELP["detect_interval"])
        df.addRow("Detect every N frames", _with_chip(
            self._detect_interval, self._detect_chip, HelpBadge(_COST_HELP["detect_interval"]),
        ))
        # Live readout of the *effective* cadence when the engine auto-adjusts it.
        self._detect_effective = QLabel("")
        self._detect_effective.setWordWrap(True)
        self._muted_captions.append(self._detect_effective)
        df.addRow("", self._detect_effective)
        d.add_widget(_wrap(df))
        self._col.addWidget(d)

        # Overlays — global on-video display toggles (mirror View ▸ Overlays).
        ov = CollapsibleGroup("Overlays")
        ovf = _form()
        self._overlay_boxes: dict[str, QCheckBox] = {}
        cur_ov = _safe(lambda: self._client.overlays(), {}) or {}
        for key, label, tip in (
            ("detection", "Detection boxes",
             "Show a box around every detected person."),
            ("faces", "Face boxes",
             "Show face-recognition boxes with the matched name "
             "(needs Face recognition enabled in Services)."),
            ("pose", "Pose skeleton",
             "Draw the selected person's body skeleton — shows for whoever you "
             "click/track (needs Pose enabled in Services)."),
            ("prediction", "Motion prediction",
             "Debug overlay: draw predicted target motion/ghost box. Off by "
             "default so the target overlay stays one true box plus PTZ aim dot."),
        ):
            cb = QCheckBox(label)
            cb.setChecked(bool(cur_ov.get(key, key == "detection")))
            cb.setToolTip(tip)
            cb.toggled.connect(lambda on, k=key: self._client.setOverlay(k, on))
            self._overlay_boxes[key] = cb
            ovf.addRow("", _with_chip(cb, HelpBadge(tip)))
        ov.add_widget(_wrap(ovf))
        self._col.addWidget(ov)
        _connect(self._client, "overlaysChanged", self._refresh_overlay_boxes)

        # Tracking
        tr = CollapsibleGroup("Tracking", expanded=False)
        # Prominent Start/Stop button — the unmistakable master switch (the #1
        # ask was an obvious STOP).  Big, full-width, accent when off / red when
        # on, with enough height that the descender in "Stop" isn't clipped.
        self._track_btn = QPushButton("Track")
        self._track_btn.setObjectName("trackToggleBtn")
        self._track_btn.setCheckable(True)
        self._track_btn.setMinimumHeight(T.fs(34))
        self._track_btn.setToolTip(
            "Master on/off for detection-driven PTZ tracking on this camera."
        )
        self._track_btn.toggled.connect(self._on_track_toggled)
        tr.add_widget(self._track_btn)
        tf = _form()
        self._target_combo = QComboBox()
        self._target_combo.setToolTip(
            "Lock tracking to a registered person — the camera follows them "
            "whenever they're recognized. Choose “— Anyone —” to follow whoever "
            "is detected."
        )
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        tf.addRow("Track person", _with_chip(self._target_combo, HelpBadge(
            "Lock tracking to a registered person — the camera follows them "
            "whenever they're recognized. Choose “— Anyone —” to follow whoever "
            "is detected. Register people in the Identities panel."
        )))
        self._tracking_mode = QComboBox()
        for value, caption in _TRACKING_MODE_CHOICES:
            self._tracking_mode.addItem(caption, value)
        self._tracking_mode.setToolTip(
            "Stable holds the selected person through crossings using appearance "
            "ReID (needs the ReID feature on in Services). Responsive follows the "
            "freshest detection with no ReID hold and less delay."
        )
        self._tracking_mode.currentIndexChanged.connect(self._schedule)
        self._tracking_mode.currentIndexChanged.connect(self._refresh_reid_gating)
        tf.addRow("Mode", _with_chip(self._tracking_mode, HelpBadge(
            "Stable is best for crowded scenes and uses ReID for person-level "
            "recovery (toggle ReID in Services). Responsive is lower latency and "
            "better for solo scenes, but can switch bodies more easily."
        )))
        # Caption shown when Stable is picked but the global ReID feature is off.
        self._mode_caption = QLabel("")
        self._mode_caption.setWordWrap(True)
        self._muted_captions.append(self._mode_caption)
        tf.addRow("", self._mode_caption)
        self._tracker = QComboBox(); self._tracker.addItems(["botsort", "deepocsort", "bytetrack"])
        self._tracker.setToolTip(_COST_HELP["tracker"])
        self._tracker.currentTextChanged.connect(self._schedule)
        self._tracker.currentTextChanged.connect(self._refresh_tracker_tip)
        self._tracker_chip = CostChip("medium"); self._tracker_chip.setToolTip(_COST_HELP["tracker"])
        tf.addRow("Tracker", _with_chip(
            self._tracker, self._tracker_chip, HelpBadge(_COST_HELP["tracker"]),
        ))
        # Unified "Framing" control — one dropdown that drives BOTH where the
        # camera centers (aim) AND how tight it zooms.  Replaces the old separate
        # "Aim at" + "Zoom framing" dropdowns.
        # Builder, part 1 — the region to frame on (drives aim height + zoom).
        self._framing = QComboBox()
        for value, caption in _FRAMING_CHOICES:
            self._framing.addItem(caption, value)
        self._framing.setToolTip(
            "Where the camera centers on the person and how tightly it zooms — "
            "from a tight face shot to the full body."
        )
        self._framing.currentIndexChanged.connect(self._schedule)
        framing_help = HelpBadge(
            "The region to build the shot around. “Face” aims high on the head and "
            "zooms in close; “Full body” aims at the torso centre and zooms out to "
            "fit the whole person. Drives BOTH the aim point and the auto-zoom "
            "tightness. Pair it with “Ignore arms” below."
        )
        tf.addRow("Frame on", _with_chip(self._framing, framing_help))
        # Builder, part 2 — whether arms are ignored (steady) or included (widen).
        self._ignore_arms = QCheckBox("Ignore arms (steadier framing)")
        self._ignore_arms.setToolTip(
            "On: the aim sits on the body (pose torso) and the zoom stays steady "
            "when arms move — raising a hand won't yank or widen the shot. "
            "Off: the shot widens to include outstretched arms."
        )
        self._ignore_arms.toggled.connect(self._schedule)
        tf.addRow("", _with_chip(self._ignore_arms, HelpBadge(
            "Builds on “Frame on”: choose whether arms are ignored for steadier "
            "framing, or included so reaching out widens the shot. The aim circle "
            "always stays on the body either way."
        )))
        # ReID is no longer a per-camera checkbox: it's the global "reid" feature
        # (Services) combined with the per-camera Mode above ("Stable" uses it).
        self._face = QCheckBox("Confirm with face recognition")
        self._face.toggled.connect(self._schedule)
        self._face_chip = CostChip("medium"); self._face_chip.setToolTip(_COST_HELP["face_confirm"])
        tf.addRow("", _with_chip(
            self._face, self._face_chip, HelpBadge(_COST_HELP["face_confirm"]),
        ))
        tr.add_widget(_wrap(tf))
        self._col.addWidget(tr)

        # PTZ
        pz = CollapsibleGroup("PTZ", expanded=False)
        pf = _form()
        self._backend = QComboBox(); self._backend.addItems(["auto", "ndi", "visca_ip", "visca_usb", "onvif"])
        self._backend.setToolTip(
            "How PTZ move commands reach the camera. “auto” probes NDI → ONVIF → "
            "VISCA-IP; pick a specific one if you know your camera."
        )
        self._backend.currentTextChanged.connect(self._schedule)
        pf.addRow("Backend", _with_chip(self._backend, HelpBadge(
            "How PTZ move commands reach the camera. “auto” probes NDI → ONVIF → "
            "VISCA-IP; pick a specific protocol if you already know what your "
            "camera speaks."
        )))
        self._ptz_address = QLineEdit(); self._ptz_address.editingFinished.connect(self._schedule)
        self._ptz_address.setToolTip("Host:port (IP backends) or serial port (VISCA-USB). Leave blank for auto.")
        pf.addRow("Address", _with_chip(self._ptz_address, HelpBadge(
            "Where to send PTZ commands: host:port for IP backends (NDI / ONVIF / "
            "VISCA-IP) or the serial port for VISCA-USB. Leave blank to let the "
            "backend auto-discover."
        )))
        self._auto_zoom = QCheckBox("Auto-zoom to frame the subject")
        self._auto_zoom.setToolTip(
            "Let the controller zoom in/out to keep the chosen Framing "
            "(set in the Tracking section)."
        )
        self._auto_zoom.toggled.connect(self._schedule)
        pf.addRow("", _with_chip(self._auto_zoom, HelpBadge(
            "Lets the controller zoom in and out automatically to keep the "
            "subject at the chosen Framing tightness (set in the Tracking "
            "section). Turn off to hold a fixed zoom."
        )))
        pz.add_widget(_wrap(pf))
        pz.add_widget(self._build_ptz_controls())
        pz.add_widget(self._build_presets())
        self._col.addWidget(pz)

        # Advanced tracking — collapsed by default so casual users never see it.
        self._col.addWidget(self._build_advanced_tracking())

        # Remove — the shared destructive button (theme-tracked, no inline color).
        self._remove_btn = DangerButton("Remove Camera")
        self._remove_btn.clicked.connect(self._remove)
        self._col.addWidget(self._remove_btn)

    def _build_advanced_tracking(self) -> CollapsibleGroup:
        """Collapsed 'Advanced tracking' tuning — gain, smoothing, prediction, safe zone.

        These map to per-camera ``PTZConfig`` fields and are pushed live to the
        running controller (no restart) via the debounced config write-back.
        """
        adv = CollapsibleGroup("Advanced tracking", expanded=False)
        af = _form()

        def _slider(lo: int, hi: int, tip: str) -> tuple[QWidget, QSlider, QLabel]:
            holder = QWidget()
            row = QHBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            s.setToolTip(tip)
            val = QLabel("")
            val.setMinimumWidth(56)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(s, 1)
            row.addWidget(val)
            return holder, s, val

        gh, self._kp, self._kp_val = _slider(
            10, 150, "How hard the camera corrects toward the subject.")
        self._kp.valueChanged.connect(
            lambda v: (self._kp_val.setText(f"{v / 100:.2f}"), self._schedule()))
        af.addRow("Gain", _with_chip(gh, HelpBadge(
            "Proportional gain (Kp). Higher reacts faster but can overshoot or "
            "oscillate; lower is calmer but slower to catch up.")))

        sh, self._smoothing, self._smoothing_val = _slider(
            0, 100, "Higher = smoother but laggier; lower = snappier.")
        self._smoothing.valueChanged.connect(
            lambda v: (self._smoothing_val.setText(f"{v}%"), self._schedule()))
        af.addRow("Smoothing", _with_chip(sh, HelpBadge(
            "Aim smoothing. Higher rejects jitter but adds lag; lower is more "
            "responsive but jumpier.")))

        lh, self._lead, self._lead_val = _slider(
            0, 50, "How far ahead the camera anticipates motion.")
        self._lead.valueChanged.connect(
            lambda v: (self._lead_val.setText(f"{v * 10} ms"), self._schedule()))
        af.addRow("Prediction", _with_chip(lh, HelpBadge(
            "Lead time: project the subject's motion forward so the camera leads "
            "rather than trails. 0 = follow only.")))

        self._safe_zone = QCheckBox("Framing box (hold still while centered)")
        self._safe_zone.toggled.connect(self._schedule)
        af.addRow("", _with_chip(self._safe_zone, HelpBadge(
            "Draw an adjustable centre box on the tile; the PTZ stays still while "
            "the subject is inside it and only moves to keep them within it. Drag "
            "inside the box to move it, or drag its handles to resize it.")))

        xh, self._safe_x, self._safe_x_val = _slider(
            -90, 90, "Horizontal centre of the framing box; 0 is frame centre.")
        self._safe_x.valueChanged.connect(
            lambda v: (self._safe_x_val.setText(_signed_pct(v)), self._schedule_framing()))
        af.addRow("Box X", _with_chip(xh, HelpBadge(
            "Move the framing box left or right. This changes the target settle "
            "point, not just the drawing: tracking holds still around this offset.")))

        yh, self._safe_y, self._safe_y_val = _slider(
            -90, 90, "Vertical centre of the framing box; positive is higher.")
        self._safe_y.valueChanged.connect(
            lambda v: (self._safe_y_val.setText(_signed_pct(v)), self._schedule_framing()))
        af.addRow("Box Y", _with_chip(yh, HelpBadge(
            "Move the framing box up or down. Positive values place the hold-still "
            "region higher in the frame.")))

        center = QPushButton("Center")
        center.setToolTip("Reset Box X and Box Y to the exact frame center.")
        center.clicked.connect(self._center_framing_box)
        af.addRow("", _with_chip(center, HelpBadge(
            "Sets the framing box centre to exact 0 / 0. Tile dragging also snaps "
            "each axis to exact center when it is within 4%."
        )))

        wh, self._safe_w, self._safe_w_val = _slider(
            3, 90, "Half-width of the framing box as a fraction of the frame.")
        self._safe_w.valueChanged.connect(
            lambda v: (self._safe_w_val.setText(f"{v}%"), self._schedule_framing()))
        af.addRow("Box width", _with_chip(wh, HelpBadge(
            "How wide the framing box is, as a fraction of the frame's half-width. "
            "Also adjustable by dragging the box on the tile.")))

        hh, self._safe_h, self._safe_h_val = _slider(
            3, 90, "Half-height of the framing box as a fraction of the frame.")
        self._safe_h.valueChanged.connect(
            lambda v: (self._safe_h_val.setText(f"{v}%"), self._schedule_framing()))
        af.addRow("Box height", _with_chip(hh, HelpBadge(
            "How tall the framing box is, as a fraction of the frame's half-height. "
            "Also adjustable by dragging the box on the tile.")))

        rh, self._safe_round, self._safe_round_val = _slider(
            0, 100, "Corner roundness of the framing region (0 = rectangle, 100 = oval).")
        self._safe_round.valueChanged.connect(
            lambda v: (self._safe_round_val.setText(f"{v}%"), self._schedule_framing()))
        af.addRow("Roundness", _with_chip(rh, HelpBadge(
            "Shape of the framing region: 0% is a sharp rectangle, 100% a full "
            "oval. Tune it to whatever frames your subject best.")))

        adv.add_widget(_wrap(af))
        return adv

    def _build_ptz_controls(self) -> QWidget:
        """Manual PTZ: a draggable joystick (pan/tilt) beside the button pad."""
        holder = QWidget()
        outer = QVBoxLayout(holder)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        head = QHBoxLayout(); head.setContentsMargins(0, 0, 0, 0); head.setSpacing(6)
        cap = QLabel("Manual control")
        self._muted_captions.append(cap)
        head.addWidget(cap)
        head.addWidget(HelpBadge(
            "Drive the camera by hand: drag the joystick or hold a D-pad arrow "
            "(including the diagonals) to pan/tilt, hold ＋/－ to zoom — release to "
            "stop. ⌂ recalls Home, ☰ opens the camera's menu. The same controls "
            "are on each camera tile (hover the top-right corner). Tracking pauses "
            "while you nudge."
        ))
        head.addStretch(1)
        outer.addLayout(head)
        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(14)
        self._joystick = JoystickPad(118)
        self._joystick.moved.connect(lambda pan, tilt: self._nudge(pan, tilt, 0.0))
        row.addWidget(self._joystick)
        row.addWidget(self._build_ptz_pad())
        row.addStretch(1)
        outer.addLayout(row)
        return holder

    def _build_ptz_pad(self) -> QWidget:
        """A clean 3×3 D-pad (8 directions, empty centre) + zoom and Home/Menu.

        Press-and-hold drives ``ptzNudge``; release stops (0,0,0).  There is no
        centre stop button by design — releasing any direction stops motion.
        The 4 diagonals drive pan and tilt together for smooth corner moves.
        """
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setContentsMargins(0, 4, 0, 0)
        col.setSpacing(8)

        # Row of [3×3 D-pad] [zoom +/- column].
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)

        self._pad_btns: list[QPushButton] = []
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        # (label, row, col, (pan, tilt, zoom)) — row 0 top; centre (1,1) is empty.
        dirs = [
            ("↖", 0, 0, (-1, 1, 0)), ("↑", 0, 1, (0, 1, 0)), ("↗", 0, 2, (1, 1, 0)),
            ("←", 1, 0, (-1, 0, 0)),                          ("→", 1, 2, (1, 0, 0)),
            ("↙", 2, 0, (-1, -1, 0)), ("↓", 2, 1, (0, -1, 0)), ("↘", 2, 2, (1, -1, 0)),
        ]
        for text, r, c, vec in dirs:
            b = QPushButton(text)
            b.setObjectName("ptzPadBtn")
            b.setFixedSize(T.fs(38), T.fs(34))
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.pressed.connect(lambda v=vec: self._nudge(*v))
            b.released.connect(lambda: self._nudge(0, 0, 0))
            grid.addWidget(b, r, c)
            self._pad_btns.append(b)
        body.addLayout(grid)

        # Prettier zoom controls: a labelled +/- stack.
        zoom_col = QVBoxLayout()
        zoom_col.setContentsMargins(0, 0, 0, 0)
        zoom_col.setSpacing(4)
        self._zoom_in = QPushButton("＋")
        self._zoom_out = QPushButton("－")
        zoom_cap = QLabel("Zoom")
        zoom_cap.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        zoom_cap.setObjectName("ptzZoomCap")
        for b, vec, tip in (
            (self._zoom_in, (0, 0, 1), "Zoom in (hold)"),
            (self._zoom_out, (0, 0, -1), "Zoom out (hold)"),
        ):
            b.setObjectName("ptzZoomBtn")
            b.setFixedSize(T.fs(38), T.fs(34))
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setToolTip(tip)
            b.pressed.connect(lambda v=vec: self._nudge(*v))
            b.released.connect(lambda: self._nudge(0, 0, 0))
        zoom_col.addWidget(self._zoom_in)
        zoom_col.addWidget(zoom_cap)
        zoom_col.addWidget(self._zoom_out)
        zoom_col.addStretch(1)
        body.addLayout(zoom_col)
        body.addStretch(1)
        col.addLayout(body)

        # Home / Menu actions row.
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self._home_btn = QPushButton("⌂  Home")
        self._home_btn.setObjectName("ptzActionBtn")
        self._home_btn.setMinimumHeight(T.fs(28))
        self._home_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._home_btn.setToolTip("Recall the camera's home position.")
        self._home_btn.clicked.connect(self._ptz_home)
        self._menu_btn = QPushButton("☰  Menu")
        self._menu_btn.setObjectName("ptzActionBtn")
        self._menu_btn.setMinimumHeight(T.fs(28))
        self._menu_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._menu_btn.setToolTip("Open the camera's on-screen menu (if supported).")
        self._menu_btn.clicked.connect(self._ptz_menu)
        actions.addWidget(self._home_btn, 1)
        actions.addWidget(self._menu_btn, 1)
        col.addLayout(actions)

        self._restyle_ptz_pad()
        return holder

    def _restyle_ptz_pad(self) -> None:
        """Theme-correct styling for the D-pad, zoom, and Home/Menu buttons."""
        pal = T.CURRENT
        pad_css = (
            f"QPushButton#ptzPadBtn, QPushButton#ptzZoomBtn {{"
            f" background: {pal.surface_hov}; color: {pal.text}; border: none;"
            f" border-radius: {T.RADIUS}px; font-size: {T.fs(15)}px; font-weight: 700;"
            f" padding-bottom: 2px; }}"
            f"QPushButton#ptzPadBtn:hover, QPushButton#ptzZoomBtn:hover {{"
            f" background: {T.ACCENT.name()}; color: #ffffff; }}"
            f"QPushButton#ptzPadBtn:pressed, QPushButton#ptzZoomBtn:pressed {{"
            f" background: {T.ACCENT.lighter(120).name()}; color: #ffffff; }}"
            f"QLabel#ptzZoomCap {{ color: {pal.muted}; font-size: {T.fs(10)}px; }}"
            f"QPushButton#ptzActionBtn {{ background: {pal.surface};"
            f" color: {pal.text}; border: 1px solid {pal.border};"
            f" border-radius: {T.RADIUS}px; font-weight: 600; padding: 5px 10px;"
            f" font-size: {T.fs(12)}px; }}"
            f"QPushButton#ptzActionBtn:hover {{ border-color: {T.ACCENT.name()};"
            f" color: {T.ACCENT.name()}; }}"
        )
        for b in getattr(self, "_pad_btns", []):
            b.setStyleSheet(pad_css)
        for name in ("_zoom_in", "_zoom_out", "_home_btn", "_menu_btn"):
            w = getattr(self, name, None)
            if w is not None:
                w.setStyleSheet(pad_css)

    # ── presets ──────────────────────────────────────────────────────────────────

    def _build_presets(self) -> QWidget:
        """A grid of up to six preset tiles (snapshot thumbnail + label).

        Click an occupied tile to recall it; click the ⋯ corner (or an empty
        tile) to save / overwrite / clear.  The thumbnail is grabbed from the
        live frame source when available (see :meth:`_capture_thumbnail`).
        """
        holder = QWidget()
        outer = QVBoxLayout(holder)
        outer.setContentsMargins(0, 8, 0, 0)
        outer.setSpacing(4)
        head = QHBoxLayout(); head.setContentsMargins(0, 0, 0, 0); head.setSpacing(6)
        cap = QLabel("Presets")
        self._muted_captions.append(cap)
        head.addWidget(cap)
        head.addWidget(HelpBadge(
            "Saved camera views. Click an occupied preset to recall it. Use a "
            "tile's ⋯ menu (or click an empty tile) to save the current view, "
            "overwrite, or clear it. Saving captures a snapshot thumbnail."
        ))
        head.addStretch(1)
        outer.addLayout(head)

        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 0)
        grid.setSpacing(6)
        self._preset_tiles: list[_PresetTile] = []
        for i in range(_PRESET_COUNT):
            tile = _PresetTile(i)
            tile.recallRequested.connect(self._on_preset_recall)
            tile.saveRequested.connect(self._on_preset_save)
            tile.clearRequested.connect(self._on_preset_clear)
            grid.addWidget(tile, i // 3, i % 3)
            self._preset_tiles.append(tile)
        outer.addLayout(grid)
        return holder

    def _preset_slots(self) -> dict[int, dict[str, Any]]:
        """Read the ``ptz.preset_slots`` map from the loaded config (slot → meta)."""
        pz = (self._cfg.get("ptz", {}) or {}) if self._cfg else {}
        raw = pz.get("preset_slots", {}) or {}
        out: dict[int, dict[str, Any]] = {}
        for k, v in raw.items():
            try:
                slot = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict):
                out[slot] = v
            elif isinstance(v, str):  # legacy plain-string label
                out[slot] = {"label": v, "thumbnail": None}
        return out

    def _refresh_presets(self) -> None:
        """Render each tile from the current ``preset_slots`` map."""
        slots = self._preset_slots()
        for tile in getattr(self, "_preset_tiles", []):
            meta = slots.get(tile.slot)
            tile.set_state(
                occupied=meta is not None,
                label=(meta or {}).get("label", "") or "",
                thumbnail=(meta or {}).get("thumbnail"),
            )

    def _on_preset_recall(self, slot: int) -> None:
        if not self._camera_id:
            return
        try:
            self._client.recallPtzPreset(self._camera_id, int(slot))
        except Exception:  # noqa: BLE001
            log.debug("recallPtzPreset failed", exc_info=True)

    def _on_preset_save(self, slot: int) -> None:
        """Prompt for a label, grab a thumbnail, and save into *slot*."""
        if not self._camera_id:
            return
        slots = self._preset_slots()
        existing = (slots.get(int(slot)) or {}).get("label", "")
        label, ok = QInputDialog.getText(
            self, "Save preset", f"Label for preset {int(slot) + 1}:",
            text=existing or f"Preset {int(slot) + 1}",
        )
        if not ok:
            return
        thumbnail = self._capture_thumbnail()
        try:
            self._client.savePtzPreset(self._camera_id, int(slot), label.strip(), thumbnail or "")
        except Exception:  # noqa: BLE001
            log.debug("savePtzPreset failed", exc_info=True)
        # Reflect immediately (the config write-back is async); configChanged
        # will re-render with the authoritative state shortly after.
        for tile in getattr(self, "_preset_tiles", []):
            if tile.slot == int(slot):
                tile.set_state(occupied=True, label=label.strip(), thumbnail=thumbnail)

    def _on_preset_clear(self, slot: int) -> None:
        """Clear a slot by saving an empty preset (label="", no thumbnail)."""
        if not self._camera_id:
            return
        if QMessageBox.question(
            self, "Clear preset", f"Clear preset {int(slot) + 1}?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self._client.savePtzPreset(self._camera_id, int(slot), "", "")
        except Exception:  # noqa: BLE001
            log.debug("clear preset (savePtzPreset) failed", exc_info=True)
        for tile in getattr(self, "_preset_tiles", []):
            if tile.slot == int(slot):
                tile.set_state(occupied=False, label="", thumbnail=None)

    def _capture_thumbnail(self) -> str:
        """Grab the current camera frame as a JPEG ``data:`` URI, or "".

        Uses the same live-frame handle the camera tiles use
        (``frame_source.latest_qimage(camera_id)``).  Returns an empty string
        when no frame source is wired or no frame is available yet — the preset
        still saves, just without a snapshot.
        """
        frames = getattr(self, "_frames", None)
        if frames is None or not self._camera_id:
            # TODO: no frame handle wired — pass ``frame_source=`` to
            # PropertiesPanel(...) to enable snapshot thumbnails.
            return ""
        try:
            img = frames.latest_qimage(self._camera_id)
        except Exception:  # noqa: BLE001
            log.debug("latest_qimage failed", exc_info=True)
            return ""
        if img is None or img.isNull():
            return ""
        try:
            scaled = img.scaled(
                QSize(160, 160),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            scaled.save(buf, "JPEG", 70)
            buf.close()
            b64 = base64.b64encode(bytes(ba)).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:  # noqa: BLE001
            log.debug("thumbnail encode failed", exc_info=True)
            return ""

    def _on_config_changed(self, camera_id: str) -> None:
        if camera_id == self._camera_id:
            # Re-pull just the preset state (cheap; avoids clobbering edits in flight).
            self._cfg = _safe(lambda: self._client.getCameraConfig(self._camera_id), self._cfg) or self._cfg
            self._refresh_presets()
            # Mirror framing changes made on the tile (drag-resize) into the sliders.
            self._sync_framing_sliders()

    def _build_fps_row(self) -> QWidget:
        """A frame-rate slider with a live value + measured-fps readout.

        The slider's max tracks the source's *detected* hardware fps ceiling
        (``client.source_fps_cap``) when known, and moving it applies the new
        rate live via ``client.setTargetFps`` on release/keyboard changes.
        """
        holder = QWidget()
        col = QVBoxLayout(holder)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        top = QHBoxLayout(); top.setContentsMargins(0, 0, 0, 0); top.setSpacing(8)
        self._fps = QSlider(Qt.Orientation.Horizontal)
        self._fps.setRange(1, 60)
        self._fps.setTracking(False)
        self._fps.setToolTip(
            "Frames per second to request from the camera. Higher is smoother for "
            "tracking but uses more CPU/GPU; the cap reflects the source's "
            "detected hardware maximum. Changes apply live."
        )
        self._fps.sliderMoved.connect(self._set_fps_value_label)
        self._fps.valueChanged.connect(self._on_fps_changed)
        self._fps_value = QLabel("30 fps")
        self._fps_value.setMinimumWidth(48)
        self._fps_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self._fps, 1)
        top.addWidget(self._fps_value)
        col.addLayout(top)
        self._fps_measured = QLabel("")
        self._fps_measured.setObjectName("fpsMeasured")
        self._restyle_fps_measured()
        col.addWidget(self._fps_measured)
        return holder

    def _restyle_fps_measured(self) -> None:
        """Theme-aware styling for the measured-fps caption (re-runs on switch)."""
        self._fps_measured.setStyleSheet(
            f"color: {T.CURRENT.muted}; font-size: {T.fs(11)}px;"
        )

    def _fps_cap(self) -> int:
        """The slider max: trusted hardware cap, else current requested rate.

        When AVFoundation/OpenCV reports a real source ceiling we trust it. When
        the cap is still unknown, do not invent a USB 60 fps range for a 30 fps
        camera; expose the configured/measured neighborhood instead.
        """
        cap = 0.0
        if self._camera_id:
            cap = _safe(lambda: float(self._client.sourceFpsCap(self._camera_id)), 0.0)
        if cap and cap >= 1.0:
            return int(round(cap))
        src = (self._cfg.get("source", {}) or {}) if self._cfg else {}
        configured = int(round(float(src.get("fps", 30) or 30)))
        measured = 0
        if self._camera_id:
            measured = int(round(_safe(
                lambda: float(self._client.cameraModel.get_record(self._camera_id).fps), 0.0,
            ) or 0.0))
        return max(1, min(120, max(30, configured, measured)))

    def _has_trusted_fps_cap(self) -> bool:
        if not self._camera_id:
            return False
        cap = _safe(lambda: float(self._client.sourceFpsCap(self._camera_id)), 0.0)
        return bool(cap and cap >= 1.0)

    def _refresh_substream_visibility(self, src: dict[str, Any]) -> None:
        """Show the substream toggle only when this source has a usable alternate stream."""
        supported = _source_supports_substream(src)
        self._substream_holder.setVisible(supported)
        self._substream.setEnabled(supported)
        if not supported:
            self._substream.blockSignals(True)
            self._substream.setChecked(False)
            self._substream.blockSignals(False)

    # ── public ─────────────────────────────────────────────────────────────────

    def set_camera(self, camera_id: str) -> None:
        self._camera_id = camera_id or ""
        self._empty.setVisible(not self._camera_id)
        self._scroll.setVisible(bool(self._camera_id))
        if self._camera_id:
            self._load()

    # ── load / push ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        cfg = _safe(lambda: self._client.getCameraConfig(self._camera_id), {}) or {}
        self._cfg = cfg
        self._loading = True
        try:
            src = cfg.get("source", {}) or {}
            tr = cfg.get("tracking", {}) or {}
            pz = cfg.get("ptz", {}) or {}
            self._name.setText(cfg.get("name", ""))
            self._source_type.setText(src.get("type", "—"))
            self._address.setText(_short(src.get("address", "")))
            self._refresh_substream_visibility(src)
            self._substream.setChecked(bool(src.get("substream", False)))
            cap = self._fps_cap()
            self._fps.blockSignals(True)
            self._fps.setRange(1, cap)
            self._fps.setValue(min(int(src.get("fps", 30) or 30), cap))
            self._fps.blockSignals(False)
            self._set_fps_value_label(self._fps.value())
            self._update_measured_fps()
            _set_combo(self._quality, tr.get("quality_floor", "auto"))
            self._detect_interval.setValue(int(tr.get("detect_interval", 1) or 1))
            _set_combo_data(self._tracking_mode, tr.get("tracking_mode", "stable"))
            _set_combo(self._tracker, tr.get("tracker", "botsort"))
            # Unified Framing: prefer ``tracking.framing``; fall back to the legacy
            # ``aim_region`` so an un-migrated config still selects sensibly.
            _set_combo_data(
                self._framing, tr.get("framing") or tr.get("aim_region") or "upper_body",
            )
            self._ignore_arms.setChecked((tr.get("aim_body_mode") or "torso") == "torso")
            self._face.setChecked(bool(tr.get("face_confirm", False)))
            _set_combo(self._backend, pz.get("backend", "auto"))
            self._ptz_address.setText(pz.get("address") or "")
            self._auto_zoom.setChecked(bool(pz.get("auto_zoom", True)))
            # Advanced tracking tuning (sliders store hundredths of the cfg value).
            self._kp.setValue(int(round(float(pz.get("kp", 0.6)) * 100)))
            self._kp_val.setText(f"{self._kp.value() / 100:.2f}")
            self._smoothing.setValue(int(round(float(pz.get("aim_smoothing", 0.5)) * 100)))
            self._smoothing_val.setText(f"{self._smoothing.value()}%")
            self._lead.setValue(int(round(float(pz.get("lead_time_s", 0.15)) * 100)))
            self._lead_val.setText(f"{self._lead.value() * 10} ms")
            self._safe_zone.setChecked(bool(pz.get("safe_zone_enabled", True)))
            self._safe_x.setValue(int(round(float(pz.get("safe_zone_x", 0.0)) * 100)))
            self._safe_x_val.setText(_signed_pct(self._safe_x.value()))
            self._safe_y.setValue(int(round(float(pz.get("safe_zone_y", 0.0)) * 100)))
            self._safe_y_val.setText(_signed_pct(self._safe_y.value()))
            self._safe_w.setValue(int(round(float(pz.get("safe_zone_w", 0.15)) * 100)))
            self._safe_w_val.setText(f"{self._safe_w.value()}%")
            self._safe_h.setValue(int(round(float(pz.get("safe_zone_h", 0.22)) * 100)))
            self._safe_h_val.setText(f"{self._safe_h.value()}%")
            self._safe_round.setValue(int(round(float(pz.get("safe_zone_roundness", 1.0)) * 100)))
            self._safe_round_val.setText(f"{self._safe_round.value()}%")
            self._refresh_presets()
            # Tracking target + on/off (driven via dedicated client calls).
            enabled = _safe(
                lambda: bool(self._client.cameraModel.get_record(self._camera_id).tracking_enabled),
                False,
            )
            self._track_btn.blockSignals(True)
            self._track_btn.setChecked(enabled)
            self._apply_track_label(enabled)
            self._track_btn.blockSignals(False)
            self._reload_targets()
        finally:
            self._loading = False
        self._refresh_cost_chips()
        self._refresh_reid_gating()

    def _refresh_reid_gating(self) -> None:
        """Grey Mode▸Stable when the global ReID feature is off, and caption why.

        Stable mode depends on the ReID feature (Services). When it's off, Stable
        can't hold via ReID, so we disable the option and note that a still-stored
        Stable selection behaves like Responsive until ReID is turned back on."""
        reid_on = bool((_safe(lambda: self._client.features(), {}) or {}).get("reid", True))
        idx = self._tracking_mode.findData("stable")
        model = self._tracking_mode.model()
        item = model.item(idx) if (idx >= 0 and hasattr(model, "item")) else None
        if item is not None:
            item.setEnabled(reid_on)
            item.setToolTip("" if reid_on else "Enable ReID in Services to use Stable")
        cur = self._tracking_mode.currentData()
        if not reid_on and cur == "stable":
            self._mode_caption.setText(
                "Stable needs the ReID feature (off in Services) — acting as "
                "Responsive until it's turned on."
            )
            self._mode_caption.setVisible(True)
        else:
            self._mode_caption.clear()
            self._mode_caption.setVisible(False)

    def _schedule(self) -> None:
        if self._loading or not self._camera_id:
            return
        self._refresh_cost_chips()
        self._push_timer.start()

    def _schedule_framing(self) -> None:
        """Coalesce framing-slider changes into a fast, *continuous* live push.

        Crucially we do NOT restart the timer on every change — a restarting
        single-shot timer only fires once the user STOPS moving, so the box never
        moved during a slow drag.  Leaving an already-running timer alone makes it
        fire every ~40 ms throughout the drag, so the oval tracks the slider live.
        """
        if self._loading or not self._camera_id:
            return
        if not self._framing_timer.isActive():
            self._framing_timer.start()

    def _push_framing(self) -> None:
        """Push ONLY the framing-box fields so the tile oval tracks the sliders live."""
        if not self._camera_id:
            return
        patch = {"ptz": {
            "safe_zone_x": self._safe_x.value() / 100.0,
            "safe_zone_y": self._safe_y.value() / 100.0,
            "safe_zone_w": self._safe_w.value() / 100.0,
            "safe_zone_h": self._safe_h.value() / 100.0,
            "safe_zone_roundness": self._safe_round.value() / 100.0,
        }}
        # Keep the panel's cached cfg in sync so a later full push doesn't revert it.
        if isinstance(self._cfg, dict):
            self._cfg.setdefault("ptz", {})
            self._cfg["ptz"].update(patch["ptz"])
        try:
            self._client.updateCameraConfigPatch(self._camera_id, patch)
        except Exception:  # noqa: BLE001
            log.debug("framing live push failed", exc_info=True)

    def _center_framing_box(self) -> None:
        """Reset framing-box X/Y to exact center and persist immediately."""
        if self._loading:
            return
        for slider, label in ((self._safe_x, self._safe_x_val), (self._safe_y, self._safe_y_val)):
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            label.setText(_signed_pct(0))
        self._push_framing()

    def _sync_framing_sliders(self) -> None:
        """Mirror the framing box config into the sliders (e.g. after a tile drag)."""
        pz = (self._cfg or {}).get("ptz") or {}
        for slider, val, key, default in (
            (self._safe_x, self._safe_x_val, "safe_zone_x", 0.0),
            (self._safe_y, self._safe_y_val, "safe_zone_y", 0.0),
            (self._safe_w, self._safe_w_val, "safe_zone_w", 0.15),
            (self._safe_h, self._safe_h_val, "safe_zone_h", 0.22),
            (self._safe_round, self._safe_round_val, "safe_zone_roundness", 1.0),
        ):
            v = int(round(float(pz.get(key, default)) * 100))
            if slider.value() != v:
                slider.blockSignals(True)
                slider.setValue(v)
                slider.blockSignals(False)
                val.setText(_signed_pct(v) if key in {"safe_zone_x", "safe_zone_y"} else f"{v}%")

    def _refresh_overlay_boxes(self) -> None:
        """Mirror the global overlay flags into the checkboxes (e.g. View-menu toggle)."""
        cur = _safe(lambda: self._client.overlays(), {}) or {}
        for key, cb in getattr(self, "_overlay_boxes", {}).items():
            on = bool(cur.get(key, key == "detection"))
            if cb.isChecked() != on:
                cb.blockSignals(True)
                cb.setChecked(on)
                cb.blockSignals(False)

    def _refresh_tracker_tip(self) -> None:
        tip = _TRACKER_HELP.get(self._tracker.currentText(), _COST_HELP["tracker"])
        self._tracker.setToolTip(tip)

    def _refresh_cost_chips(self) -> None:
        di = self._detect_interval.value()
        _restyle_chip(self._detect_chip, "heavy" if di <= 2 else "medium" if di <= 5 else "light")
        _restyle_chip(self._tracker_chip,
                      "light" if self._tracker.currentText() == "bytetrack" else "medium")
        self._face_chip.setVisible(self._face.isChecked())

    def _push(self) -> None:
        if not self._camera_id or not self._cfg:
            return
        cfg = dict(self._cfg)
        cfg.setdefault("source", {}); cfg["source"] = dict(cfg["source"])
        cfg.setdefault("tracking", {}); cfg["tracking"] = dict(cfg["tracking"])
        cfg.setdefault("ptz", {}); cfg["ptz"] = dict(cfg["ptz"])
        cfg["name"] = self._name.text().strip() or cfg.get("name", "Camera")
        cfg["source"]["substream"] = (
            self._substream.isChecked() if _source_supports_substream(cfg["source"]) else False
        )
        cfg["source"]["fps"] = float(self._fps.value())
        cfg["tracking"]["quality_floor"] = self._quality.currentText()
        cfg["tracking"]["detect_interval"] = self._detect_interval.value()
        cfg["tracking"]["tracking_mode"] = self._tracking_mode.currentData() or "stable"
        cfg["tracking"]["tracker"] = self._tracker.currentText()
        # The single Framing control drives both aim and zoom: store it as the new
        # unified ``tracking.framing`` and ALSO mirror it into the existing
        # consumers — ``tracking.aim_region`` (worker aim) and ``ptz.zoom_framing``
        # (controller zoom) — so both honor it without a separate worker bridge.
        framing = self._framing.currentData() or "upper_body"
        cfg["tracking"]["framing"] = framing
        cfg["tracking"]["aim_region"] = framing
        cfg["tracking"]["aim_body_mode"] = (
            "torso" if self._ignore_arms.isChecked() else "full_silhouette"
        )
        cfg["tracking"]["face_confirm"] = self._face.isChecked()
        cfg["ptz"]["backend"] = self._backend.currentText()
        cfg["ptz"]["address"] = self._ptz_address.text().strip() or None
        cfg["ptz"]["auto_zoom"] = self._auto_zoom.isChecked()
        cfg["ptz"]["zoom_framing"] = framing
        # Advanced tracking tuning (sliders hold hundredths of the cfg value).
        cfg["ptz"]["kp"] = self._kp.value() / 100.0
        cfg["ptz"]["aim_smoothing"] = self._smoothing.value() / 100.0
        cfg["ptz"]["lead_time_s"] = self._lead.value() / 100.0
        cfg["ptz"]["safe_zone_enabled"] = self._safe_zone.isChecked()
        cfg["ptz"]["safe_zone_x"] = self._safe_x.value() / 100.0
        cfg["ptz"]["safe_zone_y"] = self._safe_y.value() / 100.0
        cfg["ptz"]["safe_zone_w"] = self._safe_w.value() / 100.0
        cfg["ptz"]["safe_zone_h"] = self._safe_h.value() / 100.0
        cfg["ptz"]["safe_zone_roundness"] = self._safe_round.value() / 100.0
        self._cfg = cfg
        try:
            self._client.updateCameraConfig(self._camera_id, json.dumps(cfg))
        except Exception:  # noqa: BLE001
            log.debug("updateCameraConfig failed", exc_info=True)

    # ── tracking target + fps (dedicated client calls, not the cfg push) ─────────

    def _set_fps_value_label(self, value: int) -> None:
        """Show the requested fps, flagging when it's pinned at the hardware cap.

        Dragging past the source's detected ceiling has no effect (the request is
        clamped), which reads as "the slider does nothing" — so say "(max)" rather
        than silently ignore it.
        """
        at_max = self._has_trusted_fps_cap() and value >= self._fps.maximum()
        self._fps_value.setText(f"{value} fps" + ("  (max)" if at_max else ""))

    def _on_fps_changed(self, value: int) -> None:
        self._set_fps_value_label(value)
        if self._loading or not self._camera_id:
            return
        # Apply live for immediate effect (the worker re-paces ingest at once)…
        try:
            self._client.setTargetFps(self._camera_id, float(value))
        except Exception:  # noqa: BLE001
            log.debug("setTargetFps failed", exc_info=True)

    def _on_telemetry(self) -> None:
        """Per-tick UI refresh driven by ``telemetryUpdated``."""
        self._update_measured_fps()
        self._update_effective_detection()
        self._sync_track_state()

    def _update_effective_detection(self) -> None:
        """Echo what the engine is *actually* doing next to the configured values.

        When the user picks ``auto`` quality or the engine relaxes the detector
        cadence under load, show the resolved value (and the reason, on hover) so
        the auto-scaling stays transparent instead of looking like the control
        does nothing."""
        if not self._camera_id:
            return
        rec = _safe(lambda: self._client.cameraModel.get_record(self._camera_id), None)
        tel = getattr(rec, "telemetry", None) if rec is not None else None
        qs = getattr(tel, "quality_state", None) if tel is not None else None

        # Quality floor: when "auto", surface the level it's resolving to + why.
        active = str(getattr(qs, "active", "") or "") if qs is not None else ""
        reason = str(getattr(qs, "reason", "") or "") if qs is not None else ""
        if self._quality.currentText() == "auto" and active and active != "auto":
            self._quality_effective.setText(f"auto → currently {active}")
            self._quality_effective.setToolTip(reason)
            self._quality_effective.setVisible(True)
        else:
            self._quality_effective.clear()
            self._quality_effective.setVisible(False)

        # Detect-every: surface the effective cadence when the engine overrides it.
        base = self._detect_interval.value()
        eff = int(getattr(qs, "detect_interval", base) or base) if qs is not None else base
        if eff and eff != base:
            self._detect_effective.setText(
                f"effective: every {eff} frame{'s' if eff != 1 else ''} "
                "(auto-adjusted under load)"
            )
            self._detect_effective.setToolTip(reason)
            self._detect_effective.setVisible(True)
        else:
            self._detect_effective.clear()
            self._detect_effective.setVisible(False)

    def _sync_track_state(self) -> None:
        """Reflect the camera's *live* tracking on/off in the toggle button.

        Tracking can be flipped from the tile overlay or Stop-All-Tracking, so we
        re-read the authoritative record and update the button without re-emitting
        ``enableTracking`` (which would echo the command back to the engine).
        """
        if not self._camera_id or self._loading:
            return
        enabled = _safe(
            lambda: bool(self._client.cameraModel.get_record(self._camera_id).tracking_enabled),
            None,
        )
        if enabled is None or enabled == self._track_btn.isChecked():
            return
        self._track_btn.blockSignals(True)
        self._track_btn.setChecked(enabled)
        self._apply_track_label(enabled)
        self._track_btn.blockSignals(False)

    def _update_measured_fps(self) -> None:
        if not self._camera_id:
            return
        m = _safe(
            lambda: float(self._client.cameraModel.get_record(self._camera_id).fps), 0.0,
        )
        if m <= 0:
            self._fps_measured.setText("measured: —")
            return
        requested = float(self._fps.value())
        text = f"measured: {m:.1f} fps"
        # When the camera can't reach the requested rate, say so plainly instead
        # of leaving the operator to wonder why "30" runs at 15.
        if requested >= 1.0 and m < requested * 0.8 and (requested - m) >= 3.0:
            text += (
                f" — source tops out near {m:.0f}; lower the rate or this is the "
                "hardware ceiling"
                if self._has_trusted_fps_cap()
                else f" — source isn't reaching {requested:.0f} fps"
            )
        self._fps_measured.setText(text)
        # The hardware cap can arrive after load (it rides telemetry); raise the
        # slider's ceiling to the real maximum once it's known.
        cap = self._fps_cap()
        if cap != self._fps.maximum():
            cur = self._fps.value()
            new_value = min(cur, cap)
            self._fps.blockSignals(True)
            self._fps.setRange(1, cap)
            self._fps.setValue(new_value)
            self._fps.blockSignals(False)
            self._set_fps_value_label(self._fps.value())
            if new_value != cur and not self._loading and self._camera_id:
                self._on_fps_changed(new_value)

    def _on_external_target_changed(self, camera_id: str) -> None:
        """Sync the 'Track person' dropdown when the target changes elsewhere."""
        if not self._camera_id or camera_id != self._camera_id:
            return
        cfg = _safe(lambda: self._client.getCameraConfig(camera_id), None)
        if cfg and isinstance(self._cfg, dict):
            self._cfg["target"] = cfg.get("target", self._cfg.get("target", {}))
        self._reload_targets()

    def _reload_targets(self) -> None:
        """Repopulate the person dropdown from registered identities."""
        people = _safe(lambda: self._client.registeredIdentities(), []) or []
        cur = (self._cfg.get("target") or {}).get("identity_id") if self._cfg else None
        self._target_combo.blockSignals(True)
        try:
            self._target_combo.clear()
            self._target_combo.addItem("— Anyone —", "")
            for p in people:
                self._target_combo.addItem(p.get("name") or "(unnamed)", p.get("id") or "")
            idx = self._target_combo.findData(cur) if cur else 0
            self._target_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._target_combo.blockSignals(False)

    def _on_track_toggled(self, on: bool) -> None:
        self._apply_track_label(on)
        if self._loading or not self._camera_id:
            return
        try:
            self._client.enableTracking(self._camera_id, bool(on))
        except Exception:  # noqa: BLE001
            log.debug("enableTracking failed", exc_info=True)

    def _apply_track_label(self, on: bool) -> None:
        """Make the on-state unmistakable: a red ■ Stop Tracking button."""
        self._track_btn.setText("■  Stop Tracking" if on else "Track")
        self._restyle_track_btn(on)

    def _restyle_track_btn(self, on: bool) -> None:
        """Accent (off) / red (on) full-width toggle.  Re-runs on theme change.

        Padding + the button's min-height guarantee the descender in "Stop"
        clears the bottom edge (the reported clipping bug).
        """
        color = T.ERROR if on else T.ACCENT.name()
        self._track_btn.setStyleSheet(
            f"QPushButton#trackToggleBtn {{ background: {color}; color: #ffffff;"
            f" border: none; border-radius: {T.RADIUS}px; font-weight: 700;"
            f" padding: 7px 14px; font-size: {T.fs(13)}px; }}"
            f"QPushButton#trackToggleBtn:hover {{ background: {QColor(color).lighter(112).name()}; }}"
        )

    def _on_target_changed(self, _index: int) -> None:
        if self._loading or not self._camera_id:
            return
        ident = self._target_combo.currentData() or ""
        try:
            self._client.setTargetIdentity(self._camera_id, ident)
        except Exception:  # noqa: BLE001
            log.debug("setTargetIdentity failed", exc_info=True)

    # ── actions ────────────────────────────────────────────────────────────────

    def _nudge(self, pan: float, tilt: float, zoom: float) -> None:
        if not self._camera_id:
            return
        try:
            self._client.ptzNudge(self._camera_id, pan * _NUDGE, tilt * _NUDGE, zoom * _NUDGE)
        except Exception:  # noqa: BLE001
            log.debug("ptzNudge failed", exc_info=True)

    def _ptz_home(self) -> None:
        if not self._camera_id:
            return
        try:
            self._client.ptzHome(self._camera_id)
        except Exception:  # noqa: BLE001
            log.debug("ptzHome failed", exc_info=True)

    def _ptz_menu(self) -> None:
        if not self._camera_id:
            return
        try:
            self._client.ptzMenu(self._camera_id)
        except Exception:  # noqa: BLE001
            log.debug("ptzMenu failed", exc_info=True)

    def _remove(self) -> None:
        if not self._camera_id:
            return
        name = self._name.text() or self._camera_id
        if QMessageBox.question(
            self, "Remove Camera", f"Remove “{name}”?",
        ) == QMessageBox.StandardButton.Yes:
            self._client.removeCamera(self._camera_id)


# ── helpers ─────────────────────────────────────────────────────────────────────


def _form() -> QFormLayout:
    f = QFormLayout()
    f.setContentsMargins(0, 0, 0, 0)
    f.setHorizontalSpacing(14)
    f.setVerticalSpacing(8)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return f


def _wrap(layout: QFormLayout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


def _with_chip(widget: QWidget, *trailing: QWidget) -> QWidget:
    """Pack ``widget`` (stretched) with one or more trailing badges/chips.

    Accepts any number of trailing widgets so a control can carry BOTH a cost
    chip and a "?" HelpBadge in the same row.
    """
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    row.addWidget(widget, 1)
    for w in trailing:
        row.addWidget(w, 0)
    return holder


def _ro_value() -> QLabel:
    lab = QLabel("—")
    lab.setStyleSheet(f"color: {T.CURRENT.text};")
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def _restyle_chip(chip: CostChip, cost: str) -> None:
    color = {"light": T.TRACKING, "medium": T.WARNING, "heavy": T.ERROR}.get(cost, T.TRACKING)
    chip.setText(cost.upper())
    chip.setStyleSheet(
        f"color: {color}; border: 1px solid {color}; border-radius: 7px;"
        f"padding: 1px 6px; font-size: 9px; font-weight: 700;"
    )


def _set_combo(combo: QComboBox, value: str) -> None:
    i = combo.findText(str(value))
    if i >= 0:
        combo.setCurrentIndex(i)


def _set_combo_data(combo: QComboBox, value: str) -> None:
    """Select the item whose userData == *value* (for caption≠value combos)."""
    i = combo.findData(str(value))
    if i >= 0:
        combo.setCurrentIndex(i)


def _short(addr: str) -> str:
    import re
    return re.sub(r"(\w+://)[^@/]*@", r"\1", str(addr or "—")) or "—"


def _signed_pct(value: int) -> str:
    return f"{value:+d}%"


def _source_supports_substream(src: dict[str, Any]) -> bool:
    """True only when config carries a concrete alternate stream reference."""
    if not isinstance(src, dict):
        return False
    for key in ("substream_url", "substream_address", "secondary_address", "lowres_address"):
        if src.get(key):
            return True
    profiles = src.get("profiles")
    if isinstance(profiles, list) and len(profiles) > 1:
        return True
    return False


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _connect(obj: Any, name: str, slot: Any) -> None:
    try:
        getattr(obj, name).connect(slot)
    except Exception:  # noqa: BLE001
        log.debug("connect %s failed", name, exc_info=True)
