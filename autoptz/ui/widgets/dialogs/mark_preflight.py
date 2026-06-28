"""MarkPreflightDialog — the friendly pre-flight notice for AutoPTZ Mark.

A plain, jargon-free setup screen: a one-line intro, the source + profile, and
six dropdowns (Max cameras, Target FPS, Time per step, Resolution, Model, Scene)
with a live run-time estimate.  Its Start button asks a confirm ("This suspends AutoPTZ
and runs the simulation. Continue?") before accepting, since entering Mark
suspends the live app.  On accept it yields a :class:`MarkSession`.

Pure widget logic, offscreen-testable: construct, poke control values, read
:meth:`session`.  The NDI source option is disabled (with a "requires cyndilib"
suffix) when ``ndi_sim_available()`` is False, so the window never tries to spin
up NDI senders the host can't build.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from autoptz.benchmark.ndi_sim import ndi_sim_available
from autoptz.ui import theme as T
from autoptz.ui.mark_session import CLIP_LIBRARY, DEFAULT_CLIP_ID, MarkSession

_INTRO = (
    "AutoPTZ Mark is a quick simulation: it adds fake cameras one at a time and "
    "measures how many your computer can run smoothly. Each camera runs the real "
    "AutoPTZ pipeline on a built-in scene."
)
# Appended to the intro only when the bundled clip is present, so the copy never
# promises "real people from a video clip" in a build that lacks the asset.
_INTRO_CLIP = (
    " By default it plays real people from a built-in video clip; if that clip "
    "isn't installed it falls back to drawn people automatically."
)

# Dropdown options as (label, value).  The value is what lands in MarkSession.
_MAX_CAMERA_OPTS: list[tuple[str, int]] = [
    ("1 camera", 1),
    ("2 cameras", 2),
    ("4 cameras", 4),
    ("8 cameras", 8),
    ("12 cameras", 12),
    ("16 cameras", 16),
]
_STEP_OPTS: list[tuple[str, float]] = [
    ("Recommended (10 s)", 10.0),
    ("5 seconds", 5.0),
    ("15 seconds", 15.0),
    ("20 seconds", 20.0),
]
# Target FPS / Resolution are no longer a fixed global set: they are rebuilt from
# the selected scene's availability table (MarkSession.available_variants()) so the
# combos only ever offer variants the transcode cache can actually produce, and
# label synthetic variants (upscaled pixels / frame-duplicated fps) honestly.
#
# The fps order we surface (and the friendly base label for each).
_FPS_ORDER: list[float] = [24.0, 30.0, 60.0]
_FPS_BASE_LABEL: dict[float, str] = {24.0: "24 fps", 30.0: "30 fps", 60.0: "60 fps"}
# Variant ``res`` tuple (w, h) → the MarkSession resolution key + friendly base label.
_RES_KEY_FROM_SIZE: dict[tuple[int, int], str] = {
    (1280, 720): "720p",
    (1920, 1080): "1080p",
    (3840, 2160): "4k",
}
_RES_BASE_LABEL: dict[str, str] = {
    "720p": "720p (HD)",
    "1080p": "1080p (Full HD)",
    "4k": "4K (Ultra HD)",
}
# The order resolutions are surfaced, smallest → largest.
_RES_ORDER: list[str] = ["720p", "1080p", "4k"]
# Friendly word for each capability tag, for the per-scene capability hint.
_CAPABILITY_WORD: dict[str, str] = {
    "tracking": "tracking",
    "reid": "re-ID",
    "center-stage": "Center Stage",
    "face": "face recognition",
}
_MODEL_OPTS: list[tuple[str, str]] = [
    ("Auto", "auto"),
    ("Nano (fastest)", "nano"),
    ("Small (recommended)", "small"),
    ("Medium (most accurate)", "medium"),
]
# Scene options (label, clip id) built from the bundled clip library.  The clips
# play at their own native fps; Target FPS above stays the pass-floor concept.
_CLIP_OPTS: list[tuple[str, str]] = [(meta.label, meta.id) for meta in CLIP_LIBRARY.values()]

# Run-time estimate fudge factor: spin-up / tear-down / discovery overhead on top
# of the measured ramp.
_ETA_OVERHEAD_S = 10.0


def _add_options(combo: QComboBox, opts: list[tuple[str, Any]], current: Any) -> None:
    """Populate *combo* with (label, value) options and select *current*."""
    for label, value in opts:
        combo.addItem(label, value)
    idx = combo.findData(current)
    combo.setCurrentIndex(idx if idx >= 0 else 0)


def _fps_label(fps: float, fps_tag: str) -> str:
    """Friendly Target-FPS label; interpolated fps is flagged frame-duplicated.

    Native / resampled cadences are real (the master is sub-sampled at worst), so
    they read plainly; only fabricated (frame-duplicated) fps carries the tag.
    """
    base = _FPS_BASE_LABEL.get(fps, f"{int(fps)} fps")
    if fps_tag == "interpolated":
        return f"{base} (frame-duplicated)"
    return base


def _res_label(res_key: str, res_tag: str) -> str:
    """Friendly Resolution label; upscaled resolutions are flagged honestly.

    Native / downscaled sizes show their real pixels, so they read plainly; only
    upscaled (fabricated pixel) resolutions carry the "(upscaled)" tag.
    """
    base = _RES_BASE_LABEL.get(res_key, res_key)
    if res_tag == "upscaled":
        return f"{base} (upscaled)"
    return base


class MarkPreflightDialog(QDialog):
    """Friendly pre-flight notice + parameter picker for an AutoPTZ Mark run."""

    def __init__(
        self,
        *,
        defaults: MarkSession | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        d = defaults or MarkSession()
        self.setWindowTitle("Run AutoPTZ Mark")
        self.setModal(True)
        self.setMinimumWidth(460)

        col = QVBoxLayout(self)
        col.setContentsMargins(18, 18, 18, 18)
        col.setSpacing(12)

        # Honest about the source: only promise "real people from a clip" when the
        # bundled clip is actually present; otherwise say it falls back to drawn people.
        clip_ok = MarkSession().clip_available()
        intro = QLabel(_INTRO + (_INTRO_CLIP if clip_ok else ""))
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {T.CURRENT.subtext};")
        col.addWidget(intro)

        # ── profile ────────────────────────────────────────────────────────────
        profile_box = QGroupBox("What to run")
        profile_col = QVBoxLayout(profile_box)
        self._profile_group = QButtonGroup(self)
        self._full_radio = QRadioButton("Full — find, track, and frame people")
        self._streams_radio = QRadioButton("Streams — show video only (no tracking)")
        self._profile_group.addButton(self._full_radio)
        self._profile_group.addButton(self._streams_radio)
        profile_col.addWidget(self._full_radio)
        profile_col.addWidget(self._streams_radio)
        (self._streams_radio if d.profile == "streams" else self._full_radio).setChecked(True)
        col.addWidget(profile_box)

        # ── source ─────────────────────────────────────────────────────────────
        # Only two sources: the bundled clip (real decode) and real NDI senders.
        # The old "Synthetic — drawn people" option is removed — the drawn scene now
        # survives only as the env-gated ground-truth scene, never a user source.
        source_box = QGroupBox("Camera source")
        source_col = QVBoxLayout(source_box)
        self._source_group = QButtonGroup(self)
        clip_label = (
            "Bundled clip — real people (real decode)"
            if clip_ok
            else "Bundled clip — not installed (uses drawn people)"
        )
        self._clip_radio = QRadioButton(clip_label)
        ndi_ok = ndi_sim_available()
        ndi_text = "Real NDI sources" if ndi_ok else "Real NDI sources  (requires cyndilib)"
        self._ndi_radio = QRadioButton(ndi_text)
        self._ndi_radio.setEnabled(ndi_ok)
        self._source_group.addButton(self._clip_radio)
        self._source_group.addButton(self._ndi_radio)
        source_col.addWidget(self._clip_radio)
        source_col.addWidget(self._ndi_radio)
        if d.source == "ndi" and ndi_ok:
            self._ndi_radio.setChecked(True)
        else:
            # Default (and any clip / unknown / NDI-without-cyndilib) → bundled clip.
            self._clip_radio.setChecked(True)
        col.addWidget(source_box)

        # ── parameters (all dropdowns) ──────────────────────────────────────────
        params = QFormLayout()
        params.setHorizontalSpacing(14)
        params.setVerticalSpacing(8)

        self._max_combo = QComboBox()
        _add_options(self._max_combo, _MAX_CAMERA_OPTS, int(d.max_cameras))
        params.addRow("Max cameras", self._max_combo)

        # Target FPS / Resolution start empty; _on_scene_changed() (called at the
        # end of __init__) fills them from the selected scene's availability table.
        # The session defaults are the *desired* initial selection, preserved when
        # the chosen scene actually offers them.
        self._fps_combo = QComboBox()
        params.addRow("Target FPS", self._fps_combo)

        self._step_combo = QComboBox()
        _add_options(self._step_combo, _STEP_OPTS, float(d.dwell_s))
        params.addRow("Time per step", self._step_combo)
        step_hint = QLabel("How long each level is measured.")
        step_hint.setStyleSheet(f"color: {T.CURRENT.subtext}; font-size: {T.fs(11)}px;")
        params.addRow("", step_hint)

        self._res_combo = QComboBox()
        params.addRow("Resolution", self._res_combo)

        self._model_combo = QComboBox()
        _add_options(self._model_combo, _MODEL_OPTS, str(d.model).strip().lower())
        params.addRow("Model", self._model_combo)

        # Scene: which bundled clip to play.  An empty default → the library default.
        self._clip_combo = QComboBox()
        clip_default = str(d.clip_id).strip() or DEFAULT_CLIP_ID
        _add_options(self._clip_combo, _CLIP_OPTS, clip_default)
        params.addRow("Scene", self._clip_combo)

        # The capability hint sits directly under the Scene row: it names the AI
        # features the chosen scene meaningfully exercises (tracking / re-ID / …).
        self._capability_hint = QLabel("")
        self._capability_hint.setWordWrap(True)
        self._capability_hint.setStyleSheet(f"color: {T.CURRENT.subtext}; font-size: {T.fs(11)}px;")
        params.addRow("", self._capability_hint)
        col.addLayout(params)

        # The desired initial fps/res from the defaults, preserved across scene
        # changes when still valid (else the scene's native/sensible default wins).
        self._desired_fps = float(d.floor_fps)
        self._desired_res = str(d.resolution).strip().lower()
        # Set while we rebuild the fps/res combos so their currentIndexChanged
        # signals don't re-enter the repopulation logic mid-rebuild.
        self._rebuilding = False

        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet(f"color: {T.CURRENT.subtext};")
        col.addWidget(self._eta_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Start")
        # Start confirms before accepting (entering Mark suspends the live app), so
        # the box's ``accepted`` (emitted on Ok) routes through our confirm slot
        # instead of the default ``accept`` — only a Yes proceeds.
        buttons.accepted.connect(self._on_start)
        buttons.rejected.connect(self.reject)
        col.addWidget(buttons)

        # The fps/res combos follow the chosen scene; the res combo additionally
        # re-filters when the fps changes (only resolutions valid for (scene, fps)).
        self._clip_combo.currentIndexChanged.connect(self._on_scene_changed)
        self._fps_combo.currentIndexChanged.connect(self._on_fps_changed)

        # Live-update the ETA whenever a duration-affecting control changes.
        self._max_combo.currentIndexChanged.connect(self._refresh_eta)
        self._step_combo.currentIndexChanged.connect(self._refresh_eta)

        # Sync the scene-coupled combos + capability hint to the initial scene.
        self._on_scene_changed()
        self._refresh_eta()

    # ── scene-coupled availability ──────────────────────────────────────────
    def _selected_clip_id(self) -> str:
        cid = self._clip_combo.currentData()
        return str(cid) if cid else DEFAULT_CLIP_ID

    def _variants(self) -> list[dict]:
        """The selected scene's (res, fps) availability table."""
        return MarkSession(clip_id=self._selected_clip_id()).available_variants()

    def _on_scene_changed(self, *_args: Any) -> None:
        """Rebuild the fps + resolution combos and capability hint for the scene.

        Repopulates Target FPS with the distinct fps the scene offers (in
        {24,30,60} order), preserving the current fps when it's still on offer
        else falling back to the scene's native/sensible default; then rebuilds
        the resolution combo for the resulting fps.  Synthetic variants are
        labelled honestly (upscaled / frame-duplicated) while their ``itemData``
        stays the clean MarkSession key/value.
        """
        variants = self._variants()
        self._repopulate_fps(variants)
        self._repopulate_res(variants)
        self._update_capability_hint()

    def _on_fps_changed(self, *_args: Any) -> None:
        """Re-filter the resolution combo for the newly-selected (scene, fps)."""
        if self._rebuilding:
            return
        self._repopulate_res(self._variants())

    def _repopulate_fps(self, variants: list[dict]) -> None:
        """Fill Target FPS with the scene's distinct fps, preserving selection."""
        # Distinct fps offered by the scene, surfaced in the canonical order.
        offered: dict[float, str] = {}
        for v in variants:
            offered[float(v["fps"])] = str(v["fps_tag"])
        ordered = [f for f in _FPS_ORDER if f in offered] + sorted(
            f for f in offered if f not in _FPS_ORDER
        )

        prev = self._fps_combo.currentData()
        want = prev if prev in offered else self._desired_fps
        if want not in offered:
            want = self._native_or_default_fps(offered)

        self._rebuilding = True
        try:
            self._fps_combo.clear()
            for fps in ordered:
                self._fps_combo.addItem(_fps_label(fps, offered[fps]), fps)
            idx = self._fps_combo.findData(want)
            self._fps_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._rebuilding = False

    def _repopulate_res(self, variants: list[dict]) -> None:
        """Fill Resolution with the resolutions valid for the current (scene, fps)."""
        fps = self._fps_combo.currentData()
        # Resolutions valid for the selected fps (combos that don't apply vanish).
        offered: dict[str, str] = {}
        for v in variants:
            if fps is not None and float(v["fps"]) != float(fps):
                continue
            key = _RES_KEY_FROM_SIZE.get(tuple(v["res"]))
            if key is not None:
                offered[key] = str(v["res_tag"])
        ordered = [r for r in _RES_ORDER if r in offered]

        prev = self._res_combo.currentData()
        want = prev if prev in offered else self._desired_res
        if want not in offered:
            want = self._native_or_default_res(offered)

        self._rebuilding = True
        try:
            self._res_combo.clear()
            for key in ordered:
                self._res_combo.addItem(_res_label(key, offered[key]), key)
            idx = self._res_combo.findData(want)
            self._res_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._rebuilding = False

    @staticmethod
    def _native_or_default_fps(offered: dict[float, str]) -> float:
        """The scene's native fps if offered, else the first offered (canonical)."""
        for fps, tag in offered.items():
            if tag == "native":
                return fps
        for fps in _FPS_ORDER:
            if fps in offered:
                return fps
        return next(iter(offered))

    @staticmethod
    def _native_or_default_res(offered: dict[str, str]) -> str:
        """The scene's native resolution if offered, else the largest real one."""
        for key in _RES_ORDER:
            if offered.get(key) == "native":
                return key
        # Prefer a real (downscaled, non-synthetic) resolution over an upscaled one.
        for key in reversed(_RES_ORDER):
            if offered.get(key) == "downscaled":
                return key
        for key in _RES_ORDER:
            if key in offered:
                return key
        return next(iter(offered))

    def _update_capability_hint(self) -> None:
        """Refresh the per-scene capability hint from the scene's capability tags."""
        tags = MarkSession(clip_id=self._selected_clip_id()).capability_tags()
        words = [_CAPABILITY_WORD.get(t, t) for t in tags]
        if words:
            self._capability_hint.setText("Tests: " + " + ".join(words))
        else:
            self._capability_hint.setText("")

    def _on_start(self) -> None:
        """Confirm the run suspends AutoPTZ, then accept."""
        choice = QMessageBox.question(
            self,
            "Run AutoPTZ Mark?",
            "This suspends AutoPTZ and runs the simulation. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if choice == QMessageBox.StandardButton.Yes:
            self.accept()

    def session(self) -> MarkSession:
        """Read the current control values into a :class:`MarkSession`."""
        if self._ndi_radio.isEnabled() and self._ndi_radio.isChecked():
            source = "ndi"
        else:
            source = "clip"
        profile = "streams" if self._streams_radio.isChecked() else "full"
        return MarkSession(
            profile=profile,
            source=source,
            floor_fps=float(self._fps_combo.currentData()),
            max_cameras=int(self._max_combo.currentData()),
            dwell_s=float(self._step_combo.currentData()),
            resolution=str(self._res_combo.currentData()),
            model=str(self._model_combo.currentData()),
            clip_id=str(self._clip_combo.currentData()),
        )

    @staticmethod
    def estimated_seconds(session: MarkSession) -> float:
        """Worst-case run time: every step is measured before the limit is hit."""
        return session.max_cameras * session.dwell_s + _ETA_OVERHEAD_S

    def _refresh_eta(self, *_args: Any) -> None:
        secs = self.estimated_seconds(self.session())
        mins, rem = divmod(int(round(secs)), 60)
        human = f"{mins} min {rem} s" if mins else f"{rem} s"
        self._eta_label.setText(
            f"Estimated time: up to {human} (it stops early once your computer can't keep up)."
        )
