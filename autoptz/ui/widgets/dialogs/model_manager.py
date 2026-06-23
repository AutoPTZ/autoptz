"""ModelManagerDialog - interactive model cache management."""

from __future__ import annotations

import logging
import threading
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from autoptz.ui import theme as T
from autoptz.ui.widgets.common import AccentButton, CostChip, DangerButton, section_label

log = logging.getLogger(__name__)

MODEL_SETUP_REMINDER_KEY = "model_setup_dont_remind"
_DETECTOR_TIERS = (
    ("auto", "Auto"),
    ("fast", "Fast"),
    ("balanced", "Balanced"),
    ("medium", "Accurate"),
)


def model_setup_reminder_suppressed(client: Any) -> bool:
    try:
        return bool(client.getSetting(MODEL_SETUP_REMINDER_KEY, False))
    except Exception:  # noqa: BLE001
        return False


def startup_missing_model_keys() -> list[str]:
    """Return the model keys that should trigger the startup setup window."""
    try:
        from autoptz.engine.runtime.models import default_manager

        rows = default_manager().app_model_statuses()
    except Exception:  # noqa: BLE001
        log.debug("startup model inventory failed", exc_info=True)
        return []
    detector_rows = [row for row in rows if row.get("kind") == "detector"]
    pose_row = next((row for row in rows if row.get("key") == "pose"), None)
    missing: list[str] = []
    if detector_rows and not any(bool(row.get("cached")) for row in detector_rows):
        missing.append("detector_fast")
    if pose_row is not None and not bool(pose_row.get("cached")):
        missing.append("pose")
    return missing


class _ModelTask(QObject):
    progress = Signal(str, int, int)
    done = Signal(object)

    def __init__(self, action: str, keys: list[str]) -> None:
        super().__init__()
        self._action = action
        self._keys = list(keys)

    def run(self) -> None:
        try:
            from autoptz.engine.runtime.models import default_manager

            manager = default_manager()
            if self._action == "download":
                results = manager.ensure_app_models(
                    keys=self._keys,
                    progress=lambda label, value, total: self.progress.emit(
                        label,
                        value,
                        total,
                    ),
                )
            else:
                self.progress.emit("Removing models", 0, 1)
                results = manager.remove_app_models(keys=self._keys)
                self.progress.emit("Removing models", 1, 1)
        except Exception as exc:  # noqa: BLE001
            results = [
                {
                    "name": "Model operation",
                    "state": "failed",
                    "path": "",
                    "error": str(exc),
                }
            ]
        self.done.emit({"action": self._action, "results": results})


class _ModelRow(QFrame):
    def __init__(self, row: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = str(row.get("key", ""))
        self.display_name = str(row.get("name", self.key))
        self.cached = bool(row.get("cached"))
        self.bundled = bool(row.get("bundled"))
        self.removable = bool(row.get("removable", self.cached and not self.bundled))
        self.active = bool(row.get("active"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("modelRow")
        # Flat list-row (thin separator) rather than a stack of bordered boxes.
        # No hover tint — the row isn't clickable (the checkbox is the target),
        # and a full-width highlight just looked like stray text selection.
        self.setStyleSheet(
            f"QFrame#modelRow {{ background: transparent; border: none;"
            f" border-bottom: 1px solid {T.CURRENT.border}; }}"
        )
        lay = QGridLayout(self)
        lay.setContentsMargins(4, 10, 4, 10)
        lay.setHorizontalSpacing(10)
        lay.setVerticalSpacing(4)

        self.checkbox = QCheckBox()
        if self.bundled:
            # Shipped inside the app: nothing to download or remove.
            self.checkbox.setEnabled(False)
            self.checkbox.setToolTip("Included with AutoPTZ — always available, nothing to do.")
        else:
            self.checkbox.setToolTip("Include this model in Download Selected or Remove Selected.")
        lay.addWidget(self.checkbox, 0, 0, 2, 1, Qt.AlignmentFlag.AlignTop)

        title = QLabel(f"<b>{self.display_name}</b>")
        title.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(title, 0, 1)

        cost = CostChip(str(row.get("cost", "Light")))
        lay.addWidget(cost, 0, 2, Qt.AlignmentFlag.AlignLeft)

        if self.active:
            status_text = "ACTIVE"
        elif self.bundled:
            status_text = "INCLUDED"
        elif self.cached:
            status_text = "DOWNLOADED"
        else:
            status_text = "NOT DOWNLOADED"
        status = QLabel(status_text)
        status_color = T.TRACKING if self.cached else T.CURRENT.muted
        if self.active:
            status_color = T.ACCENT.name()
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setStyleSheet(
            f"color: {status_color}; border: 1px solid {status_color};"
            f" border-radius: 8px; padding: 1px 8px; font-size: {T.fs(9)}px;"
            " font-weight: 700;"
        )
        lay.addWidget(status, 0, 3, Qt.AlignmentFlag.AlignRight)

        label = str(row.get("label", ""))
        description = str(row.get("description", ""))
        why = str(row.get("why", ""))
        size = str(row.get("size", ""))
        path = str(row.get("path", ""))
        meta_bits = [bit for bit in (label, size or "not downloaded") if bit]
        detail = QLabel(
            f"<span style='color:{T.CURRENT.subtext}'>"
            f"{description}{' · ' if description else ''}{' · '.join(meta_bits)}</span>"
        )
        detail.setTextFormat(Qt.TextFormat.RichText)
        detail.setWordWrap(True)
        if why or path:
            detail.setToolTip(f"Used for: {why}\nPath: {path}")
        lay.addWidget(detail, 1, 1, 1, 3)
        lay.setColumnStretch(1, 1)

    def is_checked(self) -> bool:
        return bool(self.checkbox.isChecked())

    def set_checked(self, checked: bool) -> None:
        # Bundled rows have a disabled checkbox — Select All/Missing skip them.
        if self.checkbox.isEnabled():
            self.checkbox.setChecked(bool(checked))

    def set_controls_enabled(self, enabled: bool) -> None:
        self.checkbox.setEnabled(bool(enabled))


class _ExternalRow(QFrame):
    def __init__(self, row: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("externalModelRow")
        # Flat list-row to match _ModelRow (no nested card).
        self.setStyleSheet(
            f"QFrame#externalModelRow {{ background: transparent; border: none;"
            f" border-bottom: 1px solid {T.CURRENT.border}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 9, 6, 9)
        lay.setSpacing(3)
        state = str(row.get("state", "off"))
        color = T.TRACKING if state == "ok" else T.WARNING if state == "warn" else T.CURRENT.muted
        title = QLabel(
            f"<b>{row.get('name', row.get('key', 'External model'))}</b> "
            f"<span style='color:{color}'>({state.upper()})</span>"
        )
        title.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(title)
        detail = QLabel(
            f"<span style='color:{T.CURRENT.subtext}'>"
            f"{row.get('detail', '')} · only labels bodies the detector finds.<br>"
            f"{row.get('managed', 'Managed outside AutoPTZ.')}</span>"
        )
        detail.setTextFormat(Qt.TextFormat.RichText)
        detail.setWordWrap(True)
        detail.setToolTip(
            f"Used for: {row.get('why', 'optional feature')}\nPath: {row.get('path', '-')}"
        )
        lay.addWidget(detail)


class ModelManagerDialog(QDialog):
    """Interactive model download/removal window."""

    def __init__(
        self,
        client: Any,
        *,
        startup_prompt: bool = False,
        selected_keys: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client
        self._startup_prompt = bool(startup_prompt)
        self._initial_selected = set(selected_keys or [])
        self._rows: dict[str, _ModelRow] = {}
        self._status_by_key: dict[str, dict[str, Any]] = {}
        self._task: _ModelTask | None = None
        self.setWindowTitle("AutoPTZ Models")
        self.setModal(True)
        self.setMinimumSize(880, 700)
        self.resize(940, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(10)

        title = QLabel("Model Setup" if startup_prompt else "Models")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        root.addWidget(title)
        intro = QLabel(
            "The <b>detector</b> model is the foundation — it finds the bodies that "
            "everything else builds on. Face recognition, pose and ReID only label "
            "or stabilise bodies the detector already found, so with no detector "
            "model nothing is drawn. Face and ReID weights live in their upstream "
            "package caches: AutoPTZ doesn't delete those files, but it unloads them "
            "from memory whenever their feature is switched off in Services."
        )
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {T.CURRENT.text};")
        root.addWidget(intro)

        tier_row = QHBoxLayout()
        tier_row.setSpacing(8)
        tier_row.addWidget(QLabel("Active detector tier"))
        self._tier = QComboBox()
        for value, label in _DETECTOR_TIERS:
            self._tier.addItem(label, value)
        self._tier.setToolTip(
            "The detector model tier the engine uses (Auto maps to Fast). Tiers "
            "that aren't downloaded are greyed out unless automatic download is on."
        )
        self._tier.currentIndexChanged.connect(self._on_tier_changed)
        tier_row.addWidget(self._tier, 1)
        root.addLayout(tier_row)

        self._auto_download = QCheckBox(
            "Automatically download a missing detector tier when I select it"
        )
        self._auto_download.setToolTip(
            "When off, selecting a missing detector tier will not download/export it. "
            "Download it from this window first."
        )
        self._auto_download.setChecked(_safe_bool(lambda: self._client.autoDownloadModels()))
        self._auto_download.toggled.connect(self._set_auto_download)
        root.addWidget(self._auto_download)

        scroll = QScrollArea(self)
        scroll.setObjectName("modelScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # One clean bordered panel holding flat list-rows, instead of a stack of
        # individually-boxed cards (which read as cluttered/ugly).
        scroll.setStyleSheet(
            f"QScrollArea#modelScroll {{ border: 1px solid {T.CURRENT.border};"
            f" border-radius: {T.RADIUS}px; background: {T.CURRENT.surface}; }}"
        )
        body = QWidget()
        body.setStyleSheet(f"background: {T.CURRENT.surface};")
        self._list = QVBoxLayout(body)
        self._list.setContentsMargins(8, 6, 8, 6)
        self._list.setSpacing(0)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        root.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {T.CURRENT.subtext};")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._remind = QCheckBox("Do not remind me on startup when managed models are missing")
        self._remind.setChecked(model_setup_reminder_suppressed(client))
        self._remind.toggled.connect(self._set_reminder_suppressed)
        root.addWidget(self._remind)

        buttons = QHBoxLayout()
        self._select_missing = QPushButton("Select Missing")
        self._select_missing.clicked.connect(self._select_missing_rows)
        buttons.addWidget(self._select_missing)
        self._select_all = QPushButton("Select All")
        self._select_all.clicked.connect(lambda: self._set_all_rows(True))
        buttons.addWidget(self._select_all)
        self._select_none = QPushButton("Select None")
        self._select_none.clicked.connect(lambda: self._set_all_rows(False))
        buttons.addWidget(self._select_none)
        buttons.addStretch(1)
        self._remove = DangerButton("Remove Selected")
        self._remove.clicked.connect(self._remove_selected)
        buttons.addWidget(self._remove)
        self._download = AccentButton("Download Selected")
        self._download.clicked.connect(self._download_selected)
        buttons.addWidget(self._download)
        self._close = QPushButton("Close")
        self._close.clicked.connect(self.accept)
        buttons.addWidget(self._close)
        root.addLayout(buttons)

        self._refresh_tier()
        self._refresh_rows()
        _connect(self._client, "modelDownloadStarted", self._on_client_download_started)
        _connect(self._client, "modelDownloadProgress", self._on_progress)
        _connect(self._client, "modelDownloadFinished", self._on_client_download_finished)

    def _refresh_tier(self) -> None:
        tier = "auto"
        try:
            tier = str(self._client.getDetectorModelTier() or "auto")
        except Exception:  # noqa: BLE001
            pass
        idx = self._tier.findData(tier)
        if idx < 0:
            idx = self._tier.findData("auto")
        if idx >= 0:
            self._tier.blockSignals(True)
            self._tier.setCurrentIndex(idx)
            self._tier.blockSignals(False)

    def _on_tier_changed(self, _index: int) -> None:
        tier = str(self._tier.currentData() or "auto")
        key = _detector_key_for_tier(tier)
        row = self._status_by_key.get(key, {})
        if row and not bool(row.get("cached")) and not self._auto_download.isChecked():
            self._status.setText(
                f"{self._tier.currentText()} is not downloaded. Download it first, "
                "or enable automatic downloads."
            )
            self._refresh_tier()
            return
        try:
            self._client.setDetectorModelTier(tier)
        except Exception:  # noqa: BLE001
            log.debug("detector tier update failed", exc_info=True)

    def _set_auto_download(self, checked: bool) -> None:
        try:
            self._client.setAutoDownloadModels(bool(checked))
        except Exception:  # noqa: BLE001
            log.debug("persist auto model download preference failed", exc_info=True)
        self._refresh_tier_item_states()

    def _refresh_tier_item_states(self) -> None:
        """Grey out tier options whose model isn't downloaded (auto-download off).

        Selecting a tier you can't actually run is the confusing case to avoid:
        with auto-download off only downloaded tiers stay selectable; with it on,
        missing tiers are annotated "(will download)".
        """
        combo = getattr(self, "_tier", None)
        if combo is None:
            return
        auto_dl = self._auto_download.isChecked()
        model = combo.model()
        labels = dict(_DETECTOR_TIERS)
        combo.blockSignals(True)
        try:
            for i in range(combo.count()):
                value = str(combo.itemData(i) or "auto")
                key = _detector_key_for_tier(value)
                cached = bool(self._status_by_key.get(key, {}).get("cached"))
                if cached:
                    suffix = ""
                elif auto_dl:
                    suffix = "  (will download)"
                else:
                    suffix = "  (not downloaded)"
                combo.setItemText(i, labels.get(value, value) + suffix)
                item = model.item(i) if hasattr(model, "item") else None
                if item is not None:
                    item.setEnabled(cached or auto_dl)
        finally:
            combo.blockSignals(False)

    def _refresh_rows(self) -> None:
        selected = {key for key, row in self._rows.items() if row.is_checked()}
        if not selected:
            selected = set(self._initial_selected)
        while self._list.count():
            item = self._list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows.clear()
        self._status_by_key.clear()

        try:
            from autoptz.engine.runtime.models import default_manager

            model_rows = default_manager().app_model_statuses()
        except Exception as exc:  # noqa: BLE001
            error = QLabel(f"Model inventory failed: {exc}")
            error.setWordWrap(True)
            self._list.addWidget(error)
            model_rows = []
        active_key = _detector_key_for_tier(_safe_str(lambda: self._client.getDetectorModelTier()))
        for row in model_rows:
            row["active"] = row.get("key") == active_key
            self._status_by_key[str(row.get("key", ""))] = dict(row)

        downloaded = [row for row in model_rows if row.get("cached")]
        missing = [row for row in model_rows if not row.get("cached")]
        self._add_section_label("Downloaded", first=True)
        if downloaded:
            for row in downloaded:
                self._add_model_row(row, selected)
        else:
            self._add_empty_note("No AutoPTZ-managed models are downloaded.")

        self._add_section_label("Available to download")
        if missing:
            for row in missing:
                self._add_model_row(row, selected)
        else:
            self._add_empty_note("All AutoPTZ-managed models are downloaded.")

        self._add_section_label("Upstream-managed models")
        external = []
        try:
            external = [
                row
                for row in (self._client.optionalComponents() or [])
                if row.get("key") in {"face", "reid"}
            ]
        except Exception:  # noqa: BLE001
            log.debug("optional component inventory failed", exc_info=True)
        if external:
            for row in external:
                self._list.addWidget(_ExternalRow(row))
        else:
            self._add_empty_note("No upstream-managed optional model packs were reported.")
        self._list.addStretch(1)
        self._initial_selected.clear()
        self._refresh_action_state()
        self._refresh_tier_item_states()

    def _add_model_row(self, row: dict[str, Any], selected: set[str]) -> None:
        widget = _ModelRow(row)
        widget.set_checked(str(row.get("key", "")) in selected)
        widget.checkbox.toggled.connect(lambda _checked: self._refresh_action_state())
        self._rows[widget.key] = widget
        self._list.addWidget(widget)

    def _add_section_label(self, text: str, *, first: bool = False) -> None:
        """Add a section caption with breathing room above it (except the first)."""
        if not first:
            self._list.addSpacing(14)
        cap = section_label(text)
        cap.setContentsMargins(2, 0, 0, 4)
        self._list.addWidget(cap)

    def _add_empty_note(self, text: str) -> None:
        note = QLabel(text)
        note.setWordWrap(True)
        note.setContentsMargins(4, 4, 4, 4)
        note.setStyleSheet(f"color: {T.CURRENT.subtext};")
        self._list.addWidget(note)

    def _selected_keys(self) -> list[str]:
        return [key for key, row in self._rows.items() if row.is_checked()]

    def _select_missing_rows(self) -> None:
        for row in self._rows.values():
            row.set_checked(not row.cached)
        self._refresh_action_state()

    def _set_all_rows(self, checked: bool) -> None:
        for row in self._rows.values():
            row.set_checked(checked)
        self._refresh_action_state()

    def _set_busy(self, busy: bool) -> None:
        for row in self._rows.values():
            row.set_controls_enabled(not busy)
        for widget in (
            self._tier,
            self._select_missing,
            self._select_all,
            self._select_none,
            self._remove,
            self._download,
            self._remind,
            self._auto_download,
            self._close,
        ):
            widget.setEnabled(not busy)
        self._progress.setVisible(busy)
        if not busy:
            self._refresh_action_state()

    def _refresh_action_state(self) -> None:
        """Gate the action buttons on what the selection can actually do.

        Download lights up only when a *missing* model is selected (so once
        everything is downloaded the button is disabled); Remove only when a
        *cached* model is selected.
        """
        selected = [row for row in self._rows.values() if row.is_checked()]
        idle = self._task is None
        has_missing = any(not row.cached for row in selected)
        has_removable = any(row.removable for row in selected)
        self._download.setEnabled(has_missing and idle)
        self._remove.setEnabled(has_removable and idle)
        self._download.setToolTip(
            "Download the selected models that aren't available yet."
            if has_missing
            else "Select a model that isn't downloaded yet to enable this."
        )
        self._remove.setToolTip(
            "Remove the selected downloaded models from the local cache."
            if has_removable
            else "Select a downloaded model to enable this (included models can't be removed)."
        )

    def _set_reminder_suppressed(self, checked: bool) -> None:
        try:
            self._client.setSetting(MODEL_SETUP_REMINDER_KEY, bool(checked))
        except Exception:  # noqa: BLE001
            log.debug("persist model reminder preference failed", exc_info=True)

    def _download_selected(self) -> None:
        self._run_task("download")

    def _remove_selected(self) -> None:
        keys = self._selected_keys()
        if not keys:
            return
        names = [self._rows[key].display_name for key in keys if key in self._rows]
        answer = QMessageBox.question(
            self,
            "Remove Selected Models",
            "Remove the selected AutoPTZ-managed model files from the local cache?\n\n"
            + "\n".join(_strip_html(name) for name in names[:6]),
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._run_task("remove")

    def _run_task(self, action: str) -> None:
        if self._task is not None:
            return
        keys = self._selected_keys()
        if not keys:
            QMessageBox.information(self, "No Models Selected", "Select at least one model first.")
            return
        task = _ModelTask(action, keys)
        self._task = task
        task.progress.connect(self._on_progress)
        task.done.connect(self._on_done)
        self._set_busy(True)
        self._status.setText("Starting model operation...")
        threading.Thread(target=task.run, name=f"ui-model-{action}", daemon=True).start()

    def _on_client_download_started(self, label: str) -> None:
        if self._task is not None:
            return
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._progress.setFormat(label)
        self._status.setText(label)

    def _on_progress(self, label: str, value: int, total: int) -> None:
        total = max(1, int(total))
        value = max(0, min(total, int(value)))
        self._progress.setRange(0, total)
        self._progress.setValue(value)
        self._progress.setFormat(f"{label} ({value}/{total})")

    def _on_client_download_finished(self, ok: bool, message: str) -> None:
        if self._task is not None:
            return
        self._progress.setVisible(False)
        self._status.setText(message)
        self._refresh_rows()

    def _on_done(self, payload: object) -> None:
        self._task = None
        self._set_busy(False)
        action = ""
        results: list[Any] = []
        if isinstance(payload, dict):
            action = str(payload.get("action", ""))
            raw = payload.get("results", [])
            if isinstance(raw, list):
                results = raw
        failed = [
            row
            for row in results
            if isinstance(row, dict) and row.get("state") not in {"ok", "removed"}
        ]
        # Tell a running engine to unload + rebuild from the new cache state so a
        # removed model stops drawing boxes (and a freshly downloaded one is
        # picked up) without a manual restart.
        try:
            self._client.applyModelCacheChanged()
        except Exception:  # noqa: BLE001
            log.debug("applyModelCacheChanged failed", exc_info=True)
        self._refresh_rows()
        if failed:
            detail = "\n".join(
                f"- {row.get('name', 'Model')}: {row.get('error', 'failed')}" for row in failed[:6]
            )
            self._status.setText("Some model operations failed.")
            QMessageBox.warning(
                self,
                "Model Operation Incomplete",
                f"Some model operations failed.\n\n{detail}",
            )
            return
        if action == "remove":
            self._status.setText(f"Removed {len(results)} cached model file(s).")
        else:
            self._status.setText("Selected models are cached.")


def _strip_html(text: str) -> str:
    return text.replace("<b>", "").replace("</b>", "").replace("<br>", "\n").replace("&amp;", "&")


def _detector_key_for_tier(tier: str) -> str:
    try:
        from autoptz.engine.runtime.models import detector_key_for_tier

        return detector_key_for_tier(tier)
    except Exception:  # noqa: BLE001
        return "detector_fast"


def _connect(obj: Any, name: str, slot: Any) -> None:
    try:
        getattr(obj, name).connect(slot)
    except Exception:  # noqa: BLE001
        log.debug("connect %s failed", name, exc_info=True)


def _safe_bool(fn: Any) -> bool:
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001
        return False


def _safe_str(fn: Any) -> str:
    try:
        return str(fn() or "auto")
    except Exception:  # noqa: BLE001
        return "auto"
