"""AutoPTZ Mark pre-flight dialog (offscreen): friendly dropdowns + confirm.

The pre-flight uses plain language and dropdowns (no spinboxes / no "FPS floor" /
"Dwell" jargon) for Max cameras, Target FPS, Time per step, Resolution, and
Model, plus Source + Profile.  Its Start asks a confirm before accepting.
"""

from __future__ import annotations

import pytest

from autoptz.ui.mark_session import MarkSession


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class TestPreflight:
    def test_defaults_and_session(self, qtapp) -> None:
        from autoptz.benchmark.ndi_sim import ndi_sim_available
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        s = dlg.session()
        assert s.profile in ("full", "streams")
        # Default source is the bundled clip (real people) unless NDI was chosen.
        assert s.source == "clip"
        assert s.floor_fps in (24.0, 30.0)
        assert s.max_cameras in (1, 2, 4, 8, 12, 16)
        assert s.dwell_s in (5.0, 10.0, 15.0, 20.0)
        assert s.resolution in ("720p", "1080p", "4k")
        assert s.model in ("auto", "nano", "small", "medium")
        # The MarkSession() defaults flow straight through the dialog.
        assert s.source == "clip"
        assert s.floor_fps == 30.0
        assert s.max_cameras == 4
        assert s.dwell_s == 10.0
        assert s.resolution == "1080p"
        assert s.model == "small"
        assert ndi_sim_available() or s.source != "ndi"
        dlg.deleteLater()

    def test_defaults_round_trip_non_default(self, qtapp) -> None:
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(
            defaults=MarkSession(
                profile="streams",
                floor_fps=30.0,
                max_cameras=8,
                dwell_s=10.0,
                resolution="1080p",
                model="nano",
            )
        )
        s = dlg.session()
        assert s.profile == "streams"
        assert s.floor_fps == 30.0
        assert s.max_cameras == 8
        assert s.dwell_s == 10.0
        assert s.resolution == "1080p"
        assert s.model == "nano"
        dlg.deleteLater()

    def test_controls_are_dropdowns_not_spinboxes(self, qtapp) -> None:
        from PySide6.QtWidgets import QComboBox, QSpinBox

        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        # The parameter pickers are all combo boxes; no spinbox jargon remains.
        for name in ("_max_combo", "_fps_combo", "_step_combo", "_res_combo", "_model_combo"):
            assert isinstance(getattr(dlg, name), QComboBox)
        assert not dlg.findChildren(QSpinBox)
        dlg.deleteLater()

    def test_no_jargon_in_text(self, qtapp) -> None:
        from PySide6.QtWidgets import QLabel

        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        text = " ".join(lbl.text() for lbl in dlg.findChildren(QLabel)).lower()
        # The jargon the user called out must be gone.
        assert "fps floor" not in text
        assert "dwell" not in text
        assert "can't hold the fps floor" not in text
        assert "relaunch" not in text
        dlg.deleteLater()

    def test_resolution_and_model_options_present(self, qtapp) -> None:
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        res_keys = {dlg._res_combo.itemData(i) for i in range(dlg._res_combo.count())}
        model_keys = {dlg._model_combo.itemData(i) for i in range(dlg._model_combo.count())}
        assert {"720p", "1080p", "4k"} <= res_keys
        # All four model tiers are offered, default Small.
        assert {"auto", "nano", "small", "medium"} <= model_keys
        assert dlg._model_combo.currentData() == "small"
        dlg.deleteLater()

    def test_max_and_fps_and_step_options(self, qtapp) -> None:
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        max_keys = {dlg._max_combo.itemData(i) for i in range(dlg._max_combo.count())}
        fps_keys = {dlg._fps_combo.itemData(i) for i in range(dlg._fps_combo.count())}
        step_keys = {dlg._step_combo.itemData(i) for i in range(dlg._step_combo.count())}
        # Max cameras offers 1/2/4/8/12/16, default 4.
        assert {1, 2, 4, 8, 12, 16} <= max_keys
        assert dlg._max_combo.currentData() == 4
        # Target FPS offers 24/30, default 30.
        assert {24.0, 30.0} <= fps_keys
        assert dlg._fps_combo.currentData() == 30.0
        # Time per step offers 5/10/15/20 s, default the recommended 10 s.
        assert {5.0, 10.0, 15.0, 20.0} <= step_keys
        assert dlg._step_combo.currentData() == 10.0
        # The recommended entry is labelled so the user knows which is recommended.
        assert any(
            "recommend" in dlg._step_combo.itemText(i).lower()
            for i in range(dlg._step_combo.count())
        )
        dlg.deleteLater()

    def test_clip_is_default_source_option(self, qtapp) -> None:
        from autoptz.benchmark.ndi_sim import ndi_sim_available
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        # Three source radios: clip / synthetic / ndi; clip selected by default.
        assert dlg._clip_radio.isChecked()
        assert not dlg._synthetic_radio.isChecked()
        assert dlg.session().source == "clip"
        # NDI radio gating mirrors cyndilib availability.
        assert dlg._ndi_radio.isEnabled() == ndi_sim_available()
        dlg.deleteLater()

    def test_synthetic_source_selectable(self, qtapp) -> None:
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession(source="synthetic"))
        assert dlg._synthetic_radio.isChecked()
        assert dlg.session().source == "synthetic"
        dlg.deleteLater()

    def test_eta_formula(self, qtapp) -> None:
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        eta = MarkPreflightDialog.estimated_seconds(MarkSession(max_cameras=4, dwell_s=10.0))
        assert eta >= 40.0  # 4 * 10 + overhead

    def test_eta_label_updates_on_control_change(self, qtapp) -> None:
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession(max_cameras=4))
        before = dlg._eta_label.text()
        # Pick a different max-cameras option → ETA changes.
        idx = (dlg._max_combo.currentIndex() + 1) % dlg._max_combo.count()
        dlg._max_combo.setCurrentIndex(idx)
        after = dlg._eta_label.text()
        assert before != after
        dlg.deleteLater()

    def test_ndi_disabled_without_cyndilib(self, qtapp) -> None:
        from autoptz.benchmark.ndi_sim import ndi_sim_available
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        assert dlg._ndi_radio.isEnabled() == ndi_sim_available()
        if not ndi_sim_available():
            # NDI off → the default (clip) source stays selected, never NDI.
            assert dlg.session().source == "clip"
        dlg.deleteLater()

    def test_start_confirms_before_accept(self, qtapp, monkeypatch) -> None:
        from PySide6.QtWidgets import QDialog, QMessageBox

        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())

        # Declining the confirm keeps the dialog open (no accept).
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
        accepted: list[str] = []
        monkeypatch.setattr(QDialog, "accept", lambda self: accepted.append("accept"))
        dlg._on_start()
        assert accepted == []  # declined → no accept

        # Confirming accepts.
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
        dlg._on_start()
        assert accepted == ["accept"]
        dlg.deleteLater()
