"""AutoPTZ Mark pre-flight dialog: defaults, ETA formula, NDI gating (offscreen)."""

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
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        s = dlg.session()
        assert s.profile in ("full", "streams")
        assert s.source == "synthetic"
        assert s.floor_fps == 24.0
        assert s.max_cameras == 16
        assert 14.0 <= s.dwell_s <= 21.0
        dlg.deleteLater()

    def test_defaults_round_trip_non_default(self, qtapp) -> None:
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(
            defaults=MarkSession(profile="streams", floor_fps=20.0, max_cameras=8, dwell_s=18.0)
        )
        s = dlg.session()
        assert s.profile == "streams"
        assert s.floor_fps == 20.0
        assert s.max_cameras == 8
        assert s.dwell_s == 18.0
        dlg.deleteLater()

    def test_eta_formula(self, qtapp) -> None:
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        eta = MarkPreflightDialog.estimated_seconds(MarkSession(max_cameras=4, dwell_s=10.0))
        assert eta >= 40.0  # 4 * 10 + overhead

    def test_eta_label_updates_on_control_change(self, qtapp) -> None:
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        before = dlg._eta_label.text()
        dlg._max_spin.setValue(dlg._max_spin.value() + 4)
        after = dlg._eta_label.text()
        assert before != after
        dlg.deleteLater()

    def test_ndi_disabled_without_cyndilib(self, qtapp) -> None:
        from autoptz.benchmark.ndi_sim import ndi_sim_available
        from autoptz.ui.widgets.dialogs.mark_preflight import MarkPreflightDialog

        dlg = MarkPreflightDialog(defaults=MarkSession())
        # The NDI option is only enabled when cyndilib is present.
        assert dlg._ndi_radio.isEnabled() == ndi_sim_available()
        # When NDI is unavailable the chosen source falls back to synthetic.
        if not ndi_sim_available():
            assert dlg.session().source == "synthetic"
        dlg.deleteLater()
