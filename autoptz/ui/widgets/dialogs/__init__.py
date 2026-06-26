"""Modal dialogs for the native UI."""

from __future__ import annotations

from autoptz.ui.widgets.dialogs.about import AboutDialog
from autoptz.ui.widgets.dialogs.experimental import ExperimentalFeaturesDialog
from autoptz.ui.widgets.dialogs.model_manager import ModelManagerDialog
from autoptz.ui.widgets.dialogs.network_camera import NetworkCameraDialog
from autoptz.ui.widgets.dialogs.person_detail import PersonDetailDialog
from autoptz.ui.widgets.dialogs.register_person import RegisterPersonDialog

__all__ = [
    "AboutDialog",
    "ExperimentalFeaturesDialog",
    "ModelManagerDialog",
    "NetworkCameraDialog",
    "PersonDetailDialog",
    "RegisterPersonDialog",
]
