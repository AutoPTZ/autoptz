"""PySide6 / QML application entry point.

Usage::

    python -m autoptz.ui        # direct launch
    from autoptz.ui.app import run; run()   # programmatic

The QML engine is given two context properties:

    engineClient  — EngineClient QObject (commands / model)
    frameProvider is registered as image provider "frame"

All heavy work (ingest, inference, PTZ) runs in the engine processes.  The UI
thread only processes Qt events, model updates, and image uploads.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_QML_DIR = Path(__file__).parent / "qml"
_MAIN_QML = _QML_DIR / "CameraWall.qml"


def run(argv: list[str] | None = None) -> int:
    """Launch the AutoPTZ UI.  Returns the process exit code."""
    # Import PySide6 here so the rest of the package is importable without it.
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine

    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.providers import ShmFrameProvider

    app = QGuiApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("AutoPTZ")
    app.setOrganizationName("AutoPTZ")
    app.setApplicationVersion("2.0.0a0")

    client = EngineClient()
    provider = ShmFrameProvider()

    engine = QQmlApplicationEngine()

    # Register the image provider before loading QML
    engine.addImageProvider("frame", provider)

    # Expose Python objects to QML
    ctx = engine.rootContext()
    ctx.setContextProperty("engineClient", client)
    ctx.setContextProperty("qmlDir", str(_QML_DIR))

    if not _MAIN_QML.exists():
        log.error("Main QML not found: %s", _MAIN_QML)
        return 1

    engine.load(str(_MAIN_QML))

    roots = engine.rootObjects()
    if not roots:
        log.error("QML engine failed to load %s", _MAIN_QML)
        return 1

    log.info("AutoPTZ UI started")

    exit_code = app.exec()

    # Clean up shared-memory readers before exit
    provider.detach_all()
    log.info("AutoPTZ UI exited (code %d)", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
