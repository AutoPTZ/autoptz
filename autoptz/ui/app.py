"""PySide6 / Qt Widgets application entry point.

Usage::

    python -m autoptz.ui        # direct launch
    from autoptz.ui.app import run; run()   # programmatic

The UI is native Qt Widgets (a ``QMainWindow`` shell with dockable panels around
a camera wall).  All heavy work (ingest, inference, PTZ) runs in the engine; the
GUI thread only processes Qt events, model updates, and frame painting.

Wiring that is reused verbatim from the engine layer:
  * ``ConfigStore`` — persistence.
  * ``EngineClient`` — typed command/telemetry bridge (+ its list models).
  * the deferred ``Supervisor`` factory (heavy imports only on first start).
  * a ~30 Hz GUI-thread command pump (``sup.tick()``).
  * the in-app log bridge (``LogListModel`` + ``QtLogHandler``).
  * worker preview frames over shared memory, now painted by the camera tiles.
"""
from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def _set_macos_app_name(name: str) -> None:
    """Best-effort: set the runtime process/bundle name on macOS via PyObjC.

    Unbundled ``python -m autoptz`` makes macOS use the process name ("Python")
    for the app menu.  The definitive fix is a proper ``.app`` bundle with
    ``CFBundleName``; this is the interim best effort.  Any failure is swallowed.
    """
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle  # type: ignore[import-not-found]

        bundle = NSBundle.mainBundle()
        if bundle is not None:
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info is not None:
                info["CFBundleName"] = name
                info["CFBundleDisplayName"] = name
    except Exception:  # noqa: BLE001 — cosmetic only; never block launch
        log.debug("Could not set macOS bundle name via PyObjC", exc_info=True)


def run(argv: list[str] | None = None) -> int:
    """Launch the AutoPTZ UI.  Returns the process exit code."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from autoptz.config.store import ConfigStore
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.log_bridge import LogListModel, QtLogHandler
    from autoptz.ui.theme import ThemeController
    from autoptz.ui.widgets import MainWindow

    # Preserve fractional display scaling so our UI-scale font sizes stay crisp on
    # high-DPI screens (must be set before the QApplication is constructed).
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:  # noqa: BLE001
        log.debug("Could not set high-DPI rounding policy", exc_info=True)

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("AutoPTZ")
    app.setOrganizationName("AutoPTZ")
    app.setApplicationVersion("2.0.0a0")
    app.setApplicationDisplayName("AutoPTZ")
    _set_macos_app_name("AutoPTZ")

    # Persistent config store — creates the DB on first run.
    store = ConfigStore()

    client = EngineClient(store=store)
    frames = ShmFrameSource()

    # ── in-app logging viewer ──────────────────────────────────────────────────
    # A ring-buffered model fed by a logging.Handler on the root logger; the Logs
    # panel binds to it.  INFO by default so the console is informative; the
    # console's level control can raise it to DEBUG for the full pipeline log.
    log_model = LogListModel()
    log_handler = QtLogHandler(log_model)
    logging.getLogger().addHandler(log_handler)
    client.set_log_bridge(log_model, log_handler)
    logging.getLogger().setLevel(logging.INFO)

    # ── engine wiring ──────────────────────────────────────────────────────────
    # The supervisor is created lazily on first start (defers heavy ML imports);
    # the engine defaults to STOPPED until auto-start fires below.
    def _make_supervisor(engine_client: EngineClient):  # noqa: ANN202
        from autoptz.engine.supervisor import Supervisor
        return Supervisor(engine_client, store=store)

    client.set_supervisor_factory(_make_supervisor)

    # Bridge worker-side provider attach/detach onto the GUI thread.  Queued so
    # they run on the GUI thread even when emitted from a worker/pump thread.
    # NOTE: providerAttachRequested carries (camera_id, shm_name, w, h); the
    # frame source's attach takes (camera_id, shm_name, height, width).
    client.providerAttachRequested.connect(
        lambda cid, shm, w, h: frames.attach(cid, shm, h, w),
        Qt.ConnectionType.QueuedConnection,
    )
    client.providerDetachRequested.connect(
        frames.detach, Qt.ConnectionType.QueuedConnection,
    )

    # GUI-thread command pump: drains EngineClient commands to workers.  Safe
    # no-op while the engine is stopped.
    pump_timer = QTimer()
    pump_timer.setInterval(33)  # ~30 Hz

    def _pump() -> None:
        sup = client._supervisor  # set by startEngine via the factory
        if sup is not None and sup.is_running:
            sup.tick()

    pump_timer.timeout.connect(_pump)
    pump_timer.start()

    # ── theme + window ─────────────────────────────────────────────────────────
    theme = ThemeController(app, client)
    window = MainWindow(
        client, log_model=log_model, frame_source=frames, theme=theme,
    )
    window.show()

    log.info("AutoPTZ UI started")

    # ── engine auto-start ──────────────────────────────────────────────────────
    # Restore the last on/off state (default ON) and start after the window is
    # shown so the first paint happens before the heavy ingest/ML imports run.
    if bool(store.get_setting("engine_running", True)):
        QTimer.singleShot(0, client.startEngine)

    exit_code = app.exec()

    # Persist the engine on/off state for the next launch.
    try:
        store.set_setting("engine_running", bool(client.engineRunning))
    except Exception:  # noqa: BLE001
        log.exception("Error persisting engine_running on shutdown")

    # Orderly shutdown: stop the pump + engine before touching the store so no
    # worker thread is still pushing telemetry / draining commands.
    pump_timer.stop()
    try:
        client.stopEngine()
    except Exception:  # noqa: BLE001
        log.exception("Error stopping engine on shutdown")

    logging.getLogger().removeHandler(log_handler)

    store.flush()
    store.close()
    frames.detach_all()
    log.info("AutoPTZ UI exited (code %d)", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
