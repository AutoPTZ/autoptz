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
import os
import sys
import threading
import warnings
from typing import Any

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


def _queue_macos_camera_access_result(bridge: object, granted: bool) -> None:
    """Deliver a TCC permission result on the Qt event loop when possible."""
    resolve = getattr(bridge, "resolve", None)
    if resolve is not None:
        try:
            from PySide6.QtCore import Q_ARG, QMetaObject, Qt  # noqa: PLC0415

            queued = QMetaObject.invokeMethod(
                bridge,
                "resolve",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(bool, bool(granted)),
            )
            if queued:
                return
        except Exception:  # noqa: BLE001
            log.debug("Could not queue macOS camera access result onto Qt", exc_info=True)

    bridge.resolved.emit(bool(granted))  # type: ignore[attr-defined]


def _install_signal_shutdown(app: object, hard_exit_delay: float | None = 6.0) -> None:
    """Route SIGINT/SIGTERM through a clean Qt quit, with a hard-exit backstop.

    Qt's event loop blocks in native code, so a signal would otherwise terminate the
    process WITHOUT running the orderly shutdown after ``app.exec()`` — orphaning
    model-server camera child processes (spawned under ``AUTOPTZ_MODEL_SERVER``),
    which keep holding RAM + the accelerator. Asking the
    app to quit lets ``app.exec()`` return so ``client.stopEngine()`` →
    ``supervisor.stop()`` terminates the children cleanly. The always-on ~30 Hz pump
    timer keeps the loop returning to Python so the handler fires within a frame.

    Backstop: if the clean quit doesn't complete within ``hard_exit_delay`` seconds (a
    wedged teardown, or a C library that replaced our handler so app.quit never ran),
    force-exit so a signal ALWAYS stops the app — the spawned children reap themselves
    via their own parent-death watchdog. Pass ``hard_exit_delay=None`` to disable the
    backstop (tests). Best-effort: ``signal.signal`` only works on the main thread.
    """
    import signal  # noqa: PLC0415
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415

    def _hard_exit() -> None:
        time.sleep(hard_exit_delay or 0)
        os._exit(0)

    def _handler(_signum: int, _frame: object) -> None:
        quit_fn = getattr(app, "quit", None)
        if callable(quit_fn):
            quit_fn()
        if hard_exit_delay is not None:
            threading.Thread(target=_hard_exit, name="signal-hard-exit", daemon=True).start()

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _handler)
        except (ValueError, OSError):  # not main thread / unsupported platform
            log.debug("could not install shutdown handler for signal %s", _sig, exc_info=True)


def _start_engine_after_macos_camera_preflight(client: object, bridge: object | None) -> None:
    """Start the engine after macOS camera authorization is known.

    AVFoundation/OpenCV camera permission must be requested from the GUI process,
    not from a capture worker thread.  If the OS prompt is pending, this returns
    immediately and starts the engine from the async permission callback.

    ``AUTOPTZ_SKIP_CAMERA_PREFLIGHT`` forces the engine to start regardless of
    camera authorization — for setups with no local camera (NDI / RTSP / the
    synthetic test source) or headless/CI runs, where gating on the camera prompt
    would otherwise keep the engine from ever starting.
    """
    if os.environ.get("AUTOPTZ_SKIP_CAMERA_PREFLIGHT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        client.startEngine()  # type: ignore[attr-defined]
        return
    if sys.platform != "darwin" or bridge is None:
        client.startEngine()  # type: ignore[attr-defined]
        return
    try:
        import AVFoundation  # type: ignore  # noqa: PLC0415

        media = AVFoundation.AVMediaTypeVideo
        status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(media)
    except Exception:  # noqa: BLE001 — no AVFoundation; let non-native paths run
        log.debug("macOS camera authorization preflight unavailable", exc_info=True)
        client.startEngine()  # type: ignore[attr-defined]
        return

    if status == 3:  # AVAuthorizationStatusAuthorized
        client.startEngine()  # type: ignore[attr-defined]
        return

    def _report_blocked(reason: str) -> None:
        if getattr(sys, "frozen", False):
            # Packaged app: it has the camera-usage entitlement and can be granted
            # directly in the Camera privacy pane.
            message = (
                f"Camera access is {reason}. Grant access in System Settings > "
                "Privacy & Security > Camera, then restart AutoPTZ."
            )
        else:
            # Source run: the bare Python binary has no NSCameraUsageDescription, so
            # macOS can't show the consent prompt for it — it attributes camera use
            # to the *terminal* that launched AutoPTZ. Grant camera access to that
            # terminal app (Terminal / iTerm / VS Code / PyCharm …), or run the
            # packaged AutoPTZ app, then restart.
            message = (
                "Camera access is unavailable when running AutoPTZ from source: the "
                "Python interpreter can't request camera permission directly. Grant "
                "camera access to the terminal app you launched it from (System "
                "Settings > Privacy & Security > Camera), or use the packaged AutoPTZ "
                "app, then restart."
            )
        log.warning(message)
        try:
            client.errorOccurred.emit(message)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            log.debug("could not surface camera-access error", exc_info=True)

    if status in (1, 2):  # restricted / denied
        _report_blocked("restricted" if status == 1 else "denied")
        return

    def _on_resolved(granted: bool) -> None:
        try:
            bridge.resolved.disconnect(_on_resolved)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        if granted:
            client.startEngine()  # type: ignore[attr-defined]
        else:
            _report_blocked("denied")

    try:
        bridge.resolved.connect(_on_resolved)  # type: ignore[attr-defined]
        log.info("Requesting macOS camera access before engine auto-start.")

        def _completion_handler(granted: bool) -> None:
            try:
                _queue_macos_camera_access_result(bridge, bool(granted))
            except Exception:  # noqa: BLE001 — never let Python escape into TCC
                log.exception("macOS camera authorization callback failed")

        AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            media,
            _completion_handler,
        )
    except Exception:  # noqa: BLE001
        log.debug("macOS camera authorization request failed", exc_info=True)
        _report_blocked("unavailable")


def _build_main_window(
    client: Any,
    *,
    log_model: Any,
    frames: Any,
    theme: Any,
) -> Any:
    """Construct the normal :class:`MainWindow` (extracted for mode routing)."""
    from autoptz.ui.widgets import MainWindow

    return MainWindow(
        client,
        log_model=log_model,
        frame_source=frames,
        theme=theme,
    )


def run(argv: list[str] | None = None) -> int:
    """Launch the AutoPTZ UI.  Returns the process exit code.

    Always builds the normal :class:`MainWindow`.  AutoPTZ Mark is reached
    **in-process** from Help → Run AutoPTZ Mark… (the MainWindow suspends and shows
    an isolated :class:`MarkWindow`), so there is no subprocess relaunch or
    ``--mark`` compatibility path.
    """
    from PySide6.QtCore import QEventLoop, QObject, Qt, QTimer, Signal, Slot
    from PySide6.QtWidgets import QApplication

    from autoptz.config.store import ConfigStore
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.log_bridge import LogListModel, QtLogHandler
    from autoptz.ui.theme import ThemeController

    # Preserve fractional display scaling so our UI-scale font sizes stay crisp on
    # high-DPI screens (must be set before the QApplication is constructed).
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:  # noqa: BLE001
        log.debug("Could not set high-DPI rounding policy", exc_info=True)

    from autoptz import version as _app_version
    from autoptz.ui.branding import app_icon

    # Reuse an existing QApplication when one is already live (e.g. a relaunch
    # within the same process, or tests that route through run() repeatedly);
    # constructing a second QApplication in one process aborts.
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("AutoPTZ")
    app.setOrganizationName("AutoPTZ")
    app.setApplicationVersion(_app_version())
    if hasattr(app, "setApplicationDisplayName"):
        app.setApplicationDisplayName("AutoPTZ")
    if hasattr(app, "setWindowIcon"):
        app.setWindowIcon(app_icon())
    _set_macos_app_name("AutoPTZ")

    # In-process Mark swap (Help → Run AutoPTZ Mark…): hiding MainWindow or closing
    # MarkWindow must NOT quit the app.  Only an explicit quit (the visible
    # MainWindow closed, or Mark's Quit choice) calls app.quit().  This single line
    # is the linchpin of the suspend/resume lifecycle.
    has_qapplication_window_api = hasattr(app, "setQuitOnLastWindowClosed")
    if has_qapplication_window_api:
        app.setQuitOnLastWindowClosed(False)
    else:
        # Some tests leave a QCoreApplication behind. Keep the startup contract
        # observable without constructing a second Qt application object.
        _quit_state = {"value": False}

        try:
            app.setQuitOnLastWindowClosed = lambda value: _quit_state.__setitem__(  # type: ignore[attr-defined]
                "value", bool(value)
            )
            app.quitOnLastWindowClosed = lambda: bool(_quit_state["value"])  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            log.debug("Could not shim quitOnLastWindowClosed on Qt application", exc_info=True)

    # Persistent config store — creates the DB on first run.
    store = ConfigStore()

    client = EngineClient(store=store)
    frames = ShmFrameSource()

    # ── in-app logging viewer ──────────────────────────────────────────────────
    # A ring-buffered model fed by a logging.Handler on the root logger; the Logs
    # panel binds to it.  INFO by default in-app, while existing stderr/terminal
    # handlers stay at WARNING so development launches are not spammed.
    log_model = LogListModel()
    log_handler = QtLogHandler(log_model)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    log_handler.setLevel(logging.INFO)
    client.set_log_bridge(log_model, log_handler)
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        if handler is not log_handler:
            handler.setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface(\.|$)")

    # ── engine wiring ──────────────────────────────────────────────────────────
    # The supervisor is created lazily on first start (defers heavy ML imports);
    # the engine defaults to STOPPED until auto-start fires below.
    supervisor_holder: dict[str, Any] = {}
    supervisor_lock = threading.Lock()

    def _load_supervisor_class() -> Any:
        with supervisor_lock:
            cls = supervisor_holder.get("cls")
        if cls is not None:
            return cls
        from autoptz.engine.supervisor import Supervisor

        with supervisor_lock:
            supervisor_holder["cls"] = Supervisor
        return Supervisor

    def _preload_supervisor_class() -> None:
        try:
            _load_supervisor_class()
        except Exception as exc:  # noqa: BLE001
            with supervisor_lock:
                supervisor_holder["error"] = str(exc) or type(exc).__name__
            log.exception("Engine preload failed")

    def _make_supervisor(engine_client: EngineClient) -> Any:
        Supervisor = _load_supervisor_class()

        return Supervisor(engine_client, store=store)

    client.set_supervisor_factory(_make_supervisor)
    threading.Thread(
        target=_preload_supervisor_class,
        name="engine-supervisor-preload",
        daemon=True,
    ).start()

    # Bridge worker-side provider attach/detach onto the GUI thread.  Queued so
    # they run on the GUI thread even when emitted from a worker/pump thread.
    # NOTE: providerAttachRequested carries (camera_id, shm_name, w, h); the
    # frame source's attach takes (camera_id, shm_name, height, width).
    client.providerAttachRequested.connect(
        lambda cid, shm, w, h: frames.attach(cid, shm, h, w),
        Qt.ConnectionType.QueuedConnection,
    )
    client.providerDetachRequested.connect(
        frames.detach,
        Qt.ConnectionType.QueuedConnection,
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

    # Reap spawned child processes on signal termination: route SIGINT/SIGTERM through
    # a clean Qt quit so the post-exec teardown (stopEngine → supervisor.stop) runs
    # instead of orphaning the model-server / per-camera workers. The 30 Hz pump above
    # keeps the loop returning to Python so the handler fires promptly.
    _install_signal_shutdown(app)

    # ── theme + window ─────────────────────────────────────────────────────────
    theme = ThemeController(app, client)
    window = _build_main_window(client, log_model=log_model, frames=frames, theme=theme)
    window.show()

    def _present_window() -> None:
        """Best-effort macOS/Qt nudge so the shell is drawable before engine work."""
        try:
            if window.isMinimized():
                window.showNormal()
            else:
                window.show()
            window.raise_()
            window.activateWindow()
        except Exception:  # noqa: BLE001
            log.debug("Could not present main window", exc_info=True)

    _present_window()
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)

    log.info("AutoPTZ UI started")

    # ── dev hook: open AutoPTZ Mark on startup ─────────────────────────────────
    # The macOS menu bar is intercepted by some menu-bar managers, so the Help →
    # Run AutoPTZ Mark… item can be unreachable to automated UI drivers. When
    # AUTOPTZ_START_MARK is set (or /tmp/autoptz_start_mark exists) open the Mark
    # pre-flight automatically so the demo can be launched headlessly/scripted.
    # Inert in normal use.
    if os.path.exists("/tmp/autoptz_start_mark_now"):
        # Skip the pre-flight and enter Mark directly with a small fast session
        # (for scripted/visual validation of the Mark window itself).  The marker
        # file's first line picks the source ("clip"|"synthetic"|"ndi"; blank →
        # the MarkSession default) so a script can validate each source variant.
        from autoptz.ui.mark_session import MarkSession

        def _marker_lines() -> tuple[str, str]:
            # Line 1 → source ("clip"|"synthetic"|"ndi"; blank → default).
            # Line 2 (optional) → a CLIP_LIBRARY clip id (for clip source) so a
            # script can validate a specific bundled scene directly.
            try:
                with open("/tmp/autoptz_start_mark_now", encoding="utf-8") as fh:
                    first = fh.readline().strip().lower()
                    second = fh.readline().strip()
            except OSError:
                first, second = "", ""
            src = first if first in {"clip", "synthetic", "ndi"} else MarkSession().source
            return src, second

        _src, _clip = _marker_lines()
        QTimer.singleShot(
            1200,
            lambda: window._enter_mark_mode(
                MarkSession(source=_src, clip_id=_clip, max_cameras=4, dwell_s=6.0)
            ),
        )
    elif os.environ.get("AUTOPTZ_START_MARK") or os.path.exists("/tmp/autoptz_start_mark"):
        QTimer.singleShot(1200, window._start_mark)

    # ── engine auto-start ──────────────────────────────────────────────────────
    # Restore the last on/off state (default ON) and start after the window is
    # shown and exposed so the first paint happens before heavy ingest/ML work.
    if bool(store.get_setting("engine_running", True)):

        class _CameraAccessBridge(QObject):
            resolved = Signal(bool)

            @Slot(bool)
            def resolve(self, granted: bool) -> None:
                self.resolved.emit(bool(granted))

        camera_access = _CameraAccessBridge(app)

        def _auto_start_when_engine_ready() -> None:
            with supervisor_lock:
                ready = "cls" in supervisor_holder
                error = str(supervisor_holder.get("error", "") or "")
            if error:
                client.errorOccurred.emit(f"Engine failed to load: {error}")
                return
            if not ready:
                window.statusBar().showMessage("Preparing engine...", 1000)
                QTimer.singleShot(100, _auto_start_when_engine_ready)
                return
            _start_engine_after_macos_camera_preflight(client, camera_access)

        QTimer.singleShot(50, _present_window)
        QTimer.singleShot(750, _auto_start_when_engine_ready)

    if has_qapplication_window_api:
        exit_code = app.exec()
    else:
        # Offscreen tests can run with a QCoreApplication already installed.
        # Route through QApplication.exec so their QApplication-level stub still
        # prevents entering a real event loop.
        exit_code = QApplication.exec(app)

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
