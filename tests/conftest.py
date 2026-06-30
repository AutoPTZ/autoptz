"""Shared test fixtures and a Linux-safe shutdown guard.

Two things live here so every test file can rely on them:

``qapp``
    One process-wide Qt application object. Tests are headless, but several suites
    construct real widgets, so the shared instance must be a ``QApplication``. Qt
    allows only one application per process, so the fixture is session-scoped and
    reuses any existing instance.

``pytest_sessionfinish`` guard
    On Linux and macOS, native extensions (PySide6/Qt, AVFoundation, OpenCV, ONNX
    Runtime, torch) can corrupt the heap while their global destructors run at
    interpreter shutdown — the process aborts *after* every test has already passed,
    turning a green run red on CI. A conftest hook wrapper runs outside pytest's own,
    i.e. after the terminal summary is printed, so we flush output and hard-exit with
    pytest's own status. The result stays authoritative and the flaky native teardown
    never runs.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_DARWIN_DRAIN_QT_TEARDOWN_FOR_MARK = {
    "test_mark_window.py",
}

_DARWIN_QT_WIDGET_FIRST = {
    "test_about_mark.py",
    "test_app_mark_routing.py",
    "test_camera_context_menu.py",
    "test_camera_info_telemetry.py",
    "test_experimental_dialog.py",
    "test_logs_panel.py",
    "test_main_window_startup.py",
    "test_mark_completion_dialog.py",
    "test_mark_exit_dialog.py",
    "test_mark_engine.py",
    "test_mark_lifecycle.py",
    "test_mark_panels.py",
    "test_mark_preflight.py",
    "test_mark_smoke.py",
    "test_mark_window.py",
    "test_theme.py",
    "test_ui.py",
}

if TYPE_CHECKING:
    from pytest import ExitCode, Session


@pytest.fixture(autouse=True)
def _no_serial_autoprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never open real serial ports during tests.

    A USB camera's ``auto`` PTZ backend scans serial ports for a VISCA control
    port at worker startup; on CI that opens host serial devices and stalls the
    worker thread (breaking timing-sensitive tests). Disable it globally; the
    tests that exercise the probe path re-enable it explicitly.
    """
    monkeypatch.setenv("AUTOPTZ_PTZ_SERIAL_AUTOPROBE", "0")


@pytest.fixture(scope="session")
def qapp() -> Generator[object]:
    """One process-wide QApplication object for headless tests."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


@pytest.fixture
def wait_until() -> Callable[..., object]:
    """Poll a predicate until it becomes truthy, with an actionable timeout error."""

    def _wait_until(
        predicate: Callable[[], object],
        *,
        timeout: float = 2.0,
        interval: float = 0.01,
        message: str = "condition was not met before timeout",
        pump_qt: bool = False,
    ) -> object:
        deadline = time.monotonic() + timeout
        last: object = None
        last_assertion: AssertionError | None = None
        while True:
            if pump_qt:
                try:
                    from PySide6.QtWidgets import QApplication

                    app = QApplication.instance()
                    if app is not None:
                        app.processEvents()
                except OSError:
                    if sys.platform != "win32":
                        raise
                except Exception:
                    pass
            try:
                last = predicate()
                last_assertion = None
            except AssertionError as exc:
                last = False
                last_assertion = exc
            try:
                ok = bool(last)
            except ValueError:
                ok = last is not None
            if ok:
                return last
            if time.monotonic() >= deadline:
                detail = f"{message} after {timeout:.2f}s"
                if last_assertion is not None:
                    detail = f"{detail}: {last_assertion}"
                else:
                    detail = f"{detail}; last value={last!r}"
                pytest.fail(detail)
            time.sleep(interval)

    return _wait_until


def _drain_qt_events(app: object) -> None:
    """Best-effort event drain for widget teardown."""
    try:
        app.processEvents()
    except OSError:
        if sys.platform == "win32":
            return
        raise


@pytest.fixture(autouse=True)
def _qt_widget_cleanup(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Close leaked top-level widgets and flush deferred deletes after each test.

    In-process Qt suites should be independent from collection order. Leaving a
    MarkWindow, dialog, or queued DeferredDelete behind makes later tests depend on
    native Qt teardown timing, which is the source of the mixed-suite flake.
    """
    yield
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
    except Exception:  # noqa: BLE001
        return
    app = QApplication.instance()
    if app is None:
        return
    widgets = list(QApplication.topLevelWidgets())
    if not widgets:
        return
    for widget in widgets:
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            pass
    filename = request.node.path.name
    if (
        sys.platform == "darwin"
        and filename.startswith("test_mark_")
        and filename not in _DARWIN_DRAIN_QT_TEARDOWN_FOR_MARK
    ):
        return
    _drain_qt_events(app)
    try:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    except RuntimeError:
        pass
    _drain_qt_events(app)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Run in-process Qt widget suites before macOS media/native tests.

    AVFoundation/AppKit, torch, and OpenCV can leave native state that makes
    PySide's offscreen ``processEvents`` crash during later widget teardown on
    macOS. Widget suites are deterministic when they run first, so keep their
    relative order but move them ahead of the non-Qt engine suites.
    """
    if not sys.platform == "darwin":
        return
    original_index = {id(item): idx for idx, item in enumerate(items)}

    def _key(item: pytest.Item) -> tuple[int, int]:
        filename = item.path.name
        priority = 0 if filename in _DARWIN_QT_WIDGET_FIRST else 1
        return (priority, original_index[id(item)])

    items.sort(key=_key)


@pytest.hookimpl(wrapper=True)
def pytest_sessionfinish(
    session: Session, exitstatus: int | ExitCode
) -> Generator[None, None, None]:
    """Hard-exit before native teardown can abort the process."""
    result = yield
    if int(exitstatus) == 0 and sys.platform.startswith(("linux", "darwin")):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(exitstatus))
    return result
