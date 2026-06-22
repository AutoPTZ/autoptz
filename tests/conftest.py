"""Shared test fixtures and a Linux-safe shutdown guard.

Two things live here so every test file can rely on them:

``qapp``
    One process-wide Qt application object. These tests are headless — no widgets
    are created in-process (the widget smoke tests in ``test_ui.py`` spawn their own
    offscreen subprocess), so a ``QCoreApplication`` is enough. Qt allows only one
    application per process, so the fixture is session-scoped and reuses any existing
    instance.

``pytest_sessionfinish`` guard
    On Linux, native extensions (PySide6/Qt, OpenCV, ONNX Runtime, torch) can corrupt
    the heap while their global destructors run at interpreter shutdown — the process
    aborts ("corrupted double-linked list", exit code 134) *after* every test has
    already passed, turning a green run red on CI. A conftest hook wrapper runs outside
    pytest's own, i.e. after the terminal summary is printed, so we flush output and
    hard-exit with pytest's own status. The result stays authoritative and the flaky
    native teardown never runs. Scoped to Linux so other platforms keep their normal
    shutdown (coverage/JUnit writers and the like).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest import ExitCode, Session


@pytest.fixture(scope="session")
def qapp() -> Generator[object]:
    """One process-wide Qt application object for headless tests."""
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    yield app


@pytest.hookimpl(wrapper=True)
def pytest_sessionfinish(
    session: Session, exitstatus: int | ExitCode
) -> Generator[None, None, None]:
    """Hard-exit on Linux before native teardown can abort the process."""
    result = yield
    if sys.platform.startswith("linux"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(exitstatus))
    return result
