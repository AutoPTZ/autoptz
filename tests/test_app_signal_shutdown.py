"""SIGTERM/SIGINT must route through a clean Qt quit.

Qt's event loop blocks in native code, so a signal would otherwise terminate the
process WITHOUT running the orderly shutdown after ``app.exec()`` — leaving
model-server camera child processes (spawned under AUTOPTZ_MODEL_SERVER)
orphaned, holding RAM/accelerator. Routing the signal
through ``app.quit()`` lets ``app.exec()`` return so ``client.stopEngine()`` →
``supervisor.stop()`` terminates the children.
"""

from __future__ import annotations

import signal

from autoptz.ui.app import _install_signal_shutdown


class _FakeApp:
    def __init__(self) -> None:
        self.quit_count = 0

    def quit(self) -> None:
        self.quit_count += 1


def test_sigterm_and_sigint_handlers_request_app_quit() -> None:
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    app = _FakeApp()
    try:
        # hard_exit_delay=None disables the os._exit fallback so the test never kills
        # the pytest process.
        _install_signal_shutdown(app, hard_exit_delay=None)
        term = signal.getsignal(signal.SIGTERM)
        sigint = signal.getsignal(signal.SIGINT)
        assert callable(term), "SIGTERM handler not installed"
        assert callable(sigint), "SIGINT handler not installed"
        # Invoking either handler must ask Qt to quit so app.exec() returns and the
        # post-exec teardown (stopEngine → supervisor.stop) reaps the child processes.
        term(signal.SIGTERM, None)
        sigint(signal.SIGINT, None)
        assert app.quit_count == 2
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
