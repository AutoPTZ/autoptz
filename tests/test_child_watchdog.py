"""Parent-death watchdog for spawned children.

daemon=True multiprocessing children are only reaped on a CLEAN parent exit
(multiprocessing's atexit). A parent killed by signal (SIGTERM/SIGKILL) or a crash
orphans the model-server / per-camera workers, leaking RAM + the accelerator. The
watchdog polls the parent pid and force-exits the child when the parent goes away, so
children never outlive the app no matter how it died.
"""

from __future__ import annotations

import os
import threading

from autoptz.engine.process_worker import (
    _install_parent_death_watchdog,
    _parent_is_gone,
)


def test_parent_is_gone_detects_reparenting() -> None:
    assert _parent_is_gone(os.getppid()) is False  # parent unchanged → still alive
    assert _parent_is_gone(-12345) is True  # original ppid no longer our parent → gone


def test_install_parent_death_watchdog_starts_daemon_thread() -> None:
    # Long poll so the watchdog never actually fires os._exit during the test.
    _install_parent_death_watchdog(poll_s=999.0)
    threads = [t for t in threading.enumerate() if t.name == "parent-death-watchdog"]
    assert threads, "watchdog thread not started"
    assert threads[0].daemon, "watchdog must be a daemon thread"
