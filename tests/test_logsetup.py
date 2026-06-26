"""Tests for console colour logging (autoptz.logsetup)."""

from __future__ import annotations

import io
import logging

from autoptz.logsetup import (
    ColorFormatter,
    camera_ansi,
    install_console_logging,
)


def _record(msg: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord("autoptz.test", level, __file__, 1, msg, None, None)


class TestColorFormatter:
    def test_no_color_is_plain(self) -> None:
        fmt = ColorFormatter("%(levelname)s %(message)s", use_color=False)
        out = fmt.format(_record("camera_id=abc123 hello"))
        assert "\033[" not in out
        assert out == "INFO camera_id=abc123 hello"

    def test_color_wraps_level_and_camera(self) -> None:
        fmt = ColorFormatter("%(levelname)s %(message)s", use_color=True)
        out = fmt.format(_record("camera_id=deadbeef working", logging.WARNING))
        assert "\033[" in out  # has ANSI
        assert "\033[0m" in out  # has reset
        assert "deadbeef" in out  # id text preserved
        assert "camera_id=" in out

    def test_levelname_restored_after_format(self) -> None:
        fmt = ColorFormatter("%(levelname)s %(message)s", use_color=True)
        rec = _record("no camera here", logging.ERROR)
        fmt.format(rec)
        assert rec.levelname == "ERROR"  # not left colourised on the record

    def test_camera_ansi_stable_and_empty_safe(self) -> None:
        assert camera_ansi("") == ""
        a, b = camera_ansi("cam-1"), camera_ansi("cam-1")
        assert a == b and a.startswith("\033[38;5;")


class TestInstallConsoleLogging:
    def test_idempotent_single_handler(self) -> None:
        root = logging.getLogger()
        before = [h for h in root.handlers if getattr(h, "_autoptz_console", False)]
        for h in before:
            root.removeHandler(h)
        buf = io.StringIO()
        install_console_logging(logging.INFO, stream=buf)
        install_console_logging(logging.DEBUG, stream=buf)  # re-install
        ours = [h for h in root.handlers if getattr(h, "_autoptz_console", False)]
        assert len(ours) == 1  # not stacked
        assert root.level == logging.DEBUG
        # cleanup
        for h in ours:
            root.removeHandler(h)

    def test_non_tty_stream_has_no_color(self) -> None:
        buf = io.StringIO()  # not a tty
        install_console_logging(logging.INFO, stream=buf)
        logging.getLogger("autoptz.test").info("camera_id=xyz hi")
        assert "\033[" not in buf.getvalue()
        for h in list(logging.getLogger().handlers):
            if getattr(h, "_autoptz_console", False):
                logging.getLogger().removeHandler(h)
