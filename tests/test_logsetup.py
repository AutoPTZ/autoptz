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


# ── third-party warning spam ──────────────────────────────────────────────────
#
# insightface's face_align triggers a skimage FutureWarning ("`estimate` is
# deprecated…") on EVERY alignment call attribution point; with 5 camera child
# processes the console drowns in it. Both the app console setup and the camera
# child logging setup must suppress FutureWarnings attributed to insightface —
# and ONLY insightface, so our own deprecation signals stay visible.


def _emit_insightface_future_warning() -> None:
    import warnings

    warnings.warn_explicit(
        "`estimate` is deprecated since version 0.26 and will be removed",
        FutureWarning,
        filename="/site-packages/insightface/utils/face_align.py",
        lineno=23,
        module="insightface.utils.face_align",
    )


def test_install_console_logging_suppresses_insightface_future_warnings() -> None:
    import warnings

    from autoptz.logsetup import install_console_logging

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        install_console_logging()
        _emit_insightface_future_warning()
        # Our own FutureWarnings must NOT be swallowed.
        warnings.warn_explicit(
            "autoptz thing deprecated",
            FutureWarning,
            filename="/autoptz/engine/foo.py",
            lineno=1,
            module="autoptz.engine.foo",
        )
    msgs = [str(w.message) for w in rec if issubclass(w.category, FutureWarning)]
    assert msgs == ["autoptz thing deprecated"]


def test_child_logging_suppresses_insightface_future_warnings() -> None:
    import warnings

    from autoptz.engine.process_worker import _configure_child_logging

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _configure_child_logging()
        _emit_insightface_future_warning()
    assert [w for w in rec if issubclass(w.category, FutureWarning)] == []
