"""Optional go2rtc gateway — normalises RTSP/RTMP/USB streams.

go2rtc (https://github.com/AlexxIT/go2rtc) is a tiny, single-binary
media server that:
- Auto-reconnects to upstream RTSP/RTMP sources.
- Normalises quirky camera protocols behind a stable local RTSP URL.
- Re-exposes each stream at ``rtsp://localhost:{rtsp_port}/{name}``.

``Go2RTCGateway`` launches go2rtc as a managed subprocess.  It is entirely
optional — AutoPTZ v2 works without it; using it offloads reconnection and
protocol quirks from the engine and is especially useful when upstream
cameras have unreliable RTSP implementations.

Usage::

    gw = Go2RTCGateway()
    gw.add_stream("cam1", "rtsp://192.168.1.100/stream")
    gw.start()
    url = gw.local_url("cam1")   # "rtsp://localhost:8554/cam1"
    ...
    gw.stop()
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_RTSP_PORT = 8554
_DEFAULT_API_PORT = 1984
_HEALTH_TIMEOUT = 10.0  # seconds to wait for go2rtc to come up
_HEALTH_URL_FMT = "http://127.0.0.1:{port}/api"


@dataclass
class Go2RTCGateway:
    """Managed go2rtc subprocess with a stable local RTSP endpoint per stream.

    Args:
        binary:    Path to the go2rtc binary.  If ``None``, searched in PATH.
        rtsp_port: Local RTSP port (default 8554).
        api_port:  Local HTTP API port (default 1984).
    """

    binary: str | None = None
    rtsp_port: int = _DEFAULT_RTSP_PORT
    api_port: int = _DEFAULT_API_PORT

    _streams: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _config_file: Path | None = field(default=None, init=False, repr=False)

    # ── Stream management ─────────────────────────────────────────────────────

    def add_stream(self, name: str, source: str) -> None:
        """Register a stream source.  Must be called before ``start()``."""
        if self._process is not None:
            raise RuntimeError("Cannot add streams after start(); call stop() first.")
        self._streams[name] = source

    def local_url(self, name: str) -> str:
        """Return the stable local RTSP URL for a registered stream name."""
        return f"rtsp://127.0.0.1:{self.rtsp_port}/{name}"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Write the config and launch go2rtc."""
        binary = self._find_binary()
        if binary is None:
            raise FileNotFoundError(
                "go2rtc binary not found.  Download from https://github.com/AlexxIT/go2rtc/releases "
                "and place it in PATH (or set Go2RTCGateway.binary)."
            )

        config_path = self._write_config()
        self._config_file = config_path

        cmd = [binary, "-config", str(config_path)]
        log.info("Starting go2rtc: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        if not self._wait_healthy():
            self.stop()
            raise RuntimeError(f"go2rtc did not become healthy within {_HEALTH_TIMEOUT} s.")
        log.info(
            "go2rtc running (pid=%d, rtsp=:%d, api=:%d)",
            self._process.pid,
            self.rtsp_port,
            self.api_port,
        )

    def stop(self) -> None:
        """Terminate go2rtc and clean up the temporary config."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            log.info("go2rtc stopped")

        if self._config_file is not None and self._config_file.exists():
            self._config_file.unlink(missing_ok=True)
            self._config_file = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def __enter__(self) -> Go2RTCGateway:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _find_binary(self) -> str | None:
        if self.binary:
            return self.binary if os.path.isfile(self.binary) else None
        return shutil.which("go2rtc")

    def _write_config(self) -> Path:
        """Write a minimal go2rtc YAML config to a temp file."""
        lines = [
            f'api:\n  listen: ":{self.api_port}"\n',
            f'rtsp:\n  listen: ":{self.rtsp_port}"\n',
            "streams:\n",
        ]
        for name, source in self._streams.items():
            lines.append(f"  {name}: {source}\n")

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="autoptz_go2rtc_"
        )
        tmp.writelines(lines)
        tmp.close()
        return Path(tmp.name)

    def _wait_healthy(self) -> bool:
        url = _HEALTH_URL_FMT.format(port=self.api_port)
        deadline = time.monotonic() + _HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0):
                    return True
            except (urllib.error.URLError, OSError):
                time.sleep(0.3)
        return False
