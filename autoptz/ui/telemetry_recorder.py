"""Opt-in JSONL telemetry recorder for field diagnosis.

Set ``AUTOPTZ_TELEMETRY_LOG=/path/to/file.jsonl`` and every ``TelemetryMsg`` the
GUI receives is appended as one JSON line (plus ``recv_ts``, the GUI receive
time). Off by default; costs nothing when unset. Lets tracking quality (ID
churn, aim jitter, PTZ command oscillation, latency) be measured from a real
session instead of eyeballed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import IO, Any

log = logging.getLogger(__name__)

ENV_VAR = "AUTOPTZ_TELEMETRY_LOG"


class TelemetryRecorder:
    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._fh: IO[str] | None = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self.path = path

    def __call__(self, msg: Any) -> None:
        try:
            line = msg.model_dump(mode="json")
            line["recv_ts"] = time.time()
            text = json.dumps(line, separators=(",", ":"))
            with self._lock:
                if self._fh is None:
                    return
                self._fh.write(text + "\n")
        except Exception:  # noqa: BLE001 — diagnostics must never break telemetry
            log.debug("telemetry record failed", exc_info=True)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def install_from_env(client: Any) -> TelemetryRecorder | None:
    path = os.environ.get(ENV_VAR, "").strip()
    if not path:
        return None
    try:
        rec = TelemetryRecorder(path)
    except OSError:
        log.warning("%s=%s could not be opened; telemetry recording off", ENV_VAR, path)
        return None
    client.add_telemetry_observer(rec)
    log.info("Recording telemetry to %s", path)
    return rec
