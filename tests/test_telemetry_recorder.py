"""Opt-in JSONL telemetry recorder (AUTOPTZ_TELEMETRY_LOG) for field diagnosis."""

from __future__ import annotations

import json

from autoptz.engine.runtime.messages import TelemetryMsg


class _Client:
    def __init__(self) -> None:
        self.observers = []

    def add_telemetry_observer(self, fn) -> None:
        self.observers.append(fn)


def test_disabled_when_env_unset(monkeypatch) -> None:
    from autoptz.ui.telemetry_recorder import install_from_env

    monkeypatch.delenv("AUTOPTZ_TELEMETRY_LOG", raising=False)
    client = _Client()
    assert install_from_env(client) is None
    assert client.observers == []


def test_writes_one_json_line_per_message(monkeypatch, tmp_path) -> None:
    from autoptz.ui.telemetry_recorder import install_from_env

    path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("AUTOPTZ_TELEMETRY_LOG", str(path))
    client = _Client()
    rec = install_from_env(client)
    assert rec is not None and len(client.observers) == 1

    client.observers[0](TelemetryMsg(camera_id="cam-a", seq=1, fps=30.0))
    client.observers[0](TelemetryMsg(camera_id="cam-a", seq=2, fps=29.0))
    rec.close()

    lines = path.read_text().splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [1, 2]
    assert json.loads(lines[0])["camera_id"] == "cam-a"
    assert "recv_ts" in json.loads(lines[0])


def test_observer_never_raises_after_close(monkeypatch, tmp_path) -> None:
    from autoptz.ui.telemetry_recorder import install_from_env

    monkeypatch.setenv("AUTOPTZ_TELEMETRY_LOG", str(tmp_path / "t.jsonl"))
    client = _Client()
    rec = install_from_env(client)
    rec.close()
    client.observers[0](TelemetryMsg(camera_id="cam-a", seq=3))  # must not raise
