"""Worker-owned frame-source reconnect (reopen a dead/stalled source).

The ingest ``SourceAdapter`` has a full open/stall/reconnect loop, but it only
runs on the adapter's OWN thread — which the camera worker never starts (the
worker drives ``_open``/``_read_frame`` directly so it can feed detection).  So
a source that failed to open at startup, or a session that stops delivering
frames (the classic Continuity-Camera "session runs, no frames ever arrive"),
stayed dead FOREVER: the capture loop only backed off and re-polled ``read()``
on the dead session.  The camera showed nothing until a full service restart.

These tests cover the worker-side reopen: rebuild + reopen after a sustained
stall (``reconnect.stall_timeout_s``), retry a failed open with exponential
backoff (``backoff_initial_s`` → ``backoff_max_s``), and never touch injected
(test/synthetic) sources.
"""

from __future__ import annotations

import numpy as np

from autoptz.config.models import CameraConfig, ReconnectConfig, SourceConfig


def _config(camera_id: str, **reconnect_kw) -> CameraConfig:
    return CameraConfig(
        id=camera_id,
        name="ReopenCam",
        source=SourceConfig(type="usb", address="usb://0"),
        reconnect=ReconnectConfig(**reconnect_kw) if reconnect_kw else ReconnectConfig(),
    )


class _FakeSource:
    def __init__(self, opens: bool = True) -> None:
        self.opens = opens
        self.open_calls = 0
        self.closed = False

    def open(self) -> bool:
        self.open_calls += 1
        return self.opens

    def read(self):  # noqa: ANN201
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _worker(config: CameraConfig):  # noqa: ANN201
    from autoptz.engine.camera_worker import CameraWorker

    return CameraWorker(config.id, config, lambda _m: None)


class TestMaybeReopenSource:
    def test_dead_source_is_rebuilt_after_backoff(self, monkeypatch) -> None:
        import autoptz.engine.camera_worker as cw

        built: list[_FakeSource] = []

        def _build(_cid, _cfg):  # noqa: ANN001, ANN202
            src = _FakeSource(opens=True)
            built.append(src)
            return src

        monkeypatch.setattr(cw, "build_frame_source", _build)
        w = _worker(_config("reopen-dead", backoff_initial_s=1.0, backoff_max_s=30.0))
        w._source = None  # open failed at startup
        w._last_frame_t = 100.0

        assert w._maybe_reopen_source(now=100.0) is True, "first attempt is immediate"
        assert w._source is built[0], "a fresh source must be built and opened"
        assert built[0].open_calls == 1

    def test_stalled_source_is_closed_and_replaced(self, monkeypatch) -> None:
        import autoptz.engine.camera_worker as cw

        built: list[_FakeSource] = []

        def _build(_cid, _cfg):  # noqa: ANN001, ANN202
            src = _FakeSource(opens=True)
            built.append(src)
            return src

        monkeypatch.setattr(cw, "build_frame_source", _build)
        w = _worker(_config("reopen-stall", stall_timeout_s=5.0))
        stalled = _FakeSource()
        w._source = stalled
        w._last_frame_t = 100.0

        # Frames still fresh → no reopen.
        assert w._maybe_reopen_source(now=104.0) is False
        assert w._source is stalled

        # Past the stall timeout → close the old session, open a fresh one.
        assert w._maybe_reopen_source(now=105.1) is True
        assert stalled.closed is True
        assert w._source is built[0]

    def test_failed_open_backs_off_exponentially_to_cap(self, monkeypatch) -> None:
        import autoptz.engine.camera_worker as cw

        attempts: list[float] = []

        def _build(_cid, _cfg):  # noqa: ANN001, ANN202
            return _FakeSource(opens=False)

        monkeypatch.setattr(cw, "build_frame_source", _build)
        w = _worker(_config("reopen-backoff", backoff_initial_s=1.0, backoff_max_s=4.0))
        w._source = None
        w._last_frame_t = 0.0

        now = 100.0
        # Walk time forward in small steps; record when attempts actually fire.
        while now < 120.0:
            if w._maybe_reopen_source(now=now):
                attempts.append(now)
            now += 0.25

        assert len(attempts) >= 4
        gaps = [round(b - a, 2) for a, b in zip(attempts, attempts[1:], strict=False)]
        assert gaps[0] == 1.0, "second attempt after backoff_initial_s"
        assert gaps[1] == 2.0, "backoff doubles"
        assert all(g <= 4.0 for g in gaps), "backoff never exceeds backoff_max_s"
        assert gaps[-1] == 4.0, "backoff settles at the cap"

    def test_success_resets_backoff_and_stall_window(self, monkeypatch) -> None:
        import autoptz.engine.camera_worker as cw

        failing = _FakeSource(opens=False)
        working = _FakeSource(opens=True)
        srcs = [failing, working]

        def _build(_cid, _cfg):  # noqa: ANN001, ANN202
            return srcs.pop(0)

        monkeypatch.setattr(cw, "build_frame_source", _build)
        w = _worker(_config("reopen-reset", backoff_initial_s=1.0, backoff_max_s=30.0))
        w._source = None
        w._last_frame_t = 0.0

        assert w._maybe_reopen_source(now=100.0) is True  # fails, backoff → 2s
        assert w._source is None
        assert w._maybe_reopen_source(now=101.0) is True  # succeeds
        assert w._source is working
        # A successful reopen restarts the stall window from "now" so the fresh
        # session gets a full stall_timeout before being torn down again.
        assert w._last_frame_t == 101.0

    def test_injected_source_is_never_rebuilt(self, monkeypatch) -> None:
        import autoptz.engine.camera_worker as cw

        def _build(_cid, _cfg):  # noqa: ANN001, ANN202
            raise AssertionError("injected sources must never be rebuilt")

        monkeypatch.setattr(cw, "build_frame_source", _build)
        from autoptz.engine.camera_worker import CameraWorker

        injected = _FakeSource()
        config = _config("reopen-injected", stall_timeout_s=1.0)
        w = CameraWorker(config.id, config, lambda _m: None, frame_source=injected)
        w._source = injected
        w._last_frame_t = 0.0

        assert w._maybe_reopen_source(now=500.0) is False
        assert w._source is injected


class TestCaptureLoopReopenWiring:
    """End-to-end: a worker whose source opens but never delivers frames must
    come back on its own once the (rebuilt) source starts delivering."""

    def test_worker_recovers_from_frameless_source(self, monkeypatch, wait_until) -> None:  # noqa: ANN001
        import autoptz.engine.camera_worker as cw

        class _FramelessSource(_FakeSource):
            def read(self):  # noqa: ANN201
                import time

                time.sleep(0.01)
                return None

        sources: list[_FakeSource] = []

        def _build(_cid, _cfg):  # noqa: ANN001, ANN202
            # First (startup) build delivers nothing; the reopened one delivers.
            src = _FramelessSource() if not sources else _FakeSource()
            sources.append(src)
            return src

        monkeypatch.setattr(cw, "build_frame_source", _build)
        config = _config(
            "reopen-e2e",
            stall_timeout_s=0.2,
            backoff_initial_s=0.05,
            backoff_max_s=0.2,
        )
        w = cw.CameraWorker(config.id, config, lambda _m: None)
        w.start()
        try:
            wait_until(
                lambda: len(sources) >= 2 or None,
                timeout=10.0,
                interval=0.05,
                message="worker never attempted to reopen the frameless source",
            )
            wait_until(
                lambda: (w._frames_captured > 0) or None,
                timeout=10.0,
                interval=0.05,
                message="worker never captured frames from the reopened source",
            )
            assert sources[0].closed is True, "the dead session must be closed"
        finally:
            w.stop()
