"""Fixes from live 5-camera desktop testing (offscreen):

- the tile's cryptic ``×N`` degradation chip is gone (info moved to the "?"
  per-stage tooltip); the state chip never paints over the target label;
- the Engine EP label aggregates mixed worker EPs instead of flapping;
- this machine's own AutoPTZ NDI outputs are filtered from the NDI menu
  (feedback loop);
- the Properties tracking captions never show/hide (no layout jumping).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def qtapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


# ── tile HUD ─────────────────────────────────────────────────────────────────


def _tile(qtapp):
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.widgets.camera_tile import CameraTile

    client = EngineClient()
    cid = client.addCamera("usb://0", "Cam")
    return CameraTile(cid, client, frame_source=None)


def test_degradation_chip_paint_removed(qtapp) -> None:
    """The bare ×N tile chip is gone — its info lives in the '?' tooltip now."""
    from autoptz.ui.widgets.camera_tile import CameraTile

    assert not hasattr(CameraTile, "_paint_degradation_chip")


def test_perf_tooltip_carries_cadence_when_degraded(qtapp) -> None:
    tile = _tile(qtapp)
    rec = SimpleNamespace(
        fps=20.0,
        telemetry=SimpleNamespace(
            ingest_ms=5.0,
            detect_ms=20.0,
            face_ms=0.0,
            width=1280,
            height=720,
            quality_state=None,
        ),
        camera_config=None,
        quality_state_as_dict=lambda: {
            "configured_interval": 1,
            "detect_interval": 4,
            "reason": "over frame budget",
        },
        tracking_status_as_dict=lambda: {},
    )
    tip = tile._compose_perf_tooltip(rec)
    assert "×4" in tip  # cadence surfaced with room for words
    tile.deleteLater()


def test_target_label_sets_flag_and_state_chip_skips(qtapp) -> None:
    """_draw_target_label marks the frame; the paint loop then skips the state
    chip — the two never overlap at top-left."""
    from PySide6.QtGui import QColor, QPainter, QPixmap

    tile = _tile(qtapp)
    tile.resize(320, 180)
    assert tile._target_label_drawn is False
    pm = QPixmap(320, 180)
    p = QPainter(pm)
    try:
        tile._draw_target_label(p, "Tracking: prince  92%", QColor("red"))
    finally:
        p.end()
    assert tile._target_label_drawn is True
    tile.deleteLater()


# ── engine EP label aggregation ──────────────────────────────────────────────


def test_engine_ep_label_stable_with_mixed_workers(qtapp) -> None:
    """Mixed worker modes (model-server + threaded CoreML) must compose ONE
    stable label, not flap between the two on every telemetry message."""
    from autoptz.engine.runtime.messages import TelemetryMsg
    from autoptz.ui.engine_client import EngineClient

    c = EngineClient()
    a = c.addCamera("usb://0", "A")
    b = c.addCamera("usb://1", "B")
    changes: list[str] = []
    c.engineStateChanged.connect(lambda: changes.append(c.engineEp))

    c._on_telemetry_main(TelemetryMsg(camera_id=a, seq=1, ep="CoreMLExecutionProvider"))
    assert c.engineEp == "CoreML"
    c._on_telemetry_main(TelemetryMsg(camera_id=b, seq=1, ep="model-server"))
    assert c.engineEp == "CoreML + model-server"
    # More telemetry from either camera must NOT change the label again.
    for seq in range(2, 6):
        c._on_telemetry_main(TelemetryMsg(camera_id=a, seq=seq, ep="CoreMLExecutionProvider"))
        c._on_telemetry_main(TelemetryMsg(camera_id=b, seq=seq, ep="model-server"))
    assert c.engineEp == "CoreML + model-server"
    assert changes == ["CoreML", "CoreML + model-server"]  # exactly two updates


def test_engine_ep_label_single_mode_unchanged(qtapp) -> None:
    from autoptz.engine.runtime.messages import TelemetryMsg
    from autoptz.ui.engine_client import EngineClient

    c = EngineClient()
    a = c.addCamera("usb://0", "A")
    c._on_telemetry_main(TelemetryMsg(camera_id=a, seq=1, ep="CoreMLExecutionProvider"))
    assert c.engineEp == "CoreML"


# ── NDI own-output filtering ─────────────────────────────────────────────────


def test_is_own_ndi_output_matches_local_autoptz_feeds() -> None:
    from autoptz.ui.widgets.main_window import _is_own_ndi_output

    host = "PRINCES-MBP"
    assert _is_own_ndi_output("PRINCES-MBP (AutoPTZ MacBook Pro Camera)", host)
    # Nested loopback of a loopback is still ours.
    assert _is_own_ndi_output(
        "PRINCES-MBP (AutoPTZ PRINCES-MBP (AutoPTZ MacBook Pro Camera))", host
    )
    assert _is_own_ndi_output("princes-mbp (AutoPTZ Cam)", host)  # case-insensitive


def test_is_own_ndi_output_keeps_legitimate_sources() -> None:
    from autoptz.ui.widgets.main_window import _is_own_ndi_output

    host = "PRINCES-MBP"
    # Another machine's AutoPTZ output is a legitimate remote source.
    assert not _is_own_ndi_output("STUDIO-PC (AutoPTZ Stage Cam)", host)
    # Non-AutoPTZ senders on this machine stay listed (OBS, Test Patterns, …).
    assert not _is_own_ndi_output("PRINCES-MBP (OBS)", host)
    assert not _is_own_ndi_output("PRINCES-MBP (Test Pattern)", host)
    assert not _is_own_ndi_output("", host)
    assert not _is_own_ndi_output("no-parens-name", host)


# ── properties captions never move the layout ────────────────────────────────


def test_set_caption_reserves_space_and_elides(qtapp) -> None:
    from PySide6.QtWidgets import QLabel

    from autoptz.ui.widgets.properties_panel import PropertiesPanel

    label = QLabel(" ")
    label.resize(120, 16)
    PropertiesPanel._set_caption(label, "")
    assert label.text() == " "  # keeps its line height — no layout jump
    PropertiesPanel._set_caption(
        label, "Degraded ×4: Auto quality: over frame budget; detector cadence relaxed"
    )
    assert label.text().strip()
    assert "Degraded" in label.toolTip()  # full text on hover
    label.deleteLater()


# ── log spam: change/transition-based, never steady-state repeats ────────────


def _bare_worker():
    from autoptz.config.models import CameraConfig
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(id="cam-log", name="LogCam")
    return CameraWorker("cam-log", cfg, on_telemetry=lambda m: None)


def test_center_stage_log_fires_on_change_not_repeat(qtapp, caplog) -> None:
    import logging

    import numpy as np

    from autoptz.engine.ptz.digital import DigitalPTZBackend

    w = _bare_worker()
    w._ptz_backend = DigitalPTZBackend()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    with caplog.at_level(logging.INFO, logger="autoptz.engine.camera_worker"):
        w._framed_output(frame)  # first state (no target) → one line
        w._framed_output(frame)  # unchanged → silent
        w._framed_output(frame)  # unchanged → silent
    lines = [r for r in caplog.records if "center-stage:" in r.message]
    assert len(lines) == 1


def test_inference_behind_logs_transitions_only(qtapp, caplog) -> None:
    import logging

    w = _bare_worker()

    def window(captured_delta: int, inferred_delta: int) -> None:
        w._frames_captured += captured_delta
        w._frames_inferred += inferred_delta
        w._maybe_log_drops(w._next_drop_log_t + 1.0)  # force the window

    with caplog.at_level(logging.INFO, logger="autoptz.engine.camera_worker"):
        window(100, 15)  # enters behind → INFO
        window(100, 15)  # still behind → DEBUG only
        window(100, 15)  # still behind → DEBUG only
        window(100, 95)  # recovers → INFO "caught up"
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    behind_lines = [r for r in infos if "inference behind" in r.message]
    caught_up = [r for r in infos if "caught up" in r.message]
    assert len(behind_lines) == 1  # only the transition, not every window
    assert len(caught_up) == 1


def test_face_pass_skipped_when_no_tracks(qtapp) -> None:
    """An empty scene must not pay the SCRFD full-frame scan."""
    import numpy as np

    w = _bare_worker()
    calls: list[int] = []
    w._face = SimpleNamespace(
        recognizer=SimpleNamespace(available=True, detect=lambda f: calls.append(1) or []),
        service=None,
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    w._maybe_identify(frame, [], now=100.0)  # no tracks → no detect
    assert calls == []
    # And the timer was not stamped: a person appearing runs the pass at once.
    track = SimpleNamespace(
        track_id=1, lost=False, bbox=SimpleNamespace(x1=0, y1=0, x2=100, y2=200)
    )
    w._maybe_identify(frame, [track], now=100.05)
    assert calls == [1]


# ── output pump: sinks never run on the capture thread ──────────────────────


def test_output_sender_delivers_on_its_own_thread() -> None:
    import threading

    import numpy as np

    from autoptz.engine.pipeline.output_sender import OutputSender

    seen: list[str] = []
    done = threading.Event()

    class _Sink:
        def send_bgr(self, frame):
            seen.append(threading.current_thread().name)
            done.set()

    sender = OutputSender(name="t1")
    try:
        sender.submit(np.zeros((4, 4, 3), dtype=np.uint8), [_Sink()])
        assert done.wait(2.0)
        assert seen and "output-sender" in seen[0]  # NOT the caller's thread
    finally:
        sender.close()


def test_output_sender_drops_oldest_when_busy() -> None:
    import threading
    import time

    import numpy as np

    from autoptz.engine.pipeline.output_sender import OutputSender

    delivered: list[int] = []
    release = threading.Event()
    first_started = threading.Event()

    class _SlowSink:
        def send_bgr(self, frame):
            first_started.set()
            release.wait(2.0)  # hold the pump busy
            delivered.append(int(frame[0, 0, 0]))

    sender = OutputSender(name="t2")
    try:
        mk = lambda v: np.full((2, 2, 3), v, dtype=np.uint8)  # noqa: E731
        sink = _SlowSink()
        sender.submit(mk(1), [sink])
        assert first_started.wait(2.0)
        # While busy, park two more — only the NEWEST must survive.
        sender.submit(mk(2), [sink])
        sender.submit(mk(3), [sink])
        release.set()
        deadline = time.monotonic() + 2.0
        while len(delivered) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert delivered == [1, 3]  # frame 2 was replaced, never sent
    finally:
        sender.close()


def test_output_sender_close_is_idempotent_and_fast() -> None:
    from autoptz.engine.pipeline.output_sender import OutputSender

    sender = OutputSender(name="t3")
    sender.close()
    sender.close()  # second close must be a no-op


def test_is_own_autoptz_output_shared_helper() -> None:
    from autoptz.engine.discovery.ndi import is_own_autoptz_output

    assert is_own_autoptz_output("HOSTY (AutoPTZ Cam)", "HOSTY")
    assert not is_own_autoptz_output("OTHER (AutoPTZ Cam)", "HOSTY")
    assert not is_own_autoptz_output("HOSTY (OBS)", "HOSTY")


# ── layout audit regressions: values elide late-settling widths; floors real ─


def test_services_panel_floor_covers_trailing_pills() -> None:
    """At 260 the body overflowed the viewport by 22px (scrollbar + margins),
    clipping the ON/OK pills and Restart/Enable-all — the floor is 300 now."""
    from autoptz.ui.widgets.services_panel import ServicesPanel

    assert ServicesPanel.minimumSizeHint(ServicesPanel.__new__(ServicesPanel)).width() >= 300


def test_properties_address_elides_after_layout_settles(qtapp, tmp_path) -> None:
    """The Address used to be elided against a stale (pre-layout) width, leaving
    the full text hard-clipped in a narrow label. After set_camera + one event
    loop turn, a long address must be elided with the full value on the tooltip."""
    from pathlib import Path

    from autoptz.config.store import ConfigStore
    from autoptz.ui.engine_client import EngineClient
    from autoptz.ui.frames import ShmFrameSource
    from autoptz.ui.widgets.properties_panel import PropertiesPanel

    client = EngineClient(store=ConfigStore(db_path=Path(tmp_path) / "cfg.db", debounce_s=0))
    long_addr = "ndi://PRINCES-MBP (AutoPTZ PRINCES-MBP (AutoPTZ MacBook Pro Camera))"
    cid = client.addCamera(long_addr, "Loopy")
    panel = PropertiesPanel(client, frame_source=ShmFrameSource())
    try:
        panel.resize(300, 760)
        panel.show()
        qtapp.processEvents()
        panel.set_camera(cid)
        qtapp.processEvents()  # deferred re-elide fires here
        qtapp.processEvents()
        label = panel._address
        assert long_addr.startswith(label.toolTip()[:20])  # full value on tooltip
        shown = label.text()
        assert "…" in shown  # elided, not hard-clipped
        assert label.fontMetrics().horizontalAdvance(shown) <= label.width() + 2
    finally:
        panel.deleteLater()
