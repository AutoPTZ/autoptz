from __future__ import annotations

import numpy as np

from autoptz.engine.pipeline.vcam import VirtualCamSink


def test_sink_is_noop_without_pyvirtualcam(monkeypatch):
    import autoptz.engine.pipeline.vcam as v

    monkeypatch.setattr(v, "_probe_pyvirtualcam", lambda: False)
    sink = VirtualCamSink(640, 480)
    assert sink.available is False
    sink.send_bgr(np.zeros((480, 640, 3), dtype=np.uint8))  # must not raise
    sink.close()
