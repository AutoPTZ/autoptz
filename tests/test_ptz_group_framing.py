"""Physical PTZ honors group framing (parity with Center Stage): with group
framing on and nobody locked, the camera aims at the confident group instead of
idling. With group framing off (default), the single-target path is unchanged."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def _worker(*, group_framing: bool):
    from autoptz.config.models import CameraConfig, TrackingConfig
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(
        id="cam-grp",
        name="Grp",
        tracking=TrackingConfig(group_framing=group_framing),
    )
    return CameraWorker("cam-grp", cfg, on_telemetry=lambda m: None)


class _Ctrl:
    def set_loop_latency(self, _s: float) -> None: ...
    def last_cmd_send_ms(self) -> float:
        return 0.0


def _track(track_id, box):
    x1, y1, x2, y2 = box
    return SimpleNamespace(
        track_id=track_id,
        lost=False,
        vx=0.0,
        vy=0.0,
        bbox=SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def _drive(worker, tracks):
    published: list[dict] = []
    worker._ptz = _Ctrl()
    worker._tracking_enabled = True
    worker._publish_ptz = lambda ctrl, err, vel, height, *, track_active, now, log_label: (
        published.append({"err": err, "height": height, "active": track_active, "label": log_label})
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    worker._drive_ptz_auto(tracks, frame, now=1.0)
    return published


def test_group_framing_on_no_lock_drives_ptz_at_group() -> None:
    w = _worker(group_framing=True)
    tracks = [_track(1, (0, 40, 20, 90)), _track(2, (80, 40, 100, 90))]
    published = _drive(w, tracks)
    assert published, "controller was never stepped"
    last = published[-1]
    assert last["active"] is True
    assert last["label"] == "group"
    # Union spans both people symmetrically around center → ~zero horizontal error.
    assert abs(last["err"][0]) < 0.05


def test_group_framing_off_no_lock_idles() -> None:
    """Default behaviour (group framing off): no lock → controller idles."""
    w = _worker(group_framing=False)
    tracks = [_track(1, (0, 40, 20, 90)), _track(2, (80, 40, 100, 90))]
    published = _drive(w, tracks)
    assert published
    last = published[-1]
    assert last["active"] is False
    assert last["label"] is None


def test_group_framing_off_no_people_idles() -> None:
    w = _worker(group_framing=True)
    published = _drive(w, [])
    assert published
    assert published[-1]["active"] is False
