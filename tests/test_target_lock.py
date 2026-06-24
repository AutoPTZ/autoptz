"""Target-lock correctness tests.

Bug D — a target selected by *identity* (name) must never be re-bound to a
different person by the appearance-only ReID recovery path. ReID may hold/refresh
the current track, but re-acquiring a *named* target requires a face-identity
match, not body appearance — otherwise the camera drifts onto the next
visually-similar person, then the next unknown track.
"""

from __future__ import annotations

from autoptz.config.models import CameraConfig, SourceConfig, TrackingConfig


def _worker(identity_id: str | None = None):
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(
        id="cam-lock-00000001",
        name="t",
        source=SourceConfig(type="usb", address="usb://0"),
        tracking=TrackingConfig(tracking_mode="stable"),
    )
    w = CameraWorker("cam-lock-00000001", cfg, on_telemetry=lambda m: None)
    w._target_identity_id = identity_id
    return w


class TestReidRebindIdentityGate:
    def test_named_target_allows_rebind_only_to_same_identity(self):
        w = _worker(identity_id="A")
        w._track_identity = {5: ("A", "Alice", 0.9), 7: ("B", "Bob", 0.8)}
        assert w._reid_rebind_allows(5) is True  # same identity → allowed
        assert w._reid_rebind_allows(7) is False  # different known person → blocked

    def test_named_target_blocks_rebind_to_unknown_track(self):
        w = _worker(identity_id="A")
        w._track_identity = {}
        # Unknown track (no confirmed identity) → blocked; wait for a face match.
        assert w._reid_rebind_allows(9) is False

    def test_manual_target_allows_appearance_rebind(self):
        w = _worker(identity_id=None)
        w._track_identity = {7: ("B", "Bob", 0.8)}
        # No identity target (clicked/manual): appearance re-bind is the intended behaviour.
        assert w._reid_rebind_allows(7) is True
        assert w._reid_rebind_allows(9) is True
