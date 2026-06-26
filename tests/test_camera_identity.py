"""Tests for USB/network camera identity matching (the wrong-camera bug).

USB cameras must be matched by stable unique_id, never by the volatile
usb://<index> address — otherwise toggling/deleting one camera after the USB
enumeration order shifts could hit the wrong camera.
"""

from __future__ import annotations

from autoptz.ui.widgets.main_window import _match_camera_id


def test_usb_matches_by_unique_id_not_address() -> None:
    # Two USB cameras whose addresses have SHIFTED so cam B now sits at usb://0
    # (the address cam A was originally added under).
    cameras = [
        ("camA", "usb", "uid-A", "usb://1"),
        ("camB", "usb", "uid-B", "usb://0"),
    ]
    # Removing cam A (its real unique_id) must resolve to camA, never camB,
    # even though camB's address now collides with cam A's old usb://0 uri.
    assert _match_camera_id(cameras, "usb://0", "uid-A") == "camA"


def test_usb_does_not_fall_back_to_address_when_id_known_elsewhere() -> None:
    # A scanned device with no unique_id must NOT match a USB camera that has a
    # *different* known id at that address (the wrong-camera-delete scenario).
    cameras = [("camB", "usb", "uid-B", "usb://0")]
    assert _match_camera_id(cameras, "usb://0", "") is None


def test_usb_idless_matches_idless_at_same_uri() -> None:
    # When the device genuinely has no unique_id, an id-less USB camera at the
    # same uri is a safe match (no ambiguity with id-bearing cameras).
    cameras = [("camX", "usb", "", "usb://2")]
    assert _match_camera_id(cameras, "usb://2", "") == "camX"


def test_network_matches_by_address() -> None:
    cameras = [
        ("ndi1", "ndi", "", "ndi://STUDIO (CAM 1)"),
        ("rtsp1", "rtsp", "", "rtsp://10.0.0.5/stream"),
    ]
    assert _match_camera_id(cameras, "ndi://STUDIO (CAM 1)", "") == "ndi1"
    assert _match_camera_id(cameras, "rtsp://10.0.0.5/stream", "") == "rtsp1"


def test_no_match_returns_none() -> None:
    cameras = [("camA", "usb", "uid-A", "usb://0")]
    assert _match_camera_id(cameras, "usb://9", "uid-Z") is None


def test_unique_id_wins_over_address_collision() -> None:
    # unique_id match takes priority even if another camera shares the address.
    cameras = [
        ("camA", "usb", "uid-A", "usb://0"),
        ("camB", "usb", "uid-B", "usb://0"),  # stale collision
    ]
    assert _match_camera_id(cameras, "usb://0", "uid-B") == "camB"
