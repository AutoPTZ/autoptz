from __future__ import annotations

import pytest

from autoptz.benchmark.ndi_sim import ndi_sim_available

_SKIP = pytest.mark.skipif(not ndi_sim_available(), reason="cyndilib missing")


def test_available_flag_is_bool() -> None:
    assert isinstance(ndi_sim_available(), bool)


@_SKIP
def test_names_are_unique_and_branded() -> None:
    from autoptz.benchmark.ndi_sim import MarkNDIFleet

    fleet = MarkNDIFleet(3, width=320, height=240, fps=30.0)
    names = fleet.names()
    assert names == ["AutoPTZ Mark Cam 1", "AutoPTZ Mark Cam 2", "AutoPTZ Mark Cam 3"]
    assert len(set(names)) == 3


@_SKIP
def test_open_push_close_cycle() -> None:
    from autoptz.benchmark.ndi_sim import MarkNDIFleet

    fleet = MarkNDIFleet(2, width=160, height=120, fps=30.0)
    fleet.open()
    try:
        for _ in range(5):
            fleet.pump_once()  # composes + write_video on each sender; must not raise
    finally:
        fleet.close()


@_SKIP
def test_sender_broadcasts_clip_variant_not_drawn_scene(tmp_path) -> None:
    """NDI must broadcast the SELECTED clip (real video), not the drawn 'anim' scene.

    The fix: the sender feeds the SyntheticAdapter the clip variant path the engine
    resolved for (clip_id, resolution, fps), so every NDI tile shows the same real
    footage as clip mode — never the procedural drawn people.
    """
    from autoptz.benchmark.ndi_sim import MarkNDISender

    clip = str(tmp_path / "variant_1280x720_30fps.mp4")
    sender = MarkNDISender(0, width=160, height=120, fps=30.0, frame_source=clip)
    try:
        # The underlying synthetic adapter is fed the clip path (loops real decode),
        # NOT the drawn-scene address "anim".
        assert sender._adapter._address == clip
        assert sender._adapter._address != "anim"
    finally:
        sender.close()


@_SKIP
def test_fleet_passes_clip_variant_to_every_sender(tmp_path) -> None:
    """A clip-variant frame source threads through to every sender in the fleet."""
    from autoptz.benchmark.ndi_sim import MarkNDIFleet

    clip = str(tmp_path / "variant.mp4")
    fleet = MarkNDIFleet(3, width=160, height=120, fps=30.0, frame_source=clip)
    try:
        for s in fleet._senders:
            assert s._adapter._address == clip
    finally:
        fleet.close()


def test_resolve_full_name_matches_hostname_prefixed() -> None:
    """NDI advertises 'HOST (short)' and the ingest matches the FULL name, so the
    Mark fleet maps each short sender name to its discovered full name (the fix for
    'NDI makes no real streams'). Pure helper — runs without cyndilib."""
    from autoptz.benchmark.ndi_sim import _resolve_full_name

    discovered = ["PRINCES-MBP (AutoPTZ Mark Cam 1)", "PRINCES-MBP (AutoPTZ Mark Cam 2)"]
    assert (
        _resolve_full_name("AutoPTZ Mark Cam 1", discovered) == "PRINCES-MBP (AutoPTZ Mark Cam 1)"
    )
    assert (
        _resolve_full_name("AutoPTZ Mark Cam 2", discovered) == "PRINCES-MBP (AutoPTZ Mark Cam 2)"
    )
    assert _resolve_full_name("X", ["X"]) == "X"  # exact (no prefix) still matches
    # Not yet discovered → None (non-strict callers may fall back; Mark uses strict preflight).
    assert _resolve_full_name("AutoPTZ Mark Cam 3", discovered) is None
