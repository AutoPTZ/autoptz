"""Unit tests for autoptz.engine.discovery — USB, NDI, ONVIF.

All network/OS I/O is mocked so tests run without physical cameras,
NDI SDK, or ONVIF-capable devices.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest

from autoptz.engine.discovery.ndi import NDIDiscovery, NDISource
from autoptz.engine.discovery.onvif import ONVIFDevice, ONVIFDiscovery
from autoptz.engine.discovery.usb import USBDevice, USBDiscovery


# ── USBDiscovery ───────────────────────────────────────────────────────────────

class TestUSBDiscovery:
    """Tests using a mocked _probe_indices function."""

    def test_initial_devices_reported_as_added(self) -> None:
        events: list[tuple[str, USBDevice]] = []

        probe_results = [
            {0, 1},  # first poll: devices 0 and 1 present
            {0, 1},  # second poll: no change
        ]
        probe_iter = iter(probe_results)

        def mock_probe(max_index: int = 10) -> set[int]:
            try:
                return next(probe_iter)
            except StopIteration:
                return {0, 1}

        with patch("autoptz.engine.discovery.usb._probe_indices", mock_probe):
            discovery = USBDiscovery(poll_interval=0.1)
            discovery.on_change(lambda ev, dev: events.append((ev, dev)))
            discovery.start()
            time.sleep(0.3)
            discovery.stop()

        added_indices = {e[1].index for e in events if e[0] == "added"}
        assert 0 in added_indices
        assert 1 in added_indices
        assert all(e[0] == "added" for e in events)

    def test_device_added_after_initial_scan(self) -> None:
        events: list[tuple[str, USBDevice]] = []
        call_count = 0

        def mock_probe(max_index: int = 10) -> set[int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {0}   # initial: only device 0
            return {0, 2}    # subsequent: device 2 appeared

        with patch("autoptz.engine.discovery.usb._probe_indices", mock_probe):
            discovery = USBDiscovery(poll_interval=0.05)
            discovery.on_change(lambda ev, dev: events.append((ev, dev)))
            discovery.start()
            time.sleep(0.4)
            discovery.stop()

        added = [e for e in events if e[0] == "added"]
        added_indices = {e[1].index for e in added}
        assert 2 in added_indices

    def test_device_removed_fires_removed_event(self) -> None:
        events: list[tuple[str, USBDevice]] = []
        call_count = 0

        def mock_probe(max_index: int = 10) -> set[int]:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return {0, 1}  # initial: both present
            return {0}          # device 1 disappeared

        with patch("autoptz.engine.discovery.usb._probe_indices", mock_probe):
            discovery = USBDiscovery(poll_interval=0.05)
            discovery.on_change(lambda ev, dev: events.append((ev, dev)))
            discovery.start()
            time.sleep(0.4)
            discovery.stop()

        removed = [e for e in events if e[0] == "removed"]
        assert any(e[1].index == 1 for e in removed)

    def test_devices_property_reflects_current_state(self) -> None:
        def mock_probe(max_index: int = 10) -> set[int]:
            return {0, 1, 3}

        with patch("autoptz.engine.discovery.usb._probe_indices", mock_probe):
            discovery = USBDiscovery(poll_interval=0.05)
            discovery.start()
            time.sleep(0.2)
            devices = discovery.devices
            discovery.stop()

        indices = {d.index for d in devices}
        assert {0, 1, 3} == indices

    def test_no_events_when_nothing_changes(self) -> None:
        events: list[tuple[str, USBDevice]] = []

        def mock_probe(max_index: int = 10) -> set[int]:
            return {0}

        with patch("autoptz.engine.discovery.usb._probe_indices", mock_probe):
            discovery = USBDiscovery(poll_interval=0.05)
            discovery.on_change(lambda ev, dev: events.append((ev, dev)))
            discovery.start()
            time.sleep(0.3)
            discovery.stop()

        # Only the initial "added" event for device 0, nothing else
        event_types = [e[0] for e in events]
        assert event_types.count("removed") == 0

    def test_multiple_callbacks_all_called(self) -> None:
        events1: list[str] = []
        events2: list[str] = []

        def mock_probe(max_index: int = 10) -> set[int]:
            return {0}

        with patch("autoptz.engine.discovery.usb._probe_indices", mock_probe):
            discovery = USBDiscovery(poll_interval=0.05)
            discovery.on_change(lambda ev, _: events1.append(ev))
            discovery.on_change(lambda ev, _: events2.append(ev))
            discovery.start()
            time.sleep(0.2)
            discovery.stop()

        assert len(events1) > 0
        assert len(events2) > 0


# ── NDIDiscovery ───────────────────────────────────────────────────────────────

def _install_mock_cyndilib_for_discovery(source_names_sequence: list[list[str]]) -> None:
    """Install a mock cyndilib.finder.Finder that returns successive source lists."""
    cyn = types.ModuleType("cyndilib")
    finder_mod = types.ModuleType("cyndilib.finder")

    call_idx = [0]

    def make_sources(names: list[str]) -> list[MagicMock]:
        srcs = []
        for name in names:
            src = MagicMock()
            src.__str__ = MagicMock(return_value=name)
            src.url_address = ""
            srcs.append(src)
        return srcs

    class MockFinder:
        def open(self) -> None: ...
        def close(self) -> None: ...
        def iter_sources(self):  # type: ignore[override]
            idx = call_idx[0]
            names = (
                source_names_sequence[idx]
                if idx < len(source_names_sequence)
                else source_names_sequence[-1]
            )
            call_idx[0] += 1
            return iter(make_sources(names))

    finder_mod.Finder = MockFinder
    sys.modules["cyndilib"] = cyn
    sys.modules["cyndilib.finder"] = finder_mod


def _remove_mock_cyndilib_for_discovery() -> None:
    for mod in ["cyndilib", "cyndilib.finder"]:
        sys.modules.pop(mod, None)
    import autoptz.engine.discovery.ndi as ndi_mod  # re-import to clear state
    # Reset any cached state
    ndi_mod.NDIDiscovery  # just reference to confirm importable


class TestNDIDiscovery:
    def teardown_method(self, _: object) -> None:
        _remove_mock_cyndilib_for_discovery()

    def test_new_source_fires_added_event(self) -> None:
        events: list[tuple[str, NDISource]] = []
        # First poll: no sources. Second poll: one source appears.
        _install_mock_cyndilib_for_discovery([
            [],
            ["LAPTOP (TEST)"],
            ["LAPTOP (TEST)"],
        ])

        discovery = NDIDiscovery(poll_interval=0.05)
        discovery.on_change(lambda ev, src: events.append((ev, src)))
        discovery.start()
        time.sleep(0.5)
        discovery.stop()

        added = [e for e in events if e[0] == "added"]
        assert any(e[1].name == "LAPTOP (TEST)" for e in added)

    def test_removed_source_fires_removed_event(self) -> None:
        events: list[tuple[str, NDISource]] = []
        # First poll: source present. Second poll: gone.
        _install_mock_cyndilib_for_discovery([
            ["NDI_CAM_1"],
            [],
            [],
        ])

        discovery = NDIDiscovery(poll_interval=0.05)
        discovery.on_change(lambda ev, src: events.append((ev, src)))
        discovery.start()
        time.sleep(0.5)
        discovery.stop()

        removed = [e for e in events if e[0] == "removed"]
        assert any(e[1].name == "NDI_CAM_1" for e in removed)

    def test_sources_property(self) -> None:
        _install_mock_cyndilib_for_discovery([
            ["SOURCE_A", "SOURCE_B"],
            ["SOURCE_A", "SOURCE_B"],
        ])

        discovery = NDIDiscovery(poll_interval=0.05)
        discovery.start()
        time.sleep(0.3)
        sources = discovery.sources
        discovery.stop()

        names = {s.name for s in sources}
        assert "SOURCE_A" in names
        assert "SOURCE_B" in names

    def test_cyndilib_unavailable_does_not_raise(self) -> None:
        """If cyndilib is not installed, start() should log a warning and return."""
        _remove_mock_cyndilib_for_discovery()  # ensure cyndilib not in sys.modules

        discovery = NDIDiscovery(poll_interval=0.05)
        # Should NOT raise even without cyndilib
        discovery.start()
        time.sleep(0.1)
        discovery.stop()

        assert discovery.sources == []

    def test_no_events_when_sources_stable(self) -> None:
        events: list[tuple[str, NDISource]] = []
        _install_mock_cyndilib_for_discovery([
            ["STABLE"],
            ["STABLE"],
            ["STABLE"],
        ])

        discovery = NDIDiscovery(poll_interval=0.05)
        discovery.on_change(lambda ev, src: events.append((ev, src)))
        discovery.start()
        time.sleep(0.3)
        discovery.stop()

        # "STABLE" should appear as added exactly once
        added = [e for e in events if e[0] == "added"]
        removed = [e for e in events if e[0] == "removed"]
        assert len(added) == 1
        assert len(removed) == 0


# ── ONVIFDiscovery ─────────────────────────────────────────────────────────────

def _install_mock_wsdiscovery(services_sequence: list[list[dict]]) -> None:
    """Install a mock wsdiscovery that returns successive service lists."""
    wsd_mod = types.ModuleType("wsdiscovery")
    call_idx = [0]

    def make_service(data: dict) -> MagicMock:
        svc = MagicMock()
        svc.getXAddrs.return_value = data.get("xaddrs", [])
        svc.getTypes.return_value = data.get("types", [])
        svc.getScopes.return_value = data.get("scopes", [])
        return svc

    class MockWSDiscovery:
        def start(self) -> None: ...
        def stop(self) -> None: ...

        def searchServices(self, timeout: float = 3) -> list[MagicMock]:
            idx = call_idx[0]
            services_data = (
                services_sequence[idx]
                if idx < len(services_sequence)
                else services_sequence[-1]
            )
            call_idx[0] += 1
            return [make_service(d) for d in services_data]

    wsd_mod.WSDiscovery = MockWSDiscovery
    sys.modules["wsdiscovery"] = wsd_mod


def _remove_mock_wsdiscovery() -> None:
    sys.modules.pop("wsdiscovery", None)


class TestONVIFDiscovery:
    def teardown_method(self, _: object) -> None:
        _remove_mock_wsdiscovery()

    def test_discovered_device_fires_added(self) -> None:
        events: list[tuple[str, ONVIFDevice]] = []
        _install_mock_wsdiscovery([
            [{"xaddrs": ["http://192.168.1.10/onvif/device_service"], "types": [], "scopes": []}],
        ])

        discovery = ONVIFDiscovery(rescan_interval=60.0)
        discovery.on_change(lambda ev, dev: events.append((ev, dev)))
        discovery.start()
        time.sleep(0.5)
        discovery.stop()

        added = [e for e in events if e[0] == "added"]
        assert len(added) == 1
        assert added[0][1].host == "192.168.1.10"

    def test_device_removed_after_miss_threshold(self) -> None:
        events: list[tuple[str, ONVIFDevice]] = []
        device = {"xaddrs": ["http://10.0.0.5/onvif/device_service"], "types": [], "scopes": []}
        # Present on first scan, absent on next 3 (miss threshold = 3)
        _install_mock_wsdiscovery([
            [device],
            [],  # miss 1
            [],  # miss 2
            [],  # miss 3 → removed
        ])

        discovery = ONVIFDiscovery(rescan_interval=0.05)
        discovery.on_change(lambda ev, dev: events.append((ev, dev)))
        discovery.start()
        time.sleep(0.6)
        discovery.stop()

        added = [e for e in events if e[0] == "added"]
        removed = [e for e in events if e[0] == "removed"]
        assert len(added) == 1
        assert len(removed) == 1
        assert removed[0][1].host == "10.0.0.5"

    def test_stable_device_not_re_reported(self) -> None:
        events: list[tuple[str, ONVIFDevice]] = []
        device = {"xaddrs": ["http://172.16.0.1/onvif"], "types": [], "scopes": []}
        _install_mock_wsdiscovery([
            [device], [device], [device],
        ])

        discovery = ONVIFDiscovery(rescan_interval=0.05)
        discovery.on_change(lambda ev, dev: events.append((ev, dev)))
        discovery.start()
        time.sleep(0.4)
        discovery.stop()

        # Should only fire "added" once, not on every rescan
        added = [e for e in events if e[0] == "added"]
        assert len(added) == 1

    def test_wsdiscovery_unavailable_does_not_raise(self) -> None:
        _remove_mock_wsdiscovery()

        discovery = ONVIFDiscovery(rescan_interval=0.05)
        discovery.start()
        time.sleep(0.1)
        discovery.stop()

        assert discovery.devices == []

    def test_devices_property(self) -> None:
        device = {"xaddrs": ["http://192.168.1.20/onvif"], "types": [], "scopes": []}
        _install_mock_wsdiscovery([[device], [device]])

        discovery = ONVIFDiscovery(rescan_interval=0.05)
        discovery.start()
        time.sleep(0.3)
        devices = discovery.devices
        discovery.stop()

        assert len(devices) == 1
        assert devices[0].host == "192.168.1.20"

    def test_multiple_callbacks_all_called(self) -> None:
        results1: list[str] = []
        results2: list[str] = []
        device = {"xaddrs": ["http://192.168.0.1/onvif"], "types": [], "scopes": []}
        _install_mock_wsdiscovery([[device]])

        discovery = ONVIFDiscovery(rescan_interval=60.0)
        discovery.on_change(lambda ev, _: results1.append(ev))
        discovery.on_change(lambda ev, _: results2.append(ev))
        discovery.start()
        time.sleep(0.5)
        discovery.stop()

        assert len(results1) > 0
        assert len(results2) > 0


# ── ONVIFDevice helpers ────────────────────────────────────────────────────────

class TestONVIFDevice:
    def test_primary_xaddr(self) -> None:
        dev = ONVIFDevice(
            xaddrs=("http://192.168.1.5/onvif", "http://192.168.1.5:80/onvif"),
            types=(),
            scopes=(),
        )
        assert dev.primary_xaddr == "http://192.168.1.5/onvif"

    def test_host_extraction(self) -> None:
        dev = ONVIFDevice(
            xaddrs=("http://10.0.0.100:80/onvif/device_service",),
            types=(),
            scopes=(),
        )
        assert dev.host == "10.0.0.100"

    def test_empty_xaddrs(self) -> None:
        dev = ONVIFDevice(xaddrs=(), types=(), scopes=())
        assert dev.primary_xaddr == ""
        assert dev.host == ""
