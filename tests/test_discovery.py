"""Unit tests for autoptz.engine.discovery — USB, NDI, ONVIF.

All network/OS I/O is mocked so tests run without physical cameras,
NDI SDK, or ONVIF-capable devices.
"""

from __future__ import annotations

import sys
import time
import types
from unittest.mock import MagicMock, patch

import autoptz.engine.discovery.usb as usb_mod
from autoptz.engine.discovery.ndi import NDIDiscovery, NDISource
from autoptz.engine.discovery.onvif import ONVIFDevice, ONVIFDiscovery
from autoptz.engine.discovery.usb import USBDevice, USBDiscovery, enumerate_cameras


def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses; return its final value.

    Discovery runs a background poll thread, so waiting for the asserted condition
    (rather than sleeping a fixed window) keeps these tests fast and immune to a
    loaded CI runner stalling the thread.
    """
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(interval)
    return predicate()


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
            _wait_until(lambda: {0, 1} <= {e[1].index for e in events if e[0] == "added"})
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
                return {0}  # initial: only device 0
            return {0, 2}  # subsequent: device 2 appeared

        with patch("autoptz.engine.discovery.usb._probe_indices", mock_probe):
            discovery = USBDiscovery(poll_interval=0.05)
            discovery.on_change(lambda ev, dev: events.append((ev, dev)))
            discovery.start()
            _wait_until(lambda: 2 in {e[1].index for e in events if e[0] == "added"})
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
            return {0}  # device 1 disappeared

        with patch("autoptz.engine.discovery.usb._probe_indices", mock_probe):
            discovery = USBDiscovery(poll_interval=0.05)
            discovery.on_change(lambda ev, dev: events.append((ev, dev)))
            discovery.start()
            _wait_until(lambda: any(e[0] == "removed" and e[1].index == 1 for e in events))
            discovery.stop()

        removed = [e for e in events if e[0] == "removed"]
        assert any(e[1].index == 1 for e in removed)

    def test_devices_property_reflects_current_state(self) -> None:
        def mock_probe(max_index: int = 10) -> set[int]:
            return {0, 1, 3}

        with patch("autoptz.engine.discovery.usb._probe_indices", mock_probe):
            discovery = USBDiscovery(poll_interval=0.05)
            discovery.start()
            _wait_until(lambda: {0, 1, 3} <= {d.index for d in discovery.devices})
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
            _wait_until(lambda: any(e[0] == "added" and e[1].index == 0 for e in events))
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
            _wait_until(lambda: events1 and events2)
            discovery.stop()

        assert len(events1) > 0
        assert len(events2) > 0


# ── _probe_indices: confirm a real frame, not just isOpened() ──────────────────


class _FakeCap:
    """Fake cv2.VideoCapture: configurable open + first-frame behaviour."""

    def __init__(self, *, opened: bool, has_frame: bool) -> None:
        self._opened = opened
        self._has_frame = has_frame
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 — mirrors cv2 API
        return self._opened

    def read(self):
        import numpy as np

        if self._has_frame:
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        return False, None

    def release(self) -> None:
        self.released = True


class TestProbeIndices:
    def test_only_indices_that_open_and_read_are_returned(self, monkeypatch) -> None:
        # Index 0 opens + reads (real); 1 opens but no frame (phantom);
        # 2 doesn't open; the rest don't open.
        def fake_videocapture(i, backend):
            if i == 0:
                return _FakeCap(opened=True, has_frame=True)
            if i == 1:
                return _FakeCap(opened=True, has_frame=False)  # phantom
            return _FakeCap(opened=False, has_frame=False)

        monkeypatch.setattr(usb_mod.cv2, "VideoCapture", fake_videocapture)
        found = usb_mod._probe_indices(max_index=4)
        assert found == {0}

    def test_a_bad_index_does_not_abort_scan(self, monkeypatch) -> None:
        def fake_videocapture(i, backend):
            if i == 0:
                raise RuntimeError("driver explosion")
            if i == 2:
                return _FakeCap(opened=True, has_frame=True)
            return _FakeCap(opened=False, has_frame=False)

        monkeypatch.setattr(usb_mod.cv2, "VideoCapture", fake_videocapture)
        found = usb_mod._probe_indices(max_index=4)
        assert found == {2}


# ── enumerate_cameras: real/openable devices only ──────────────────────────────


class TestEnumerateCameras:
    def test_macos_uses_avfoundation_real_names(self, monkeypatch) -> None:
        """When AVFoundation enumerates, those real devices are returned verbatim."""
        monkeypatch.setattr(usb_mod.platform, "system", lambda: "Darwin")
        fake = [
            {
                "name": "FaceTime HD Camera",
                "unique_id": "0x111",
                "index": 0,
                "is_continuity": False,
            },
            {"name": "iPhone", "unique_id": "0x222", "index": 1, "is_continuity": True},
        ]
        monkeypatch.setattr(usb_mod, "_enumerate_macos_cameras", lambda: fake)
        # If AVFoundation succeeds we must NOT probe cv2 at all.
        monkeypatch.setattr(
            usb_mod,
            "_probed_fallback_cameras",
            lambda: (_ for _ in ()).throw(AssertionError("probed!")),
        )
        cams = enumerate_cameras()
        assert cams == fake
        assert all(isinstance(c["name"], str) for c in cams)

    def test_fallback_probes_openable_indices_only(self, monkeypatch) -> None:
        """No AVFoundation → probe real openable indices, never blind 0-3."""
        monkeypatch.setattr(usb_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(usb_mod, "_enumerate_macos_cameras", lambda: None)
        # Only index 0 is real and openable.
        monkeypatch.setattr(usb_mod, "_probe_indices", lambda *a, **k: {0})
        cams = enumerate_cameras()
        assert [c["index"] for c in cams] == [0]
        assert cams[0]["name"] == "Camera 0"
        assert cams[0]["unique_id"] is None
        assert cams[0]["is_continuity"] is False

    def test_fallback_returns_empty_when_nothing_openable(self, monkeypatch) -> None:
        """No real cameras → empty list, NOT phantom Camera 0-3."""
        monkeypatch.setattr(usb_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr(usb_mod, "_probe_indices", lambda *a, **k: set())
        assert enumerate_cameras() == []

    def test_never_raises_on_probe_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(usb_mod.platform, "system", lambda: "Linux")

        def boom(*a, **k):
            raise RuntimeError("probe broke")

        monkeypatch.setattr(usb_mod, "_probe_indices", boom)
        assert enumerate_cameras() == []


class TestEnumerationIsCheap:
    """``usb_enumeration_is_cheap`` gates the UI's background hotplug poll: only
    the AVFoundation listing (no device opens) is cheap enough to poll."""

    def test_non_darwin_is_never_cheap(self, monkeypatch) -> None:
        monkeypatch.setattr(usb_mod.platform, "system", lambda: "Linux")
        assert usb_mod.usb_enumeration_is_cheap() is False

    def test_darwin_with_avfoundation_is_cheap(self, monkeypatch) -> None:
        monkeypatch.setattr(usb_mod.platform, "system", lambda: "Darwin")
        # AVFoundation importable → cheap (stub the module so it "imports").
        monkeypatch.setitem(sys.modules, "AVFoundation", types.ModuleType("AVFoundation"))
        assert usb_mod.usb_enumeration_is_cheap() is True

    def test_darwin_without_avfoundation_is_not_cheap(self, monkeypatch) -> None:
        monkeypatch.setattr(usb_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setitem(sys.modules, "AVFoundation", None)  # import → ImportError
        assert usb_mod.usb_enumeration_is_cheap() is False


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
    assert ndi_mod.NDIDiscovery is not None  # confirm importable


class TestNDIDiscovery:
    def teardown_method(self, _: object) -> None:
        _remove_mock_cyndilib_for_discovery()

    def test_new_source_fires_added_event(self) -> None:
        events: list[tuple[str, NDISource]] = []
        # First poll: no sources. Second poll: one source appears.
        _install_mock_cyndilib_for_discovery(
            [
                [],
                ["LAPTOP (TEST)"],
                ["LAPTOP (TEST)"],
            ]
        )

        discovery = NDIDiscovery(poll_interval=0.05)
        discovery.on_change(lambda ev, src: events.append((ev, src)))
        discovery.start()
        _wait_until(lambda: any(e[0] == "added" and e[1].name == "LAPTOP (TEST)" for e in events))
        discovery.stop()

        added = [e for e in events if e[0] == "added"]
        assert any(e[1].name == "LAPTOP (TEST)" for e in added)

    def test_removed_source_fires_removed_event(self) -> None:
        events: list[tuple[str, NDISource]] = []
        # First poll: source present. Second poll: gone.
        _install_mock_cyndilib_for_discovery(
            [
                ["NDI_CAM_1"],
                [],
                [],
            ]
        )

        discovery = NDIDiscovery(poll_interval=0.05)
        discovery.on_change(lambda ev, src: events.append((ev, src)))
        discovery.start()
        _wait_until(lambda: any(e[0] == "removed" and e[1].name == "NDI_CAM_1" for e in events))
        discovery.stop()

        removed = [e for e in events if e[0] == "removed"]
        assert any(e[1].name == "NDI_CAM_1" for e in removed)

    def test_sources_property(self) -> None:
        _install_mock_cyndilib_for_discovery(
            [
                ["SOURCE_A", "SOURCE_B"],
                ["SOURCE_A", "SOURCE_B"],
            ]
        )

        discovery = NDIDiscovery(poll_interval=0.05)
        discovery.start()
        _wait_until(lambda: {"SOURCE_A", "SOURCE_B"} <= {s.name for s in discovery.sources})
        sources = discovery.sources
        discovery.stop()

        names = {s.name for s in sources}
        assert "SOURCE_A" in names
        assert "SOURCE_B" in names

    def test_cyndilib_unavailable_does_not_raise(self) -> None:
        """If cyndilib is not installed, start() should log a warning and return."""
        _remove_mock_cyndilib_for_discovery()  # ensure cyndilib not in sys.modules

        discovery = NDIDiscovery(poll_interval=0.05)
        # Should NOT raise even without cyndilib (no poll thread starts).
        discovery.start()
        discovery.stop()

        assert discovery.sources == []

    def test_no_events_when_sources_stable(self) -> None:
        events: list[tuple[str, NDISource]] = []
        _install_mock_cyndilib_for_discovery(
            [
                ["STABLE"],
                ["STABLE"],
                ["STABLE"],
            ]
        )

        discovery = NDIDiscovery(poll_interval=0.05)
        discovery.on_change(lambda ev, src: events.append((ev, src)))
        discovery.start()
        _wait_until(lambda: any(e[0] == "added" for e in events))
        discovery.stop()

        # "STABLE" should appear as added exactly once (dedup holds across polls).
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
                services_sequence[idx] if idx < len(services_sequence) else services_sequence[-1]
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
        _install_mock_wsdiscovery(
            [
                [
                    {
                        "xaddrs": ["http://192.168.1.10/onvif/device_service"],
                        "types": [],
                        "scopes": [],
                    }
                ],
            ]
        )

        discovery = ONVIFDiscovery(rescan_interval=60.0)
        discovery.on_change(lambda ev, dev: events.append((ev, dev)))
        discovery.start()
        _wait_until(lambda: any(e[0] == "added" for e in events))
        discovery.stop()

        added = [e for e in events if e[0] == "added"]
        assert len(added) == 1
        assert added[0][1].host == "192.168.1.10"

    def test_device_removed_after_miss_threshold(self) -> None:
        events: list[tuple[str, ONVIFDevice]] = []
        device = {"xaddrs": ["http://10.0.0.5/onvif/device_service"], "types": [], "scopes": []}
        # Present on first scan, absent on next 3 (miss threshold = 3)
        _install_mock_wsdiscovery(
            [
                [device],
                [],  # miss 1
                [],  # miss 2
                [],  # miss 3 → removed
            ]
        )

        discovery = ONVIFDiscovery(rescan_interval=0.05)
        discovery.on_change(lambda ev, dev: events.append((ev, dev)))
        discovery.start()
        _wait_until(lambda: any(e[0] == "removed" for e in events))
        discovery.stop()

        added = [e for e in events if e[0] == "added"]
        removed = [e for e in events if e[0] == "removed"]
        assert len(added) == 1
        assert len(removed) == 1
        assert removed[0][1].host == "10.0.0.5"

    def test_stable_device_not_re_reported(self) -> None:
        events: list[tuple[str, ONVIFDevice]] = []
        device = {"xaddrs": ["http://172.16.0.1/onvif"], "types": [], "scopes": []}
        _install_mock_wsdiscovery(
            [
                [device],
                [device],
                [device],
            ]
        )

        discovery = ONVIFDiscovery(rescan_interval=0.05)
        discovery.on_change(lambda ev, dev: events.append((ev, dev)))
        discovery.start()
        _wait_until(lambda: any(e[0] == "added" for e in events))
        discovery.stop()

        # Should only fire "added" once, not on every rescan (dedup holds).
        added = [e for e in events if e[0] == "added"]
        assert len(added) == 1

    def test_wsdiscovery_unavailable_does_not_raise(self) -> None:
        _remove_mock_wsdiscovery()

        discovery = ONVIFDiscovery(rescan_interval=0.05)
        discovery.start()
        discovery.stop()

        assert discovery.devices == []

    def test_devices_property(self) -> None:
        device = {"xaddrs": ["http://192.168.1.20/onvif"], "types": [], "scopes": []}
        _install_mock_wsdiscovery([[device], [device]])

        discovery = ONVIFDiscovery(rescan_interval=0.05)
        discovery.start()
        _wait_until(lambda: len(discovery.devices) >= 1)
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
        _wait_until(lambda: results1 and results2)
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
