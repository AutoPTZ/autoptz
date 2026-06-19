"""Phase 6: PTZ backend factory + worker auto/manual control-loop tests.

Everything is mocked — no real serial ports, sockets, NDI runtime, or ONVIF
devices are touched.  Covers:

* ``build_backend`` dispatch per backend kind (visca_usb / visca_ip / ndi / onvif).
* Graceful ``None`` (never raises) when an optional dep is absent or a device is
  unreachable, and when ``backend`` is unconfigured / ``"auto"`` with nothing
  probable.
* The ``"auto"`` probe (NDI source → NDI PTZ; address → ONVIF/VISCA-IP).
* The framing-name → target subject-height mapping in the controller.
* The ``PTZConfig.zoom_framing`` model change + legacy migration.
* CameraWorker integration: manual nudge → ``backend.move_velocity``; auto target
  error → command in the right direction; manual override suspends then resumes
  auto; ``stop()`` always halts the backend.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from autoptz.config.models import PTZConfig
from autoptz.engine.ptz import factory
from autoptz.engine.ptz.base import PTZBackend, PTZCaps, PTZState
from autoptz.engine.ptz.controller import (
    _DEFAULT_ZOOM_FRAMING_TARGET,
    _ZOOM_FRAMING_TARGETS,
)
from autoptz.engine.ptz.factory import _split_host_port, build_backend


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


class RecordingBackend(PTZBackend):
    """Records every command; optionally reports an absolute position."""

    def __init__(self, has_position: bool = False) -> None:
        super().__init__()
        self.caps = PTZCaps(continuous_pan_tilt=True, continuous_zoom=True)
        self.moves: list[tuple[float, float, float]] = []
        self.stop_count = 0
        self.closed = False
        self._pos = PTZState(pan=0.1, tilt=0.2, zoom=0.3) if has_position else None

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        self.moves.append((pan, tilt, zoom))

    def stop(self) -> None:
        self.stop_count += 1

    def get_position(self) -> PTZState | None:
        return self._pos

    def goto_preset(self, idx: int) -> None:  # pragma: no cover - unused here
        pass

    def save_preset(self, idx: int) -> None:  # pragma: no cover - unused here
        pass

    def close(self) -> None:
        self.closed = True


def _cfg(**kw: object) -> PTZConfig:
    return PTZConfig(**kw)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Factory dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestFactoryDispatch:
    def test_visca_usb_dispatch(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake_ctor(port, *a, **k):
            seen["port"] = port
            return RecordingBackend()

        monkeypatch.setattr("autoptz.engine.ptz.visca_usb.ViscaUSBBackend", fake_ctor)
        b = build_backend(_cfg(backend="visca_usb", address="/dev/tty.usbserial"))
        assert b is not None
        assert seen["port"] == "/dev/tty.usbserial"

    def test_visca_usb_no_address_returns_none(self) -> None:
        assert build_backend(_cfg(backend="visca_usb", address=None)) is None

    def test_visca_ip_dispatch_with_port(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake_ctor(host, port, *a, **k):
            seen["host"] = host
            seen["port"] = port
            return RecordingBackend()

        monkeypatch.setattr("autoptz.engine.ptz.visca_ip.ViscaIPBackend", fake_ctor)
        b = build_backend(_cfg(backend="visca_ip", address="192.168.1.50:5678"))
        assert b is not None
        assert seen["host"] == "192.168.1.50"
        assert seen["port"] == 5678

    def test_visca_ip_default_port(self, monkeypatch) -> None:
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "autoptz.engine.ptz.visca_ip.ViscaIPBackend",
            lambda host, port, *a, **k: seen.update(host=host, port=port)
            or RecordingBackend(),
        )
        build_backend(_cfg(backend="visca_ip", address="cam.local"))
        assert seen["port"] == factory._DEFAULT_VISCA_IP_PORT

    def test_ndi_dispatch_with_receiver(self, monkeypatch) -> None:
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "autoptz.engine.ptz.ndi_ptz.NDIPTZBackend",
            lambda recv, *a, **k: seen.update(recv=recv) or RecordingBackend(),
        )
        recv = object()
        b = build_backend(_cfg(backend="ndi"), ndi_source=recv)
        assert b is not None
        assert seen["recv"] is recv

    def test_ndi_without_receiver_returns_none(self) -> None:
        # No receiver available → graceful None (NDI PTZ rides the receiver).
        assert build_backend(_cfg(backend="ndi"), ndi_source=None) is None

    def test_onvif_dispatch(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake_ctor(host, port, *, username, password):
            seen.update(host=host, port=port, username=username, password=password)
            return RecordingBackend()

        monkeypatch.setattr("autoptz.engine.ptz.onvif_ptz.ONVIFPTZBackend", fake_ctor)
        b = build_backend(_cfg(backend="onvif", address="onvif://10.0.0.5:8080"))
        assert b is not None
        assert seen["host"] == "10.0.0.5"
        assert seen["port"] == 8080

    def test_unknown_backend_returns_none(self) -> None:
        # bypass the Literal validation by constructing then mutating via a dict
        cfg = _cfg(backend="auto")
        cfg = cfg.model_copy(update={"backend": "bogus"})
        assert build_backend(cfg) is None


# ─────────────────────────────────────────────────────────────────────────────
# Graceful failure — never raises
# ─────────────────────────────────────────────────────────────────────────────


class TestFactoryGraceful:
    def test_missing_optional_dep_returns_none(self, monkeypatch) -> None:
        # ViscaUSB ctor raises ImportError (pyserial missing) → None, no raise.
        def boom(*a, **k):
            raise ImportError("pyserial not installed")

        monkeypatch.setattr("autoptz.engine.ptz.visca_usb.ViscaUSBBackend", boom)
        assert build_backend(_cfg(backend="visca_usb", address="/dev/ttyX")) is None

    def test_unreachable_device_returns_none(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr("autoptz.engine.ptz.visca_ip.ViscaIPBackend", boom)
        assert build_backend(_cfg(backend="visca_ip", address="10.0.0.9")) is None

    def test_onvif_dep_missing_returns_none(self, monkeypatch) -> None:
        def boom(*a, **k):
            raise ImportError("onvif-zeep not installed")

        monkeypatch.setattr("autoptz.engine.ptz.onvif_ptz.ONVIFPTZBackend", boom)
        assert build_backend(_cfg(backend="onvif", address="10.0.0.9")) is None

    def test_none_backend_returns_none(self) -> None:
        cfg = _cfg(backend="auto").model_copy(update={"backend": ""})
        assert build_backend(cfg) is None

    def test_auto_no_source_no_address_returns_none(self) -> None:
        assert build_backend(_cfg(backend="auto", address=None)) is None


# ─────────────────────────────────────────────────────────────────────────────
# Auto probe
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoProbe:
    def test_auto_prefers_ndi_when_source_present(self, monkeypatch) -> None:
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "autoptz.engine.ptz.ndi_ptz.NDIPTZBackend",
            lambda recv, *a, **k: seen.update(recv=recv) or RecordingBackend(),
        )
        recv = object()
        b = build_backend(_cfg(backend="auto"), ndi_source=recv)
        assert b is not None
        assert seen["recv"] is recv

    def test_auto_tries_onvif_then_visca_by_address(self, monkeypatch) -> None:
        # ONVIF fails (unreachable) → factory falls through to VISCA-IP.
        monkeypatch.setattr(
            "autoptz.engine.ptz.onvif_ptz.ONVIFPTZBackend",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no onvif")),
        )
        built = RecordingBackend()
        monkeypatch.setattr(
            "autoptz.engine.ptz.visca_ip.ViscaIPBackend",
            lambda *a, **k: built,
        )
        b = build_backend(_cfg(backend="auto", address="192.168.0.10"))
        assert b is built

    def test_auto_returns_none_when_nothing_reachable(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "autoptz.engine.ptz.onvif_ptz.ONVIFPTZBackend",
            lambda *a, **k: (_ for _ in ()).throw(OSError()),
        )
        monkeypatch.setattr(
            "autoptz.engine.ptz.visca_ip.ViscaIPBackend",
            lambda *a, **k: (_ for _ in ()).throw(OSError()),
        )
        assert build_backend(_cfg(backend="auto", address="192.168.0.10")) is None


class TestAddressParsing:
    def test_split_plain_host(self) -> None:
        assert _split_host_port("cam.local", 80) == ("cam.local", 80)

    def test_split_host_port(self) -> None:
        assert _split_host_port("1.2.3.4:9000", 80) == ("1.2.3.4", 9000)

    def test_split_scheme_and_path(self) -> None:
        assert _split_host_port("onvif://1.2.3.4:88/path", 80) == ("1.2.3.4", 88)

    def test_split_empty(self) -> None:
        assert _split_host_port("", 80) == ("", 80)
        assert _split_host_port(None, 52381) == ("", 52381)


# ─────────────────────────────────────────────────────────────────────────────
# zoom_framing model change + framing → target-height map
# ─────────────────────────────────────────────────────────────────────────────


class TestFramingModel:
    def test_default_is_upper_body(self) -> None:
        assert PTZConfig().zoom_framing == "upper_body"

    @pytest.mark.parametrize(
        "name", ["face", "head_shoulders", "upper_body", "full_body", "wide"],
    )
    def test_five_named_presets_accepted(self, name) -> None:
        assert _cfg(zoom_framing=name).zoom_framing == name

    def test_legacy_medium_migrates_to_upper_body(self) -> None:
        # legacy "medium" → "upper_body" (preserves the 0.45 target)
        assert _cfg(zoom_framing="medium").zoom_framing == "upper_body"

    def test_legacy_tight_migrates(self) -> None:
        assert _cfg(zoom_framing="tight").zoom_framing == "head_shoulders"

    def test_legacy_wide_still_valid(self) -> None:
        assert _cfg(zoom_framing="wide").zoom_framing == "wide"


class TestPresetSlotsModel:
    """The ``PTZConfig.preset_slots`` occupancy map (tile quick-recall row)."""

    def test_default_is_empty(self) -> None:
        assert PTZConfig().preset_slots == {}

    def test_round_trips_through_json_with_int_keys(self) -> None:
        # model_dump_json emits dict keys as strings; the before-validator must
        # coerce them back to int so the PTZ section maps slot index → preset.
        import json

        cfg = _cfg(preset_slots={
            0: {"label": "Stage", "thumbnail": "data:image/png;base64,AA=="},
            3: {"label": "Podium", "thumbnail": None},
        })
        restored = PTZConfig.model_validate(json.loads(cfg.model_dump_json()))
        assert restored.preset_slots[0].label == "Stage"
        assert restored.preset_slots[0].thumbnail == "data:image/png;base64,AA=="
        assert restored.preset_slots[3].label == "Podium"
        assert all(isinstance(k, int) for k in restored.preset_slots)

    def test_legacy_string_label_migrates(self) -> None:
        # An older stored config used {slot: "label"} plain strings.
        slots = _cfg(preset_slots={0: "Stage", 3: "Podium"}).preset_slots
        assert slots[0].label == "Stage" and slots[0].thumbnail is None
        assert slots[3].label == "Podium"

    def test_backward_compatible_when_field_absent(self) -> None:
        # An older stored config (no preset_slots key) still validates → {}.
        assert PTZConfig.model_validate({"backend": "auto"}).preset_slots == {}

    def test_bad_keys_are_dropped_not_raised(self) -> None:
        slots = _cfg(preset_slots={"x": "nope", 2: "ok"}).preset_slots
        assert set(slots) == {2} and slots[2].label == "ok"


class TestFramingTargetMap:
    def test_ordering_tighter_is_larger_fraction(self) -> None:
        t = _ZOOM_FRAMING_TARGETS
        assert (
            t["face"] > t["head_shoulders"] > t["upper_body"]
            > t["full_body"] > t["wide"]
        )

    def test_expected_values(self) -> None:
        t = _ZOOM_FRAMING_TARGETS
        assert t["face"] == pytest.approx(0.80)
        assert t["head_shoulders"] == pytest.approx(0.60)
        assert t["upper_body"] == pytest.approx(0.45)
        assert t["full_body"] == pytest.approx(0.30)
        assert t["wide"] == pytest.approx(0.20)

    def test_default_target_matches_upper_body(self) -> None:
        assert _DEFAULT_ZOOM_FRAMING_TARGET == pytest.approx(
            _ZOOM_FRAMING_TARGETS["upper_body"]
        )

    def test_controller_uses_named_target(self) -> None:
        from autoptz.engine.ptz.controller import PTZController

        # face framing target=0.80; subject_height 0.2 << 0.80 → zoom in (+).
        ctrl = PTZController(
            RecordingBackend(),
            _cfg(auto_zoom=True, zoom_framing="face", max_zoom_speed=1.0),
        )
        _, _, zoom = ctrl.step((0.0, 0.0), (0.0, 0.0), 0.2, True, t=0.0)
        assert zoom > 0.0

        # wide framing target=0.20; subject_height 0.8 >> 0.20 → zoom out (-).
        ctrl2 = PTZController(
            RecordingBackend(),
            _cfg(auto_zoom=True, zoom_framing="wide", max_zoom_speed=1.0),
        )
        _, _, zoom2 = ctrl2.step((0.0, 0.0), (0.0, 0.0), 0.8, True, t=0.0)
        assert zoom2 < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Backend home() / osd_menu() — byte commands + safe no-op defaults
# ─────────────────────────────────────────────────────────────────────────────


class TestViscaHomeMenuBytes:
    def test_home_cmd_bytes(self) -> None:
        from autoptz.engine.ptz.base import visca_home_cmd
        assert visca_home_cmd() == bytes([0x81, 0x01, 0x06, 0x04, 0xFF])

    def test_menu_cmd_bytes(self) -> None:
        from autoptz.engine.ptz.base import visca_menu_cmd
        assert visca_menu_cmd() == bytes([0x81, 0x01, 0x06, 0x06, 0x10, 0xFF])


class TestBackendHomeMenuDefaults:
    def test_base_home_menu_are_noop(self) -> None:
        # The RecordingBackend doesn't override home()/osd_menu(), so it inherits
        # the base no-op defaults — calling them must not raise or move.
        backend = RecordingBackend()
        backend.home()
        backend.osd_menu()
        assert backend.moves == []
        assert backend.stop_count == 0

    def test_visca_backends_emit_home_menu(self) -> None:
        # A fake VISCA backend (both IP + USB share the same byte helpers via
        # _send) records the wire bytes for home()/osd_menu().
        from autoptz.engine.ptz.base import (
            PTZBackend,
            visca_home_cmd,
            visca_menu_cmd,
        )

        class FakeVisca(PTZBackend):
            def __init__(self) -> None:
                super().__init__()
                self.sent: list[bytes] = []

            def _send(self, cmd: bytes) -> None:
                self.sent.append(cmd)

            def move_velocity(self, pan, tilt, zoom=0.0):  # pragma: no cover
                pass

            def stop(self):  # pragma: no cover
                pass

            def goto_preset(self, idx):  # pragma: no cover
                pass

            def save_preset(self, idx):  # pragma: no cover
                pass

            def close(self):  # pragma: no cover
                pass

            def home(self) -> None:
                self._send(visca_home_cmd())

            def osd_menu(self) -> None:
                self._send(visca_menu_cmd())

        fake = FakeVisca()
        fake.home()
        fake.osd_menu()
        assert fake.sent == [visca_home_cmd(), visca_menu_cmd()]


# ─────────────────────────────────────────────────────────────────────────────
# CameraWorker integration (auto / manual control loop)
# ─────────────────────────────────────────────────────────────────────────────


def _bbox(x1, y1, x2, y2):
    from autoptz.engine.runtime.messages import BBox
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _track(track_id, bbox):
    from autoptz.engine.runtime.messages import TrackInfo
    return TrackInfo(track_id=track_id, bbox=bbox)


def _worker(camera_id="ptzcam0xabcd", *, ptz_backend=None, ptz_controller=None,
            mode="manual"):
    from autoptz.config.models import (
        CameraConfig,
        PTZConfig,
        SourceConfig,
        TargetConfig,
    )
    from autoptz.engine.camera_worker import CameraWorker

    cfg = CameraConfig(
        id=camera_id,
        name="PTZ Cam",
        source=SourceConfig(type="usb", address="usb://0"),
        ptz=PTZConfig(backend="visca_ip", address="cam.local",
                      kp=1.0, kd=0.0, kv=0.0, deadzone_x=0.0, deadzone_y=0.0,
                      max_pan_speed=1.0, max_tilt_speed=1.0),
        target=TargetConfig(mode=mode),
    )
    return CameraWorker(
        camera_id, cfg, lambda m: None,
        ptz_backend=ptz_backend, ptz_controller=ptz_controller,
    )


class TestWorkerManualNudge:
    def test_nudge_drives_backend_move_velocity(self) -> None:
        backend = RecordingBackend()
        w = _worker(ptz_backend=backend)
        w._drive_ptz_nudge(0.7, -0.2, 0.0)
        assert backend.moves
        assert backend.moves[-1] == pytest.approx((0.7, -0.2, 0.0))

    def test_nudge_opens_manual_override_window(self) -> None:
        backend = RecordingBackend()
        w = _worker(ptz_backend=backend)
        w._drive_ptz_nudge(0.5, 0.0, 0.0)
        assert w._manual_override_active(time.monotonic()) is True

    def test_nudge_safe_without_backend(self) -> None:
        w = _worker(ptz_backend=None, ptz_controller=None)
        # Override default-config controller build to ensure no backend.
        w._ptz = None
        w._ptz_backend = None
        w._drive_ptz_nudge(0.5, 0.0, 0.0)  # must not raise


class TestWorkerAutoControl:
    def _ctrl_worker(self, backend):
        from autoptz.engine.ptz.controller import PTZController
        w = _worker(ptz_backend=backend, mode="manual")
        # Wire a real controller around the recording backend.
        w._ptz = PTZController(backend, w.config.ptz)
        w._ptz_backend = backend
        w._tracking_enabled = True
        w._target_track_id = 7
        return w

    def test_auto_target_right_of_center_pans_right(self) -> None:
        backend = RecordingBackend()
        w = self._ctrl_worker(backend)
        # bbox centered well right of frame center (w=1000) → ex>0 → pan right>0
        frame = np.zeros((720, 1000, 3), dtype=np.uint8)
        tracks = [_track(7, _bbox(800, 300, 900, 500))]
        w._drive_ptz_auto(tracks, frame, now=0.0)
        assert backend.moves, "auto control sent no command"
        pan = backend.moves[-1][0]
        assert pan > 0.0, f"expected pan right (>0), got {pan}"

    def test_auto_target_left_of_center_pans_left(self) -> None:
        backend = RecordingBackend()
        w = self._ctrl_worker(backend)
        frame = np.zeros((720, 1000, 3), dtype=np.uint8)
        tracks = [_track(7, _bbox(50, 300, 150, 500))]  # far left → ex<0
        w._drive_ptz_auto(tracks, frame, now=0.0)
        assert backend.moves
        assert backend.moves[-1][0] < 0.0

    def test_auto_target_below_center_tilts_down(self) -> None:
        backend = RecordingBackend()
        w = self._ctrl_worker(backend)
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # bbox center well below frame center → image-y large → tilt down (<0)
        tracks = [_track(7, _bbox(450, 800, 550, 950))]
        w._drive_ptz_auto(tracks, frame, now=0.0)
        assert backend.moves
        assert backend.moves[-1][1] < 0.0

    def test_manual_override_suspends_then_resumes_auto(self) -> None:
        backend = RecordingBackend()
        w = self._ctrl_worker(backend)
        frame = np.zeros((720, 1000, 3), dtype=np.uint8)
        tracks = [_track(7, _bbox(800, 300, 900, 500))]

        # Open a manual override (nudge) → auto must be suspended.
        w._drive_ptz_nudge(0.9, 0.0, 0.0)
        nudge_moves = len(backend.moves)
        w._drive_ptz_auto(tracks, frame, now=time.monotonic())
        assert len(backend.moves) == nudge_moves, "auto ran during manual override"

        # After the window expires, auto resumes and drives the backend again.
        future = w._manual_override_until + 0.01
        w._drive_ptz_auto(tracks, frame, now=future)
        assert len(backend.moves) > nudge_moves, "auto did not resume after override"
        assert backend.moves[-1][0] > 0.0  # pans toward the right-of-center target

    def test_auto_no_target_drives_idle_then_stops(self) -> None:
        from autoptz.engine.ptz.controller import ControllerState, PTZController
        backend = RecordingBackend()
        w = _worker(ptz_backend=backend, mode="manual")
        w._ptz = PTZController(backend, w.config.ptz, coast_window_ms=0)
        w._ptz_backend = backend
        w._tracking_enabled = True
        w._target_track_id = 7
        frame = np.zeros((720, 1000, 3), dtype=np.uint8)

        # Acquire then lose the target → controller coasts → searches → stops.
        w._drive_ptz_auto([_track(7, _bbox(800, 300, 900, 500))], frame, now=0.0)
        w._drive_ptz_auto([], frame, now=0.1)   # lost → coast
        w._drive_ptz_auto([], frame, now=0.2)   # coast window 0 → searching/stop
        assert w._ptz.state in (ControllerState.SEARCHING, ControllerState.COASTING)


class TestWorkerPtzState:
    def test_telemetry_reports_backend_position(self) -> None:
        backend = RecordingBackend(has_position=True)
        w = _worker(ptz_backend=backend)
        st = w._ptz_state()
        assert st.pan == pytest.approx(0.1)
        assert st.tilt == pytest.approx(0.2)
        assert st.zoom == pytest.approx(0.3)
        assert st.moving is True
        assert st.backend == "visca_ip"

    def test_telemetry_falls_back_to_last_cmd(self) -> None:
        backend = RecordingBackend(has_position=False)
        w = _worker(ptz_backend=backend)
        w._drive_ptz_nudge(0.4, 0.0, 0.0)
        st = w._ptz_state()
        assert st.pan == pytest.approx(0.4)
        assert st.state == "manual"  # inside the manual-override window

    def test_telemetry_empty_when_no_ptz(self) -> None:
        w = _worker(ptz_backend=None, ptz_controller=None)
        w._ptz = None
        w._ptz_backend = None
        st = w._ptz_state()
        assert st.pan == 0.0 and st.tilt == 0.0 and st.zoom == 0.0
        assert st.moving is False


class TestWorkerStopHaltsBackend:
    def test_close_resources_stops_backend(self) -> None:
        backend = RecordingBackend()
        w = _worker(ptz_backend=backend)
        w._close_resources()
        assert backend.stop_count >= 1

    def test_close_resources_closes_owned_controller(self) -> None:
        from autoptz.engine.ptz.controller import PTZController
        backend = RecordingBackend()
        w = _worker(ptz_backend=None)
        w._ptz = PTZController(backend, w.config.ptz)
        w._ptz_backend = backend
        w._ptz_owned = True
        w._close_resources()
        # owned controller is close()d → backend stopped and closed
        assert backend.stop_count >= 1
        assert backend.closed is True
