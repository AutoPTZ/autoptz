"""Tests for PTZ stop-on-loss safety features.

Covers:
  1. ONVIF dead-man's switch: ContinuousMove carries a Timeout field.
  2. ONVIF transport-error resilience: move_velocity / stop never raise.
  3. VISCA IP halt-on-reconnect: a stop is always sent before the retried
     command after a reconnect so a mid-pan camera halts on reconnect.
  4. VISCA USB halt-on-reconnect: same guarantee for the serial backend.

No real hardware or network — all services/transports are mocked.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import autoptz.engine.ptz.visca_ip as visca_ip_mod
import autoptz.engine.ptz.visca_usb as visca_usb_mod
from autoptz.engine.ptz.base import visca_stop_cmd, visca_zoom_stop_cmd
from autoptz.engine.ptz.visca_ip import ViscaIPBackend
from autoptz.engine.ptz.visca_usb import ViscaUSBBackend

# ─────────────────────────────────────────────────────────────────────────────
# Fake transports (shared by VISCA tests)
# ─────────────────────────────────────────────────────────────────────────────


class FakeSocket:
    """Fake TCP socket that records sendall calls and can be flipped to raise."""

    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.should_fail: bool = False
        self.closed: bool = False

    def sendall(self, data: bytes) -> None:
        if self.should_fail:
            raise OSError("simulated network failure")
        self.sent.append(data)

    def settimeout(self, t: float) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeSerial:
    """Fake pyserial Serial instance."""

    def __init__(self, port: str, baud: int, timeout: float = 0.1) -> None:
        self.port = port
        self.baud = baud
        self.written: list[bytes] = []
        self.should_fail: bool = False
        self.closed: bool = False
        self.in_waiting: int = 0

    def write(self, data: bytes) -> None:
        if self.should_fail:
            raise visca_usb_mod.serial.SerialException("simulated serial failure")
        self.written.append(data)

    def read(self, n: int) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


class FakeSerialFactory:
    """Collects all FakeSerial instances created so tests can inspect them."""

    def __init__(self) -> None:
        self.instances: list[FakeSerial] = []

    def __call__(self, port: str, baud: int, timeout: float = 0.1) -> FakeSerial:
        s = FakeSerial(port, baud, timeout)
        self.instances.append(s)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# 1. ONVIF dead-man's switch
# ─────────────────────────────────────────────────────────────────────────────


class TestONVIFDeadMansSwitch:
    """ContinuousMove must carry a Timeout so the camera self-stops on loss."""

    def _make(self) -> tuple[Any, MagicMock]:
        """Return (backend, mock_ptz) with all ONVIF network calls stubbed."""
        from autoptz.engine.ptz.onvif_ptz import ONVIFPTZBackend

        mock_cam = MagicMock()
        mock_ptz = MagicMock()
        mock_media = MagicMock()

        mock_cam.create_ptz_service.return_value = mock_ptz
        mock_cam.create_media_service.return_value = mock_media

        fake_profile = MagicMock()
        fake_profile.token = "Profile_1"
        mock_media.GetProfiles.return_value = [fake_profile]
        mock_ptz.GetConfigurationOptions.return_value = MagicMock(Spaces=MagicMock())

        # create_type returns a plain settable object so attribute assignments
        # are captured without MagicMock auto-creating everything.
        class _Req:
            pass

        mock_ptz.create_type.side_effect = lambda _name: _Req()

        with patch("autoptz.engine.ptz.onvif_ptz._require_onvif") as mock_req:
            mock_req.return_value = lambda *a, **kw: mock_cam
            backend = ONVIFPTZBackend("192.0.2.1")

        return backend, mock_ptz

    def test_continuousmove_sets_timeout_field(self) -> None:
        """move_velocity must set request.Timeout to the expected timedelta."""
        from autoptz.engine.ptz.onvif_ptz import _CONTINUOUSMOVE_TIMEOUT

        backend, mock_ptz = self._make()

        # Capture the request object passed to ContinuousMove.
        captured: list[Any] = []
        mock_ptz.ContinuousMove.side_effect = lambda req: captured.append(req)

        backend.move_velocity(0.3, 0.0, 0.0)

        assert len(captured) == 1, "ContinuousMove was not called"
        req = captured[0]
        assert hasattr(req, "Timeout"), "request.Timeout was not set"
        assert req.Timeout == _CONTINUOUSMOVE_TIMEOUT

    def test_timeout_is_timedelta_of_one_second(self) -> None:
        """The dead-man timeout must be a timedelta of exactly 1 second."""
        from autoptz.engine.ptz.onvif_ptz import _CONTINUOUSMOVE_TIMEOUT

        assert isinstance(_CONTINUOUSMOVE_TIMEOUT, timedelta)
        assert _CONTINUOUSMOVE_TIMEOUT == timedelta(seconds=1)

    def test_move_velocity_velocity_fields_correct(self) -> None:
        """move_velocity must still set PanTilt and Zoom on the request."""
        backend, mock_ptz = self._make()

        captured: list[Any] = []
        mock_ptz.ContinuousMove.side_effect = lambda req: captured.append(req)

        backend.move_velocity(0.5, -0.3, 0.1)

        req = captured[0]
        assert req.Velocity["PanTilt"]["x"] == pytest.approx(0.5)
        assert req.Velocity["PanTilt"]["y"] == pytest.approx(-0.3)
        assert req.Velocity["Zoom"]["x"] == pytest.approx(0.1)

    def test_transport_error_does_not_propagate(self) -> None:
        """A SOAP/transport error in move_velocity must be caught — never raised."""
        backend, mock_ptz = self._make()
        mock_ptz.ContinuousMove.side_effect = Exception("simulated SOAP transport error")

        backend.move_velocity(0.3, 0.0, 0.0)  # must not raise

    def test_stop_transport_error_does_not_propagate(self) -> None:
        """A transport error in stop() must be caught — never raised."""
        backend, mock_ptz = self._make()
        mock_ptz.Stop.side_effect = Exception("simulated SOAP transport error")

        backend.stop()  # must not raise

    def test_repeated_errors_throttle_log_warnings(self, caplog: pytest.LogCaptureFixture) -> None:
        """Repeated transport errors within the throttle window emit only one WARNING."""
        import autoptz.engine.ptz.onvif_ptz as onvif_mod

        backend, mock_ptz = self._make()
        mock_ptz.ContinuousMove.side_effect = Exception("transport down")

        # Fix the monotonic clock at t=100.0 and set last_err_log well in the
        # past (t=0) so the first call is past the throttle window and emits a
        # WARNING, while the second call (also at t=100.0) is within the window
        # and must be silenced.
        backend._last_err_log_t = 0.0  # far in the past relative to now=100.0

        # Patch time.monotonic in the onvif_ptz module namespace so the
        # throttle gate sees our fixed clock.
        with patch.object(onvif_mod.time, "monotonic", return_value=100.0):
            with caplog.at_level(logging.WARNING, logger="autoptz.engine.ptz.onvif_ptz"):
                backend.move_velocity(0.3, 0.0)  # first error → WARNING emitted
                n_after_first = caplog.text.count("transport error")
                backend.move_velocity(0.3, 0.0)  # still at t=100.0 → throttled
                n_after_second = caplog.text.count("transport error")

        assert n_after_first == 1
        assert n_after_second == 1  # throttled: still only one


# ─────────────────────────────────────────────────────────────────────────────
# 2. VISCA IP halt-on-reconnect
# ─────────────────────────────────────────────────────────────────────────────


class TestViscaIPHaltOnReconnect:
    """ViscaIPBackend must always send a stop before retrying a command after
    a reconnect, whether or not the camera was previously moving."""

    def _make_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[ViscaIPBackend, list[FakeSocket]]:
        sockets: list[FakeSocket] = []

        def fake_create_connection(
            addr: tuple[str, int], timeout: float | None = None
        ) -> FakeSocket:
            s = FakeSocket()
            sockets.append(s)
            return s

        monkeypatch.setattr(visca_ip_mod.socket, "create_connection", fake_create_connection)
        backend = ViscaIPBackend("192.0.2.1", port=52381, mode="raw", timeout=0.5)
        return backend, sockets

    def test_halt_sent_before_move_after_reconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After a connection drop, reconnect must send stop bytes before the move."""
        backend, sockets = self._make_backend(monkeypatch)

        # Issue a non-zero move, then simulate a drop on the next send.
        backend.move_velocity(0.5, 0.0, 0.0)
        sockets[0].should_fail = True

        # Trigger send → disconnect → reconnect.
        backend.move_velocity(0.5, 0.0, 0.0)

        # Two sockets: original (dropped) + fresh (reconnect).
        assert len(sockets) == 2

        fresh_sock = sockets[1]
        # Fresh socket must receive at least: stop, zoom-stop, retried move.
        assert len(fresh_sock.sent) >= 3, (
            f"Expected ≥3 frames on reconnect (stop, zoom-stop, move), got {len(fresh_sock.sent)}"
        )

        # First two frames must be the stop commands (raw mode: no framing).
        assert fresh_sock.sent[0] == visca_stop_cmd(), (
            f"First frame must be stop cmd, got {fresh_sock.sent[0].hex()}"
        )
        assert fresh_sock.sent[1] == visca_zoom_stop_cmd(), (
            f"Second frame must be zoom-stop cmd, got {fresh_sock.sent[1].hex()}"
        )

    def test_halt_sent_even_when_camera_was_stopped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stop is sent on reconnect regardless of prior command (always-safe)."""
        backend, sockets = self._make_backend(monkeypatch)

        # Explicitly stop the camera, then simulate a connection failure.
        backend.stop()
        sockets[0].should_fail = True
        backend.move_velocity(0.3, 0.0, 0.0)

        assert len(sockets) == 2
        fresh_sock = sockets[1]

        # Even though the camera was stopped, halt bytes are still sent first.
        assert len(fresh_sock.sent) >= 3
        assert fresh_sock.sent[0] == visca_stop_cmd()
        assert fresh_sock.sent[1] == visca_zoom_stop_cmd()

    def test_stop_bytes_precede_retried_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pantilt move is always retried AFTER the stop sequence."""
        from autoptz.engine.ptz.base import visca_pantilt_cmd

        backend, sockets = self._make_backend(monkeypatch)
        backend.move_velocity(0.5, 0.0, 0.0)
        sockets[0].should_fail = True
        backend.move_velocity(0.5, 0.0, 0.0)

        fresh_sock = sockets[1]
        # Determine which frame is the pantilt move cmd.
        pantilt_bytes = visca_pantilt_cmd(0.5, 0.0)
        stop_bytes = visca_stop_cmd()

        stop_idx = next((i for i, b in enumerate(fresh_sock.sent) if b == stop_bytes), None)
        move_idx = next((i for i, b in enumerate(fresh_sock.sent) if b == pantilt_bytes), None)

        assert stop_idx is not None, "stop cmd not found in fresh socket frames"
        assert move_idx is not None, "pantilt move cmd not found in fresh socket frames"
        assert stop_idx < move_idx, (
            f"stop (idx {stop_idx}) must precede pantilt move (idx {move_idx})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISCA USB halt-on-reconnect
# ─────────────────────────────────────────────────────────────────────────────


class TestViscaUSBHaltOnReconnect:
    """ViscaUSBBackend must always send a stop before retrying a command after
    a reconnect, whether or not the camera was previously moving."""

    def _make_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[ViscaUSBBackend, FakeSerialFactory]:
        factory = FakeSerialFactory()
        monkeypatch.setattr(visca_usb_mod.serial, "Serial", factory)
        backend = ViscaUSBBackend("/dev/ttyUSB0", baud=9600)
        return backend, factory

    def test_halt_sent_before_move_after_reconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After a port drop, reconnect must send stop bytes before the move."""
        backend, factory = self._make_backend(monkeypatch)

        # Issue a non-zero move, then simulate a port failure on the next send.
        backend.move_velocity(0.5, 0.0, 0.0)
        factory.instances[0].should_fail = True

        # Trigger send → disconnect → reconnect.
        backend.move_velocity(0.5, 0.0, 0.0)

        # Two serial instances: original (dropped) + fresh (reconnect).
        assert len(factory.instances) == 2

        fresh_ser = factory.instances[1]
        # Address byte is patched to 0x81 (camera addr 1).
        # Fresh port must receive: stop, zoom-stop, then the retried move.
        assert len(fresh_ser.written) >= 3, (
            f"Expected ≥3 writes on reconnect (stop, zoom-stop, move), got {len(fresh_ser.written)}"
        )

        # Build expected stop bytes with address patched.
        expected_stop = bytes([0x81]) + visca_stop_cmd()[1:]
        expected_zoom_stop = bytes([0x81]) + visca_zoom_stop_cmd()[1:]

        assert fresh_ser.written[0] == expected_stop, (
            f"First write must be stop cmd, got {fresh_ser.written[0].hex()}"
        )
        assert fresh_ser.written[1] == expected_zoom_stop, (
            f"Second write must be zoom-stop cmd, got {fresh_ser.written[1].hex()}"
        )

    def test_halt_sent_even_when_camera_was_stopped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stop is sent on reconnect regardless of prior command (always-safe)."""
        backend, factory = self._make_backend(monkeypatch)

        backend.stop()
        factory.instances[0].should_fail = True
        backend.move_velocity(0.3, 0.0, 0.0)

        assert len(factory.instances) == 2
        fresh_ser = factory.instances[1]

        expected_stop = bytes([0x81]) + visca_stop_cmd()[1:]
        expected_zoom_stop = bytes([0x81]) + visca_zoom_stop_cmd()[1:]

        assert len(fresh_ser.written) >= 3
        assert fresh_ser.written[0] == expected_stop
        assert fresh_ser.written[1] == expected_zoom_stop

    def test_stop_bytes_precede_retried_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pantilt move is always retried AFTER the stop sequence."""
        from autoptz.engine.ptz.base import visca_pantilt_cmd

        backend, factory = self._make_backend(monkeypatch)
        backend.move_velocity(0.5, 0.0, 0.0)
        factory.instances[0].should_fail = True
        backend.move_velocity(0.5, 0.0, 0.0)

        fresh_ser = factory.instances[1]
        # Address-patched bytes.
        expected_stop = bytes([0x81]) + visca_stop_cmd()[1:]
        expected_pantilt = bytes([0x81]) + visca_pantilt_cmd(0.5, 0.0)[1:]

        stop_idx = next((i for i, b in enumerate(fresh_ser.written) if b == expected_stop), None)
        move_idx = next((i for i, b in enumerate(fresh_ser.written) if b == expected_pantilt), None)

        assert stop_idx is not None, "stop cmd not found in fresh serial writes"
        assert move_idx is not None, "pantilt move cmd not found in fresh serial writes"
        assert stop_idx < move_idx, (
            f"stop (idx {stop_idx}) must precede pantilt move (idx {move_idx})"
        )
