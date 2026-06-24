"""Tests for PTZ transport auto-reconnect (R4').

No real hardware — socket and serial are replaced by fake implementations
injected via monkeypatch.
"""

from __future__ import annotations

from typing import Any

import pytest

import autoptz.engine.ptz.visca_ip as visca_ip_mod
import autoptz.engine.ptz.visca_usb as visca_usb_mod
from autoptz.engine.ptz.reconnect import ReconnectPolicy
from autoptz.engine.ptz.visca_ip import ViscaIPBackend
from autoptz.engine.ptz.visca_usb import ViscaUSBBackend

# ─────────────────────────────────────────────────────────────────────────────
# ReconnectPolicy unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReconnectPolicy:
    def test_initially_allowed(self) -> None:
        """Policy starts with _next_allowed_t=0 so any 'now' passes the gate."""
        policy = ReconnectPolicy()
        assert policy.should_attempt(0.0) is True
        assert policy.should_attempt(100.0) is True

    def test_failure_blocks_immediate_retry(self) -> None:
        """After the first failure the next attempt must wait at least base_s."""
        policy = ReconnectPolicy(base_s=1.0, cap_s=30.0)
        policy.record_failure(now=0.0)
        # Just-after the failure: still blocked
        assert policy.should_attempt(0.0) is False
        assert policy.should_attempt(0.99) is False

    def test_failure_allows_after_backoff(self) -> None:
        policy = ReconnectPolicy(base_s=1.0, cap_s=30.0)
        policy.record_failure(now=0.0)
        # Exactly at base_s the gate should open
        assert policy.should_attempt(1.0) is True
        assert policy.should_attempt(2.0) is True

    def test_exponential_backoff_values(self) -> None:
        """Delays should double each failure: 1, 2, 4, 8 (capped at 30)."""
        policy = ReconnectPolicy(base_s=1.0, cap_s=30.0)
        t = 0.0
        expected_delays = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
        for expected in expected_delays:
            policy.record_failure(now=t)
            # Gate should open exactly at t+expected
            assert policy.should_attempt(t + expected - 0.001) is False
            assert policy.should_attempt(t + expected) is True
            t += expected  # advance time past the backoff for the next iteration

    def test_cap_limits_backoff(self) -> None:
        """Backoff must never exceed cap_s."""
        policy = ReconnectPolicy(base_s=1.0, cap_s=5.0)
        t = 0.0
        for _ in range(10):
            policy.record_failure(now=t)
            t += 5.0  # advance past any cap
        # After many failures the delay is still capped at cap_s
        policy.record_failure(now=t)
        assert policy.should_attempt(t + 5.0) is True
        assert policy.should_attempt(t + 4.99) is False

    def test_record_success_resets(self) -> None:
        """record_success must reset the attempts counter and unblock immediately."""
        policy = ReconnectPolicy(base_s=1.0, cap_s=30.0)
        policy.record_failure(now=0.0)
        policy.record_failure(now=1.0)
        assert policy.should_attempt(1.0) is False  # still blocked
        policy.record_success()
        assert policy.should_attempt(0.0) is True  # fully reset

    def test_success_resets_exponential_growth(self) -> None:
        """After success the next failure starts back from base_s."""
        policy = ReconnectPolicy(base_s=1.0, cap_s=30.0)
        # Wind up to a large backoff
        t = 0.0
        for _ in range(5):
            policy.record_failure(now=t)
            t += 32.0  # advance well past any delay
        # Reset
        policy.record_success()
        # First failure after reset should use base_s again
        policy.record_failure(now=t)
        assert policy.should_attempt(t + 1.0) is True
        assert policy.should_attempt(t + 0.5) is False


# ─────────────────────────────────────────────────────────────────────────────
# ViscaIPBackend reconnect tests
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


class TestViscaIPReconnect:
    def _make_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[ViscaIPBackend, list[FakeSocket]]:
        """Build a ViscaIPBackend with a monkeypatched create_connection.

        Returns (backend, sockets_created) so tests can inspect/flip the fakes.
        """
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

    def test_initial_connection_made(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor must call create_connection exactly once."""
        backend, sockets = self._make_backend(monkeypatch)
        assert len(sockets) == 1
        assert backend.connected is True

    def test_normal_send_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """move_velocity over a healthy socket sends bytes without reconnecting."""
        backend, sockets = self._make_backend(monkeypatch)
        backend.move_velocity(0.5, 0.0)
        assert len(sockets[0].sent) == 2  # pantilt + zoom

    def test_error_disconnects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSError on sendall marks the backend as disconnected."""
        backend, sockets = self._make_backend(monkeypatch)
        sockets[0].should_fail = True
        # Policy initially allows an attempt → a reconnect socket is created.
        # The reconnect itself succeeds (new fake socket won't fail).
        backend.move_velocity(0.5, 0.0)
        # After reconnect succeeds, connected should be True
        assert backend.connected is True

    def test_reconnects_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After a transport error create_connection is called again."""
        backend, sockets = self._make_backend(monkeypatch)
        sockets[0].should_fail = True
        backend.move_velocity(0.5, 0.0)
        # A second socket should have been created during the reconnect
        assert len(sockets) == 2

    def test_retry_command_sent_on_reconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The failed command is retried on the fresh socket after reconnect."""
        backend, sockets = self._make_backend(monkeypatch)
        sockets[0].should_fail = True
        backend.move_velocity(0.5, 0.0)
        # The fresh socket (sockets[1]) should have received the retry
        assert len(sockets) == 2
        assert len(sockets[1].sent) >= 1

    def test_connected_true_after_reconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """connected property is True after successful reconnect."""
        backend, sockets = self._make_backend(monkeypatch)
        sockets[0].should_fail = True
        backend.move_velocity(0.5, 0.0)
        assert backend.connected is True

    def test_backoff_throttles_second_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second immediate failure after a reconnect is throttled by the policy.

        Uses a fake clock so the test is deterministic on any CI load.
        """
        # Pin the clock at t=0 so both _send calls see the same 'now'.
        fake_now = [0.0]
        monkeypatch.setattr(visca_ip_mod.time, "monotonic", lambda: fake_now[0])

        backend, sockets = self._make_backend(monkeypatch)
        # First failure → reconnect succeeds but we flip the new socket too
        sockets[0].should_fail = True

        def fake_create_connection_fail(
            addr: tuple[str, int], timeout: float | None = None
        ) -> FakeSocket:
            s = FakeSocket()
            s.should_fail = True  # reconnect socket also fails
            sockets.append(s)
            return s

        monkeypatch.setattr(visca_ip_mod.socket, "create_connection", fake_create_connection_fail)

        # First call at t=0: original socket fails → reconnect socket also fails →
        # policy records failure at t=0, sets _next_allowed_t = 1.0 → disconnected
        backend.move_velocity(0.5, 0.0)
        assert backend.connected is False
        sockets_after_first = len(sockets)

        # Second call still at t=0 (< _next_allowed_t=1.0): policy blocks
        backend.move_velocity(0.5, 0.0)
        assert len(sockets) == sockets_after_first  # no new socket created

        # Advance clock past the backoff window → policy now allows a reconnect
        fake_now[0] = 2.0
        backend.move_velocity(0.5, 0.0)
        assert len(sockets) > sockets_after_first  # new reconnect attempt was made

    def test_no_exception_from_move_velocity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """move_velocity must never raise even when the socket stays broken."""
        backend, sockets = self._make_backend(monkeypatch)
        sockets[0].should_fail = True

        def always_fail(addr: tuple[str, int], timeout: float | None = None) -> FakeSocket:
            raise OSError("still down")

        monkeypatch.setattr(visca_ip_mod.socket, "create_connection", always_fail)
        backend.move_velocity(0.5, 0.0)  # must not raise

    def test_no_exception_from_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stop() must never raise even when the socket stays broken."""
        backend, sockets = self._make_backend(monkeypatch)
        sockets[0].should_fail = True

        def always_fail(addr: tuple[str, int], timeout: float | None = None) -> FakeSocket:
            raise OSError("still down")

        monkeypatch.setattr(visca_ip_mod.socket, "create_connection", always_fail)
        backend.stop()  # must not raise

    def test_zoom_send_failure_triggers_reconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """move_velocity: pantilt send succeeds, zoom send fails → reconnect, no exception.

        move_velocity calls _send(pantilt) then _send(zoom).  If the socket dies
        between the two sends (e.g. the camera rebooted mid-frame), the zoom
        _send must trigger a reconnect and must not raise into the caller.
        """
        backend, sockets = self._make_backend(monkeypatch)

        # The initial socket (sockets[0]) succeeds the first sendall (pantilt)
        # then fails on the second (zoom): flip should_fail after the first call.
        send_count = [0]
        _orig_sendall = sockets[0].sendall

        def first_socket_sendall(data: bytes) -> None:
            send_count[0] += 1
            if send_count[0] >= 2:
                raise OSError("simulated mid-move failure on zoom")
            sockets[0].sent.append(data)

        sockets[0].sendall = first_socket_sendall  # type: ignore[method-assign]

        # move_velocity: first _send (pantilt) succeeds, second _send (zoom) fails
        # → _send closes the socket, policy allows reconnect (first failure), reconnects
        backend.move_velocity(0.5, 0.2)  # must not raise
        # A second socket was created during the reconnect
        assert len(sockets) == 2
        assert backend.connected is True  # reconnect succeeded

    def test_connected_false_when_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """connected is False when every attempt fails."""
        backend, sockets = self._make_backend(monkeypatch)
        sockets[0].should_fail = True

        def always_fail(addr: tuple[str, int], timeout: float | None = None) -> FakeSocket:
            raise OSError("still down")

        monkeypatch.setattr(visca_ip_mod.socket, "create_connection", always_fail)
        backend.move_velocity(0.5, 0.0)
        assert backend.connected is False


# ─────────────────────────────────────────────────────────────────────────────
# ViscaUSBBackend reconnect tests
# ─────────────────────────────────────────────────────────────────────────────


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


class TestViscaUSBReconnect:
    def _make_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[ViscaUSBBackend, FakeSerialFactory]:
        factory = FakeSerialFactory()
        monkeypatch.setattr(visca_usb_mod.serial, "Serial", factory)
        backend = ViscaUSBBackend("/dev/ttyUSB0", baud=9600)
        return backend, factory

    def test_initial_open_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor must open the serial port exactly once."""
        backend, factory = self._make_backend(monkeypatch)
        assert len(factory.instances) == 1
        assert backend.connected is True

    def test_normal_write_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """move_velocity over a healthy port writes bytes without reconnecting."""
        backend, factory = self._make_backend(monkeypatch)
        backend.move_velocity(0.5, 0.0)
        assert len(factory.instances[0].written) == 2  # pantilt + zoom

    def test_error_triggers_reconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SerialException causes a reconnect (second Serial() call)."""
        backend, factory = self._make_backend(monkeypatch)
        factory.instances[0].should_fail = True
        backend.move_velocity(0.5, 0.0)
        assert len(factory.instances) == 2

    def test_retry_on_fresh_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The command is retried on the fresh serial port after reconnect."""
        backend, factory = self._make_backend(monkeypatch)
        factory.instances[0].should_fail = True
        backend.move_velocity(0.5, 0.0)
        assert len(factory.instances) == 2
        assert len(factory.instances[1].written) >= 1

    def test_connected_true_after_reconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend, factory = self._make_backend(monkeypatch)
        factory.instances[0].should_fail = True
        backend.move_velocity(0.5, 0.0)
        assert backend.connected is True

    def test_backoff_throttles_second_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backoff blocks a second reconnect attempt immediately after a failure.

        Uses a fake clock so the test is deterministic on any CI load.
        """
        # Pin the clock at t=0 so both _send calls see the same 'now'.
        fake_now = [0.0]
        monkeypatch.setattr(visca_usb_mod.time, "monotonic", lambda: fake_now[0])

        backend, factory = self._make_backend(monkeypatch)
        factory.instances[0].should_fail = True

        # Override factory so second Serial() also fails
        call_count = [0]

        def failing_factory(port: str, baud: int, timeout: float = 0.1) -> FakeSerial:
            call_count[0] += 1
            s = FakeSerial(port, baud, timeout)
            s.should_fail = True
            factory.instances.append(s)
            return s

        monkeypatch.setattr(visca_usb_mod.serial, "Serial", failing_factory)

        # First call at t=0: write fails → reconnect also fails →
        # policy records failure at t=0, sets _next_allowed_t = 1.0 → disconnected
        backend.move_velocity(0.5, 0.0)
        assert backend.connected is False
        calls_after_first = call_count[0]

        # Second call still at t=0 (< _next_allowed_t=1.0): policy blocks
        backend.move_velocity(0.5, 0.0)
        assert call_count[0] == calls_after_first  # no new Serial() call

        # Advance clock past the backoff window → policy now allows a reconnect
        fake_now[0] = 2.0
        backend.move_velocity(0.5, 0.0)
        assert call_count[0] > calls_after_first  # new Serial() call was made

    def test_no_exception_from_move_velocity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """move_velocity must never raise even with a broken port."""
        backend, factory = self._make_backend(monkeypatch)
        factory.instances[0].should_fail = True

        def always_fail(port: str, baud: int, timeout: float = 0.1) -> Any:
            raise visca_usb_mod.serial.SerialException("still down")

        monkeypatch.setattr(visca_usb_mod.serial, "Serial", always_fail)
        backend.move_velocity(0.5, 0.0)  # must not raise

    def test_no_exception_from_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stop() must never raise even with a broken port."""
        backend, factory = self._make_backend(monkeypatch)
        factory.instances[0].should_fail = True

        def always_fail(port: str, baud: int, timeout: float = 0.1) -> Any:
            raise visca_usb_mod.serial.SerialException("still down")

        monkeypatch.setattr(visca_usb_mod.serial, "Serial", always_fail)
        backend.stop()  # must not raise

    def test_connected_false_when_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """connected is False when every Serial() attempt fails."""
        backend, factory = self._make_backend(monkeypatch)
        factory.instances[0].should_fail = True

        def always_fail(port: str, baud: int, timeout: float = 0.1) -> Any:
            raise visca_usb_mod.serial.SerialException("still down")

        monkeypatch.setattr(visca_usb_mod.serial, "Serial", always_fail)
        backend.move_velocity(0.5, 0.0)
        assert backend.connected is False
