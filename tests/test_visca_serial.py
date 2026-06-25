"""VISCA-over-USB serial auto-discovery.

A USB PTZ camera presents a UVC *video* device plus a companion USB-serial
control port (e.g. ``/dev/cu.usbserial-XXXX``) that speaks VISCA.  These tests
cover the safe, non-moving discovery used to auto-wire such a camera:

* ``is_visca_reply`` — recognising a VISCA completion frame.
* ``probe_visca_baud`` — sending a (non-moving) version inquiry and detecting
  the baud the camera actually answers on.
* ``candidate_ports`` — filtering enumerated serial ports down to USB-serial
  adapters (never Bluetooth / debug / virtual ports).
* ``discover_visca_usb`` — scanning candidates and returning ``(port, baud)``.

Everything is mocked — no real serial port is opened.
"""

from __future__ import annotations

from autoptz.engine.ptz import visca_serial

# A real CAM_VersionInq completion captured from hardware (vendor 0x0001 …).
_REAL_REPLY = bytes([0x90, 0x50, 0x00, 0x01, 0x05, 0x04, 0x01, 0x04, 0x02, 0xFF])


def _fake_serial(good: dict[str, int] | None = None, busy: set[str] | None = None):
    """Build a ``serial.Serial`` stand-in.

    ``good`` maps ``port -> baud`` that replies with a VISCA frame; opening a
    port in ``busy`` raises (already in use); every other (port, baud) opens but
    stays silent.
    """
    good = good or {}
    busy = busy or set()

    class _FakeSerial:
        def __init__(self, port, baud, timeout=None):
            if port in busy:
                raise OSError(f"{port} busy")
            self.port = port
            self.baud = baud
            self._out = _REAL_REPLY if good.get(port) == baud else b""

        def reset_input_buffer(self):
            pass

        def write(self, data):
            pass

        @property
        def in_waiting(self):
            return len(self._out)

        def read(self, n):
            chunk, self._out = self._out[:n], self._out[n:]
            return chunk

        def close(self):
            pass

    return _FakeSerial


class _Port:
    """Minimal ``serial.tools.list_ports`` ListPortInfo stand-in."""

    def __init__(self, device: str) -> None:
        self.device = device


# ─────────────────────────────────────────────────────────────────────────────
# is_visca_reply
# ─────────────────────────────────────────────────────────────────────────────


class TestIsViscaReply:
    def test_accepts_real_completion_frame(self) -> None:
        assert visca_serial.is_visca_reply(_REAL_REPLY) is True

    def test_rejects_empty(self) -> None:
        assert visca_serial.is_visca_reply(b"") is False

    def test_rejects_missing_terminator(self) -> None:
        assert visca_serial.is_visca_reply(bytes([0x90, 0x50, 0x00])) is False

    def test_rejects_non_camera_address_byte(self) -> None:
        # 0x80 is not a valid VISCA reply address (must be 0x90..0xF0).
        assert visca_serial.is_visca_reply(bytes([0x80, 0x50, 0xFF])) is False


# ─────────────────────────────────────────────────────────────────────────────
# probe_visca_baud
# ─────────────────────────────────────────────────────────────────────────────


class TestProbeViscaBaud:
    def test_returns_baud_that_replies(self, monkeypatch) -> None:
        monkeypatch.setattr(
            visca_serial, "serial", type("S", (), {"Serial": _fake_serial({"/dev/x": 115200})})
        )
        baud = visca_serial.probe_visca_baud("/dev/x", (9600, 115200, 38400), settle_s=0.0)
        assert baud == 115200

    def test_tries_bauds_in_order_and_short_circuits(self, monkeypatch) -> None:
        monkeypatch.setattr(
            visca_serial, "serial", type("S", (), {"Serial": _fake_serial({"/dev/x": 9600})})
        )
        baud = visca_serial.probe_visca_baud("/dev/x", (9600, 115200, 38400), settle_s=0.0)
        assert baud == 9600

    def test_returns_none_when_silent(self, monkeypatch) -> None:
        monkeypatch.setattr(visca_serial, "serial", type("S", (), {"Serial": _fake_serial({})}))
        assert visca_serial.probe_visca_baud("/dev/x", (9600, 115200), settle_s=0.0) is None

    def test_returns_none_when_port_busy(self, monkeypatch) -> None:
        monkeypatch.setattr(
            visca_serial, "serial", type("S", (), {"Serial": _fake_serial({}, busy={"/dev/x"})})
        )
        assert visca_serial.probe_visca_baud("/dev/x", (9600, 115200), settle_s=0.0) is None


# ─────────────────────────────────────────────────────────────────────────────
# candidate_ports — filter to USB-serial adapters only
# ─────────────────────────────────────────────────────────────────────────────


class TestCandidatePorts:
    def test_includes_usbserial_excludes_bluetooth_and_debug(self, monkeypatch) -> None:
        ports = [
            _Port("/dev/cu.Bluetooth-Incoming-Port"),
            _Port("/dev/cu.debug-console"),
            _Port("/dev/cu.usbserial-21310"),
            _Port("/dev/cu.usbmodem1234"),
        ]
        monkeypatch.setattr(visca_serial.list_ports, "comports", lambda: ports)
        got = visca_serial.candidate_ports()
        assert "/dev/cu.usbserial-21310" in got
        assert "/dev/cu.usbmodem1234" in got
        assert "/dev/cu.Bluetooth-Incoming-Port" not in got
        assert "/dev/cu.debug-console" not in got


# ─────────────────────────────────────────────────────────────────────────────
# discover_visca_usb — scan candidates, return (port, baud)
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscoverViscaUsb:
    def test_returns_first_visca_port_and_baud(self, monkeypatch) -> None:
        monkeypatch.setattr(
            visca_serial,
            "candidate_ports",
            lambda: ["/dev/cu.usbserial-1", "/dev/cu.usbserial-2"],
        )
        monkeypatch.setattr(
            visca_serial,
            "serial",
            type("S", (), {"Serial": _fake_serial({"/dev/cu.usbserial-2": 115200})}),
        )
        assert visca_serial.discover_visca_usb(bauds=(9600, 115200), settle_s=0.0) == (
            "/dev/cu.usbserial-2",
            115200,
        )

    def test_returns_none_when_no_candidates(self, monkeypatch) -> None:
        monkeypatch.setattr(visca_serial, "candidate_ports", lambda: [])
        assert visca_serial.discover_visca_usb(settle_s=0.0) is None

    def test_returns_none_when_no_port_speaks_visca(self, monkeypatch) -> None:
        monkeypatch.setattr(visca_serial, "candidate_ports", lambda: ["/dev/cu.usbserial-1"])
        monkeypatch.setattr(visca_serial, "serial", type("S", (), {"Serial": _fake_serial({})}))
        assert visca_serial.discover_visca_usb(bauds=(9600, 115200), settle_s=0.0) is None
