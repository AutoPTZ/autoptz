"""UDP transport coverage for VISCA-over-IP cameras."""

from __future__ import annotations

import socket
import struct

import pytest

import autoptz.engine.ptz.visca_ip as visca_ip_mod
from autoptz.engine.ptz.base import visca_stop_cmd
from autoptz.engine.ptz.visca_ip import ViscaIPBackend


class FakeDatagramSocket:
    def __init__(self, responses: list[bytes] | None = None) -> None:
        self.peer: tuple[str, int] | None = None
        self.sent: list[bytes] = []
        self.closed = False
        self.responses = list(responses or [])

    def settimeout(self, timeout: float) -> None:
        pass

    def connect(self, peer: tuple[str, int]) -> None:
        self.peer = peer

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, size: int) -> bytes:
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def test_udp_opens_datagram_socket_and_sends_visca(monkeypatch: pytest.MonkeyPatch) -> None:
    sock = FakeDatagramSocket()
    calls: list[tuple[int, int]] = []

    def fake_socket(family: int, kind: int) -> FakeDatagramSocket:
        calls.append((family, kind))
        return sock

    monkeypatch.setattr(visca_ip_mod.socket, "socket", fake_socket)

    backend = ViscaIPBackend("192.0.2.20", port=52381, transport="udp")
    backend.stop()

    assert calls == [(socket.AF_INET, socket.SOCK_DGRAM)]
    assert sock.peer == ("192.0.2.20", 52381)
    assert sock.sent[0] == visca_stop_cmd()


def test_tcp_remains_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[tuple[str, int], float | None]] = []
    sock = FakeDatagramSocket()

    def fake_create_connection(
        peer: tuple[str, int], timeout: float | None = None
    ) -> FakeDatagramSocket:
        seen.append((peer, timeout))
        return sock

    monkeypatch.setattr(visca_ip_mod.socket, "create_connection", fake_create_connection)

    ViscaIPBackend("192.0.2.21", port=5678)

    assert seen == [(("192.0.2.21", 5678), 2.0)]


def test_sony_udp_query_reads_header_and_payload_from_one_datagram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytes([0x90, 0x50, 0x00, 0x01, 0x02, 0x03, 0xFF])
    response = struct.pack(">HHI", 0x0111, len(payload), 7) + payload
    sock = FakeDatagramSocket([response])
    monkeypatch.setattr(visca_ip_mod.socket, "socket", lambda *args: sock)

    backend = ViscaIPBackend("192.0.2.20", mode="sony", transport="udp")

    assert backend._query(bytes([0x81, 0x09, 0x04, 0x47, 0xFF]), len(payload)) == payload


def test_rejects_unknown_transport() -> None:
    with pytest.raises(ValueError, match="transport"):
        ViscaIPBackend("192.0.2.22", transport="sctp")
