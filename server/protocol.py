"""WTK1 UDP protocol helpers.

This module keeps packet parsing and echo packet construction out of the
FastAPI application layer. The protocol is binary and fixed-width, so keeping it
isolated makes route and server code easier to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass

MAGIC = b"WTK1"
HEADER_LEN = 34
DEVICE_LEN = 16
SERVER_DEVICE = b"server-echo"

PKT_TYPES = {
    1: "register",
    2: "channel",
    3: "ptt_start",
    4: "audio",
    5: "ptt_stop",
    6: "heartbeat",
}


@dataclass
class Packet:
    """Parsed WTK1 packet."""

    packet_type: int
    channel: int
    seq: int
    timestamp_ms: int
    device: str
    payload: bytes


def read_u16(data: bytes, offset: int) -> int:
    """Read a little-endian unsigned 16-bit integer."""
    return data[offset] | (data[offset + 1] << 8)


def read_u32(data: bytes, offset: int) -> int:
    """Read a little-endian unsigned 32-bit integer."""
    return (
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24)
    )


def parse_packet(data: bytes) -> Packet | None:
    """Parse one WTK1 datagram.

    Returns ``None`` for non-WTK1 input or truncated packets so the UDP server
    can log and ignore bad data without raising.
    """
    if len(data) < HEADER_LEN or data[:4] != MAGIC:
        return None

    header_len = data[5]
    payload_len = read_u16(data, 32)
    if header_len != HEADER_LEN or len(data) < header_len + payload_len:
        return None

    device_raw = data[16:32].split(b"\x00", 1)[0]
    return Packet(
        packet_type=data[4],
        channel=read_u16(data, 6),
        seq=read_u32(data, 8),
        timestamp_ms=read_u32(data, 12),
        device=device_raw.decode("utf-8", errors="replace"),
        payload=data[header_len : header_len + payload_len],
    )


def make_server_echo(data: bytes) -> bytes:
    """Return an echo packet with the device field replaced by the server id."""
    out = bytearray(data)
    out[16:32] = b"\x00" * DEVICE_LEN
    out[16 : 16 + len(SERVER_DEVICE)] = SERVER_DEVICE
    return bytes(out)
