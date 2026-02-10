"""
VESC binary packet framing: encode, decode, CRC16, and dual-mode stream assembly.

Packet format:
  Short (payload <= 255 bytes):
    [0x02] [len:1] [payload:N] [crc_hi] [crc_lo] [0x03]

  Long (payload 256-65535 bytes):
    [0x03] [len_hi:1] [len_lo:1] [payload:N] [crc_hi] [crc_lo] [0x03]
"""

import struct
from binascii import crc_hqx


class VescPacket:
    START_SHORT = 0x02
    START_LONG = 0x03
    END_BYTE = 0x03

    @staticmethod
    def encode(payload: bytes) -> bytes:
        """Wrap payload into a VESC framed packet."""
        crc = crc_hqx(payload, 0)
        if len(payload) <= 255:
            return (
                bytes([0x02, len(payload)])
                + payload
                + struct.pack(">H", crc)
                + bytes([0x03])
            )
        else:
            return (
                bytes([0x03])
                + struct.pack(">H", len(payload))
                + payload
                + struct.pack(">H", crc)
                + bytes([0x03])
            )

    @staticmethod
    def decode(data: bytes) -> tuple:
        """
        Attempt to decode one packet from buffer.
        Returns (payload, bytes_consumed) or (None, 0) if incomplete/invalid.
        """
        if len(data) < 2:
            return None, 0

        start = data[0]

        if start == 0x02:
            # Short packet
            if len(data) < 2:
                return None, 0
            length = data[1]
            total = 2 + length + 2 + 1  # start+len + payload + crc + end
            if len(data) < total:
                return None, 0
            payload = data[2 : 2 + length]
            crc_received = struct.unpack_from(">H", data, 2 + length)[0]
            end = data[2 + length + 2]
            if end != 0x03:
                return None, 0
            crc_calc = crc_hqx(bytes(payload), 0)
            if crc_calc != crc_received:
                return None, 0
            return bytes(payload), total

        elif start == 0x03:
            # Long packet
            if len(data) < 3:
                return None, 0
            length = struct.unpack_from(">H", data, 1)[0]
            if length <= 255:
                # This is likely an end byte from a previous packet, not a long start
                return None, 0
            total = 3 + length + 2 + 1  # start+len(2) + payload + crc + end
            if len(data) < total:
                return None, 0
            payload = data[3 : 3 + length]
            crc_received = struct.unpack_from(">H", data, 3 + length)[0]
            end = data[3 + length + 2]
            if end != 0x03:
                return None, 0
            crc_calc = crc_hqx(bytes(payload), 0)
            if crc_calc != crc_received:
                return None, 0
            return bytes(payload), total

        return None, 0


class PacketAssembler:
    """
    Accumulates incoming bytes and yields complete VESC packets.
    Also extracts plain-text lines for terminal mode (dual-mode stream).
    """

    def __init__(self):
        self._buffer = bytearray()
        self._text_buffer = ""

    def feed(self, data: bytes) -> list:
        """
        Feed raw bytes. Returns list of decoded VESC payloads.
        Any non-packet bytes are accumulated as text (retrieve via get_text_lines).
        """
        self._buffer.extend(data)
        packets = []

        while len(self._buffer) > 0:
            # Try to find a packet start byte
            idx = -1
            for i, b in enumerate(self._buffer):
                if b == 0x02 or b == 0x03:
                    idx = i
                    break

            if idx < 0:
                # No potential packet start — everything is text
                self._text_buffer += self._buffer.decode("utf-8", errors="ignore")
                self._buffer.clear()
                break

            if idx > 0:
                # Bytes before the start byte are text
                text_bytes = self._buffer[:idx]
                self._text_buffer += text_bytes.decode("utf-8", errors="ignore")
                self._buffer = self._buffer[idx:]

            # Try to decode a packet at the current position
            payload, consumed = VescPacket.decode(bytes(self._buffer))
            if payload is not None:
                packets.append(payload)
                self._buffer = self._buffer[consumed:]
            else:
                # Could be incomplete packet — wait for more data
                # But if buffer is large enough that it should have decoded, skip this byte
                if len(self._buffer) > 6:
                    # Check if this is truly a start byte by peeking at length
                    if self._buffer[0] == 0x02:
                        expected_len = self._buffer[1]
                        expected_total = 2 + expected_len + 3
                        if len(self._buffer) >= expected_total:
                            # Full data present but decode failed (bad CRC) — skip byte
                            self._text_buffer += chr(self._buffer[0])
                            self._buffer = self._buffer[1:]
                            continue
                    elif self._buffer[0] == 0x03:
                        if len(self._buffer) >= 3:
                            maybe_len = struct.unpack_from(">H", self._buffer, 1)[0]
                            if maybe_len <= 255:
                                # Not a long packet start — treat as text
                                self._text_buffer += chr(self._buffer[0])
                                self._buffer = self._buffer[1:]
                                continue
                # Wait for more data
                break

        return packets

    def get_text_lines(self) -> list:
        """Extract complete text lines from buffer. Keeps partial lines for next call."""
        lines = []
        while "\n" in self._text_buffer:
            line, self._text_buffer = self._text_buffer.split("\n", 1)
            lines.append(line + "\n")
        return lines

    def flush_text(self) -> str:
        """Flush any remaining text (even without newline)."""
        text = self._text_buffer
        self._text_buffer = ""
        return text
