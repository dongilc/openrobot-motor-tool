"""
Thread-safe serial transport with dual-mode support (VESC packets + plain text).
"""

import time
import threading

import serial
import serial.tools.list_ports
from PyQt6.QtCore import QObject, pyqtSignal

from .packet import VescPacket, PacketAssembler


def list_serial_ports() -> list:
    return [p.device for p in serial.tools.list_ports.comports()]


class SerialTransport(QObject):
    """
    Centralized serial I/O with thread-safe writes and a background reader.
    Emits Qt signals for both VESC binary packets and plain text lines.
    """

    packet_received = pyqtSignal(bytes)      # decoded VESC payload
    text_received = pyqtSignal(str)          # plain text line
    connection_lost = pyqtSignal(str)        # error reason
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    # Debug signals for raw UART data
    debug_tx = pyqtSignal(bytes)             # raw TX bytes
    debug_rx = pyqtSignal(bytes)             # raw RX bytes

    def __init__(self):
        super().__init__()
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self._assembler = PacketAssembler()
        self._reader_thread: threading.Thread | None = None
        self._running = False
        # Byte counters for bitrate calculation
        self._rx_bytes = 0
        self._tx_bytes = 0
        self._rx_bytes_lock = threading.Lock()

    @property
    def port_name(self) -> str:
        if self._ser and self._ser.is_open:
            return self._ser.port
        return ""

    @property
    def baudrate(self) -> int:
        if self._ser and self._ser.is_open:
            return self._ser.baudrate
        return 0

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def take_byte_counts(self) -> tuple[int, int]:
        """Return (rx_bytes, tx_bytes) since last call and reset counters."""
        with self._rx_bytes_lock:
            rx, tx = self._rx_bytes, self._tx_bytes
            self._rx_bytes = 0
            self._tx_bytes = 0
        return rx, tx

    def connect(self, port: str, baud: int) -> None:
        """Open serial port and start reader thread."""
        self.disconnect()
        self._ser = serial.Serial(port, baud, timeout=0.1)
        self._assembler = PacketAssembler()
        self._running = True
        with self._rx_bytes_lock:
            self._rx_bytes = 0
            self._tx_bytes = 0
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="SerialReader"
        )
        self._reader_thread.start()
        self.connected.emit()

    def disconnect(self) -> None:
        """Stop reader and close serial port."""
        self._running = False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None

        with self._lock:
            if self._ser:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
        self.disconnected.emit()

    def send_packet(self, payload: bytes) -> None:
        """Send a VESC-framed packet. Thread-safe."""
        with self._lock:
            if self._ser and self._ser.is_open:
                data = VescPacket.encode(payload)
                self._ser.write(data)
                self._ser.flush()
                with self._rx_bytes_lock:
                    self._tx_bytes += len(data)
                self.debug_tx.emit(data)

    def send_raw(self, data: bytes) -> None:
        """Send raw bytes (for text commands or firmware upload). Thread-safe."""
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.write(data)
                self._ser.flush()
                with self._rx_bytes_lock:
                    self._tx_bytes += len(data)
                self.debug_tx.emit(data)

    def get_serial(self) -> serial.Serial | None:
        """Direct access to serial object (for firmware uploader). Use with caution."""
        return self._ser

    def _reader_loop(self):
        """Background thread: read bytes, separate packets from text."""
        while self._running:
            try:
                if not self._ser or not self._ser.is_open:
                    time.sleep(0.05)
                    continue

                n = self._ser.in_waiting
                if n:
                    raw = self._ser.read(n)
                    with self._rx_bytes_lock:
                        self._rx_bytes += len(raw)
                    self.debug_rx.emit(raw)
                    packets = self._assembler.feed(raw)
                    for p in packets:
                        self.packet_received.emit(p)
                    for line in self._assembler.get_text_lines():
                        self.text_received.emit(line)

                time.sleep(0.002)

            except (serial.SerialException, OSError, PermissionError) as e:
                self._running = False
                self.connection_lost.emit(str(e))
                return
            except Exception as e:
                self._running = False
                self.connection_lost.emit(f"Unexpected: {e}")
                return
