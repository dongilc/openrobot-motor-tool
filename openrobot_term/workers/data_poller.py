"""
Periodic COMM_GET_VALUES poller thread.
"""

import time
from PyQt6.QtCore import QThread, pyqtSignal

from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import build_get_values


class DataPoller(QThread):
    """Periodically sends COMM_GET_VALUES requests at the configured rate."""

    tick = pyqtSignal()  # emitted after each request sent (for timing debug)

    def __init__(self, transport: SerialTransport, interval_ms: int = 50):
        super().__init__()
        self._transport = transport
        self._interval_s = interval_ms / 1000.0
        self._running = False

    @property
    def interval_ms(self) -> int:
        return int(self._interval_s * 1000)

    @interval_ms.setter
    def interval_ms(self, ms: int):
        self._interval_s = ms / 1000.0

    def start(self):
        """Start polling. Sets running flag before thread starts to avoid race."""
        self._running = True
        super().start()

    def run(self):
        # Don't set _running here - it's set in start() to avoid race condition
        while self._running:
            if self._transport.is_connected():
                self._transport.send_packet(build_get_values())
                self.tick.emit()
            time.sleep(self._interval_s)

    def stop(self):
        self._running = False
