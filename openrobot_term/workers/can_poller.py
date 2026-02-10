"""
Periodic CAN status poller thread — mirrors DataPoller pattern.
Sends READ_MOTOR_STATUS_2 (0x9C) + READ_MOTOR_STATUS_3 (0x9D) at configurable rate.
Uses perf_counter busy-wait for sub-ms accuracy at high rates (>=200 Hz).
"""

import time
from PyQt6.QtCore import QThread, pyqtSignal

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import build_read_motor_status_2, build_read_motor_status_3

class CanPoller(QThread):
    """Periodically sends READ_MOTOR_STATUS_2 requests for realtime plotting."""

    tick = pyqtSignal()

    def __init__(self, can_transport: PcanTransport, interval_ms: int = 100):
        super().__init__()
        self._transport = can_transport
        self._interval_s = interval_ms / 1000.0
        self._running = False

    @property
    def interval_ms(self) -> int:
        return int(self._interval_s * 1000)

    @interval_ms.setter
    def interval_ms(self, ms: int):
        self._interval_s = ms / 1000.0

    def start(self):
        self._running = True
        super().start()

    def run(self):
        frame_9c = build_read_motor_status_2()
        frame_9d = build_read_motor_status_3()
        perf = time.perf_counter
        sleep = time.sleep
        t_next = perf()

        while self._running:
            if self._transport.is_connected():
                try:
                    self._transport.send_frame(frame_9d)   # 0x9D first so response arrives before 0x9C
                    self._transport.send_frame(frame_9c)
                except Exception:
                    break  # transport error — exit cleanly
                if self._running:
                    self.tick.emit()

            t_next += self._interval_s
            remaining = t_next - perf()

            if remaining <= 0:
                # Fell behind — reset deadline
                t_next = perf()
            else:
                # Sleep most of the interval, then yield-spin the last bit
                if remaining > 0.002:
                    sleep(remaining - 0.002)
                while perf() < t_next and self._running:
                    sleep(0.0002)  # 200µs yield — prevents CPU 100%

    def stop(self):
        self._running = False
