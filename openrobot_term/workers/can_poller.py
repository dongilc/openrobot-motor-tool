"""
Periodic CAN status poller thread — mirrors DataPoller pattern.
Sends READ_MOTOR_STATUS_2 (0x9C) + READ_MOTOR_STATUS_3 (0x9D) at configurable rate.
Uses perf_counter busy-wait for sub-ms accuracy at high rates (>=200 Hz).

Supports multi-device polling: when motor_ids is provided, polls all IDs
each tick using send_frame_to(). Otherwise falls back to send_frame()
(single target selected in connection_bar).
"""

import time
from PyQt6.QtCore import QThread, pyqtSignal

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import build_read_motor_status_2, build_read_motor_status_3

class CanPoller(QThread):
    """Periodically sends READ_MOTOR_STATUS_2 requests for realtime plotting."""

    tick = pyqtSignal()

    def __init__(self, can_transport: PcanTransport, interval_ms: int = 100,
                 motor_ids: list[int] | None = None):
        super().__init__()
        self._transport = can_transport
        self._interval_s = interval_ms / 1000.0
        self._running = False
        self._motor_ids = list(motor_ids) if motor_ids else []

    @property
    def interval_ms(self) -> int:
        return int(self._interval_s * 1000)

    @interval_ms.setter
    def interval_ms(self, ms: int):
        self._interval_s = ms / 1000.0

    def set_motor_ids(self, ids: list[int]):
        """Update the list of motor IDs to poll (thread-safe: list replace is atomic)."""
        self._motor_ids = list(ids)

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
                    ids = self._motor_ids
                    if ids:
                        # Multi-device: poll all discovered IDs
                        for mid in ids:
                            self._transport.send_frame_to(mid, frame_9d)
                            self._transport.send_frame_to(mid, frame_9c)
                    else:
                        # Single target (connection_bar selected ID)
                        self._transport.send_frame(frame_9d)
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
