"""
Thread-safe PCAN transport with Qt signals — mirrors SerialTransport pattern.
Gracefully degrades if PCAN DLL is not installed.
"""

import time
import threading

from PyQt6.QtCore import QObject, pyqtSignal

from .can_commands import (
    CAN_HEADER_ID, RmdCommand, STATUS_RETURN_COMMANDS,
    parse_status, parse_status3, parse_multi_turn_angle, format_response_log,
)

# Attempt to import PCANBasic — graceful fallback if DLL not available
PCAN_AVAILABLE = False
try:
    from .PCANBasic import (
        PCANBasic, TPCANMsg, TPCANTimestamp,
        PCAN_USBBUS1, PCAN_BAUD_1M, PCAN_ERROR_OK, PCAN_ERROR_QRCVEMPTY,
        PCAN_MESSAGE_STANDARD,
    )
    # Verify the DLL actually loads
    _test = PCANBasic()
    PCAN_AVAILABLE = True
    del _test
except Exception:
    PCAN_AVAILABLE = False


class PcanTransport(QObject):
    """
    PCAN-USB CAN bus transport layer.
    Emits Qt signals for received frames and parsed RMD status.
    """

    frame_received = pyqtSignal(int, int, float, list)  # can_id, dlc, timestamp, data_list
    status_received = pyqtSignal(object)                 # RmdStatus (from 0x9C polling only)
    cmd_status_received = pyqtSignal(object)             # RmdStatus (from 0xA1-A4 control responses)
    status3_received = pyqtSignal(object)                # RmdStatus3 (0x9D)
    multiturn_received = pyqtSignal(float)               # degrees (0x92)
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._pc = PCANBasic() if PCAN_AVAILABLE else None
        self._lock = threading.Lock()
        self._connected = False
        self._motor_id = 1
        self._reader_thread = None
        self._running = False
        self._last_msg = None

        # Periodic resend state
        self._periodic_running = False
        self._periodic_thread = None
        self._periodic_freq = 10  # Hz

        # Frame counters (reset each read via take_frame_counts)
        self._tx_frames = 0
        self._rx_frames = 0
        self._counter_lock = threading.Lock()

        # Raw message buffer for display
        self._rxmsg = []
        self._rxmsg_max = 100

    @property
    def available(self) -> bool:
        return PCAN_AVAILABLE

    def is_connected(self) -> bool:
        return self._connected

    def set_motor_id(self, motor_id: int):
        self._motor_id = motor_id

    @property
    def motor_id(self) -> int:
        return self._motor_id

    def connect(self) -> bool:
        """Open PCAN-USB channel at 1 Mbps. Returns True on success."""
        if not PCAN_AVAILABLE:
            self.log_message.emit("PCAN DLL not available")
            return False

        self.disconnect()

        # Force-release any stale PCAN handle (e.g. from previous crash)
        try:
            self._pc.Uninitialize(PCAN_USBBUS1)
        except Exception:
            pass

        result = self._pc.Initialize(PCAN_USBBUS1, PCAN_BAUD_1M)
        if result != PCAN_ERROR_OK:
            err = self._pc.GetErrorText(result)
            self.log_message.emit(f"[PCAN] Init failed: {err[1]}")
            return False

        self._last_msg = TPCANMsg()
        self._rxmsg.clear()
        self._connected = True

        # Start reader thread
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="PCANReader"
        )
        self._reader_thread.start()

        self.log_message.emit("[PCAN] Connected (1 Mbps)")
        self.connected.emit()
        return True

    def disconnect(self):
        """Close PCAN channel and stop threads."""
        self._running = False
        self.stop_periodic()

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None

        if self._connected and self._pc:
            with self._lock:
                self._pc.Uninitialize(PCAN_USBBUS1)
            self._connected = False
            self.log_message.emit("[PCAN] Disconnected")
            self.disconnected.emit()

    def send_frame(self, data: bytes) -> bool:
        """Send a CAN frame to the current motor_id. Thread-safe."""
        if not self._connected:
            return False

        with self._lock:
            msg = TPCANMsg()
            msg.MSGTYPE = PCAN_MESSAGE_STANDARD
            msg.ID = CAN_HEADER_ID + self._motor_id
            msg.LEN = 8
            for i in range(min(len(data), 8)):
                msg.DATA[i] = data[i]

            self._last_msg = msg
            result = self._pc.Write(PCAN_USBBUS1, msg)

        if result != PCAN_ERROR_OK:
            self.log_message.emit(f"[PCAN] Send error: 0x{result:02X}")
            return False
        with self._counter_lock:
            self._tx_frames += 1
        return True

    def start_periodic(self):
        """Start periodic resend of last message."""
        if self._periodic_running:
            return
        self._periodic_running = True
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop, daemon=True, name="PCANPeriodic"
        )
        self._periodic_thread.start()

    def stop_periodic(self):
        """Stop periodic resend."""
        self._periodic_running = False
        if self._periodic_thread and self._periodic_thread.is_alive():
            self._periodic_thread.join(timeout=1.0)
        self._periodic_thread = None

    @property
    def periodic_running(self) -> bool:
        return self._periodic_running

    @property
    def periodic_freq(self) -> int:
        return self._periodic_freq

    @periodic_freq.setter
    def periodic_freq(self, hz: int):
        self._periodic_freq = max(1, hz)

    def get_rxmsg(self) -> list:
        """Return buffered raw messages and clear buffer."""
        msgs = list(self._rxmsg)
        if len(self._rxmsg) >= self._rxmsg_max:
            self._rxmsg.clear()
        return msgs

    def clear_rxmsg(self):
        self._rxmsg.clear()

    def take_frame_counts(self) -> tuple[int, int]:
        """Return (rx_frames, tx_frames) since last call and reset counters."""
        with self._counter_lock:
            rx, tx = self._rx_frames, self._tx_frames
            self._rx_frames = 0
            self._tx_frames = 0
        return rx, tx

    # ── Internal threads ──

    def _reader_loop(self):
        """Background thread: read all pending CAN frames."""
        while self._running:
            if not self._connected:
                time.sleep(0.05)
                continue
            try:
                self._read_messages()
            except Exception:
                pass
            time.sleep(0.001)

    def _read_messages(self):
        """Read all pending messages from PCAN receive queue."""
        while True:
            with self._lock:
                result = self._pc.Read(PCAN_USBBUS1)

            if result[0] == PCAN_ERROR_QRCVEMPTY:
                break
            if result[0] != PCAN_ERROR_OK:
                break

            self._process_message(result[1], result[2])

    def _process_message(self, msg, ts):
        """Process a received CAN message."""
        with self._counter_lock:
            self._rx_frames += 1

        # Compute timestamp in seconds
        timestamp = (ts.micros + 1000 * ts.millis +
                     0x100000000 * 1000 * ts.millis_overflow) / 1_000_000.0

        data_list = [msg.DATA[i] for i in range(min(msg.LEN, 8))]

        # Buffer for raw display
        if len(self._rxmsg) < self._rxmsg_max:
            self._rxmsg.append({
                'id': msg.ID,
                'dlc': msg.LEN,
                'timestamp': timestamp,
                'data': data_list,
            })

        # Emit raw frame signal
        self.frame_received.emit(msg.ID, msg.LEN, timestamp, data_list)

        # Parse RMD protocol response (standard frame only)
        if msg.MSGTYPE in (0x00, 0x02) and msg.LEN == 8:
            cmd = data_list[0]

            if cmd == RmdCommand.READ_MOTOR_STATUS_2:
                # 0x9C polling response → status_received (for graphs)
                try:
                    status = parse_status(data_list)
                    self.status_received.emit(status)
                except Exception:
                    pass
            elif cmd in STATUS_RETURN_COMMANDS:
                # 0xA1-A4 control command responses → separate signal
                try:
                    status = parse_status(data_list)
                    self.cmd_status_received.emit(status)
                except Exception:
                    pass
            elif cmd == RmdCommand.READ_MOTOR_STATUS_3:
                try:
                    status3 = parse_status3(data_list)
                    self.status3_received.emit(status3)
                except Exception:
                    pass
            elif cmd == RmdCommand.READ_MULTI_TURN_ANGLE:
                try:
                    angle = parse_multi_turn_angle(data_list)
                    self.multiturn_received.emit(angle)
                except Exception:
                    pass
                log_text = format_response_log(cmd, data_list)
                self.log_message.emit(f"[CAN RX] {log_text}")
            else:
                # Log non-status responses (encoder, PID, fault, etc.)
                log_text = format_response_log(cmd, data_list)
                self.log_message.emit(f"[CAN RX] {log_text}")

    def _periodic_loop(self):
        """Periodically resend the last CAN message."""
        while self._periodic_running and self._connected:
            if self._last_msg is not None:
                with self._lock:
                    result = self._pc.Write(PCAN_USBBUS1, self._last_msg)
                if result != PCAN_ERROR_OK:
                    self.log_message.emit(f"[PCAN] Periodic send error: 0x{result:02X}")
                else:
                    with self._counter_lock:
                        self._tx_frames += 1
            time.sleep(1.0 / self._periodic_freq)
