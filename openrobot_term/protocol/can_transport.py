"""
Thread-safe PCAN transport with Qt signals — mirrors SerialTransport pattern.
Gracefully degrades if PCAN DLL is not installed.

v2: Added VESC EID (Extended ID) multi-frame send/receive for VESC commands
    over CAN bus (firmware update, MCCONF read/write, etc.).
v2.2: Combined FW+BL upload, config backup/restore on firmware update.
"""

import struct
import time
import threading
from binascii import crc_hqx

from PyQt6.QtCore import QObject, pyqtSignal

from .can_commands import (
    CAN_HEADER_ID, RmdCommand, STATUS_RETURN_COMMANDS, FAULT_CODE,
    parse_status, parse_status3, parse_multi_turn_angle, format_response_log,
    build_read_motor_status_2,
)
from .commands import build_get_fw_version

# Attempt to import PCANBasic — graceful fallback if DLL not available
PCAN_AVAILABLE = False
try:
    from .PCANBasic import (
        PCANBasic, TPCANMsg, TPCANTimestamp,
        PCAN_USBBUS1, PCAN_BAUD_1M, PCAN_ERROR_OK, PCAN_ERROR_QRCVEMPTY,
        PCAN_MESSAGE_STANDARD, PCAN_MESSAGE_EXTENDED,
    )
    # Verify the DLL actually loads
    _test = PCANBasic()
    PCAN_AVAILABLE = True
    del _test
except Exception:
    PCAN_AVAILABLE = False

# ── VESC CAN EID protocol constants ──────────────────────────────
VESC_CAN_PACKET_FILL_RX_BUFFER       = 5
VESC_CAN_PACKET_FILL_RX_BUFFER_LONG  = 6
VESC_CAN_PACKET_PROCESS_RX_BUFFER    = 7
VESC_CAN_PACKET_PROCESS_SHORT_BUFFER = 8

PC_SENDER_ID = 0xFE  # Host PC virtual CAN ID

# Fault broadcast
FAULT_BROADCAST_MARKER = 0xBF


class PcanTransport(QObject):
    """
    PCAN-USB CAN bus transport layer.
    Emits Qt signals for received frames and parsed RMD status.
    """

    frame_received = pyqtSignal(int, int, float, list)  # can_id, dlc, timestamp, data_list
    status_received = pyqtSignal(int, object)             # (motor_id, RmdStatus) from 0x9C polling
    cmd_status_received = pyqtSignal(int, object)        # (motor_id, RmdStatus) from 0xA1-A4 control
    status3_received = pyqtSignal(int, object)           # (motor_id, RmdStatus3) from 0x9D
    multiturn_received = pyqtSignal(float)               # degrees (0x92)
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    log_message = pyqtSignal(str)

    # v2 signals
    vesc_response_received = pyqtSignal(int, bytes)      # (sender_id, reassembled_payload)
    fault_detected = pyqtSignal(int, int, str)           # (can_id, fault_code, fault_name)
    scan_complete = pyqtSignal(list)                      # [int, ...] — found motor IDs

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

        # EID multi-frame RX reassembly buffer: {sender_id: bytearray}
        self._eid_rx_buf: dict[int, bytearray] = {}

        # CAN scan state
        self._scan_mode: bool = False
        self._scan_found: set[int] = set()

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

    def send_frame_to(self, motor_id: int, data: bytes) -> bool:
        """Send a CAN frame to a specific motor_id without changing the global motor_id."""
        if not self._connected:
            return False

        with self._lock:
            msg = TPCANMsg()
            msg.MSGTYPE = PCAN_MESSAGE_STANDARD
            msg.ID = CAN_HEADER_ID + motor_id
            msg.LEN = 8
            for i in range(min(len(data), 8)):
                msg.DATA[i] = data[i]
            result = self._pc.Write(PCAN_USBBUS1, msg)

        if result != PCAN_ERROR_OK:
            return False
        with self._counter_lock:
            self._tx_frames += 1
        return True

    def start_scan(self, id_min: int = 1, id_max: int = 253):
        """Send 0x9C status query and VESC COMM_FW_VERSION to id_min~id_max to discover active controllers."""
        self._scan_found.clear()
        self._scan_mode = True

        # Phase 1: RMD SID probe (0x9C)
        probe = build_read_motor_status_2()
        for mid in range(id_min, id_max + 1):
            self.send_frame_to(mid, probe)
            time.sleep(0.002)

        # Phase 2: VESC EID probe (COMM_FW_VERSION)
        fw_probe = build_get_fw_version()
        for mid in range(id_min, id_max + 1):
            self.send_vesc_command(mid, fw_probe, send_mode=0)
            time.sleep(0.002)

    def finish_scan(self) -> list[int]:
        """End scan mode and return sorted list of discovered motor IDs."""
        self._scan_mode = False
        found = sorted(self._scan_found)
        self._scan_found.clear()
        return found

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

    # ── VESC EID (Extended ID) multi-frame protocol ──

    def send_vesc_to_target(self, payload: bytes, send_mode: int = 0) -> bool:
        """Send a VESC command to the current motor_id via EID multi-frame.
        Drop-in replacement for SerialTransport.send_packet()."""
        return self.send_vesc_command(self._motor_id, payload, send_mode)

    def send_vesc_command(self, target_id: int, payload: bytes,
                          send_mode: int = 0) -> bool:
        """
        Send a VESC command via EID multi-frame CAN protocol.

        Args:
            target_id: Target controller ID (0-254, or 255 for broadcast)
            payload: VESC command payload (e.g. [COMM_ID] + data)
            send_mode: 0=process+respond, 1=respond only, 2=process only (no response)
        Returns:
            True if all frames were sent successfully
        """
        if not self._connected:
            return False
        if len(payload) <= 6:
            return self._send_short_buffer(target_id, payload, send_mode)
        else:
            return self._send_long_buffer(target_id, payload, send_mode)

    def _send_short_buffer(self, target_id: int, payload: bytes,
                           send_mode: int) -> bool:
        """Send a short VESC command (<=6 bytes) using PROCESS_SHORT_BUFFER."""
        # EID = (CAN_PACKET_PROCESS_SHORT_BUFFER << 8) | target_id
        eid = (VESC_CAN_PACKET_PROCESS_SHORT_BUFFER << 8) | (target_id & 0xFF)
        # Data: [sender_id] [send_mode] [payload...]
        data = bytes([PC_SENDER_ID, send_mode & 0xFF]) + payload
        return self._send_eid_frame(eid, data)

    def _send_long_buffer(self, target_id: int, payload: bytes,
                          send_mode: int) -> bool:
        """Send a long VESC command using FILL_RX_BUFFER + PROCESS_RX_BUFFER."""
        target = target_id & 0xFF
        offset = 0
        total = len(payload)

        # Phase 1: Fill RX buffer
        while offset < total:
            if offset <= 255:
                # FILL_RX_BUFFER: [index:1][payload:7]
                eid = (VESC_CAN_PACKET_FILL_RX_BUFFER << 8) | target
                chunk_max = 7
                chunk = payload[offset:offset + chunk_max]
                data = bytes([offset & 0xFF]) + chunk
            else:
                # FILL_RX_BUFFER_LONG: [index_H:1][index_L:1][payload:6]
                eid = (VESC_CAN_PACKET_FILL_RX_BUFFER_LONG << 8) | target
                chunk_max = 6
                chunk = payload[offset:offset + chunk_max]
                data = bytes([(offset >> 8) & 0xFF, offset & 0xFF]) + chunk

            if not self._send_eid_frame(eid, data):
                return False
            offset += len(chunk)

        # Phase 2: Process RX buffer
        crc = crc_hqx(payload, 0)
        eid = (VESC_CAN_PACKET_PROCESS_RX_BUFFER << 8) | target
        data = bytes([
            PC_SENDER_ID,
            send_mode & 0xFF,
            (total >> 8) & 0xFF, total & 0xFF,
            (crc >> 8) & 0xFF, crc & 0xFF,
        ])
        return self._send_eid_frame(eid, data)

    def _send_eid_frame(self, eid: int, data: bytes) -> bool:
        """Send a single CAN Extended ID frame. Thread-safe."""
        if not self._connected:
            return False

        with self._lock:
            msg = TPCANMsg()
            msg.MSGTYPE = PCAN_MESSAGE_EXTENDED
            msg.ID = eid & 0x1FFFFFFF  # 29-bit
            msg.LEN = min(len(data), 8)
            for i in range(msg.LEN):
                msg.DATA[i] = data[i]
            result = self._pc.Write(PCAN_USBBUS1, msg)

        if result != PCAN_ERROR_OK:
            self.log_message.emit(f"[PCAN] EID send error: 0x{result:02X}")
            return False
        with self._counter_lock:
            self._tx_frames += 1
        return True

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
        """Process a received CAN message (SID or EID)."""
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
                'ext': bool(msg.MSGTYPE & 0x02),
            })

        # Emit raw frame signal
        self.frame_received.emit(msg.ID, msg.LEN, timestamp, data_list)

        # ── EID frame handling (VESC protocol responses) ──
        if msg.MSGTYPE & 0x02:
            self._process_eid_frame(msg.ID, data_list)
            return

        # ── SID frame handling ──

        # Check for fault broadcast: SID = 0x140 + motor_id, DATA[0] == 0xBF
        if msg.LEN >= 2 and data_list[0] == FAULT_BROADCAST_MARKER:
            can_id = msg.ID
            motor_id = can_id - CAN_HEADER_ID if can_id >= CAN_HEADER_ID else can_id
            fault_code = data_list[1]
            fault_name = FAULT_CODE.get(fault_code, f'UNKNOWN_FAULT_{fault_code}')
            self.fault_detected.emit(motor_id, fault_code, fault_name)
            self.log_message.emit(
                f"[FAULT] Motor {motor_id}: {fault_name} (code={fault_code})"
            )
            return

        # Parse RMD protocol response (standard frame only)
        if msg.LEN == 8:
            cmd = data_list[0]
            motor_id = msg.ID - CAN_HEADER_ID if msg.ID >= CAN_HEADER_ID else 0

            if cmd == RmdCommand.READ_MOTOR_STATUS_2:
                # Scan mode: collect responding motor IDs
                if self._scan_mode:
                    if motor_id > 0:
                        self._scan_found.add(motor_id)
                # 0x9C polling response → status_received (for graphs)
                try:
                    status = parse_status(data_list)
                    self.status_received.emit(motor_id, status)
                except Exception:
                    pass
            elif cmd in STATUS_RETURN_COMMANDS:
                # 0xA1-A4 control command responses → separate signal
                try:
                    status = parse_status(data_list)
                    self.cmd_status_received.emit(motor_id, status)
                except Exception:
                    pass
            elif cmd == RmdCommand.READ_MOTOR_STATUS_3:
                try:
                    status3 = parse_status3(data_list)
                    self.status3_received.emit(motor_id, status3)
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

    def _process_eid_frame(self, eid: int, data: list):
        """Process a VESC Extended ID CAN frame (multi-frame reassembly)."""
        packet_type = (eid >> 8) & 0xFF
        sender_id = eid & 0xFF

        if packet_type == VESC_CAN_PACKET_FILL_RX_BUFFER:
            # [index:1][payload:1-7]
            if len(data) < 2:
                return
            index = data[0]
            payload = bytes(data[1:])
            buf = self._eid_rx_buf.setdefault(sender_id, bytearray())
            # Extend buffer to accommodate new data
            needed = index + len(payload)
            if len(buf) < needed:
                buf.extend(b'\x00' * (needed - len(buf)))
            buf[index:index + len(payload)] = payload

        elif packet_type == VESC_CAN_PACKET_FILL_RX_BUFFER_LONG:
            # [index_H:1][index_L:1][payload:1-6]
            if len(data) < 3:
                return
            index = (data[0] << 8) | data[1]
            payload = bytes(data[2:])
            buf = self._eid_rx_buf.setdefault(sender_id, bytearray())
            needed = index + len(payload)
            if len(buf) < needed:
                buf.extend(b'\x00' * (needed - len(buf)))
            buf[index:index + len(payload)] = payload

        elif packet_type == VESC_CAN_PACKET_PROCESS_RX_BUFFER:
            # [sender_id:1][send:1][len_H:1][len_L:1][crc_H:1][crc_L:1]
            if len(data) < 6:
                return
            rx_sender = data[0]
            length = (data[2] << 8) | data[3]
            crc_expected = (data[4] << 8) | data[5]

            buf = self._eid_rx_buf.pop(sender_id, None)
            if buf is None or len(buf) < length:
                self.log_message.emit(
                    f"[EID] Buffer underflow from {rx_sender}: "
                    f"got {len(buf) if buf else 0}, expected {length}"
                )
                return

            assembled = bytes(buf[:length])
            crc_actual = crc_hqx(assembled, 0)
            if crc_actual != crc_expected:
                self.log_message.emit(
                    f"[EID] CRC mismatch from {rx_sender}: "
                    f"expected 0x{crc_expected:04X}, got 0x{crc_actual:04X}"
                )
                return

            if self._scan_mode and rx_sender > 0:
                self._scan_found.add(rx_sender)
            self.vesc_response_received.emit(rx_sender, assembled)

        elif packet_type == VESC_CAN_PACKET_PROCESS_SHORT_BUFFER:
            # [sender_id:1][send:1][payload:1-6]
            if len(data) < 3:
                return
            vesc_sender = data[0]
            payload = bytes(data[2:])
            if self._scan_mode and vesc_sender > 0:
                self._scan_found.add(vesc_sender)
            self.vesc_response_received.emit(vesc_sender, payload)

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
