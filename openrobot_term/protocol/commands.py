"""
VESC COMM packet type definitions, data structures, and builder functions.

NOTE: The VescValues field layout follows standard VESC 6.x firmware.
If your custom firmware has a different byte layout, adjust from_payload() accordingly.
"""

import struct
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class CommPacketId(IntEnum):
    """
    Command IDs based on OpenRobot VESC firmware (bldc_5.02_openrobot_ver5).
    Values must match the COMM_PACKET_ID enum in datatypes.h
    """
    COMM_FW_VERSION = 0
    COMM_JUMP_TO_BOOTLOADER = 1
    COMM_ERASE_NEW_APP = 2
    COMM_WRITE_NEW_APP_DATA = 3
    COMM_GET_VALUES = 4
    COMM_SET_DUTY = 5
    COMM_SET_CURRENT = 6
    COMM_SET_CURRENT_BRAKE = 7
    COMM_SET_RPM = 8
    COMM_SET_POS = 9
    COMM_SET_HANDBRAKE = 10
    COMM_SET_DETECT = 11
    COMM_SET_SERVO_POS = 12
    COMM_SET_MCCONF = 13
    COMM_GET_MCCONF = 14
    COMM_GET_MCCONF_DEFAULT = 15
    COMM_SET_APPCONF = 16
    COMM_GET_APPCONF = 17
    COMM_GET_APPCONF_DEFAULT = 18
    COMM_SAMPLE_PRINT = 19
    COMM_TERMINAL_CMD = 20
    COMM_PRINT = 21  # Terminal output (printf)
    COMM_ROTOR_POSITION = 22
    COMM_EXPERIMENT_SAMPLE = 23
    COMM_REBOOT = 29
    COMM_ALIVE = 30
    COMM_GET_VALUES_SETUP = 47
    COMM_TERMINAL_CMD_SYNC = 64
    COMM_ERASE_BOOTLOADER = 73
    COMM_PLOT_INIT = 75
    COMM_PLOT_DATA = 76
    COMM_PLOT_ADD_GRAPH = 77
    COMM_PLOT_SET_GRAPH = 78
    COMM_WRITE_NEW_APP_DATA_LZO = 81


class FaultCode(IntEnum):
    FAULT_CODE_NONE = 0
    FAULT_CODE_OVER_VOLTAGE = 1
    FAULT_CODE_UNDER_VOLTAGE = 2
    FAULT_CODE_DRV = 3
    FAULT_CODE_ABS_OVER_CURRENT = 4
    FAULT_CODE_OVER_TEMP_FET = 5
    FAULT_CODE_OVER_TEMP_MOTOR = 6
    FAULT_CODE_GATE_DRIVER_OVER_VOLTAGE = 7
    FAULT_CODE_GATE_DRIVER_UNDER_VOLTAGE = 8
    FAULT_CODE_MCU_UNDER_VOLTAGE = 9
    FAULT_CODE_BOOTING_FROM_WATCHDOG_RESET = 10
    FAULT_CODE_ENCODER_FAULT = 11


@dataclass
class VescValues:
    """Parsed response from COMM_GET_VALUES."""
    temp_mosfet: float = 0.0
    temp_motor: float = 0.0
    avg_motor_current: float = 0.0
    avg_input_current: float = 0.0
    id_current: float = 0.0
    iq_current: float = 0.0
    duty_now: float = 0.0
    rpm: float = 0.0
    v_in: float = 0.0
    amp_hours: float = 0.0
    amp_hours_charged: float = 0.0
    watt_hours: float = 0.0
    watt_hours_charged: float = 0.0
    tachometer: int = 0
    tachometer_abs: int = 0
    fault_code: int = 0
    pid_pos: float = 0.0
    vesc_id: int = 0

    @classmethod
    def from_payload(cls, data: bytes) -> "VescValues":
        """
        Parse COMM_GET_VALUES response payload (after COMM_ID byte is stripped).

        Standard VESC 6.x layout (big-endian):
          temp_mosfet:       int16  / 10
          temp_motor:        int16  / 10
          avg_motor_current: int32  / 100
          avg_input_current: int32  / 100
          id_current:        int32  / 100
          iq_current:        int32  / 100
          duty_now:          int16  / 1000
          rpm:               int32
          v_in:              int16  / 10
          amp_hours:         int32  / 10000
          amp_hours_charged: int32  / 10000
          watt_hours:        int32  / 10000
          watt_hours_charged:int32  / 10000
          tachometer:        int32
          tachometer_abs:    int32
          fault_code:        int8
          pid_pos:           int32  / 1000000
          vesc_id:           uint8
        """
        v = cls()
        idx = 0
        try:
            v.temp_mosfet = struct.unpack_from(">h", data, idx)[0] / 10.0; idx += 2
            v.temp_motor = struct.unpack_from(">h", data, idx)[0] / 10.0; idx += 2
            v.avg_motor_current = struct.unpack_from(">i", data, idx)[0] / 100.0; idx += 4
            v.avg_input_current = struct.unpack_from(">i", data, idx)[0] / 100.0; idx += 4
            v.id_current = struct.unpack_from(">i", data, idx)[0] / 100.0; idx += 4
            v.iq_current = struct.unpack_from(">i", data, idx)[0] / 100.0; idx += 4
            v.duty_now = struct.unpack_from(">h", data, idx)[0] / 1000.0; idx += 2
            v.rpm = struct.unpack_from(">i", data, idx)[0]; idx += 4
            v.v_in = struct.unpack_from(">h", data, idx)[0] / 10.0; idx += 2
            v.amp_hours = struct.unpack_from(">i", data, idx)[0] / 10000.0; idx += 4
            v.amp_hours_charged = struct.unpack_from(">i", data, idx)[0] / 10000.0; idx += 4
            v.watt_hours = struct.unpack_from(">i", data, idx)[0] / 10000.0; idx += 4
            v.watt_hours_charged = struct.unpack_from(">i", data, idx)[0] / 10000.0; idx += 4
            v.tachometer = struct.unpack_from(">i", data, idx)[0]; idx += 4
            v.tachometer_abs = struct.unpack_from(">i", data, idx)[0]; idx += 4
            v.fault_code = struct.unpack_from(">b", data, idx)[0]; idx += 1
            if idx + 4 <= len(data):
                v.pid_pos = struct.unpack_from(">i", data, idx)[0] / 1000000.0; idx += 4
            if idx + 1 <= len(data):
                v.vesc_id = struct.unpack_from(">B", data, idx)[0]; idx += 1
        except struct.error:
            pass  # Partial data — return what we have
        return v


import math

def decode_float32_auto(data: bytes, offset: int) -> tuple:
    """
    Decode VESC float32_auto format (4 bytes).
    Returns (value, new_offset).
    """
    if offset + 4 > len(data):
        return (0.0, offset + 4)

    res = struct.unpack_from(">I", data, offset)[0]

    e = (res >> 23) & 0xFF
    sig_i = res & 0x7FFFFF
    neg = bool(res & (1 << 31))

    sig = 0.0
    if e != 0 or sig_i != 0:
        sig = sig_i / (8388608.0 * 2.0) + 0.5
        e -= 126

    if neg:
        sig = -sig

    value = math.ldexp(sig, e)
    return (value, offset + 4)


@dataclass
class WaveformSamples:
    """
    Parsed waveform sample data from COMM_SAMPLE_PRINT.

    Firmware format (34 bytes per sample):
    - curr0: float32_auto (4 bytes) - Current sensor 0
    - curr1: float32_auto (4 bytes) - Current sensor 1
    - ph1: float32_auto (4 bytes) - Phase 1 voltage
    - ph2: float32_auto (4 bytes) - Phase 2 voltage
    - ph3: float32_auto (4 bytes) - Phase 3 voltage
    - vzero: float32_auto (4 bytes) - Zero voltage
    - curr_fir: float32_auto (4 bytes) - FIR filtered current
    - f_sw: float32_auto (4 bytes) - Switching frequency
    - status: uint8 (1 byte)
    - phase: int8 (1 byte)
    """
    # Current measurements (only 2 sensors in hardware)
    curr0: list = field(default_factory=list)  # Current sensor 0
    curr1: list = field(default_factory=list)  # Current sensor 1

    # Phase voltages
    ph1_voltage: list = field(default_factory=list)
    ph2_voltage: list = field(default_factory=list)
    ph3_voltage: list = field(default_factory=list)

    # Additional data
    vzero: list = field(default_factory=list)
    curr_fir: list = field(default_factory=list)
    f_sw: list = field(default_factory=list)
    status: list = field(default_factory=list)
    phase: list = field(default_factory=list)

    num_samples: int = 0

    # Aliases for backward compatibility (map to actual data)
    @property
    def phase_a(self):
        return self.curr0

    @property
    def phase_b(self):
        return self.curr1

    @property
    def phase_c(self):
        """Third current calculated as -(curr0 + curr1) for balanced 3-phase."""
        return [-(c0 + c1) for c0, c1 in zip(self.curr0, self.curr1)]

    @classmethod
    def from_payload(cls, data: bytes) -> "WaveformSamples":
        """
        Parse COMM_SAMPLE_PRINT response.
        Format: 34 bytes per sample (8 x float32_auto + 2 bytes)
        """
        w = cls()
        idx = 0
        sample_size = 34  # 8 * 4 + 2 bytes

        try:
            while idx + sample_size <= len(data):
                curr0, idx = decode_float32_auto(data, idx)
                curr1, idx = decode_float32_auto(data, idx)
                ph1, idx = decode_float32_auto(data, idx)
                ph2, idx = decode_float32_auto(data, idx)
                ph3, idx = decode_float32_auto(data, idx)
                vzero, idx = decode_float32_auto(data, idx)
                curr_fir, idx = decode_float32_auto(data, idx)
                f_sw, idx = decode_float32_auto(data, idx)
                status = data[idx]; idx += 1
                phase = struct.unpack_from(">b", data, idx)[0]; idx += 1

                w.curr0.append(curr0)
                w.curr1.append(curr1)
                w.ph1_voltage.append(ph1)
                w.ph2_voltage.append(ph2)
                w.ph3_voltage.append(ph3)
                w.vzero.append(vzero)
                w.curr_fir.append(curr_fir)
                w.f_sw.append(f_sw)
                w.status.append(status)
                w.phase.append(phase)
                w.num_samples += 1
        except (struct.error, IndexError):
            pass
        return w


# ---- Packet builder functions ----

def build_get_values() -> bytes:
    return bytes([CommPacketId.COMM_GET_VALUES])


def build_get_fw_version() -> bytes:
    return bytes([CommPacketId.COMM_FW_VERSION])


def build_terminal_cmd(cmd: str, sync: bool = False) -> bytes:
    """
    Build terminal command packet.
    sync=False (default): Use COMM_TERMINAL_CMD (20) - standard VESC Tool behavior
    sync=True: Use COMM_TERMINAL_CMD_SYNC (64) - guarantees acknowledgment response

    Format: [COMM_ID] [command string]
    Note: VESC Tool does NOT use null-terminated string.
    """
    cmd_id = CommPacketId.COMM_TERMINAL_CMD_SYNC if sync else CommPacketId.COMM_TERMINAL_CMD
    # No null terminator - matches VESC Tool behavior
    return bytes([cmd_id]) + cmd.encode("utf-8")


def build_set_duty(duty: float) -> bytes:
    """Set duty cycle. duty in range -1.0 to 1.0, sent as int32 * 100000."""
    return bytes([CommPacketId.COMM_SET_DUTY]) + struct.pack(">i", int(duty * 100000))


def build_set_current(current_a: float) -> bytes:
    """Set motor current in amps, sent as int32 * 1000."""
    return bytes([CommPacketId.COMM_SET_CURRENT]) + struct.pack(">i", int(current_a * 1000))


def build_set_rpm(rpm: int) -> bytes:
    """Set eRPM target."""
    return bytes([CommPacketId.COMM_SET_RPM]) + struct.pack(">i", rpm)


def build_set_pos(pos_deg: float) -> bytes:
    """Set position in degrees, sent as int32 * 1000000."""
    return bytes([CommPacketId.COMM_SET_POS]) + struct.pack(">i", int(pos_deg * 1000000))


class DebugSamplingMode(IntEnum):
    """Debug sampling modes for COMM_SAMPLE_PRINT."""
    OFF = 0
    NOW = 1                    # Sample immediately
    START = 2                  # Start sampling
    TRIGGER_START = 3          # Trigger-based start
    TRIGGER_FAULT = 4          # Trigger on fault
    TRIGGER_START_NOSEND = 5   # Trigger start without auto-send
    TRIGGER_FAULT_NOSEND = 6   # Trigger fault without auto-send
    SEND_LAST_SAMPLES = 7      # Send last captured samples


def build_sample_request(num_samples: int, decimation: int,
                         mode: DebugSamplingMode = DebugSamplingMode.START) -> bytes:
    """
    Request waveform sample capture.

    Args:
        num_samples: Number of samples to capture (max 2000)
        decimation: Decimation factor (1 = every sample, 2 = every 2nd, etc.)
        mode: Sampling mode (default: START)

    Firmware format: [mode:1] [num_samples:2] [decimation:1]
    """
    return (
        bytes([CommPacketId.COMM_SAMPLE_PRINT])
        + struct.pack(">B", mode)           # mode (1 byte)
        + struct.pack(">H", num_samples)    # num_samples (2 bytes)
        + struct.pack(">B", decimation)     # decimation (1 byte)
    )


def build_reboot() -> bytes:
    return bytes([CommPacketId.COMM_REBOOT])


def build_set_pid_gains(kp: float, ki: float, kd: float, kd_filter: float = 0.0) -> bytes:
    """
    Send PID gains to VESC. This uses a custom command format.
    Adjust the COMM ID and format to match your firmware implementation.

    Format: [COMM_SET_MCCONF subset] or custom command
    Here we use COMM_TERMINAL_CMD with a text command as a fallback.
    """
    cmd = f"pid_gains {kp:.6f} {ki:.6f} {kd:.6f} {kd_filter:.6f}"
    return build_terminal_cmd(cmd)


def build_get_mcconf() -> bytes:
    """Request motor configuration (includes PID gains)."""
    return bytes([CommPacketId.COMM_GET_MCCONF])


def build_get_mcconf_default() -> bytes:
    """Request default motor configuration (factory defaults)."""
    return bytes([CommPacketId.COMM_GET_MCCONF_DEFAULT])


def build_get_appconf() -> bytes:
    """Request application configuration."""
    return bytes([CommPacketId.COMM_GET_APPCONF])


def build_get_appconf_default() -> bytes:
    """Request default application configuration (factory defaults)."""
    return bytes([CommPacketId.COMM_GET_APPCONF_DEFAULT])


# ── Rotor position display modes (DISP_POS_MODE_*) ──────────────
DISP_POS_MODE_NONE = 0
DISP_POS_MODE_INDUCTANCE = 1
DISP_POS_MODE_OBSERVER = 2
DISP_POS_MODE_ENCODER = 3
DISP_POS_MODE_PID_POS = 4
DISP_POS_MODE_PID_POS_ERROR = 5
DISP_POS_MODE_ENCODER_OBSERVER_ERROR = 6
DISP_POS_MODE_ACCUM = 7


def build_set_detect(mode: int) -> bytes:
    """Start/stop rotor position streaming. mode: DISP_POS_MODE_* constant."""
    return bytes([CommPacketId.COMM_SET_DETECT, mode & 0xFF])


def decode_rotor_position(data: bytes) -> float:
    """Decode COMM_ROTOR_POSITION payload → degrees."""
    if len(data) < 4:
        return 0.0
    val = struct.unpack_from(">i", data, 0)[0]
    return val / 100000.0


def encode_float32_auto(value: float) -> bytes:
    """
    Encode a float to VESC's float32_auto format.

    Format: 4 bytes big-endian
    - Bit 31: sign
    - Bits 23-30: exponent (8 bits)
    - Bits 0-22: significand (23 bits)
    """
    import math

    if value == 0.0:
        return struct.pack(">I", 0)

    neg = value < 0
    value = abs(value)

    # Get exponent and significand using frexp (value = sig * 2^e, 0.5 <= sig < 1)
    sig, e = math.frexp(value)

    # Convert to VESC format
    sig_i = int((sig - 0.5) * 2.0 * 8388608.0)  # 2^23
    e += 126

    res = ((e & 0xFF) << 23) | (sig_i & 0x7FFFFF)
    if neg:
        res |= (1 << 31)

    return struct.pack(">I", res)


# PID offsets in MCCONF payload (after SIGNATURE, 0-indexed)
# Verified from confgenerator.c (bldc_5.02_openrobot_spn-mc1_v1R2)
MCCONF_SPEED_PID_OFFSETS = {
    "kp": 329,
    "ki": 333,
    "kd": 337,
    "kd_filter": 341,
    "min_erpm": 345,       # s_pid_min_erpm
    # s_pid_allow_braking at 349 (1 byte)
    "ramp_erpms_s": 350,   # s_pid_ramp_erpms_s - affects transient response
}

MCCONF_POSITION_PID_OFFSETS = {
    "kp": 354,
    "ki": 358,
    "kd": 362,
    "kd_filter": 366,
}

# FOC current controller gains offsets in MCCONF
# Based on mc_configuration struct in datatypes.h (bldc_5.02_openrobot_spn-mc1_v1R2)
MCCONF_FOC_CURRENT_OFFSETS = {
    "kp": 153,   # foc_current_kp
    "ki": 157,   # foc_current_ki
}


def build_set_mcconf_with_foc_cc(
    original_mcconf: bytes,
    current_kp: float,
    current_ki: float,
) -> bytes:
    """
    Build COMM_SET_MCCONF packet with modified FOC current controller gains.

    Args:
        original_mcconf: Original MCCONF payload (without cmd_id byte)
        current_kp: FOC current controller Kp
        current_ki: FOC current controller Ki
    """
    data = bytearray(original_mcconf)

    # Write FOC current gains
    for name, value in [("kp", current_kp), ("ki", current_ki)]:
        offset = MCCONF_FOC_CURRENT_OFFSETS[name]
        encoded = encode_float32_auto(value)
        data[offset:offset+4] = encoded

    return bytes([CommPacketId.COMM_SET_MCCONF]) + bytes(data)


def build_set_mcconf_with_pid(
    original_mcconf: bytes,
    kp: float, ki: float, kd: float, kd_filter: float,
    position_mode: bool = False,
    ramp_erpms_s: float = None,  # Speed ramp rate (only for speed mode)
) -> bytes:
    """
    Build COMM_SET_MCCONF packet with modified PID values.

    Args:
        original_mcconf: Original MCCONF payload (without cmd_id byte)
        kp, ki, kd, kd_filter: New PID values
        position_mode: If True, modify position PID; else modify speed PID
        ramp_erpms_s: Speed setpoint ramp rate in eRPM/s (speed mode only)
    """
    # Make a mutable copy
    data = bytearray(original_mcconf)

    # Select offsets based on mode
    offsets = MCCONF_POSITION_PID_OFFSETS if position_mode else MCCONF_SPEED_PID_OFFSETS

    # Write new PID values
    for name, value in [("kp", kp), ("ki", ki), ("kd", kd), ("kd_filter", kd_filter)]:
        offset = offsets[name]
        encoded = encode_float32_auto(value)
        data[offset:offset+4] = encoded

    # Write ramp rate for speed mode only
    if not position_mode and ramp_erpms_s is not None:
        offset = offsets["ramp_erpms_s"]
        encoded = encode_float32_auto(ramp_erpms_s)
        data[offset:offset+4] = encoded

    # Build packet: [COMM_SET_MCCONF] + [modified data]
    return bytes([CommPacketId.COMM_SET_MCCONF]) + bytes(data)


# ---- float32_auto decoding (VESC custom format) ----

def _decode_float32_auto_single(data: bytes, offset: int) -> float:
    """
    Decode VESC's float32_auto format.

    Format: 4 bytes big-endian
    - Bit 31: sign
    - Bits 23-30: exponent (8 bits)
    - Bits 0-22: significand (23 bits)

    Decoding:
    - sig = significand / (2^24) + 0.5
    - e = exponent - 126
    - result = sig * 2^e (with sign)
    """
    res = struct.unpack_from(">I", data, offset)[0]

    e = (res >> 23) & 0xFF
    sig_i = res & 0x7FFFFF
    neg = bool(res & (1 << 31))

    sig = 0.0
    if e != 0 or sig_i != 0:
        sig = float(sig_i) / (8388608.0 * 2.0) + 0.5
        e -= 126

    if neg:
        sig = -sig

    # ldexp: sig * 2^e
    import math
    return math.ldexp(sig, e)


@dataclass
class McconfPid:
    """Parsed PID values from COMM_GET_MCCONF response."""
    # FOC Current Controller
    foc_current_kp: float = 0.0
    foc_current_ki: float = 0.0

    # Speed PID
    s_pid_kp: float = 0.0
    s_pid_ki: float = 0.0
    s_pid_kd: float = 0.0
    s_pid_kd_filter: float = 0.0
    s_pid_min_erpm: float = 0.0
    s_pid_allow_braking: bool = False
    s_pid_ramp_erpms_s: float = 0.0

    # Position PID
    p_pid_kp: float = 0.0
    p_pid_ki: float = 0.0
    p_pid_kd: float = 0.0
    p_pid_kd_filter: float = 0.0
    p_pid_ang_div: float = 0.0

    @classmethod
    def from_mcconf_payload(cls, data: bytes) -> "McconfPid":
        """
        Parse PID values from COMM_GET_MCCONF response.

        The payload starts with MCCONF_SIGNATURE (4 bytes: 0x83C3E1AA).
        PID offsets are relative to start of payload (after cmd_id byte is stripped):
        - Speed PID: Kp(329), Ki(333), Kd(337), Kd_filter(341), min_erpm(345), allow_braking(349), ramp(350)
        - Position PID: Kp(354), Ki(358), Kd(362), Kd_filter(366), ang_div(370)
        """
        pid = cls()

        # Verify MCCONF_SIGNATURE
        if len(data) < 4:
            return pid
        signature = struct.unpack_from(">I", data, 0)[0]
        expected_signature = 2211848314  # 0x83D6207A
        if signature != expected_signature:
            # Try to parse anyway, but log warning
            pass

        try:
            # FOC Current Controller (offsets 153, 157)
            if len(data) > 160:
                pid.foc_current_kp = _decode_float32_auto_single(data, 153)
                pid.foc_current_ki = _decode_float32_auto_single(data, 157)

            # Speed PID (offsets from start of payload)
            if len(data) > 353:
                pid.s_pid_kp = _decode_float32_auto_single(data, 329)
                pid.s_pid_ki = _decode_float32_auto_single(data, 333)
                pid.s_pid_kd = _decode_float32_auto_single(data, 337)
                pid.s_pid_kd_filter = _decode_float32_auto_single(data, 341)
                pid.s_pid_min_erpm = _decode_float32_auto_single(data, 345)
                pid.s_pid_allow_braking = bool(data[349])
                pid.s_pid_ramp_erpms_s = _decode_float32_auto_single(data, 350)

            # Position PID
            if len(data) > 373:
                pid.p_pid_kp = _decode_float32_auto_single(data, 354)
                pid.p_pid_ki = _decode_float32_auto_single(data, 358)
                pid.p_pid_kd = _decode_float32_auto_single(data, 362)
                pid.p_pid_kd_filter = _decode_float32_auto_single(data, 366)
                pid.p_pid_ang_div = _decode_float32_auto_single(data, 370)

        except (struct.error, IndexError):
            pass  # Return what we have

        return pid


# ---- Firmware update commands ----

def build_jump_to_bootloader() -> bytes:
    """Jump to bootloader mode for firmware update."""
    return bytes([CommPacketId.COMM_JUMP_TO_BOOTLOADER])


def build_erase_new_app(app_size: int) -> bytes:
    """
    Erase flash area for new firmware.

    Args:
        app_size: Size of the firmware to be uploaded (bytes)

    Response: [COMM_ERASE_NEW_APP:1][success:1]
    success = non-zero if OK, 0 if error
    """
    return bytes([CommPacketId.COMM_ERASE_NEW_APP]) + struct.pack(">I", app_size)


def build_write_new_app_data(offset: int, data: bytes) -> bytes:
    """
    Write firmware data chunk to flash.

    Args:
        offset: Byte offset in flash (starting from 0)
        data: Firmware data chunk to write (max 384 bytes recommended)

    Response: [COMM_WRITE_NEW_APP_DATA:1][success:1][offset:4 (optional)]
    success = non-zero if OK
    """
    return bytes([CommPacketId.COMM_WRITE_NEW_APP_DATA]) + struct.pack(">I", offset) + data


def build_erase_bootloader() -> bytes:
    """
    Erase bootloader area (for bootloader update).

    Response: [COMM_ERASE_BOOTLOADER:1][success:1]
    """
    return bytes([CommPacketId.COMM_ERASE_BOOTLOADER])
