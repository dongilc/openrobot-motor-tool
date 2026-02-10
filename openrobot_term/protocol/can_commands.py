"""
RMD protocol command builders, response parsers, and constants.
Ported from the PCAN test program (vesc_pcan.py + general_defines.py).
"""

from dataclasses import dataclass
from enum import IntEnum
import struct


# ── Constants ──────────────────────────────────────────────────────

CAN_HEADER_ID = 0x140
CNT2DEG = 360.0 / 16384.0
DIGIT2AMPHERE = 33.0 / 2048.0


class RmdCommand(IntEnum):
    READ_PID = 0x30
    WRITE_PID_TO_RAM = 0x31
    WRITE_PID_TO_ROM = 0x32
    READ_ACCELERATION = 0x33
    WRITE_ACCELERATION_TO_RAM = 0x34
    READ_ENCODER = 0x90
    WRITE_ENCODER_OFFSET = 0x91
    WRITE_CURRENT_POS_TO_ROM = 0x19
    READ_MULTI_TURN_ANGLE = 0x92
    READ_SINGLE_TURN_ANGLE = 0x94
    READ_MOTOR_STATUS_1 = 0x9A
    CLEAR_ERROR_FLAG = 0x9B
    READ_MOTOR_STATUS_2 = 0x9C
    READ_MOTOR_STATUS_3 = 0x9D
    MOTOR_OFF = 0x80
    MOTOR_STOP = 0x81
    MOTOR_START = 0x88
    TORQUE_CLOSED_LOOP = 0xA1
    SPEED_CLOSED_LOOP = 0xA2
    POSITION_CLOSED_LOOP_1 = 0xA3
    SET_MULTITURN_POSITION = 0xA4
    POSITION_CLOSED_LOOP_3 = 0xA5
    POSITION_CLOSED_LOOP_4 = 0xA6
    READ_MULTITURN_ENCODER_POSITION = 0x60
    READ_MULTITURN_ORIGINAL_ENCODER_POSITION = 0x61
    READ_MULTITURN_ENCODER_POSITION_OFFSET = 0x62
    READ_FAULT_CODE = 0xB0
    READ_MAX_CURRENT = 0xB1
    WRITE_MAX_CURRENT_TO_ROM = 0xB2


# Fault broadcast marker (sent by firmware on fault, SID = 0x140 + motor_id)
FAULT_BROADCAST = 0xBF


def parse_fault_broadcast(data: list | bytes) -> tuple[int, str]:
    """Parse fault broadcast frame. Returns (fault_code, fault_name)."""
    fault_code = data[1] if len(data) > 1 else 0
    fault_name = FAULT_CODE.get(fault_code, f'UNKNOWN_FAULT_{fault_code}')
    return fault_code, fault_name


# Commands that return motor status (temp/torque/speed/enc_pos)
STATUS_RETURN_COMMANDS = frozenset({
    RmdCommand.TORQUE_CLOSED_LOOP,
    RmdCommand.SPEED_CLOSED_LOOP,
    RmdCommand.POSITION_CLOSED_LOOP_1,
    RmdCommand.SET_MULTITURN_POSITION,
})


FAULT_CODE = {
    0: 'FAULT_CODE_NONE',
    1: 'FAULT_CODE_OVER_VOLTAGE',
    2: 'FAULT_CODE_UNDER_VOLTAGE',
    3: 'FAULT_CODE_DRV',
    4: 'FAULT_CODE_ABS_OVER_CURRENT',
    5: 'FAULT_CODE_OVER_TEMP_FET',
    6: 'FAULT_CODE_OVER_TEMP_MOTOR',
    7: 'FAULT_CODE_GATE_DRIVER_OVER_VOLTAGE',
    8: 'FAULT_CODE_GATE_DRIVER_UNDER_VOLTAGE',
    9: 'FAULT_CODE_MCU_UNDER_VOLTAGE',
    10: 'FAULT_CODE_BOOTING_FROM_WATCHDOG_RESET',
    11: 'FAULT_CODE_ENCODER_SPI',
    12: 'FAULT_CODE_ENCODER_SINCOS_BELOW_MIN_AMPLITUDE',
    13: 'FAULT_CODE_ENCODER_SINCOS_ABOVE_MAX_AMPLITUDE',
    14: 'FAULT_CODE_FLASH_CORRUPTION',
    15: 'FAULT_CODE_HIGH_OFFSET_CURRENT_SENSOR_1',
    16: 'FAULT_CODE_HIGH_OFFSET_CURRENT_SENSOR_2',
    17: 'FAULT_CODE_HIGH_OFFSET_CURRENT_SENSOR_3',
    18: 'FAULT_CODE_UNBALANCED_CURRENTS',
    19: 'FAULT_CODE_BRK',
    20: 'FAULT_CODE_RESOLVER_LOT',
    21: 'FAULT_CODE_RESOLVER_DOS',
    22: 'FAULT_CODE_RESOLVER_LOS',
    23: 'FAULT_CODE_FLASH_CORRUPTION_APP_CFG',
    24: 'FAULT_CODE_FLASH_CORRUPTION_MC_CFG',
    25: 'FAULT_CODE_ENCODER_NO_MAGNET',
}


# ── Response dataclasses ───────────────────────────────────────────

@dataclass
class RmdStatus:
    motor_temp: float       # celsius
    torque_curr: float      # amps
    speed_dps: int          # degrees per second
    enc_pos: float          # degrees


@dataclass
class RmdStatus3:
    """0x9D response: control mode + phase currents A/B/C."""
    control_mode: int       # firmware CONTROL_MODE enum
    phase_a: float          # amps (phase A current)
    phase_b: float          # amps (phase B current)
    phase_c: float          # amps (phase C current)


@dataclass
class RmdEncoder:
    enc_pos: int
    enc_pos_ori: int
    enc_pos_offset: int
    enc_pos_deg: float
    enc_pos_ori_deg: float
    enc_pos_offset_deg: float
    err_bit: int
    warning_bit: int
    crc_err_bit: int


@dataclass
class RmdPid:
    kp: float
    ki: float
    kd: float


@dataclass
class RmdMaxCurrent:
    drv8301_oc_mode: int
    motor_current_max: float
    motor_current_abs_max: float
    bat_current_max: float


# ── Signed integer helpers (two's complement) ──────────────────────

def _signed_2byte(val: int) -> int:
    if val & 0x8000:
        return -((~val & 0xFFFF) + 1)
    return val


def _signed_7byte(val: int) -> int:
    if val & 0x80000000000000:
        return -((~val & 0xFFFFFFFFFFFFFF) + 1)
    return val


# ── Command builders (return 8-byte data for CAN frame) ────────────

def build_read_encoder() -> bytes:
    return bytes([RmdCommand.READ_ENCODER, 0, 0, 0, 0, 0, 0, 0])


def build_write_encoder_offset(offset: int) -> bytes:
    # offset as 16-bit LE in bytes 6-7
    lo = offset & 0xFF
    hi = (offset >> 8) & 0xFF
    return bytes([RmdCommand.WRITE_ENCODER_OFFSET, 0, 0, 0, 0, 0, lo, hi])


def build_write_current_pos_to_rom() -> bytes:
    return bytes([RmdCommand.WRITE_CURRENT_POS_TO_ROM, 0, 0, 0, 0, 0, 0, 0])


def build_read_multi_turn_angle() -> bytes:
    return bytes([RmdCommand.READ_MULTI_TURN_ANGLE, 0, 0, 0, 0, 0, 0, 0])


def build_motor_off() -> bytes:
    return bytes([RmdCommand.MOTOR_OFF, 0, 0, 0, 0, 0, 0, 0])


def build_motor_stop() -> bytes:
    return bytes([RmdCommand.MOTOR_STOP, 0, 0, 0, 0, 0, 0, 0])


def build_motor_start() -> bytes:
    return bytes([RmdCommand.MOTOR_START, 0, 0, 0, 0, 0, 0, 0])


def build_torque_closed_loop(current_a: float) -> bytes:
    # Convert amps to raw digit: current / DIGIT2AMPHERE
    raw = int(current_a / DIGIT2AMPHERE)
    raw = max(-2048, min(2048, raw))
    val = raw & 0xFFFF
    lo = val & 0xFF
    hi = (val >> 8) & 0xFF
    return bytes([RmdCommand.TORQUE_CLOSED_LOOP, 0, 0, 0, lo, hi, 0, 0])


def build_speed_closed_loop(speed: float, mode: int = 0) -> bytes:
    """Build 0xA2 speed command.
    mode 0: DPS control (0.01 dps/LSB) — OpenRobot custom loop
    mode 1: eRPM control (1 eRPM/LSB)  — VESC built-in PID speed loop
    """
    if mode == 1:
        raw = int(round(speed))       # eRPM, integer
    else:
        raw = int(round(speed * 100))  # 0.01 dps/LSB
    val = struct.pack('<i', raw)
    return bytes([RmdCommand.SPEED_CLOSED_LOOP, mode, 0, 0]) + val


def build_position_closed_loop_1(angle_deg: float) -> bytes:
    # Angle in 0.01 deg/LSB, little-endian 4 bytes at [4:8]
    raw = int(angle_deg * 100)
    val = struct.pack('<i', raw)
    return bytes([RmdCommand.POSITION_CLOSED_LOOP_1, 0, 0, 0]) + val


def build_set_multiturn_position(dps_limit: int, pos_deg: float) -> bytes:
    # Bytes 2-3: speed limit (dps, LE 16-bit)
    # Bytes 4-7: position (0.01 deg/LSB, LE 32-bit signed)
    speed_val = dps_limit & 0xFFFF
    pos_raw = int(pos_deg * 100)
    return bytes([
        RmdCommand.SET_MULTITURN_POSITION, 0,
        speed_val & 0xFF, (speed_val >> 8) & 0xFF,
    ]) + struct.pack('<i', pos_raw)


def build_read_pid() -> bytes:
    return bytes([RmdCommand.READ_PID, 0, 0, 0, 0, 0, 0, 0])


def build_write_pid_to_ram(kp: float, ki: float, kd: float) -> bytes:
    kp_raw = int(kp * 1000 + 0.5) & 0xFFFF
    ki_raw = int(ki * 100000 + 0.5) & 0xFFFF
    kd_raw = int(kd * 100000 + 0.5) & 0xFFFF
    return bytes([
        RmdCommand.WRITE_PID_TO_RAM, 0,
        kp_raw & 0xFF, (kp_raw >> 8) & 0xFF,
        ki_raw & 0xFF, (ki_raw >> 8) & 0xFF,
        kd_raw & 0xFF, (kd_raw >> 8) & 0xFF,
    ])


def build_write_pid_to_rom(kp: float, ki: float, kd: float) -> bytes:
    kp_raw = int(kp * 1000 + 0.5) & 0xFFFF
    ki_raw = int(ki * 100000 + 0.5) & 0xFFFF
    kd_raw = int(kd * 100000 + 0.5) & 0xFFFF
    return bytes([
        RmdCommand.WRITE_PID_TO_ROM, 0,
        kp_raw & 0xFF, (kp_raw >> 8) & 0xFF,
        ki_raw & 0xFF, (ki_raw >> 8) & 0xFF,
        kd_raw & 0xFF, (kd_raw >> 8) & 0xFF,
    ])


def build_read_motor_status_2() -> bytes:
    return bytes([RmdCommand.READ_MOTOR_STATUS_2, 0, 0, 0, 0, 0, 0, 0])


def build_read_motor_status_3() -> bytes:
    return bytes([RmdCommand.READ_MOTOR_STATUS_3, 0, 0, 0, 0, 0, 0, 0])


def build_read_fault_code() -> bytes:
    return bytes([RmdCommand.READ_FAULT_CODE, 0, 0, 0, 0, 0, 0, 0])


def build_read_max_current() -> bytes:
    return bytes([RmdCommand.READ_MAX_CURRENT, 0, 0, 0, 0, 0, 0, 0])


def build_write_max_current_to_rom(oc_mode: int, motor_max: float,
                                    abs_max: float, bat_max: float) -> bytes:
    m = int(motor_max * 100) & 0xFFFF
    a = int(abs_max * 100) & 0xFFFF
    b = int(bat_max * 100) & 0xFFFF
    return bytes([RmdCommand.WRITE_MAX_CURRENT_TO_ROM, oc_mode & 0xFF,
                  m & 0xFF, (m >> 8) & 0xFF,
                  a & 0xFF, (a >> 8) & 0xFF,
                  b & 0xFF, (b >> 8) & 0xFF])


# ── Response parsers ───────────────────────────────────────────────

PHASE_CURR_SCALE = 1.0 / 64.0   # 0x9D phase current: 1A/64 LSB


def parse_status(data: list | bytes) -> RmdStatus:
    """Parse motor status from responses (0xA1/A2/A3/A4/0x9C).
    DATA[1]=motor_temp, DATA[2:3]=torque, DATA[4:5]=speed, DATA[6:7]=encoder."""
    torque_raw = _signed_2byte((data[3] << 8) | data[2])
    speed_raw = _signed_2byte((data[5] << 8) | data[4])
    enc_raw = _signed_2byte((data[7] << 8) | data[6])

    return RmdStatus(
        motor_temp=float(data[1]),
        torque_curr=torque_raw * DIGIT2AMPHERE,
        speed_dps=speed_raw,
        enc_pos=enc_raw * CNT2DEG,
    )


def parse_status3(data: list | bytes) -> RmdStatus3:
    """Parse 0x9D response: DATA[1]=control_mode, DATA[2:7]=phase A/B/C (1A/64 LSB)."""
    phase_a = _signed_2byte((data[3] << 8) | data[2])
    phase_b = _signed_2byte((data[5] << 8) | data[4])
    phase_c = _signed_2byte((data[7] << 8) | data[6])

    return RmdStatus3(
        control_mode=int(data[1]),
        phase_a=phase_a * PHASE_CURR_SCALE,
        phase_b=phase_b * PHASE_CURR_SCALE,
        phase_c=phase_c * PHASE_CURR_SCALE,
    )


def parse_encoder(data: list | bytes) -> RmdEncoder:
    """Parse READ_ENCODER (0x90) response."""
    err_bit = data[1] & 0x01
    warning_bit = (data[1] >> 1) & 0x01
    crc_err_bit = (data[1] >> 2) & 0x01

    enc_pos = _signed_2byte((data[3] << 8) | data[2])
    enc_pos_ori = _signed_2byte((data[5] << 8) | data[4])
    enc_pos_offset = _signed_2byte((data[7] << 8) | data[6])

    return RmdEncoder(
        enc_pos=enc_pos,
        enc_pos_ori=enc_pos_ori,
        enc_pos_offset=enc_pos_offset,
        enc_pos_deg=enc_pos * CNT2DEG,
        enc_pos_ori_deg=enc_pos_ori * CNT2DEG,
        enc_pos_offset_deg=enc_pos_offset * CNT2DEG,
        err_bit=err_bit,
        warning_bit=warning_bit,
        crc_err_bit=crc_err_bit,
    )


def parse_encoder_offset(data: list | bytes) -> tuple[int, float]:
    """Parse WRITE_ENCODER_OFFSET (0x91) or WRITE_CURRENT_POS_TO_ROM (0x19) response."""
    offset = _signed_2byte((data[7] << 8) | data[6])
    return offset, offset * CNT2DEG


def parse_multi_turn_angle(data: list | bytes) -> float:
    """Parse READ_MULTI_TURN_ANGLE (0x92) response. Returns degrees."""
    raw = 0
    for i in range(7, 0, -1):
        raw = (raw << 8) | data[i]
    raw = _signed_7byte(raw)
    return raw / 100.0


def parse_pid(data: list | bytes) -> RmdPid:
    """Parse READ_PID (0x30) response."""
    kp_raw = _signed_2byte((data[3] << 8) | data[2])
    ki_raw = _signed_2byte((data[5] << 8) | data[4])
    kd_raw = _signed_2byte((data[7] << 8) | data[6])
    return RmdPid(
        kp=kp_raw / 1000.0,
        ki=ki_raw / 100000.0,
        kd=kd_raw / 100000.0,
    )


def parse_fault_code(data: list | bytes) -> list[str]:
    """Parse READ_FAULT_CODE (0xB0) response. Returns list of fault strings."""
    faults = []
    for i in range(1, 8):
        if data[i] != 0x00:
            faults.append(FAULT_CODE.get(data[i], f'UNKNOWN_FAULT_{data[i]}'))
    if not faults:
        faults.append(FAULT_CODE[0])
    return faults


def parse_max_current(data: list | bytes) -> RmdMaxCurrent:
    """Parse READ_MAX_CURRENT (0xB1) response."""
    drv8301_oc_mode = data[1]
    motor_current_max = _signed_2byte((data[3] << 8) | data[2]) / 100.0
    motor_current_abs_max = _signed_2byte((data[5] << 8) | data[4]) / 100.0
    bat_current_max = _signed_2byte((data[7] << 8) | data[6]) / 100.0
    return RmdMaxCurrent(
        drv8301_oc_mode=drv8301_oc_mode,
        motor_current_max=motor_current_max,
        motor_current_abs_max=motor_current_abs_max,
        bat_current_max=bat_current_max,
    )


def format_response_log(cmd: int, data: list | bytes) -> str:
    """Format a CAN response into a human-readable log string."""
    try:
        if cmd == RmdCommand.READ_ENCODER:
            e = parse_encoder(data)
            return (f"enc_pos:{e.enc_pos}({e.enc_pos_deg:.2f}deg) "
                    f"/ enc_pos_ori:{e.enc_pos_ori}({e.enc_pos_ori_deg:.2f}deg) "
                    f"/ enc_pos_offset:{e.enc_pos_offset}({e.enc_pos_offset_deg:.2f}deg) "
                    f"/ err:{e.err_bit} warn:{e.warning_bit} crc:{e.crc_err_bit} (1 is ok)")

        elif cmd in (RmdCommand.WRITE_ENCODER_OFFSET, RmdCommand.WRITE_CURRENT_POS_TO_ROM):
            offset, deg = parse_encoder_offset(data)
            return f"enc_offset:{offset}({deg:.2f}deg)"

        elif cmd == RmdCommand.READ_MULTI_TURN_ANGLE:
            angle = parse_multi_turn_angle(data)
            return f"enc_multiturn_angle: {angle:.2f} deg"

        elif cmd == RmdCommand.READ_PID:
            pid = parse_pid(data)
            return f"kp:{pid.kp:.5f} / ki:{pid.ki:.5f} / kd:{pid.kd:.5f}"

        elif cmd == RmdCommand.READ_FAULT_CODE:
            faults = parse_fault_code(data)
            return "fault_code: " + ", ".join(faults)

        elif cmd == RmdCommand.READ_MAX_CURRENT:
            mc = parse_max_current(data)
            return (f"motor_current_max:{mc.motor_current_max:.2f}A "
                    f"/ motor_current_abs_max:{mc.motor_current_abs_max:.2f}A "
                    f"/ bat_current_max:{mc.bat_current_max:.2f}A "
                    f"/ drv8301_oc_mode:{mc.drv8301_oc_mode}")

        elif cmd in STATUS_RETURN_COMMANDS:
            s = parse_status(data)
            return (f"temp:{s.motor_temp:.0f}C torque:{s.torque_curr:.3f}A "
                    f"speed:{s.speed_dps}dps enc:{s.enc_pos:.2f}deg")

        elif cmd in (RmdCommand.MOTOR_OFF, RmdCommand.MOTOR_STOP, RmdCommand.MOTOR_START):
            return f"cmd {RmdCommand(cmd).name} ACK"

        return f"cmd=0x{cmd:02X} data={' '.join(f'{b:02x}' for b in data)}"

    except Exception as ex:
        return f"parse error (cmd=0x{cmd:02X}): {ex}"
