"""
VESC MCCONF / APPCONF binary parser and XML export/import.

Field tables transcribed 1:1 from confgenerator.c
(bldc_5.02_openrobot_spn-mc1_v1R2).
"""

import struct
import math
from collections import OrderedDict
from typing import NamedTuple, List
import xml.etree.ElementTree as ET

# ── Signatures (from confgenerator.h, decimal values) ─────────────────
MCCONF_SIGNATURE = 2211848314   # 0x83D6207A
APPCONF_SIGNATURE = 3264926020  # 0xC29AD144


# ── Field descriptor ──────────────────────────────────────────────────
class ConfField(NamedTuple):
    name: str        # e.g. "l_current_max"
    dtype: str       # uint8, uint16, int16, uint32, int32, float32_auto, float16
    category: str    # UI group label
    scale: int = 0   # float16 scale factor (only used when dtype == "float16")


# ── float32_auto decode/encode (VESC custom format) ───────────────────

def _decode_float32_auto(data: bytes, offset: int) -> tuple:
    """Decode VESC float32_auto. Returns (value, new_offset)."""
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
    return (math.ldexp(sig, e), offset + 4)


def _encode_float32_auto(value: float) -> bytes:
    """Encode float to VESC float32_auto (4 bytes, big-endian)."""
    if value == 0.0:
        return struct.pack(">I", 0)
    neg = value < 0
    value = abs(value)
    sig, e = math.frexp(value)
    sig_i = int((sig - 0.5) * 2.0 * 8388608.0)
    e += 126
    res = ((e & 0xFF) << 23) | (sig_i & 0x7FFFFF)
    if neg:
        res |= (1 << 31)
    return struct.pack(">I", res)


# ── MCCONF field table (confgenerator.c lines 7-164) ─────────────────
# Exact serialization order. Signature (uint32) is handled separately.

MCCONF_FIELDS: List[ConfField] = [
    # General
    ConfField("pwm_mode",           "uint8",       "General"),
    ConfField("comm_mode",          "uint8",       "General"),
    ConfField("motor_type",         "uint8",       "General"),
    ConfField("sensor_mode",        "uint8",       "General"),
    # Current Limits
    ConfField("l_current_max",      "float32_auto","Current Limits"),
    ConfField("l_current_min",      "float32_auto","Current Limits"),
    ConfField("l_in_current_max",   "float32_auto","Current Limits"),
    ConfField("l_in_current_min",   "float32_auto","Current Limits"),
    ConfField("l_abs_current_max",  "float32_auto","Current Limits"),
    # Speed Limits
    ConfField("l_min_erpm",         "float32_auto","Speed Limits"),
    ConfField("l_max_erpm",         "float32_auto","Speed Limits"),
    ConfField("l_erpm_start",       "float32_auto","Speed Limits"),
    ConfField("l_max_erpm_fbrake",  "float32_auto","Speed Limits"),
    ConfField("l_max_erpm_fbrake_cc","float32_auto","Speed Limits"),
    # Voltage Limits
    ConfField("l_min_vin",          "float32_auto","Voltage Limits"),
    ConfField("l_max_vin",          "float32_auto","Voltage Limits"),
    ConfField("l_battery_cut_start","float32_auto","Voltage Limits"),
    ConfField("l_battery_cut_end",  "float32_auto","Voltage Limits"),
    # Temperature
    ConfField("l_slow_abs_current", "uint8",       "Temperature"),
    ConfField("l_temp_fet_start",   "float32_auto","Temperature"),
    ConfField("l_temp_fet_end",     "float32_auto","Temperature"),
    ConfField("l_temp_motor_start", "float32_auto","Temperature"),
    ConfField("l_temp_motor_end",   "float32_auto","Temperature"),
    ConfField("l_temp_accel_dec",   "float32_auto","Temperature"),
    # Duty / Power
    ConfField("l_min_duty",         "float32_auto","Duty / Power"),
    ConfField("l_max_duty",         "float32_auto","Duty / Power"),
    ConfField("l_watt_max",         "float32_auto","Duty / Power"),
    ConfField("l_watt_min",         "float32_auto","Duty / Power"),
    ConfField("l_current_max_scale","float32_auto","Duty / Power"),
    ConfField("l_current_min_scale","float32_auto","Duty / Power"),
    ConfField("l_duty_start",       "float32_auto","Duty / Power"),
    # Sensorless
    ConfField("sl_min_erpm",                       "float32_auto","Sensorless"),
    ConfField("sl_min_erpm_cycle_int_limit",       "float32_auto","Sensorless"),
    ConfField("sl_max_fullbreak_current_dir_change","float32_auto","Sensorless"),
    ConfField("sl_cycle_int_limit",                "float32_auto","Sensorless"),
    ConfField("sl_phase_advance_at_br",            "float32_auto","Sensorless"),
    ConfField("sl_cycle_int_rpm_br",               "float32_auto","Sensorless"),
    ConfField("sl_bemf_coupling_k",                "float32_auto","Sensorless"),
    # Hall Sensors
    ConfField("hall_table_0",       "uint8",       "Hall Sensors"),
    ConfField("hall_table_1",       "uint8",       "Hall Sensors"),
    ConfField("hall_table_2",       "uint8",       "Hall Sensors"),
    ConfField("hall_table_3",       "uint8",       "Hall Sensors"),
    ConfField("hall_table_4",       "uint8",       "Hall Sensors"),
    ConfField("hall_table_5",       "uint8",       "Hall Sensors"),
    ConfField("hall_table_6",       "uint8",       "Hall Sensors"),
    ConfField("hall_table_7",       "uint8",       "Hall Sensors"),
    ConfField("hall_sl_erpm",       "float32_auto","Hall Sensors"),
    # FOC Current
    ConfField("foc_current_kp",     "float32_auto","FOC Current"),
    ConfField("foc_current_ki",     "float32_auto","FOC Current"),
    ConfField("foc_f_sw",           "float32_auto","FOC Current"),
    ConfField("foc_dt_us",          "float32_auto","FOC Current"),
    # FOC Encoder
    ConfField("foc_encoder_inverted","uint8",      "FOC Encoder"),
    ConfField("foc_encoder_offset", "float32_auto","FOC Encoder"),
    ConfField("foc_encoder_ratio",  "float32_auto","FOC Encoder"),
    ConfField("foc_encoder_sin_gain","float32_auto","FOC Encoder"),
    ConfField("foc_encoder_cos_gain","float32_auto","FOC Encoder"),
    ConfField("foc_encoder_sin_offset","float32_auto","FOC Encoder"),
    ConfField("foc_encoder_cos_offset","float32_auto","FOC Encoder"),
    ConfField("foc_encoder_sincos_filter_constant","float32_auto","FOC Encoder"),
    # FOC Observer
    ConfField("foc_sensor_mode",    "uint8",       "FOC Observer"),
    ConfField("foc_pll_kp",         "float32_auto","FOC Observer"),
    ConfField("foc_pll_ki",         "float32_auto","FOC Observer"),
    ConfField("foc_motor_l",        "float32_auto","FOC Observer"),
    ConfField("foc_motor_ld_lq_diff","float32_auto","FOC Observer"),
    ConfField("foc_motor_r",        "float32_auto","FOC Observer"),
    ConfField("foc_motor_flux_linkage","float32_auto","FOC Observer"),
    ConfField("foc_observer_gain",  "float32_auto","FOC Observer"),
    ConfField("foc_observer_gain_slow","float32_auto","FOC Observer"),
    ConfField("foc_duty_dowmramp_kp","float32_auto","FOC Observer"),
    ConfField("foc_duty_dowmramp_ki","float32_auto","FOC Observer"),
    # FOC Openloop
    ConfField("foc_openloop_rpm",   "float32_auto","FOC Openloop"),
    ConfField("foc_openloop_rpm_low","float16",    "FOC Openloop", 1000),
    ConfField("foc_d_gain_scale_start","float32_auto","FOC Openloop"),
    ConfField("foc_d_gain_scale_max_mod","float32_auto","FOC Openloop"),
    ConfField("foc_sl_openloop_hyst","float16",    "FOC Openloop", 100),
    ConfField("foc_sl_openloop_time_lock","float16","FOC Openloop", 100),
    ConfField("foc_sl_openloop_time_ramp","float16","FOC Openloop", 100),
    ConfField("foc_sl_openloop_time","float16",    "FOC Openloop", 100),
    # FOC Hall
    ConfField("foc_hall_table_0",   "uint8",       "FOC Hall"),
    ConfField("foc_hall_table_1",   "uint8",       "FOC Hall"),
    ConfField("foc_hall_table_2",   "uint8",       "FOC Hall"),
    ConfField("foc_hall_table_3",   "uint8",       "FOC Hall"),
    ConfField("foc_hall_table_4",   "uint8",       "FOC Hall"),
    ConfField("foc_hall_table_5",   "uint8",       "FOC Hall"),
    ConfField("foc_hall_table_6",   "uint8",       "FOC Hall"),
    ConfField("foc_hall_table_7",   "uint8",       "FOC Hall"),
    ConfField("foc_hall_interp_erpm","float32_auto","FOC Hall"),
    ConfField("foc_sl_erpm",        "float32_auto","FOC Hall"),
    # FOC Misc
    ConfField("foc_sample_v0_v7",   "uint8",       "FOC Misc"),
    ConfField("foc_sample_high_current","uint8",   "FOC Misc"),
    ConfField("foc_sat_comp",       "float16",     "FOC Misc", 1000),
    ConfField("foc_temp_comp",      "uint8",       "FOC Misc"),
    ConfField("foc_temp_comp_base_temp","float16", "FOC Misc", 100),
    ConfField("foc_current_filter_const","float32_auto","FOC Misc"),
    ConfField("foc_cc_decoupling",  "uint8",       "FOC Misc"),
    ConfField("foc_observer_type",  "uint8",       "FOC Misc"),
    # FOC HFI
    ConfField("foc_hfi_voltage_start","float32_auto","FOC HFI"),
    ConfField("foc_hfi_voltage_run","float32_auto","FOC HFI"),
    ConfField("foc_hfi_voltage_max","float32_auto","FOC HFI"),
    ConfField("foc_sl_erpm_hfi",    "float32_auto","FOC HFI"),
    ConfField("foc_hfi_start_samples","uint16",    "FOC HFI"),
    ConfField("foc_hfi_obs_ovr_sec","float32_auto","FOC HFI"),
    ConfField("foc_hfi_samples",    "uint8",       "FOC HFI"),
    # GPD
    ConfField("gpd_buffer_notify_left","int16",    "GPD"),
    ConfField("gpd_buffer_interpol","int16",       "GPD"),
    ConfField("gpd_current_filter_const","float32_auto","GPD"),
    ConfField("gpd_current_kp",     "float32_auto","GPD"),
    ConfField("gpd_current_ki",     "float32_auto","GPD"),
    # Speed PID
    ConfField("s_pid_kp",           "float32_auto","Speed PID"),
    ConfField("s_pid_ki",           "float32_auto","Speed PID"),
    ConfField("s_pid_kd",           "float32_auto","Speed PID"),
    ConfField("s_pid_kd_filter",    "float32_auto","Speed PID"),
    ConfField("s_pid_min_erpm",     "float32_auto","Speed PID"),
    ConfField("s_pid_allow_braking","uint8",       "Speed PID"),
    ConfField("s_pid_ramp_erpms_s", "float32_auto","Speed PID"),
    # Position PID
    ConfField("p_pid_kp",           "float32_auto","Position PID"),
    ConfField("p_pid_ki",           "float32_auto","Position PID"),
    ConfField("p_pid_kd",           "float32_auto","Position PID"),
    ConfField("p_pid_kd_filter",    "float32_auto","Position PID"),
    ConfField("p_pid_ang_div",      "float32_auto","Position PID"),
    # Current Control
    ConfField("cc_startup_boost_duty","float32_auto","Current Control"),
    ConfField("cc_min_current",     "float32_auto","Current Control"),
    ConfField("cc_gain",            "float32_auto","Current Control"),
    ConfField("cc_ramp_step_max",   "float32_auto","Current Control"),
    # Motor Misc
    ConfField("m_fault_stop_time_ms","int32",      "Motor Misc"),
    ConfField("m_duty_ramp_step",   "float32_auto","Motor Misc"),
    ConfField("m_current_backoff_gain","float32_auto","Motor Misc"),
    ConfField("m_encoder_counts",   "uint32",      "Motor Misc"),
    ConfField("m_sensor_port_mode", "uint8",       "Motor Misc"),
    ConfField("m_invert_direction", "uint8",       "Motor Misc"),
    ConfField("m_drv8301_oc_mode",  "uint8",       "Motor Misc"),
    ConfField("m_drv8301_oc_adj",   "uint8",       "Motor Misc"),
    ConfField("m_bldc_f_sw_min",    "float32_auto","Motor Misc"),
    ConfField("m_bldc_f_sw_max",    "float32_auto","Motor Misc"),
    ConfField("m_dc_f_sw",          "float32_auto","Motor Misc"),
    ConfField("m_ntc_motor_beta",   "float32_auto","Motor Misc"),
    ConfField("m_out_aux_mode",     "uint8",       "Motor Misc"),
    ConfField("m_motor_temp_sens_type","uint8",    "Motor Misc"),
    ConfField("m_ptc_motor_coeff",  "float32_auto","Motor Misc"),
    ConfField("m_hall_extra_samples","uint8",      "Motor Misc"),
    # Setup Info
    ConfField("si_motor_poles",     "uint8",       "Setup Info"),
    ConfField("si_gear_ratio",      "float32_auto","Setup Info"),
    ConfField("si_wheel_diameter",  "float32_auto","Setup Info"),
    ConfField("si_battery_type",    "uint8",       "Setup Info"),
    ConfField("si_battery_cells",   "uint8",       "Setup Info"),
    ConfField("si_battery_ah",      "float32_auto","Setup Info"),
    # BMS
    ConfField("bms.type",           "uint8",       "BMS"),
    ConfField("bms.t_limit_start",  "float16",     "BMS", 100),
    ConfField("bms.t_limit_end",    "float16",     "BMS", 100),
    ConfField("bms.soc_limit_start","float16",     "BMS", 1000),
    ConfField("bms.soc_limit_end",  "float16",     "BMS", 1000),
]


# ── APPCONF field table (confgenerator.c lines 167-322) ───────────────

APPCONF_FIELDS: List[ConfField] = [
    # General
    ConfField("controller_id",      "uint8",       "General"),
    ConfField("timeout_msec",       "uint32",      "General"),
    ConfField("timeout_brake_current","float32_auto","General"),
    ConfField("send_can_status",    "uint8",       "General"),
    ConfField("send_can_status_rate_hz","uint16",  "General"),
    ConfField("can_baud_rate",      "uint8",       "General"),
    ConfField("pairing_done",       "uint8",       "General"),
    ConfField("permanent_uart_enabled","uint8",    "General"),
    ConfField("shutdown_mode",      "uint8",       "General"),
    ConfField("can_mode",           "uint8",       "General"),
    ConfField("uavcan_esc_index",   "uint8",       "General"),
    ConfField("uavcan_raw_mode",    "uint8",       "General"),
    ConfField("app_to_use",         "uint8",       "General"),
    # PPM
    ConfField("app_ppm_conf.ctrl_type",       "uint8",       "PPM"),
    ConfField("app_ppm_conf.pid_max_erpm",    "float32_auto","PPM"),
    ConfField("app_ppm_conf.hyst",            "float32_auto","PPM"),
    ConfField("app_ppm_conf.pulse_start",     "float32_auto","PPM"),
    ConfField("app_ppm_conf.pulse_end",       "float32_auto","PPM"),
    ConfField("app_ppm_conf.pulse_center",    "float32_auto","PPM"),
    ConfField("app_ppm_conf.median_filter",   "uint8",       "PPM"),
    ConfField("app_ppm_conf.safe_start",      "uint8",       "PPM"),
    ConfField("app_ppm_conf.throttle_exp",    "float32_auto","PPM"),
    ConfField("app_ppm_conf.throttle_exp_brake","float32_auto","PPM"),
    ConfField("app_ppm_conf.throttle_exp_mode","uint8",      "PPM"),
    ConfField("app_ppm_conf.ramp_time_pos",   "float32_auto","PPM"),
    ConfField("app_ppm_conf.ramp_time_neg",   "float32_auto","PPM"),
    ConfField("app_ppm_conf.multi_esc",       "uint8",       "PPM"),
    ConfField("app_ppm_conf.tc",              "uint8",       "PPM"),
    ConfField("app_ppm_conf.tc_max_diff",     "float32_auto","PPM"),
    ConfField("app_ppm_conf.max_erpm_for_dir","float32_auto","PPM"),
    ConfField("app_ppm_conf.smart_rev_max_duty","float32_auto","PPM"),
    ConfField("app_ppm_conf.smart_rev_ramp_time","float32_auto","PPM"),
    # ADC
    ConfField("app_adc_conf.ctrl_type",       "uint8",       "ADC"),
    ConfField("app_adc_conf.hyst",            "float32_auto","ADC"),
    ConfField("app_adc_conf.voltage_start",   "float32_auto","ADC"),
    ConfField("app_adc_conf.voltage_end",     "float32_auto","ADC"),
    ConfField("app_adc_conf.voltage_center",  "float32_auto","ADC"),
    ConfField("app_adc_conf.voltage2_start",  "float32_auto","ADC"),
    ConfField("app_adc_conf.voltage2_end",    "float32_auto","ADC"),
    ConfField("app_adc_conf.use_filter",      "uint8",       "ADC"),
    ConfField("app_adc_conf.safe_start",      "uint8",       "ADC"),
    ConfField("app_adc_conf.cc_button_inverted","uint8",     "ADC"),
    ConfField("app_adc_conf.rev_button_inverted","uint8",    "ADC"),
    ConfField("app_adc_conf.voltage_inverted","uint8",       "ADC"),
    ConfField("app_adc_conf.voltage2_inverted","uint8",      "ADC"),
    ConfField("app_adc_conf.throttle_exp",    "float32_auto","ADC"),
    ConfField("app_adc_conf.throttle_exp_brake","float32_auto","ADC"),
    ConfField("app_adc_conf.throttle_exp_mode","uint8",      "ADC"),
    ConfField("app_adc_conf.ramp_time_pos",   "float32_auto","ADC"),
    ConfField("app_adc_conf.ramp_time_neg",   "float32_auto","ADC"),
    ConfField("app_adc_conf.multi_esc",       "uint8",       "ADC"),
    ConfField("app_adc_conf.tc",              "uint8",       "ADC"),
    ConfField("app_adc_conf.tc_max_diff",     "float32_auto","ADC"),
    ConfField("app_adc_conf.update_rate_hz",  "uint16",      "ADC"),
    # UART
    ConfField("app_uart_baudrate",            "uint32",      "UART"),
    # Nunchuk
    ConfField("app_chuk_conf.ctrl_type",      "uint8",       "Nunchuk"),
    ConfField("app_chuk_conf.hyst",           "float32_auto","Nunchuk"),
    ConfField("app_chuk_conf.ramp_time_pos",  "float32_auto","Nunchuk"),
    ConfField("app_chuk_conf.ramp_time_neg",  "float32_auto","Nunchuk"),
    ConfField("app_chuk_conf.stick_erpm_per_s_in_cc","float32_auto","Nunchuk"),
    ConfField("app_chuk_conf.throttle_exp",   "float32_auto","Nunchuk"),
    ConfField("app_chuk_conf.throttle_exp_brake","float32_auto","Nunchuk"),
    ConfField("app_chuk_conf.throttle_exp_mode","uint8",     "Nunchuk"),
    ConfField("app_chuk_conf.multi_esc",      "uint8",       "Nunchuk"),
    ConfField("app_chuk_conf.tc",             "uint8",       "Nunchuk"),
    ConfField("app_chuk_conf.tc_max_diff",    "float32_auto","Nunchuk"),
    ConfField("app_chuk_conf.use_smart_rev",  "uint8",       "Nunchuk"),
    ConfField("app_chuk_conf.smart_rev_max_duty","float32_auto","Nunchuk"),
    ConfField("app_chuk_conf.smart_rev_ramp_time","float32_auto","Nunchuk"),
    # NRF
    ConfField("app_nrf_conf.speed",           "uint8",       "NRF"),
    ConfField("app_nrf_conf.power",           "uint8",       "NRF"),
    ConfField("app_nrf_conf.crc_type",        "uint8",       "NRF"),
    ConfField("app_nrf_conf.retry_delay",     "uint8",       "NRF"),
    ConfField("app_nrf_conf.retries",         "uint8",       "NRF"),
    ConfField("app_nrf_conf.channel",         "uint8",       "NRF"),
    ConfField("app_nrf_conf.address_0",       "uint8",       "NRF"),
    ConfField("app_nrf_conf.address_1",       "uint8",       "NRF"),
    ConfField("app_nrf_conf.address_2",       "uint8",       "NRF"),
    ConfField("app_nrf_conf.send_crc_ack",    "uint8",       "NRF"),
    # Balance
    ConfField("app_balance_conf.kp",          "float32_auto","Balance"),
    ConfField("app_balance_conf.ki",          "float32_auto","Balance"),
    ConfField("app_balance_conf.kd",          "float32_auto","Balance"),
    ConfField("app_balance_conf.hertz",       "uint16",      "Balance"),
    ConfField("app_balance_conf.fault_pitch", "float32_auto","Balance"),
    ConfField("app_balance_conf.fault_roll",  "float32_auto","Balance"),
    ConfField("app_balance_conf.fault_duty",  "float32_auto","Balance"),
    ConfField("app_balance_conf.fault_adc1",  "float32_auto","Balance"),
    ConfField("app_balance_conf.fault_adc2",  "float32_auto","Balance"),
    ConfField("app_balance_conf.fault_delay_pitch","uint16", "Balance"),
    ConfField("app_balance_conf.fault_delay_roll","uint16",  "Balance"),
    ConfField("app_balance_conf.fault_delay_duty","uint16",  "Balance"),
    ConfField("app_balance_conf.fault_delay_switch_half","uint16","Balance"),
    ConfField("app_balance_conf.fault_delay_switch_full","uint16","Balance"),
    ConfField("app_balance_conf.fault_adc_half_erpm","uint16","Balance"),
    ConfField("app_balance_conf.tiltback_angle","float32_auto","Balance"),
    ConfField("app_balance_conf.tiltback_speed","float32_auto","Balance"),
    ConfField("app_balance_conf.tiltback_duty","float32_auto","Balance"),
    ConfField("app_balance_conf.tiltback_high_voltage","float32_auto","Balance"),
    ConfField("app_balance_conf.tiltback_low_voltage","float32_auto","Balance"),
    ConfField("app_balance_conf.tiltback_constant","float32_auto","Balance"),
    ConfField("app_balance_conf.tiltback_constant_erpm","uint16","Balance"),
    ConfField("app_balance_conf.startup_pitch_tolerance","float32_auto","Balance"),
    ConfField("app_balance_conf.startup_roll_tolerance","float32_auto","Balance"),
    ConfField("app_balance_conf.startup_speed","float32_auto","Balance"),
    ConfField("app_balance_conf.deadzone",    "float32_auto","Balance"),
    ConfField("app_balance_conf.current_boost","float32_auto","Balance"),
    ConfField("app_balance_conf.multi_esc",   "uint8",       "Balance"),
    ConfField("app_balance_conf.yaw_kp",      "float32_auto","Balance"),
    ConfField("app_balance_conf.yaw_ki",      "float32_auto","Balance"),
    ConfField("app_balance_conf.yaw_kd",      "float32_auto","Balance"),
    ConfField("app_balance_conf.roll_steer_kp","float32_auto","Balance"),
    ConfField("app_balance_conf.roll_steer_erpm_kp","float32_auto","Balance"),
    ConfField("app_balance_conf.brake_current","float32_auto","Balance"),
    ConfField("app_balance_conf.yaw_current_clamp","float32_auto","Balance"),
    ConfField("app_balance_conf.setpoint_pitch_filter","float32_auto","Balance"),
    ConfField("app_balance_conf.setpoint_target_filter","float32_auto","Balance"),
    ConfField("app_balance_conf.setpoint_filter_clamp","float32_auto","Balance"),
    ConfField("app_balance_conf.kd_pt1_frequency","uint16",  "Balance"),
    # PAS
    ConfField("app_pas_conf.ctrl_type",       "uint8",       "PAS"),
    ConfField("app_pas_conf.sensor_type",     "uint8",       "PAS"),
    ConfField("app_pas_conf.current_scaling", "float16",     "PAS", 1000),
    ConfField("app_pas_conf.pedal_rpm_start", "float16",     "PAS", 10),
    ConfField("app_pas_conf.pedal_rpm_end",   "float16",     "PAS", 10),
    ConfField("app_pas_conf.invert_pedal_direction","uint8", "PAS"),
    ConfField("app_pas_conf.magnets",         "uint16",      "PAS"),
    ConfField("app_pas_conf.use_filter",      "uint8",       "PAS"),
    ConfField("app_pas_conf.ramp_time_pos",   "float16",     "PAS", 100),
    ConfField("app_pas_conf.ramp_time_neg",   "float16",     "PAS", 100),
    ConfField("app_pas_conf.update_rate_hz",  "uint16",      "PAS"),
    # IMU
    ConfField("imu_conf.type",                "uint8",       "IMU"),
    ConfField("imu_conf.mode",                "uint8",       "IMU"),
    ConfField("imu_conf.sample_rate_hz",      "uint16",      "IMU"),
    ConfField("imu_conf.accel_confidence_decay","float32_auto","IMU"),
    ConfField("imu_conf.mahony_kp",           "float32_auto","IMU"),
    ConfField("imu_conf.mahony_ki",           "float32_auto","IMU"),
    ConfField("imu_conf.madgwick_beta",       "float32_auto","IMU"),
    ConfField("imu_conf.rot_roll",            "float32_auto","IMU"),
    ConfField("imu_conf.rot_pitch",           "float32_auto","IMU"),
    ConfField("imu_conf.rot_yaw",             "float32_auto","IMU"),
    ConfField("imu_conf.accel_offsets_0",     "float32_auto","IMU"),
    ConfField("imu_conf.accel_offsets_1",     "float32_auto","IMU"),
    ConfField("imu_conf.accel_offsets_2",     "float32_auto","IMU"),
    ConfField("imu_conf.gyro_offsets_0",      "float32_auto","IMU"),
    ConfField("imu_conf.gyro_offsets_1",      "float32_auto","IMU"),
    ConfField("imu_conf.gyro_offsets_2",      "float32_auto","IMU"),
    ConfField("imu_conf.gyro_offset_comp_fact_0","float32_auto","IMU"),
    ConfField("imu_conf.gyro_offset_comp_fact_1","float32_auto","IMU"),
    ConfField("imu_conf.gyro_offset_comp_fact_2","float32_auto","IMU"),
    ConfField("imu_conf.gyro_offset_comp_clamp","float32_auto","IMU"),
]


# ── Field descriptions ────────────────────────────────────────────────
# Maps field name → human-readable description (units, enum values, etc.)

FIELD_DESCRIPTIONS: dict[str, str] = {
    # ── MCCONF: General ──
    "pwm_mode":           "0=Nonsynchronous, 1=Synchronous, 2=Bipolar",
    "comm_mode":          "0=Integrate, 1=Delay",
    "motor_type":         "0=BLDC, 1=DC, 2=FOC, 3=GPD",
    "sensor_mode":        "0=Sensorless, 1=Sensored, 2=Hybrid",
    # ── MCCONF: Current Limits ──
    "l_current_max":      "Max motor current (A)",
    "l_current_min":      "Max motor braking current (A, negative)",
    "l_in_current_max":   "Max battery current (A)",
    "l_in_current_min":   "Max battery regen current (A, negative)",
    "l_abs_current_max":  "Absolute max current (A)",
    # ── MCCONF: Speed Limits ──
    "l_min_erpm":         "Min electrical RPM (negative = reverse)",
    "l_max_erpm":         "Max electrical RPM",
    "l_erpm_start":       "ERPM at which current limiting starts",
    "l_max_erpm_fbrake":  "Max ERPM for full brake in sensorless",
    "l_max_erpm_fbrake_cc": "Max ERPM for full brake in current control",
    # ── MCCONF: Voltage Limits ──
    "l_min_vin":          "Min input voltage (V)",
    "l_max_vin":          "Max input voltage (V)",
    "l_battery_cut_start": "Battery cutoff start voltage (V)",
    "l_battery_cut_end":  "Battery cutoff end voltage (V)",
    # ── MCCONF: Temperature ──
    "l_slow_abs_current": "1=Slow down instead of hard abs current limit",
    "l_temp_fet_start":   "FET temp derating start (°C)",
    "l_temp_fet_end":     "FET temp cutoff end (°C)",
    "l_temp_motor_start": "Motor temp derating start (°C)",
    "l_temp_motor_end":   "Motor temp cutoff end (°C)",
    "l_temp_accel_dec":   "Temp deceleration factor",
    # ── MCCONF: Duty / Power ──
    "l_min_duty":         "Min duty cycle (0.0-1.0)",
    "l_max_duty":         "Max duty cycle (0.0-1.0)",
    "l_watt_max":         "Max power output (W)",
    "l_watt_min":         "Max regen power (W, negative)",
    "l_current_max_scale": "Motor current max scale (0.0-1.0)",
    "l_current_min_scale": "Braking current max scale (0.0-1.0)",
    "l_duty_start":       "Duty cycle where current limiting starts",
    # ── MCCONF: Sensorless ──
    "sl_min_erpm":        "Min ERPM for sensorless operation",
    "sl_min_erpm_cycle_int_limit": "Min ERPM for cycle integrator limit",
    "sl_max_fullbreak_current_dir_change": "Max full brake current at dir change (A)",
    "sl_cycle_int_limit": "Cycle integrator limit",
    "sl_phase_advance_at_br": "Phase advance at BEMF rate (deg)",
    "sl_cycle_int_rpm_br": "Cycle integrator BEMF rate (ERPM)",
    "sl_bemf_coupling_k": "BEMF coupling constant",
    # ── MCCONF: Hall Sensors ──
    "hall_table_0":       "Hall sensor lookup table [0]",
    "hall_table_1":       "Hall sensor lookup table [1]",
    "hall_table_2":       "Hall sensor lookup table [2]",
    "hall_table_3":       "Hall sensor lookup table [3]",
    "hall_table_4":       "Hall sensor lookup table [4]",
    "hall_table_5":       "Hall sensor lookup table [5]",
    "hall_table_6":       "Hall sensor lookup table [6]",
    "hall_table_7":       "Hall sensor lookup table [7]",
    "hall_sl_erpm":       "Hall to sensorless transition ERPM",
    # ── MCCONF: FOC Current ──
    "foc_current_kp":     "FOC current controller Kp",
    "foc_current_ki":     "FOC current controller Ki",
    "foc_f_sw":           "FOC switching frequency (Hz)",
    "foc_dt_us":          "FOC dead time (μs)",
    # ── MCCONF: FOC Encoder ──
    "foc_encoder_inverted": "1=Encoder direction inverted",
    "foc_encoder_offset": "Encoder offset (degrees)",
    "foc_encoder_ratio":  "Encoder gear ratio",
    "foc_encoder_sin_gain": "SinCos encoder sine gain",
    "foc_encoder_cos_gain": "SinCos encoder cosine gain",
    "foc_encoder_sin_offset": "SinCos encoder sine offset",
    "foc_encoder_cos_offset": "SinCos encoder cosine offset",
    "foc_encoder_sincos_filter_constant": "SinCos encoder filter constant",
    # ── MCCONF: FOC Observer ──
    "foc_sensor_mode":    "0=Sensorless, 1=Encoder, 2=Hall, 3=HFI",
    "foc_pll_kp":         "FOC PLL proportional gain",
    "foc_pll_ki":         "FOC PLL integral gain",
    "foc_motor_l":        "Motor inductance (H)",
    "foc_motor_ld_lq_diff": "Ld-Lq inductance difference (H)",
    "foc_motor_r":        "Motor resistance (Ω)",
    "foc_motor_flux_linkage": "Motor flux linkage (Wb)",
    "foc_observer_gain":  "FOC observer gain",
    "foc_observer_gain_slow": "FOC observer gain at low speed",
    "foc_duty_dowmramp_kp": "Duty downramp Kp",
    "foc_duty_dowmramp_ki": "Duty downramp Ki",
    # ── MCCONF: FOC Openloop ──
    "foc_openloop_rpm":   "Open loop ERPM threshold",
    "foc_openloop_rpm_low": "Open loop low ERPM threshold",
    "foc_d_gain_scale_start": "D-axis gain scale start",
    "foc_d_gain_scale_max_mod": "D-axis gain scale max modulation",
    "foc_sl_openloop_hyst": "Sensorless-openloop hysteresis",
    "foc_sl_openloop_time_lock": "Sensorless-openloop lock time (s)",
    "foc_sl_openloop_time_ramp": "Sensorless-openloop ramp time (s)",
    "foc_sl_openloop_time": "Sensorless-openloop time (s)",
    # ── MCCONF: FOC Hall ──
    "foc_hall_table_0":   "FOC Hall sensor table [0]",
    "foc_hall_table_1":   "FOC Hall sensor table [1]",
    "foc_hall_table_2":   "FOC Hall sensor table [2]",
    "foc_hall_table_3":   "FOC Hall sensor table [3]",
    "foc_hall_table_4":   "FOC Hall sensor table [4]",
    "foc_hall_table_5":   "FOC Hall sensor table [5]",
    "foc_hall_table_6":   "FOC Hall sensor table [6]",
    "foc_hall_table_7":   "FOC Hall sensor table [7]",
    "foc_hall_interp_erpm": "FOC Hall interpolation ERPM",
    "foc_sl_erpm":        "FOC sensorless ERPM threshold",
    # ── MCCONF: FOC Misc ──
    "foc_sample_v0_v7":   "1=Sample V0 and V7 (center-aligned)",
    "foc_sample_high_current": "1=Sample at high current",
    "foc_sat_comp":       "Saturation compensation factor",
    "foc_temp_comp":      "1=Enable temperature compensation",
    "foc_temp_comp_base_temp": "Base temperature for compensation (°C)",
    "foc_current_filter_const": "Current filter time constant",
    "foc_cc_decoupling":  "0=Disabled, 1=Cross, 2=BEMF, 3=Cross+BEMF",
    "foc_observer_type":  "0=Original, 1=Ortega original, 2=Ortega iterative",
    # ── MCCONF: FOC HFI ──
    "foc_hfi_voltage_start": "HFI injection start voltage (V)",
    "foc_hfi_voltage_run": "HFI injection running voltage (V)",
    "foc_hfi_voltage_max": "HFI injection max voltage (V)",
    "foc_sl_erpm_hfi":    "Sensorless ERPM for HFI transition",
    "foc_hfi_start_samples": "HFI startup samples count",
    "foc_hfi_obs_ovr_sec": "HFI observer override time (s)",
    "foc_hfi_samples":    "HFI samples: 0=8, 1=16, 2=32",
    # ── MCCONF: GPD ──
    "gpd_buffer_notify_left": "GPD buffer notify left",
    "gpd_buffer_interpol": "GPD buffer interpolation",
    "gpd_current_filter_const": "GPD current filter constant",
    "gpd_current_kp":     "GPD current controller Kp",
    "gpd_current_ki":     "GPD current controller Ki",
    # ── MCCONF: Speed PID ──
    "s_pid_kp":           "Speed PID proportional gain",
    "s_pid_ki":           "Speed PID integral gain",
    "s_pid_kd":           "Speed PID derivative gain",
    "s_pid_kd_filter":    "Speed PID derivative filter",
    "s_pid_min_erpm":     "Min ERPM for speed PID",
    "s_pid_allow_braking": "1=Allow braking in speed mode",
    "s_pid_ramp_erpms_s": "Speed PID ramp rate (ERPM/s)",
    # ── MCCONF: Position PID ──
    "p_pid_kp":           "Position PID proportional gain",
    "p_pid_ki":           "Position PID integral gain",
    "p_pid_kd":           "Position PID derivative gain",
    "p_pid_kd_filter":    "Position PID derivative filter",
    "p_pid_ang_div":      "Position PID angle divisor",
    # ── MCCONF: Current Control ──
    "cc_startup_boost_duty": "Startup boost duty cycle",
    "cc_min_current":     "Min current for current control (A)",
    "cc_gain":            "Current controller gain",
    "cc_ramp_step_max":   "Current ramp max step size",
    # ── MCCONF: Motor Misc ──
    "m_fault_stop_time_ms": "Fault stop time (ms)",
    "m_duty_ramp_step":   "Duty cycle ramp step per cycle",
    "m_current_backoff_gain": "Current backoff gain",
    "m_encoder_counts":   "Encoder CPR (counts per revolution)",
    "m_sensor_port_mode": "0=Hall, 1=ABI, 2=AS5047_SPI, 3=AD2S1205, etc.",
    "m_invert_direction": "1=Invert motor direction",
    "m_drv8301_oc_mode":  "DRV8301 overcurrent mode, DRV8301_OC_LIMIT = 0, DRV8301_OC_DISABLED = 3",
    "m_drv8301_oc_adj":   "DRV8301 overcurrent adjustment",
    "m_bldc_f_sw_min":    "BLDC min switching frequency (Hz)",
    "m_bldc_f_sw_max":    "BLDC max switching frequency (Hz)",
    "m_dc_f_sw":          "DC mode switching frequency (Hz)",
    "m_ntc_motor_beta":   "NTC motor temp sensor beta value",
    "m_out_aux_mode":     "0=OFF, 1=ON after FW, 2=ON w/ modulation",
    "m_motor_temp_sens_type": "0=NTC 10K 3380, 1=PTC 1K, 2=NTC/PTC custom",
    "m_ptc_motor_coeff":  "PTC motor temperature coefficient",
    "m_hall_extra_samples": "Extra Hall sensor samples",
    # ── MCCONF: Setup Info ──
    "si_motor_poles":     "Number of motor poles",
    "si_gear_ratio":      "Gear ratio (motor/wheel)",
    "si_wheel_diameter":  "Wheel diameter (m)",
    "si_battery_type":    "0=Li-ion, 1=LiFePO4, 2=Lead-acid",
    "si_battery_cells":   "Number of battery cells in series",
    "si_battery_ah":      "Battery capacity (Ah)",
    # ── MCCONF: BMS ──
    "bms.type":           "0=None, 1=UART, etc.",
    "bms.t_limit_start":  "BMS temp limit start (°C)",
    "bms.t_limit_end":    "BMS temp limit end (°C)",
    "bms.soc_limit_start": "BMS SOC limit start (0.0-1.0)",
    "bms.soc_limit_end":  "BMS SOC limit end (0.0-1.0)",

    # ── APPCONF: General ──
    "controller_id":      "VESC CAN ID (0-253)",
    "timeout_msec":       "Command timeout (ms)",
    "timeout_brake_current": "Brake current on timeout (A)",
    "send_can_status":    "0=Off, 1-5=CAN status message format",
    "send_can_status_rate_hz": "CAN status send rate (Hz)",
    "can_baud_rate":      "0=125K, 1=250K, 2=500K, 3=1M, 4=10K, 5=20K, 6=50K",
    "pairing_done":       "1=Pairing completed",
    "permanent_uart_enabled": "1=UART always enabled",
    "shutdown_mode":      "0=Always on, 1=Off after timeout, 2=Off no input",
    "can_mode":           "0=VESC, 1=UAVCAN",
    "uavcan_esc_index":   "UAVCAN ESC index",
    "uavcan_raw_mode":    "1=UAVCAN raw mode enabled",
    "app_to_use":         "0=None, 1=PPM, 2=ADC, 3=UART, 4=PPM+UART, 5=ADC+UART, 6=Nunchuk, 7=NRF, 8=Custom, 9=Balance, 10=PAS, 11=ADC+PAS",
    # ── APPCONF: PPM ──
    "app_ppm_conf.ctrl_type": "0=None, 1=Cur, 2=CurNoRev, 3=CurNoRevBrk, 4=Duty, 5=DutyNoRev, 6=PID, 7=PIDNoRev",
    "app_ppm_conf.pid_max_erpm": "Max ERPM for PID mode",
    "app_ppm_conf.hyst":  "PPM hysteresis",
    "app_ppm_conf.pulse_start": "Pulse start (μs)",
    "app_ppm_conf.pulse_end": "Pulse end (μs)",
    "app_ppm_conf.pulse_center": "Pulse center (μs)",
    "app_ppm_conf.median_filter": "1=Enable median filter",
    "app_ppm_conf.safe_start": "0=Disabled, 1=Enabled, 2=Regular only",
    "app_ppm_conf.throttle_exp": "Throttle exponential curve (-1.0 to 1.0)",
    "app_ppm_conf.throttle_exp_brake": "Brake exponential curve (-1.0 to 1.0)",
    "app_ppm_conf.throttle_exp_mode": "0=Exp, 1=Natural, 2=Polynomial",
    "app_ppm_conf.ramp_time_pos": "Positive ramp time (s)",
    "app_ppm_conf.ramp_time_neg": "Negative ramp time (s)",
    "app_ppm_conf.multi_esc": "1=Multiple ESCs via CAN",
    "app_ppm_conf.tc":    "1=Enable traction control",
    "app_ppm_conf.tc_max_diff": "Traction control max ERPM diff",
    "app_ppm_conf.max_erpm_for_dir": "Max ERPM for direction change",
    "app_ppm_conf.smart_rev_max_duty": "Smart reverse max duty cycle",
    "app_ppm_conf.smart_rev_ramp_time": "Smart reverse ramp time (s)",
    # ── APPCONF: ADC ──
    "app_adc_conf.ctrl_type": "0=None, 1=Cur, 2=CurRev, 3=CurRevBtn, 4=CurRevBtnBrk, 5=Duty, 6=DutyRev, 7=PID, 8=PIDRev",
    "app_adc_conf.hyst":  "ADC hysteresis",
    "app_adc_conf.voltage_start": "ADC1 voltage start (V)",
    "app_adc_conf.voltage_end": "ADC1 voltage end (V)",
    "app_adc_conf.voltage_center": "ADC1 voltage center (V)",
    "app_adc_conf.voltage2_start": "ADC2 voltage start (V)",
    "app_adc_conf.voltage2_end": "ADC2 voltage end (V)",
    "app_adc_conf.use_filter": "1=Enable input filter",
    "app_adc_conf.safe_start": "0=Disabled, 1=Enabled, 2=Regular only",
    "app_adc_conf.cc_button_inverted": "1=Cruise control button inverted",
    "app_adc_conf.rev_button_inverted": "1=Reverse button inverted",
    "app_adc_conf.voltage_inverted": "1=ADC1 voltage inverted",
    "app_adc_conf.voltage2_inverted": "1=ADC2 voltage inverted",
    "app_adc_conf.throttle_exp": "Throttle exponential curve (-1.0 to 1.0)",
    "app_adc_conf.throttle_exp_brake": "Brake exponential curve (-1.0 to 1.0)",
    "app_adc_conf.throttle_exp_mode": "0=Exp, 1=Natural, 2=Polynomial",
    "app_adc_conf.ramp_time_pos": "Positive ramp time (s)",
    "app_adc_conf.ramp_time_neg": "Negative ramp time (s)",
    "app_adc_conf.multi_esc": "1=Multiple ESCs via CAN",
    "app_adc_conf.tc":    "1=Enable traction control",
    "app_adc_conf.tc_max_diff": "Traction control max ERPM diff",
    "app_adc_conf.update_rate_hz": "ADC update rate (Hz)",
    # ── APPCONF: UART ──
    "app_uart_baudrate":  "UART baud rate (bps)",
    # ── APPCONF: Nunchuk ──
    "app_chuk_conf.ctrl_type": "0=None, 1=Current, 2=CurrentNoRev",
    "app_chuk_conf.hyst": "Nunchuk hysteresis",
    "app_chuk_conf.ramp_time_pos": "Positive ramp time (s)",
    "app_chuk_conf.ramp_time_neg": "Negative ramp time (s)",
    "app_chuk_conf.stick_erpm_per_s_in_cc": "Stick ERPM/s in cruise control",
    "app_chuk_conf.throttle_exp": "Throttle exponential curve",
    "app_chuk_conf.throttle_exp_brake": "Brake exponential curve",
    "app_chuk_conf.throttle_exp_mode": "0=Exp, 1=Natural, 2=Polynomial",
    "app_chuk_conf.multi_esc": "1=Multiple ESCs via CAN",
    "app_chuk_conf.tc":   "1=Enable traction control",
    "app_chuk_conf.tc_max_diff": "Traction control max ERPM diff",
    "app_chuk_conf.use_smart_rev": "1=Use smart reverse",
    "app_chuk_conf.smart_rev_max_duty": "Smart reverse max duty cycle",
    "app_chuk_conf.smart_rev_ramp_time": "Smart reverse ramp time (s)",
    # ── APPCONF: NRF ──
    "app_nrf_conf.speed": "0=250Kbps, 1=1Mbps, 2=2Mbps",
    "app_nrf_conf.power": "0=-18dBm, 1=-12dBm, 2=-6dBm, 3=0dBm",
    "app_nrf_conf.crc_type": "0=Disabled, 1=1byte, 2=2byte",
    "app_nrf_conf.retry_delay": "Auto retransmit delay (0-15)",
    "app_nrf_conf.retries": "Max retransmit count (0-15)",
    "app_nrf_conf.channel": "RF channel (0-125)",
    "app_nrf_conf.address_0": "NRF address byte 0",
    "app_nrf_conf.address_1": "NRF address byte 1",
    "app_nrf_conf.address_2": "NRF address byte 2",
    "app_nrf_conf.send_crc_ack": "1=Send CRC ACK",
    # ── APPCONF: Balance ──
    "app_balance_conf.kp": "Balance PID proportional gain",
    "app_balance_conf.ki": "Balance PID integral gain",
    "app_balance_conf.kd": "Balance PID derivative gain",
    "app_balance_conf.hertz": "Balance loop rate (Hz)",
    "app_balance_conf.fault_pitch": "Fault pitch angle (deg)",
    "app_balance_conf.fault_roll": "Fault roll angle (deg)",
    "app_balance_conf.fault_duty": "Fault duty cycle threshold",
    "app_balance_conf.fault_adc1": "Fault ADC1 threshold (V)",
    "app_balance_conf.fault_adc2": "Fault ADC2 threshold (V)",
    "app_balance_conf.fault_delay_pitch": "Fault delay for pitch (ms)",
    "app_balance_conf.fault_delay_roll": "Fault delay for roll (ms)",
    "app_balance_conf.fault_delay_duty": "Fault delay for duty (ms)",
    "app_balance_conf.fault_delay_switch_half": "Fault delay switch half (ms)",
    "app_balance_conf.fault_delay_switch_full": "Fault delay switch full (ms)",
    "app_balance_conf.fault_adc_half_erpm": "Fault ADC half ERPM threshold",
    "app_balance_conf.tiltback_angle": "Tiltback angle (deg)",
    "app_balance_conf.tiltback_speed": "Tiltback speed (deg/s)",
    "app_balance_conf.tiltback_duty": "Tiltback duty cycle threshold",
    "app_balance_conf.tiltback_high_voltage": "Tiltback high voltage (V)",
    "app_balance_conf.tiltback_low_voltage": "Tiltback low voltage (V)",
    "app_balance_conf.tiltback_constant": "Constant tiltback angle (deg)",
    "app_balance_conf.tiltback_constant_erpm": "Constant tiltback ERPM threshold",
    "app_balance_conf.startup_pitch_tolerance": "Startup pitch tolerance (deg)",
    "app_balance_conf.startup_roll_tolerance": "Startup roll tolerance (deg)",
    "app_balance_conf.startup_speed": "Startup speed limit",
    "app_balance_conf.deadzone": "Input deadzone",
    "app_balance_conf.current_boost": "Current boost factor",
    "app_balance_conf.multi_esc": "1=Multiple ESCs via CAN",
    "app_balance_conf.yaw_kp": "Yaw PID proportional gain",
    "app_balance_conf.yaw_ki": "Yaw PID integral gain",
    "app_balance_conf.yaw_kd": "Yaw PID derivative gain",
    "app_balance_conf.roll_steer_kp": "Roll steering Kp",
    "app_balance_conf.roll_steer_erpm_kp": "Roll steering ERPM Kp",
    "app_balance_conf.brake_current": "Balance brake current (A)",
    "app_balance_conf.yaw_current_clamp": "Yaw current clamp (A)",
    "app_balance_conf.setpoint_pitch_filter": "Setpoint pitch filter constant",
    "app_balance_conf.setpoint_target_filter": "Setpoint target filter constant",
    "app_balance_conf.setpoint_filter_clamp": "Setpoint filter clamp (deg)",
    "app_balance_conf.kd_pt1_frequency": "Kd PT1 filter frequency (Hz)",
    # ── APPCONF: PAS ──
    "app_pas_conf.ctrl_type": "0=None, 1=Current",
    "app_pas_conf.sensor_type": "0=Quadrature, 1=Direction",
    "app_pas_conf.current_scaling": "PAS current scaling factor",
    "app_pas_conf.pedal_rpm_start": "Pedal RPM start threshold",
    "app_pas_conf.pedal_rpm_end": "Pedal RPM end threshold",
    "app_pas_conf.invert_pedal_direction": "1=Invert pedal direction",
    "app_pas_conf.magnets": "Number of PAS magnets",
    "app_pas_conf.use_filter": "1=Enable PAS filter",
    "app_pas_conf.ramp_time_pos": "Positive ramp time (s)",
    "app_pas_conf.ramp_time_neg": "Negative ramp time (s)",
    "app_pas_conf.update_rate_hz": "PAS update rate (Hz)",
    # ── APPCONF: IMU ──
    "imu_conf.type":      "0=Off, 1=Internal, 2=ExtMPU9x50, 3=ExtICM20948, 4=ExtBMI160",
    "imu_conf.mode":      "0=Off, 1=9DOF, 2=6DOF",
    "imu_conf.sample_rate_hz": "IMU sample rate (Hz)",
    "imu_conf.accel_confidence_decay": "Accelerometer confidence decay",
    "imu_conf.mahony_kp": "Mahony filter proportional gain",
    "imu_conf.mahony_ki": "Mahony filter integral gain",
    "imu_conf.madgwick_beta": "Madgwick filter beta",
    "imu_conf.rot_roll":  "IMU rotation roll offset (deg)",
    "imu_conf.rot_pitch": "IMU rotation pitch offset (deg)",
    "imu_conf.rot_yaw":   "IMU rotation yaw offset (deg)",
    "imu_conf.accel_offsets_0": "Accelerometer X offset",
    "imu_conf.accel_offsets_1": "Accelerometer Y offset",
    "imu_conf.accel_offsets_2": "Accelerometer Z offset",
    "imu_conf.gyro_offsets_0": "Gyroscope X offset",
    "imu_conf.gyro_offsets_1": "Gyroscope Y offset",
    "imu_conf.gyro_offsets_2": "Gyroscope Z offset",
    "imu_conf.gyro_offset_comp_fact_0": "Gyro offset compensation X",
    "imu_conf.gyro_offset_comp_fact_1": "Gyro offset compensation Y",
    "imu_conf.gyro_offset_comp_fact_2": "Gyro offset compensation Z",
    "imu_conf.gyro_offset_comp_clamp": "Gyro offset compensation clamp",
}


# ── Binary parser ─────────────────────────────────────────────────────

def parse_conf(data: bytes, fields: List[ConfField]) -> OrderedDict:
    """
    Parse a MCCONF or APPCONF binary payload.

    The payload starts with a 4-byte signature (already validated by caller).
    Fields are decoded sequentially in the order defined by the field table.

    Returns OrderedDict { field_name: value }.
    """
    values = OrderedDict()
    ind = 4  # skip signature (uint32)

    for f in fields:
        if ind >= len(data):
            values[f.name] = 0
            continue

        try:
            if f.dtype == "uint8":
                values[f.name] = data[ind]
                ind += 1
            elif f.dtype == "uint16":
                values[f.name] = struct.unpack_from(">H", data, ind)[0]
                ind += 2
            elif f.dtype == "int16":
                values[f.name] = struct.unpack_from(">h", data, ind)[0]
                ind += 2
            elif f.dtype == "uint32":
                values[f.name] = struct.unpack_from(">I", data, ind)[0]
                ind += 4
            elif f.dtype == "int32":
                values[f.name] = struct.unpack_from(">i", data, ind)[0]
                ind += 4
            elif f.dtype == "float32_auto":
                val, ind = _decode_float32_auto(data, ind)
                values[f.name] = val
            elif f.dtype == "float16":
                raw = struct.unpack_from(">h", data, ind)[0]
                values[f.name] = raw / f.scale
                ind += 2
            else:
                values[f.name] = 0
        except (struct.error, IndexError):
            values[f.name] = 0

    return values


def serialize_conf(values: OrderedDict, fields: List[ConfField],
                   signature: int) -> bytes:
    """
    Serialize an OrderedDict of config values back to binary.
    Returns bytes including the 4-byte signature header.
    """
    buf = bytearray()
    buf += struct.pack(">I", signature)

    for f in fields:
        val = values.get(f.name, 0)
        if f.dtype == "uint8":
            buf.append(int(val) & 0xFF)
        elif f.dtype == "uint16":
            buf += struct.pack(">H", int(val) & 0xFFFF)
        elif f.dtype == "int16":
            buf += struct.pack(">h", int(val))
        elif f.dtype == "uint32":
            buf += struct.pack(">I", int(val) & 0xFFFFFFFF)
        elif f.dtype == "int32":
            buf += struct.pack(">i", int(val))
        elif f.dtype == "float32_auto":
            buf += _encode_float32_auto(float(val))
        elif f.dtype == "float16":
            raw = int(round(float(val) * f.scale))
            buf += struct.pack(">h", max(-32768, min(32767, raw)))

    return bytes(buf)


# ── XML export / import ───────────────────────────────────────────────

def conf_to_xml(values: OrderedDict, root_tag: str) -> str:
    """
    Convert parsed config values to VESC-Tool compatible XML string.

    root_tag: "MCConfiguration" or "APPConfiguration"
    """
    root = ET.Element(root_tag)

    for name, val in values.items():
        elem = ET.SubElement(root, name)
        if isinstance(val, float):
            elem.text = f"{val:.6f}"
        else:
            elem.text = str(int(val))

    # Indent for readability
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    )


def xml_to_conf(xml_path: str) -> tuple:
    """
    Load config from a VESC-Tool XML file.

    Returns (root_tag, OrderedDict).
    root_tag is "MCConfiguration" or "APPConfiguration".
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    root_tag = root.tag

    # Select the right field table
    if root_tag == "MCConfiguration":
        fields = MCCONF_FIELDS
    elif root_tag == "APPConfiguration":
        fields = APPCONF_FIELDS
    else:
        raise ValueError(f"Unknown root tag: {root_tag}")

    # Build a name→dtype lookup
    dtype_map = {f.name: f.dtype for f in fields}

    values = OrderedDict()
    for f in fields:
        elem = root.find(f.name)
        if elem is not None and elem.text is not None:
            text = elem.text.strip()
            if f.dtype in ("float32_auto", "float16"):
                values[f.name] = float(text)
            else:
                values[f.name] = int(float(text))
        else:
            values[f.name] = 0

    return root_tag, values
