"""
Local signal processing engine for motor control quality analysis.

Computes metrics from real-time data: FFT, THD, ripple, settling time,
overshoot, steady-state error, and an overall quality score.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal


@dataclass
class MotorMetrics:
    """Comprehensive motor control quality metrics."""
    # Speed control quality
    rpm_mean: float = 0.0
    rpm_std: float = 0.0
    rpm_ripple_pct: float = 0.0
    target_rpm: float = 0.0
    steady_state_error: float = 0.0
    steady_state_error_pct: float = 0.0

    # Transient response
    settling_time: float = 0.0         # seconds to reach ±2% band
    overshoot_pct: float = 0.0
    rise_time: float = 0.0            # 10% to 90%

    # Current quality
    current_thd: float = 0.0          # Total Harmonic Distortion (%)
    current_ripple_pct: float = 0.0

    # Frequency analysis
    fft_dominant_freq: float = 0.0    # Hz
    fft_dominant_magnitude: float = 0.0
    fft_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    fft_magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))

    # Power quality
    power_mean: float = 0.0
    power_std: float = 0.0
    efficiency_indicator: float = 0.0  # simplified: useful_power / input_power

    # Overall score
    quality_score: float = 0.0         # 0-100

    # Raw time-domain data for step response plotting
    time_data: np.ndarray = field(default_factory=lambda: np.array([]))
    rpm_data: np.ndarray = field(default_factory=lambda: np.array([]))


def foc_lp_filter(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    VESC FOC 1st-order IIR low-pass filter (UTILS_LP_FAST equivalent).

    Firmware macro: value -= alpha * (value - sample)
    Equivalent to:  y[n] = y[n-1] * (1 - alpha) + x[n] * alpha

    Default alpha=0.1 matches MCCONF_FOC_CURRENT_FILTER_CONST (fc ≈ 399 Hz @ 25 kHz).
    """
    if len(data) == 0:
        return data.copy()
    out = np.empty_like(data, dtype=float)
    out[0] = data[0]
    coeff = 1.0 - alpha
    for i in range(1, len(data)):
        out[i] = out[i - 1] * coeff + data[i] * alpha
    return out


def compute_fft(data: np.ndarray, sample_rate: float) -> tuple:
    """
    Compute FFT magnitude spectrum.
    Returns (frequencies, magnitudes) excluding DC component.
    """
    if len(data) < 2:
        return np.array([]), np.array([])

    # Remove DC offset
    data_ac = data - np.mean(data)

    # Apply Hanning window
    windowed = data_ac * np.hanning(len(data_ac))

    fft_vals = np.fft.rfft(windowed)
    magnitudes = 2.0 * np.abs(fft_vals) / len(data)
    freqs = np.fft.rfftfreq(len(data), d=1.0 / sample_rate)

    # Skip DC
    return freqs[1:], magnitudes[1:]


def compute_thd(data: np.ndarray, sample_rate: float, fundamental_freq: Optional[float] = None) -> float:
    """
    Compute Total Harmonic Distortion.
    If fundamental_freq is not specified, the dominant frequency is used.
    Returns THD as percentage.
    """
    freqs, mags = compute_fft(data, sample_rate)
    if len(mags) < 3:
        return 0.0

    if fundamental_freq is None:
        fund_idx = np.argmax(mags)
    else:
        # Find bin closest to fundamental frequency, but search in a small window
        # to find the actual peak near the expected frequency
        fund_idx = np.argmin(np.abs(freqs - fundamental_freq))

        # Search ±5 bins around the expected frequency for the actual peak
        window = 5
        start_idx = max(0, fund_idx - window)
        end_idx = min(len(mags), fund_idx + window + 1)
        local_peak_idx = start_idx + np.argmax(mags[start_idx:end_idx])
        fund_idx = local_peak_idx

    fundamental_mag = mags[fund_idx]
    if fundamental_mag < 1e-10:
        return 0.0

    # Sum of squares of harmonics
    harmonic_sum_sq = 0.0
    for h in range(2, 11):  # 2nd through 10th harmonic
        target_freq = freqs[fund_idx] * h
        if target_freq > freqs[-1]:
            break
        h_idx = np.argmin(np.abs(freqs - target_freq))

        # Search ±3 bins around harmonic for actual peak
        window = 3
        start_idx = max(0, h_idx - window)
        end_idx = min(len(mags), h_idx + window + 1)
        local_peak_idx = start_idx + np.argmax(mags[start_idx:end_idx])
        harmonic_sum_sq += mags[local_peak_idx] ** 2

    thd = np.sqrt(harmonic_sum_sq) / fundamental_mag * 100.0

    # Cap THD at reasonable maximum (100% is already very high distortion)
    return min(thd, 100.0)


def compute_settling_time(data: np.ndarray, target: float, sample_rate: float,
                          tolerance_pct: float = 2.0) -> float:
    """
    Compute settling time: time for signal to stay within ±tolerance% of target.
    Scans from end of data backward to find last exit from the band.
    Returns seconds. Returns total duration if never settles.
    """
    if len(data) < 2 or target == 0:
        return 0.0

    band = abs(target) * tolerance_pct / 100.0
    lower = target - band
    upper = target + band

    # Find last index outside the band (scanning from end)
    for i in range(len(data) - 1, -1, -1):
        if data[i] < lower or data[i] > upper:
            return (i + 1) / sample_rate

    return 0.0  # Already within band from the start


def find_steady_state_start(data: np.ndarray, target: float, sample_rate: float,
                            tolerance_pct: float = 2.0, min_duration_s: float = 1.0) -> int:
    """
    Find when steady-state begins: first point where signal enters and stays in ±tolerance% band.
    Returns index of steady-state start, or len(data)//2 as fallback.
    """
    if len(data) < 10 or target == 0:
        return len(data) // 2

    band = abs(target) * tolerance_pct / 100.0
    lower = target - band
    upper = target + band
    min_samples = int(min_duration_s * sample_rate)

    # Find first index where signal enters band and stays
    for i in range(len(data) - min_samples):
        # Check if signal stays within band for min_duration
        segment = data[i:i + min_samples]
        if np.all((segment >= lower) & (segment <= upper)):
            return i

    # Fallback: use last portion
    return max(0, len(data) - min_samples)


def compute_overshoot(data: np.ndarray, target: float) -> float:
    """Compute overshoot as percentage of target value."""
    if target == 0 or len(data) == 0:
        return 0.0

    if target > 0:
        peak = np.max(data)
        if peak > target:
            return (peak - target) / abs(target) * 100.0
    else:
        peak = np.min(data)
        if peak < target:
            return (target - peak) / abs(target) * 100.0

    return 0.0


def compute_rise_time(data: np.ndarray, target: float, sample_rate: float) -> float:
    """Compute rise time: time from 10% to 90% of target value."""
    if target == 0 or len(data) < 2:
        return 0.0

    low = 0.1 * target
    high = 0.9 * target

    t_low = None
    t_high = None

    for i, v in enumerate(data):
        if target > 0:
            if t_low is None and v >= low:
                t_low = i
            if t_low is not None and t_high is None and v >= high:
                t_high = i
                break
        else:
            if t_low is None and v <= low:
                t_low = i
            if t_low is not None and t_high is None and v <= high:
                t_high = i
                break

    if t_low is not None and t_high is not None:
        return (t_high - t_low) / sample_rate
    return 0.0


def detect_step_start(data: np.ndarray, target: float, threshold_pct: float = 5.0) -> int:
    """
    Detect when the step response actually starts (motor begins moving).
    Returns the index where signal first exceeds threshold_pct of target.
    """
    if len(data) < 2 or target == 0:
        return 0

    threshold = abs(target) * threshold_pct / 100.0

    for i, v in enumerate(data):
        if abs(v) >= threshold:
            return max(0, i - 1)  # Return one sample before threshold crossing

    return 0


def analyze_speed_control(rpm_data: np.ndarray, target_rpm: float,
                          sample_rate: float,
                          current_data: Optional[np.ndarray] = None,
                          voltage_data: Optional[np.ndarray] = None,
                          input_current_data: Optional[np.ndarray] = None) -> MotorMetrics:
    """
    Comprehensive speed control quality analysis.

    Args:
        rpm_data: Array of RPM measurements
        target_rpm: Target RPM setpoint
        sample_rate: Data sample rate in Hz
        current_data: Optional motor current data
        voltage_data: Optional input voltage data
        input_current_data: Optional input current data
    """
    m = MotorMetrics()
    m.target_rpm = target_rpm

    if len(rpm_data) < 5:
        return m

    # Detect step start (when motor actually begins moving)
    step_start_idx = detect_step_start(rpm_data, target_rpm, threshold_pct=5.0)

    # Store raw data for time-domain plotting (adjusted to start from step)
    m.rpm_data = rpm_data.copy()
    # Time axis: shift so step starts at t=0
    m.time_data = (np.arange(len(rpm_data)) - step_start_idx) / sample_rate

    # Data from step start onwards (for transient analysis)
    step_data = rpm_data[step_start_idx:]

    # Find where steady-state actually begins (signal enters and stays in ±2% band)
    steady_start_in_step = find_steady_state_start(step_data, target_rpm, sample_rate,
                                                    tolerance_pct=2.0, min_duration_s=1.0)
    steady_data = step_data[steady_start_in_step:] if len(step_data) > 10 else step_data

    # Basic RPM statistics (steady-state portion only)
    if len(steady_data) > 5:
        m.rpm_mean = float(np.mean(steady_data))
        m.rpm_std = float(np.std(steady_data))
        m.rpm_ripple_pct = (m.rpm_std / abs(m.rpm_mean) * 100.0) if abs(m.rpm_mean) > 1 else 0.0
    else:
        # Fallback if steady data is too short
        m.rpm_mean = float(np.mean(step_data))
        m.rpm_std = float(np.std(step_data))
        m.rpm_ripple_pct = (m.rpm_std / abs(m.rpm_mean) * 100.0) if abs(m.rpm_mean) > 1 else 0.0

    # Steady-state error (from steady portion)
    m.steady_state_error = m.rpm_mean - target_rpm
    m.steady_state_error_pct = (
        abs(m.steady_state_error) / abs(target_rpm) * 100.0
        if abs(target_rpm) > 1 else 0.0
    )

    # Transient analysis (from step start, first 0.5 seconds)
    transient_samples = min(len(step_data), int(0.5 * sample_rate))
    transient_data = step_data[:transient_samples]
    m.settling_time = compute_settling_time(transient_data, target_rpm, sample_rate)
    m.overshoot_pct = compute_overshoot(transient_data, target_rpm)
    m.rise_time = compute_rise_time(transient_data, target_rpm, sample_rate)

    # FFT of RPM signal (steady-state only, after 0.5 second)
    steady_fft_start = int(0.5 * sample_rate)  # Skip first 0.5 second (transient)
    if steady_fft_start < len(step_data):
        fft_data = step_data[steady_fft_start:]
    else:
        fft_data = step_data
    freqs, mags = compute_fft(fft_data, sample_rate)
    m.fft_frequencies = freqs
    m.fft_magnitudes = mags

    if len(mags) > 0:
        dom_idx = np.argmax(mags)
        m.fft_dominant_freq = float(freqs[dom_idx])
        m.fft_dominant_magnitude = float(mags[dom_idx])

    # Current analysis
    if current_data is not None and len(current_data) > 5:
        m.current_thd = compute_thd(current_data, sample_rate)
        i_mean = np.mean(np.abs(current_data))
        i_std = np.std(current_data)
        m.current_ripple_pct = (i_std / i_mean * 100.0) if i_mean > 0.01 else 0.0

    # Power analysis
    if voltage_data is not None and input_current_data is not None:
        power = voltage_data * input_current_data
        m.power_mean = float(np.mean(power))
        m.power_std = float(np.std(power))

    # Quality score (0-100)
    m.quality_score = calculate_quality_score(m)

    return m


def calculate_quality_score(m: MotorMetrics) -> float:
    """
    Calculate overall quality score from 0 (poor) to 100 (excellent).

    Scoring breakdown (Speed PID focused):
    - Steady-state (50 pts total):
      - Ripple (25 pts): lower RPM ripple = better
      - SS error (25 pts): smaller error = better
    - Transient (40 pts total):
      - Settling time (15 pts): faster settling = better
      - Overshoot (15 pts): less overshoot = better
      - Rise time (10 pts): faster rise = better (but not too fast)
    - Current quality (10 pts):
      - THD (10 pts): reduced weight since speed PID can't directly fix this
    """
    score = 0.0

    # === STEADY-STATE (50 pts) ===
    # Ripple score (25 pts): <0.5% = 25, >10% = 0
    ripple_score = max(0, 25 * (1 - m.rpm_ripple_pct / 10.0))
    score += min(25, ripple_score)

    # SS error score (25 pts): <0.1% = 25, >5% = 0
    ss_score = max(0, 25 * (1 - m.steady_state_error_pct / 5.0))
    score += min(25, ss_score)

    # === TRANSIENT (40 pts) ===
    # Settling score (15 pts): <0.5s = 15, >5s = 0
    settling_score = max(0, 15 * (1 - m.settling_time / 5.0))
    score += min(15, settling_score)

    # Overshoot score (15 pts): <2% = 15, >30% = 0
    overshoot_score = max(0, 15 * (1 - m.overshoot_pct / 30.0))
    score += min(15, overshoot_score)

    # Rise time score (10 pts): optimal around 0.3-1.0s
    # Too fast (<0.1s) may indicate noise, too slow (>3s) is sluggish
    if m.rise_time <= 0:
        rise_score = 0  # No rise detected (already at target or no step)
    elif m.rise_time < 0.1:
        rise_score = 5  # Too fast, might be unstable
    elif m.rise_time < 0.3:
        rise_score = 8  # Fast but acceptable
    elif m.rise_time <= 1.0:
        rise_score = 10  # Optimal range
    elif m.rise_time <= 2.0:
        rise_score = 7  # Slow but acceptable
    elif m.rise_time <= 3.0:
        rise_score = 4  # Too slow
    else:
        rise_score = 0  # Very slow
    score += rise_score

    # === CURRENT QUALITY (10 pts) ===
    # THD score (10 pts): reduced weight - speed PID can't directly fix this
    thd_score = max(0, 10 * (1 - m.current_thd / 20.0))
    score += min(10, thd_score)

    return max(0, min(100, score))
