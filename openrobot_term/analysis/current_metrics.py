"""
Current control loop quality analysis.

Computes metrics for current/torque control evaluation:
- Current ripple and THD
- Current tracking error
- Step response characteristics
- Bandwidth estimation
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from .signal_metrics import compute_fft, compute_thd


@dataclass
class CurrentMetrics:
    """Comprehensive current control quality metrics."""
    # Target and measured
    target_current: float = 0.0
    current_mean: float = 0.0
    current_std: float = 0.0

    # Tracking quality
    tracking_error: float = 0.0          # Absolute error (A)
    tracking_error_pct: float = 0.0      # Percentage error

    # Ripple and distortion
    current_ripple_pct: float = 0.0      # Peak-to-peak ripple %
    current_ripple_pp: float = 0.0       # Peak-to-peak ripple (A)
    current_thd: float = 0.0             # Total Harmonic Distortion %

    # Transient response (for step test)
    rise_time_ms: float = 0.0            # 10% to 90% (milliseconds)
    settling_time_ms: float = 0.0        # Time to reach ±5% band
    overshoot_pct: float = 0.0           # Overshoot percentage

    # Frequency analysis
    bandwidth_hz: float = 0.0            # -3dB bandwidth estimate
    fft_dominant_freq: float = 0.0
    fft_dominant_magnitude: float = 0.0
    fft_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    fft_magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))

    # Power analysis
    power_mean: float = 0.0
    efficiency_indicator: float = 0.0

    # Overall score
    quality_score: float = 0.0

    # Raw data for plotting
    time_data: np.ndarray = field(default_factory=lambda: np.array([]))
    current_data: np.ndarray = field(default_factory=lambda: np.array([]))
    target_data: np.ndarray = field(default_factory=lambda: np.array([]))


def detect_current_step_start(data: np.ndarray, target: float, threshold_pct: float = 10.0) -> int:
    """
    Detect when current step response starts.
    Returns index where current first exceeds threshold_pct of target.
    """
    if len(data) < 2 or target == 0:
        return 0

    threshold = abs(target) * threshold_pct / 100.0

    for i, v in enumerate(data):
        if abs(v) >= threshold:
            return max(0, i - 1)

    return 0


def compute_current_rise_time(data: np.ndarray, target: float, sample_rate: float) -> float:
    """
    Compute rise time: time from 10% to 90% of target current.
    Returns time in milliseconds.
    """
    if target == 0 or len(data) < 2:
        return 0.0

    low = 0.1 * abs(target)
    high = 0.9 * abs(target)

    t_low = None
    t_high = None

    for i, v in enumerate(data):
        if t_low is None and abs(v) >= low:
            t_low = i
        if t_low is not None and t_high is None and abs(v) >= high:
            t_high = i
            break

    if t_low is not None and t_high is not None:
        return (t_high - t_low) / sample_rate * 1000.0  # Convert to ms
    return 0.0


def compute_current_settling_time(data: np.ndarray, target: float, sample_rate: float,
                                   tolerance_pct: float = 5.0) -> float:
    """
    Compute settling time for current to stay within ±tolerance% of target.
    Returns time in milliseconds.
    """
    if len(data) < 2 or target == 0:
        return 0.0

    band = abs(target) * tolerance_pct / 100.0
    lower = abs(target) - band
    upper = abs(target) + band

    # Find last index outside the band (scanning from end)
    for i in range(len(data) - 1, -1, -1):
        if abs(data[i]) < lower or abs(data[i]) > upper:
            return (i + 1) / sample_rate * 1000.0  # Convert to ms

    return 0.0


def compute_current_overshoot(data: np.ndarray, target: float) -> float:
    """Compute overshoot as percentage of target current."""
    if target == 0 or len(data) == 0:
        return 0.0

    peak = np.max(np.abs(data))
    target_abs = abs(target)

    if peak > target_abs:
        return (peak - target_abs) / target_abs * 100.0
    return 0.0


def find_current_steady_state_start(data: np.ndarray, target: float, sample_rate: float,
                                     tolerance_pct: float = 5.0, min_duration_s: float = 0.1) -> int:
    """
    Find when steady-state begins for current control.
    Returns index of steady-state start.
    """
    if len(data) < 10 or target == 0:
        return len(data) // 2

    band = abs(target) * tolerance_pct / 100.0
    lower = abs(target) - band
    upper = abs(target) + band
    min_samples = int(min_duration_s * sample_rate)

    for i in range(len(data) - min_samples):
        segment = np.abs(data[i:i + min_samples])
        if np.all((segment >= lower) & (segment <= upper)):
            return i

    return max(0, len(data) - min_samples)


def analyze_current_step(current_data: np.ndarray, target_current: float,
                          sample_rate: float,
                          voltage_data: Optional[np.ndarray] = None) -> CurrentMetrics:
    """
    Analyze current control step response (Phase 1: Acceleration test).

    Args:
        current_data: Array of motor current measurements
        target_current: Target current setpoint (A)
        sample_rate: Data sample rate in Hz
        voltage_data: Optional voltage data for power analysis
    """
    m = CurrentMetrics()
    m.target_current = target_current

    if len(current_data) < 5:
        return m

    # Detect step start
    step_start_idx = detect_current_step_start(current_data, target_current, threshold_pct=10.0)

    # Store raw data for plotting (shifted to start from step)
    m.current_data = current_data.copy()
    m.time_data = (np.arange(len(current_data)) - step_start_idx) / sample_rate  # seconds
    m.target_data = np.full_like(current_data, target_current)

    # Data from step start onwards
    step_data = current_data[step_start_idx:]

    if len(step_data) < 5:
        return m

    # Transient analysis (first 100ms or available data)
    transient_samples = min(len(step_data), int(0.1 * sample_rate))
    transient_data = step_data[:transient_samples]

    m.rise_time_ms = compute_current_rise_time(transient_data, target_current, sample_rate)
    m.settling_time_ms = compute_current_settling_time(transient_data, target_current, sample_rate)
    m.overshoot_pct = compute_current_overshoot(transient_data, target_current)

    # Find steady-state region
    steady_start = find_current_steady_state_start(step_data, target_current, sample_rate)
    steady_data = step_data[steady_start:] if steady_start < len(step_data) else step_data[-50:]

    if len(steady_data) > 5:
        # Basic statistics
        m.current_mean = float(np.mean(np.abs(steady_data)))
        m.current_std = float(np.std(steady_data))

        # Tracking error
        m.tracking_error = abs(m.current_mean - abs(target_current))
        m.tracking_error_pct = (m.tracking_error / abs(target_current) * 100.0) if target_current != 0 else 0.0

        # Ripple
        m.current_ripple_pp = float(np.max(steady_data) - np.min(steady_data))
        m.current_ripple_pct = (m.current_ripple_pp / m.current_mean * 100.0) if m.current_mean > 0.01 else 0.0

        # THD
        m.current_thd = compute_thd(steady_data, sample_rate)

    # FFT analysis on steady-state
    if len(steady_data) > 10:
        freqs, mags = compute_fft(steady_data, sample_rate)
        m.fft_frequencies = freqs
        m.fft_magnitudes = mags

        if len(mags) > 0:
            dom_idx = np.argmax(mags)
            m.fft_dominant_freq = float(freqs[dom_idx])
            m.fft_dominant_magnitude = float(mags[dom_idx])

    # Power analysis
    if voltage_data is not None and len(voltage_data) == len(current_data):
        power = voltage_data * np.abs(current_data)
        m.power_mean = float(np.mean(power))

    # Quality score
    m.quality_score = calculate_current_quality_score(m)

    return m


def analyze_current_steady_state(current_data: np.ndarray, target_current: float,
                                  sample_rate: float, rpm_data: Optional[np.ndarray] = None,
                                  voltage_data: Optional[np.ndarray] = None) -> CurrentMetrics:
    """
    Analyze current control at steady-state (Phase 2: During speed control).

    Assumes motor is already at steady speed, focuses on current quality.
    """
    m = CurrentMetrics()
    m.target_current = target_current

    if len(current_data) < 10:
        return m

    # Use all data as steady-state (skip first 10% for any transients)
    skip = len(current_data) // 10
    steady_data = current_data[skip:]

    # Store raw data
    m.current_data = current_data.copy()
    m.time_data = np.arange(len(current_data)) / sample_rate  # seconds
    m.target_data = np.full_like(current_data, target_current)

    # Basic statistics
    m.current_mean = float(np.mean(np.abs(steady_data)))
    m.current_std = float(np.std(steady_data))

    # Tracking error (use mean as reference if target not specified)
    if target_current != 0:
        m.tracking_error = abs(m.current_mean - abs(target_current))
        m.tracking_error_pct = m.tracking_error / abs(target_current) * 100.0
    else:
        # No target specified, use mean as baseline
        m.tracking_error = m.current_std
        m.tracking_error_pct = (m.current_std / m.current_mean * 100.0) if m.current_mean > 0.01 else 0.0

    # Ripple analysis
    m.current_ripple_pp = float(np.max(steady_data) - np.min(steady_data))
    m.current_ripple_pct = (m.current_ripple_pp / m.current_mean * 100.0) if m.current_mean > 0.01 else 0.0

    # THD
    m.current_thd = compute_thd(steady_data, sample_rate)

    # FFT analysis
    freqs, mags = compute_fft(steady_data, sample_rate)
    m.fft_frequencies = freqs
    m.fft_magnitudes = mags

    if len(mags) > 0:
        dom_idx = np.argmax(mags)
        m.fft_dominant_freq = float(freqs[dom_idx])
        m.fft_dominant_magnitude = float(mags[dom_idx])

    # Power analysis
    if voltage_data is not None and len(voltage_data) >= len(steady_data):
        voltage_steady = voltage_data[skip:skip + len(steady_data)]
        power = voltage_steady * np.abs(steady_data)
        m.power_mean = float(np.mean(power))

    # No transient metrics for steady-state analysis
    m.rise_time_ms = 0.0
    m.settling_time_ms = 0.0
    m.overshoot_pct = 0.0

    # Quality score
    m.quality_score = calculate_current_quality_score(m)

    return m


def calculate_current_quality_score(m: CurrentMetrics) -> float:
    """
    Calculate overall current control quality score (0-100).

    Scoring breakdown:
    - Tracking accuracy (30 pts): Lower tracking error = better
    - Current ripple (25 pts): Lower ripple = better
    - THD (20 pts): Lower THD = better
    - Transient (25 pts): Fast rise, low overshoot, quick settling
    """
    score = 0.0

    # === TRACKING ACCURACY (30 pts) ===
    # <1% error = 30 pts, >10% error = 0 pts
    tracking_score = max(0, 30 * (1 - m.tracking_error_pct / 10.0))
    score += min(30, tracking_score)

    # === CURRENT RIPPLE (25 pts) ===
    # <2% ripple = 25 pts, >15% ripple = 0 pts
    ripple_score = max(0, 25 * (1 - m.current_ripple_pct / 15.0))
    score += min(25, ripple_score)

    # === THD (20 pts) ===
    # <3% THD = 20 pts, >20% THD = 0 pts
    thd_score = max(0, 20 * (1 - m.current_thd / 20.0))
    score += min(20, thd_score)

    # === TRANSIENT RESPONSE (25 pts) ===
    # Only if transient data available
    if m.rise_time_ms > 0 or m.settling_time_ms > 0:
        # Rise time (10 pts): <5ms = 10, >50ms = 0
        if m.rise_time_ms > 0:
            rise_score = max(0, 10 * (1 - m.rise_time_ms / 50.0))
        else:
            rise_score = 5  # No data, neutral score
        score += min(10, rise_score)

        # Settling time (8 pts): <20ms = 8, >100ms = 0
        if m.settling_time_ms > 0:
            settling_score = max(0, 8 * (1 - m.settling_time_ms / 100.0))
        else:
            settling_score = 4
        score += min(8, settling_score)

        # Overshoot (7 pts): <5% = 7, >30% = 0
        overshoot_score = max(0, 7 * (1 - m.overshoot_pct / 30.0))
        score += min(7, overshoot_score)
    else:
        # No transient data, give partial credit
        score += 12.5

    return max(0, min(100, score))
