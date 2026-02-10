"""
CAN Position control quality analysis for RMD motors.

Computes position step response metrics: settling time, overshoot,
rise time, steady-state position error, position ripple, and quality score.
Reuses generic signal processing functions from signal_metrics.py.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .signal_metrics import (
    compute_fft,
    compute_settling_time,
    compute_overshoot,
    compute_rise_time,
    detect_step_start,
    find_steady_state_start,
)


@dataclass
class CanPositionMetrics:
    """Position control quality metrics for CAN/RMD motors."""

    # Position control
    target_pos_deg: float = 0.0
    pos_mean_deg: float = 0.0
    pos_std_deg: float = 0.0
    pos_ripple_deg: float = 0.0
    pos_ripple_pct: float = 0.0
    steady_state_error_deg: float = 0.0
    steady_state_error_pct: float = 0.0

    # Transient response
    settling_time_s: float = 0.0
    overshoot_pct: float = 0.0
    rise_time_s: float = 0.0

    # Torque during step
    torque_peak_a: float = 0.0
    torque_mean_a: float = 0.0

    # Frequency analysis (on position error signal)
    fft_dominant_freq: float = 0.0
    fft_dominant_magnitude: float = 0.0
    fft_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    fft_magnitudes: np.ndarray = field(default_factory=lambda: np.array([]))

    # Overall score
    quality_score: float = 0.0

    # Raw time-domain data for plotting
    time_data: np.ndarray = field(default_factory=lambda: np.array([]))
    pos_data: np.ndarray = field(default_factory=lambda: np.array([]))
    error_data: np.ndarray = field(default_factory=lambda: np.array([]))
    torque_data: np.ndarray = field(default_factory=lambda: np.array([]))


def analyze_position_step(
    pos_data: np.ndarray,
    target_deg: float,
    sample_rate: float,
    torque_data: Optional[np.ndarray] = None,
    initial_pos: float = 0.0,
) -> CanPositionMetrics:
    """
    Analyze position step response from CAN/RMD motor.

    Args:
        pos_data: Array of position measurements (degrees, unwrapped).
        target_deg: Target position command (degrees).
        sample_rate: Data sample rate in Hz.
        torque_data: Optional torque current during step.
        initial_pos: Position before step command was sent.
    """
    m = CanPositionMetrics()
    m.target_pos_deg = target_deg

    if len(pos_data) < 5:
        return m

    # Step magnitude (relative displacement expected)
    step_magnitude = target_deg - initial_pos
    if abs(step_magnitude) < 0.01:
        return m

    # Relative position from initial
    relative_pos = pos_data - initial_pos

    # Detect step start (when movement begins)
    step_start_idx = detect_step_start(relative_pos, step_magnitude, threshold_pct=5.0)

    # Store raw data for plotting, time axis centered on step start
    m.pos_data = pos_data.copy()
    m.time_data = (np.arange(len(pos_data)) - step_start_idx) / sample_rate
    m.error_data = target_deg - pos_data

    # Torque analysis
    if torque_data is not None and len(torque_data) > 0:
        m.torque_data = torque_data.copy()
        m.torque_peak_a = float(np.max(np.abs(torque_data)))
        m.torque_mean_a = float(np.mean(np.abs(torque_data)))

    # Step data from start onwards
    step_data = relative_pos[step_start_idx:]
    if len(step_data) < 5:
        return m

    # Find steady-state region (enters ±2% band and stays for 0.5s)
    steady_start = find_steady_state_start(
        step_data, step_magnitude, sample_rate,
        tolerance_pct=2.0, min_duration_s=0.5,
    )
    steady_data = step_data[steady_start:]

    # Steady-state statistics
    if len(steady_data) > 5:
        m.pos_mean_deg = float(np.mean(steady_data)) + initial_pos
        m.pos_std_deg = float(np.std(steady_data))
        m.pos_ripple_deg = float(np.max(steady_data) - np.min(steady_data))
        m.pos_ripple_pct = m.pos_ripple_deg / abs(step_magnitude) * 100.0
    else:
        m.pos_mean_deg = float(np.mean(step_data)) + initial_pos
        m.pos_std_deg = float(np.std(step_data))

    # Steady-state error
    m.steady_state_error_deg = m.pos_mean_deg - target_deg
    m.steady_state_error_pct = (
        abs(m.steady_state_error_deg) / abs(step_magnitude) * 100.0
    )

    # Transient analysis (up to 2 seconds from step start)
    transient_samples = min(len(step_data), int(2.0 * sample_rate))
    transient_data = step_data[:transient_samples]
    m.settling_time_s = compute_settling_time(transient_data, step_magnitude, sample_rate)
    m.overshoot_pct = compute_overshoot(transient_data, step_magnitude)
    m.rise_time_s = compute_rise_time(transient_data, step_magnitude, sample_rate)

    # FFT on position error signal (steady-state portion)
    if len(steady_data) > 20:
        error_steady = target_deg - (steady_data + initial_pos)
        freqs, mags = compute_fft(error_steady, sample_rate)
        m.fft_frequencies = freqs
        m.fft_magnitudes = mags
        if len(mags) > 0:
            dom_idx = np.argmax(mags)
            m.fft_dominant_freq = float(freqs[dom_idx])
            m.fft_dominant_magnitude = float(mags[dom_idx])

    # Quality score
    m.quality_score = calculate_position_quality_score(m)
    return m


def calculate_position_quality_score(m: CanPositionMetrics) -> float:
    """
    Position control quality score (0-100).

    Scoring:
      Steady-state (50 pts):
        - Position ripple  25 pts: <0.5% = 25, >5% = 0
        - SS error         25 pts: <0.1% = 25, >5% = 0
      Transient (50 pts):
        - Settling time    20 pts: <0.5s = 20, >5s = 0
        - Overshoot        20 pts: <2% = 20, >20% = 0 (stricter for position)
        - Rise time        10 pts: optimal 0.1-0.5s
    """
    score = 0.0

    # Ripple (25 pts)
    score += min(25.0, max(0.0, 25.0 * (1.0 - m.pos_ripple_pct / 5.0)))

    # SS error (25 pts)
    score += min(25.0, max(0.0, 25.0 * (1.0 - m.steady_state_error_pct / 5.0)))

    # Settling time (20 pts)
    score += min(20.0, max(0.0, 20.0 * (1.0 - m.settling_time_s / 5.0)))

    # Overshoot (20 pts) — heavier penalty for position control
    score += min(20.0, max(0.0, 20.0 * (1.0 - m.overshoot_pct / 20.0)))

    # Rise time (10 pts)
    rt = m.rise_time_s
    if rt <= 0:
        rise_score = 0.0
    elif rt < 0.05:
        rise_score = 5.0
    elif rt < 0.1:
        rise_score = 8.0
    elif rt <= 0.5:
        rise_score = 10.0
    elif rt <= 1.0:
        rise_score = 7.0
    elif rt <= 2.0:
        rise_score = 4.0
    else:
        rise_score = 0.0
    score += rise_score

    return max(0.0, min(100.0, score))
