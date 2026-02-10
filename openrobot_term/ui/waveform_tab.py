"""
FOC waveform capture and display tab — VESC-Tool style.

Features:
- Auto motor start in speed mode
- Wait for steady state before capture
- 3-phase current waveform analysis (THD, balance, amplitude)
- AI-powered FFT/waveform interpretation
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QDoubleSpinBox, QMessageBox, QCheckBox, QGroupBox,
    QGridLayout, QTextEdit, QComboBox, QSplitter,
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal

import numpy as np
import pyqtgraph as pg

from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import WaveformSamples, build_sample_request, build_set_rpm
from ..analysis.signal_metrics import compute_fft, compute_thd, foc_lp_filter
from .plot_style import style_plot, graph_pen, Crosshair, style_legend, GRAPH_COLORS

CAPTURE_TIMEOUT_MS = 10000
MOTOR_SETTLE_MS = 2000  # Wait 2 seconds for motor to reach steady state


class WaveformTab(QWidget):
    # Signals for thread-safe AI result delivery
    _ai_result_signal = pyqtSignal(object)
    _ai_error_signal = pyqtSignal(str)

    def __init__(self, transport: SerialTransport):
        super().__init__()
        self._transport = transport
        self._samples = WaveformSamples()
        self._capturing = False
        self._waiting_for_settle = False
        self._timeout_timer = QTimer(self)
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.timeout.connect(self._on_capture_timeout)

        # Motor settle timer
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.timeout.connect(self._on_motor_settled)

        # Periodic RPM command timer (VESC needs repeated commands to keep motor running)
        self._rpm_keepalive_timer = QTimer(self)
        self._rpm_keepalive_timer.timeout.connect(self._send_rpm_keepalive)
        self._target_rpm = 0

        # Delayed auto-stop timer (stop motor 1 second after capture)
        self._auto_stop_timer = QTimer(self)
        self._auto_stop_timer.setSingleShot(True)
        self._auto_stop_timer.timeout.connect(self._on_auto_stop)

        # Cache for last analysis data (used by AI interpret)
        self._last_analysis_data = None

        # Connect AI signals
        self._ai_result_signal.connect(self._on_ai_result)
        self._ai_error_signal.connect(self._on_ai_error)

        layout = QVBoxLayout(self)

        # Controls Row 1: Motor + Sample settings + buttons
        ctrl1 = QHBoxLayout()
        layout.addLayout(ctrl1)

        self.auto_start_cb = QCheckBox("Auto Start")
        self.auto_start_cb.setChecked(True)
        self.auto_start_cb.setToolTip("Automatically start motor in speed mode before capture")
        ctrl1.addWidget(self.auto_start_cb)

        ctrl1.addWidget(QLabel("eRPM:"))
        self.rpm_spin = QSpinBox()
        self.rpm_spin.setRange(1000, 50000)
        self.rpm_spin.setValue(6000)
        self.rpm_spin.setSingleStep(1000)
        ctrl1.addWidget(self.rpm_spin)

        ctrl1.addWidget(QLabel("Settle:"))
        self.settle_spin = QSpinBox()
        self.settle_spin.setRange(500, 5000)
        self.settle_spin.setValue(2000)
        self.settle_spin.setSuffix(" ms")
        self.settle_spin.setSingleStep(500)
        self.settle_spin.setToolTip("Time to wait for motor to reach steady state")
        ctrl1.addWidget(self.settle_spin)

        ctrl1.addWidget(QLabel("N:"))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(100, 5000)
        self.samples_spin.setValue(1000)
        self.samples_spin.setToolTip("Number of samples")
        ctrl1.addWidget(self.samples_spin)

        ctrl1.addWidget(QLabel("Dec:"))
        self.decim_spin = QSpinBox()
        self.decim_spin.setRange(1, 20)
        self.decim_spin.setValue(1)
        self.decim_spin.setToolTip("Decimation factor")
        ctrl1.addWidget(self.decim_spin)

        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.start_capture)
        ctrl1.addWidget(self.capture_btn)

        self.stop_btn = QPushButton("Stop Motor")
        self.stop_btn.clicked.connect(self._stop_motor)
        ctrl1.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_plot)
        ctrl1.addWidget(self.clear_btn)

        ctrl1.addStretch()

        # Controls Row 2: Filter & display options + AI
        ctrl2 = QHBoxLayout()
        layout.addLayout(ctrl2)

        ctrl2.addWidget(QLabel("Filter \u03b1:"))
        self.filter_alpha_spin = QDoubleSpinBox()
        self.filter_alpha_spin.setRange(0.01, 1.0)
        self.filter_alpha_spin.setValue(0.1)
        self.filter_alpha_spin.setSingleStep(0.01)
        self.filter_alpha_spin.setDecimals(2)
        self.filter_alpha_spin.setToolTip(
            "FOC IIR low-pass filter constant (UTILS_LP_FAST)\n"
            "0.1 = VESC default (fc\u2248399Hz@25kHz)\n"
            "Lower = stronger smoothing"
        )
        ctrl2.addWidget(self.filter_alpha_spin)

        ctrl2.addSpacing(20)

        # AI Interpret button and model selector
        ctrl2.addWidget(QLabel("AI:"))
        self.ai_model_combo = QComboBox()
        self.ai_model_combo.addItems(["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"])
        self.ai_model_combo.setMaximumWidth(100)
        ctrl2.addWidget(self.ai_model_combo)

        self.ai_lang_combo = QComboBox()
        self.ai_lang_combo.addItems(["\ud55c\uad6d\uc5b4", "English"])
        self.ai_lang_combo.setMaximumWidth(80)
        ctrl2.addWidget(self.ai_lang_combo)

        self.ai_interpret_btn = QPushButton("AI Interpret")
        self.ai_interpret_btn.setToolTip("Ask AI to interpret FFT peaks and waveform quality")
        self.ai_interpret_btn.clicked.connect(self._run_ai_interpret)
        self.ai_interpret_btn.setEnabled(False)
        ctrl2.addWidget(self.ai_interpret_btn)

        self.show_raw_api_cb = QCheckBox("Show Raw")
        self.show_raw_api_cb.setToolTip("Show raw API request/response for debugging")
        ctrl2.addWidget(self.show_raw_api_cb)

        self.ai_clear_btn = QPushButton("Clear AI Interp.")
        self.ai_clear_btn.clicked.connect(lambda: self.ai_text.clear())
        ctrl2.addWidget(self.ai_clear_btn)

        ctrl2.addSpacing(10)
        self.info_label = QLabel("")
        ctrl2.addWidget(self.info_label)
        ctrl2.addStretch()

        # Analysis results panel
        self.analysis_group = QGroupBox("Waveform Analysis")
        analysis_layout = QGridLayout(self.analysis_group)

        # Row 0: Phase amplitudes
        analysis_layout.addWidget(QLabel("Phase A Amp:"), 0, 0)
        self.amp_a_label = QLabel("-")
        analysis_layout.addWidget(self.amp_a_label, 0, 1)

        analysis_layout.addWidget(QLabel("Phase B Amp:"), 0, 2)
        self.amp_b_label = QLabel("-")
        analysis_layout.addWidget(self.amp_b_label, 0, 3)

        analysis_layout.addWidget(QLabel("Phase C Amp:"), 0, 4)
        self.amp_c_label = QLabel("-")
        analysis_layout.addWidget(self.amp_c_label, 0, 5)

        # Row 1: THD values
        analysis_layout.addWidget(QLabel("THD A:"), 1, 0)
        self.thd_a_label = QLabel("-")
        analysis_layout.addWidget(self.thd_a_label, 1, 1)

        analysis_layout.addWidget(QLabel("THD B:"), 1, 2)
        self.thd_b_label = QLabel("-")
        analysis_layout.addWidget(self.thd_b_label, 1, 3)

        analysis_layout.addWidget(QLabel("THD C:"), 1, 4)
        self.thd_c_label = QLabel("-")
        analysis_layout.addWidget(self.thd_c_label, 1, 5)

        # Row 2: Balance and frequency
        analysis_layout.addWidget(QLabel("Phase Balance:"), 2, 0)
        self.balance_label = QLabel("-")
        analysis_layout.addWidget(self.balance_label, 2, 1)

        analysis_layout.addWidget(QLabel("Fund. Freq (raw/filt):"), 2, 2)
        self.freq_label = QLabel("-")
        analysis_layout.addWidget(self.freq_label, 2, 3)

        analysis_layout.addWidget(QLabel("Quality:"), 2, 4)
        self.quality_label = QLabel("-")
        self.quality_label.setStyleSheet("font-weight: bold;")
        analysis_layout.addWidget(self.quality_label, 2, 5)

        layout.addWidget(self.analysis_group)

        # Splitter: plots on top, AI text on bottom
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self._splitter, stretch=1)

        # Plots container
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)

        # Phase current plot — VESC-Tool colors
        self.plot = pg.PlotWidget()
        style_plot(self.plot, title="FOC Waveform - Phase Currents",
                   left_label="Current", left_unit="A",
                   bottom_label="Sample", bottom_unit="")
        style_legend(self.plot)

        # Raw curves (semi-transparent, thin)
        from PyQt6.QtGui import QColor as _QC
        raw_alpha = 80  # 0-255 transparency
        pen_a_raw = pg.mkPen(_QC(200, 52, 52, raw_alpha), width=1)
        pen_b_raw = pg.mkPen(_QC(127, 200, 127, raw_alpha), width=1)
        pen_c_raw = pg.mkPen(_QC(77, 127, 196, raw_alpha), width=1)
        self.curve_a = self.plot.plot(pen=pen_a_raw, name="Phase A (raw)")
        self.curve_b = self.plot.plot(pen=pen_b_raw, name="Phase B (raw)")
        self.curve_c = self.plot.plot(pen=pen_c_raw, name="Phase C (raw)")

        # Filtered curves (solid, thick)
        self.curve_a_filt = self.plot.plot(pen=graph_pen(1, width=2.0), name="Phase A (filtered)")
        self.curve_b_filt = self.plot.plot(pen=graph_pen(2, width=2.0), name="Phase B (filtered)")
        self.curve_c_filt = self.plot.plot(pen=graph_pen(0, width=2.0), name="Phase C (filtered)")

        Crosshair(self.plot)  # crosshair OK on static capture plots
        plots_layout.addWidget(self.plot)

        # FFT plot
        self.fft_plot = pg.PlotWidget()
        style_plot(self.fft_plot, title="FFT - Phase A",
                   left_label="Magnitude", left_unit="",
                   bottom_label="Frequency", bottom_unit="Hz")
        style_legend(self.fft_plot)
        self.curve_fft_raw = self.fft_plot.plot(pen=graph_pen(4), name="Raw")       # yellow
        self.curve_fft_filt = self.fft_plot.plot(pen=graph_pen(5), name="Filtered")  # cyan

        # Fundamental frequency marker line
        self._fund_marker = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#ff4444', width=1.5, style=Qt.PenStyle.DashLine),
        )
        self._fund_marker.setVisible(False)
        self.fft_plot.addItem(self._fund_marker)

        # Label for fundamental marker
        from .plot_style import TEXT_LIGHT
        self._fund_label = pg.TextItem(color=TEXT_LIGHT, anchor=(0, 1))
        from PyQt6.QtGui import QFont as _QF
        self._fund_label.setFont(_QF("Consolas", 8))
        self._fund_label.setVisible(False)
        self.fft_plot.addItem(self._fund_label, ignoreBounds=True)

        Crosshair(self.fft_plot)
        plots_layout.addWidget(self.fft_plot)

        self._splitter.addWidget(plots_widget)

        # AI Interpretation text area
        self.ai_text = QTextEdit()
        self.ai_text.setReadOnly(True)
        self.ai_text.setPlaceholderText(
            "AI waveform interpretation will appear here.\n"
            "Capture a waveform first, then click 'AI Interpret'."
        )
        self.ai_text.setMaximumHeight(200)
        self._splitter.addWidget(self.ai_text)

        # Set splitter proportions: plots get most space
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 1)

    def start_capture(self):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to VESC first.")
            return

        self._samples = WaveformSamples()
        self.capture_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        if self.auto_start_cb.isChecked():
            # Start motor first, then wait for settle time
            rpm = self.rpm_spin.value()
            settle_ms = self.settle_spin.value()

            self._target_rpm = rpm
            self._transport.send_packet(build_set_rpm(rpm))
            # Start keepalive timer to maintain motor running (VESC timeout is typically 250ms)
            self._rpm_keepalive_timer.start(100)  # Send every 100ms

            self._waiting_for_settle = True
            self.capture_btn.setText(f"Starting motor...")
            self.info_label.setText(f"Motor starting at {rpm} eRPM, waiting {settle_ms}ms...")

            self._settle_timer.start(settle_ms)
        else:
            # Direct capture (motor should already be running)
            self._request_samples()

    def _on_motor_settled(self):
        """Called when motor has reached steady state."""
        self._waiting_for_settle = False
        self.info_label.setText("Motor steady, capturing waveform...")
        self._request_samples()

    def _request_samples(self):
        """Send sample request to VESC."""
        self._capturing = True
        n = self.samples_spin.value()
        d = self.decim_spin.value()
        self._transport.send_packet(build_sample_request(n, d))
        self.capture_btn.setText("Capturing...")
        self.info_label.setText(f"Requesting {n} samples (dec={d})...")
        self._timeout_timer.start(CAPTURE_TIMEOUT_MS)

    def _send_rpm_keepalive(self):
        """Send periodic RPM command to keep motor running."""
        if self._transport.is_connected() and self._target_rpm != 0:
            self._transport.send_packet(build_set_rpm(self._target_rpm))

    def _stop_motor(self):
        """Stop motor (set RPM to 0)."""
        self._rpm_keepalive_timer.stop()
        self._auto_stop_timer.stop()
        self._target_rpm = 0
        if self._transport.is_connected():
            self._transport.send_packet(build_set_rpm(0))
            # Don't overwrite analysis info; append motor status
            cur = self.info_label.text()
            if cur and "samples" in cur:
                self.info_label.setText(cur + "  |  Motor stopped")
            else:
                self.info_label.setText("Motor stopped")

    def _schedule_auto_stop(self, delay_ms: int = 1000):
        """Schedule motor stop after delay."""
        if self.auto_start_cb.isChecked():
            self._auto_stop_timer.start(delay_ms)

    def _on_auto_stop(self):
        """Called when auto-stop timer fires."""
        self._stop_motor()

    def _on_capture_timeout(self):
        if not self._capturing:
            return
        received = self._samples.num_samples
        self._capturing = False

        self.capture_btn.setEnabled(True)
        self.capture_btn.setText("Capture")
        self.stop_btn.setEnabled(True)

        if received > 0:
            self._update_plots()
            self._analyze_waveform()
            self.info_label.setText(
                f"Timeout: captured {received} samples (partial)")
            # Auto-stop motor 1 second after partial capture
            self._schedule_auto_stop(1000)
        else:
            # No response - stop motor immediately
            self._stop_motor()
            self.info_label.setText(
                "Timeout: no response. Motor must be running for COMM_SAMPLE_PRINT.")
            self.info_label.setStyleSheet("color: #ff8800;")

    def on_sample_data(self, data: bytes):
        if not self._capturing:
            return
        self.info_label.setStyleSheet("")

        new_samples = WaveformSamples.from_payload(data)
        # Extend raw current data (curr0, curr1)
        self._samples.curr0.extend(new_samples.curr0)
        self._samples.curr1.extend(new_samples.curr1)
        # Also extend voltage data
        self._samples.ph1_voltage.extend(new_samples.ph1_voltage)
        self._samples.ph2_voltage.extend(new_samples.ph2_voltage)
        self._samples.ph3_voltage.extend(new_samples.ph3_voltage)
        self._samples.f_sw.extend(new_samples.f_sw)
        self._samples.num_samples += new_samples.num_samples

        expected = self.samples_spin.value()
        self.info_label.setText(f"Received {self._samples.num_samples}/{expected} samples")

        if self._samples.num_samples >= expected or new_samples.num_samples == 0:
            self._timeout_timer.stop()
            self._capturing = False

            self._update_plots()
            self._analyze_waveform()
            self.capture_btn.setEnabled(True)
            self.capture_btn.setText("Capture")
            self.stop_btn.setEnabled(True)

            # Auto-stop motor 1 second after capture
            self._schedule_auto_stop(1000)

    def _update_plots(self):
        n = self._samples.num_samples
        if n == 0:
            return

        x = np.arange(n)
        a = np.array(self._samples.phase_a[:n])
        b = np.array(self._samples.phase_b[:n])
        c = np.array(self._samples.phase_c[:n])

        # Raw curves
        self.curve_a.setData(x, a)
        self.curve_b.setData(x, b)
        self.curve_c.setData(x, c)

        # Filtered curves (FOC IIR low-pass)
        alpha = self.filter_alpha_spin.value()
        af = foc_lp_filter(a, alpha)
        bf = foc_lp_filter(b, alpha)
        cf = foc_lp_filter(c, alpha)
        self.curve_a_filt.setData(x, af)
        self.curve_b_filt.setData(x, bf)
        self.curve_c_filt.setData(x, cf)

        # Cache for FFT update
        self._cached_a_raw = a
        self._cached_a_filt = af

        # Update FFT
        self._update_fft()

        self.info_label.setText(f"Captured {n} samples")

    def _get_sample_rate(self) -> float:
        """
        Get actual sample rate from VESC f_sw data.

        VESC sends f_sw (switching frequency) with each sample.
        Effective sample rate = f_sw / decimation.
        """
        decimation = self.decim_spin.value()
        if self._samples.f_sw:
            # Use median f_sw (avoids outliers)
            f_sw = float(np.median(self._samples.f_sw))
            if f_sw > 0:
                return f_sw / decimation
        # Fallback: typical VESC FOC switching freq
        return 25000.0 / decimation

    def _find_fundamental(self, data: np.ndarray, sample_rate: float,
                          expected_freq: float) -> float:
        """
        Find fundamental frequency near expected electrical frequency.

        FOC motor phase currents have strong harmonics that can exceed
        the fundamental in magnitude. Instead of global argmax, search
        within +/-50% of the expected frequency for the local peak.
        """
        freqs, mags = compute_fft(data, sample_rate)
        if len(mags) == 0 or expected_freq <= 0:
            return 0.0

        # Search window: 50% to 150% of expected frequency
        low = expected_freq * 0.5
        high = expected_freq * 1.5
        mask = (freqs >= low) & (freqs <= high)

        if not np.any(mask):
            # Fallback: argmax of full spectrum
            return float(freqs[np.argmax(mags)])

        # Find peak within the search window
        masked_mags = mags.copy()
        masked_mags[~mask] = 0
        peak_idx = np.argmax(masked_mags)
        return float(freqs[peak_idx])

    def _update_fft(self):
        """Update FFT plot based on Raw/Filtered checkboxes."""
        if not hasattr(self, '_cached_a_raw') or self._cached_a_raw is None:
            return
        n = len(self._cached_a_raw)
        if n < 2:
            return

        sample_rate = self._get_sample_rate()
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

        max_mag = 0.0

        fft_raw = np.abs(np.fft.rfft(self._cached_a_raw - np.mean(self._cached_a_raw)))
        self.curve_fft_raw.setData(freqs[1:], fft_raw[1:])
        max_mag = max(max_mag, float(np.max(fft_raw[1:])) if len(fft_raw) > 1 else 0)

        fft_filt = np.abs(np.fft.rfft(self._cached_a_filt - np.mean(self._cached_a_filt)))
        self.curve_fft_filt.setData(freqs[1:], fft_filt[1:])
        max_mag = max(max_mag, float(np.max(fft_filt[1:])) if len(fft_filt) > 1 else 0)

        # Update fundamental frequency marker
        erpm = self.rpm_spin.value()
        expected_freq = erpm / 60.0
        if expected_freq > 0:
            fund_freq = self._find_fundamental(self._cached_a_raw, sample_rate, expected_freq)
            if fund_freq > 0:
                self._fund_marker.setValue(fund_freq)
                self._fund_marker.setVisible(True)
                self._fund_label.setText(f" Fund: {fund_freq:.0f} Hz")
                self._fund_label.setPos(fund_freq, max_mag * 0.9)
                self._fund_label.setVisible(True)
            else:
                self._fund_marker.setVisible(False)
                self._fund_label.setVisible(False)

    def _analyze_waveform(self):
        """Analyze captured waveform using FOC-filtered data."""
        n = self._samples.num_samples
        if n < 10:
            return

        a_raw = np.array(self._samples.phase_a[:n])
        b_raw = np.array(self._samples.phase_b[:n])
        c_raw = np.array(self._samples.phase_c[:n])

        # Apply FOC IIR low-pass filter (same as VESC firmware UTILS_LP_FAST)
        alpha = self.filter_alpha_spin.value()
        a = foc_lp_filter(a_raw, alpha)
        b = foc_lp_filter(b_raw, alpha)
        c = foc_lp_filter(c_raw, alpha)

        # Get actual sample rate from VESC f_sw
        sample_rate = self._get_sample_rate()

        # Calculate amplitudes from filtered data (peak-to-peak / 2)
        amp_a = (np.max(a) - np.min(a)) / 2
        amp_b = (np.max(b) - np.min(b)) / 2
        amp_c = (np.max(c) - np.min(c)) / 2

        self.amp_a_label.setText(f"{amp_a:.2f} A")
        self.amp_b_label.setText(f"{amp_b:.2f} A")
        self.amp_c_label.setText(f"{amp_c:.2f} A")

        # THD from filtered data (reference info only, not scored)
        erpm = self.rpm_spin.value()
        expected_electrical_freq = erpm / 60.0  # Hz
        thd_a = compute_thd(a, sample_rate, fundamental_freq=expected_electrical_freq)
        thd_b = compute_thd(b, sample_rate, fundamental_freq=expected_electrical_freq)
        thd_c = compute_thd(c, sample_rate, fundamental_freq=expected_electrical_freq)

        self.thd_a_label.setText(f"{thd_a:.1f}%")
        self.thd_b_label.setText(f"{thd_b:.1f}%")
        self.thd_c_label.setText(f"{thd_c:.1f}%")

        # Phase balance from filtered amplitudes
        avg_amp = (amp_a + amp_b + amp_c) / 3
        if avg_amp > 0.01:
            max_dev = max(abs(amp_a - avg_amp), abs(amp_b - avg_amp), abs(amp_c - avg_amp))
            balance_pct = (1 - max_dev / avg_amp) * 100
        else:
            balance_pct = 0

        if balance_pct >= 95:
            self.balance_label.setText(f"{balance_pct:.1f}% (Excellent)")
            self.balance_label.setStyleSheet("color: #00ff00;")
        elif balance_pct >= 85:
            self.balance_label.setText(f"{balance_pct:.1f}% (Good)")
            self.balance_label.setStyleSheet("color: #88ff00;")
        elif balance_pct >= 70:
            self.balance_label.setText(f"{balance_pct:.1f}% (Fair)")
            self.balance_label.setStyleSheet("color: #ffaa00;")
        else:
            self.balance_label.setText(f"{balance_pct:.1f}% (Poor)")
            self.balance_label.setStyleSheet("color: #ff4400;")

        # Fundamental frequency — search near expected eRPM/60 (not global argmax)
        # FOC motor currents have strong harmonics; argmax often picks a harmonic
        fund_raw = self._find_fundamental(a_raw, sample_rate, expected_electrical_freq)
        fund_filt = self._find_fundamental(a, sample_rate, expected_electrical_freq)
        if fund_raw > 0 or fund_filt > 0:
            self.freq_label.setText(f"{fund_raw:.1f} / {fund_filt:.1f} Hz")
        else:
            self.freq_label.setText("-")

        # Overall quality score (0-100)
        # Based on: Phase Balance (50%) + Amplitude consistency (50%)
        # Evaluated on FOC-filtered data (same filter as actual controller)

        # Phase Balance score (50 pts): 85%+ = 50, 50% = 0
        balance_score = max(0, min(50, 50 * (balance_pct - 50) / 35))

        # Amplitude consistency (50 pts): CV <5% = 50, CV 30%+ = 0
        amp_std = np.std([amp_a, amp_b, amp_c])
        amp_cv = (amp_std / avg_amp * 100) if avg_amp > 0.01 else 100
        amp_score = max(0, min(50, 50 * (1 - amp_cv / 30.0)))

        quality = min(100, balance_score + amp_score)

        if quality >= 85:
            self.quality_label.setText(f"{quality:.0f}/100 (Excellent)")
            self.quality_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        elif quality >= 70:
            self.quality_label.setText(f"{quality:.0f}/100 (Good)")
            self.quality_label.setStyleSheet("font-weight: bold; color: #88ff00;")
        elif quality >= 55:
            self.quality_label.setText(f"{quality:.0f}/100 (Fair)")
            self.quality_label.setStyleSheet("font-weight: bold; color: #ffaa00;")
        else:
            self.quality_label.setText(f"{quality:.0f}/100 (Poor)")
            self.quality_label.setStyleSheet("font-weight: bold; color: #ff4400;")

        sr = self._get_sample_rate()
        self.info_label.setText(
            f"Captured {n} samples (\u03b1={alpha:.2f}, Fs={sr:.0f}Hz)"
        )

        # Extract top FFT peaks with pre-computed harmonic analysis
        fft_peaks = self._extract_fft_peaks(
            a_raw, sample_rate, fund_raw if fund_raw > 0 else expected_electrical_freq, top_n=10
        )

        # Switching frequency (actual, not sampling)
        f_sw_actual = sr * self.decim_spin.value()  # Undo decimation to get switching freq
        nyquist = sr / 2

        # Cache analysis data for AI interpretation
        self._last_analysis_data = {
            "erpm": erpm,
            "sample_rate_hz": round(sr, 1),
            "switching_freq_hz": round(f_sw_actual, 1),
            "nyquist_hz": round(nyquist, 1),
            "fft_resolution_hz": round(sr / n, 2),
            "filter_alpha": alpha,
            "num_samples": n,
            "phase_amplitudes_A": {
                "A": round(amp_a, 3),
                "B": round(amp_b, 3),
                "C": round(amp_c, 3),
            },
            "thd_pct": {
                "A": round(thd_a, 2),
                "B": round(thd_b, 2),
                "C": round(thd_c, 2),
            },
            "phase_balance_pct": round(balance_pct, 2),
            "quality_score": round(quality, 1),
            "fundamental_freq_hz": {
                "raw": round(fund_raw, 2),
                "filtered": round(fund_filt, 2),
            },
            "expected_electrical_freq_hz": round(expected_electrical_freq, 2),
            "aliasing_note": (
                f"Sampling at {sr:.0f} Hz (switching freq {f_sw_actual:.0f} Hz / 2). "
                f"Nyquist = {nyquist:.0f} Hz. "
                f"Switching sidebands at {f_sw_actual:.0f} +/- n*f_e alias onto n*f_e harmonics. "
                f"Frequencies above {nyquist:.0f} Hz fold back."
            ),
            "fft_peaks_with_harmonic_analysis": fft_peaks,
        }

        # Enable AI button
        self.ai_interpret_btn.setEnabled(True)

    def _extract_fft_peaks(self, data: np.ndarray, sample_rate: float,
                           fundamental_freq: float, top_n: int = 10) -> list:
        """
        Extract top N FFT peaks with pre-computed harmonic analysis.

        For each peak, calculates:
        - Which harmonic of f_e it's closest to (integer)
        - Deviation from the nearest harmonic (Hz)
        - Whether it's odd/even harmonic
        - Relative magnitude vs fundamental
        """
        freqs, mags = compute_fft(data, sample_rate)
        if len(mags) == 0 or fundamental_freq <= 0:
            return []

        # Find fundamental magnitude for relative comparison
        fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fund_mag = float(mags[fund_idx]) if fund_idx < len(mags) else 1.0
        if fund_mag < 1e-10:
            fund_mag = 1.0

        # Find top peaks (sorted by magnitude)
        indices = np.argsort(mags)[::-1][:top_n]
        peaks = []
        for idx in indices:
            if mags[idx] <= 0:
                continue
            freq = float(freqs[idx])
            mag = float(mags[idx])

            # Determine nearest harmonic number
            harmonic_n = round(freq / fundamental_freq) if fundamental_freq > 0 else 0
            nearest_harmonic_freq = harmonic_n * fundamental_freq
            deviation_hz = freq - nearest_harmonic_freq

            peak_info = {
                "freq_hz": round(freq, 2),
                "magnitude": round(mag, 4),
                "relative_to_fundamental_dB": round(20 * np.log10(mag / fund_mag), 1) if mag > 0 else -999,
                "nearest_harmonic": harmonic_n,
                "harmonic_type": "even" if harmonic_n % 2 == 0 else "odd",
                "deviation_from_harmonic_hz": round(deviation_hz, 2),
            }
            peaks.append(peak_info)
        return peaks

    # ── AI Interpretation ─────────────────────────────────────────────

    def _run_ai_interpret(self):
        """Send waveform analysis data to LLM for interpretation."""
        if self._last_analysis_data is None:
            QMessageBox.warning(self, "No data", "Capture a waveform first.")
            return

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key or openai_key == "your-api-key-here":
            QMessageBox.warning(
                self, "API Key",
                "Set OPENAI_API_KEY in .env file to use AI interpretation."
            )
            return

        self.ai_interpret_btn.setEnabled(False)
        self.ai_interpret_btn.setText("Analyzing...")
        self.ai_text.setPlainText("Sending waveform data to AI for interpretation...")

        model_name = self.ai_model_combo.currentText()
        lang = "Korean" if self.ai_lang_combo.currentText() == "\ud55c\uad6d\uc5b4" else "English"

        import threading

        waveform_data = self._last_analysis_data.copy()

        def _run():
            try:
                from ..analysis.llm_advisor import WaveformAdvisor
                advisor = WaveformAdvisor(
                    api_key=openai_key, model=model_name, language=lang
                )
                result = advisor.analyze_waveform(waveform_data)
                self._ai_result_signal.emit(result)
            except Exception as e:
                import traceback
                self._ai_error_signal.emit(f"Error: {e}\n{traceback.format_exc()}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _on_ai_result(self, result):
        """Handle AI interpretation result (called on main thread via signal)."""
        self.ai_interpret_btn.setEnabled(True)
        self.ai_interpret_btn.setText("AI Interpret")

        # Format the result text
        text = ""

        # Health indicator
        health = result.motor_health
        if health:
            health_colors = {
                "excellent": "#00ff00", "good": "#88ff00",
                "fair": "#ffaa00", "poor": "#ff4400",
            }
            color = health_colors.get(health.lower(), "#B4B4B4")
            text += f"Motor Health: {health.upper()}\n"
            text += f"Confidence: {result.confidence:.0%}\n\n"

        # Summary
        if result.summary:
            text += f"{result.summary}\n\n"

        # FFT Peak Interpretation
        if result.fft_interpretation:
            text += "=== FFT Peak Interpretation ===\n"
            for peak in result.fft_interpretation:
                freq = peak.get("freq_hz", "?")
                desc = peak.get("description", "")
                text += f"  {freq} Hz: {desc}\n"
            text += "\n"

        # Issues
        if result.issues:
            text += "=== Issues ===\n"
            for issue in result.issues:
                text += f"  - {issue}\n"
            text += "\n"

        # Recommendations
        if result.recommendations:
            text += "=== Recommendations ===\n"
            for rec in result.recommendations:
                text += f"  - {rec}\n"

        # Show raw API request/response if checkbox is checked
        if self.show_raw_api_cb.isChecked():
            text += "\n" + "=" * 60 + "\n"
            text += "=== RAW REQUEST ===\n"
            text += result.raw_request + "\n\n"
            text += "=== RAW RESPONSE ===\n"
            text += result.raw_response + "\n"

        self.ai_text.setPlainText(text)

    def _on_ai_error(self, err_msg: str):
        """Handle AI interpretation error."""
        self.ai_interpret_btn.setEnabled(True)
        self.ai_interpret_btn.setText("AI Interpret")
        self.ai_text.setPlainText(f"AI Error:\n{err_msg}")

    def _clear_analysis(self):
        """Clear analysis labels."""
        self.amp_a_label.setText("-")
        self.amp_b_label.setText("-")
        self.amp_c_label.setText("-")
        self.thd_a_label.setText("-")
        self.thd_b_label.setText("-")
        self.thd_c_label.setText("-")
        self.balance_label.setText("-")
        self.balance_label.setStyleSheet("")
        self.freq_label.setText("-")
        self.quality_label.setText("-")
        self.quality_label.setStyleSheet("font-weight: bold;")

    def clear_plot(self):
        self._samples = WaveformSamples()
        self._cached_a_raw = None
        self._cached_a_filt = None
        self._last_analysis_data = None
        self.ai_interpret_btn.setEnabled(False)
        self.curve_a.setData([], [])
        self.curve_b.setData([], [])
        self.curve_c.setData([], [])
        self.curve_a_filt.setData([], [])
        self.curve_b_filt.setData([], [])
        self.curve_c_filt.setData([], [])
        self.curve_fft_raw.setData([], [])
        self.curve_fft_filt.setData([], [])
        self._fund_marker.setVisible(False)
        self._fund_label.setVisible(False)
        self._clear_analysis()
        self.info_label.setText("")
        self.ai_text.clear()
