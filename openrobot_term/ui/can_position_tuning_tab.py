"""
CAN AI Analysis tab — position step response and speed eRPM testing
with AI-driven analysis for RMD motors over CAN bus.
"""

import os
import time
import json
import threading
from collections import deque
from typing import Optional, Union

import numpy as np
import pyqtgraph as pg
from .plot_style import style_plot, graph_pen, Crosshair, style_legend
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
    QTextEdit, QProgressBar, QGroupBox, QComboBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QSplitter,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from ..protocol.commands import build_get_mcconf, build_get_mcconf_default, McconfPid, build_set_mcconf_with_pid
from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import (
    RmdCommand, RmdStatus, RmdPid, parse_pid,
    build_read_pid, build_write_pid_to_rom,
    build_position_closed_loop_1, build_set_multiturn_position,
    build_speed_closed_loop,
    build_motor_off, build_motor_stop, build_read_motor_status_2,
)
from ..analysis.can_position_metrics import CanPositionMetrics, analyze_position_step
from ..analysis.signal_metrics import MotorMetrics, analyze_speed_control
from ..analysis.can_auto_tuner import CanAutoTuner, RmdPidGains, CanTuningIteration
from ..analysis.can_speed_auto_tuner import CanSpeedAutoTuner, CanSpeedTuningIteration
from ..analysis.llm_advisor import LLMAdvisor, CanPositionAdvisor, PIDGains, AnalysisResult
from ..workers.can_poller import CanPoller


class CanPositionTuningTab(QWidget):
    _llm_result_signal = pyqtSignal(object)
    _llm_error_signal = pyqtSignal(str)

    def __init__(self, can_transport: PcanTransport):
        super().__init__()
        self._transport = can_transport
        self._auto_tuner: Optional[CanAutoTuner] = None
        self._speed_auto_tuner: Optional[CanSpeedAutoTuner] = None
        self._original_mcconf: Optional[bytes] = None
        self._collecting = False
        self._poller: Optional[CanPoller] = None
        self._progress_timer: Optional[QTimer] = None
        self._baseline_timer: Optional[QTimer] = None

        # Encoder unwrap
        self._prev_enc_pos: Optional[float] = None
        self._enc_offset = 0.0

        # Data buffers (position mode)
        self._time_buf: list[float] = []
        self._pos_buf: list[float] = []
        self._torque_buf: list[float] = []
        self._t0: Optional[float] = None
        self._initial_pos = 0.0

        # Speed mode state
        self._speed_collecting = False
        self._speed_buf: list[float] = []
        self._speed_torque_buf: list[float] = []
        self._speed_time_buf: list[float] = []
        self._speed_t0: Optional[float] = None
        self._speed_target: int = 0
        self._speed_mode: int = 1  # 0=DPS, 1=eRPM (default eRPM)
        self._speed_test_duration: float = 3.0
        self._speed_cmd_timer: Optional[QTimer] = None
        self._last_speed_metrics: Optional[MotorMetrics] = None

        # Safety: always track last encoder position
        self._last_enc_pos_raw: Optional[float] = None

        # Analysis state
        self._last_metrics: Optional[Union[CanPositionMetrics, MotorMetrics]] = None
        self._suggested_pid: Optional[PIDGains] = None
        self._pid_history: list[dict] = []
        self._history_counter = 0
        self._motor_confirmed = False

        self._build_ui()

        # Thread-safe LLM result signals
        self._llm_result_signal.connect(self._on_llm_result)
        self._llm_error_signal.connect(self._on_llm_error)

        # Wire CAN status signal
        self._transport.status_received.connect(self._on_status)

    # ── UI Construction ─────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # === Row 1: Control Bar ===
        ctrl_group = QGroupBox("AI Test")
        ctrl_layout = QHBoxLayout(ctrl_group)

        # Command mode
        ctrl_layout.addWidget(QLabel("Cmd:"))
        self.cmd_combo = QComboBox()
        self.cmd_combo.addItems([
            "Position (0xA3)", "Multiturn+DPS (0xA4)",
            "Speed eRPM (0xA2)", "Speed DPS (0xA2)",
        ])
        self.cmd_combo.currentIndexChanged.connect(self._on_cmd_mode_changed)
        ctrl_layout.addWidget(self.cmd_combo)

        # Position controls (Start, Target, DPS)
        self._start_label = QLabel("Start:")
        ctrl_layout.addWidget(self._start_label)
        self.start_pos_spin = QDoubleSpinBox()
        self.start_pos_spin.setRange(-36000, 36000)
        self.start_pos_spin.setValue(0.0)
        self.start_pos_spin.setSuffix("°")
        self.start_pos_spin.setDecimals(1)
        ctrl_layout.addWidget(self.start_pos_spin)

        self._target_label = QLabel("Target:")
        ctrl_layout.addWidget(self._target_label)
        self.target_pos_spin = QDoubleSpinBox()
        self.target_pos_spin.setRange(-36000, 36000)
        self.target_pos_spin.setValue(90.0)
        self.target_pos_spin.setSuffix("°")
        self.target_pos_spin.setDecimals(1)
        ctrl_layout.addWidget(self.target_pos_spin)

        # DPS limit (used for 0xA4 mode and safety-limited moves)
        self._dps_label = QLabel("DPS:")
        ctrl_layout.addWidget(self._dps_label)
        self.dps_spin = QSpinBox()
        self.dps_spin.setRange(1, 25000)
        self.dps_spin.setValue(500)
        self.dps_spin.setToolTip("Speed limit (deg/s) for 0xA4 mode and safety-limited moves")
        ctrl_layout.addWidget(self.dps_spin)

        # Speed controls (initially hidden)
        self._speed_label = QLabel("eRPM:")
        ctrl_layout.addWidget(self._speed_label)
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(-100000, 100000)
        self.speed_spin.setValue(5000)
        self.speed_spin.setToolTip("Target speed in electrical RPM")
        ctrl_layout.addWidget(self.speed_spin)
        self._speed_unit_label = QLabel("eRPM")
        ctrl_layout.addWidget(self._speed_unit_label)
        self._pole_label = QLabel("Poles:")
        ctrl_layout.addWidget(self._pole_label)
        self.pole_spin = QSpinBox()
        self.pole_spin.setRange(1, 50)
        self.pole_spin.setValue(21)
        self.pole_spin.setToolTip("Motor pole pairs (eRPM ↔ DPS conversion)")
        ctrl_layout.addWidget(self.pole_spin)
        # Hide speed controls by default
        self._speed_label.hide()
        self.speed_spin.hide()
        self._speed_unit_label.hide()
        self._pole_label.hide()
        self.pole_spin.hide()

        ctrl_layout.addWidget(QLabel("Duration:"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 30.0)
        self.duration_spin.setValue(3.0)
        self.duration_spin.setSuffix(" s")
        ctrl_layout.addWidget(self.duration_spin)

        ctrl_layout.addWidget(QLabel("Rate:"))
        self.rate_combo = QComboBox()
        for r in ["50 Hz", "100 Hz", "200 Hz", "500 Hz"]:
            self.rate_combo.addItem(r)
        self.rate_combo.setCurrentText("100 Hz")
        ctrl_layout.addWidget(self.rate_combo)

        self.collect_btn = QPushButton("Collect && Analyze")
        self.collect_btn.clicked.connect(self.start_collection)
        ctrl_layout.addWidget(self.collect_btn)

        self.collect_progress = QProgressBar()
        self.collect_progress.setRange(0, 100)
        self.collect_progress.setValue(0)
        self.collect_progress.setMaximumWidth(120)
        ctrl_layout.addWidget(self.collect_progress)

        ctrl_layout.addStretch()

        self.autotune_btn = QPushButton("Auto-Tune")
        self.autotune_btn.setStyleSheet("background-color: #2d5a27; font-weight: bold;")
        self.autotune_btn.clicked.connect(self.start_auto_tune)
        ctrl_layout.addWidget(self.autotune_btn)

        self.stop_tune_btn = QPushButton("Stop")
        self.stop_tune_btn.setStyleSheet("background-color: #8b0000;")
        self.stop_tune_btn.clicked.connect(self.stop_auto_tune)
        self.stop_tune_btn.setEnabled(False)
        ctrl_layout.addWidget(self.stop_tune_btn)

        layout.addWidget(ctrl_group)

        # === Row 2: Quality Score ===
        score_row = QHBoxLayout()
        self.score_label = QLabel("Quality Score: --")
        self.score_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px;")
        score_row.addWidget(self.score_label)

        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setTextVisible(True)
        self.score_bar.setFormat("%v / 100")
        self.score_bar.setMinimumHeight(30)
        score_row.addWidget(self.score_bar)
        layout.addLayout(score_row)

        # === Main content: Plots | Metrics+PID+History | AI ===
        main_content = QHBoxLayout()
        layout.addLayout(main_content, stretch=1)

        # LEFT PANEL — Plots (full vertical space)
        left_panel = QVBoxLayout()
        main_content.addLayout(left_panel, stretch=5)

        # Plots: Step Response + Error
        plot_splitter = QSplitter(Qt.Orientation.Vertical)

        self.step_plot = pg.PlotWidget()
        style_plot(self.step_plot, title="Position Step Response",
                   left_label="Position", left_unit="deg",
                   bottom_label="Time", bottom_unit="s")
        self.step_curve = self.step_plot.plot(pen=graph_pen(0), name="Actual")
        self.target_line = self.step_plot.plot(
            pen=pg.mkPen(color='#ff6666', width=2, style=Qt.PenStyle.DashLine), name="Target"
        )
        self.settle_upper = self.step_plot.plot(
            pen=pg.mkPen(color='#66ff66', width=1, style=Qt.PenStyle.DotLine), name="±2% band"
        )
        self.settle_lower = self.step_plot.plot(
            pen=pg.mkPen(color='#66ff66', width=1, style=Qt.PenStyle.DotLine)
        )
        style_legend(self.step_plot)
        Crosshair(self.step_plot)
        plot_splitter.addWidget(self.step_plot)

        self.fft_plot = pg.PlotWidget()
        style_plot(self.fft_plot, title="FFT (Steady-State)",
                   left_label="Magnitude", left_unit="",
                   bottom_label="Frequency", bottom_unit="Hz")
        self.fft_curve = self.fft_plot.plot(pen=graph_pen(1), name="FFT")
        self.fft_dominant_line = self.fft_plot.plot(
            pen=pg.mkPen(color='#ff6666', width=2, style=Qt.PenStyle.DashLine), name="Dominant"
        )
        style_legend(self.fft_plot)
        Crosshair(self.fft_plot)
        plot_splitter.addWidget(self.fft_plot)

        plot_splitter.setSizes([300, 150])
        left_panel.addWidget(plot_splitter, stretch=1)

        # RIGHT WRAPPER — Metrics+PID | AI (top) + Score History (bottom)
        right_wrapper = QVBoxLayout()
        main_content.addLayout(right_wrapper, stretch=6)

        # Top row inside right wrapper: Metrics+PID | AI
        right_top = QHBoxLayout()
        right_wrapper.addLayout(right_top, stretch=1)

        # MIDDLE PANEL — Metrics + PID
        mid_panel = QVBoxLayout()
        right_top.addLayout(mid_panel, stretch=3)

        # Metrics grid
        metrics_group = QGroupBox("Metrics")
        metrics_grid = QGridLayout(metrics_group)
        metrics_grid.setSpacing(4)
        self.metrics_labels: dict[str, QLabel] = {}
        self._metrics_name_labels: dict[str, QLabel] = {}
        items = [
            "Pos Ripple", "SS Error",
            "Settling Time", "Overshoot",
            "Rise Time", "Peak Torque",
            "Dominant Freq", "Score",
        ]
        for i, name in enumerate(items):
            row = i // 2
            col = (i % 2) * 2
            lbl = QLabel(f"{name}:")
            lbl.setStyleSheet("font-size: 11px;")
            val = QLabel("--")
            val.setStyleSheet("font-family: monospace; font-size: 11px;")
            metrics_grid.addWidget(lbl, row, col)
            metrics_grid.addWidget(val, row, col + 1)
            self.metrics_labels[name] = val
            self._metrics_name_labels[name] = lbl
        mid_panel.addWidget(metrics_group)

        # PID Tuning (compact grid: label | current | → | suggested)
        self.pid_group_widget = QGroupBox("Position PID Tuning")
        pid_grid = QGridLayout(self.pid_group_widget)
        self.pid_kp = QDoubleSpinBox()
        self.pid_ki = QDoubleSpinBox()
        self.pid_kd = QDoubleSpinBox()
        self.pid_kd_flt = QDoubleSpinBox()
        self.sug_kp = QLabel("--")
        self.sug_ki = QLabel("--")
        self.sug_kd = QLabel("--")
        self.sug_kd_flt = QLabel("--")
        for i, (lbl_text, spin, sug) in enumerate([
            ("Kp:", self.pid_kp, self.sug_kp),
            ("Ki:", self.pid_ki, self.sug_ki),
            ("Kd:", self.pid_kd, self.sug_kd),
            ("Kd Flt:", self.pid_kd_flt, self.sug_kd_flt),
        ]):
            spin.setDecimals(5)
            spin.setRange(0, 100)
            spin.setSingleStep(0.001)
            sug.setStyleSheet("font-family: monospace; font-weight: bold;")
            pid_grid.addWidget(QLabel(lbl_text), i, 0)
            pid_grid.addWidget(spin, i, 1)
            arrow = QLabel("\u2192")
            arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pid_grid.addWidget(arrow, i, 2)
            pid_grid.addWidget(sug, i, 3)
        pid_btn_row = QHBoxLayout()
        self.read_pid_btn = QPushButton("Read")
        self.read_pid_btn.clicked.connect(self._read_pid)
        pid_btn_row.addWidget(self.read_pid_btn)
        self.write_rom_btn = QPushButton("Write MCCONF")
        self.write_rom_btn.setStyleSheet("background-color: #884400;")
        self.write_rom_btn.clicked.connect(self._write_pid_rom)
        pid_btn_row.addWidget(self.write_rom_btn)
        self.apply_btn = QPushButton("Apply >>>")
        self.apply_btn.clicked.connect(self._apply_suggested)
        self.apply_btn.setEnabled(False)
        pid_btn_row.addWidget(self.apply_btn)
        pid_grid.addLayout(pid_btn_row, 4, 0, 1, 4)
        mid_panel.addWidget(self.pid_group_widget)

        # Current Speed PID (read from serial MCCONF)
        self.cur_pid_group = QGroupBox("Current Speed PID")
        cur_grid = QGridLayout(self.cur_pid_group)
        self.cur_kp = QDoubleSpinBox()
        self.cur_ki = QDoubleSpinBox()
        self.cur_kd = QDoubleSpinBox()
        self.cur_kd_flt = QDoubleSpinBox()
        self.cur_ramp = QDoubleSpinBox()
        for i, (lbl, spin) in enumerate([
            ("Kp:", self.cur_kp), ("Ki:", self.cur_ki),
            ("Kd:", self.cur_kd), ("Kd Flt:", self.cur_kd_flt),
        ]):
            spin.setDecimals(6)
            spin.setRange(0, 100)
            spin.setSingleStep(0.001)
            cur_grid.addWidget(QLabel(lbl), i, 0)
            cur_grid.addWidget(spin, i, 1)
        self.cur_ramp.setDecimals(0)
        self.cur_ramp.setRange(-1, 1000000)
        self.cur_ramp.setSingleStep(1000)
        cur_grid.addWidget(QLabel("Ramp:"), 4, 0)
        cur_grid.addWidget(self.cur_ramp, 4, 1)
        cur_btn_row = QHBoxLayout()
        self.read_gains_btn = QPushButton("Read Gains")
        self.read_gains_btn.clicked.connect(self._read_mcconf_gains)
        cur_btn_row.addWidget(self.read_gains_btn)
        self.read_default_btn = QPushButton("Read Default")
        self.read_default_btn.clicked.connect(self._read_mcconf_default)
        cur_btn_row.addWidget(self.read_default_btn)
        self.write_speed_pid_btn = QPushButton("Write")
        self.write_speed_pid_btn.setStyleSheet("background-color: #884400;")
        self.write_speed_pid_btn.clicked.connect(self._write_speed_pid)
        cur_btn_row.addWidget(self.write_speed_pid_btn)
        cur_grid.addLayout(cur_btn_row, 5, 0, 1, 2)
        mid_panel.addWidget(self.cur_pid_group)
        self.cur_pid_group.hide()  # Hidden by default (Position mode)
        mid_panel.addStretch()

        # RIGHT PANEL — AI Analysis
        right_panel = QVBoxLayout()
        right_top.addLayout(right_panel, stretch=3)

        ai_group = QGroupBox("AI Analysis")
        ai_layout = QVBoxLayout(ai_group)

        llm_row = QHBoxLayout()
        llm_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"])
        llm_row.addWidget(self.model_combo)

        llm_row.addWidget(QLabel("Lang:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["한국어", "English"])
        llm_row.addWidget(self.lang_combo)

        self.ai_btn = QPushButton("Ask AI")
        self.ai_btn.clicked.connect(self._run_llm)
        self.ai_btn.setEnabled(False)
        llm_row.addWidget(self.ai_btn)

        self.api_label = QLabel()
        self._update_api_status()
        llm_row.addWidget(self.api_label)
        llm_row.addStretch()

        self.show_raw_cb = QCheckBox("Raw")
        self.show_raw_cb.setToolTip("Show raw API request/response")
        llm_row.addWidget(self.show_raw_cb)
        ai_layout.addLayout(llm_row)

        self.ai_text = QTextEdit()
        self.ai_text.setReadOnly(True)
        self.ai_text.setPlaceholderText("AI analysis results will appear here...")
        ai_layout.addWidget(self.ai_text)
        right_panel.addWidget(ai_group, stretch=1)

        # Score History (below Metrics+AI, same width)
        history_group = QGroupBox("Score History")
        history_layout = QHBoxLayout(history_group)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels([
            "#", "Kp", "Ki", "Kd", "Score", "Ripple%", "SS Err%"
        ])
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setMaximumHeight(120)
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.history_table.setColumnWidth(0, 30)
        for col in range(1, 7):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.history_table.cellDoubleClicked.connect(self._on_history_double_click)
        history_layout.addWidget(self.history_table)
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(60)
        clear_btn.clicked.connect(lambda: (
            self.history_table.setRowCount(0),
            self._pid_history.clear(),
            setattr(self, '_history_counter', 0),
        ))
        history_layout.addWidget(clear_btn)
        right_wrapper.addWidget(history_group)

    # ── Command Mode Switching ─────────────────────────────────────

    def _on_cmd_mode_changed(self, index: int):
        """Switch UI between position and speed modes."""
        is_speed_erpm = (index == 2)
        is_speed_dps = (index == 3)
        is_speed = is_speed_erpm or is_speed_dps

        # Save old speed mode before updating (for value conversion)
        prev_speed_mode = self._speed_mode  # 0=DPS, 1=eRPM

        # Track speed sub-mode
        if is_speed_erpm:
            self._speed_mode = 1
        elif is_speed_dps:
            self._speed_mode = 0

        # Position controls
        for w in [self.start_pos_spin, self.target_pos_spin, self.dps_spin,
                  self._start_label, self._target_label, self._dps_label]:
            w.setVisible(not is_speed)

        # Speed controls
        for w in [self.speed_spin, self._speed_label, self._speed_unit_label,
                  self._pole_label, self.pole_spin]:
            w.setVisible(is_speed)

        # Speed spin label/range per sub-mode — convert current value
        pp = self.pole_spin.value()
        old_val = self.speed_spin.value()
        if is_speed_dps:
            self._speed_label.setText("DPS:")
            self._speed_unit_label.setText("dps")
            if prev_speed_mode == 1:  # was eRPM → convert to DPS
                converted = int(round(old_val * 6.0 / pp))
            else:
                converted = old_val
            self.speed_spin.setRange(-25000, 25000)
            self.speed_spin.setValue(max(-25000, min(25000, converted)))
        elif is_speed_erpm:
            self._speed_label.setText("eRPM:")
            self._speed_unit_label.setText("eRPM")
            if prev_speed_mode == 0:  # was DPS → convert to eRPM
                converted = int(round(old_val * pp / 6.0))
            else:
                converted = old_val
            self.speed_spin.setRange(-100000, 100000)
            self.speed_spin.setValue(max(-100000, min(100000, converted)))

        # Plot titles
        if is_speed:
            self.step_plot.setTitle("Speed Step Response")
            self.step_plot.setLabel('left', 'Speed', units='dps')
        else:
            self.step_plot.setTitle("Position Step Response")
            self.step_plot.setLabel('left', 'Position', units='deg')

        # PID panel visibility
        self.pid_group_widget.setVisible(not is_speed)
        self.cur_pid_group.setVisible(is_speed_erpm)

        # Update metrics labels
        self._update_metrics_labels(is_speed)

    def _update_metrics_labels(self, is_speed: bool):
        """Swap metrics label text between position and speed modes."""
        if is_speed:
            self._metrics_name_labels["Pos Ripple"].setText("Speed Ripple:")
            self._metrics_name_labels["SS Error"].setText("SS Error (dps):")
            self._metrics_name_labels["Peak Torque"].setText("Torque RMS:")
        else:
            self._metrics_name_labels["Pos Ripple"].setText("Pos Ripple:")
            self._metrics_name_labels["SS Error"].setText("SS Error:")
            self._metrics_name_labels["Peak Torque"].setText("Peak Torque:")

    # ── CAN Data Ingestion ──────────────────────────────────────────

    def _on_status(self, motor_id: int, status: RmdStatus):
        """Receive CAN motor status during data collection."""
        # Always track last encoder position (for safety checks)
        self._last_enc_pos_raw = status.enc_pos

        # Speed mode collection
        if self._speed_collecting:
            self._speed_buf.append(float(status.speed_dps))
            self._speed_torque_buf.append(float(status.torque_curr))
            if self._speed_t0 is None:
                self._speed_t0 = time.time()
            self._speed_time_buf.append(time.time() - self._speed_t0)
            return

        if not self._collecting:
            return

        # Unwrap encoder
        raw = status.enc_pos
        if self._prev_enc_pos is not None:
            delta = raw - self._prev_enc_pos
            if delta < -180.0:
                self._enc_offset += 360.0
            elif delta > 180.0:
                self._enc_offset -= 360.0
        self._prev_enc_pos = raw
        unwrapped = raw + self._enc_offset

        self._pos_buf.append(unwrapped)
        self._torque_buf.append(status.torque_curr)
        if self._t0 is None:
            self._t0 = time.time()
        self._time_buf.append(time.time() - self._t0)

    def on_frame_received(self, can_id: int, dlc: int, timestamp: float, data: list):
        """Handle raw CAN frames for PID read responses."""
        if len(data) >= 8 and data[0] == RmdCommand.READ_PID:
            try:
                pid = parse_pid(data)
                self.pid_kp.setValue(pid.kp)
                self.pid_ki.setValue(pid.ki)
                self.pid_kd.setValue(pid.kd)
            except Exception:
                pass

    # ── Manual Collection & Analysis ────────────────────────────────

    def start_collection(self):
        """Start data collection — dispatches to position or speed mode."""
        if self.cmd_combo.currentIndex() in (2, 3):
            self._start_speed_collection()
            return

        # Position mode (existing flow)
        target = self.target_pos_spin.value()
        start = self.start_pos_spin.value()
        duration = self.duration_spin.value()

        step_mag = abs(target - start)
        mode_text = self.cmd_combo.currentText()

        # Motor movement confirmation
        reply = QMessageBox.warning(
            self, "Motor Warning",
            f"Position Step Test:\n\n"
            f"  Start: {start:.1f}\u00b0 (via 0xA4 DPS 1000)\n"
            f"  Target: {target:.1f}\u00b0 (via {mode_text})\n"
            f"  Step: {step_mag:.1f}\u00b0\n"
            f"  Duration: {duration:.1f}s\n\n"
            f"Motor will move. Ensure it is safe!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Read current PID from MCU (response arrives during 3s move)
        self._transport.send_frame(build_read_pid())

        # Move to start position — ALWAYS use DPS-limited command (safe)
        self.collect_btn.setEnabled(False)
        self.collect_btn.setText("Reading PID + moving to start...")
        self._safe_move(start)
        QTimer.singleShot(3000, lambda: self._begin_step_test(start, target, duration))

    # ── Speed eRPM Collection ──────────────────────────────────────

    def _start_speed_collection(self):
        """Start speed step response collection (eRPM or DPS)."""
        target_val = self.speed_spin.value()
        duration = self.duration_spin.value()
        mode_str = "DPS (mode=0)" if self._speed_mode == 0 else "eRPM (mode=1)"
        unit_str = "dps" if self._speed_mode == 0 else "eRPM"

        reply = QMessageBox.warning(
            self, "Motor Warning",
            f"Speed Test:\n\n"
            f"  Target: {target_val} {unit_str} (0xA2 {mode_str})\n"
            f"  Duration: {duration:.1f}s\n\n"
            f"Motor will spin. Ensure it is safe!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.collect_btn.setEnabled(False)
        self.collect_btn.setText("Stopping motor...")

        # Release motor first (MOTOR_OFF 0x80 → MOTOR_RELEASE).
        # Must NOT use MOTOR_STOP (0x81) here because it activates
        # DPS_CONTROL_DURATION position lock which fights against
        # the VESC built-in speed PID used by eRPM mode=1.
        self._transport.send_frame(build_motor_off())
        self._speed_target = target_val
        self._speed_test_duration = duration
        QTimer.singleShot(1000, self._begin_speed_test)

    def _begin_speed_test(self):
        """Start poller, wait for baseline, then send speed command."""
        # Clear speed buffers
        self._speed_buf.clear()
        self._speed_torque_buf.clear()
        self._speed_time_buf.clear()
        self._speed_t0 = None

        self.collect_btn.setText("Waiting for baseline...")
        self.collect_progress.setValue(0)

        rate_hz = self._get_rate_hz()
        rate_ms = int(1000.0 / rate_hz)
        self._poll_duration = self._speed_test_duration
        self._speed_collecting = True

        self._poller = CanPoller(self._transport, rate_ms)
        self._poller.start()

        # Wait for baseline samples before sending speed command
        self._baseline_check_count = 0
        self._baseline_timer = QTimer(self)
        self._baseline_timer.setInterval(10)
        self._baseline_timer.timeout.connect(self._check_speed_baseline_ready)
        self._baseline_timer.start()

    def _check_speed_baseline_ready(self):
        """Fire speed command once poller has delivered baseline samples."""
        self._baseline_check_count += 1
        if len(self._speed_buf) >= 3 or self._baseline_check_count >= 50:
            self._baseline_timer.stop()
            self._baseline_timer = None
            self._fire_speed_step()

    def _fire_speed_step(self):
        """Send speed command (eRPM or DPS) after baseline."""
        self._poll_start = time.time()

        # Reset buffers after baseline (keep clean data from step start)
        self._speed_buf.clear()
        self._speed_torque_buf.clear()
        self._speed_time_buf.clear()
        self._speed_t0 = None

        # Send speed command and keep re-sending periodically.
        # VESC speed mode uses mc_interface_set_pid_speed() which has a timeout —
        # the command must be refreshed or the motor stops.
        self._transport.send_frame(
            build_speed_closed_loop(self._speed_target, mode=self._speed_mode)
        )
        self._speed_cmd_timer = QTimer(self)
        self._speed_cmd_timer.setInterval(50)  # 50 ms refresh
        self._speed_cmd_timer.timeout.connect(self._resend_speed_cmd)
        self._speed_cmd_timer.start()

        self.collect_btn.setText("Collecting speed data...")

        # Progress update timer
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(100)
        self._progress_timer.timeout.connect(self._update_speed_collect_progress)
        self._progress_timer.start()

        # Stop after duration
        QTimer.singleShot(
            int(self._speed_test_duration * 1000),
            self._finish_speed_collection,
        )

    def _resend_speed_cmd(self):
        """Periodically re-send speed command to prevent VESC timeout."""
        self._transport.send_frame(
            build_speed_closed_loop(self._speed_target, mode=self._speed_mode)
        )

    def _update_speed_collect_progress(self):
        elapsed = time.time() - self._poll_start
        progress = min(100, int(elapsed / self._poll_duration * 100))
        self.collect_progress.setValue(progress)
        self.collect_btn.setText(f"Collecting... ({len(self._speed_buf)} samples)")

    def _finish_speed_collection(self):
        # Stop speed command refresh first
        if self._speed_cmd_timer:
            self._speed_cmd_timer.stop()
            self._speed_cmd_timer = None
        if self._baseline_timer:
            self._baseline_timer.stop()
            self._baseline_timer = None
        # Stop poller
        if self._poller:
            self._poller.stop()
            deadline = time.time() + 2.0
            while self._poller.isRunning() and time.time() < deadline:
                QApplication.processEvents()
                time.sleep(0.005)
            if self._poller.isRunning():
                self._poller.terminate()
                self._poller.wait(500)
            self._poller = None
        if self._progress_timer:
            self._progress_timer.stop()
            self._progress_timer = None

        # Release motor (no position lock needed for speed test)
        self._transport.send_frame(build_motor_off())
        self._speed_collecting = False
        self.collect_btn.setEnabled(True)
        self.collect_btn.setText("Collect && Analyze")
        self.collect_progress.setValue(100)
        self._run_speed_analysis()

    def _run_speed_analysis(self):
        """Analyze speed step response data using signal_metrics."""
        n = len(self._speed_buf)
        if n < 10:
            self.score_label.setText(
                f"Quality Score: -- (insufficient data: {n} samples received)"
            )
            self.ai_text.append(
                f"\n[DEBUG] Speed collection: {n} samples.\n"
                f"If 0: check CAN polling. If few: motor may not respond to 0x9C."
            )
            return

        actual_rate = n / self._poll_duration if self._poll_duration > 0 else 100.0
        speed_data = np.array(self._speed_buf)  # dps from CAN status

        # Target in dps: convert eRPM to dps for analysis.
        # For analysis purposes we pass dps data directly as "rpm_data" —
        # the metrics (ripple%, settling time, overshoot) are unit-agnostic ratios.
        target_dps = float(self._speed_target)  # placeholder: unit label shows dps
        # Note: actual relationship is eRPM → dps depends on pole pairs.
        # We use the raw dps feedback and auto-detect the steady-state target.
        # Use last 20% average as effective target for more accurate analysis.
        tail_n = max(5, n // 5)
        effective_target = float(np.mean(speed_data[-tail_n:]))

        torque_data = np.array(self._speed_torque_buf) if self._speed_torque_buf else None
        metrics = analyze_speed_control(
            rpm_data=speed_data,
            target_rpm=effective_target,
            sample_rate=actual_rate,
            current_data=torque_data,
        )
        # Store the commanded eRPM for display
        metrics.target_rpm = effective_target

        self._last_metrics = metrics
        self._last_speed_metrics = metrics
        self._update_speed_display(metrics)
        self.ai_btn.setEnabled(True)

    def _update_speed_display(self, m: MotorMetrics):
        """Update UI with speed analysis results."""
        score = m.quality_score
        self.score_label.setText(f"Quality Score: {score:.1f}")
        self.score_bar.setValue(int(score))
        self.score_bar.setFormat(f"{int(score)} / 100")

        if score >= 80:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #44bb44; }")
        elif score >= 50:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #ddaa00; }")
        else:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #dd4444; }")

        # Metrics labels (speed mode)
        self.metrics_labels["Pos Ripple"].setText(f"{m.rpm_ripple_pct:.2f}%")
        self.metrics_labels["SS Error"].setText(
            f"{m.steady_state_error:.1f} dps ({m.steady_state_error_pct:.2f}%)"
        )
        self.metrics_labels["Settling Time"].setText(f"{m.settling_time:.3f} s")
        self.metrics_labels["Overshoot"].setText(f"{m.overshoot_pct:.2f} %")
        self.metrics_labels["Rise Time"].setText(f"{m.rise_time:.3f} s")
        self.metrics_labels["Peak Torque"].setText(
            f"{m.current_ripple_pct:.2f}%" if m.current_ripple_pct > 0 else "--"
        )
        self.metrics_labels["Dominant Freq"].setText(f"{m.fft_dominant_freq:.1f} Hz")
        self.metrics_labels["Score"].setText(f"{score:.1f} / 100")

        # Step response plot (speed)
        if len(m.time_data) > 0 and len(m.rpm_data) > 0:
            self.step_curve.setData(m.time_data, m.rpm_data)

            target = m.target_rpm
            t_max = m.time_data[-1]
            t_min = m.time_data[0]
            self.target_line.setData([t_min, t_max], [target, target])

            band = abs(target) * 0.02 if abs(target) > 0.01 else 10.0
            self.settle_upper.setData([t_min, t_max], [target + band, target + band])
            self.settle_lower.setData([t_min, t_max], [target - band, target - band])

        # FFT plot
        self._update_fft_plot(m.fft_frequencies, m.fft_magnitudes,
                              m.fft_dominant_freq)

    # ── Position Collection (existing flow) ────────────────────────

    def _begin_step_test(self, start: float, target: float, duration: float):
        """After motor has reached start position, run the step test."""
        # Stop briefly
        self._transport.send_frame(build_motor_stop())
        QTimer.singleShot(500, lambda: self._run_step_collection(start, target, duration))

    def _run_step_collection(self, start: float, target: float, duration: float):
        # Clear buffers
        self._time_buf.clear()
        self._pos_buf.clear()
        self._torque_buf.clear()
        self._t0 = None
        self._prev_enc_pos = None
        self._enc_offset = 0.0

        # Brief pre-read for initial position
        self._collecting = True
        self._transport.send_frame(build_read_motor_status_2())
        QTimer.singleShot(100, lambda: self._start_step_polling(start, target, duration))

    def _start_step_polling(self, start: float, target: float, duration: float):
        self._initial_pos = self._pos_buf[-1] if self._pos_buf else start
        self._time_buf.clear()
        self._pos_buf.clear()
        self._torque_buf.clear()
        self._t0 = None

        self.collect_btn.setText("Waiting for baseline...")
        self.collect_progress.setValue(0)

        # Start CAN poller FIRST — wait for real baseline data before step
        rate_hz = self._get_rate_hz()
        rate_ms = int(1000.0 / rate_hz)
        self._poll_duration = duration
        self._collecting = True

        self._poller = CanPoller(self._transport, rate_ms)
        self._poller.start()

        # Poll until poller delivers baseline samples, then fire step
        self._step_target = target
        self._step_duration = duration
        self._baseline_check_count = 0
        self._baseline_timer = QTimer(self)
        self._baseline_timer.setInterval(10)
        self._baseline_timer.timeout.connect(self._check_baseline_ready)
        self._baseline_timer.start()

    def _check_baseline_ready(self):
        """Fire step command once poller has delivered baseline samples."""
        self._baseline_check_count += 1
        if len(self._pos_buf) >= 3 or self._baseline_check_count >= 50:
            self._baseline_timer.stop()
            self._baseline_timer = None
            self._fire_step(self._step_target, self._step_duration)

    def _fire_step(self, target: float, duration: float):
        """Send step command after poller is confirmed running."""
        self._poll_start = time.time()
        self._send_position(target)

        # Progress update timer (UI feedback)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(100)
        self._progress_timer.timeout.connect(self._update_collect_progress)
        self._progress_timer.start()

        # Stop collection after duration
        QTimer.singleShot(int(duration * 1000), self._finish_collection)

    def _update_collect_progress(self):
        elapsed = time.time() - self._poll_start
        progress = min(100, int(elapsed / self._poll_duration * 100))
        self.collect_progress.setValue(progress)
        self.collect_btn.setText(f"Collecting... ({len(self._pos_buf)} samples)")

    def _finish_collection(self):
        if self._baseline_timer:
            self._baseline_timer.stop()
            self._baseline_timer = None
        # Stop poller (non-blocking: keep Qt event loop alive so queued
        # signals from the PCAN reader thread can drain normally)
        if self._poller:
            self._poller.stop()
            deadline = time.time() + 2.0
            while self._poller.isRunning() and time.time() < deadline:
                QApplication.processEvents()
                time.sleep(0.005)
            if self._poller.isRunning():
                self._poller.terminate()
                self._poller.wait(500)
            self._poller = None
        if self._progress_timer:
            self._progress_timer.stop()
            self._progress_timer = None
        self._collecting = False
        self.collect_btn.setEnabled(True)
        self.collect_btn.setText("Collect && Analyze")
        self.collect_progress.setValue(100)
        self._run_analysis()

    def _run_analysis(self):
        n = len(self._pos_buf)
        if n < 10:
            self.score_label.setText(
                f"Quality Score: -- (insufficient data: {n} samples received)"
            )
            self.ai_text.append(
                f"\n[DEBUG] Data collection result: {n} samples in pos_buf, "
                f"{len(self._torque_buf)} in torque_buf.\n"
                f"If 0 samples: check that CAN polling works (try CAN Realtime tab first).\n"
                f"If few samples: motor may not respond to READ_MOTOR_STATUS_2 (0x9C)."
            )
            return

        target = self.target_pos_spin.value()
        start = self.start_pos_spin.value()
        actual_rate = len(self._pos_buf) / self._poll_duration if self._poll_duration > 0 else 100.0

        # Normalize encoder frame → commanded frame.
        # Single-turn encoder near 0°/360° boundary can introduce ±360° offset
        # (e.g. motor at 0° reads as 359.5°, unwrap then adds 360° on crossing).
        frame_offset = round((self._initial_pos - start) / 360.0) * 360.0
        pos_data = np.array(self._pos_buf) - frame_offset

        metrics = analyze_position_step(
            pos_data=pos_data,
            target_deg=target,
            sample_rate=actual_rate,
            torque_data=np.array(self._torque_buf) if self._torque_buf else None,
            initial_pos=self._initial_pos - frame_offset,
        )

        self._last_metrics = metrics
        self._update_display(metrics)
        self.ai_btn.setEnabled(True)

    # ── Display Update ──────────────────────────────────────────────

    def _update_display(self, m: CanPositionMetrics):
        score = m.quality_score
        self.score_label.setText(f"Quality Score: {score:.1f}")
        self.score_bar.setValue(int(score))
        self.score_bar.setFormat(f"{int(score)} / 100")

        if score >= 80:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #44bb44; }")
        elif score >= 50:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #ddaa00; }")
        else:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #dd4444; }")

        # Metrics labels
        self.metrics_labels["Pos Ripple"].setText(f"{m.pos_ripple_deg:.3f}\u00b0 ({m.pos_ripple_pct:.2f}%)")
        self.metrics_labels["SS Error"].setText(f"{m.steady_state_error_deg:.3f}\u00b0 ({m.steady_state_error_pct:.2f}%)")
        self.metrics_labels["Settling Time"].setText(f"{m.settling_time_s:.3f} s")
        self.metrics_labels["Overshoot"].setText(f"{m.overshoot_pct:.2f} %")
        self.metrics_labels["Rise Time"].setText(f"{m.rise_time_s:.3f} s")
        self.metrics_labels["Peak Torque"].setText(f"{m.torque_peak_a:.2f} A")
        self.metrics_labels["Dominant Freq"].setText(f"{m.fft_dominant_freq:.1f} Hz")
        self.metrics_labels["Score"].setText(f"{score:.1f} / 100")

        # Step response plot
        if len(m.time_data) > 0 and len(m.pos_data) > 0:
            self.step_curve.setData(m.time_data, m.pos_data)

            target = m.target_pos_deg
            t_max = m.time_data[-1]
            t_min = m.time_data[0]
            self.target_line.setData([t_min, t_max], [target, target])

            step_mag = abs(target - (m.pos_data[0] if len(m.pos_data) > 0 else 0))
            band = step_mag * 0.02 if step_mag > 0.01 else 0.5
            self.settle_upper.setData([t_min, t_max], [target + band, target + band])
            self.settle_lower.setData([t_min, t_max], [target - band, target - band])

        # FFT plot
        self._update_fft_plot(m.fft_frequencies, m.fft_magnitudes,
                              m.fft_dominant_freq)

    # ── PID Read/Write ──────────────────────────────────────────────

    def _read_pid(self):
        if not self._transport.is_connected():
            return
        self._transport.send_vesc_to_target(build_get_mcconf())
        self.ai_text.append("[MCCONF] Reading position PID...")

    def _write_pid_rom(self):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected",
                                "PCAN connection required to write MCCONF.")
            return
        if self._original_mcconf is None:
            QMessageBox.warning(
                self, "No MCCONF",
                "Read MCCONF first (use 'Read Gains') before writing."
            )
            return
        kp = self.pid_kp.value()
        ki = self.pid_ki.value()
        kd = self.pid_kd.value()
        kd_flt = self.pid_kd_flt.value()
        reply = QMessageBox.warning(
            self, "Write Position PID",
            f"Write Position PID to VESC MCCONF?\n\n"
            f"Kp={kp:.5f}  Ki={ki:.5f}\nKd={kd:.5f}  Kd Flt={kd_flt:.5f}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            packet = build_set_mcconf_with_pid(
                self._original_mcconf,
                kp, ki, kd, kd_flt,
                position_mode=True,
            )
            self._transport.send_vesc_to_target(packet)
            self.ai_text.append(
                f"[Pos PID] Written MCCONF: Kp={kp:.5f} Ki={ki:.5f} "
                f"Kd={kd:.5f} Kd_flt={kd_flt:.5f}"
            )

    def _apply_suggested(self):
        """Apply suggested PID to motor via MCCONF."""
        if self._suggested_pid is None:
            return
        if self._original_mcconf is None:
            QMessageBox.warning(
                self, "No MCCONF",
                "Read MCCONF first (use 'Read Gains') before applying."
            )
            return
        kp = self._suggested_pid.kp
        ki = self._suggested_pid.ki
        kd = self._suggested_pid.kd
        kd_flt = self.pid_kd_flt.value()
        packet = build_set_mcconf_with_pid(
            self._original_mcconf,
            kp, ki, kd, kd_flt,
            position_mode=True,
        )
        self._transport.send_vesc_to_target(packet)
        self.pid_kp.setValue(kp)
        self.pid_ki.setValue(ki)
        self.pid_kd.setValue(kd)
        self.ai_text.append(
            f"[Pos PID] Applied MCCONF: Kp={kp:.5f} Ki={ki:.5f} "
            f"Kd={kd:.5f} Kd_flt={kd_flt:.5f}"
        )

        # Add to history
        score = self._last_metrics.quality_score if self._last_metrics else None
        ripple = getattr(self._last_metrics, 'pos_ripple_pct', None) if self._last_metrics else None
        ss_err = getattr(self._last_metrics, 'steady_state_error_pct', None) if self._last_metrics else None
        self._add_history_entry(kp, ki, kd, score, ripple, ss_err, self._last_metrics)

    # ── Auto-Tune ───────────────────────────────────────────────────

    def start_auto_tune(self):
        is_speed = self.cmd_combo.currentIndex() in (2, 3)

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-api-key-here":
            QMessageBox.warning(self, "API Key", "Set OPENAI_API_KEY in .env file.")
            return

        if is_speed:
            # Speed eRPM auto-tune requires VESC connection for MCCONF PID writes
            if not self._transport.is_connected():
                QMessageBox.warning(
                    self, "Not Connected",
                    "PCAN connection required for speed auto-tune.\n"
                    "Speed PID is written via COMM_SET_MCCONF over CAN."
                )
                return
            if self._original_mcconf is None:
                QMessageBox.warning(
                    self, "MCCONF Not Loaded",
                    "Read MCCONF first (click 'Read Gains').\n"
                    "Original MCCONF data is needed to write PID changes."
                )
                return
            target_val = self.speed_spin.value()
            mode_str = "DPS (mode=0)" if self._speed_mode == 0 else "eRPM (mode=1)"
            unit_str = "dps" if self._speed_mode == 0 else "eRPM"
            if not self._motor_confirmed:
                reply = QMessageBox.warning(
                    self, "Auto-Tune Confirmation",
                    f"Speed Auto-Tune:\n\n"
                    f"  Target: {target_val} {unit_str} (0xA2 {mode_str})\n"
                    f"  Iterations: 5\n"
                    f"  Duration: {self.duration_spin.value():.1f}s per iteration\n\n"
                    f"Motor will spin repeatedly.\n"
                    f"Speed PID will be modified via serial MCCONF.\n\n"
                    f"Ensure safety!\n\nProceed?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                self._motor_confirmed = True

            self.autotune_btn.setEnabled(False)
            self.autotune_btn.setText("Starting...")
            QTimer.singleShot(100, self._launch_speed_auto_tuner)
        else:
            # Position auto-tune
            if not self._motor_confirmed:
                target = self.target_pos_spin.value()
                start = self.start_pos_spin.value()
                reply = QMessageBox.warning(
                    self, "Auto-Tune Confirmation",
                    f"Auto-Tune will control the motor automatically.\n\n"
                    f"  Start: {start:.1f}\u00b0\n"
                    f"  Target: {target:.1f}\u00b0\n"
                    f"  Iterations: 5\n\n"
                    f"Motor will move repeatedly. Ensure safety!\n\nProceed?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                self._motor_confirmed = True

            # Read current PID from MCU before starting
            self.autotune_btn.setEnabled(False)
            self.autotune_btn.setText("Reading PID...")
            self._transport.send_frame(build_read_pid())
            QTimer.singleShot(300, self._launch_auto_tuner)

    def _launch_auto_tuner(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model_name = self.model_combo.currentText()
        lang = "Korean" if self.lang_combo.currentText() == "\ud55c\uad6d\uc5b4" else "English"
        advisor = CanPositionAdvisor(api_key=api_key, model=model_name, language=lang)

        initial_pid = RmdPidGains(
            kp=self.pid_kp.value(),
            ki=self.pid_ki.value(),
            kd=self.pid_kd.value(),
        )

        use_multiturn = self.cmd_combo.currentIndex() == 1
        self._auto_tuner = CanAutoTuner(
            transport=self._transport,
            advisor=advisor,
            target_pos_deg=self.target_pos_spin.value(),
            initial_pid=initial_pid,
            use_multiturn_cmd=use_multiturn,
            dps_limit=self.dps_spin.value(),
            max_iterations=5,
            collect_duration_s=self.duration_spin.value(),
            poll_rate_hz=self._get_rate_hz(),
            return_pos_deg=self.start_pos_spin.value(),
        )
        self._auto_tuner.status_update.connect(self._on_tune_status)
        self._auto_tuner.iteration_complete.connect(self._on_tune_iteration)
        self._auto_tuner.tuning_finished.connect(self._on_tune_finished)
        self._auto_tuner.data_collecting.connect(
            lambda p: self.collect_progress.setValue(int(p * 100))
        )
        self._auto_tuner.metrics_ready.connect(self._on_metrics_ready)

        # Wire status for auto-tuner data collection
        self._transport.status_received.connect(self._auto_tuner.on_status)

        self._auto_tuner.start()

        self.autotune_btn.setText("Auto-Tune")
        self.stop_tune_btn.setEnabled(True)
        self.collect_btn.setEnabled(False)
        self.ai_text.setPlainText(
            f"[Auto-Tune] Started CAN position PID tuning...\n"
            f"  Initial PID (from MCU): Kp={initial_pid.kp:.5f} Ki={initial_pid.ki:.5f} Kd={initial_pid.kd:.5f}\n"
            f"Motor will move..."
        )

    def _launch_speed_auto_tuner(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model_name = self.model_combo.currentText()
        lang = "Korean" if self.lang_combo.currentText() == "\ud55c\uad6d\uc5b4" else "English"
        advisor = LLMAdvisor(api_key=api_key, model=model_name, language=lang)

        initial_pid = PIDGains(
            kp=self.cur_kp.value(),
            ki=self.cur_ki.value(),
            kd=self.cur_kd.value(),
            kd_filter=self.cur_kd_flt.value(),
            ramp_erpms_s=self.cur_ramp.value(),
        )

        self._speed_auto_tuner = CanSpeedAutoTuner(
            can_transport=self._transport,
            advisor=advisor,
            target_erpm=self.speed_spin.value(),
            initial_pid=initial_pid,
            original_mcconf=self._original_mcconf,
            max_iterations=5,
            collect_duration_s=self.duration_spin.value(),
            poll_rate_hz=self._get_rate_hz(),
        )
        self._speed_auto_tuner.status_update.connect(self._on_tune_status)
        self._speed_auto_tuner.iteration_complete.connect(self._on_speed_tune_iteration)
        self._speed_auto_tuner.tuning_finished.connect(self._on_speed_tune_finished)
        self._speed_auto_tuner.data_collecting.connect(
            lambda p: self.collect_progress.setValue(int(p * 100))
        )
        self._speed_auto_tuner.metrics_ready.connect(self._on_speed_metrics_ready)

        # Wire CAN status for data collection
        self._transport.status_received.connect(self._speed_auto_tuner.on_status)

        self._speed_auto_tuner.start()

        self.autotune_btn.setText("Auto-Tune")
        self.stop_tune_btn.setEnabled(True)
        self.collect_btn.setEnabled(False)
        self.ai_text.setPlainText(
            f"[Auto-Tune] Started CAN speed eRPM PID tuning...\n"
            f"  Target: {self.speed_spin.value()} eRPM\n"
            f"  Initial Speed PID: Kp={initial_pid.kp:.6f} Ki={initial_pid.ki:.6f} "
            f"Kd={initial_pid.kd:.6f} Kd_flt={initial_pid.kd_filter:.4f} "
            f"Ramp={initial_pid.ramp_erpms_s:.0f}\n"
            f"Motor will spin..."
        )

    def stop_auto_tune(self):
        if self._auto_tuner:
            self._auto_tuner.stop()
            self._transport.send_frame(build_motor_stop())
            self.ai_text.append("\n[Auto-Tune] Stop requested \u2014 motor stopped.")
        if self._speed_auto_tuner:
            self._speed_auto_tuner.stop()
            self._transport.send_frame(build_motor_off())
            self.ai_text.append("\n[Auto-Tune] Stop requested \u2014 motor released.")

    def _on_tune_status(self, msg: str):
        self.ai_text.append(msg)

    def _on_metrics_ready(self, metrics: CanPositionMetrics):
        self._last_metrics = metrics
        self._update_display(metrics)

    def _on_speed_metrics_ready(self, metrics: MotorMetrics):
        self._last_metrics = metrics
        self._last_speed_metrics = metrics
        self._update_speed_display(metrics)

    def _on_tune_iteration(self, result: CanTuningIteration):
        if result.metrics:
            self._last_metrics = result.metrics
            self._update_display(result.metrics)

        if result.analysis:
            self.ai_text.append(f"\n--- Iteration {result.iteration} ---")
            self.ai_text.append(result.analysis.summary)
            if self.show_raw_cb.isChecked():
                self.ai_text.append("\n[RAW REQUEST]\n" + result.analysis.raw_request)
                self.ai_text.append("\n[RAW RESPONSE]\n" + result.analysis.raw_response)

            if result.analysis.suggested_pid:
                sp = result.analysis.suggested_pid
                self.sug_kp.setText(f"{sp.kp:.5f}")
                self.sug_ki.setText(f"{sp.ki:.5f}")
                self.sug_kd.setText(f"{sp.kd:.5f}")
                self.pid_kp.setValue(sp.kp)
                self.pid_ki.setValue(sp.ki)
                self.pid_kd.setValue(sp.kd)

        # History
        pid = result.pid
        score = result.metrics.quality_score if result.metrics else None
        ripple = result.metrics.pos_ripple_pct if result.metrics else None
        ss_err = result.metrics.steady_state_error_pct if result.metrics else None
        self._add_history_entry(pid.kp, pid.ki, pid.kd, score, ripple, ss_err, result.metrics)

    def _on_speed_tune_iteration(self, result: CanSpeedTuningIteration):
        if result.metrics:
            self._last_metrics = result.metrics
            self._last_speed_metrics = result.metrics
            self._update_speed_display(result.metrics)

        if result.analysis:
            self.ai_text.append(f"\n--- Iteration {result.iteration} ---")
            self.ai_text.append(result.analysis.summary)
            if self.show_raw_cb.isChecked():
                self.ai_text.append("\n[RAW REQUEST]\n" + result.analysis.raw_request)
                self.ai_text.append("\n[RAW RESPONSE]\n" + result.analysis.raw_response)

            if result.analysis.suggested_pid:
                sp = result.analysis.suggested_pid
                self.cur_kp.setValue(sp.kp)
                self.cur_ki.setValue(sp.ki)
                self.cur_kd.setValue(sp.kd)
                self.cur_kd_flt.setValue(sp.kd_filter)
                self.cur_ramp.setValue(sp.ramp_erpms_s)

        # History
        pid = result.pid
        score = result.metrics.quality_score if result.metrics else None
        ripple = result.metrics.rpm_ripple_pct if result.metrics else None
        ss_err = result.metrics.steady_state_error_pct if result.metrics else None
        self._add_history_entry(pid.kp, pid.ki, pid.kd, score, ripple, ss_err, result.metrics)

    def _on_tune_finished(self, reason: str):
        self.ai_text.append(f"\n[Auto-Tune Finished] {reason}")
        self._transport.send_frame(build_motor_stop())
        self.ai_text.append("[Motor stopped]")
        self.autotune_btn.setEnabled(True)
        self.stop_tune_btn.setEnabled(False)
        self.collect_btn.setEnabled(True)

        # Disconnect auto-tuner status
        if self._auto_tuner:
            try:
                self._transport.status_received.disconnect(self._auto_tuner.on_status)
            except TypeError:
                pass
        self._auto_tuner = None
        self._motor_confirmed = False

    def _on_speed_tune_finished(self, reason: str):
        self.ai_text.append(f"\n[Auto-Tune Finished] {reason}")
        self._transport.send_frame(build_motor_off())
        self.ai_text.append("[Motor released]")
        self.autotune_btn.setEnabled(True)
        self.stop_tune_btn.setEnabled(False)
        self.collect_btn.setEnabled(True)

        # Disconnect speed auto-tuner status
        if self._speed_auto_tuner:
            try:
                self._transport.status_received.disconnect(self._speed_auto_tuner.on_status)
            except TypeError:
                pass
        self._speed_auto_tuner = None
        self._motor_confirmed = False

        # Re-read MCCONF to update UI with final PID values
        if self._transport.is_connected():
            self._transport.send_vesc_to_target(build_get_mcconf())

    # ── LLM Analysis ────────────────────────────────────────────────

    def _run_llm(self):
        if self._last_metrics is None:
            return
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your-api-key-here":
            QMessageBox.warning(self, "API Key", "Set OPENAI_API_KEY in .env file.")
            return

        self.ai_btn.setEnabled(False)
        model_name = self.model_combo.currentText()
        lang = "Korean" if self.lang_combo.currentText() == "\ud55c\uad6d\uc5b4" else "English"
        is_speed = self.cmd_combo.currentIndex() in (2, 3)

        if is_speed:
            # Speed mode: use parent LLMAdvisor (speed PID)
            advisor = LLMAdvisor(api_key=api_key, model=model_name, language=lang)
            current_pid = PIDGains(
                kp=self.cur_kp.value(), ki=self.cur_ki.value(),
                kd=self.cur_kd.value(), kd_filter=self.cur_kd_flt.value(),
                ramp_erpms_s=self.cur_ramp.value(),
            )
            mode_str = "DPS (mode=0)" if self._speed_mode == 0 else "eRPM (mode=1)"
            unit_str = "dps" if self._speed_mode == 0 else "eRPM"
            context = (
                f"Control: CAN Speed (0xA2 {mode_str})\n"
                f"Target: {self._speed_target} {unit_str}\n"
                f"VESC Speed PID is configured via VESC-Tool serial, "
                f"not adjustable via CAN RMD. Provide analysis and tuning suggestions."
            )
        else:
            # Position mode: use CanPositionAdvisor
            advisor = CanPositionAdvisor(api_key=api_key, model=model_name, language=lang)
            current_pid = PIDGains(
                kp=self.pid_kp.value(), ki=self.pid_ki.value(), kd=self.pid_kd.value(),
            )
            context = (
                f"Control: CAN RMD Position\n"
                f"Target: {self.target_pos_spin.value():.1f}\u00b0\n"
                f"Start: {self.start_pos_spin.value():.1f}\u00b0"
            )

        m_dict, pid_dict, _ = advisor.build_request_data(self._last_metrics, current_pid, context)
        self.ai_text.setPlainText(
            f"=== Sending to {model_name} ===\n\n"
            f"PID:\n{json.dumps(pid_dict, indent=2)}\n\n"
            f"Metrics:\n{json.dumps(m_dict, indent=2)}\n\n"
            f"--- Waiting for AI response... ---"
        )

        def _run():
            try:
                result = advisor.analyze_and_recommend(self._last_metrics, current_pid, context)
                self._llm_result_signal.emit(result)
            except Exception as e:
                import traceback
                self._llm_error_signal.emit(f"Error: {e}\n{traceback.format_exc()}")

        threading.Thread(target=_run, daemon=True).start()

    def _on_llm_result(self, result: AnalysisResult):
        self.ai_btn.setEnabled(True)
        text = result.summary + "\n\n"
        if result.recommendations:
            text += "Recommendations:\n"
            for r in result.recommendations:
                text += f"  - {r}\n"
        if self.show_raw_cb.isChecked():
            text += "\n" + "=" * 50 + "\n"
            text += "=== RAW REQUEST ===\n" + result.raw_request + "\n\n"
            text += "=== RAW RESPONSE ===\n" + result.raw_response + "\n"
        self.ai_text.setPlainText(text)

        is_speed = self.cmd_combo.currentIndex() in (2, 3)
        if result.suggested_pid and not is_speed:
            self._suggested_pid = result.suggested_pid
            self.sug_kp.setText(f"{result.suggested_pid.kp:.5f}")
            self.sug_ki.setText(f"{result.suggested_pid.ki:.5f}")
            self.sug_kd.setText(f"{result.suggested_pid.kd:.5f}")
            self.apply_btn.setEnabled(True)

    def _on_llm_error(self, msg: str):
        self.ai_btn.setEnabled(True)
        self.ai_text.setPlainText(msg)

    # ── History Table ───────────────────────────────────────────────

    def _add_history_entry(self, kp, ki, kd, score=None, ripple=None, ss_err=None, metrics=None):
        self._history_counter += 1
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)

        items = [
            str(self._history_counter),
            f"{kp:.5f}", f"{ki:.5f}", f"{kd:.5f}",
            f"{score:.1f}" if score is not None else "--",
            f"{ripple:.2f}" if ripple is not None else "--",
            f"{ss_err:.2f}" if ss_err is not None else "--",
        ]
        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.history_table.setItem(row, col, item)
        self.history_table.scrollToBottom()

        self._pid_history.append({
            "kp": kp, "ki": ki, "kd": kd,
            "score": score, "ripple": ripple, "ss_err": ss_err,
            "metrics": metrics,
        })

    def _on_history_double_click(self, row: int, col: int):
        if row < 0 or row >= len(self._pid_history):
            return
        entry = self._pid_history[row]
        self._suggested_pid = PIDGains(kp=entry["kp"], ki=entry["ki"], kd=entry["kd"])
        self.sug_kp.setText(f"{entry['kp']:.5f}")
        self.sug_ki.setText(f"{entry['ki']:.5f}")
        self.sug_kd.setText(f"{entry['kd']:.5f}")
        self.apply_btn.setEnabled(True)

        if entry.get("metrics"):
            self._last_metrics = entry["metrics"]
            if isinstance(entry["metrics"], MotorMetrics):
                self._update_speed_display(entry["metrics"])
            else:
                self._update_display(entry["metrics"])
            self.ai_text.append(
                f"\n[History] Restored #{row + 1}: Score={entry['score']:.1f}"
            )

    # ── Helpers ──────────────────────────────────────────────────────

    def _safe_move(self, deg: float, dps: int = 1000):
        """Move to position using DPS-limited command (0xA4). Always safe."""
        self._transport.send_frame(build_set_multiturn_position(dps, deg))

    def _send_position(self, deg: float):
        """Send position command using selected mode."""
        if self.cmd_combo.currentIndex() == 1:  # Multiturn+DPS (0xA4)
            self._transport.send_frame(
                build_set_multiturn_position(self.dps_spin.value(), deg)
            )
        else:  # Position (0xA3)
            self._transport.send_frame(build_position_closed_loop_1(deg))

    def _get_rate_hz(self) -> float:
        return float(self.rate_combo.currentText().replace(" Hz", ""))

    # ── Current PID (serial MCCONF) ─────────────────────────────────

    def _read_mcconf_gains(self):
        """Read current PID gains from VESC via CAN EID COMM_GET_MCCONF."""
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected",
                                "PCAN connection required to read MCCONF.")
            return
        self._transport.send_vesc_to_target(build_get_mcconf())
        self.ai_text.append("[MCCONF] Reading current gains...")

    def _write_speed_pid(self):
        """Write current speed PID values to VESC via COMM_SET_MCCONF over CAN."""
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected",
                                "PCAN connection required to write MCCONF.")
            return
        if self._original_mcconf is None:
            QMessageBox.warning(
                self, "No MCCONF",
                "Read MCCONF first (use 'Read Gains') before writing."
            )
            return
        kp = self.cur_kp.value()
        ki = self.cur_ki.value()
        kd = self.cur_kd.value()
        kd_flt = self.cur_kd_flt.value()
        ramp = self.cur_ramp.value()
        packet = build_set_mcconf_with_pid(
            self._original_mcconf,
            kp, ki, kd, kd_flt,
            position_mode=False,
            ramp_erpms_s=ramp,
        )
        self._transport.send_vesc_to_target(packet)
        self.ai_text.append(
            f"[Speed PID] Written: Kp={kp:.6f} Ki={ki:.6f} "
            f"Kd={kd:.6f} Kd_flt={kd_flt:.6f} Ramp={ramp:.0f}"
        )

    def _read_mcconf_default(self):
        """Read default PID gains from VESC via CAN EID COMM_GET_MCCONF_DEFAULT."""
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected",
                                "PCAN connection required to read MCCONF.")
            return
        self._transport.send_vesc_to_target(build_get_mcconf_default())
        self.ai_text.append("[MCCONF] Reading default gains...")

    def on_mcconf_received(self, pid: McconfPid, raw_data: bytes = None,
                           is_default: bool = False):
        """Handle MCCONF PID data dispatched from MainWindow."""
        if raw_data and not is_default:
            self._original_mcconf = raw_data
        # Speed PID section
        self.cur_kp.setValue(pid.s_pid_kp)
        self.cur_ki.setValue(pid.s_pid_ki)
        self.cur_kd.setValue(pid.s_pid_kd)
        self.cur_kd_flt.setValue(pid.s_pid_kd_filter)
        self.cur_ramp.setValue(pid.s_pid_ramp_erpms_s)
        # Position PID Tuning section
        self.pid_kp.setValue(pid.p_pid_kp)
        self.pid_ki.setValue(pid.p_pid_ki)
        self.pid_kd.setValue(pid.p_pid_kd)
        self.pid_kd_flt.setValue(pid.p_pid_kd_filter)
        tag = "DEFAULT" if is_default else "MCCONF"
        self.ai_text.append(
            f"[{tag}] Speed PID: Kp={pid.s_pid_kp:.6f} Ki={pid.s_pid_ki:.6f} "
            f"Kd={pid.s_pid_kd:.6f} Kd_flt={pid.s_pid_kd_filter:.6f} "
            f"Ramp={pid.s_pid_ramp_erpms_s:.0f}"
        )
        self.ai_text.append(
            f"[{tag}] Pos PID: Kp={pid.p_pid_kp:.6f} Ki={pid.p_pid_ki:.6f} "
            f"Kd={pid.p_pid_kd:.6f} Kd_flt={pid.p_pid_kd_filter:.6f}"
        )

    def set_pole_pairs(self, value: int):
        """Set pole pairs from external source (e.g. mcconf foc_encoder_ratio)."""
        self.pole_spin.setValue(value)

    def _update_fft_plot(self, freqs, mags, dominant_freq: float):
        """Update FFT plot with frequency data and dominant frequency marker."""
        if freqs is None or mags is None or len(freqs) == 0 or len(mags) == 0:
            self.fft_curve.setData([], [])
            self.fft_dominant_line.setData([], [])
            return

        self.fft_curve.setData(freqs, mags)

        # Mark dominant frequency with vertical line
        if dominant_freq > 0:
            y_max = float(np.max(mags))
            self.fft_dominant_line.setData(
                [dominant_freq, dominant_freq], [0, y_max * 1.1]
            )
        else:
            self.fft_dominant_line.setData([], [])

    def _update_api_status(self):
        key = os.environ.get("OPENAI_API_KEY", "")
        if key and key != "your-api-key-here":
            self.api_label.setText("API \u2713")
            self.api_label.setStyleSheet("color: green;")
        else:
            self.api_label.setText("API: not set")
            self.api_label.setStyleSheet("color: red;")

    def cleanup(self):
        if self._speed_cmd_timer:
            self._speed_cmd_timer.stop()
            self._speed_cmd_timer = None
        if self._baseline_timer:
            self._baseline_timer.stop()
            self._baseline_timer = None
        if self._progress_timer:
            self._progress_timer.stop()
        if self._poller:
            self._poller.stop()
            deadline = time.time() + 2.0
            while self._poller.isRunning() and time.time() < deadline:
                QApplication.processEvents()
                time.sleep(0.005)
            if self._poller.isRunning():
                self._poller.terminate()
                self._poller.wait(500)
            self._poller = None
        if self._auto_tuner:
            self._auto_tuner.stop()
            try:
                self._transport.status_received.disconnect(self._auto_tuner.on_status)
            except TypeError:
                pass
            self._auto_tuner.wait(2000)
        if self._speed_auto_tuner:
            self._speed_auto_tuner.stop()
            try:
                self._transport.status_received.disconnect(self._speed_auto_tuner.on_status)
            except TypeError:
                pass
            self._speed_auto_tuner.wait(2000)
        self._collecting = False
        self._speed_collecting = False
