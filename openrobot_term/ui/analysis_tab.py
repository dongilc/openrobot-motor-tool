"""
AI Analysis + PID Tuning tab.

Combines local signal processing with LLM-based interpretation and recommendations.
Supports both manual analysis and automatic PID tuning loop.
"""

import time
import json
import os
from collections import deque

import numpy as np
import pyqtgraph as pg
from .plot_style import (
    style_plot, graph_pen, make_fill_brush,
    Crosshair, style_legend,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QTextEdit, QProgressBar, QGroupBox, QComboBox, QMessageBox,
    QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QMetaObject, Q_ARG, Qt as QtCore

from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import VescValues, build_get_values, build_set_pid_gains, McconfPid, build_set_mcconf_with_pid
from ..analysis.signal_metrics import MotorMetrics, analyze_speed_control
from ..analysis.current_metrics import CurrentMetrics, analyze_current_step, analyze_current_steady_state
from ..analysis.llm_advisor import LLMAdvisor, PIDGains, AnalysisResult, CurrentControlAdvisor, FOCCurrentGains
from ..analysis.auto_tuner import AutoTuner, TuningIteration
from ..analysis.current_auto_tuner import CurrentAutoTuner, CurrentTuningIteration


class AnalysisTab(QWidget):
    # Signals for thread-safe UI updates
    _llm_result_signal = pyqtSignal(object)
    _llm_error_signal = pyqtSignal(str)

    def __init__(self, transport: SerialTransport):
        super().__init__()
        self._transport = transport
        self._advisor = None
        self._auto_tuner = None
        self._collecting = False
        self._collect_timer = None

        # Data buffers for manual analysis
        self._rpm_buf = deque(maxlen=10000)
        self._current_buf = deque(maxlen=10000)
        self._voltage_buf = deque(maxlen=10000)
        self._input_current_buf = deque(maxlen=10000)
        self._time_buf = deque(maxlen=10000)
        self._t0 = None

        self._build_ui()

        # Connect signals for thread-safe UI updates
        self._llm_result_signal.connect(self._on_llm_result)
        self._llm_error_signal.connect(self._on_llm_error)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ===== Row 1: Collection controls =====
        collect_group = QGroupBox("Data Collection & Analysis")
        collect_layout = QHBoxLayout(collect_group)

        # Control mode selector
        collect_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Speed Control"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        collect_layout.addWidget(self.mode_combo)

        self.target_label = QLabel("Target RPM:")
        collect_layout.addWidget(self.target_label)
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 100000)
        self.target_spin.setValue(5000)
        collect_layout.addWidget(self.target_spin)

        # Target current (for Current Control mode, hidden initially)
        self.target_current_label = QLabel("Target A:")
        self.target_current_label.hide()
        collect_layout.addWidget(self.target_current_label)
        self.target_current_spin = QDoubleSpinBox()
        self.target_current_spin.setRange(0.1, 100.0)
        self.target_current_spin.setValue(1.0)
        self.target_current_spin.setSuffix(" A")
        self.target_current_spin.hide()
        collect_layout.addWidget(self.target_current_spin)

        collect_layout.addWidget(QLabel("Duration:"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 30.0)
        self.duration_spin.setValue(5.0)
        self.duration_spin.setSuffix(" s")
        collect_layout.addWidget(self.duration_spin)

        collect_layout.addWidget(QLabel("Rate:"))
        self.rate_combo = QComboBox()
        for r in ["10 Hz", "20 Hz", "50 Hz"]:
            self.rate_combo.addItem(r)
        self.rate_combo.setCurrentText("50 Hz")
        collect_layout.addWidget(self.rate_combo)

        self.collect_btn = QPushButton("Collect & Analyze")
        self.collect_btn.clicked.connect(self.start_collection)
        collect_layout.addWidget(self.collect_btn)

        self.collect_progress = QProgressBar()
        self.collect_progress.setRange(0, 100)
        self.collect_progress.setValue(0)
        self.collect_progress.setMaximumWidth(150)
        collect_layout.addWidget(self.collect_progress)

        # Auto-Tune buttons (right side)
        collect_layout.addStretch()

        self.autotune_btn = QPushButton("Auto-Tune")
        self.autotune_btn.setStyleSheet("background-color: #2d5a27; font-weight: bold;")
        self.autotune_btn.clicked.connect(self.start_auto_tune)
        collect_layout.addWidget(self.autotune_btn)

        self.stop_tune_btn = QPushButton("Stop")
        self.stop_tune_btn.setStyleSheet("background-color: #8b0000;")
        self.stop_tune_btn.clicked.connect(self.stop_auto_tune)
        self.stop_tune_btn.setEnabled(False)
        collect_layout.addWidget(self.stop_tune_btn)

        layout.addWidget(collect_group)

        # ===== Row 2: Quality Score =====
        score_layout = QHBoxLayout()
        layout.addLayout(score_layout)

        self.score_label = QLabel("Quality Score: --")
        self.score_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 8px;"
        )
        score_layout.addWidget(self.score_label)

        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setTextVisible(True)
        self.score_bar.setFormat("%v / 100")
        self.score_bar.setMinimumHeight(30)
        score_layout.addWidget(self.score_bar)

        # ===== Main content area: Left (Plots + PID) | Right (Metrics + AI) =====
        main_content = QHBoxLayout()
        layout.addLayout(main_content, stretch=1)

        # ===== LEFT PANEL: Plots + PID Tuning =====
        left_panel = QVBoxLayout()
        main_content.addLayout(left_panel, stretch=7)

        # === Plots Row: Step Response (left) | FFT (right) ===
        plots_layout = QHBoxLayout()
        left_panel.addLayout(plots_layout, stretch=1)

        # Step Response Plot (Time Domain)
        self.step_plot = pg.PlotWidget()
        style_plot(self.step_plot, title="Step Response (Transient)",
                   left_label="RPM", left_unit="",
                   bottom_label="Time", bottom_unit="s")
        self.step_curve = self.step_plot.plot(pen=graph_pen(0))  # Blue for actual RPM
        self.target_line = self.step_plot.plot(pen=pg.mkPen(color='#ff6666', width=2, style=Qt.PenStyle.DashLine))  # Red dashed for target
        self.settle_band_upper = self.step_plot.plot(pen=pg.mkPen(color='#66ff66', width=1, style=Qt.PenStyle.DotLine))  # Green dotted for ±2% band
        self.settle_band_lower = self.step_plot.plot(pen=pg.mkPen(color='#66ff66', width=1, style=Qt.PenStyle.DotLine))
        Crosshair(self.step_plot)
        plots_layout.addWidget(self.step_plot, stretch=1)

        # FFT Plot (Frequency Domain)
        self.fft_plot = pg.PlotWidget()
        style_plot(self.fft_plot, title="RPM FFT Spectrum",
                   left_label="Magnitude", left_unit="",
                   bottom_label="Frequency", bottom_unit="Hz")
        self.fft_curve = self.fft_plot.plot(pen=graph_pen(4))
        Crosshair(self.fft_plot)
        plots_layout.addWidget(self.fft_plot, stretch=1)

        # PID Tuning
        self.pid_group = QGroupBox("PID Tuning")
        self.pid_group.setMaximumHeight(200)
        pid_layout = QHBoxLayout(self.pid_group)

        # Current PID
        self.cur_group = QGroupBox("Current PID")
        cur_layout = QGridLayout(self.cur_group)
        self.pid_kp = QDoubleSpinBox()
        self.pid_ki = QDoubleSpinBox()
        self.pid_kd = QDoubleSpinBox()
        self.pid_kd_filter = QDoubleSpinBox()
        self.pid_ramp = QDoubleSpinBox()

        # Store labels for show/hide
        self.pid_kp_label = QLabel("Kp:")
        self.pid_ki_label = QLabel("Ki:")
        self.pid_kd_label = QLabel("Kd:")
        self.pid_kd_filter_label = QLabel("Kd Flt:")
        self.pid_ramp_label = QLabel("Ramp:")

        for i, (label, spin) in enumerate([
            (self.pid_kp_label, self.pid_kp), (self.pid_ki_label, self.pid_ki),
            (self.pid_kd_label, self.pid_kd), (self.pid_kd_filter_label, self.pid_kd_filter)
        ]):
            spin.setDecimals(6)
            spin.setRange(0, 100)
            spin.setSingleStep(0.001)
            cur_layout.addWidget(label, i, 0)
            cur_layout.addWidget(spin, i, 1)
        # Ramp has different range (eRPM/s), -1 means disabled
        self.pid_ramp.setDecimals(0)
        self.pid_ramp.setRange(-1, 1000000)
        self.pid_ramp.setSingleStep(1000)
        cur_layout.addWidget(self.pid_ramp_label, 4, 0)
        cur_layout.addWidget(self.pid_ramp, 4, 1)

        self.read_pid_btn = QPushButton("Read Gains")
        self.read_pid_btn.clicked.connect(self.read_current_pid)
        cur_layout.addWidget(self.read_pid_btn, 5, 0)

        self.read_default_btn = QPushButton("Read Default")
        self.read_default_btn.clicked.connect(self.load_default_pid)
        cur_layout.addWidget(self.read_default_btn, 5, 1)
        pid_layout.addWidget(self.cur_group)

        # Copy to Suggested button (between Current and Suggested)
        mid_layout = QVBoxLayout()
        mid_layout.addStretch()
        self.copy_to_suggested_btn = QPushButton("Apply\n>>>")
        self.copy_to_suggested_btn.setStyleSheet("font-size: 10px; padding: 8px;")
        self.copy_to_suggested_btn.clicked.connect(self.copy_to_suggested)
        mid_layout.addWidget(self.copy_to_suggested_btn)
        mid_layout.addStretch()
        pid_layout.addLayout(mid_layout)

        # Suggested PID
        self.sug_group = QGroupBox("Suggested PID")
        sug_layout = QGridLayout(self.sug_group)
        self.sug_kp = QLabel("--")
        self.sug_ki = QLabel("--")
        self.sug_kd = QLabel("--")
        self.sug_kd_filter = QLabel("--")
        self.sug_ramp = QLabel("--")

        # Store labels for show/hide
        self.sug_kp_label = QLabel("Kp:")
        self.sug_ki_label = QLabel("Ki:")
        self.sug_kd_label = QLabel("Kd:")
        self.sug_kd_filter_label = QLabel("Kd Flt:")
        self.sug_ramp_label = QLabel("Ramp:")

        for i, (label, widget) in enumerate([
            (self.sug_kp_label, self.sug_kp), (self.sug_ki_label, self.sug_ki),
            (self.sug_kd_label, self.sug_kd), (self.sug_kd_filter_label, self.sug_kd_filter),
            (self.sug_ramp_label, self.sug_ramp)
        ]):
            widget.setStyleSheet("font-family: monospace; font-weight: bold;")
            sug_layout.addWidget(label, i, 0)
            sug_layout.addWidget(widget, i, 1)
        pid_layout.addWidget(self.sug_group)

        # Apply button (centered vertically)
        btn_layout = QVBoxLayout()
        btn_layout.addStretch()
        self.apply_btn = QPushButton("Set\nParameter")
        self.apply_btn.setStyleSheet("font-size: 12px; font-weight: bold; padding: 12px 16px;")
        self.apply_btn.clicked.connect(self.apply_suggested_pid)
        self.apply_btn.setEnabled(False)
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addStretch()
        pid_layout.addLayout(btn_layout)
        left_panel.addWidget(self.pid_group)

        # ===== RIGHT PANEL: Metrics + AI Analysis =====
        right_panel = QVBoxLayout()
        main_content.addLayout(right_panel, stretch=1)

        # Metrics panel (top right) - compact 2-column grid
        metrics_group = QGroupBox("Metrics")
        metrics_group.setMaximumWidth(550)
        metrics_grid = QGridLayout(metrics_group)
        metrics_grid.setSpacing(4)
        self.metrics_labels = {}
        metrics_items = [
            "RPM Ripple", "THD", "Settling Time", "Overshoot",
            "Rise Time", "SS Error", "Power Mean", "Dominant Freq"
        ]
        for i, name in enumerate(metrics_items):
            row = i // 2
            col = (i % 2) * 2  # 0 or 2
            label = QLabel(f"{name}:")
            label.setStyleSheet("font-size: 11px;")
            value = QLabel("--")
            value.setStyleSheet("font-family: monospace; font-size: 11px;")
            metrics_grid.addWidget(label, row, col)
            metrics_grid.addWidget(value, row, col + 1)
            self.metrics_labels[name] = value
        metrics_group.setMaximumHeight(140)
        right_panel.addWidget(metrics_group)

        # AI Analysis (bottom right - takes remaining vertical space)
        ai_group = QGroupBox("AI Analysis")
        ai_group.setMaximumWidth(550)
        ai_layout = QVBoxLayout(ai_group)

        # LLM settings row
        llm_row = QHBoxLayout()
        llm_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"])
        llm_row.addWidget(self.model_combo)

        # Language selector
        llm_row.addWidget(QLabel("Lang:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["한국어", "English"])
        self.lang_combo.setToolTip("AI response language")
        llm_row.addWidget(self.lang_combo)

        self.ai_analyze_btn = QPushButton("Ask AI")
        self.ai_analyze_btn.clicked.connect(self.run_llm_analysis)
        self.ai_analyze_btn.setEnabled(False)
        llm_row.addWidget(self.ai_analyze_btn)

        self.api_status_label = QLabel()
        self._update_api_status()
        llm_row.addWidget(self.api_status_label)

        llm_row.addStretch()

        # Show Raw checkbox for debugging AI conversations
        from PyQt6.QtWidgets import QCheckBox
        self.show_raw_cb = QCheckBox("Show Raw")
        self.show_raw_cb.setToolTip("Show raw API request/response for debugging")
        llm_row.addWidget(self.show_raw_cb)

        ai_layout.addLayout(llm_row)

        self.ai_text = QTextEdit()
        self.ai_text.setReadOnly(True)
        self.ai_text.setPlaceholderText("AI analysis results will appear here...")
        ai_layout.addWidget(self.ai_text)

        # Store raw conversation data
        self._last_raw_request = ""
        self._last_raw_response = ""
        right_panel.addWidget(ai_group, stretch=1)  # AI takes remaining vertical space

        # ===== PID Tuning History Table =====
        history_group = QGroupBox("PID Tuning History")
        history_layout = QVBoxLayout(history_group)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(9)
        self.history_table.setHorizontalHeaderLabels([
            "#", "Kp", "Ki", "Kd", "Kd_flt", "Ramp", "Score", "Ripple%", "SS Err%"
        ])
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setMaximumHeight(125)

        # Adjust column widths
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.history_table.setColumnWidth(0, 30)  # #
        for col in range(1, 9):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

        history_layout.addWidget(self.history_table)

        # Connect double-click to restore PID and graphs
        self.history_table.cellDoubleClicked.connect(self._on_history_double_click)

        # Clear history button
        clear_btn = QPushButton("Clear History")
        clear_btn.setMaximumWidth(100)
        clear_btn.clicked.connect(self._clear_history)
        history_layout.addWidget(clear_btn)

        left_panel.addWidget(history_group)

        # Internal state
        self._last_metrics = None
        self._last_current_metrics = None  # For current control mode
        self._suggested_pid = None
        self._tune_history = []
        self._control_mode = "speed"  # "speed" or "current"
        self._original_mcconf = None  # Store original MCCONF for PID updates
        self._pid_history = []  # List of (pid, metrics) tuples for history table
        self._history_counter = 0
        self._pending_auto_tune = False  # Flag for auto-loading MCCONF before auto-tune
        self._pending_apply_suggested = False  # Flag for auto-loading MCCONF before apply
        self._motor_confirmed = False  # Flag for user motor rotation confirmation
        self._requesting_default = False  # Flag for default MCCONF request
        self._silent_mcconf_load = False  # Flag to load MCCONF without updating UI

    def _clear_history(self):
        """Clear the PID tuning history table."""
        self.history_table.setRowCount(0)
        self._pid_history.clear()
        self._history_counter = 0

    def _add_history_entry(self, kp: float, ki: float, kd: float, kd_filter: float,
                           score: float = None, ripple: float = None, ss_err: float = None,
                           metrics: MotorMetrics = None, ramp_erpms_s: float = None):
        """Add an entry to the PID tuning history table."""
        self._history_counter += 1
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)

        # Add items
        items = [
            str(self._history_counter),
            f"{kp:.6f}",
            f"{ki:.6f}",
            f"{kd:.6f}",
            f"{kd_filter:.4f}",
            f"{ramp_erpms_s:.0f}" if ramp_erpms_s is not None else "--",
            f"{score:.1f}" if score is not None else "--",
            f"{ripple:.2f}" if ripple is not None else "--",
            f"{ss_err:.2f}" if ss_err is not None else "--",
        ]

        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.history_table.setItem(row, col, item)

        # Scroll to the new row
        self.history_table.scrollToBottom()

        # Store in internal list (including metrics for graph restoration)
        self._pid_history.append({
            "kp": kp, "ki": ki, "kd": kd, "kd_filter": kd_filter,
            "ramp_erpms_s": ramp_erpms_s,
            "score": score, "ripple": ripple, "ss_err": ss_err,
            "metrics": metrics  # Store metrics for graph restoration
        })

    def _add_current_history_entry(self, result: CurrentTuningIteration):
        """Add a Current Control tuning entry to the history table."""
        self._history_counter += 1
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)

        gains = result.gains
        score = result.score
        ripple = result.steady_metrics.current_ripple_pct if result.steady_metrics else None
        thd = result.steady_metrics.current_thd if result.steady_metrics else None

        # Add items (reuse columns: #, Kp, Ki, Kd(--), Kd_flt(--), Ramp(--), Score, Ripple%, SS Err%->THD%)
        items = [
            str(self._history_counter),
            f"{gains.kp:.6f}",
            f"{gains.ki:.3f}",
            "--",  # Kd not used for current control
            "--",  # Kd_filter not used
            "--",  # Ramp not used
            f"{score:.1f}" if score is not None else "--",
            f"{ripple:.2f}" if ripple is not None else "--",
            f"{thd:.2f}" if thd is not None else "--",  # THD instead of SS Err
        ]

        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.history_table.setItem(row, col, item)

        self.history_table.scrollToBottom()

        # Store in internal list
        self._pid_history.append({
            "kp": gains.kp, "ki": gains.ki, "kd": 0, "kd_filter": 0,
            "ramp_erpms_s": 0,
            "score": score, "ripple": ripple, "ss_err": thd,
            "metrics": result.steady_metrics,
            "is_current_control": True,
        })

    def _on_history_double_click(self, row: int, col: int):
        """Handle double-click on history table row to restore PID and graphs."""
        if row < 0 or row >= len(self._pid_history):
            return

        entry = self._pid_history[row]
        kp = entry["kp"]
        ki = entry["ki"]
        kd = entry["kd"]
        kd_filter = entry["kd_filter"]
        metrics = entry.get("metrics")

        # Update Suggested PID
        ramp = entry.get("ramp_erpms_s", self.pid_ramp.value())
        self.sug_kp.setText(f"{kp:.6f}")
        self.sug_ki.setText(f"{ki:.6f}")
        self.sug_kd.setText(f"{kd:.6f}")
        self.sug_kd_filter.setText(f"{kd_filter:.6f}")
        self.sug_ramp.setText(f"{ramp:.0f}")

        # Store as suggested PID for apply button
        self._suggested_pid = PIDGains(kp=kp, ki=ki, kd=kd, kd_filter=kd_filter, ramp_erpms_s=ramp)
        self.apply_btn.setEnabled(True)

        # Restore graphs if metrics available
        if metrics is not None:
            self._last_metrics = metrics
            self._update_metrics_display(metrics)
            self.ai_text.append(
                f"\n[History] Restored iteration #{row + 1}: "
                f"Score={entry['score']:.1f}, PID(Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f})"
            )
        else:
            self.ai_text.append(
                f"\n[History] Selected iteration #{row + 1}: "
                f"PID(Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}) - No graph data available"
            )

    def _on_mode_changed(self, text: str):
        """Handle control mode change."""
        self._control_mode = "speed"
        self.target_label.setText("Target RPM:")
        self.target_spin.setRange(0, 100000)
        self.target_spin.setValue(5000)
        self.target_label.show()
        self.target_spin.show()
        self.target_current_label.hide()
        self.target_current_spin.hide()
        # Reset plot titles and Y-axis labels
        self.step_plot.setTitle("Step Response (Transient)")
        self.step_plot.setLabel('left', 'RPM', units='')
        self.step_plot.setLabel('bottom', 'Time', units='s')
        self.fft_plot.setTitle("RPM FFT Spectrum")
        # Restore PID tuning section for speed control
        self.pid_group.setTitle("PID Tuning")
        self.cur_group.setTitle("Current PID")
        self.sug_group.setTitle("Suggested PID")
        self._show_all_pid_fields()
        # Restore metrics labels for speed control
        self._update_metrics_labels_for_speed()

    def _show_all_pid_fields(self):
        """Show all PID fields (used when switching back from Current Control mode)."""
        self.pid_kd_label.show()
        self.pid_kd.show()
        self.pid_kd_filter_label.show()
        self.pid_kd_filter.show()
        self.pid_ramp_label.show()
        self.pid_ramp.show()
        self.sug_kd_label.show()
        self.sug_kd.show()
        self.sug_kd_filter_label.show()
        self.sug_kd_filter.show()
        self.sug_ramp_label.show()
        self.sug_ramp.show()

    def _update_metrics_labels_for_current(self):
        """Update metrics labels for Current Control mode."""
        # Find and update the "RPM Ripple" label to "Current Ripple"
        for i in range(self.metrics_labels["RPM Ripple"].parent().layout().count()):
            item = self.metrics_labels["RPM Ripple"].parent().layout().itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QLabel) and widget.text() == "RPM Ripple:":
                    widget.setText("Current Ripple:")
                    break

    def _update_metrics_labels_for_speed(self):
        """Restore metrics labels for Speed/Position Control mode."""
        # Find and update the "Current Ripple" label back to "RPM Ripple"
        for i in range(self.metrics_labels["RPM Ripple"].parent().layout().count()):
            item = self.metrics_labels["RPM Ripple"].parent().layout().itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QLabel) and widget.text() == "Current Ripple:":
                    widget.setText("RPM Ripple:")
                    break

    def read_current_pid(self):
        """Request current PID values from VESC via COMM_GET_MCCONF."""
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to VESC first.")
            return
        from ..protocol.commands import build_get_mcconf
        self._transport.send_packet(build_get_mcconf())
        self.ai_text.setPlainText("Sent COMM_GET_MCCONF (14) to VESC.\nWaiting for response...")

    def on_mcconf_received(self, pid: McconfPid, raw_data: bytes = None):
        """Handle received MCCONF PID values."""
        is_default = self._requesting_default
        is_silent = self._silent_mcconf_load
        self._requesting_default = False
        self._silent_mcconf_load = False

        # Store original MCCONF data for later PID updates (not for default request)
        if raw_data and not is_default:
            self._original_mcconf = raw_data
            print(f"[MCCONF] Stored original data: {len(raw_data)} bytes", flush=True)

        # Skip UI update if silent load (for Apply without overwriting user values)
        if is_silent:
            self.ai_text.append("\n[MCCONF] Template loaded (UI not updated)")
            # Check pending operations
            if self._pending_apply_suggested:
                self._pending_apply_suggested = False
                self.ai_text.append("\n[Apply] Applying suggested PID...")
                QTimer.singleShot(200, self.apply_suggested_pid)
            return

        # Determine label prefix
        label = "DEFAULT " if is_default else ""

        # Update UI based on current control mode
        if self._control_mode == "current":
            # FOC Current Controller gains
            self.pid_kp.setValue(pid.foc_current_kp)
            self.pid_ki.setValue(pid.foc_current_ki)
            self.ai_text.setPlainText(
                f"{label}FOC Current Controller received from VESC:\n"
                f"  Current Kp = {pid.foc_current_kp:.6f}\n"
                f"  Current Ki = {pid.foc_current_ki:.6f}\n\n"
                f"{label}Speed PID (for reference):\n"
                f"  Kp = {pid.s_pid_kp:.6f}\n"
                f"  Ki = {pid.s_pid_ki:.6f}\n"
                f"  Kd = {pid.s_pid_kd:.6f}\n"
                f"  Kd Filter = {pid.s_pid_kd_filter:.6f}"
            )
        else:
            # Speed control mode
            self.pid_kp.setValue(pid.s_pid_kp)
            self.pid_ki.setValue(pid.s_pid_ki)
            self.pid_kd.setValue(pid.s_pid_kd)
            self.pid_kd_filter.setValue(pid.s_pid_kd_filter)
            self.pid_ramp.setValue(pid.s_pid_ramp_erpms_s)
            self.ai_text.setPlainText(
                f"{label}Speed PID received from VESC:\n"
                f"  Kp = {pid.s_pid_kp:.6f}\n"
                f"  Ki = {pid.s_pid_ki:.6f}\n"
                f"  Kd = {pid.s_pid_kd:.6f}\n"
                f"  Kd Filter = {pid.s_pid_kd_filter:.6f}\n"
                f"  Ramp = {pid.s_pid_ramp_erpms_s:.0f} eRPM/s\n"
                f"  Min ERPM = {pid.s_pid_min_erpm:.1f}\n"
                f"  Allow Braking = {pid.s_pid_allow_braking}\n\n"
                f"{label}FOC Current Controller (for reference):\n"
                f"  Current Kp = {pid.foc_current_kp:.6f}\n"
                f"  Current Ki = {pid.foc_current_ki:.2f}\n\n"
                f"{label}Position PID (for reference):\n"
                f"  Kp = {pid.p_pid_kp:.6f}\n"
                f"  Ki = {pid.p_pid_ki:.6f}\n"
                f"  Kd = {pid.p_pid_kd:.6f}\n"
                f"  Kd Filter = {pid.p_pid_kd_filter:.6f}"
            )

        # Check if auto-tune was pending MCCONF load
        if self._pending_auto_tune:
            self._pending_auto_tune = False
            self.ai_text.append("\n[Auto-Tune] MCCONF loaded. Starting auto-tune in 500ms...")
            QTimer.singleShot(500, self._start_auto_tune_internal)

        # Check if apply suggested was pending MCCONF load
        if self._pending_apply_suggested:
            self._pending_apply_suggested = False
            self.ai_text.append("\n[Apply] MCCONF loaded. Applying suggested PID...")
            QTimer.singleShot(200, self.apply_suggested_pid)

    def load_default_pid(self):
        """Request default PID values from VESC via COMM_GET_MCCONF_DEFAULT."""
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to VESC first.")
            return
        from ..protocol.commands import build_get_mcconf_default
        self._requesting_default = True
        self._transport.send_packet(build_get_mcconf_default())
        self.ai_text.setPlainText("Sent COMM_GET_MCCONF_DEFAULT (15) to VESC.\nWaiting for default configuration...")

    def _update_api_status(self):
        openai_key = os.environ.get("OPENAI_API_KEY", "")

        if openai_key and openai_key != "your-api-key-here":
            self.api_status_label.setText("API: OpenAI ✓")
            self.api_status_label.setStyleSheet("color: green;")
        else:
            self.api_status_label.setText("API Key: not set (.env)")
            self.api_status_label.setStyleSheet("color: red;")

    def _get_api_key(self) -> str:
        """Get OpenAI API key."""
        return os.environ.get("OPENAI_API_KEY", "")

    def _get_rate_hz(self) -> float:
        return float(self.rate_combo.currentText().replace(" Hz", ""))

    def on_values(self, v: VescValues):
        """Called by main window with each COMM_GET_VALUES response during collection."""
        # Feed auto-tuner if running (MUST be before the _collecting check!)
        if self._auto_tuner and self._auto_tuner.isRunning():
            self._auto_tuner.on_values(v)

        # Manual collection mode
        if not self._collecting:
            return
        if self._t0 is None:
            self._t0 = time.time()
        self._time_buf.append(time.time() - self._t0)
        self._rpm_buf.append(v.rpm)
        self._current_buf.append(v.avg_motor_current)
        self._voltage_buf.append(v.v_in)
        self._input_current_buf.append(v.avg_input_current)

    def start_collection(self):
        """Start collecting data for manual analysis."""
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to VESC first.")
            return

        # For Speed Control mode, warn user
        if self._control_mode == "speed":
            target_rpm = int(self.target_spin.value())
            duration = self.duration_spin.value()
            reply = QMessageBox.warning(
                self, "Motor Warning",
                f"Speed Control Test:\n\n"
                f"  Target: {target_rpm:,} RPM\n"
                f"  Duration: {duration:.1f}s\n\n"
                f"The motor will spin during the test.\n"
                f"Make sure the motor is safe to run!",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._rpm_buf.clear()
        self._current_buf.clear()
        self._voltage_buf.clear()
        self._input_current_buf.clear()
        self._time_buf.clear()
        self._t0 = None
        self._collecting = True
        self.collect_btn.setEnabled(False)
        self.collect_btn.setText("Collecting...")
        self.collect_progress.setValue(0)

        duration_s = self.duration_spin.value()
        rate_hz = self._get_rate_hz()
        interval_ms = int(1000 / rate_hz)

        # Clean up any existing timer
        if hasattr(self, '_poll_timer') and self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer.deleteLater()
            self._poll_timer = None

        # For Speed Control mode, start motor automatically
        if self._control_mode == "speed":
            from ..protocol.commands import build_set_rpm
            target_rpm = int(self.target_spin.value())
            self._transport.send_packet(build_set_rpm(target_rpm))

        # Timer for sending COMM_GET_VALUES
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(interval_ms)
        self._poll_timer.timeout.connect(self._poll_tick)
        self._poll_start = time.time()
        self._poll_duration = duration_s
        self._poll_timer.start()

    def _poll_tick(self):
        elapsed = time.time() - self._poll_start
        progress = min(100, int(elapsed / self._poll_duration * 100))
        self.collect_progress.setValue(progress)

        if self._transport.is_connected():
            # For Speed Control mode: keep sending RPM command (VESC timeout)
            if self._control_mode == "speed":
                from ..protocol.commands import build_set_rpm
                target_rpm = int(self.target_spin.value())
                self._transport.send_packet(build_set_rpm(target_rpm))

            self._transport.send_packet(build_get_values())

        if elapsed >= self._poll_duration:
            self._poll_timer.stop()
            self._collecting = False
            self.collect_btn.setEnabled(True)
            self.collect_btn.setText("Collect & Analyze")
            self.collect_progress.setValue(100)

            # Stop motor after collection
            if self._transport.is_connected():
                from ..protocol.commands import build_set_current
                self._transport.send_packet(build_set_current(0.0))  # Stop motor
            self._run_local_analysis()

    def _run_local_analysis(self):
        """Run signal processing analysis on collected data."""
        # Speed/Position control analysis
        if len(self._rpm_buf) < 10:
            self.score_label.setText("Quality Score: -- (insufficient data)")
            return

        # Speed control analysis
        metrics = analyze_speed_control(
            rpm_data=np.array(self._rpm_buf),
            target_rpm=float(self.target_spin.value()),
            sample_rate=self._get_rate_hz(),
            current_data=np.array(self._current_buf),
            voltage_data=np.array(self._voltage_buf),
            input_current_data=np.array(self._input_current_buf),
        )

        self._last_metrics = metrics
        self._update_metrics_display(metrics)
        self.ai_analyze_btn.setEnabled(True)

    def _update_metrics_display(self, m: MotorMetrics):
        # Score
        score = m.quality_score
        print(f"[UI] _update_metrics_display called with score={score:.1f}", flush=True)
        self.score_label.setText(f"Quality Score: {score:.1f}")
        self.score_bar.setValue(int(score))
        self.score_bar.setFormat(f"{int(score)} / 100")

        # Color the score bar
        if score >= 80:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #44bb44; }")
        elif score >= 50:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #ddaa00; }")
        else:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #dd4444; }")

        # Metrics
        self.metrics_labels["RPM Ripple"].setText(f"{m.rpm_ripple_pct:.2f} %")
        self.metrics_labels["THD"].setText(f"{m.current_thd:.2f} %")
        self.metrics_labels["Settling Time"].setText(f"{m.settling_time:.3f} s")
        self.metrics_labels["Overshoot"].setText(f"{m.overshoot_pct:.2f} %")
        self.metrics_labels["Rise Time"].setText(f"{m.rise_time:.3f} s")
        self.metrics_labels["SS Error"].setText(
            f"{m.steady_state_error:.1f} RPM ({m.steady_state_error_pct:.2f}%)"
        )
        self.metrics_labels["Power Mean"].setText(f"{m.power_mean:.1f} W")
        self.metrics_labels["Dominant Freq"].setText(f"{m.fft_dominant_freq:.1f} Hz")

        # FFT plot
        if len(m.fft_frequencies) > 0:
            self.fft_curve.setData(m.fft_frequencies, m.fft_magnitudes)

        # Step Response plot (Time Domain) - show 0 to 0.5 seconds (transient only)
        if len(m.time_data) > 0 and len(m.rpm_data) > 0:
            self.step_curve.setData(m.time_data, m.rpm_data)

            # Set X-axis range to focus on transient (0 to 0.5 seconds)
            self.step_plot.setXRange(0, 0.5, padding=0.02)

            # Target line (horizontal)
            target = m.target_rpm
            self.target_line.setData([0, 0.5], [target, target])

            # ±2% settling band
            band = abs(target) * 0.02
            self.settle_band_upper.setData([0, 0.5], [target + band, target + band])
            self.settle_band_lower.setData([0, 0.5], [target - band, target - band])

        print(f"[UI] Updated metrics display: score={m.quality_score:.1f}", flush=True)

    def _update_current_metrics_display(self, m: CurrentMetrics):
        """Update UI with current control metrics."""
        score = m.quality_score
        print(f"[UI] _update_current_metrics_display called with score={score:.1f}", flush=True)
        self.score_label.setText(f"Quality Score: {score:.1f}")
        self.score_bar.setValue(int(score))
        self.score_bar.setFormat(f"{int(score)} / 100")

        # Color the score bar
        if score >= 80:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #44bb44; }")
        elif score >= 50:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #ddaa00; }")
        else:
            self.score_bar.setStyleSheet("QProgressBar::chunk { background: #dd4444; }")

        # Metrics - adapt labels for current control
        self.metrics_labels["RPM Ripple"].setText(f"{m.current_ripple_pct:.2f} %")  # Current Ripple
        self.metrics_labels["THD"].setText(f"{m.current_thd:.2f} %")
        self.metrics_labels["Settling Time"].setText(f"{m.settling_time_ms:.1f} ms")
        self.metrics_labels["Overshoot"].setText(f"{m.overshoot_pct:.2f} %")
        self.metrics_labels["Rise Time"].setText(f"{m.rise_time_ms:.1f} ms")
        self.metrics_labels["SS Error"].setText(
            f"{m.tracking_error:.2f} A ({m.tracking_error_pct:.2f}%)"
        )
        self.metrics_labels["Power Mean"].setText(f"{m.power_mean:.1f} W")
        self.metrics_labels["Dominant Freq"].setText(f"{m.fft_dominant_freq:.1f} Hz")

        # FFT plot
        if len(m.fft_frequencies) > 0:
            self.fft_curve.setData(m.fft_frequencies, m.fft_magnitudes)

        # Current Waveform plot (Time Domain in seconds)
        if len(m.time_data) > 0 and len(m.current_data) > 0:
            self.step_curve.setData(m.time_data, m.current_data)

            # Set X-axis range to match duration setting
            max_time = self.duration_spin.value()
            self.step_plot.setXRange(0, max_time, padding=0.02)

            # Mean current line (horizontal) - use measured mean
            mean_current = m.current_mean if m.current_mean != 0 else np.mean(m.current_data)
            self.target_line.setData([0, max_time], [mean_current, mean_current])

            # ±ripple band (show actual ripple range)
            ripple_half = m.current_ripple_pp / 2 if m.current_ripple_pp > 0 else abs(mean_current) * 0.05
            self.settle_band_upper.setData([0, max_time], [mean_current + ripple_half, mean_current + ripple_half])
            self.settle_band_lower.setData([0, max_time], [mean_current - ripple_half, mean_current - ripple_half])

        print(f"[UI] Updated current metrics display: score={m.quality_score:.1f}", flush=True)

    def run_llm_analysis(self):
        """Send metrics to LLM for analysis."""
        if self._last_metrics is None:
            return

        model_name = self.model_combo.currentText()
        openai_key = self._get_api_key()

        if not openai_key or openai_key == "your-api-key-here":
            QMessageBox.warning(self, "API Key", "Set OPENAI_API_KEY in .env file.")
            return

        self.ai_analyze_btn.setEnabled(False)

        current_pid = PIDGains(
            kp=self.pid_kp.value(),
            ki=self.pid_ki.value(),
            kd=self.pid_kd.value(),
            kd_filter=self.pid_kd_filter.value(),
            ramp_erpms_s=self.pid_ramp.value(),
        )

        # Get language setting
        lang = "Korean" if self.lang_combo.currentText() == "한국어" else "English"
        advisor = LLMAdvisor(api_key=openai_key, model=model_name, language=lang)

        # Add control mode context
        mode_context = (
            f"Control Mode: SPEED CONTROL\n"
            f"Target: {self.target_spin.value()} RPM"
        )

        # Build request data and show what we're sending
        m, pid, _ = advisor.build_request_data(self._last_metrics, current_pid, mode_context)

        import json
        preview = (
            f"=== Sending to {model_name} ===\n\n"
            f"Mode: {self._control_mode.upper()} CONTROL\n\n"
            f"Current PID:\n{json.dumps(pid, indent=2)}\n\n"
            f"Metrics:\n{json.dumps(m, indent=2)}\n\n"
            f"--- Waiting for AI response... ---"
        )
        self.ai_text.setPlainText(preview)

        # Run in a simple thread to avoid blocking UI
        import threading
        import traceback
        import sys

        def _run():
            print("[Thread] Starting LLM request...", flush=True)
            sys.stdout.flush()
            try:
                result = advisor.analyze_and_recommend(self._last_metrics, current_pid, mode_context)
                print(f"[Thread] Got result: {result.summary[:50] if result.summary else 'empty'}...", flush=True)
                # Update UI from main thread via signal
                self._llm_result_signal.emit(result)
                print("[Thread] Signal emitted", flush=True)
            except Exception as e:
                err_msg = f"Error: {e}\n{traceback.format_exc()}"
                print(f"[Thread] Error: {err_msg}", flush=True)
                self._llm_error_signal.emit(err_msg)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        print(f"[Main] Thread started: {t.name}", flush=True)

    def _on_llm_error(self, err_msg: str):
        """Handle LLM analysis error."""
        self.ai_analyze_btn.setEnabled(True)
        self.ai_text.setPlainText(err_msg)

    def _on_llm_result(self, result: AnalysisResult):
        print(f"[UI] _on_llm_result called", flush=True)
        self.ai_analyze_btn.setEnabled(True)

        # Store raw data for later viewing
        self._last_raw_request = result.raw_request
        self._last_raw_response = result.raw_response

        # Display analysis
        text = result.summary + "\n\n"
        if result.recommendations:
            text += "Recommendations:\n"
            for r in result.recommendations:
                text += f"  - {r}\n"

        # Add raw data if checkbox is checked
        if self.show_raw_cb.isChecked():
            text += "\n" + "=" * 50 + "\n"
            text += "=== RAW REQUEST ===\n"
            text += result.raw_request + "\n\n"
            text += "=== RAW RESPONSE ===\n"
            text += result.raw_response + "\n"

        print(f"[UI] Setting text: {text[:100]}...", flush=True)
        self.ai_text.setPlainText(text)

        # Display suggested PID
        if result.suggested_pid:
            self._suggested_pid = result.suggested_pid
            self.sug_kp.setText(f"{result.suggested_pid.kp:.6f}")
            self.sug_ki.setText(f"{result.suggested_pid.ki:.6f}")
            self.sug_kd.setText(f"{result.suggested_pid.kd:.6f}")
            self.sug_kd_filter.setText(f"{result.suggested_pid.kd_filter:.6f}")
            self.sug_ramp.setText(f"{result.suggested_pid.ramp_erpms_s:.0f}")
            self.apply_btn.setEnabled(True)

    def copy_to_suggested(self):
        """Copy current gains spinbox values to Suggested section."""
        kp = self.pid_kp.value()
        ki = self.pid_ki.value()

        if self._control_mode == "current":
            # Current Control mode - only Kp, Ki
            self.sug_kp.setText(f"{kp:.6f}")
            self.sug_ki.setText(f"{ki:.2f}")
            # Store as suggested (Kp, Ki only matter for current mode)
            self._suggested_pid = PIDGains(kp=kp, ki=ki, kd=0, kd_filter=0, ramp_erpms_s=0)
            self.apply_btn.setEnabled(True)
            self.ai_text.setPlainText(
                f"Current Gains copied to Suggested:\n"
                f"  Current Kp={kp:.6f}\n"
                f"  Current Ki={ki:.2f}\n\n"
                f"Click 'Set Parameter' to send to VESC."
            )
        else:
            # Speed/Position mode - full PID
            kd = self.pid_kd.value()
            kd_filter = self.pid_kd_filter.value()
            ramp = self.pid_ramp.value()

            self.sug_kp.setText(f"{kp:.6f}")
            self.sug_ki.setText(f"{ki:.6f}")
            self.sug_kd.setText(f"{kd:.6f}")
            self.sug_kd_filter.setText(f"{kd_filter:.6f}")
            self.sug_ramp.setText(f"{ramp:.0f}")

            self._suggested_pid = PIDGains(kp=kp, ki=ki, kd=kd, kd_filter=kd_filter, ramp_erpms_s=ramp)
            self.apply_btn.setEnabled(True)

            self.ai_text.setPlainText(
                f"Current PID copied to Suggested:\n"
                f"  Kp={kp:.6f}\n"
                f"  Ki={ki:.6f}\n"
                f"  Kd={kd:.6f}\n"
                f"  Kd_filter={kd_filter:.6f}\n"
                f"  Ramp={ramp:.0f} eRPM/s\n\n"
                f"Click 'Set Parameter' to send to VESC."
            )

    def apply_suggested_pid(self):
        """Send suggested PID gains to VESC via COMM_SET_MCCONF."""
        if self._suggested_pid is None:
            QMessageBox.warning(self, "No PID", "No suggested PID values to apply.")
            return

        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to VESC first.")
            return

        # Auto-load MCCONF if not loaded (silent - don't update UI)
        if self._original_mcconf is None:
            self.ai_text.setPlainText("[Apply] Loading MCCONF template from VESC...")
            self._pending_apply_suggested = True
            self._silent_mcconf_load = True  # Don't overwrite user's values
            self.read_current_pid()
            return

        pid = self._suggested_pid

        # Handle Current Control mode (FOC current gains)
        if self._control_mode == "current":
            from ..protocol.commands import build_set_mcconf_with_foc_cc
            packet = build_set_mcconf_with_foc_cc(
                self._original_mcconf,
                pid.kp, pid.ki,  # Use Kp, Ki for FOC current gains
            )
            self._transport.send_packet(packet)

            # Update current gains display
            self.pid_kp.setValue(pid.kp)
            self.pid_ki.setValue(pid.ki)

            self.ai_text.append(
                f"\n[Applied FOC Current Gains via COMM_SET_MCCONF]\n"
                f"  Current Kp={pid.kp:.6f} Current Ki={pid.ki:.2f}"
            )
            print(f"[FOC] Sent SET_MCCONF with new FOC current gains", flush=True)

            # Add to history
            score = self._last_current_metrics.quality_score if self._last_current_metrics else None
            ripple = self._last_current_metrics.current_ripple_pct if self._last_current_metrics else None
            thd = self._last_current_metrics.current_thd if self._last_current_metrics else None
            self._add_history_entry(pid.kp, pid.ki, 0, 0, score, ripple, thd,
                                    metrics=self._last_current_metrics, ramp_erpms_s=0)
            return

        # Speed mode
        packet = build_set_mcconf_with_pid(
            self._original_mcconf,
            pid.kp, pid.ki, pid.kd, pid.kd_filter,
            position_mode=False,
            ramp_erpms_s=pid.ramp_erpms_s,
        )
        self._transport.send_packet(packet)

        # Update current PID display
        self.pid_kp.setValue(pid.kp)
        self.pid_ki.setValue(pid.ki)
        self.pid_kd.setValue(pid.kd)
        self.pid_kd_filter.setValue(pid.kd_filter)
        self.pid_ramp.setValue(pid.ramp_erpms_s)

        self.ai_text.append(
            f"\n[Applied Speed PID via COMM_SET_MCCONF]\n"
            f"  Kp={pid.kp:.6f} Ki={pid.ki:.6f} Kd={pid.kd:.6f} Kd_filter={pid.kd_filter:.6f} Ramp={pid.ramp_erpms_s:.0f}"
        )
        print(f"[PID] Sent SET_MCCONF with new {mode_str} PID values", flush=True)

        # Add to history with metrics for graph restoration
        score = self._last_metrics.quality_score if self._last_metrics else None
        ripple = self._last_metrics.rpm_ripple_pct if self._last_metrics else None
        ss_err = self._last_metrics.steady_state_error_pct if self._last_metrics else None
        self._add_history_entry(pid.kp, pid.ki, pid.kd, pid.kd_filter, score, ripple, ss_err,
                                metrics=self._last_metrics, ramp_erpms_s=pid.ramp_erpms_s)

    def start_auto_tune(self):
        """Start automatic PID tuning loop."""
        model_name = self.model_combo.currentText()
        openai_key = self._get_api_key()

        if not openai_key or openai_key == "your-api-key-here":
            QMessageBox.warning(self, "API Key", "Set OPENAI_API_KEY in .env file.")
            return

        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to VESC first.")
            return

        # Auto-load MCCONF if not loaded
        if self._original_mcconf is None:
            self.ai_text.setPlainText("[Auto-Tune] Loading MCCONF from VESC...")
            self._pending_auto_tune = True  # Flag to continue after MCCONF received
            self.read_current_pid()
            return

        self._start_auto_tune_internal()

    def _start_auto_tune_internal(self):
        """Internal method to start auto-tune after MCCONF is loaded."""
        # Current Control mode uses different tuner
        if self._control_mode == "current":
            self._start_current_auto_tune()
            return

        # First time: ask user confirmation for motor rotation
        if not self._motor_confirmed:
            target = self.target_spin.value()

            reply = QMessageBox.warning(
                self,
                "Motor Rotation Confirmation",
                f"Auto-Tune will now start and control the motor automatically.\n\n"
                f"Target: {target} RPM (Speed mode)\n"
                f"Iterations: 5\n"
                f"Duration per iteration: {self.duration_spin.value()}s\n\n"
                f"The motor will rotate during data collection.\n"
                f"Make sure the motor is safe to run!\n\n"
                f"Do you want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                self.ai_text.setPlainText("[Auto-Tune] Cancelled by user.")
                return

            self._motor_confirmed = True

        openai_key = self._get_api_key()
        model_name = self.model_combo.currentText()
        lang = "Korean" if self.lang_combo.currentText() == "한국어" else "English"
        advisor = LLMAdvisor(api_key=openai_key, model=model_name, language=lang)
        initial_pid = PIDGains(
            kp=self.pid_kp.value(),
            ki=self.pid_ki.value(),
            kd=self.pid_kd.value(),
            kd_filter=self.pid_kd_filter.value(),
            ramp_erpms_s=self.pid_ramp.value(),
        )

        self._tune_history.clear()
        self._auto_tuner = AutoTuner(
            transport=self._transport,
            advisor=advisor,
            target_rpm=float(self.target_spin.value()),
            initial_pid=initial_pid,
            original_mcconf=self._original_mcconf,
            position_mode=False,
            max_iterations=5,
            collect_duration_s=self.duration_spin.value(),
            sample_rate_hz=self._get_rate_hz(),
        )
        self._auto_tuner.status_update.connect(self._on_tune_status)
        self._auto_tuner.iteration_complete.connect(self._on_tune_iteration)
        self._auto_tuner.tuning_finished.connect(self._on_tune_finished)
        self._auto_tuner.data_collecting.connect(
            lambda p: self.collect_progress.setValue(int(p * 100))
        )
        self._auto_tuner.metrics_ready.connect(self._on_metrics_ready)
        self._auto_tuner.start()

        self.autotune_btn.setEnabled(False)
        self.stop_tune_btn.setEnabled(True)
        self.collect_btn.setEnabled(False)
        self.ai_text.setPlainText("[Auto-Tune] Started automatic PID tuning...\nMotor will start rotating...")

    def _start_current_auto_tune(self):
        """Start Current Control auto-tuning."""
        # First time: ask user confirmation
        if not self._motor_confirmed:
            target_current = self.target_current_spin.value()
            target_rpm = self.target_spin.value()

            reply = QMessageBox.warning(
                self,
                "Current Control Tuning Confirmation",
                f"Current Control Auto-Tune will start.\n\n"
                f"Phase 1: Apply {target_current:.1f}A step (transient test)\n"
                f"Phase 2: Run at {target_rpm} RPM (steady-state test)\n"
                f"Iterations: 5\n\n"
                f"The motor will rotate during testing.\n"
                f"Make sure the motor is safe to run!\n\n"
                f"Do you want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                self.ai_text.setPlainText("[Current Tune] Cancelled by user.")
                return

            self._motor_confirmed = True

        openai_key = self._get_api_key()
        model_name = self.model_combo.currentText()
        lang = "Korean" if self.lang_combo.currentText() == "한국어" else "English"
        advisor = CurrentControlAdvisor(api_key=openai_key, model=model_name, language=lang)

        # Use Kp/Ki spinboxes for FOC current gains
        initial_gains = FOCCurrentGains(
            kp=self.pid_kp.value(),
            ki=self.pid_ki.value(),
        )

        self._tune_history.clear()
        self._auto_tuner = CurrentAutoTuner(
            transport=self._transport,
            advisor=advisor,
            target_current=self.target_current_spin.value(),
            initial_gains=initial_gains,
            original_mcconf=self._original_mcconf,
            target_rpm_for_steady=float(self.target_spin.value()) if self.target_spin.value() > 0 else 3000.0,
            max_iterations=5,
            step_duration_s=1.0,
            steady_duration_s=self.duration_spin.value(),
            sample_rate_hz=self._get_rate_hz(),
        )
        self._auto_tuner.status_update.connect(self._on_tune_status)
        self._auto_tuner.iteration_complete.connect(self._on_current_tune_iteration)
        self._auto_tuner.tuning_finished.connect(self._on_tune_finished)
        self._auto_tuner.data_collecting.connect(
            lambda p: self.collect_progress.setValue(int(p * 100))
        )
        self._auto_tuner.metrics_ready.connect(self._on_current_metrics_ready)
        self._auto_tuner.start()

        self.autotune_btn.setEnabled(False)
        self.stop_tune_btn.setEnabled(True)
        self.collect_btn.setEnabled(False)
        self.ai_text.setPlainText("[Current Tune] Started FOC current control tuning...\nMotor will start rotating...")

    def _on_current_metrics_ready(self, metrics: CurrentMetrics):
        """Handle current metrics immediately after analysis."""
        self._last_current_metrics = metrics
        self._update_current_metrics_display(metrics)
        print(f"[UI] Current metrics ready: score={metrics.quality_score:.1f}", flush=True)

    def _on_current_tune_iteration(self, result: CurrentTuningIteration):
        """Handle Current Control tuning iteration complete."""
        self._tune_history.append(result)

        # Update metrics display (prefer steady metrics for display)
        if result.steady_metrics:
            self._last_current_metrics = result.steady_metrics
            self._update_current_metrics_display(result.steady_metrics)

        # Update AI text
        if result.analysis:
            self.ai_text.append(f"\n--- Iteration {result.iteration} ---")
            self.ai_text.append(result.analysis.summary)

            # Show raw data if checkbox is checked
            if self.show_raw_cb.isChecked():
                self.ai_text.append("\n[RAW REQUEST]\n" + result.analysis.raw_request)
                self.ai_text.append("\n[RAW RESPONSE]\n" + result.analysis.raw_response)

            # Update Suggested gains display
            if result.analysis.suggested_gains:
                sg = result.analysis.suggested_gains
                self.sug_kp.setText(f"{sg.kp:.6f}")
                self.sug_ki.setText(f"{sg.ki:.6f}")
                self.sug_kd.setText("--")
                self.sug_kd_filter.setText("--")
                self.sug_ramp.setText("--")
                # Update Current gains spinboxes
                self.pid_kp.setValue(sg.kp)
                self.pid_ki.setValue(sg.ki)

        # Add to history table
        self._add_current_history_entry(result)

    def stop_auto_tune(self):
        if self._auto_tuner:
            self._auto_tuner.stop()
            # Immediately stop motor as safety measure
            from ..protocol.commands import build_set_current
            if self._transport.is_connected():
                self._transport.send_packet(build_set_current(0.0))
                self.ai_text.append("\n[Auto-Tune] Stop requested - motor stopped.")

    def _on_tune_status(self, msg: str):
        self.ai_text.append(msg)

    def _on_metrics_ready(self, metrics: MotorMetrics):
        """Handle metrics immediately after analysis (before LLM consultation)."""
        self._last_metrics = metrics
        self._update_metrics_display(metrics)
        print(f"[UI] Metrics ready: score={metrics.quality_score:.1f} (immediate update)", flush=True)

    def _on_tune_iteration(self, result: TuningIteration):
        self._tune_history.append(result)

        # Update metrics display
        if result.metrics:
            self._last_metrics = result.metrics
            self._update_metrics_display(result.metrics)
            print(f"[UI] Updated metrics display: score={result.metrics.quality_score:.1f}", flush=True)
        else:
            print(f"[UI] No metrics in result for iteration {result.iteration}", flush=True)

        # Update AI text
        if result.analysis:
            self.ai_text.append(f"\n--- Iteration {result.iteration} ---")
            self.ai_text.append(result.analysis.summary)

            # Show raw data if checkbox is checked
            if self.show_raw_cb.isChecked():
                self.ai_text.append("\n[RAW REQUEST]\n" + result.analysis.raw_request)
                self.ai_text.append("\n[RAW RESPONSE]\n" + result.analysis.raw_response)

            # Store raw data
            self._last_raw_request = result.analysis.raw_request
            self._last_raw_response = result.analysis.raw_response

            # Add to history table with the PID that was USED (result.pid), not the suggested PID
            # The score was achieved by result.pid, not the suggested PID
            used_pid = result.pid
            score = result.metrics.quality_score if result.metrics else None
            ripple = result.metrics.rpm_ripple_pct if result.metrics else None
            ss_err = result.metrics.steady_state_error_pct if result.metrics else None
            self._add_history_entry(used_pid.kp, used_pid.ki, used_pid.kd, used_pid.kd_filter,
                                    score, ripple, ss_err, metrics=result.metrics,
                                    ramp_erpms_s=used_pid.ramp_erpms_s)

            # Update Suggested PID display (this is for the NEXT iteration)
            if result.analysis.suggested_pid:
                sp = result.analysis.suggested_pid
                self.sug_kp.setText(f"{sp.kp:.6f}")
                self.sug_ki.setText(f"{sp.ki:.6f}")
                self.sug_kd.setText(f"{sp.kd:.6f}")
                self.sug_kd_filter.setText(f"{sp.kd_filter:.6f}")
                self.sug_ramp.setText(f"{sp.ramp_erpms_s:.0f}")
                # Update Current PID spinboxes to show what will be applied next
                self.pid_kp.setValue(sp.kp)
                self.pid_ki.setValue(sp.ki)
                self.pid_kd.setValue(sp.kd)
                self.pid_kd_filter.setValue(sp.kd_filter)
                self.pid_ramp.setValue(sp.ramp_erpms_s)

    def _on_tune_finished(self, reason: str):
        self.ai_text.append(f"\n[Auto-Tune Finished] {reason}")

        # Ensure motor is stopped
        from ..protocol.commands import build_set_current
        if self._transport.is_connected():
            self._transport.send_packet(build_set_current(0.0))
            self.ai_text.append("[Motor stopped]")

        self.autotune_btn.setEnabled(True)
        self.stop_tune_btn.setEnabled(False)
        self.collect_btn.setEnabled(True)
        self._auto_tuner = None
        self._motor_confirmed = False  # Reset for next session

    def cleanup(self):
        if self._auto_tuner:
            self._auto_tuner.stop()
            self._auto_tuner.wait(2000)
