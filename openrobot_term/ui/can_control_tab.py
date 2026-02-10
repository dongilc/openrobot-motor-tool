"""
CAN Control panel — RMD motor control (torque/speed/position/multiturn).
Designed for right-side dock (narrow vertical layout with scroll).
Mirrors MotorControlTab pattern.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QSlider, QComboBox,
    QMessageBox, QGridLayout, QScrollArea, QFrame, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

import math
import time
import threading

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import (
    build_torque_closed_loop, build_speed_closed_loop,
    build_position_closed_loop_1, build_set_multiturn_position,
    build_motor_off, build_motor_stop, build_motor_start,
)

class CanControlTab(QWidget):
    torque_cmd_sent = pyqtSignal(float)  # emitted on each torque command send
    pos_cmd_sent = pyqtSignal(float)     # emitted on each position/multiturn command send

    def __init__(self, can_transport: PcanTransport):
        super().__init__()
        self._transport = can_transport
        self._active_mode = None
        self._send_timer = QTimer(self)
        self._send_timer.timeout.connect(self._send_command)

        # LPF state
        self._lpf_out = 0.0
        self._lpf_thread = None
        self._lpf_running = False

        self._build_ui()
        self._update_alpha_display()

        # Wire signals — live feedback from both polling (0x9C) and control responses (0xA1-A4)
        self._transport.status_received.connect(self._on_status_received)
        self._transport.cmd_status_received.connect(self._on_status_received)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # Scroll area for compact dock
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)
        scroll.setWidget(container)
        outer.addWidget(scroll)

        # ── Send rate row ──
        top_row = QHBoxLayout()
        layout.addLayout(top_row)

        self.periodic_chk = QCheckBox("Periodic")
        self.periodic_chk.setToolTip("Send commands repeatedly at Send Rate")
        self.periodic_chk.toggled.connect(self._on_periodic_toggled)
        top_row.addWidget(self.periodic_chk)

        top_row.addWidget(QLabel("Send Rate:"))
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["10 Hz", "20 Hz", "50 Hz", "100 Hz", "200 Hz", "500 Hz"])
        self.rate_combo.setCurrentText("50 Hz")
        self.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        top_row.addWidget(self.rate_combo)

        top_row.addStretch()

        # ── LPF row ──
        lpf_row = QHBoxLayout()
        layout.addLayout(lpf_row)

        self.lpf_chk = QCheckBox("LPF")
        self.lpf_chk.setToolTip("Low-pass filter on slider commands (implies periodic)")
        self.lpf_chk.toggled.connect(self._on_lpf_toggled)
        lpf_row.addWidget(self.lpf_chk)

        lpf_row.addWidget(QLabel("Cutoff:"))
        self.lpf_cutoff_spin = QDoubleSpinBox()
        self.lpf_cutoff_spin.setRange(0.1, 50.0)
        self.lpf_cutoff_spin.setSingleStep(0.1)
        self.lpf_cutoff_spin.setValue(1.0)
        self.lpf_cutoff_spin.setDecimals(1)
        self.lpf_cutoff_spin.setSuffix(" Hz")
        self.lpf_cutoff_spin.setMaximumWidth(110)
        self.lpf_cutoff_spin.valueChanged.connect(self._update_alpha_display)
        lpf_row.addWidget(self.lpf_cutoff_spin)

        self.lpf_alpha_label = QLabel()
        self.lpf_alpha_label.setStyleSheet("font-family: monospace; color: #aaa;")
        lpf_row.addWidget(self.lpf_alpha_label)

        lpf_row.addStretch()

        # ── Motor OFF / STOP / START ──
        motor_row = QHBoxLayout()
        layout.addLayout(motor_row)

        off_btn = QPushButton("Motor OFF")
        off_btn.setMinimumHeight(28)
        off_btn.setStyleSheet(
            "QPushButton { background-color: #884400; color: white; font-weight: bold; }"
            "QPushButton:pressed { background-color: #442200; border: 2px solid #ffaa00; }"
            "QPushButton:hover { background-color: #995500; }"
        )
        off_btn.clicked.connect(lambda: self._motor_cmd(build_motor_off()))
        motor_row.addWidget(off_btn)

        stop_motor_btn = QPushButton("Motor STOP")
        stop_motor_btn.setMinimumHeight(28)
        stop_motor_btn.setStyleSheet(
            "QPushButton:pressed { background-color: #222; border: 2px solid #aaa; }"
        )
        stop_motor_btn.clicked.connect(lambda: self._motor_cmd(build_motor_stop()))
        motor_row.addWidget(stop_motor_btn)

        start_motor_btn = QPushButton("Motor START")
        start_motor_btn.setMinimumHeight(28)
        start_motor_btn.setStyleSheet(
            "QPushButton { background-color: #338833; color: white; font-weight: bold; }"
            "QPushButton:pressed { background-color: #114411; border: 2px solid #66ff66; }"
            "QPushButton:hover { background-color: #449944; }"
        )
        start_motor_btn.clicked.connect(lambda: self._motor_cmd(build_motor_start()))
        motor_row.addWidget(start_motor_btn)

        # ── Torque Control ──
        torque_group = QGroupBox("Torque (0xA1)")
        tl = QVBoxLayout(torque_group)
        tl.setSpacing(4)

        tr1 = QHBoxLayout()
        self.torque_spin = QDoubleSpinBox()
        self.torque_spin.setRange(-33.0, 33.0)
        self.torque_spin.setSingleStep(0.1)
        self.torque_spin.setDecimals(2)
        self.torque_spin.setMinimumHeight(32)
        self.torque_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        tr1.addWidget(self.torque_spin, stretch=1)
        tr1.addWidget(QLabel("A"))
        tl.addLayout(tr1)

        self.torque_slider = QSlider(Qt.Orientation.Horizontal)
        self.torque_slider.setRange(-3300, 3300)
        self.torque_slider.valueChanged.connect(lambda v: self.torque_spin.setValue(v / 100.0))
        self.torque_spin.valueChanged.connect(lambda v: self.torque_slider.setValue(int(v * 100)))
        tl.addWidget(self.torque_slider)

        # Auto-send on value change when torque mode is active
        self.torque_spin.valueChanged.connect(self._on_torque_value_changed)

        self.torque_start_btn = QPushButton("Start Torque")
        self.torque_start_btn.setMinimumHeight(30)
        self.torque_start_btn.setStyleSheet(
            "QPushButton { background-color: #338833; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.torque_start_btn.clicked.connect(lambda: self._start("torque"))
        tl.addWidget(self.torque_start_btn)
        layout.addWidget(torque_group)

        # ── Speed Control ──
        speed_group = QGroupBox("Speed (0xA2)")
        sl = QVBoxLayout(speed_group)
        sl.setSpacing(4)

        # Mode selector row
        smode_row = QHBoxLayout()
        smode_row.addWidget(QLabel("Mode:"))
        self.speed_mode_combo = QComboBox()
        self.speed_mode_combo.addItems(["DPS", "eRPM"])
        self.speed_mode_combo.setToolTip("DPS: OpenRobot custom loop, eRPM: VESC built-in PID")
        self.speed_mode_combo.currentTextChanged.connect(self._on_speed_mode_changed)
        smode_row.addWidget(self.speed_mode_combo)
        smode_row.addStretch()
        sl.addLayout(smode_row)

        sr1 = QHBoxLayout()
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(-25000.0, 25000.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setDecimals(1)
        self.speed_spin.setMinimumHeight(32)
        self.speed_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        sr1.addWidget(self.speed_spin, stretch=1)
        self.speed_unit_label = QLabel("dps")
        sr1.addWidget(self.speed_unit_label)
        sl.addLayout(sr1)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(-25000, 25000)
        self.speed_slider.valueChanged.connect(lambda v: self.speed_spin.setValue(float(v)))
        self.speed_spin.valueChanged.connect(lambda v: self.speed_slider.setValue(int(v)))
        sl.addWidget(self.speed_slider)

        # Auto-send on value change when speed mode is active
        self.speed_spin.valueChanged.connect(self._on_speed_value_changed)

        self.speed_start_btn = QPushButton("Start Speed")
        self.speed_start_btn.setMinimumHeight(30)
        self.speed_start_btn.setStyleSheet(
            "QPushButton { background-color: #338833; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.speed_start_btn.clicked.connect(lambda: self._start("speed"))
        sl.addWidget(self.speed_start_btn)
        layout.addWidget(speed_group)

        # ── Position Control ──
        pos_group = QGroupBox("Position (0xA3)")
        pl = QVBoxLayout(pos_group)
        pl.setSpacing(4)

        pr1 = QHBoxLayout()
        self.pos_spin = QDoubleSpinBox()
        self.pos_spin.setRange(-3600.0, 3600.0)
        self.pos_spin.setSingleStep(1.0)
        self.pos_spin.setDecimals(2)
        self.pos_spin.setMinimumHeight(32)
        self.pos_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        pr1.addWidget(self.pos_spin, stretch=1)
        pr1.addWidget(QLabel("deg"))
        pl.addLayout(pr1)

        self.pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.pos_slider.setRange(-360000, 360000)
        self.pos_slider.valueChanged.connect(lambda v: self.pos_spin.setValue(v / 100.0))
        self.pos_spin.valueChanged.connect(lambda v: self.pos_slider.setValue(int(v * 100)))
        pl.addWidget(self.pos_slider)

        # Auto-send on value change when position mode is active
        self.pos_spin.valueChanged.connect(self._on_pos_value_changed)

        self.pos_start_btn = QPushButton("Start Position")
        self.pos_start_btn.setMinimumHeight(30)
        self.pos_start_btn.setStyleSheet(
            "QPushButton { background-color: #338833; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.pos_start_btn.clicked.connect(lambda: self._start("position"))
        pl.addWidget(self.pos_start_btn)
        layout.addWidget(pos_group)

        # ── Multiturn Position ──
        mt_group = QGroupBox("Multiturn Position (0xA4)")
        ml = QVBoxLayout(mt_group)
        ml.setSpacing(4)

        mr1 = QHBoxLayout()
        mr1.addWidget(QLabel("DPS limit:"))
        self.mt_dps_spin = QSpinBox()
        self.mt_dps_spin.setRange(1, 25000)
        self.mt_dps_spin.setValue(1000)
        mr1.addWidget(self.mt_dps_spin, stretch=1)
        ml.addLayout(mr1)

        mr2 = QHBoxLayout()
        self.mt_pos_spin = QDoubleSpinBox()
        self.mt_pos_spin.setRange(-3600.0, 3600.0)
        self.mt_pos_spin.setSingleStep(1.0)
        self.mt_pos_spin.setDecimals(2)
        self.mt_pos_spin.setMinimumHeight(32)
        self.mt_pos_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        mr2.addWidget(self.mt_pos_spin, stretch=1)
        mr2.addWidget(QLabel("deg"))
        ml.addLayout(mr2)

        self.mt_pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.mt_pos_slider.setRange(-360000, 360000)
        self.mt_pos_slider.valueChanged.connect(lambda v: self.mt_pos_spin.setValue(v / 100.0))
        self.mt_pos_spin.valueChanged.connect(lambda v: self.mt_pos_slider.setValue(int(v * 100)))
        ml.addWidget(self.mt_pos_slider)

        # Auto-send on value change when multiturn mode is active
        self.mt_pos_spin.valueChanged.connect(self._on_mt_value_changed)

        self.mt_start_btn = QPushButton("Start Multiturn")
        self.mt_start_btn.setMinimumHeight(30)
        self.mt_start_btn.setStyleSheet(
            "QPushButton { background-color: #338833; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.mt_start_btn.clicked.connect(lambda: self._start("multiturn"))
        ml.addWidget(self.mt_start_btn)
        layout.addWidget(mt_group)

        # ── Live Feedback ──
        fb_group = QGroupBox("Live Feedback")
        fb_layout = QGridLayout(fb_group)
        fb_layout.setSpacing(2)

        self.fb_labels = {}
        items = [
            ("Temp:", "temp"), ("Torque:", "torque"),
            ("Speed:", "speed"), ("Pos:", "position"),
        ]
        for i, (label, key) in enumerate(items):
            row, col = i // 2, (i % 2) * 2
            lb = QLabel(label)
            lb.setStyleSheet("font-size: 11px; font-weight: bold;")
            fb_layout.addWidget(lb, row, col)
            val = QLabel("--")
            val.setStyleSheet("font-family: monospace; font-size: 11px;")
            fb_layout.addWidget(val, row, col + 1)
            self.fb_labels[key] = val

        layout.addWidget(fb_group)

        layout.addStretch()

    # ── Speed mode helpers ──

    def _get_speed_mode(self) -> int:
        """Return 0 for DPS, 1 for eRPM."""
        return 1 if self.speed_mode_combo.currentText() == "eRPM" else 0

    def _on_speed_mode_changed(self, text: str):
        """Switch speed control mode between DPS and eRPM."""
        if text == "eRPM":
            self.speed_spin.setDecimals(0)
            self.speed_spin.setSingleStep(100)
            self.speed_spin.setRange(-100000.0, 100000.0)
            self.speed_slider.setRange(-100000, 100000)
            self.speed_unit_label.setText("eRPM")
        else:
            self.speed_spin.setDecimals(1)
            self.speed_spin.setSingleStep(0.1)
            self.speed_spin.setRange(-25000.0, 25000.0)
            self.speed_slider.setRange(-25000, 25000)
            self.speed_unit_label.setText("dps")
        self.speed_spin.setValue(0)

    # ── Helpers ──

    def _get_rate_hz(self) -> int:
        return int(self.rate_combo.currentText().replace(" Hz", ""))

    def _get_target(self) -> float:
        """Return the current target value for the active mode."""
        if self._active_mode == "torque":
            return self.torque_spin.value()
        elif self._active_mode == "speed":
            return float(self.speed_spin.value())
        elif self._active_mode == "position":
            return self.pos_spin.value()
        elif self._active_mode == "multiturn":
            return self.mt_pos_spin.value()
        return 0.0

    def _is_periodic(self) -> bool:
        """Whether periodic send is active (explicit or implied by LPF)."""
        return self.periodic_chk.isChecked() or self.lpf_chk.isChecked()

    # ── Control actions ──

    def _start(self, mode: str):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Open PCAN connection first.")
            return

        # Stop previous mode first (auto-cancel)
        self._stop_loop()

        self._active_mode = mode

        # Disable all start buttons while active
        self.torque_start_btn.setEnabled(False)
        self.speed_start_btn.setEnabled(False)
        self.pos_start_btn.setEnabled(False)
        self.mt_start_btn.setEnabled(False)
        self.rate_combo.setEnabled(False)

        # Start appropriate sender and send initial value
        self._apply_send_mode()
        self._send_current_value()

    def _stop_loop(self):
        """Stop the active control loop (timer + LPF thread, no CAN command)."""
        self._send_timer.stop()
        self._stop_lpf_thread()
        self._active_mode = None
        self.torque_start_btn.setEnabled(True)
        self.speed_start_btn.setEnabled(True)
        self.pos_start_btn.setEnabled(True)
        self.mt_start_btn.setEnabled(True)
        self.rate_combo.setEnabled(True)

    def _motor_cmd(self, data: bytes):
        """Send a motor OFF/STOP/START command, stopping any active control loop first."""
        self._stop_loop()
        self.torque_cmd_sent.emit(0.0)
        self.pos_cmd_sent.emit(float('nan'))
        self._send_once(data)

    def _send_current_value(self):
        """Send the current target value once for the active mode."""
        if not self._transport.is_connected():
            return
        mode = self._active_mode
        if mode == "torque":
            val = self.torque_spin.value()
            self._transport.send_frame(build_torque_closed_loop(val))
            self.torque_cmd_sent.emit(val)
        elif mode == "speed":
            val = self.speed_spin.value()
            self._transport.send_frame(build_speed_closed_loop(val, self._get_speed_mode()))
        elif mode == "position":
            val = self.pos_spin.value()
            self._transport.send_frame(build_position_closed_loop_1(val))
            self.pos_cmd_sent.emit(val)
        elif mode == "multiturn":
            val = self.mt_pos_spin.value()
            self._transport.send_frame(
                build_set_multiturn_position(self.mt_dps_spin.value(), val)
            )
            self.pos_cmd_sent.emit(val)

    # ── Send mode management (single source of truth) ──

    def _apply_send_mode(self):
        """Stop all senders, then restart the correct one for current state.
        Called on: start, periodic toggle, LPF toggle, rate change."""
        # Always stop both first
        self._send_timer.stop()
        self._stop_lpf_thread()

        if self._active_mode is None:
            return

        if self.lpf_chk.isChecked():
            # LPF thread (implies periodic)
            self._lpf_out = self._get_target()
            self._start_lpf_thread()
        elif self.periodic_chk.isChecked():
            # QTimer periodic
            self._send_timer.start(int(1000 / self._get_rate_hz()))
        # else: single-shot via valueChanged handlers

    def _on_periodic_toggled(self, _checked: bool):
        self._apply_send_mode()

    def _on_lpf_toggled(self, _checked: bool):
        self._apply_send_mode()

    def _on_rate_changed(self, text: str):
        self._apply_send_mode()
        self._update_alpha_display()

    def _compute_alpha(self) -> float:
        """Compute LPF alpha from cutoff frequency and send rate."""
        fc = self.lpf_cutoff_spin.value()
        fs = self._get_rate_hz()
        return 1.0 - math.exp(-2.0 * math.pi * fc / fs)

    def _update_alpha_display(self):
        alpha = self._compute_alpha()
        self.lpf_alpha_label.setText(f"(α={alpha:.3f})")

    # ── LPF thread (precise timing for smooth commands) ──

    def _start_lpf_thread(self):
        self._stop_lpf_thread()
        self._lpf_running = True
        self._lpf_thread = threading.Thread(
            target=self._lpf_loop, daemon=True, name="LPFSender"
        )
        self._lpf_thread.start()

    def _stop_lpf_thread(self):
        self._lpf_running = False
        if self._lpf_thread and self._lpf_thread.is_alive():
            self._lpf_thread.join(timeout=1.0)
        self._lpf_thread = None

    def _lpf_loop(self):
        """Background thread: send LPF-filtered commands with precise timing."""
        perf = time.perf_counter
        sleep = time.sleep
        t_next = perf()

        while self._lpf_running and self._transport.is_connected():
            interval = 1.0 / self._get_rate_hz()
            alpha = self._compute_alpha()
            mode = self._active_mode

            if mode == "torque":
                target = self.torque_spin.value()
                self._lpf_out += alpha * (target - self._lpf_out)
                self._transport.send_frame(build_torque_closed_loop(self._lpf_out))
                self.torque_cmd_sent.emit(self._lpf_out)
            elif mode == "speed":
                target = float(self.speed_spin.value())
                self._lpf_out += alpha * (target - self._lpf_out)
                self._transport.send_frame(build_speed_closed_loop(self._lpf_out, self._get_speed_mode()))
            elif mode == "position":
                target = self.pos_spin.value()
                self._lpf_out += alpha * (target - self._lpf_out)
                self._transport.send_frame(build_position_closed_loop_1(self._lpf_out))
                self.pos_cmd_sent.emit(self._lpf_out)
            elif mode == "multiturn":
                target = self.mt_pos_spin.value()
                self._lpf_out += alpha * (target - self._lpf_out)
                self._transport.send_frame(
                    build_set_multiturn_position(self.mt_dps_spin.value(), self._lpf_out)
                )
                self.pos_cmd_sent.emit(self._lpf_out)
            else:
                break

            t_next += interval
            remaining = t_next - perf()

            if remaining <= 0:
                t_next = perf()
            else:
                # Sleep most of the interval, then yield-spin the last bit
                if remaining > 0.002:
                    sleep(remaining - 0.002)
                while perf() < t_next and self._lpf_running:
                    sleep(0.0002)  # 200µs yield — prevents CPU 100%

    # ── Timer-based periodic send (non-LPF) ──

    def _send_command(self):
        """QTimer callback for periodic non-LPF send."""
        if not self._transport.is_connected():
            self._stop_loop()
            return
        self._send_current_value()

    # ── Value-changed handlers (single-shot send when not periodic) ──

    def _on_torque_value_changed(self):
        if self._active_mode == "torque" and not self._is_periodic():
            val = self.torque_spin.value()
            self._transport.send_frame(build_torque_closed_loop(val))
            self.torque_cmd_sent.emit(val)

    def _on_speed_value_changed(self):
        if self._active_mode == "speed" and not self._is_periodic():
            self._transport.send_frame(build_speed_closed_loop(self.speed_spin.value(), self._get_speed_mode()))

    def _on_pos_value_changed(self):
        if self._active_mode == "position" and not self._is_periodic():
            val = self.pos_spin.value()
            self._transport.send_frame(build_position_closed_loop_1(val))
            self.pos_cmd_sent.emit(val)

    def _on_mt_value_changed(self):
        if self._active_mode == "multiturn" and not self._is_periodic():
            val = self.mt_pos_spin.value()
            self._transport.send_frame(
                build_set_multiturn_position(self.mt_dps_spin.value(), val)
            )
            self.pos_cmd_sent.emit(val)

    def _send_once(self, data: bytes):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Open PCAN connection first.")
            return
        self._transport.send_frame(data)

    # ── Live Feedback ──

    def _on_status_received(self, status):
        self.fb_labels["temp"].setText(f"{status.motor_temp:.0f} C")
        self.fb_labels["torque"].setText(f"{status.torque_curr:.3f} A")
        self.fb_labels["speed"].setText(f"{status.speed_dps} dps")
        self.fb_labels["position"].setText(f"{status.enc_pos:.2f} deg")

    def cleanup(self):
        self._stop_loop()
