"""
Motor Control panel: Duty / Current / Speed / Position — all visible simultaneously.
One active at a time; Start locks selection, Stop releases.
"""

import math

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QSlider, QComboBox,
    QMessageBox, QFrame, QScrollArea, QSizePolicy, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer


from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import (
    VescValues, build_set_duty, build_set_current,
    build_set_rpm, build_set_pos, build_get_values,
)


class _ControlCard(QGroupBox):
    """A single control mode card with spinbox, slider, and Start button."""

    def __init__(self, title: str, unit: str,
                 min_val: float, max_val: float, step: float, decimals: int,
                 slider_scale: int, slider_max_override: float = None):
        super().__init__(title)
        self._slider_scale = slider_scale
        self._updating = False

        # Slider range can be smaller than spin range (e.g. slider ±10A, spin ±100A)
        slider_min = min_val
        slider_max = max_val
        if slider_max_override is not None:
            slider_max = abs(slider_max_override)
            slider_min = -slider_max if min_val < 0 else 0.0
        self._slider_min = slider_min
        self._slider_max = slider_max

        # Preset 100% = slider max (not spin max)
        self._preset_max = slider_max
        self._preset_min = slider_min

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # Spin + unit
        row = QHBoxLayout()
        self.spin = QDoubleSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setValue(0 if min_val <= 0 else min_val)
        self.spin.setMinimumHeight(32)
        self.spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.spin.valueChanged.connect(self._spin_to_slider)
        row.addWidget(self.spin, stretch=1)

        unit_lbl = QLabel(unit)
        unit_lbl.setStyleSheet("font-size: 12px; color: #aaa; padding-left: 4px;")
        row.addWidget(unit_lbl)
        layout.addLayout(row)

        # Slider (uses slider range, not spin range)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(slider_min * slider_scale), int(slider_max * slider_scale))
        self.slider.setValue(0 if min_val <= 0 else int(min_val * slider_scale))
        self.slider.valueChanged.connect(self._slider_to_spin)
        layout.addWidget(self.slider)

        # Preset buttons — incremental (0 is absolute)
        preset_row = QHBoxLayout()
        preset_row.setSpacing(2)
        if min_val < 0:
            presets = [("-50%", -0.5), ("-10%", -0.1), ("-1%", -0.01),
                       ("0", None),
                       ("+1%", 0.01), ("+10%", 0.1), ("+50%", 0.5)]
        else:
            presets = [("0", None),
                       ("+1%", 0.01), ("+5%", 0.05), ("+10%", 0.1),
                       ("+25%", 0.25), ("+50%", 0.5)]
        for pct_label, pct in presets:
            btn = QPushButton(pct_label)
            btn.setFixedHeight(22)
            btn.setStyleSheet("font-size: 10px; padding: 1px 3px;")
            if pct is None:
                # "0" button — absolute zero
                btn.clicked.connect(lambda _: self.spin.setValue(0))
            else:
                # Incremental: add delta = preset_max * pct to current value
                delta = self._preset_max * pct
                btn.clicked.connect(lambda _, d=delta: self.spin.setValue(
                    max(min_val, min(max_val, self.spin.value() + d))))
            preset_row.addWidget(btn)
        layout.addLayout(preset_row)

        # Start button
        self.start_btn = QPushButton(f"Start {title}")
        self.start_btn.setMinimumHeight(30)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #338833; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        layout.addWidget(self.start_btn)

    def _spin_to_slider(self, val):
        if self._updating:
            return
        self._updating = True
        # Clamp to slider range
        clamped = max(self._slider_min, min(self._slider_max, val))
        self.slider.setValue(int(clamped * self._slider_scale))
        self._updating = False

    def _slider_to_spin(self, val):
        if self._updating:
            return
        self._updating = True
        self.spin.setValue(val / self._slider_scale)
        self._updating = False

    def set_active_style(self, active: bool):
        if active:
            self.setStyleSheet(
                "QGroupBox { border: 2px solid #44bb44; border-radius: 6px; "
                "margin-top: 6px; padding-top: 14px; }"
                "QGroupBox::title { color: #44bb44; font-weight: bold; }"
            )
        else:
            self.setStyleSheet("")

    def set_controls_enabled(self, enabled: bool):
        self.spin.setEnabled(enabled)
        self.slider.setEnabled(enabled)


class MotorControlTab(QWidget):
    def __init__(self, transport: SerialTransport):
        super().__init__()
        self._transport = transport
        self._active_mode = None
        self._send_timer = QTimer(self)
        self._send_timer.timeout.connect(self._send_command)

        # Low-pass filter state
        self._lpf_enabled = False
        self._lpf_cutoff_hz = 2.0  # default cutoff frequency
        self._smoothed_value = 0.0  # current filtered output

        self._build_ui()

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

        # Send rate + Stop at top
        top_row = QHBoxLayout()
        layout.addLayout(top_row)

        top_row.addWidget(QLabel("Send Rate:"))
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["10 Hz", "20 Hz", "50 Hz", "100 Hz"])
        self.rate_combo.setCurrentText("50 Hz")
        self.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        top_row.addWidget(self.rate_combo)

        top_row.addStretch()

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setMinimumHeight(30)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #cc3333; color: white; "
            "font-weight: bold; font-size: 13px; padding: 4px 20px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.stop_btn.clicked.connect(self.stop_control)
        self.stop_btn.setEnabled(False)
        top_row.addWidget(self.stop_btn)

        # LPF row
        lpf_row = QHBoxLayout()
        layout.addLayout(lpf_row)

        self.lpf_check = QCheckBox("LPF")
        self.lpf_check.setToolTip("Enable low-pass filter smoothing on command output")
        self.lpf_check.setChecked(False)
        self.lpf_check.toggled.connect(self._on_lpf_toggled)
        lpf_row.addWidget(self.lpf_check)

        lpf_row.addWidget(QLabel("Cutoff:"))
        self.lpf_cutoff_spin = QDoubleSpinBox()
        self.lpf_cutoff_spin.setRange(0.1, 20.0)
        self.lpf_cutoff_spin.setValue(2.0)
        self.lpf_cutoff_spin.setSingleStep(0.5)
        self.lpf_cutoff_spin.setDecimals(1)
        self.lpf_cutoff_spin.setSuffix(" Hz")
        self.lpf_cutoff_spin.setFixedWidth(110)
        self.lpf_cutoff_spin.setEnabled(False)
        self.lpf_cutoff_spin.valueChanged.connect(self._on_lpf_cutoff_changed)
        lpf_row.addWidget(self.lpf_cutoff_spin)

        self.lpf_status_label = QLabel("")
        self.lpf_status_label.setStyleSheet("font-size: 10px; color: #888;")
        lpf_row.addWidget(self.lpf_status_label)
        lpf_row.addStretch()

        # Four control cards
        self.duty_card = _ControlCard(
            "Duty", "(-1.0 ~ 1.0)", -1.0, 1.0, 0.01, 3, 1000
        )
        self.current_card = _ControlCard(
            "Current", "(A)", -100.0, 100.0, 0.1, 2, 100,
            slider_max_override=10.0
        )
        self.speed_card = _ControlCard(
            "Speed", "(eRPM)", -100000, 100000, 100, 0, 1,
            slider_max_override=50000
        )
        self.position_card = _ControlCard(
            "Position", "(deg)", 0.0, 360.0, 1.0, 2, 100
        )

        self._cards = {
            "duty": self.duty_card,
            "current": self.current_card,
            "speed": self.speed_card,
            "position": self.position_card,
        }

        # Connect start buttons
        self.duty_card.start_btn.clicked.connect(lambda: self._start("duty"))
        self.current_card.start_btn.clicked.connect(lambda: self._start("current"))
        self.speed_card.start_btn.clicked.connect(lambda: self._start("speed"))
        self.position_card.start_btn.clicked.connect(lambda: self._start("position"))

        for card in self._cards.values():
            layout.addWidget(card)

        # Live Feedback
        fb_group = QGroupBox("Live Feedback")
        fb_layout = QGridLayout(fb_group)
        fb_layout.setSpacing(2)

        self.fb_labels = {}
        items = [
            ("Duty:", "duty"), ("I_motor:", "i_motor"),
            ("I_input:", "i_input"), ("RPM:", "rpm"),
            ("Position:", "position"), ("V_in:", "v_in"),
            ("Power:", "power"), ("T_mos:", "temp_mos"),
            ("T_mot:", "temp_mot"), ("Fault:", "fault"),
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

        # Warning
        warn = QLabel("Motor will spin when Start is pressed!")
        warn.setStyleSheet(
            "color: #ff8800; font-weight: bold; font-size: 10px; padding: 4px; "
            "border: 1px solid #ff8800; border-radius: 3px;"
        )
        warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(warn)

        layout.addStretch()

    def _start(self, mode: str):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to VESC first.")
            return

        self._active_mode = mode
        # Initialize smoothed value to current spinbox value
        self._smoothed_value = self._cards[mode].spin.value()

        # Disable all start buttons, enable only the active card's controls
        for key, card in self._cards.items():
            card.start_btn.setEnabled(False)
            if key == mode:
                card.set_active_style(True)
                card.set_controls_enabled(True)
            else:
                card.set_active_style(False)
                card.set_controls_enabled(False)

        self.stop_btn.setEnabled(True)
        self.rate_combo.setEnabled(False)

        rate_hz = int(self.rate_combo.currentText().replace(" Hz", ""))
        self._send_timer.start(int(1000 / rate_hz))

    def stop_control(self):
        self._send_timer.stop()
        self._active_mode = None

        # Release motor
        if self._transport.is_connected():
            self._transport.send_packet(build_set_current(0.0))

        # Re-enable all
        for card in self._cards.values():
            card.start_btn.setEnabled(True)
            card.set_active_style(False)
            card.set_controls_enabled(True)

        self.stop_btn.setEnabled(False)
        self.rate_combo.setEnabled(True)

    def _on_rate_changed(self, text: str):
        if self._active_mode:
            rate_hz = int(text.replace(" Hz", ""))
            self._send_timer.setInterval(int(1000 / rate_hz))
        if self._lpf_enabled:
            self._update_lpf_status()

    def _on_lpf_toggled(self, checked: bool):
        self._lpf_enabled = checked
        self.lpf_cutoff_spin.setEnabled(checked)
        if checked:
            self._update_lpf_status()
        else:
            self.lpf_status_label.setText("")

    def _on_lpf_cutoff_changed(self, val: float):
        self._lpf_cutoff_hz = val
        self._update_lpf_status()

    def _update_lpf_status(self):
        alpha = self._calc_alpha()
        self.lpf_status_label.setText(f"α={alpha:.3f}")

    def _calc_alpha(self) -> float:
        """Calculate IIR filter coefficient: alpha = dt / (RC + dt)."""
        rate_hz = int(self.rate_combo.currentText().replace(" Hz", ""))
        dt = 1.0 / rate_hz
        rc = 1.0 / (2.0 * math.pi * self._lpf_cutoff_hz)
        return dt / (rc + dt)

    def _get_filtered_value(self, raw: float) -> float:
        """Apply first-order IIR low-pass filter."""
        alpha = self._calc_alpha()
        self._smoothed_value = alpha * raw + (1.0 - alpha) * self._smoothed_value
        return self._smoothed_value

    def _send_command(self):
        if not self._transport.is_connected():
            self.stop_control()
            return

        mode = self._active_mode
        if mode == "duty":
            val = self.duty_card.spin.value()
            if self._lpf_enabled:
                val = self._get_filtered_value(val)
            self._transport.send_packet(build_set_duty(val))
        elif mode == "current":
            val = self.current_card.spin.value()
            if self._lpf_enabled:
                val = self._get_filtered_value(val)
            self._transport.send_packet(build_set_current(val))
        elif mode == "speed":
            val = self.speed_card.spin.value()
            if self._lpf_enabled:
                val = self._get_filtered_value(val)
            self._transport.send_packet(build_set_rpm(int(val)))
        elif mode == "position":
            val = self.position_card.spin.value()
            if self._lpf_enabled:
                val = self._get_filtered_value(val)
            self._transport.send_packet(build_set_pos(val))

        # Request feedback values
        self._transport.send_packet(build_get_values())

    def on_values(self, v: VescValues):
        """Update live feedback from COMM_GET_VALUES."""
        self.fb_labels["duty"].setText(f"{v.duty_now:.3f}")
        self.fb_labels["i_motor"].setText(f"{v.avg_motor_current:.2f} A")
        self.fb_labels["i_input"].setText(f"{v.avg_input_current:.2f} A")
        self.fb_labels["rpm"].setText(f"{v.rpm:.0f}")
        self.fb_labels["position"].setText(f"{v.pid_pos:.2f} deg")
        self.fb_labels["v_in"].setText(f"{v.v_in:.1f} V")
        self.fb_labels["power"].setText(f"{v.v_in * v.avg_input_current:.1f} W")
        self.fb_labels["temp_mos"].setText(f"{v.temp_mosfet:.1f} C")
        self.fb_labels["temp_mot"].setText(f"{v.temp_motor:.1f} C")
        self.fb_labels["fault"].setText(f"{v.fault_code}")

        if v.fault_code != 0:
            self.fb_labels["fault"].setStyleSheet(
                "font-family: monospace; font-size: 11px; color: red; font-weight: bold;"
            )
        else:
            self.fb_labels["fault"].setStyleSheet("font-family: monospace; font-size: 11px;")

    def cleanup(self):
        if self._active_mode:
            self.stop_control()
