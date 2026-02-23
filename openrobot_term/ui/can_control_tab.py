"""
Motor Control panel — unified RMD CAN + VESC EID control.

RMD modes: torque (0xA1), speed (0xA2), position (0xA3), multiturn (0xA4)
VESC modes: duty, current, speed (eRPM), position (deg)

One active at a time; Start locks selection, Motor OFF/STOP releases all.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QSlider, QComboBox,
    QMessageBox, QGridLayout, QScrollArea, QFrame, QCheckBox, QTabWidget,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import (
    build_torque_closed_loop, build_speed_closed_loop,
    build_position_closed_loop_1, build_set_multiturn_position,
    build_motor_off, build_motor_stop, build_motor_start,
    build_duty_closed_loop,
    build_mit_control, build_mit_enter, build_mit_exit, build_mit_set_zero,
)
from ..protocol.commands import (
    VescValues, build_set_duty, build_set_current,
    build_set_rpm, build_set_pos, build_get_values,
)

# VESC EID modes cap at 50 Hz (CAN EID bandwidth limit)
_VESC_MAX_RATE_HZ = 50


class CanControlTab(QWidget):
    torque_cmd_sent = pyqtSignal(float)  # emitted on each torque command send
    pos_cmd_sent = pyqtSignal(float)     # emitted on each position/multiturn command send

    def __init__(self, can_transport: PcanTransport):
        super().__init__()
        self._transport = can_transport
        self._active_mode = None
        self._send_timer = QTimer(self)
        self._send_timer.timeout.connect(self._send_command)

        self._discovered_ids: list[int] = []
        self._vesc_poll_idx = 0  # round-robin index for VESC get_values

        self._build_ui()

        # Wire signals — live feedback from both polling (0x9C) and control responses (0xA1-A4)
        self._transport.status_received.connect(self._on_status_received)
        self._transport.cmd_status_received.connect(self._on_status_received)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # ── Target + Periodic + Send Rate (single row) ──
        top_row = QHBoxLayout()
        outer.addLayout(top_row)

        top_row.addWidget(QLabel("Target:"))
        self.target_mode_combo = QComboBox()
        self.target_mode_combo.addItem("Solo", "solo")
        self.target_mode_combo.addItem("Broadcast", "broadcast")
        self.target_mode_combo.setToolTip(
            "Solo: send to selected target ID only\n"
            "Broadcast: send same command to all discovered IDs"
        )
        top_row.addWidget(self.target_mode_combo)

        self.periodic_chk = QCheckBox()
        self.periodic_chk.setToolTip("Send commands repeatedly at Send Rate")
        self.periodic_chk.setChecked(True)
        self.periodic_chk.toggled.connect(self._on_periodic_toggled)
        top_row.addWidget(self.periodic_chk)

        top_row.addWidget(QLabel("Send Rate:"))
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["10 Hz", "20 Hz", "50 Hz", "100 Hz", "200 Hz", "500 Hz"])
        self.rate_combo.setCurrentText("50 Hz")
        self.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        top_row.addWidget(self.rate_combo)

        top_row.addStretch()

        # ── STOP ALL button (shared, always visible) ──
        self.stop_all_btn = QPushButton("STOP ALL")
        self.stop_all_btn.setMinimumHeight(34)
        self.stop_all_btn.setStyleSheet(
            "QPushButton { background-color: #cc3333; color: white; "
            "font-weight: bold; font-size: 13px; padding: 4px 20px; }"
            "QPushButton:pressed { background-color: #881111; border: 2px solid #ff4444; }"
            "QPushButton:hover { background-color: #dd4444; }"
        )
        self.stop_all_btn.clicked.connect(self._stop_all)
        outer.addWidget(self.stop_all_btn)

        # ══════════════════════════════════════════════════
        # ── Tab widget: VESC EID / RMD CAN ──
        # ══════════════════════════════════════════════════
        self.control_tabs = QTabWidget()
        self.control_tabs.setStyleSheet(
            "QTabBar::tab { background: #3a3a3a; color: #888; "
            "padding: 6px 16px; border: 1px solid #555; "
            "border-bottom: none; margin-right: 2px; }"
            "QTabBar::tab:selected { background: #505050; color: #fff; "
            "font-weight: bold; border-bottom: 2px solid #4FC3F7; }"
            "QTabBar::tab:hover:!selected { background: #444; color: #bbb; }"
        )
        outer.addWidget(self.control_tabs, stretch=1)

        # Shared zero-reset button style
        _zero_btn_style = (
            "QPushButton { background-color: #555; color: #ccc; font-weight: bold; "
            "padding: 0px 4px; min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px; }"
            "QPushButton:hover { background-color: #777; }"
            "QPushButton:pressed { background-color: #333; }"
        )

        # ── VESC EID Tab ──
        vesc_scroll = QScrollArea()
        vesc_scroll.setWidgetResizable(True)
        vesc_scroll.setFrameShape(QFrame.Shape.NoFrame)
        vesc_container = QWidget()
        vesc_layout = QVBoxLayout(vesc_container)
        vesc_layout.setSpacing(6)
        vesc_scroll.setWidget(vesc_container)
        self.control_tabs.addTab(vesc_scroll, "VESC EID")

        # ── VESC Duty ──
        vd_group = QGroupBox("Duty")
        vdl = QVBoxLayout(vd_group)
        vdl.setSpacing(4)
        vdr1 = QHBoxLayout()
        self.vesc_duty_spin = QDoubleSpinBox()
        self.vesc_duty_spin.setRange(-1.0, 1.0)
        self.vesc_duty_spin.setSingleStep(0.01)
        self.vesc_duty_spin.setDecimals(3)
        self.vesc_duty_spin.setMinimumHeight(32)
        self.vesc_duty_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        vdr1.addWidget(self.vesc_duty_spin, stretch=1)
        vdr1.addWidget(QLabel("(-1~1)"))
        _vd_zero = QPushButton("0")
        _vd_zero.setStyleSheet(_zero_btn_style)
        _vd_zero.clicked.connect(lambda: self.vesc_duty_spin.setValue(0.0))
        vdr1.addWidget(_vd_zero)
        vdl.addLayout(vdr1)
        self.vesc_duty_slider = QSlider(Qt.Orientation.Horizontal)
        self.vesc_duty_slider.setRange(-1000, 1000)
        self.vesc_duty_slider.valueChanged.connect(lambda v: self.vesc_duty_spin.setValue(v / 1000.0))
        self.vesc_duty_spin.valueChanged.connect(lambda v: self.vesc_duty_slider.setValue(int(v * 1000)))
        vdl.addWidget(self.vesc_duty_slider)
        self.vesc_duty_spin.valueChanged.connect(self._on_vesc_duty_value_changed)
        self.vesc_duty_start_btn = QPushButton("Start Duty")
        self.vesc_duty_start_btn.setMinimumHeight(30)
        self.vesc_duty_start_btn.setStyleSheet(
            "QPushButton { background-color: #886622; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.vesc_duty_start_btn.clicked.connect(lambda: self._start("vesc_duty"))
        vdl.addWidget(self.vesc_duty_start_btn)
        vesc_layout.addWidget(vd_group)

        # ── VESC Current ──
        vc_group = QGroupBox("Current")
        vcl = QVBoxLayout(vc_group)
        vcl.setSpacing(4)
        vcr1 = QHBoxLayout()
        self.vesc_current_spin = QDoubleSpinBox()
        self.vesc_current_spin.setRange(-100.0, 100.0)
        self.vesc_current_spin.setSingleStep(0.1)
        self.vesc_current_spin.setDecimals(2)
        self.vesc_current_spin.setMinimumHeight(32)
        self.vesc_current_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        vcr1.addWidget(self.vesc_current_spin, stretch=1)
        vcr1.addWidget(QLabel("A"))
        _vc_zero = QPushButton("0")
        _vc_zero.setStyleSheet(_zero_btn_style)
        _vc_zero.clicked.connect(lambda: self.vesc_current_spin.setValue(0.0))
        vcr1.addWidget(_vc_zero)
        vcl.addLayout(vcr1)
        self.vesc_current_slider = QSlider(Qt.Orientation.Horizontal)
        self.vesc_current_slider.setRange(-1000, 1000)  # ±10A slider range
        self.vesc_current_slider.valueChanged.connect(lambda v: self.vesc_current_spin.setValue(v / 100.0))
        self.vesc_current_spin.valueChanged.connect(
            lambda v: self.vesc_current_slider.setValue(int(max(-10.0, min(10.0, v)) * 100))
        )
        vcl.addWidget(self.vesc_current_slider)
        self.vesc_current_spin.valueChanged.connect(self._on_vesc_current_value_changed)
        self.vesc_current_start_btn = QPushButton("Start Current")
        self.vesc_current_start_btn.setMinimumHeight(30)
        self.vesc_current_start_btn.setStyleSheet(
            "QPushButton { background-color: #886622; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.vesc_current_start_btn.clicked.connect(lambda: self._start("vesc_current"))
        vcl.addWidget(self.vesc_current_start_btn)
        vesc_layout.addWidget(vc_group)

        # ── VESC Speed ──
        vs_group = QGroupBox("Speed (eRPM)")
        vsl = QVBoxLayout(vs_group)
        vsl.setSpacing(4)
        vsr1 = QHBoxLayout()
        self.vesc_speed_spin = QDoubleSpinBox()
        self.vesc_speed_spin.setRange(-100000, 100000)
        self.vesc_speed_spin.setSingleStep(100)
        self.vesc_speed_spin.setDecimals(0)
        self.vesc_speed_spin.setMinimumHeight(32)
        self.vesc_speed_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        vsr1.addWidget(self.vesc_speed_spin, stretch=1)
        vsr1.addWidget(QLabel("eRPM"))
        _vs_zero = QPushButton("0")
        _vs_zero.setStyleSheet(_zero_btn_style)
        _vs_zero.clicked.connect(lambda: self.vesc_speed_spin.setValue(0.0))
        vsr1.addWidget(_vs_zero)
        vsl.addLayout(vsr1)
        self.vesc_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.vesc_speed_slider.setRange(-50000, 50000)
        self.vesc_speed_slider.valueChanged.connect(lambda v: self.vesc_speed_spin.setValue(float(v)))
        self.vesc_speed_spin.valueChanged.connect(lambda v: self.vesc_speed_slider.setValue(int(max(-50000, min(50000, v)))))
        vsl.addWidget(self.vesc_speed_slider)
        self.vesc_speed_spin.valueChanged.connect(self._on_vesc_speed_value_changed)
        self.vesc_speed_start_btn = QPushButton("Start Speed")
        self.vesc_speed_start_btn.setMinimumHeight(30)
        self.vesc_speed_start_btn.setStyleSheet(
            "QPushButton { background-color: #886622; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.vesc_speed_start_btn.clicked.connect(lambda: self._start("vesc_speed"))
        vsl.addWidget(self.vesc_speed_start_btn)
        vesc_layout.addWidget(vs_group)

        # ── VESC Position ──
        vp_group = QGroupBox("Position (deg)")
        vpl = QVBoxLayout(vp_group)
        vpl.setSpacing(4)
        vpr1 = QHBoxLayout()
        self.vesc_pos_spin = QDoubleSpinBox()
        self.vesc_pos_spin.setRange(0.0, 360.0)
        self.vesc_pos_spin.setSingleStep(1.0)
        self.vesc_pos_spin.setDecimals(2)
        self.vesc_pos_spin.setMinimumHeight(32)
        self.vesc_pos_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        vpr1.addWidget(self.vesc_pos_spin, stretch=1)
        vpr1.addWidget(QLabel("deg"))
        _vp_zero = QPushButton("0")
        _vp_zero.setStyleSheet(_zero_btn_style)
        _vp_zero.clicked.connect(lambda: self.vesc_pos_spin.setValue(0.0))
        vpr1.addWidget(_vp_zero)
        vpl.addLayout(vpr1)
        self.vesc_pos_slider = QSlider(Qt.Orientation.Horizontal)
        self.vesc_pos_slider.setRange(0, 36000)
        self.vesc_pos_slider.valueChanged.connect(lambda v: self.vesc_pos_spin.setValue(v / 100.0))
        self.vesc_pos_spin.valueChanged.connect(lambda v: self.vesc_pos_slider.setValue(int(v * 100)))
        vpl.addWidget(self.vesc_pos_slider)
        self.vesc_pos_spin.valueChanged.connect(self._on_vesc_pos_value_changed)
        self.vesc_pos_start_btn = QPushButton("Start Position")
        self.vesc_pos_start_btn.setMinimumHeight(30)
        self.vesc_pos_start_btn.setStyleSheet(
            "QPushButton { background-color: #886622; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.vesc_pos_start_btn.clicked.connect(lambda: self._start("vesc_position"))
        vpl.addWidget(self.vesc_pos_start_btn)
        vesc_layout.addWidget(vp_group)

        # VESC feedback
        vesc_fb_group = QGroupBox("VESC Feedback")
        vesc_fb_layout = QGridLayout(vesc_fb_group)
        vesc_fb_layout.setSpacing(2)

        self.vesc_fb_labels = {}
        vesc_items = [
            ("Duty:", "duty"), ("I_motor:", "i_motor"),
            ("I_input:", "i_input"), ("RPM:", "rpm"),
            ("Position:", "position"), ("V_in:", "v_in"),
            ("Power:", "power"), ("T_mos:", "temp_mos"),
            ("T_mot:", "temp_mot"), ("Fault:", "fault"),
        ]
        for i, (label, key) in enumerate(vesc_items):
            row, col = i // 2, (i % 2) * 2
            lb = QLabel(label)
            lb.setStyleSheet("font-size: 11px; font-weight: bold;")
            vesc_fb_layout.addWidget(lb, row, col)
            val = QLabel("--")
            val.setStyleSheet("font-family: monospace; font-size: 11px;")
            vesc_fb_layout.addWidget(val, row, col + 1)
            self.vesc_fb_labels[key] = val

        vesc_layout.addWidget(vesc_fb_group)
        vesc_layout.addStretch()

        # ── RMD CAN Tab ──
        rmd_scroll = QScrollArea()
        rmd_scroll.setWidgetResizable(True)
        rmd_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rmd_container = QWidget()
        rmd_layout = QVBoxLayout(rmd_container)
        rmd_layout.setSpacing(6)
        rmd_scroll.setWidget(rmd_container)
        self.control_tabs.addTab(rmd_scroll, "RMD CAN")

        # ── Motor OFF / STOP / START (RMD) ──
        motor_row = QHBoxLayout()
        rmd_layout.addLayout(motor_row)

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
        _tq_zero = QPushButton("0")
        _tq_zero.setStyleSheet(_zero_btn_style)
        _tq_zero.clicked.connect(lambda: self.torque_spin.setValue(0.0))
        tr1.addWidget(_tq_zero)
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
        rmd_layout.addWidget(torque_group)

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
        _sp_zero = QPushButton("0")
        _sp_zero.setStyleSheet(_zero_btn_style)
        _sp_zero.clicked.connect(lambda: self.speed_spin.setValue(0.0))
        sr1.addWidget(_sp_zero)
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
        rmd_layout.addWidget(speed_group)

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
        _ps_zero = QPushButton("0")
        _ps_zero.setStyleSheet(_zero_btn_style)
        _ps_zero.clicked.connect(lambda: self.pos_spin.setValue(0.0))
        pr1.addWidget(_ps_zero)
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
        rmd_layout.addWidget(pos_group)

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
        _mt_zero = QPushButton("0")
        _mt_zero.setStyleSheet(_zero_btn_style)
        _mt_zero.clicked.connect(lambda: self.mt_pos_spin.setValue(0.0))
        mr2.addWidget(_mt_zero)
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
        rmd_layout.addWidget(mt_group)

        # ── Duty Control ──
        duty_group = QGroupBox("Duty (0xA5)")
        dl = QVBoxLayout(duty_group)
        dl.setSpacing(4)

        dr1 = QHBoxLayout()
        self.duty_spin = QDoubleSpinBox()
        self.duty_spin.setRange(-1.0, 1.0)
        self.duty_spin.setSingleStep(0.01)
        self.duty_spin.setDecimals(3)
        self.duty_spin.setMinimumHeight(32)
        self.duty_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        dr1.addWidget(self.duty_spin, stretch=1)
        dr1.addWidget(QLabel("(-1~1)"))
        _dt_zero = QPushButton("0")
        _dt_zero.setStyleSheet(_zero_btn_style)
        _dt_zero.clicked.connect(lambda: self.duty_spin.setValue(0.0))
        dr1.addWidget(_dt_zero)
        dl.addLayout(dr1)
        self.duty_slider = QSlider(Qt.Orientation.Horizontal)
        self.duty_slider.setRange(-1000, 1000)
        self.duty_slider.valueChanged.connect(lambda v: self.duty_spin.setValue(v / 1000.0))
        self.duty_spin.valueChanged.connect(lambda v: self.duty_slider.setValue(int(v * 1000)))
        dl.addWidget(self.duty_slider)

        self.duty_spin.valueChanged.connect(self._on_duty_value_changed)

        self.duty_start_btn = QPushButton("Start Duty")
        self.duty_start_btn.setMinimumHeight(30)
        self.duty_start_btn.setStyleSheet(
            "QPushButton { background-color: #338833; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.duty_start_btn.clicked.connect(lambda: self._start("duty"))
        dl.addWidget(self.duty_start_btn)
        rmd_layout.addWidget(duty_group)

        # RMD feedback
        rmd_fb_group = QGroupBox("RMD Feedback")
        rmd_fb_layout = QGridLayout(rmd_fb_group)
        rmd_fb_layout.setSpacing(2)

        self.fb_labels = {}
        rmd_items = [
            ("Temp:", "temp"), ("Torque:", "torque"),
            ("Speed:", "speed"), ("Pos:", "position"),
        ]
        for i, (label, key) in enumerate(rmd_items):
            row, col = i // 2, (i % 2) * 2
            lb = QLabel(label)
            lb.setStyleSheet("font-size: 11px; font-weight: bold;")
            rmd_fb_layout.addWidget(lb, row, col)
            val = QLabel("--")
            val.setStyleSheet("font-family: monospace; font-size: 11px;")
            rmd_fb_layout.addWidget(val, row, col + 1)
            self.fb_labels[key] = val

        rmd_layout.addWidget(rmd_fb_group)
        rmd_layout.addStretch()

        # ── MIT Mode Tab ──
        mit_scroll = QScrollArea()
        mit_scroll.setWidgetResizable(True)
        mit_scroll.setFrameShape(QFrame.Shape.NoFrame)
        mit_container = QWidget()
        mit_layout = QVBoxLayout(mit_container)
        mit_layout.setSpacing(6)
        mit_scroll.setWidget(mit_container)
        self.control_tabs.addTab(mit_scroll, "MIT Mode")

        # Motor mode buttons
        mit_mode_row = QHBoxLayout()
        mit_layout.addLayout(mit_mode_row)

        mit_enter_btn = QPushButton("Enter Motor Mode")
        mit_enter_btn.setMinimumHeight(28)
        mit_enter_btn.setStyleSheet(
            "QPushButton { background-color: #338833; color: white; font-weight: bold; }"
            "QPushButton:pressed { background-color: #114411; border: 2px solid #66ff66; }"
            "QPushButton:hover { background-color: #449944; }"
        )
        mit_enter_btn.clicked.connect(lambda: self._motor_cmd(build_mit_enter()))
        mit_mode_row.addWidget(mit_enter_btn)

        mit_exit_btn = QPushButton("Exit Motor Mode")
        mit_exit_btn.setMinimumHeight(28)
        mit_exit_btn.setStyleSheet(
            "QPushButton { background-color: #884400; color: white; font-weight: bold; }"
            "QPushButton:pressed { background-color: #442200; border: 2px solid #ffaa00; }"
            "QPushButton:hover { background-color: #995500; }"
        )
        mit_exit_btn.clicked.connect(lambda: self._motor_cmd(build_mit_exit()))
        mit_mode_row.addWidget(mit_exit_btn)

        # Set Zero Pos on separate row
        mit_zero_row = QHBoxLayout()
        mit_layout.addLayout(mit_zero_row)

        mit_zero_btn = QPushButton("Set Zero Pos")
        mit_zero_btn.setMinimumHeight(28)
        mit_zero_btn.setStyleSheet(
            "QPushButton:pressed { background-color: #222; border: 2px solid #aaa; }"
        )
        mit_zero_btn.clicked.connect(self._on_mit_set_zero)
        mit_zero_row.addWidget(mit_zero_btn)
        mit_zero_row.addStretch()

        # MIT Impedance Control group
        mit_ctrl_group = QGroupBox("MIT Impedance Control (0xC0)")
        mcl = QVBoxLayout(mit_ctrl_group)
        mcl.setSpacing(4)

        # Position (rad)
        mp_row = QHBoxLayout()
        mp_row.addWidget(QLabel("Position (rad):"))
        self.mit_p_spin = QDoubleSpinBox()
        self.mit_p_spin.setRange(-12.5, 12.5)
        self.mit_p_spin.setSingleStep(0.1)
        self.mit_p_spin.setDecimals(3)
        self.mit_p_spin.setMinimumHeight(32)
        self.mit_p_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        mp_row.addWidget(self.mit_p_spin, stretch=1)
        _p_zero = QPushButton("0")
        _p_zero.setStyleSheet(_zero_btn_style)
        _p_zero.clicked.connect(lambda: self.mit_p_spin.setValue(0.0))
        mp_row.addWidget(_p_zero)
        mcl.addLayout(mp_row)
        self.mit_p_slider = QSlider(Qt.Orientation.Horizontal)
        self.mit_p_slider.setRange(-12500, 12500)
        self.mit_p_slider.valueChanged.connect(lambda v: self.mit_p_spin.setValue(v / 1000.0))
        self.mit_p_spin.valueChanged.connect(lambda v: self.mit_p_slider.setValue(int(v * 1000)))
        mcl.addWidget(self.mit_p_slider)

        # Velocity (rad/s)
        mv_row = QHBoxLayout()
        mv_row.addWidget(QLabel("Velocity (rad/s):"))
        self.mit_v_spin = QDoubleSpinBox()
        self.mit_v_spin.setRange(-76.0, 76.0)
        self.mit_v_spin.setSingleStep(1.0)
        self.mit_v_spin.setDecimals(2)
        self.mit_v_spin.setMinimumHeight(32)
        self.mit_v_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        mv_row.addWidget(self.mit_v_spin, stretch=1)
        _v_zero = QPushButton("0")
        _v_zero.setStyleSheet(_zero_btn_style)
        _v_zero.clicked.connect(lambda: self.mit_v_spin.setValue(0.0))
        mv_row.addWidget(_v_zero)
        mcl.addLayout(mv_row)
        self.mit_v_slider = QSlider(Qt.Orientation.Horizontal)
        self.mit_v_slider.setRange(-7600, 7600)
        self.mit_v_slider.valueChanged.connect(lambda v: self.mit_v_spin.setValue(v / 100.0))
        self.mit_v_spin.valueChanged.connect(lambda v: self.mit_v_slider.setValue(int(v * 100)))
        mcl.addWidget(self.mit_v_slider)

        # Kp (A/rad)
        mkp_row = QHBoxLayout()
        mkp_row.addWidget(QLabel("Kp (A/rad):"))
        self.mit_kp_spin = QDoubleSpinBox()
        self.mit_kp_spin.setRange(0.0, 500.0)
        self.mit_kp_spin.setSingleStep(1.0)
        self.mit_kp_spin.setDecimals(1)
        self.mit_kp_spin.setValue(1.0)
        self.mit_kp_spin.setMinimumHeight(32)
        self.mit_kp_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        mkp_row.addWidget(self.mit_kp_spin, stretch=1)
        _kp_zero = QPushButton("0")
        _kp_zero.setStyleSheet(_zero_btn_style)
        _kp_zero.clicked.connect(lambda: self.mit_kp_spin.setValue(0.0))
        mkp_row.addWidget(_kp_zero)
        mcl.addLayout(mkp_row)
        self.mit_kp_slider = QSlider(Qt.Orientation.Horizontal)
        self.mit_kp_slider.setRange(0, 5000)
        self.mit_kp_slider.valueChanged.connect(lambda v: self.mit_kp_spin.setValue(v / 10.0))
        self.mit_kp_spin.valueChanged.connect(lambda v: self.mit_kp_slider.setValue(int(v * 10)))
        mcl.addWidget(self.mit_kp_slider)

        # Kd (A*s/rad)
        mkd_row = QHBoxLayout()
        mkd_row.addWidget(QLabel("Kd (A*s/rad):"))
        self.mit_kd_spin = QDoubleSpinBox()
        self.mit_kd_spin.setRange(0.0, 5.0)
        self.mit_kd_spin.setSingleStep(0.01)
        self.mit_kd_spin.setDecimals(3)
        self.mit_kd_spin.setValue(0.1)
        self.mit_kd_spin.setMinimumHeight(32)
        self.mit_kd_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        mkd_row.addWidget(self.mit_kd_spin, stretch=1)
        _kd_zero = QPushButton("0")
        _kd_zero.setStyleSheet(_zero_btn_style)
        _kd_zero.clicked.connect(lambda: self.mit_kd_spin.setValue(0.0))
        mkd_row.addWidget(_kd_zero)
        mcl.addLayout(mkd_row)
        self.mit_kd_slider = QSlider(Qt.Orientation.Horizontal)
        self.mit_kd_slider.setRange(0, 5000)
        self.mit_kd_slider.valueChanged.connect(lambda v: self.mit_kd_spin.setValue(v / 1000.0))
        self.mit_kd_spin.valueChanged.connect(lambda v: self.mit_kd_slider.setValue(int(v * 1000)))
        mcl.addWidget(self.mit_kd_slider)

        # Torque FF (A)
        mt_row = QHBoxLayout()
        mt_row.addWidget(QLabel("Torque FF (A):"))
        self.mit_tff_spin = QDoubleSpinBox()
        self.mit_tff_spin.setRange(-33.0, 33.0)
        self.mit_tff_spin.setSingleStep(0.1)
        self.mit_tff_spin.setDecimals(2)
        self.mit_tff_spin.setMinimumHeight(32)
        self.mit_tff_spin.setStyleSheet("font-size: 15px; font-weight: bold;")
        mt_row.addWidget(self.mit_tff_spin, stretch=1)
        _tff_zero = QPushButton("0")
        _tff_zero.setStyleSheet(_zero_btn_style)
        _tff_zero.clicked.connect(lambda: self.mit_tff_spin.setValue(0.0))
        mt_row.addWidget(_tff_zero)
        mcl.addLayout(mt_row)
        self.mit_tff_slider = QSlider(Qt.Orientation.Horizontal)
        self.mit_tff_slider.setRange(-3300, 3300)
        self.mit_tff_slider.valueChanged.connect(lambda v: self.mit_tff_spin.setValue(v / 100.0))
        self.mit_tff_spin.valueChanged.connect(lambda v: self.mit_tff_slider.setValue(int(v * 100)))
        mcl.addWidget(self.mit_tff_slider)

        # Value changed handlers
        self.mit_p_spin.valueChanged.connect(self._on_mit_value_changed)
        self.mit_v_spin.valueChanged.connect(self._on_mit_value_changed)
        self.mit_kp_spin.valueChanged.connect(self._on_mit_value_changed)
        self.mit_kd_spin.valueChanged.connect(self._on_mit_value_changed)
        self.mit_tff_spin.valueChanged.connect(self._on_mit_value_changed)

        self.mit_start_btn = QPushButton("Start MIT")
        self.mit_start_btn.setMinimumHeight(30)
        self.mit_start_btn.setStyleSheet(
            "QPushButton { background-color: #338833; color: white; "
            "font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self.mit_start_btn.clicked.connect(lambda: self._start("mit"))
        mcl.addWidget(self.mit_start_btn)
        mit_layout.addWidget(mit_ctrl_group)

        # MIT feedback
        mit_fb_group = QGroupBox("MIT Feedback")
        mit_fb_layout = QGridLayout(mit_fb_group)
        mit_fb_layout.setSpacing(2)

        self.mit_fb_labels = {}
        mit_fb_items = [
            ("Temp:", "temp"), ("Torque:", "torque"),
            ("Speed:", "speed"), ("Pos:", "position"),
        ]
        for i, (label, key) in enumerate(mit_fb_items):
            row, col = i // 2, (i % 2) * 2
            lb = QLabel(label)
            lb.setStyleSheet("font-size: 11px; font-weight: bold;")
            mit_fb_layout.addWidget(lb, row, col)
            val = QLabel("--")
            val.setStyleSheet("font-family: monospace; font-size: 11px;")
            mit_fb_layout.addWidget(val, row, col + 1)
            self.mit_fb_labels[key] = val

        mit_layout.addWidget(mit_fb_group)
        mit_layout.addStretch()

        # Collect all start buttons for enable/disable management
        self._all_start_btns = [
            self.torque_start_btn, self.speed_start_btn,
            self.pos_start_btn, self.mt_start_btn,
            self.duty_start_btn, self.mit_start_btn,
            self.vesc_duty_start_btn, self.vesc_current_start_btn,
            self.vesc_speed_start_btn, self.vesc_pos_start_btn,
        ]

    # ── Helpers ──

    def _is_vesc_mode(self, mode: str = None) -> bool:
        """Check if a mode uses VESC EID protocol."""
        m = mode or self._active_mode
        return m is not None and m.startswith("vesc_")

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

    def _get_rate_hz(self) -> int:
        return int(self.rate_combo.currentText().replace(" Hz", ""))

    def _get_effective_rate_hz(self) -> int:
        """Get rate clamped by VESC max for VESC modes."""
        rate = self._get_rate_hz()
        if self._is_vesc_mode():
            return min(rate, _VESC_MAX_RATE_HZ)
        return rate

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
        elif self._active_mode == "duty":
            return self.duty_spin.value()
        elif self._active_mode == "mit":
            return self.mit_p_spin.value()
        elif self._active_mode == "vesc_duty":
            return self.vesc_duty_spin.value()
        elif self._active_mode == "vesc_current":
            return self.vesc_current_spin.value()
        elif self._active_mode == "vesc_speed":
            return self.vesc_speed_spin.value()
        elif self._active_mode == "vesc_position":
            return self.vesc_pos_spin.value()
        return 0.0

    def _is_broadcast(self) -> bool:
        """Whether broadcast mode is selected."""
        return self.target_mode_combo.currentData() == "broadcast"

    def _broadcast_or_solo_rmd(self, data: bytes):
        """Solo: send_frame(), Broadcast: send_frame_to() for all discovered IDs."""
        if self._is_broadcast():
            for mid in self._discovered_ids:
                self._transport.send_frame_to(mid, data)
        else:
            self._transport.send_frame(data)

    def _broadcast_or_solo_vesc(self, payload: bytes):
        """Solo: send_vesc_to_target(), Broadcast: target_id=255 with send_mode=2.
        Broadcast uses send_mode=2 (process only, no response) to prevent
        multi-frame EID response collision from multiple motors."""
        if self._is_broadcast():
            self._transport.send_vesc_command(255, payload, send_mode=2)
        else:
            self._transport.send_vesc_to_target(payload)

    def _poll_vesc_values(self):
        """Poll VESC get_values — round-robin in broadcast to avoid EID collision.
        Multiple motors responding with multi-frame EID simultaneously corrupts
        the shared reassembly buffer (all use dest=0xFE). Round-robin ensures
        only one motor responds per tick."""
        if self._is_broadcast() and self._discovered_ids:
            mid = self._discovered_ids[self._vesc_poll_idx % len(self._discovered_ids)]
            self._vesc_poll_idx += 1
            self._transport.send_vesc_command(mid, build_get_values(), send_mode=0)
        else:
            self._transport.send_vesc_to_target(build_get_values())

    def _is_periodic(self) -> bool:
        """Whether periodic send is active."""
        return self.periodic_chk.isChecked()

    # ── Control actions ──

    def _start(self, mode: str):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Open PCAN connection first.")
            return

        # Stop previous mode first (auto-cancel)
        self._stop_loop()

        self._active_mode = mode

        # Disable all start buttons while active
        for btn in self._all_start_btns:
            btn.setEnabled(False)
        self.rate_combo.setEnabled(False)

        # Start appropriate sender and send initial value
        self._apply_send_mode()
        self._send_current_value()

    def _stop_loop(self):
        """Stop the active control loop (timer, no CAN command)."""
        self._send_timer.stop()
        self._active_mode = None
        for btn in self._all_start_btns:
            btn.setEnabled(True)
        self.rate_combo.setEnabled(True)

    def update_discovered_ids(self, ids: list[int]):
        """Store discovered CAN motor IDs for broadcast stop."""
        self._discovered_ids = list(ids)

    def _stop_all(self):
        """STOP ALL: stop loop + send RMD off to all IDs + VESC current=0."""
        self._stop_loop()
        self.torque_cmd_sent.emit(0.0)
        self.pos_cmd_sent.emit(float('nan'))
        if self._transport.is_connected():
            self._transport.stop_periodic()
            ids = self._discovered_ids if self._discovered_ids else range(1, 9)
            for mid in ids:
                self._transport.send_frame_to(mid, build_motor_off())
            self._transport.send_vesc_to_target(build_set_current(0.0))

    def _motor_cmd(self, data: bytes):
        """Send a motor OFF/STOP/START command, stopping any active control loop first."""
        self._stop_loop()
        self.torque_cmd_sent.emit(0.0)
        self.pos_cmd_sent.emit(float('nan'))
        # Send RMD command
        self._send_once(data)
        # Also stop VESC side for safety
        if self._transport.is_connected():
            self._transport.send_vesc_to_target(build_set_current(0.0))

    def _send_current_value(self):
        """Send the current target value once for the active mode."""
        if not self._transport.is_connected():
            return
        mode = self._active_mode

        # RMD modes
        if mode == "torque":
            val = self.torque_spin.value()
            self._broadcast_or_solo_rmd(build_torque_closed_loop(val))
            self.torque_cmd_sent.emit(val)
        elif mode == "speed":
            val = self.speed_spin.value()
            self._broadcast_or_solo_rmd(build_speed_closed_loop(val, self._get_speed_mode()))
        elif mode == "position":
            val = self.pos_spin.value()
            self._broadcast_or_solo_rmd(build_position_closed_loop_1(val))
            self.pos_cmd_sent.emit(val)
        elif mode == "multiturn":
            val = self.mt_pos_spin.value()
            self._broadcast_or_solo_rmd(
                build_set_multiturn_position(self.mt_dps_spin.value(), val)
            )
            self.pos_cmd_sent.emit(val)
        elif mode == "duty":
            val = self.duty_spin.value()
            self._broadcast_or_solo_rmd(build_duty_closed_loop(val))
        elif mode == "mit":
            self._broadcast_or_solo_rmd(build_mit_control(
                self.mit_p_spin.value(), self.mit_v_spin.value(),
                self.mit_kp_spin.value(), self.mit_kd_spin.value(),
                self.mit_tff_spin.value()
            ))

        # VESC EID modes
        elif mode == "vesc_duty":
            val = self.vesc_duty_spin.value()
            self._broadcast_or_solo_vesc(build_set_duty(val))
        elif mode == "vesc_current":
            val = self.vesc_current_spin.value()
            self._broadcast_or_solo_vesc(build_set_current(val))
        elif mode == "vesc_speed":
            val = self.vesc_speed_spin.value()
            self._broadcast_or_solo_vesc(build_set_rpm(int(val)))
        elif mode == "vesc_position":
            val = self.vesc_pos_spin.value()
            self._broadcast_or_solo_vesc(build_set_pos(val))

        # VESC modes also poll feedback (round-robin to avoid EID collision)
        if self._is_vesc_mode(mode):
            self._poll_vesc_values()

    # ── Send mode management (single source of truth) ──

    def _apply_send_mode(self):
        """Stop timer, then restart if periodic. Called on: start, periodic toggle, rate change."""
        self._send_timer.stop()

        if self._active_mode is None:
            return

        if self.periodic_chk.isChecked():
            self._send_timer.start(int(1000 / self._get_effective_rate_hz()))
        # else: single-shot via valueChanged handlers

    def _on_periodic_toggled(self, _checked: bool):
        self._apply_send_mode()

    def _on_rate_changed(self, text: str):
        self._apply_send_mode()

    # ── Timer-based periodic send ──

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
            self._broadcast_or_solo_rmd(build_torque_closed_loop(val))
            self.torque_cmd_sent.emit(val)

    def _on_speed_value_changed(self):
        if self._active_mode == "speed" and not self._is_periodic():
            self._broadcast_or_solo_rmd(build_speed_closed_loop(self.speed_spin.value(), self._get_speed_mode()))

    def _on_pos_value_changed(self):
        if self._active_mode == "position" and not self._is_periodic():
            val = self.pos_spin.value()
            self._broadcast_or_solo_rmd(build_position_closed_loop_1(val))
            self.pos_cmd_sent.emit(val)

    def _on_mt_value_changed(self):
        if self._active_mode == "multiturn" and not self._is_periodic():
            val = self.mt_pos_spin.value()
            self._broadcast_or_solo_rmd(
                build_set_multiturn_position(self.mt_dps_spin.value(), val)
            )
            self.pos_cmd_sent.emit(val)

    def _on_duty_value_changed(self):
        if self._active_mode == "duty" and not self._is_periodic():
            self._broadcast_or_solo_rmd(build_duty_closed_loop(self.duty_spin.value()))

    def _on_mit_value_changed(self):
        if self._active_mode == "mit" and not self._is_periodic():
            self._broadcast_or_solo_rmd(build_mit_control(
                self.mit_p_spin.value(), self.mit_v_spin.value(),
                self.mit_kp_spin.value(), self.mit_kd_spin.value(),
                self.mit_tff_spin.value()
            ))

    def _on_vesc_duty_value_changed(self):
        if self._active_mode == "vesc_duty" and not self._is_periodic():
            self._broadcast_or_solo_vesc(build_set_duty(self.vesc_duty_spin.value()))

    def _on_vesc_current_value_changed(self):
        if self._active_mode == "vesc_current" and not self._is_periodic():
            self._broadcast_or_solo_vesc(build_set_current(self.vesc_current_spin.value()))

    def _on_vesc_speed_value_changed(self):
        if self._active_mode == "vesc_speed" and not self._is_periodic():
            self._broadcast_or_solo_vesc(build_set_rpm(int(self.vesc_speed_spin.value())))

    def _on_vesc_pos_value_changed(self):
        if self._active_mode == "vesc_position" and not self._is_periodic():
            self._broadcast_or_solo_vesc(build_set_pos(self.vesc_pos_spin.value()))

    def _on_mit_set_zero(self):
        """Set Zero Pos: stop MIT loop, exit motor mode, zero position, then set-zero."""
        # 1) Stop periodic MIT send
        if self._active_mode == "mit":
            self._stop_loop()
        # 2) Exit motor mode — release motor so FW clears MIT control
        self._send_once(build_mit_exit())
        # 3) Reset position spinbox to 0
        self.mit_p_spin.setValue(0.0)
        # 4) Send set zero command to FW
        self._send_once(build_mit_set_zero())

    def _send_once(self, data: bytes):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Open PCAN connection first.")
            return
        self._broadcast_or_solo_rmd(data)

    # ── Live Feedback ──

    def _on_status_received(self, motor_id: int, status):
        """RMD status feedback (0x9C or 0xA1-A5/0xC0 responses)."""
        self.fb_labels["temp"].setText(f"{status.motor_temp:.0f} C")
        self.fb_labels["torque"].setText(f"{status.torque_curr:.3f} A")
        self.fb_labels["speed"].setText(f"{status.speed_dps} dps")
        self.fb_labels["position"].setText(f"{status.enc_pos:.2f} deg")
        # MIT tab shares same feedback format
        self.mit_fb_labels["temp"].setText(f"{status.motor_temp:.0f} C")
        self.mit_fb_labels["torque"].setText(f"{status.torque_curr:.3f} A")
        self.mit_fb_labels["speed"].setText(f"{status.speed_dps} dps")
        self.mit_fb_labels["position"].setText(f"{status.enc_pos:.2f} deg")

    def on_values(self, v: VescValues):
        """VESC feedback from COMM_GET_VALUES (called by main_window dispatcher)."""
        self.vesc_fb_labels["duty"].setText(f"{v.duty_now:.3f}")
        self.vesc_fb_labels["i_motor"].setText(f"{v.avg_motor_current:.2f} A")
        self.vesc_fb_labels["i_input"].setText(f"{v.avg_input_current:.2f} A")
        self.vesc_fb_labels["rpm"].setText(f"{v.rpm:.0f}")
        self.vesc_fb_labels["position"].setText(f"{v.pid_pos:.2f} deg")
        self.vesc_fb_labels["v_in"].setText(f"{v.v_in:.1f} V")
        self.vesc_fb_labels["power"].setText(f"{v.v_in * v.avg_input_current:.1f} W")
        self.vesc_fb_labels["temp_mos"].setText(f"{v.temp_mosfet:.1f} C")
        self.vesc_fb_labels["temp_mot"].setText(f"{v.temp_motor:.1f} C")
        self.vesc_fb_labels["fault"].setText(f"{v.fault_code}")

        if v.fault_code != 0:
            self.vesc_fb_labels["fault"].setStyleSheet(
                "font-family: monospace; font-size: 11px; color: red; font-weight: bold;"
            )
        else:
            self.vesc_fb_labels["fault"].setStyleSheet("font-family: monospace; font-size: 11px;")

    def cleanup(self):
        self._stop_loop()
