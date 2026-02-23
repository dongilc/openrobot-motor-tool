"""
CAN Data tab — Encoder/Config, PID, Diagnostics, Real-time Graphs,
Raw CAN data (collapsible).
"""

import csv
import time
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFileDialog, QApplication,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QTimer, QEvent, pyqtSignal

import numpy as np
import pyqtgraph as pg

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import (
    CAN_HEADER_ID, RmdCommand, RmdStatus, RmdStatus3,
    parse_pid, parse_encoder, parse_multi_turn_angle,
    parse_max_current,
    build_read_encoder, build_read_multi_turn_angle,
    build_write_encoder_offset, build_write_current_pos_to_rom,
    build_read_pid, build_write_pid_to_rom,
    build_read_fault_code, build_read_max_current,
    build_write_max_current_to_rom,
    build_motor_off,
)
from ..workers.can_poller import CanPoller
from .plot_style import style_plot, graph_pen, style_legend, set_curve_visible, GRAPH_COLORS

RENDER_INTERVAL_MS = 33  # ~30 fps
BUF_CAP = 12000
BUF_COMPACT = 24000

# Control mode enum (matches firmware CONTROL_MODE in app_openrobot.h)
_CONTROL_MODE_NAMES = {
    -1: "N/A",
    0: "NONE",
    1: "RELEASE",
    2: "BRAKE",
    3: "DUTY",
    4: "CURRENT",
    5: "DAMP_CURR",
    6: "POS",
    7: "DIRECT_PID",
    8: "DPS_TOUT",
    9: "DPS_DUR",
    10: "SERVO",
    11: "TRAJ",
    12: "FAULT",
}


class _GrowBuffer:
    """Flat growing numpy buffer with zero-copy slice view."""
    __slots__ = ('_buf', '_len')

    def __init__(self, cap: int):
        self._buf = np.empty(cap, dtype=np.float64)
        self._len = 0

    def append(self, val: float):
        if self._len >= len(self._buf):
            new = np.empty(len(self._buf) * 2, dtype=np.float64)
            new[:self._len] = self._buf[:self._len]
            self._buf = new
        self._buf[self._len] = val
        self._len += 1

    def clear(self):
        self._len = 0

    def __len__(self):
        return self._len

    def array(self) -> np.ndarray:
        return self._buf[:self._len]

    def compact(self, keep: int):
        if self._len > keep:
            start = self._len - keep
            self._buf[:keep] = self._buf[start:self._len]
            self._len = keep


class _DeviceBuffers:
    """Complete buffer set for one motor device."""

    def __init__(self):
        self.time = _GrowBuffer(BUF_CAP)
        self.mode = _GrowBuffer(BUF_CAP)
        self.torque = _GrowBuffer(BUF_CAP)
        self.torque_cmd = _GrowBuffer(BUF_CAP)
        self.speed = _GrowBuffer(BUF_CAP)
        self.pos = _GrowBuffer(BUF_CAP)
        self.phase_a = _GrowBuffer(BUF_CAP)
        self.phase_b = _GrowBuffer(BUF_CAP)
        self.phase_c = _GrowBuffer(BUF_CAP)
        self.temp = _GrowBuffer(BUF_CAP)
        self.pos_cmd = _GrowBuffer(BUF_CAP)
        # Position unwrapping state
        self.prev_enc_pos = None
        self.enc_offset = 0.0
        # Per-device status3 (0x9D)
        self.last_status3 = None

    def all_buffers(self) -> list[_GrowBuffer]:
        return [
            self.time, self.mode, self.torque, self.torque_cmd,
            self.speed, self.pos, self.phase_a, self.phase_b,
            self.phase_c, self.temp, self.pos_cmd,
        ]

    def clear(self):
        for rb in self.all_buffers():
            rb.clear()
        self.prev_enc_pos = None
        self.enc_offset = 0.0
        self.last_status3 = None

    def compact(self, keep: int):
        for rb in self.all_buffers():
            rb.compact(keep)


# ── CAN Data Tab ──

class CanDataTab(QWidget):
    rescan_requested = pyqtSignal()  # request connection_bar to re-scan before polling

    def __init__(self, can_transport: PcanTransport):
        super().__init__()
        self._transport = can_transport

        # ── Graph state ──
        self._poller = None
        self._t0 = None
        self._csv_file = None
        self._csv_writer = None
        self._polling = False
        self._pending_poll_start = False  # waiting for re-scan before polling
        self._auto_range = True
        self._x_window = 10.0
        self._dirty = False
        self._last_status = None
        self._frame_count = 0

        self._last_torque_cmd = 0.0
        self._last_pos_cmd = float('nan')
        self._mt_init_angle = None  # multiturn angle read at polling start
        self._discovered_ids: list[int] = []  # populated by CAN scan

        # Per-device buffers: {motor_id: _DeviceBuffers}
        self._dev_buffers: dict[int, _DeviceBuffers] = {}

        # Overlay mode: dynamically created curves {motor_id: {curve_key: PlotDataItem}}
        self._overlay_curves: dict[int, dict[str, pg.PlotDataItem]] = {}
        self._overlay_active = False  # True when currently in overlay mode

        self._build_ui()

        # Wire signals
        self._transport.frame_received.connect(self._on_frame_received)
        self._transport.status_received.connect(self._on_status)
        self._transport.status3_received.connect(self._on_status3)
        self._transport.multiturn_received.connect(self._on_multiturn_init)
        self._transport.fault_detected.connect(self._on_fault_detected)

        # Render timer
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._render_frame)
        self._render_timer.start()

    def showEvent(self, event):
        """Auto-read config values from MCU when tab becomes visible."""
        super().showEvent(event)
        if self._transport.is_connected():
            self._send_once(build_read_encoder())
            QTimer.singleShot(50, lambda: self._send_once(build_read_multi_turn_angle()))
            QTimer.singleShot(100, lambda: self._send_once(build_read_pid()))
            QTimer.singleShot(150, lambda: self._send_once(build_read_max_current()))

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # ═══ Top: Encoder | PID | Current | Diagnostics ═══
        top = QGridLayout()
        top.setSpacing(4)
        layout.addLayout(top)

        _grp_margin = (4, 2, 4, 2)  # left, top, right, bottom

        # ── Encoder Config ──
        enc_group = QGroupBox("Encoder Config")
        el = QVBoxLayout(enc_group)
        el.setContentsMargins(*_grp_margin)
        el.setSpacing(2)

        enc_btn_row = QHBoxLayout()
        read_enc_btn = QPushButton("Read Encoder")
        read_enc_btn.clicked.connect(lambda: self._send_once(build_read_encoder()))
        enc_btn_row.addWidget(read_enc_btn)
        read_mt_btn = QPushButton("Read Multi-Turn")
        read_mt_btn.clicked.connect(lambda: self._send_once(build_read_multi_turn_angle()))
        enc_btn_row.addWidget(read_mt_btn)
        el.addLayout(enc_btn_row)

        # Encoder readout: Pos | Offset | Multi-turn in one row
        lbl_style = "font-family: monospace; font-size: 11px; color: #66ff66;"
        name_style = "font-size: 11px; font-weight: bold;"

        info_row = QHBoxLayout()
        info_row.setSpacing(8)
        for name, attr in [("Pos:", "enc_lbl_pos"), ("Offset:", "enc_lbl_offset"), ("Multi-turn:", "enc_lbl_multiturn")]:
            lb = QLabel(name)
            lb.setStyleSheet(name_style)
            info_row.addWidget(lb)
            val = QLabel("--")
            val.setStyleSheet(lbl_style)
            val.setMinimumWidth(40)
            info_row.addWidget(val, stretch=1)
            setattr(self, attr, val)
        el.addLayout(info_row)

        # Offset write row: [spinbox] [Write Offset] [Write ROM]
        offset_row = QHBoxLayout()
        offset_row.setSpacing(4)
        ofs_label = QLabel("Offset:")
        ofs_label.setStyleSheet(name_style)
        offset_row.addWidget(ofs_label)
        self.enc_offset_spin = QSpinBox()
        self.enc_offset_spin.setRange(-16383, 16383)
        self.enc_offset_spin.setMaximumWidth(90)
        offset_row.addWidget(self.enc_offset_spin)
        write_offset_btn = QPushButton("Write Offset")
        write_offset_btn.clicked.connect(self._write_encoder_offset)
        offset_row.addWidget(write_offset_btn)
        offset_row.addSpacing(8)
        write_rom_btn = QPushButton("Write Current Pos to ROM as Motor Zero")
        write_rom_btn.setStyleSheet(
            "QPushButton { background-color: #884400; color: white; }"
            "QPushButton:pressed { background-color: #663300; }"
        )
        write_rom_btn.clicked.connect(self._write_pos_to_rom)
        offset_row.addWidget(write_rom_btn)
        el.addLayout(offset_row)
        top.addWidget(enc_group, 0, 0)

        # ── Position Control PID ──
        pid_group = QGroupBox("Position Control PID")
        pid_layout = QGridLayout(pid_group)
        pid_layout.setContentsMargins(*_grp_margin)
        pid_layout.setSpacing(2)
        for row_i, (name, attr) in enumerate([("Kp", "pid_kp"), ("Ki", "pid_ki"), ("Kd", "pid_kd")]):
            pid_layout.addWidget(QLabel(f"{name}:"), row_i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(0, 100)
            spin.setDecimals(5)
            spin.setSingleStep(0.001 if name == "Kp" else 0.00001)
            pid_layout.addWidget(spin, row_i, 1)
            setattr(self, attr, spin)
        pid_btn_row = QHBoxLayout()
        read_pid_btn = QPushButton("Read")
        read_pid_btn.clicked.connect(lambda: self._send_once(build_read_pid()))
        pid_btn_row.addWidget(read_pid_btn)
        write_pid_rom_btn = QPushButton("Write ROM")
        write_pid_rom_btn.setStyleSheet(
            "QPushButton { background-color: #884400; color: white; }"
            "QPushButton:pressed { background-color: #663300; }"
        )
        write_pid_rom_btn.clicked.connect(self._write_pid_rom)
        pid_btn_row.addWidget(write_pid_rom_btn)
        pid_layout.addLayout(pid_btn_row, 3, 0, 1, 2)
        top.addWidget(pid_group, 0, 1)

        # ── Current Setting ──
        curr_group = QGroupBox("Current Setting")
        cl = QVBoxLayout(curr_group)
        cl.setContentsMargins(*_grp_margin)
        cl.setSpacing(2)

        curr_grid = QGridLayout()
        curr_grid.setSpacing(2)
        curr_grid.addWidget(QLabel("OC Mode:"), 0, 0)
        self.curr_oc_mode = QSpinBox()
        self.curr_oc_mode.setRange(0, 10)
        self.curr_oc_mode.setValue(3)
        curr_grid.addWidget(self.curr_oc_mode, 0, 1)

        curr_grid.addWidget(QLabel("Motor MAX:"), 1, 0)
        self.curr_motor_max = QDoubleSpinBox()
        self.curr_motor_max.setRange(0, 200)
        self.curr_motor_max.setDecimals(1)
        self.curr_motor_max.setValue(60.0)
        curr_grid.addWidget(self.curr_motor_max, 1, 1)

        curr_grid.addWidget(QLabel("ABS MAX:"), 2, 0)
        self.curr_abs_max = QDoubleSpinBox()
        self.curr_abs_max.setRange(0, 200)
        self.curr_abs_max.setDecimals(1)
        self.curr_abs_max.setValue(100.0)
        curr_grid.addWidget(self.curr_abs_max, 2, 1)

        curr_grid.addWidget(QLabel("BAT MAX:"), 3, 0)
        self.curr_bat_max = QDoubleSpinBox()
        self.curr_bat_max.setRange(0, 200)
        self.curr_bat_max.setDecimals(1)
        self.curr_bat_max.setValue(60.0)
        curr_grid.addWidget(self.curr_bat_max, 3, 1)
        cl.addLayout(curr_grid)

        curr_btn_row = QHBoxLayout()
        read_curr_btn = QPushButton("Read")
        read_curr_btn.clicked.connect(lambda: self._send_once(build_read_max_current()))
        curr_btn_row.addWidget(read_curr_btn)
        write_curr_btn = QPushButton("Write ROM")
        write_curr_btn.setStyleSheet(
            "QPushButton { background-color: #884400; color: white; }"
            "QPushButton:pressed { background-color: #663300; }"
        )
        write_curr_btn.clicked.connect(self._write_max_current_rom)
        curr_btn_row.addWidget(write_curr_btn)
        cl.addLayout(curr_btn_row)
        top.addWidget(curr_group, 0, 2)

        # ── Fault Log ──
        diag_group = QGroupBox("Fault Log")
        diag_layout = QVBoxLayout(diag_group)
        diag_layout.setContentsMargins(*_grp_margin)
        diag_layout.setSpacing(2)

        fault_btn = QPushButton("Read Fault Code")
        fault_btn.clicked.connect(lambda: self._send_once(build_read_fault_code()))
        diag_layout.addWidget(fault_btn)

        # Fault log table (fills remaining height)
        self.fault_table = QTableWidget(0, 3)
        self.fault_table.setHorizontalHeaderLabels(["Time", "Motor ID", "Fault"])
        self.fault_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self.fault_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.fault_table.verticalHeader().setDefaultSectionSize(20)
        self.fault_table.verticalHeader().hide()
        diag_layout.addWidget(self.fault_table)

        top.addWidget(diag_group, 0, 3)

        # All groups same max height — Current (tallest content) is ~130px
        for grp in (enc_group, pid_group, curr_group, diag_group):
            grp.setMaximumHeight(140)

        top.setColumnStretch(0, 2)
        top.setColumnStretch(1, 2)
        top.setColumnStretch(2, 2)
        top.setColumnStretch(3, 3)

        # ═══ Graph controls bar ═══
        ctrl = QHBoxLayout()
        layout.addLayout(ctrl)

        self.start_btn = QPushButton("Start Polling")
        self.start_btn.clicked.connect(self.toggle_polling)
        ctrl.addWidget(self.start_btn)

        ctrl.addWidget(QLabel("Rate:"))
        self.rate_combo = QComboBox()
        for r in ["10 Hz", "20 Hz", "50 Hz", "100 Hz"]:
            self.rate_combo.addItem(r)
        self.rate_combo.setCurrentText("100 Hz")
        self.rate_combo.currentTextChanged.connect(self._on_rate_changed)
        ctrl.addWidget(self.rate_combo)

        self.csv_btn = QPushButton("Log to CSV")
        self.csv_btn.setCheckable(True)
        self.csv_btn.clicked.connect(self.toggle_csv)
        ctrl.addWidget(self.csv_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_data)
        ctrl.addWidget(self.clear_btn)

        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("Pole Pairs:"))
        self.pole_spin = QSpinBox()
        self.pole_spin.setRange(1, 50)
        self.pole_spin.setValue(21)
        self.pole_spin.setToolTip("Motor pole pairs (for eRPM calculation)")
        ctrl.addWidget(self.pole_spin)

        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("Graph Device:"))
        self._graph_device_combo = QComboBox()
        self._graph_device_combo.setMinimumWidth(80)
        self._graph_device_combo.addItem("All", "all")
        self._graph_device_combo.setToolTip(
            "All: overlay main curves from all devices\n"
            "Specific ID: full detail view for one device"
        )
        self._graph_device_combo.currentIndexChanged.connect(self._on_graph_device_changed)
        ctrl.addWidget(self._graph_device_combo)

        ctrl.addStretch()

        ctrl.addWidget(QLabel("X Range:"))
        self.xrange_combo = QComboBox()
        self.xrange_combo.addItems(["5 s", "10 s", "30 s", "60 s", "All"])
        self.xrange_combo.setCurrentText("10 s")
        self.xrange_combo.currentTextChanged.connect(self._on_xrange_changed)
        ctrl.addWidget(self.xrange_combo)

        self.autorange_chk = QCheckBox("Auto Y")
        self.autorange_chk.setChecked(True)
        self.autorange_chk.toggled.connect(self._on_autorange_toggled)
        ctrl.addWidget(self.autorange_chk)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(self._fit_all)
        ctrl.addWidget(self.fit_btn)

        self.raw_toggle_btn = QPushButton("Raw Data \u25bc")
        self.raw_toggle_btn.setCheckable(True)
        self.raw_toggle_btn.setChecked(False)
        self.raw_toggle_btn.clicked.connect(self._toggle_raw_panel)
        ctrl.addWidget(self.raw_toggle_btn)

        # ═══ Middle: Graphs + Raw Data ═══
        self._left_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self._left_splitter, stretch=1)

        # ── Graph area (2x2 grid) ──
        self._graph_widget = QWidget()
        graph_widget = self._graph_widget
        graph_outer = QVBoxLayout(graph_widget)
        graph_outer.setContentsMargins(0, 0, 0, 0)
        graph_outer.setSpacing(2)

        graph_grid = QGridLayout()
        graph_outer.addLayout(graph_grid, stretch=1)

        self.plot_mode = pg.PlotWidget()
        style_plot(self.plot_mode, title="Control Mode / Temp",
                   left_label="Mode", left_unit="")
        style_legend(self.plot_mode)
        self.curve_mode = self.plot_mode.plot(pen=graph_pen(1), name="Mode")

        # Right Y-axis for temperature (separate scale)
        self._temp_vb = pg.ViewBox()
        self.plot_mode.showAxis('right')
        self.plot_mode.getAxis('right').setLabel('Temp', units='°C')
        self.plot_mode.getAxis('right').setPen(pg.mkPen(color='#ff8888'))
        self.plot_mode.scene().addItem(self._temp_vb)
        self.plot_mode.getAxis('right').linkToView(self._temp_vb)
        self._temp_vb.setXLink(self.plot_mode)
        self.curve_temp = pg.PlotDataItem(pen=graph_pen(8), name="Temp(°C)")
        self._temp_vb.addItem(self.curve_temp)
        # Sync geometry on resize
        self.plot_mode.getViewBox().sigResized.connect(self._sync_temp_vb)

        # Add temp curve to legend manually
        legend = self.plot_mode.plotItem.legend
        if legend:
            legend.addItem(self.curve_temp, "Temp(°C)")

        graph_grid.addWidget(self.plot_mode, 0, 0)

        self.plot_torque = pg.PlotWidget()
        style_plot(self.plot_torque, title="Torque / Phase Current",
                   left_label="Current", left_unit="A")
        style_legend(self.plot_torque)
        self.curve_torque = self.plot_torque.plot(pen=graph_pen(0), name="Torque")
        self.curve_torque_cmd = self.plot_torque.plot(
            pen=graph_pen(4), name="Torque Cmd")
        self.curve_phase_a = self.plot_torque.plot(
            pen=graph_pen(5), name="PhA")
        self.curve_phase_b = self.plot_torque.plot(
            pen=graph_pen(6), name="PhB")
        self.curve_phase_c = self.plot_torque.plot(
            pen=graph_pen(7), name="PhC")
        # Hide phase currents by default (toggle via legend click)
        torque_legend = self.plot_torque.plotItem.legend
        set_curve_visible(torque_legend, self.curve_phase_a, False)
        set_curve_visible(torque_legend, self.curve_phase_b, False)
        set_curve_visible(torque_legend, self.curve_phase_c, False)
        graph_grid.addWidget(self.plot_torque, 0, 1)

        self.plot_speed = pg.PlotWidget()
        style_plot(self.plot_speed, title="Speed / RPM",
                   left_label="Value", left_unit="")
        style_legend(self.plot_speed)
        self.curve_speed = self.plot_speed.plot(pen=graph_pen(2), name="dps")
        self.curve_rpm = self.plot_speed.plot(pen=graph_pen(5), name="RPM")
        self.curve_erpm = self.plot_speed.plot(pen=graph_pen(6), name="eRPM")
        # Hide RPM/eRPM by default (toggle via legend click)
        speed_legend = self.plot_speed.plotItem.legend
        set_curve_visible(speed_legend, self.curve_rpm, False)
        set_curve_visible(speed_legend, self.curve_erpm, False)
        graph_grid.addWidget(self.plot_speed, 1, 0)

        self.plot_pos = pg.PlotWidget()
        style_plot(self.plot_pos, title="Encoder Position",
                   left_label="Position", left_unit="deg")
        style_legend(self.plot_pos)
        self.curve_pos = self.plot_pos.plot(pen=graph_pen(3), name="Position")
        self.curve_pos_cmd = self.plot_pos.plot(pen=graph_pen(4), name="Pos Cmd")
        graph_grid.addWidget(self.plot_pos, 1, 1)

        self._all_plots = [self.plot_mode, self.plot_torque, self.plot_speed, self.plot_pos]
        self._graph_grid = graph_grid
        self._graph_positions = {
            self.plot_mode: (0, 0), self.plot_torque: (0, 1),
            self.plot_speed: (1, 0), self.plot_pos: (1, 1),
        }

        # Optimize software rendering: downsample + clip to visible range
        self._all_curves = [
            self.curve_mode, self.curve_temp,
            self.curve_torque, self.curve_torque_cmd,
            self.curve_phase_a, self.curve_phase_b, self.curve_phase_c,
            self.curve_speed, self.curve_rpm, self.curve_erpm,
            self.curve_pos, self.curve_pos_cmd,
        ]
        for c in self._all_curves:
            c.setDownsampling(auto=True, method='peak')
            c.setClipToView(True)

        self._maximized_plot = None
        for pw in self._all_plots:
            pw.disableAutoRange()
            pw.setXRange(0, 10, padding=0)
            pw.installEventFilter(self)

        # Status label
        self.status_label = QLabel("Status: not polling")
        self.status_label.setStyleSheet("font-family: monospace; font-size: 12px; padding: 4px;")
        graph_outer.addWidget(self.status_label)

        self._left_splitter.addWidget(graph_widget)

        # ═══ Raw CAN Data panel (collapsed by default via splitter) ═══
        self._raw_panel = QWidget()
        raw_layout = QVBoxLayout(self._raw_panel)
        raw_layout.setContentsMargins(4, 4, 4, 4)
        raw_layout.setSpacing(2)

        raw_header = QHBoxLayout()
        raw_title = QLabel("Raw CAN Data")
        raw_title.setStyleSheet("font-weight: bold; color: #aaa;")
        raw_header.addWidget(raw_title)
        raw_header.addStretch()
        clear_raw_btn = QPushButton("Clear")
        clear_raw_btn.setMaximumWidth(60)
        clear_raw_btn.clicked.connect(self._clear_raw)
        raw_header.addWidget(clear_raw_btn)
        raw_layout.addLayout(raw_header)

        self.raw_table = QTableWidget()
        self.raw_table.setColumnCount(11)
        self.raw_table.setHorizontalHeaderLabels([
            "Time", "ID", "Len", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
        ])
        hdr = self.raw_table.horizontalHeader()
        for c in range(11):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.raw_table.setAlternatingRowColors(True)
        self.raw_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        raw_layout.addWidget(self.raw_table)

        self._left_splitter.addWidget(self._raw_panel)
        self._raw_visible = False
        # Collapse raw panel initially (all space to graphs)
        self._left_splitter.setSizes([1, 0])

    # ── Graph double-click maximize ──

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonDblClick and obj in self._all_plots:
            self._toggle_maximize_plot(obj)
            return True
        return super().eventFilter(obj, event)

    def _toggle_maximize_plot(self, pw):
        grid = self._graph_grid
        if self._maximized_plot is None:
            # Maximize: hide others, span clicked plot to 2x2
            for p in self._all_plots:
                grid.removeWidget(p)
                if p is not pw:
                    p.hide()
            grid.addWidget(pw, 0, 0, 2, 2)
            self._maximized_plot = pw
        else:
            # Restore 2x2 with equal sizes
            grid.removeWidget(self._maximized_plot)
            for p in self._all_plots:
                r, c = self._graph_positions[p]
                grid.addWidget(p, r, c)
                p.show()
            for i in range(2):
                grid.setRowStretch(i, 1)
                grid.setColumnStretch(i, 1)
            self._maximized_plot = None

    # ── Sync right Y-axis ViewBox geometry ──

    def _sync_temp_vb(self):
        self._temp_vb.setGeometry(self.plot_mode.getViewBox().sceneBoundingRect())

    # ── Raw Data panel toggle ──

    def _toggle_raw_panel(self, checked: bool):
        h = max(self._left_splitter.height(), 1)
        if checked:
            self._left_splitter.setSizes([0, h])
            self._raw_visible = True
            self.raw_toggle_btn.setText("Raw Data \u25b2")
        else:
            self._left_splitter.setSizes([h, 0])
            self._raw_visible = False
            self._dirty = True
            self.raw_toggle_btn.setText("Raw Data \u25bc")

    # ── Actions ──

    def _send_once(self, data: bytes):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Open PCAN connection first.")
            return
        self._transport.send_frame(data)

    def _on_fault_detected(self, motor_id: int, fault_code: int, fault_name: str):
        """Handle fault broadcast from firmware: auto-stop the faulted motor, log it."""
        # Send Motor Off to the faulted motor
        if self._transport.is_connected():
            self._transport.send_frame_to(motor_id, build_motor_off())

        # Add to fault log table
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        row = self.fault_table.rowCount()
        self.fault_table.insertRow(row)
        self.fault_table.setItem(row, 0, QTableWidgetItem(ts))
        self.fault_table.setItem(row, 1, QTableWidgetItem(str(motor_id)))
        fault_item = QTableWidgetItem(fault_name)
        fault_item.setForeground(QColor("#FF5252"))
        self.fault_table.setItem(row, 2, fault_item)
        self.fault_table.scrollToBottom()

    def update_discovered_ids(self, ids: list[int]):
        """Called from main_window when CAN scan finds motor IDs."""
        self._discovered_ids = list(ids)
        # Refresh Graph Device combo
        prev = self._graph_device_combo.currentData()
        self._graph_device_combo.blockSignals(True)
        self._graph_device_combo.clear()
        self._graph_device_combo.addItem("All", "all")
        for mid in sorted(ids):
            self._graph_device_combo.addItem(f"ID {mid}", mid)
        # Restore previous selection if still valid
        for i in range(self._graph_device_combo.count()):
            if self._graph_device_combo.itemData(i) == prev:
                self._graph_device_combo.setCurrentIndex(i)
                break
        self._graph_device_combo.blockSignals(False)
        self._dirty = True
        # Update running poller if active
        if self._poller and self._polling:
            self._poller.set_motor_ids(self._discovered_ids)
        # If polling start was pending on re-scan, start now with fresh IDs
        if self._pending_poll_start:
            self._pending_poll_start = False
            self._do_start_polling()

    def _write_encoder_offset(self):
        if not self._transport.is_connected():
            return
        self._transport.send_frame(build_write_encoder_offset(self.enc_offset_spin.value()))

    def _write_pos_to_rom(self):
        if not self._transport.is_connected():
            return
        reply = QMessageBox.question(
            self, "Confirm",
            "Write current position to ROM as motor zero?\nThis is permanent!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._transport.send_frame(build_write_current_pos_to_rom())

    def _write_max_current_rom(self):
        if not self._transport.is_connected():
            return
        reply = QMessageBox.question(
            self, "Confirm",
            "Write max current parameters to ROM?\nThis is permanent!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._transport.send_frame(build_write_max_current_to_rom(
                self.curr_oc_mode.value(), self.curr_motor_max.value(),
                self.curr_abs_max.value(), self.curr_bat_max.value(),
            ))


    def _write_pid_rom(self):
        if not self._transport.is_connected():
            return
        reply = QMessageBox.question(
            self, "Confirm",
            "Write PID parameters to ROM?\nThis is permanent!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._transport.send_frame(
                build_write_pid_to_rom(self.pid_kp.value(), self.pid_ki.value(), self.pid_kd.value())
            )

    # ── Frame handler: auto-fill + raw table ──

    def _on_frame_received(self, can_id: int, dlc: int, timestamp: float, data: list):
        if len(data) >= 8:
            cmd = data[0]
            # Auto-fill PID
            if cmd == RmdCommand.READ_PID:
                try:
                    pid = parse_pid(data)
                    self.pid_kp.setValue(pid.kp)
                    self.pid_ki.setValue(pid.ki)
                    self.pid_kd.setValue(pid.kd)
                except Exception:
                    pass
            # Auto-fill encoder readout
            elif cmd == RmdCommand.READ_ENCODER:
                try:
                    enc = parse_encoder(data)
                    self.enc_lbl_pos.setText(
                        f"{enc.enc_pos} ({enc.enc_pos_deg:.2f} deg)")
                    self.enc_lbl_offset.setText(
                        f"{enc.enc_pos_offset} ({enc.enc_pos_offset_deg:.2f} deg)")
                except Exception:
                    pass
            # Auto-fill max current
            elif cmd == RmdCommand.READ_MAX_CURRENT:
                try:
                    mc = parse_max_current(data)
                    self.curr_oc_mode.setValue(mc.drv8301_oc_mode)
                    self.curr_motor_max.setValue(mc.motor_current_max)
                    self.curr_abs_max.setValue(mc.motor_current_abs_max)
                    self.curr_bat_max.setValue(mc.bat_current_max)
                except Exception:
                    pass
            # Auto-fill multi-turn
            elif cmd == RmdCommand.READ_MULTI_TURN_ANGLE:
                try:
                    angle = parse_multi_turn_angle(data)
                    self.enc_lbl_multiturn.setText(f"{angle:.2f} deg")
                except Exception:
                    pass

        # Raw table (only when panel is visible)
        if self._raw_visible:
            try:
                row = self.raw_table.rowCount()
                if row >= 50:
                    self.raw_table.removeRow(0)
                    row = self.raw_table.rowCount()
                self.raw_table.insertRow(row)
                self.raw_table.setItem(row, 0, QTableWidgetItem(f"{timestamp:.3f}"))
                self.raw_table.setItem(row, 1, QTableWidgetItem(f"0x{can_id:03X}"))
                self.raw_table.setItem(row, 2, QTableWidgetItem(str(dlc)))
                for i in range(min(dlc, 8)):
                    self.raw_table.setItem(row, 3 + i, QTableWidgetItem(f"0x{data[i]:02X}"))
                self.raw_table.scrollToBottom()
            except Exception:
                pass

    def _clear_raw(self):
        self.raw_table.setRowCount(0)
        self._transport.clear_rxmsg()

    # ── Data ingestion (real-time graphs) ──

    def _on_torque_cmd(self, val: float):
        self._last_torque_cmd = val

    def _on_pos_cmd(self, val: float):
        self._last_pos_cmd = val

    def _on_multiturn_init(self, angle_deg: float):
        """Receive multiturn angle (0x92) for initial offset calibration."""
        if self._mt_init_angle is None:
            self._mt_init_angle = angle_deg

    def _get_or_create_buffers(self, motor_id: int) -> _DeviceBuffers:
        """Get existing or create new buffer set for a motor ID."""
        bufs = self._dev_buffers.get(motor_id)
        if bufs is None:
            bufs = _DeviceBuffers()
            self._dev_buffers[motor_id] = bufs
            # Auto-add to combo if not present
            found = False
            for i in range(self._graph_device_combo.count()):
                if self._graph_device_combo.itemData(i) == motor_id:
                    found = True
                    break
            if not found:
                self._graph_device_combo.addItem(f"ID {motor_id}", motor_id)
        return bufs

    def _on_status3(self, motor_id: int, status3: RmdStatus3):
        """Store latest 0x9D data (control mode + phase currents) per device."""
        bufs = self._get_or_create_buffers(motor_id)
        bufs.last_status3 = status3

    def _on_status(self, motor_id: int, status: RmdStatus):
        if self._t0 is None:
            self._t0 = time.time()

        bufs = self._get_or_create_buffers(motor_id)

        t = time.time() - self._t0
        bufs.time.append(t)
        bufs.torque.append(status.torque_curr)
        bufs.torque_cmd.append(self._last_torque_cmd)
        bufs.speed.append(float(status.speed_dps))

        # Use latest 0x9D data if available (per device)
        s3 = bufs.last_status3
        if s3 is not None:
            bufs.mode.append(float(s3.control_mode))
            bufs.phase_a.append(s3.phase_a)
            bufs.phase_b.append(s3.phase_b)
            bufs.phase_c.append(s3.phase_c)
        else:
            bufs.mode.append(0.0)
            bufs.phase_a.append(0.0)
            bufs.phase_b.append(0.0)
            bufs.phase_c.append(0.0)

        bufs.temp.append(float(status.motor_temp))

        # Unwrap encoder position (single-turn → continuous)
        raw_pos = status.enc_pos
        if bufs.prev_enc_pos is not None:
            delta = raw_pos - bufs.prev_enc_pos
            if delta < -180.0:
                bufs.enc_offset += 360.0
            elif delta > 180.0:
                bufs.enc_offset -= 360.0
        else:
            # First sample: calibrate offset from multiturn angle
            if self._mt_init_angle is not None:
                bufs.enc_offset = self._mt_init_angle - raw_pos
        bufs.prev_enc_pos = raw_pos
        bufs.pos.append(raw_pos + bufs.enc_offset)
        bufs.pos_cmd.append(self._last_pos_cmd)

        self._dirty = True
        self._last_status = status

        if self._csv_writer:
            mode = s3.control_mode if s3 else -1
            self._csv_writer.writerow([
                f"{t:.4f}", f"{status.motor_temp:.0f}", f"{mode}",
                f"{status.torque_curr:.3f}", f"{self._last_torque_cmd:.3f}",
                f"{status.speed_dps}", f"{status.enc_pos:.2f}",
            ])

    # ── Graph device mode switching ──

    def _on_graph_device_changed(self, _index: int):
        """Switch between overlay (All) and single-device modes."""
        self._dirty = True
        selected = self._graph_device_combo.currentData()
        if selected == "all":
            if not self._overlay_active:
                self._enter_overlay_mode()
        else:
            if self._overlay_active:
                self._exit_overlay_mode()

    def _enter_overlay_mode(self):
        """Hide single-device curves, prepare for dynamic overlay curves."""
        self._overlay_active = True
        # Hide all static single-device curves
        for c in self._all_curves:
            c.setData([], [])
            c.setVisible(False)
        self.curve_temp.setData([], [])
        self.curve_temp.setVisible(False)

    def _exit_overlay_mode(self):
        """Remove overlay curves, restore single-device static curves."""
        self._overlay_active = False
        # Remove all dynamic overlay curves
        for mid, curve_map in self._overlay_curves.items():
            for key, curve in curve_map.items():
                if key == "mode":
                    self.plot_mode.removeItem(curve)
                elif key == "torque":
                    self.plot_torque.removeItem(curve)
                elif key == "speed":
                    self.plot_speed.removeItem(curve)
                elif key == "pos":
                    self.plot_pos.removeItem(curve)
        self._overlay_curves.clear()
        # Remove overlay legend entries — rebuild legends
        for pw in self._all_plots:
            legend = pw.plotItem.legend
            if legend:
                legend.clear()
        # Re-add static curves to legends
        self.plot_mode.plotItem.legend.addItem(self.curve_mode, "Mode")
        self.plot_mode.plotItem.legend.addItem(self.curve_temp, "Temp(°C)")
        self.plot_torque.plotItem.legend.addItem(self.curve_torque, "Torque")
        self.plot_torque.plotItem.legend.addItem(self.curve_torque_cmd, "Torque Cmd")
        self.plot_torque.plotItem.legend.addItem(self.curve_phase_a, "PhA")
        self.plot_torque.plotItem.legend.addItem(self.curve_phase_b, "PhB")
        self.plot_torque.plotItem.legend.addItem(self.curve_phase_c, "PhC")
        self.plot_speed.plotItem.legend.addItem(self.curve_speed, "dps")
        self.plot_speed.plotItem.legend.addItem(self.curve_rpm, "RPM")
        self.plot_speed.plotItem.legend.addItem(self.curve_erpm, "eRPM")
        self.plot_pos.plotItem.legend.addItem(self.curve_pos, "Position")
        self.plot_pos.plotItem.legend.addItem(self.curve_pos_cmd, "Pos Cmd")
        # Show all static curves
        for c in self._all_curves:
            c.setVisible(True)
        self.curve_temp.setVisible(True)
        # Re-hide phase/RPM/eRPM by default
        torque_legend = self.plot_torque.plotItem.legend
        set_curve_visible(torque_legend, self.curve_phase_a, False)
        set_curve_visible(torque_legend, self.curve_phase_b, False)
        set_curve_visible(torque_legend, self.curve_phase_c, False)
        speed_legend = self.plot_speed.plotItem.legend
        set_curve_visible(speed_legend, self.curve_rpm, False)
        set_curve_visible(speed_legend, self.curve_erpm, False)

    def _get_overlay_curves(self, motor_id: int, dev_idx: int) -> dict:
        """Get or create overlay curves for a motor ID."""
        if motor_id in self._overlay_curves:
            return self._overlay_curves[motor_id]

        color = GRAPH_COLORS[dev_idx % len(GRAPH_COLORS)]
        pen = pg.mkPen(color, width=1.5)

        curves = {}
        curves["mode"] = self.plot_mode.plot(
            pen=pen, name=f"Mode (ID {motor_id})")
        curves["torque"] = self.plot_torque.plot(
            pen=pen, name=f"Torque (ID {motor_id})")
        curves["speed"] = self.plot_speed.plot(
            pen=pen, name=f"Speed (ID {motor_id})")
        curves["pos"] = self.plot_pos.plot(
            pen=pen, name=f"Pos (ID {motor_id})")

        for c in curves.values():
            c.setDownsampling(auto=True, method='peak')
            c.setClipToView(True)

        self._overlay_curves[motor_id] = curves
        return curves

    # ── Rendering ──

    def _render_frame(self):
        if not self._dirty or self._raw_visible:
            return
        self._dirty = False
        self._frame_count += 1

        try:
            selected = self._graph_device_combo.currentData()
            if selected == "all":
                self._render_overlay()
            else:
                self._render_single(selected)
        except Exception:
            pass  # never let rendering errors crash the app / PCAN

    def _render_single(self, motor_id):
        """Render full detail view for a single device (same as original)."""
        bufs = self._dev_buffers.get(motor_id)
        if bufs is None or len(bufs.time) == 0:
            return

        n = len(bufs.time)
        if n > BUF_COMPACT:
            bufs.compact(BUF_CAP)
            n = BUF_CAP

        x_all = bufs.time.array()
        t_now = x_all[n - 1]

        if self._x_window <= 0:
            t_min = x_all[0]
            s = 0
        else:
            t_min = max(x_all[0], t_now - self._x_window)
            s = int(np.searchsorted(x_all, t_min))

        x = x_all[s:]
        if len(x) == 0:
            return

        d_mode = bufs.mode.array()[s:]
        d_torque = bufs.torque.array()[s:]
        d_torque_cmd = bufs.torque_cmd.array()[s:]
        d_speed = bufs.speed.array()[s:]
        d_pos = bufs.pos.array()[s:]
        d_phase_a = bufs.phase_a.array()[s:]
        d_phase_b = bufs.phase_b.array()[s:]
        d_phase_c = bufs.phase_c.array()[s:]

        d_rpm = d_speed / 6.0
        pp = self.pole_spin.value()
        d_erpm = d_rpm * pp

        d_temp = bufs.temp.array()[s:]

        self.curve_mode.setData(x, d_mode, skipFiniteCheck=True)
        self.curve_temp.setData(x, d_temp, skipFiniteCheck=True)
        self.curve_torque.setData(x, d_torque, skipFiniteCheck=True)
        self.curve_torque_cmd.setData(x, d_torque_cmd, skipFiniteCheck=True)
        self.curve_phase_a.setData(x, d_phase_a, skipFiniteCheck=True)
        self.curve_phase_b.setData(x, d_phase_b, skipFiniteCheck=True)
        self.curve_phase_c.setData(x, d_phase_c, skipFiniteCheck=True)
        self.curve_speed.setData(x, d_speed, skipFiniteCheck=True)
        self.curve_rpm.setData(x, d_rpm, skipFiniteCheck=True)
        self.curve_erpm.setData(x, d_erpm, skipFiniteCheck=True)
        d_pos_cmd = bufs.pos_cmd.array()[s:]
        self.curve_pos.setData(x, d_pos, skipFiniteCheck=True)
        self.curve_pos_cmd.setData(x, d_pos_cmd, skipFiniteCheck=True)

        for pw in self._all_plots:
            pw.setXRange(t_min, t_now, padding=0)

        if self._auto_range:
            self._auto_range_single(
                d_mode, d_temp, d_torque, d_torque_cmd,
                d_phase_a, d_phase_b, d_phase_c,
                d_speed, d_rpm, d_erpm, d_pos, d_pos_cmd,
            )

        # Status bar
        if (self._frame_count & 7) == 0 and self._last_status:
            st = self._last_status
            s3 = bufs.last_status3
            mode_name = _CONTROL_MODE_NAMES.get(
                s3.control_mode, f"?({s3.control_mode})") if s3 else "N/A"
            self.status_label.setText(
                f"[ID {motor_id}]  Mode={mode_name}  Temp={st.motor_temp:.0f}C  "
                f"Torque={st.torque_curr:.3f}A  "
                f"Speed={st.speed_dps}dps  Pos={st.enc_pos:.2f}deg"
            )

    def _render_overlay(self):
        """Render overlay: main curves from all devices on each plot."""
        if not self._dev_buffers:
            return

        # Find global time range
        t_now = -1e30
        t_min_global = 1e30
        for bufs in self._dev_buffers.values():
            if len(bufs.time) > 0:
                arr = bufs.time.array()
                t_now = max(t_now, arr[-1])
                t_min_global = min(t_min_global, arr[0])
        if t_now < 0:
            return

        if self._x_window > 0:
            t_min = max(t_min_global, t_now - self._x_window)
        else:
            t_min = t_min_global

        # Accumulators for auto-range
        all_mode, all_torque, all_speed, all_pos = [], [], [], []

        sorted_ids = sorted(self._dev_buffers.keys())
        for dev_idx, mid in enumerate(sorted_ids):
            bufs = self._dev_buffers[mid]
            n = len(bufs.time)
            if n == 0:
                continue

            if n > BUF_COMPACT:
                bufs.compact(BUF_CAP)
                n = BUF_CAP

            x_all = bufs.time.array()
            s = int(np.searchsorted(x_all, t_min)) if self._x_window > 0 else 0
            x = x_all[s:]
            if len(x) == 0:
                continue

            d_mode = bufs.mode.array()[s:]
            d_torque = bufs.torque.array()[s:]
            d_speed = bufs.speed.array()[s:]
            d_pos = bufs.pos.array()[s:]

            ov = self._get_overlay_curves(mid, dev_idx)
            ov["mode"].setData(x, d_mode, skipFiniteCheck=True)
            ov["torque"].setData(x, d_torque, skipFiniteCheck=True)
            ov["speed"].setData(x, d_speed, skipFiniteCheck=True)
            ov["pos"].setData(x, d_pos, skipFiniteCheck=True)

            all_mode.append(d_mode)
            all_torque.append(d_torque)
            all_speed.append(d_speed)
            all_pos.append(d_pos)

        for pw in self._all_plots:
            pw.setXRange(t_min, t_now, padding=0)

        if self._auto_range and all_mode:
            def _yr(arrays):
                a = np.concatenate(arrays) if arrays else np.array([0.0])
                if len(a) == 0:
                    return -1, 1
                lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
                if not np.isfinite(lo):
                    lo = -1
                if not np.isfinite(hi):
                    hi = 1
                m = (hi - lo) * 0.05 if hi != lo else 1.0
                return lo - m, hi + m

            self.plot_mode.setYRange(*_yr(all_mode), padding=0)
            self.plot_torque.setYRange(*_yr(all_torque), padding=0)
            self.plot_speed.setYRange(*_yr(all_speed), padding=0)
            self.plot_pos.setYRange(*_yr(all_pos), padding=0)

        # Status bar: show device count
        if (self._frame_count & 7) == 0:
            n_dev = len(self._dev_buffers)
            total_pts = sum(len(b.time) for b in self._dev_buffers.values())
            self.status_label.setText(
                f"[All]  {n_dev} devices, {total_pts} samples total"
            )

    def _auto_range_single(self, d_mode, d_temp, d_torque, d_torque_cmd,
                            d_phase_a, d_phase_b, d_phase_c,
                            d_speed, d_rpm, d_erpm, d_pos, d_pos_cmd):
        """Auto Y-range for single-device mode."""
        def _yr(a):
            if len(a) == 0:
                return -1, 1
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            if not np.isfinite(lo):
                lo = -1
            if not np.isfinite(hi):
                hi = 1
            m = (hi - lo) * 0.05 if hi != lo else 1.0
            return lo - m, hi + m

        self.plot_mode.setYRange(*_yr(d_mode), padding=0)
        if len(d_temp) > 0:
            lo, hi = float(np.nanmin(d_temp)), float(np.nanmax(d_temp))
            if np.isfinite(lo) and np.isfinite(hi):
                m = (hi - lo) * 0.05 if hi != lo else 2.0
                self._temp_vb.setYRange(lo - m, hi + m, padding=0)
        curr_all = np.concatenate([d_torque, d_torque_cmd,
                                   d_phase_a, d_phase_b, d_phase_c])
        self.plot_torque.setYRange(*_yr(curr_all), padding=0)
        speed_all = np.concatenate([d_speed, d_rpm, d_erpm])
        self.plot_speed.setYRange(*_yr(speed_all), padding=0)
        valid_cmd = d_pos_cmd[np.isfinite(d_pos_cmd)]
        if len(valid_cmd) > 0:
            pos_all = np.concatenate([d_pos, valid_cmd])
        else:
            pos_all = d_pos
        self.plot_pos.setYRange(*_yr(pos_all), padding=0)

    # ── Polling controls ──

    def toggle_polling(self):
        if self._polling or self._pending_poll_start:
            self.stop_polling()
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start Polling")
        else:
            self.start_polling()

    def start_polling(self):
        if not self._transport.is_connected():
            return
        if self._polling or self._pending_poll_start:
            return
        if not self._discovered_ids:
            # No IDs known yet — re-scan first, then start polling
            self._pending_poll_start = True
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Scanning...")
            self.status_label.setText("Status: Scanning for active devices...")
            self.status_label.setStyleSheet(
                "font-family: monospace; font-size: 12px; padding: 4px; color: #ffaa00;"
            )
            self.rescan_requested.emit()
        else:
            # IDs already known from previous scan — start immediately
            self._do_start_polling()

    def _do_start_polling(self):
        """Actually start the polling loop (called after re-scan completes)."""
        self.start_btn.setEnabled(True)
        if not self._transport.is_connected():
            self.start_btn.setText("Start Polling")
            return
        if self._polling:
            return
        # Request multiturn angle for initial offset calibration
        self._mt_init_angle = None
        self._transport.send_frame(build_read_multi_turn_angle())
        self._t0 = time.time()
        rate_ms = int(1000 / int(self.rate_combo.currentText().replace(" Hz", "")))
        self._poller = CanPoller(self._transport, rate_ms,
                                 motor_ids=self._discovered_ids)
        self._poller.start()
        self._polling = True
        self.start_btn.setText("Stop Polling")
        n = len(self._discovered_ids)
        id_str = ", ".join(str(i) for i in self._discovered_ids) if n else "single target"
        self.status_label.setText(f"Status: Polling {n} device(s) [{id_str}]")
        self.status_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; padding: 4px; color: #66ff66;"
        )

    def stop_polling(self):
        self._pending_poll_start = False
        if self._poller:
            self._poller.stop()
            # Non-blocking wait: keep Qt event loop alive so queued signals
            # from the PCAN reader thread can drain normally instead of
            # piling up and causing a burst when wait() returns.
            deadline = time.time() + 2.0
            while self._poller.isRunning() and time.time() < deadline:
                QApplication.processEvents()
                time.sleep(0.005)
            if self._poller.isRunning():
                self._poller.terminate()
                self._poller.wait(500)
            self._poller.deleteLater()
            self._poller = None
        self._polling = False
        self.start_btn.setText("Start Polling")
        self.status_label.setText("Status: Polling stopped")
        self.status_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; padding: 4px; color: #ffaa00;"
        )

    def _on_rate_changed(self, text: str):
        if self._poller and self._polling:
            self._poller.interval_ms = int(1000 / int(text.replace(" Hz", "")))

    def _on_xrange_changed(self, text: str):
        self._x_window = 0.0 if text == "All" else float(text.replace(" s", ""))
        self._dirty = True

    def _on_autorange_toggled(self, checked: bool):
        self._auto_range = checked
        if checked:
            self._dirty = True

    def _fit_all(self):
        old = self._x_window
        self._x_window = 0.0
        self._dirty = True
        self._render_frame()
        self._x_window = old

    def toggle_csv(self, checked: bool):
        if checked:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", f"can_log_{ts}.csv", "CSV (*.csv)"
            )
            if path:
                self._csv_file = open(path, "w", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow([
                    "time_s", "motor_temp_C", "control_mode",
                    "torque_curr_A", "torque_cmd_A",
                    "speed_dps", "enc_pos_deg",
                ])
            else:
                self.csv_btn.setChecked(False)
        else:
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def clear_data(self):
        for bufs in self._dev_buffers.values():
            bufs.clear()
        self._t0 = time.time() if self._polling else None
        self._mt_init_angle = None
        self._last_pos_cmd = float('nan')
        self._dirty = True

    def set_pole_pairs(self, value: int):
        """Set pole pairs from external source (e.g. mcconf foc_encoder_ratio)."""
        if 1 <= value <= 50:
            self.pole_spin.setValue(value)

    def cleanup(self):
        self._render_timer.stop()
        self.stop_polling()
        if self._csv_file:
            self._csv_file.close()
