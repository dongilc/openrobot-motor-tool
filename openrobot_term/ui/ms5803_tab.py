"""
MS5803 External Sensor tab — real-time temperature / pressure monitoring.
Polls READ_EXT_SENSOR (0xD0) via CAN → RS485 → XIAO RA4M1 → MS5803.

Layout:
  Top    — Read / Start / Stop buttons, poll rate, window selector, status
  Middle — Current sensor values (Temperature, Pressure, ATM, Valid)
  Bottom — Dual-axis pyqtgraph (Temperature left, Pressure right)
"""

import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout,
)
from PyQt6.QtCore import QTimer, pyqtSlot
from PyQt6.QtGui import QFont

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import build_read_ext_sensor, ExtSensorData
from .plot_style import style_plot, graph_pen, style_legend, COLORS, TEXT_NORMAL


BUF_CAP = 6000  # ~10 min at 10 Hz


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


class Ms5803Tab(QWidget):
    def __init__(self, transport: PcanTransport):
        super().__init__()
        self._transport = transport
        self._polling = False
        self._t0 = 0.0
        self._send_err_cnt = 0

        # Time-series buffers
        self._time = _GrowBuffer(BUF_CAP)
        self._temp = _GrowBuffer(BUF_CAP)
        self._pres = _GrowBuffer(BUF_CAP)
        self._atm = _GrowBuffer(BUF_CAP)

        self._build_ui()
        self._connect_signals()

        # Poll timer (sends 0xD0 request)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_tick)

        # Render timer (30 fps graph update)
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_frame)
        self._dirty = False

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Description ──
        desc = QLabel(
            "VESCAT HIGH V2R3 — MS5803 pressure/temperature monitoring "
            "(CAN \u2192 RS485 \u2192 XIAO RA4M1 \u2192 MS5803)"
        )
        desc.setStyleSheet(f"color: {TEXT_NORMAL}; font-size: 10pt; padding: 2px 0px;")
        root.addWidget(desc)

        # ── Controls row ──
        ctrl = QHBoxLayout()

        self._btn_read = QPushButton("Read")
        self._btn_read.setFixedWidth(70)
        self._btn_read.clicked.connect(self._on_read_once)
        ctrl.addWidget(self._btn_read)

        self._btn_start = QPushButton("Start")
        self._btn_start.setFixedWidth(70)
        self._btn_start.clicked.connect(self._on_start)
        ctrl.addWidget(self._btn_start)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setFixedWidth(70)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        ctrl.addWidget(self._btn_stop)

        ctrl.addWidget(QLabel("Rate:"))
        self._combo_rate = QComboBox()
        self._combo_rate.addItems(["1 Hz", "2 Hz", "5 Hz", "10 Hz"])
        self._combo_rate.setCurrentIndex(2)  # default 5 Hz
        self._combo_rate.setFixedWidth(80)
        ctrl.addWidget(self._combo_rate)

        ctrl.addWidget(QLabel("Window:"))
        self._combo_window = QComboBox()
        self._combo_window.addItems(["30 s", "60 s", "120 s", "All"])
        self._combo_window.setCurrentIndex(1)
        self._combo_window.setFixedWidth(80)
        ctrl.addWidget(self._combo_window)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedWidth(60)
        self._btn_clear.clicked.connect(self._on_clear)
        ctrl.addWidget(self._btn_clear)

        self._lbl_status = QLabel("")
        self._lbl_status.setFont(QFont("Consolas", 10))
        ctrl.addWidget(self._lbl_status)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ── Status labels ──
        status_group = QGroupBox("Current Values")
        sg = QGridLayout(status_group)
        sg.setContentsMargins(8, 4, 8, 4)

        label_font = QFont("Consolas", 12)
        bold_font = QFont("Consolas", 12, QFont.Weight.Bold)

        self._lbl_valid = QLabel("--")
        self._lbl_valid.setFont(bold_font)
        self._lbl_temperature = QLabel("--")
        self._lbl_temperature.setFont(label_font)
        self._lbl_pressure = QLabel("--")
        self._lbl_pressure.setFont(label_font)
        self._lbl_atm = QLabel("--")
        self._lbl_atm.setFont(label_font)

        sg.addWidget(QLabel("Valid:"), 0, 0)
        sg.addWidget(self._lbl_valid, 0, 1)
        sg.addWidget(QLabel("Temperature:"), 0, 2)
        sg.addWidget(self._lbl_temperature, 0, 3)
        sg.addWidget(QLabel("Pressure:"), 0, 4)
        sg.addWidget(self._lbl_pressure, 0, 5)
        sg.addWidget(QLabel("ATM:"), 0, 6)
        sg.addWidget(self._lbl_atm, 0, 7)
        sg.setColumnStretch(8, 1)

        root.addWidget(status_group)

        # ── Graph: Temperature (left axis) + Pressure (right axis) ──
        self._pw = pg.PlotWidget()
        style_plot(self._pw, title="MS5803 Sensor Data",
                   left_label="Temperature", left_unit="\u00b0C",
                   bottom_label="Time", bottom_unit="s")

        # Legend (click-to-toggle)
        self._legend = style_legend(self._pw)

        self._curve_temp = self._pw.plot(
            pen=graph_pen(0, width=2.0), name="Temperature")

        # Right axis for pressure
        self._vb_pres = pg.ViewBox()
        plot_item = self._pw.getPlotItem()
        plot_item.scene().addItem(self._vb_pres)
        plot_item.getAxis('right').linkToView(self._vb_pres)
        self._vb_pres.setXLink(self._pw)
        ax_right = plot_item.getAxis('right')
        ax_right.setLabel("Pressure", units="mbar", **{"color": COLORS["red"]})
        ax_right.show()

        self._curve_pres = pg.PlotDataItem(
            pen=graph_pen(1, width=2.0), name="Pressure")
        self._vb_pres.addItem(self._curve_pres)
        # Add pressure curve to legend manually (it's in a separate ViewBox)
        self._legend.addItem(self._curve_pres, "Pressure")

        # Mouse wheel: Y-axis zoom only, X-axis auto-scrolls with data
        self._pw.setMouseEnabled(x=False, y=True)
        self._vb_pres.setMouseEnabled(x=False, y=True)

        # Sync viewbox geometry on resize
        plot_item.vb.sigResized.connect(self._sync_viewbox)

        root.addWidget(self._pw, stretch=1)

    def _sync_viewbox(self):
        self._vb_pres.setGeometry(
            self._pw.getPlotItem().vb.sceneBoundingRect())

    # ── Signals ───────────────────────────────────────────────────────

    def _connect_signals(self):
        self._transport.ext_sensor_received.connect(self._on_ext_sensor_data)

    # ── Controls ──────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_read_once(self):
        if not self._transport.is_connected():
            self._lbl_status.setText("Not connected")
            self._lbl_status.setStyleSheet("color: #C83434;")
            return
        self._transport.send_frame(build_read_ext_sensor())

    @pyqtSlot()
    def _on_start(self):
        if self._polling:
            return
        if not self._transport.is_connected():
            self._lbl_status.setText("Not connected")
            self._lbl_status.setStyleSheet("color: #C83434;")
            return

        self._polling = True
        self._send_err_cnt = 0
        self._t0 = time.monotonic()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_read.setEnabled(False)
        self._combo_rate.setEnabled(False)

        hz = self._get_rate_hz()
        self._poll_timer.start(int(1000 / hz))
        self._render_timer.start(33)  # 30 fps

        self._lbl_status.setText(f"Polling {hz} Hz")
        self._lbl_status.setStyleSheet("color: #4FCBCB;")

    @pyqtSlot()
    def _on_stop(self):
        self._polling = False
        self._poll_timer.stop()
        self._render_timer.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_read.setEnabled(True)
        self._combo_rate.setEnabled(True)
        self._lbl_status.setText("Stopped")
        self._lbl_status.setStyleSheet("color: #B4B4B4;")

    @pyqtSlot()
    def _on_clear(self):
        for buf in (self._time, self._temp, self._pres, self._atm):
            buf.clear()
        self._curve_temp.setData([], [])
        self._curve_pres.setData([], [])
        for lbl in (self._lbl_valid, self._lbl_temperature,
                    self._lbl_pressure, self._lbl_atm):
            lbl.setText("--")
        self._lbl_valid.setStyleSheet("")

    # ── Polling ───────────────────────────────────────────────────────

    def _poll_tick(self):
        if not self._transport.is_connected():
            return
        ok = self._transport.send_frame(build_read_ext_sensor())
        if ok:
            self._send_err_cnt = 0
        else:
            self._send_err_cnt += 1
            if self._send_err_cnt >= 5:
                self._on_stop()
                self._lbl_status.setText("CAN Error")
                self._lbl_status.setStyleSheet("color: #C83434;")

    # ── Data handler ──────────────────────────────────────────────────

    @pyqtSlot(int, object)
    def _on_ext_sensor_data(self, motor_id: int, sensor: ExtSensorData):
        # Update labels
        if sensor.valid:
            self._lbl_valid.setText("OK")
            self._lbl_valid.setStyleSheet("color: #7FC87F;")
        else:
            self._lbl_valid.setText("INVALID")
            self._lbl_valid.setStyleSheet("color: #C83434; font-weight: bold;")

        self._lbl_temperature.setText(f"{sensor.temperature:.2f} \u00b0C")
        self._lbl_pressure.setText(f"{sensor.pressure:.0f} mbar")
        self._lbl_atm.setText(f"{sensor.atm:.4f}")

        # Only append to graph buffers if valid
        if not sensor.valid:
            return

        t = time.monotonic() - self._t0 if self._polling else 0.0
        self._time.append(t)
        self._temp.append(sensor.temperature)
        self._pres.append(sensor.pressure)
        self._atm.append(sensor.atm)

        # Compact if over capacity
        if len(self._time) > BUF_CAP:
            for buf in (self._time, self._temp, self._pres, self._atm):
                buf.compact(BUF_CAP)

        self._dirty = True

    # ── Render ────────────────────────────────────────────────────────

    def _render_frame(self):
        if not self._dirty:
            return
        self._dirty = False

        ta = self._time.array()
        if len(ta) == 0:
            return

        window = self._get_window_sec()
        if window > 0:
            t_min = ta[-1] - window
            mask = ta >= t_min
            tx = ta[mask]
            self._curve_temp.setData(tx, self._temp.array()[mask])
            self._curve_pres.setData(tx, self._pres.array()[mask])
        else:
            self._curve_temp.setData(ta, self._temp.array())
            self._curve_pres.setData(ta, self._pres.array())

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_rate_hz(self) -> int:
        text = self._combo_rate.currentText()
        return int(text.split()[0])

    def _get_window_sec(self) -> int:
        text = self._combo_window.currentText()
        if "All" in text:
            return 0
        return int(text.split()[0])

    def cleanup(self):
        self._on_stop()
