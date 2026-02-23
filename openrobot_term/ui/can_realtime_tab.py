"""
CAN Real-time data monitoring tab — 4 pyqtgraph plots for RMD motor telemetry.
Mirrors RealtimeTab pattern with simpler data channels.
"""

import time
import csv
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QCheckBox, QSpinBox,
)
from PyQt6.QtCore import Qt, QTimer

import numpy as np
import pyqtgraph as pg

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import RmdStatus
from ..workers.can_poller import CanPoller
from .plot_style import style_plot, graph_pen, style_legend

RENDER_INTERVAL_MS = 33  # ~30 fps
BUF_CAP = 12000
BUF_COMPACT = 24000


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


class CanRealtimeTab(QWidget):
    def __init__(self, can_transport: PcanTransport):
        super().__init__()
        self._transport = can_transport
        self._poller = None
        self._t0 = None
        self._csv_file = None
        self._csv_writer = None
        self._polling = False
        self._auto_range = True
        self._x_window = 10.0
        self._dirty = False
        self._last_status = None
        self._frame_count = 0

        # Buffers
        self._rb_time = _GrowBuffer(BUF_CAP)
        self._rb_temp = _GrowBuffer(BUF_CAP)
        self._rb_torque = _GrowBuffer(BUF_CAP)
        self._rb_speed = _GrowBuffer(BUF_CAP)
        self._rb_pos = _GrowBuffer(BUF_CAP)

        self._all_rbs = [
            self._rb_time, self._rb_temp, self._rb_torque,
            self._rb_speed, self._rb_pos,
        ]

        self._build_ui()

        # Wire transport signal
        self._transport.status_received.connect(self._on_status)

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._render_frame)
        self._render_timer.start()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()
        layout.addLayout(ctrl)

        self.start_btn = QPushButton("Start Polling")
        self.start_btn.clicked.connect(self.toggle_polling)
        ctrl.addWidget(self.start_btn)

        ctrl.addWidget(QLabel("Rate:"))
        self.rate_combo = QComboBox()
        for r in ["10 Hz", "20 Hz", "50 Hz", "100 Hz"]:
            self.rate_combo.addItem(r)
        self.rate_combo.setCurrentText("50 Hz")
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
        self.pole_spin.setValue(7)
        self.pole_spin.setToolTip("Motor pole pairs (for eRPM calculation)")
        ctrl.addWidget(self.pole_spin)

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

        # ── 4 plots in 2x2 grid ──
        graph_grid = QGridLayout()
        layout.addLayout(graph_grid, stretch=1)

        self.plot_temp = pg.PlotWidget()
        style_plot(self.plot_temp, title="Motor Temperature",
                   left_label="Temperature", left_unit="°C")
        style_legend(self.plot_temp)
        self.curve_temp = self.plot_temp.plot(pen=graph_pen(1), name="Temp")
        graph_grid.addWidget(self.plot_temp, 0, 0)

        self.plot_torque = pg.PlotWidget()
        style_plot(self.plot_torque, title="Torque Current",
                   left_label="Current", left_unit="A")
        style_legend(self.plot_torque)
        self.curve_torque = self.plot_torque.plot(pen=graph_pen(0), name="Torque")
        graph_grid.addWidget(self.plot_torque, 0, 1)

        self.plot_speed = pg.PlotWidget()
        style_plot(self.plot_speed, title="Speed / RPM",
                   left_label="Value", left_unit="")
        style_legend(self.plot_speed)
        self.curve_speed = self.plot_speed.plot(pen=graph_pen(2), name="dps")
        self.curve_rpm = self.plot_speed.plot(pen=graph_pen(5), name="RPM")
        self.curve_erpm = self.plot_speed.plot(pen=graph_pen(6), name="eRPM")
        graph_grid.addWidget(self.plot_speed, 1, 0)

        self.plot_pos = pg.PlotWidget()
        style_plot(self.plot_pos, title="Encoder Position",
                   left_label="Position", left_unit="deg")
        style_legend(self.plot_pos)
        self.curve_pos = self.plot_pos.plot(pen=graph_pen(3), name="Position")
        graph_grid.addWidget(self.plot_pos, 1, 1)

        self._all_plots = [self.plot_temp, self.plot_torque, self.plot_speed, self.plot_pos]
        for pw in self._all_plots:
            pw.disableAutoRange()
            pw.setXRange(0, 10, padding=0)

        # Status bar
        self.status_label = QLabel("Status: not polling")
        self.status_label.setStyleSheet("font-family: monospace; font-size: 12px; padding: 4px;")
        layout.addWidget(self.status_label)

    # ── Data ingestion ──

    def _on_status(self, motor_id: int, status: RmdStatus):
        if self._t0 is None:
            self._t0 = time.time()

        t = time.time() - self._t0
        self._rb_time.append(t)
        self._rb_temp.append(status.motor_temp)
        self._rb_torque.append(status.torque_curr)
        self._rb_speed.append(float(status.speed_dps))
        self._rb_pos.append(status.enc_pos)

        self._dirty = True
        self._last_status = status

        if self._csv_writer:
            self._csv_writer.writerow([
                f"{t:.4f}", f"{status.motor_temp:.1f}",
                f"{status.torque_curr:.3f}", f"{status.speed_dps}",
                f"{status.enc_pos:.2f}",
            ])

    # ── Rendering ──

    def _render_frame(self):
        if not self._dirty:
            return
        self._dirty = False
        self._frame_count += 1

        n = len(self._rb_time)
        if n == 0:
            return

        if n > BUF_COMPACT:
            for rb in self._all_rbs:
                rb.compact(BUF_CAP)
            n = BUF_CAP

        x_all = self._rb_time.array()
        t_now = x_all[n - 1]

        if self._x_window <= 0:
            t_min = x_all[0]
            s = 0
        else:
            t_min = max(x_all[0], t_now - self._x_window)
            s = int(np.searchsorted(x_all, t_min))

        x = x_all[s:]
        d_temp = self._rb_temp.array()[s:]
        d_torque = self._rb_torque.array()[s:]
        d_speed = self._rb_speed.array()[s:]
        d_pos = self._rb_pos.array()[s:]

        d_rpm = d_speed / 6.0
        pp = self.pole_spin.value()
        d_erpm = d_rpm * pp

        self.curve_temp.setData(x, d_temp, skipFiniteCheck=True)
        self.curve_torque.setData(x, d_torque, skipFiniteCheck=True)
        self.curve_speed.setData(x, d_speed, skipFiniteCheck=True)
        self.curve_rpm.setData(x, d_rpm, skipFiniteCheck=True)
        self.curve_erpm.setData(x, d_erpm, skipFiniteCheck=True)
        self.curve_pos.setData(x, d_pos, skipFiniteCheck=True)

        if len(x) > 0:
            for pw in self._all_plots:
                pw.setXRange(t_min, t_now, padding=0)

        if self._auto_range and len(x) > 0:
            def _yr(a):
                lo, hi = float(a.min()), float(a.max())
                m = (hi - lo) * 0.05 if hi != lo else 1.0
                return lo - m, hi + m

            self.plot_temp.setYRange(*_yr(d_temp), padding=0)
            self.plot_torque.setYRange(*_yr(d_torque), padding=0)
            # Speed plot Y-range covers all 3 curves (dps, RPM, eRPM)
            speed_all = np.concatenate([d_speed, d_rpm, d_erpm])
            self.plot_speed.setYRange(*_yr(speed_all), padding=0)
            self.plot_pos.setYRange(*_yr(d_pos), padding=0)

        # Status bar
        if (self._frame_count & 7) == 0 and self._last_status:
            s = self._last_status
            self.status_label.setText(
                f"Temp={s.motor_temp:.0f}°C  Torque={s.torque_curr:.3f}A  "
                f"Speed={s.speed_dps}dps  Pos={s.enc_pos:.2f}deg"
            )

    # ── Controls ──

    def toggle_polling(self):
        if self._polling:
            self.stop_polling()
        else:
            self.start_polling()

    def start_polling(self):
        if not self._transport.is_connected():
            return
        self._t0 = time.time()
        rate_ms = int(1000 / int(self.rate_combo.currentText().replace(" Hz", "")))
        self._poller = CanPoller(self._transport, rate_ms)
        self._poller.start()
        self._polling = True
        self.start_btn.setText("Stop Polling")
        self.status_label.setText("Status: Polling started...")
        self.status_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; padding: 4px; color: #66ff66;"
        )

    def stop_polling(self):
        if self._poller:
            self._poller.stop()
            self._poller.wait(2000)
            if self._poller.isRunning():
                self._poller.terminate()
                self._poller.wait(500)
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
                    "time_s", "motor_temp_C", "torque_curr_A",
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
        for rb in self._all_rbs:
            rb.clear()
        self._t0 = time.time() if self._polling else None
        self._dirty = True

    def cleanup(self):
        self._render_timer.stop()
        self.stop_polling()
        if self._csv_file:
            self._csv_file.close()
