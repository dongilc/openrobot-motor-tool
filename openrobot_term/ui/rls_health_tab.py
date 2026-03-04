"""
RLS Encoder Health monitoring tab — real-time airgap/signal/error graphs.
Polls READ_ENCODER_STATUS (0x9E) + READ_ENCODER (0x90) via independent QTimer.
Scan 360: slow DPS rotation to collect full airgap profile.

Layout:
  Left  — Airgap vs Time (top), Signal/SPI Error vs Time (bottom)
  Right — Airgap vs Angle (top), Error/Warning vs Angle (bottom)
"""

import time
import csv

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QGroupBox, QGridLayout, QFileDialog,
    QSplitter, QSpinBox,
)
from PyQt6.QtCore import QTimer, pyqtSlot, Qt
from PyQt6.QtGui import QFont

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import (
    build_read_encoder_status, build_read_encoder, RmdEncoderHealth, RmdEncoder,
    build_speed_closed_loop, build_motor_off,
)
from .plot_style import style_plot, graph_pen, COLORS, BG_DARK, TEXT_LIGHT, TEXT_NORMAL


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


class RlsHealthTab(QWidget):
    def __init__(self, transport: PcanTransport):
        super().__init__()
        self._transport = transport
        self._polling = False
        self._t0 = 0.0

        # Time-series buffers (left graphs — always in sync)
        self._time = _GrowBuffer(BUF_CAP)
        self._airgap = _GrowBuffer(BUF_CAP)
        self._signal = _GrowBuffer(BUF_CAP)
        self._spi_err = _GrowBuffer(BUF_CAP)
        self._last_enc_deg = 0.0

        # Angle-domain buffers (right graphs — cleared on each scan)
        self._ang_deg = _GrowBuffer(BUF_CAP)
        self._ang_air = _GrowBuffer(BUF_CAP)
        self._err_ang = _GrowBuffer(BUF_CAP)
        self._err_y = _GrowBuffer(BUF_CAP)
        self._wrn_ang = _GrowBuffer(BUF_CAP)
        self._wrn_y = _GrowBuffer(BUF_CAP)

        # Scan state
        self._scanning = False
        self._scan_prev_deg = 0.0
        self._scan_accum_deg = 0.0
        self._scan_target_deg = 360.0
        self._scan_phase = False  # alternates health/encoder during scan
        self._send_err_cnt = 0    # consecutive send-error counter

        # CSV logging
        self._csv_file = None
        self._csv_writer = None

        self._build_ui()
        self._connect_signals()

        # Poll timer
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_tick)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Controls: Monitor | Scan (single row) ──
        ctrl = QHBoxLayout()

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
        self._combo_rate.addItems(["5 Hz", "10 Hz"])
        self._combo_rate.setCurrentIndex(1)
        self._combo_rate.setFixedWidth(80)
        ctrl.addWidget(self._combo_rate)

        ctrl.addWidget(QLabel("Window:"))
        self._combo_window = QComboBox()
        self._combo_window.addItems(["10 s", "30 s", "60 s", "All"])
        self._combo_window.setCurrentIndex(1)
        self._combo_window.setFixedWidth(80)
        ctrl.addWidget(self._combo_window)

        self._chk_csv = QCheckBox("CSV Log")
        ctrl.addWidget(self._chk_csv)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedWidth(60)
        self._btn_clear.clicked.connect(self._on_clear)
        ctrl.addWidget(self._btn_clear)

        # ── separator ──
        sep = QLabel("|")
        sep.setStyleSheet("color: #666; font-size: 16px;")
        ctrl.addWidget(sep)

        self._btn_scan = QPushButton("Scan 360\u00b0")
        self._btn_scan.setFixedWidth(90)
        self._btn_scan.clicked.connect(self._on_scan_start)
        ctrl.addWidget(self._btn_scan)

        self._btn_scan_stop = QPushButton("Stop Scan")
        self._btn_scan_stop.setFixedWidth(80)
        self._btn_scan_stop.setEnabled(False)
        self._btn_scan_stop.clicked.connect(self._on_scan_stop)
        ctrl.addWidget(self._btn_scan_stop)

        ctrl.addWidget(QLabel("DPS:"))
        self._spin_dps = QSpinBox()
        self._spin_dps.setRange(5, 120)
        self._spin_dps.setValue(20)
        self._spin_dps.setSuffix(" \u00b0/s")
        self._spin_dps.setFixedWidth(120)
        ctrl.addWidget(self._spin_dps)

        ctrl.addWidget(QLabel("Revs:"))
        self._combo_revs = QComboBox()
        self._combo_revs.addItems(["1", "2", "3", "4", "5"])
        self._combo_revs.setCurrentIndex(1)
        self._combo_revs.setFixedWidth(50)
        ctrl.addWidget(self._combo_revs)

        self._lbl_scan_status = QLabel("")
        self._lbl_scan_status.setFont(QFont("Consolas", 10))
        ctrl.addWidget(self._lbl_scan_status)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ── Status labels ──
        status_group = QGroupBox("Current Values")
        sg = QGridLayout(status_group)
        sg.setContentsMargins(8, 4, 8, 4)

        label_font = QFont("Consolas", 11)
        self._lbl_airgap = QLabel("--")
        self._lbl_airgap.setFont(label_font)
        self._lbl_signal = QLabel("--")
        self._lbl_signal.setFont(label_font)
        self._lbl_spi_err = QLabel("--")
        self._lbl_spi_err.setFont(label_font)
        self._lbl_crc = QLabel("--")
        self._lbl_crc.setFont(label_font)
        self._lbl_error = QLabel("--")
        self._lbl_error.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self._lbl_warning = QLabel("--")
        self._lbl_warning.setFont(label_font)
        self._lbl_angle = QLabel("--")
        self._lbl_angle.setFont(label_font)

        sg.addWidget(QLabel("Airgap:"), 0, 0)
        sg.addWidget(self._lbl_airgap, 0, 1)
        sg.addWidget(QLabel("Signal Level:"), 0, 2)
        sg.addWidget(self._lbl_signal, 0, 3)
        sg.addWidget(QLabel("SPI Error Rate:"), 0, 4)
        sg.addWidget(self._lbl_spi_err, 0, 5)
        sg.addWidget(QLabel("CRC:"), 0, 6)
        sg.addWidget(self._lbl_crc, 0, 7)
        sg.addWidget(QLabel("Error:"), 0, 8)
        sg.addWidget(self._lbl_error, 0, 9)
        sg.addWidget(QLabel("Warning:"), 0, 10)
        sg.addWidget(self._lbl_warning, 0, 11)
        sg.addWidget(QLabel("Angle:"), 0, 12)
        sg.addWidget(self._lbl_angle, 0, 13)
        sg.setColumnStretch(14, 1)

        root.addWidget(status_group)

        # ── Graphs: left (time-series) | right (angle) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -- Left: Airgap + Signal vs Time --
        left_widget = QWidget()
        left_lay = QVBoxLayout(left_widget)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        self._pw_airgap = pg.PlotWidget()
        style_plot(self._pw_airgap, title="Airgap vs Time",
                   left_label="Airgap", left_unit="\u00b5m",
                   bottom_label="Time", bottom_unit="s")
        self._curve_airgap = self._pw_airgap.plot(
            pen=graph_pen(0, width=2.0), name="Airgap")
        left_lay.addWidget(self._pw_airgap, stretch=3)

        self._pw_signal = pg.PlotWidget()
        style_plot(self._pw_signal, title="Signal Level / SPI Error vs Time",
                   left_label="Signal", left_unit="",
                   bottom_label="Time", bottom_unit="s")
        self._curve_signal = self._pw_signal.plot(
            pen=graph_pen(2, width=1.5), name="Signal Level")

        self._vb_err = pg.ViewBox()
        self._pw_signal.getPlotItem().scene().addItem(self._vb_err)
        self._pw_signal.getPlotItem().getAxis('right').linkToView(self._vb_err)
        self._vb_err.setXLink(self._pw_signal)
        ax_right = self._pw_signal.getPlotItem().getAxis('right')
        ax_right.setLabel("SPI Error Rate", color=COLORS["orange"])
        ax_right.show()
        self._curve_spi_err = pg.PlotDataItem(
            pen=graph_pen(3, width=1.5), name="SPI Err")
        self._vb_err.addItem(self._curve_spi_err)
        self._pw_signal.getPlotItem().vb.sigResized.connect(self._sync_viewbox)
        left_lay.addWidget(self._pw_signal, stretch=2)

        splitter.addWidget(left_widget)

        # -- Right: Airgap vs Angle + Error/Warning vs Angle --
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        # Airgap vs Angle
        self._pw_angle = pg.PlotWidget()
        style_plot(self._pw_angle, title="Airgap vs Angle",
                   left_label="Airgap", left_unit="\u00b5m",
                   bottom_label="Angle", bottom_unit="\u00b0")
        self._pw_angle.setXRange(0, 360)
        self._pw_angle.setYRange(-100, 500)
        self._pw_angle.getPlotItem().getViewBox().setLimits(yMin=-100, yMax=500)
        # Raw data as scatter dots (semi-transparent)
        self._scatter_airgap = pg.ScatterPlotItem(
            size=5, pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 100, 100, 80))
        self._pw_angle.addItem(self._scatter_airgap)
        # Binned average line (bright cyan, smooth profile)
        self._curve_airgap_avg = self._pw_angle.plot(
            pen=pg.mkPen('#4FC3F7', width=2.5), name="Avg")
        # Current position dot (white highlight)
        self._dot_current = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen('#FFFFFF', width=1.5),
            brush=pg.mkBrush('#FFFFFF'))
        self._pw_angle.addItem(self._dot_current)
        right_lay.addWidget(self._pw_angle, stretch=3)

        # Error / Warning vs Angle
        self._pw_errwrn = pg.PlotWidget()
        style_plot(self._pw_errwrn, title="Error / Warning vs Angle",
                   left_label="", left_unit="",
                   bottom_label="Angle", bottom_unit="\u00b0")
        self._pw_errwrn.setXRange(0, 360)
        self._pw_errwrn.setYRange(-0.5, 1.5)
        self._pw_errwrn.getPlotItem().getViewBox().setLimits(
            yMin=-0.5, yMax=1.5, xMin=0, xMax=360)
        ytick = self._pw_errwrn.getPlotItem().getAxis('left')
        ytick.setTicks([[(1.0, 'Error'), (0.0, 'Warn')]])
        self._scatter_error = pg.ScatterPlotItem(
            size=14, symbol='x',
            pen=pg.mkPen(COLORS["red"], width=2.5), brush=pg.mkBrush(None))
        self._pw_errwrn.addItem(self._scatter_error)
        self._scatter_warning = pg.ScatterPlotItem(
            size=12, symbol='x',
            pen=pg.mkPen(COLORS["orange"], width=2.0), brush=pg.mkBrush(None))
        self._pw_errwrn.addItem(self._scatter_warning)
        right_lay.addWidget(self._pw_errwrn, stretch=2)

        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

    def _sync_viewbox(self):
        self._vb_err.setGeometry(
            self._pw_signal.getPlotItem().vb.sceneBoundingRect())

    # ── Signals ───────────────────────────────────────────────────────

    def _connect_signals(self):
        self._transport.encoder_health_received.connect(self._on_health_data)
        self._transport.encoder_received.connect(self._on_encoder_data)

    # ── Monitor controls ──────────────────────────────────────────────

    @pyqtSlot()
    def _on_start(self):
        if self._polling:
            return
        self._polling = True
        self._send_err_cnt = 0
        self._t0 = time.monotonic()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)

        if self._chk_csv.isChecked():
            self._start_csv()

        rate_text = self._combo_rate.currentText()
        hz = 10 if "10" in rate_text else 5
        self._poll_timer.start(int(1000 / hz))

    @pyqtSlot()
    def _on_stop(self):
        if self._scanning:
            self._on_scan_stop()
        self._polling = False
        self._poll_timer.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._stop_csv()

    @pyqtSlot()
    def _on_clear(self):
        for buf in (self._time, self._airgap, self._signal, self._spi_err,
                    self._ang_deg, self._ang_air,
                    self._err_ang, self._err_y, self._wrn_ang, self._wrn_y):
            buf.clear()
        self._last_enc_deg = 0.0
        self._curve_airgap.setData([], [])
        self._curve_signal.setData([], [])
        self._curve_spi_err.setData([], [])
        self._scatter_airgap.setData([], [])
        self._curve_airgap_avg.setData([], [])
        self._dot_current.setData([], [])
        self._scatter_error.setData([], [])
        self._scatter_warning.setData([], [])
        for lbl in (self._lbl_airgap, self._lbl_signal, self._lbl_spi_err,
                    self._lbl_crc, self._lbl_error, self._lbl_warning,
                    self._lbl_angle):
            lbl.setText("--")

    # ── Scan controls ─────────────────────────────────────────────────

    @pyqtSlot()
    def _on_scan_start(self):
        if self._scanning:
            return
        if not self._transport.is_connected():
            self._lbl_scan_status.setText("Not connected")
            return

        # Auto-start polling if not running
        if not self._polling:
            self._on_start()

        # Clear angle-domain buffers only (time-series untouched)
        for buf in (self._ang_deg, self._ang_air,
                    self._err_ang, self._err_y, self._wrn_ang, self._wrn_y):
            buf.clear()
        self._scatter_airgap.setData([], [])
        self._curve_airgap_avg.setData([], [])
        self._scatter_error.setData([], [])
        self._scatter_warning.setData([], [])

        revs = int(self._combo_revs.currentText())
        self._scan_target_deg = revs * 360.0
        self._scan_accum_deg = 0.0
        self._scan_prev_deg = self._last_enc_deg
        self._scanning = True

        self._btn_scan.setEnabled(False)
        self._btn_scan_stop.setEnabled(True)
        self._spin_dps.setEnabled(False)
        self._combo_revs.setEnabled(False)
        self._lbl_scan_status.setText("Scanning... 0\u00b0")
        self._lbl_scan_status.setStyleSheet("color: #4FCBCB;")

    @pyqtSlot()
    def _on_scan_stop(self):
        if not self._scanning:
            return
        self._scanning = False

        # Stop motor
        if self._transport.is_connected():
            self._transport.send_frame(build_motor_off())

        self._btn_scan.setEnabled(True)
        self._btn_scan_stop.setEnabled(False)
        self._spin_dps.setEnabled(True)
        self._combo_revs.setEnabled(True)
        self._lbl_scan_status.setText(
            f"Done: {abs(self._scan_accum_deg):.0f}\u00b0")
        self._lbl_scan_status.setStyleSheet("color: #7FC87F;")

        # Auto-stop polling when scan finishes
        self._on_stop()

    def _scan_track_angle(self, cur_deg: float):
        """Track accumulated rotation during scan."""
        if not self._scanning:
            return
        delta = cur_deg - self._scan_prev_deg
        # Handle 0/360 wraparound
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0
        self._scan_accum_deg += delta
        self._scan_prev_deg = cur_deg

        progress = abs(self._scan_accum_deg)
        self._lbl_scan_status.setText(
            f"Scanning... {progress:.0f}\u00b0 / {self._scan_target_deg:.0f}\u00b0")

        # Check completion
        if progress >= self._scan_target_deg:
            self._on_scan_stop()

    # ── Polling ───────────────────────────────────────────────────────

    def _poll_tick(self):
        if not self._transport.is_connected():
            return

        if self._scanning:
            # During scan: alternate health/encoder to keep ≤2 frames/tick
            dps = float(self._spin_dps.value())
            self._scan_phase = not self._scan_phase
            if self._scan_phase:
                ok = self._transport.send_frame(build_read_encoder_status())
            else:
                ok = self._transport.send_frame(build_read_encoder())
            ok2 = self._transport.send_frame(build_speed_closed_loop(dps))
            ok = ok and ok2
        else:
            ok = self._transport.send_frame(build_read_encoder_status())
            ok2 = self._transport.send_frame(build_read_encoder())
            ok = ok and ok2

        # Auto-stop on consecutive CAN errors (bus-off / TX queue full)
        if ok:
            self._send_err_cnt = 0
        else:
            self._send_err_cnt += 1
            if self._send_err_cnt >= 5:
                self._on_stop()
                self._lbl_scan_status.setText("CAN Error \u2014 Auto-stopped")
                self._lbl_scan_status.setStyleSheet("color: #C83434;")

    # ── Data handlers ─────────────────────────────────────────────────

    @pyqtSlot(int, object)
    def _on_encoder_data(self, motor_id: int, enc: RmdEncoder):
        self._last_enc_deg = enc.enc_pos_ori_deg % 360.0
        self._lbl_angle.setText(f"{self._last_enc_deg:.1f}\u00b0")
        self._scan_track_angle(self._last_enc_deg)

    @pyqtSlot(int, object)
    def _on_health_data(self, motor_id: int, health: RmdEncoderHealth):
        t = time.monotonic() - self._t0

        # Time-series buffers (always appended, always in sync)
        self._time.append(t)
        self._airgap.append(health.airgap_um)
        self._signal.append(float(health.signal_level))
        self._spi_err.append(health.spi_error_rate)

        # Compact time-series if over capacity
        if len(self._time) > BUF_CAP:
            for buf in (self._time, self._airgap, self._signal, self._spi_err):
                buf.compact(BUF_CAP)

        # Angle-domain buffers (for right graphs)
        self._ang_deg.append(self._last_enc_deg)
        self._ang_air.append(health.airgap_um)

        # Record error/warning angle positions
        if health.error:
            self._err_ang.append(self._last_enc_deg)
            self._err_y.append(1.0)
        if health.warning:
            self._wrn_ang.append(self._last_enc_deg)
            self._wrn_y.append(0.0)

        # Update labels
        self._lbl_airgap.setText(f"{health.airgap_um:.0f} \u00b5m")
        self._lbl_signal.setText(f"{health.signal_level}")
        self._lbl_spi_err.setText(f"{health.spi_error_rate:.3f}")
        self._lbl_crc.setText("OK" if health.crc_ok else "FAIL")
        self._lbl_crc.setStyleSheet(
            "color: #7FC87F;" if health.crc_ok else "color: #C83434; font-weight: bold;")
        err_cnt = len(self._err_ang)
        wrn_cnt = len(self._wrn_ang)
        self._lbl_error.setText(f"ERR({err_cnt})" if health.error else f"OK ({err_cnt})")
        self._lbl_error.setStyleSheet(
            "color: #C83434;" if health.error else "color: #7FC87F;")
        self._lbl_warning.setText(f"WRN({wrn_cnt})" if health.warning else f"OK ({wrn_cnt})")
        self._lbl_warning.setStyleSheet(
            "color: #CE7D2C;" if health.warning else "color: #7FC87F;")

        # ── Left graphs: time-series ──
        ta = self._time.array()
        window = self._get_window_sec()
        if window > 0 and len(ta) > 0:
            t_min = ta[-1] - window
            mask = ta >= t_min
            tx = ta[mask]
            self._curve_airgap.setData(tx, self._airgap.array()[mask])
            self._curve_signal.setData(tx, self._signal.array()[mask])
            self._curve_spi_err.setData(tx, self._spi_err.array()[mask])
        else:
            self._curve_airgap.setData(ta, self._airgap.array())
            self._curve_signal.setData(ta, self._signal.array())
            self._curve_spi_err.setData(ta, self._spi_err.array())

        # ── Right graphs: angle domain ──
        ang = self._ang_deg.array()
        air = self._ang_air.array()
        self._scatter_airgap.setData(ang, air)

        # Binned average (2° bins → smooth profile line)
        if len(ang) > 2:
            bins = np.arange(0, 362, 2)
            sums, _ = np.histogram(ang, bins=bins, weights=air)
            counts, _ = np.histogram(ang, bins=bins)
            valid = counts > 0
            centers = (bins[:-1] + bins[1:]) * 0.5
            means = np.zeros_like(sums)
            means[valid] = sums[valid] / counts[valid]
            self._curve_airgap_avg.setData(centers[valid], means[valid])

        self._dot_current.setData(
            [self._last_enc_deg], [health.airgap_um])
        if len(self._err_ang) > 0:
            self._scatter_error.setData(
                self._err_ang.array(), self._err_y.array())
        if len(self._wrn_ang) > 0:
            self._scatter_warning.setData(
                self._wrn_ang.array(), self._wrn_y.array())

        # CSV write
        if self._csv_writer is not None:
            self._csv_writer.writerow([
                f"{t:.3f}", f"{self._last_enc_deg:.1f}",
                f"{health.airgap_um:.0f}",
                health.signal_level, f"{health.spi_error_rate:.3f}",
                1 if health.crc_ok else 0,
                1 if health.error else 0,
                1 if health.warning else 0,
            ])

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_window_sec(self) -> int:
        text = self._combo_window.currentText()
        if "All" in text:
            return 0
        return int(text.split()[0])

    def _start_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "rls_health_log.csv", "CSV Files (*.csv)")
        if not path:
            self._chk_csv.setChecked(False)
            return
        self._csv_file = open(path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "time_s", "angle_deg", "airgap_um", "signal_level",
            "spi_error_rate", "crc_ok", "error", "warning",
        ])

    def _stop_csv(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def cleanup(self):
        self._on_stop()
