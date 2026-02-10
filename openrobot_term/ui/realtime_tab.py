"""
Real-time data monitoring tab — VESC-Tool style.
Uses pre-allocated numpy ring buffers and renders only visible data.
"""

import time
import csv
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer

import numpy as np
import pyqtgraph as pg

from ..protocol.can_transport import PcanTransport
from ..protocol.commands import VescValues, CommPacketId
from ..workers.data_poller import DataPoller


from .plot_style import (
    style_plot, graph_pen, style_legend,
    AXIS_COLOR, TEXT_LIGHT, TEXT_DISABLED,
)

RENDER_INTERVAL_MS = 33  # ~30 fps
MARKER_AVG_SEC = 2.0  # min/max markers use 2-second rolling window
BUF_CAP = 12000  # initial capacity
BUF_COMPACT = 24000  # compact when exceeding this


class _GrowBuffer:
    """Flat growing numpy buffer. array() is always a zero-copy slice view."""
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
        """Zero-copy slice view — no concatenate ever."""
        return self._buf[:self._len]

    def compact(self, keep: int):
        """Keep only the last `keep` elements, shift in-place."""
        if self._len > keep:
            start = self._len - keep
            self._buf[:keep] = self._buf[start:self._len]
            self._len = keep


class RealtimeTab(QWidget):
    values_received = pyqtSignal(object)

    def __init__(self, transport: PcanTransport):
        super().__init__()
        self._transport = transport
        self._poller = None
        self._t0 = None
        self._csv_file = None
        self._csv_writer = None
        self._polling = False
        self._auto_range = True
        self._x_window = 10.0
        self._dirty = False
        self._last_values = None
        self._frame_count = 0
        self._programmatic_update = False

        # Flat growing buffers — array() is always zero-copy slice
        self._rb_time = _GrowBuffer(BUF_CAP)
        self._rb_vin = _GrowBuffer(BUF_CAP)
        self._rb_im = _GrowBuffer(BUF_CAP)
        self._rb_ii = _GrowBuffer(BUF_CAP)
        self._rb_rpm = _GrowBuffer(BUF_CAP)
        self._rb_tmos = _GrowBuffer(BUF_CAP)
        self._rb_tmot = _GrowBuffer(BUF_CAP)
        self._rb_duty = _GrowBuffer(BUF_CAP)
        self._rb_id = _GrowBuffer(BUF_CAP)
        self._rb_iq = _GrowBuffer(BUF_CAP)

        self._all_rbs = [
            self._rb_time, self._rb_vin, self._rb_im, self._rb_ii,
            self._rb_rpm, self._rb_tmos, self._rb_tmot,
            self._rb_duty, self._rb_id, self._rb_iq,
        ]

        self._build_ui()

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._render_frame)
        self._render_timer.start()

        # Auto-resume timer: re-enable auto-range after manual zoom pause
        self._auto_resume_timer = QTimer(self)
        self._auto_resume_timer.setSingleShot(True)
        self._auto_resume_timer.setInterval(2000)  # 2 seconds
        self._auto_resume_timer.timeout.connect(self._auto_resume)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        layout.addLayout(ctrl)

        self.start_btn = QPushButton("Start Polling")
        self.start_btn.clicked.connect(self.toggle_polling)
        ctrl.addWidget(self.start_btn)

        ctrl.addWidget(QLabel("Rate:"))
        self.rate_combo = QComboBox()
        for r in ["10 Hz", "20 Hz", "50 Hz"]:
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

        ctrl.addStretch()

        ctrl.addWidget(QLabel("X Range:"))
        self.xrange_combo = QComboBox()
        self.xrange_combo.addItems(["5 s", "10 s", "30 s", "60 s", "All"])
        self.xrange_combo.setCurrentText("10 s")
        self.xrange_combo.currentTextChanged.connect(self._on_xrange_changed)
        ctrl.addWidget(self.xrange_combo)

        self.autorange_chk = QCheckBox("Auto")
        self.autorange_chk.setChecked(True)
        self.autorange_chk.toggled.connect(self._on_autorange_toggled)
        ctrl.addWidget(self.autorange_chk)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("Reset all plots to full data range")
        self.fit_btn.clicked.connect(self._fit_all)
        ctrl.addWidget(self.fit_btn)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Graph FPS:"))
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["10", "15", "20", "30", "60", "120"])
        self.fps_combo.setCurrentText("30")
        self.fps_combo.currentTextChanged.connect(self._on_fps_changed)
        ctrl.addWidget(self.fps_combo)

        # ── Plots ──
        graph_grid = QGridLayout()
        layout.addLayout(graph_grid, stretch=1)

        self.plot_vi = pg.PlotWidget()
        style_plot(self.plot_vi, title="Voltage / Current",
                   left_label="Voltage", left_unit="V")
        vi_legend = style_legend(self.plot_vi)
        self.curve_vin = self.plot_vi.plot(pen=graph_pen(4), name="V_in")

        self.vi_viewbox = pg.ViewBox()
        self.plot_vi.scene().addItem(self.vi_viewbox)
        self.plot_vi.getAxis("right").linkToView(self.vi_viewbox)
        self.vi_viewbox.setXLink(self.plot_vi)
        self.plot_vi.getAxis("right").setLabel("Current", "A",
                                                **{"color": AXIS_COLOR, "font-size": "9pt"})
        self.plot_vi.showAxis("right")
        self.curve_im = pg.PlotCurveItem(pen=graph_pen(1), name="I_motor")
        self.curve_ii = pg.PlotCurveItem(pen=graph_pen(3), name="I_input")
        self.vi_viewbox.addItem(self.curve_im)
        self.vi_viewbox.addItem(self.curve_ii)
        # Manually add secondary ViewBox curves to legend
        vi_legend.addItem(self.curve_im, "I_motor")
        vi_legend.addItem(self.curve_ii, "I_input")
        self.plot_vi.getViewBox().sigResized.connect(self._sync_vi_viewbox)
        graph_grid.addWidget(self.plot_vi, 0, 0)

        self.plot_rpm = pg.PlotWidget()
        style_plot(self.plot_rpm, title="RPM", left_label="RPM", left_unit="")
        style_legend(self.plot_rpm)
        self.curve_rpm = self.plot_rpm.plot(pen=graph_pen(0), name="RPM")
        graph_grid.addWidget(self.plot_rpm, 0, 1)

        self.plot_temp = pg.PlotWidget()
        style_plot(self.plot_temp, title="Temperature",
                   left_label="Temperature", left_unit="°C")
        style_legend(self.plot_temp)
        self.curve_tmos = self.plot_temp.plot(pen=graph_pen(1), name="MOSFET")
        self.curve_tmot = self.plot_temp.plot(pen=graph_pen(3), name="Motor")
        graph_grid.addWidget(self.plot_temp, 1, 0)

        self.plot_duty = pg.PlotWidget()
        style_plot(self.plot_duty, title="Duty / FOC Currents",
                   left_label="Duty", left_unit="")
        duty_legend = style_legend(self.plot_duty)
        self.curve_duty = self.plot_duty.plot(pen=graph_pen(2), name="Duty")

        # Secondary Y axis for Id/Iq
        self.duty_viewbox = pg.ViewBox()
        self.plot_duty.scene().addItem(self.duty_viewbox)
        self.plot_duty.getAxis("right").linkToView(self.duty_viewbox)
        self.duty_viewbox.setXLink(self.plot_duty)
        self.plot_duty.getAxis("right").setLabel("Current", "A",
                                                  **{"color": AXIS_COLOR, "font-size": "9pt"})
        self.plot_duty.showAxis("right")
        self.curve_id = pg.PlotCurveItem(pen=graph_pen(6), name="Id")
        self.curve_iq = pg.PlotCurveItem(pen=graph_pen(5), name="Iq")
        self.duty_viewbox.addItem(self.curve_id)
        self.duty_viewbox.addItem(self.curve_iq)
        # Manually add secondary ViewBox curves to legend
        duty_legend.addItem(self.curve_id, "Id")
        duty_legend.addItem(self.curve_iq, "Iq")
        self.plot_duty.getViewBox().sigResized.connect(self._sync_duty_viewbox)
        graph_grid.addWidget(self.plot_duty, 1, 1)

        self._all_plots = [self.plot_vi, self.plot_rpm, self.plot_temp, self.plot_duty]
        for pw in self._all_plots:
            pw.disableAutoRange()
            pw.setXRange(0, 10, padding=0)
        self.vi_viewbox.disableAutoRange()
        self.duty_viewbox.disableAutoRange()

        # Min/Max overlay QLabels — fixed pixel positions, no trembling
        _mkstyle_l = (
            "font-family: Consolas; font-size: 9px; color: #ccc; "
            "background: rgba(30,30,30,160); padding: 1px 3px; border-radius: 2px;"
        )
        _mkstyle_r = (
            "font-family: Consolas; font-size: 9px; color: #ccc; "
            "background: rgba(30,30,30,160); padding: 1px 3px; border-radius: 2px;"
        )

        def _make_overlay(parent, align_right=False):
            lbl = QLabel("--", parent)
            lbl.setStyleSheet(_mkstyle_r if align_right else _mkstyle_l)
            lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            lbl.raise_()
            return lbl

        # VI plot markers: voltage (left), current (right)
        self._mk_vin_max = _make_overlay(self.plot_vi)
        self._mk_vin_min = _make_overlay(self.plot_vi)
        self._mk_cur_max = _make_overlay(self.plot_vi, align_right=True)
        self._mk_cur_min = _make_overlay(self.plot_vi, align_right=True)

        # RPM
        self._mk_rpm_max = _make_overlay(self.plot_rpm)
        self._mk_rpm_min = _make_overlay(self.plot_rpm)

        # Temp
        self._mk_temp_max = _make_overlay(self.plot_temp)
        self._mk_temp_min = _make_overlay(self.plot_temp)

        # Duty (left), FOC Id/Iq (right)
        self._mk_duty_max = _make_overlay(self.plot_duty)
        self._mk_duty_min = _make_overlay(self.plot_duty)
        self._mk_foc_max = _make_overlay(self.plot_duty, align_right=True)
        self._mk_foc_min = _make_overlay(self.plot_duty, align_right=True)

        for pw in self._all_plots:
            pw.getPlotItem().getViewBox().sigRangeChangedManually.connect(self._on_manual_zoom)

        self.status_label = QLabel("Status: not polling")
        self.status_label.setStyleSheet("font-family: monospace; font-size: 12px; padding: 4px;")
        layout.addWidget(self.status_label)

    # ── Data ingestion (ultra-fast — single float append per channel) ──

    def on_values(self, v: VescValues):
        if self._t0 is None:
            self._t0 = time.time()

        t = time.time() - self._t0
        self._rb_time.append(t)
        self._rb_vin.append(v.v_in)
        self._rb_im.append(v.avg_motor_current)
        self._rb_ii.append(v.avg_input_current)
        self._rb_rpm.append(v.rpm)
        self._rb_tmos.append(v.temp_mosfet)
        self._rb_tmot.append(v.temp_motor)
        self._rb_duty.append(v.duty_now)
        self._rb_id.append(v.id_current)
        self._rb_iq.append(v.iq_current)

        self._dirty = True
        self._last_values = v
        self.values_received.emit(v)

        if self._csv_writer:
            self._csv_writer.writerow([
                f"{t:.4f}", f"{v.v_in:.2f}", f"{v.avg_motor_current:.2f}",
                f"{v.avg_input_current:.2f}", f"{v.rpm:.0f}", f"{v.duty_now:.4f}",
                f"{v.temp_mosfet:.1f}", f"{v.temp_motor:.1f}",
                f"{v.id_current:.2f}", f"{v.iq_current:.2f}", f"{v.fault_code}",
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

        # Periodic compaction — prevent unbounded memory growth
        if n > BUF_COMPACT:
            keep = BUF_CAP
            for rb in self._all_rbs:
                rb.compact(keep)
            n = keep

        # Zero-copy slice views (no concatenate, no allocation)
        x_all = self._rb_time.array()
        t_now = x_all[n - 1]

        # Compute visible window (clamp to first sample — never go negative)
        if self._x_window <= 0:
            t_min = x_all[0]
            s = 0
        else:
            t_min = max(x_all[0], t_now - self._x_window)
            s = int(np.searchsorted(x_all, t_min))

        # Visible slices — all zero-copy views into flat buffers
        x = x_all[s:]
        d_vin = self._rb_vin.array()[s:]
        d_im = self._rb_im.array()[s:]
        d_ii = self._rb_ii.array()[s:]
        d_rpm = self._rb_rpm.array()[s:]
        d_tmos = self._rb_tmos.array()[s:]
        d_tmot = self._rb_tmot.array()[s:]
        d_duty = self._rb_duty.array()[s:]
        d_id = self._rb_id.array()[s:]
        d_iq = self._rb_iq.array()[s:]

        # Sync secondary ViewBox geometry before update
        self._sync_vi_viewbox()
        self._sync_duty_viewbox()

        # Flag to suppress sigRangeChangedManually during programmatic updates
        self._programmatic_update = True

        # setData — skipFiniteCheck avoids per-element isfinite scan
        self.curve_vin.setData(x, d_vin, skipFiniteCheck=True)
        self.curve_im.setData(x, d_im, skipFiniteCheck=True)
        self.curve_ii.setData(x, d_ii, skipFiniteCheck=True)
        self.curve_rpm.setData(x, d_rpm, skipFiniteCheck=True)
        self.curve_tmos.setData(x, d_tmos, skipFiniteCheck=True)
        self.curve_tmot.setData(x, d_tmot, skipFiniteCheck=True)
        self.curve_duty.setData(x, d_duty, skipFiniteCheck=True)
        self.curve_id.setData(x, d_id, skipFiniteCheck=True)
        self.curve_iq.setData(x, d_iq, skipFiniteCheck=True)

        # X axis always scrolls — set range OUTSIDE blockSignals so it actually takes effect
        if len(x) > 0:
            pad = 0.0 if self._x_window > 0 else 0.02
            for pw in self._all_plots:
                pw.setXRange(t_min, t_now, padding=pad)
            self.vi_viewbox.setXRange(t_min, t_now, padding=pad)
            self.duty_viewbox.setXRange(t_min, t_now, padding=pad)

        # Y axis auto-range — only when enabled
        if self._auto_range and len(x) > 0:
            def _yr(a, b=None):
                if b is not None:
                    lo = min(float(a.min()), float(b.min()))
                    hi = max(float(a.max()), float(b.max()))
                else:
                    lo, hi = float(a.min()), float(a.max())
                m = (hi - lo) * 0.05 if hi != lo else 1.0
                return lo - m, hi + m

            self.plot_vi.setYRange(*_yr(d_vin), padding=0)
            self.vi_viewbox.setYRange(*_yr(d_im, d_ii), padding=0)
            self.plot_rpm.setYRange(*_yr(d_rpm), padding=0)
            self.plot_temp.setYRange(*_yr(d_tmos, d_tmot), padding=0)
            self.plot_duty.setYRange(-1.05, 1.05, padding=0)
            self.duty_viewbox.setYRange(*_yr(d_id, d_iq), padding=0)

        self._programmatic_update = False

        # Min/max overlay labels — 2-second rolling window, fixed pixel positions
        fc = self._frame_count
        if len(x) > 0 and (fc & 3) == 0:
            m_start = int(np.searchsorted(x, t_now - MARKER_AVG_SEC))
            mv = d_vin[m_start:]
            mi = d_im[m_start:]
            mii = d_ii[m_start:]
            mr = d_rpm[m_start:]
            mtmos = d_tmos[m_start:]
            mtmot = d_tmot[m_start:]
            mdu = d_duty[m_start:]
            mid_ = d_id[m_start:]
            miq = d_iq[m_start:]

            if len(mv) > 0:
                # Pixel offsets from plot widget edges
                _TOP = 24   # below title + legend
                _BOT_OFF = 42  # above X axis
                _LEFT = 50  # right of Y axis
                _RIGHT_OFF = 50  # left of right Y axis

                def _place(pw, mk_max, mk_min, max_txt, min_txt, right=False):
                    mk_max.setText(max_txt)
                    mk_min.setText(min_txt)
                    mk_max.adjustSize()
                    mk_min.adjustSize()
                    h = pw.height()
                    w = pw.width()
                    if right:
                        mk_max.move(w - _RIGHT_OFF - mk_max.width(), _TOP)
                        mk_min.move(w - _RIGHT_OFF - mk_min.width(), h - _BOT_OFF - mk_min.height())
                    else:
                        mk_max.move(_LEFT, _TOP)
                        mk_min.move(_LEFT, h - _BOT_OFF - mk_min.height())

                v_lo, v_hi = float(mv.min()), float(mv.max())
                c_lo = min(float(mi.min()), float(mii.min()))
                c_hi = max(float(mi.max()), float(mii.max()))
                _place(self.plot_vi, self._mk_vin_max, self._mk_vin_min,
                       f"max {v_hi:.1f}V", f"min {v_lo:.1f}V")
                _place(self.plot_vi, self._mk_cur_max, self._mk_cur_min,
                       f"max {c_hi:.2f}A", f"min {c_lo:.2f}A", right=True)

                r_lo, r_hi = float(mr.min()), float(mr.max())
                _place(self.plot_rpm, self._mk_rpm_max, self._mk_rpm_min,
                       f"max {r_hi:.0f}", f"min {r_lo:.0f}")

                tmp_lo = min(float(mtmos.min()), float(mtmot.min()))
                tmp_hi = max(float(mtmos.max()), float(mtmot.max()))
                _place(self.plot_temp, self._mk_temp_max, self._mk_temp_min,
                       f"max {tmp_hi:.1f}\u00b0C", f"min {tmp_lo:.1f}\u00b0C")

                du_lo, du_hi = float(mdu.min()), float(mdu.max())
                foc_lo = min(float(mid_.min()), float(miq.min()))
                foc_hi = max(float(mid_.max()), float(miq.max()))
                _place(self.plot_duty, self._mk_duty_max, self._mk_duty_min,
                       f"max {du_hi:.3f}", f"min {du_lo:.3f}")
                _place(self.plot_duty, self._mk_foc_max, self._mk_foc_min,
                       f"max {foc_hi:.2f}A", f"min {foc_lo:.2f}A", right=True)

        # Status bar — throttled to ~4 fps
        if (fc & 7) == 0:
            v = self._last_values
            if v:
                power = v.v_in * v.avg_input_current
                self.status_label.setText(
                    f"V_in={v.v_in:.1f}V  I_mot={v.avg_motor_current:.1f}A  "
                    f"I_in={v.avg_input_current:.1f}A  P={power:.1f}W  "
                    f"RPM={v.rpm:.0f}  Duty={v.duty_now:.3f}  "
                    f"T_mos={v.temp_mosfet:.1f}°C  T_mot={v.temp_motor:.1f}°C  "
                    f"Fault={v.fault_code}"
                )

    # ── Controls ──

    def _on_fps_changed(self, text: str):
        fps = int(text)
        self._render_timer.setInterval(int(1000 / fps))

    def _on_xrange_changed(self, text: str):
        self._x_window = 0.0 if text == "All" else float(text.replace(" s", ""))
        self._dirty = True

    def _on_autorange_toggled(self, checked: bool):
        self._auto_range = checked
        if checked:
            self._auto_resume_timer.stop()
            self._dirty = True

    def _on_manual_zoom(self):
        # Ignore range changes triggered by our own programmatic setXRange/setYRange
        if getattr(self, '_programmatic_update', False):
            return
        if self._auto_range:
            self._auto_range = False
            self.autorange_chk.setChecked(False)
        # Restart auto-resume countdown on every manual interaction
        if self._polling:
            self._auto_resume_timer.start()

    def _auto_resume(self):
        """Re-enable auto-range after manual zoom timeout."""
        if self._polling and not self._auto_range:
            self._auto_range = True
            self.autorange_chk.setChecked(True)
            self._dirty = True

    def _fit_all(self):
        old = self._x_window
        self._x_window = 0.0
        self._dirty = True
        self._render_frame()
        self._x_window = old

    def showEvent(self, event):
        super().showEvent(event)
        # Initial geometry sync for secondary ViewBoxes
        self._sync_vi_viewbox()
        self._sync_duty_viewbox()

    def _sync_vi_viewbox(self):
        self.vi_viewbox.setGeometry(self.plot_vi.getViewBox().sceneBoundingRect())

    def _sync_duty_viewbox(self):
        self.duty_viewbox.setGeometry(self.plot_duty.getViewBox().sceneBoundingRect())

    def _get_rate_ms(self) -> int:
        return int(1000 / int(self.rate_combo.currentText().replace(" Hz", "")))

    def toggle_polling(self):
        if self._polling:
            self.stop_polling()
        else:
            self.start_polling()

    def start_polling(self):
        if not self._transport.is_connected():
            return
        self._t0 = time.time()
        self._poller = DataPoller(self._transport, self._get_rate_ms())
        self._poller.start()
        self._polling = True
        self.start_btn.setText("Stop Polling")
        self.status_label.setText("Status: Polling started...")
        self.status_label.setStyleSheet("font-family: monospace; font-size: 12px; padding: 4px; color: #66ff66;")

    def stop_polling(self):
        if self._poller:
            self._poller.stop()
            self._poller.wait(2000)  # increased timeout
            if self._poller.isRunning():
                self._poller.terminate()  # force terminate if still running
                self._poller.wait(500)
            self._poller = None
        self._polling = False
        self.start_btn.setText("Start Polling")
        self.status_label.setText("Status: Polling stopped")
        self.status_label.setStyleSheet("font-family: monospace; font-size: 12px; padding: 4px; color: #ffaa00;")

    def _on_rate_changed(self, text: str):
        if self._poller and self._polling:
            self._poller.interval_ms = self._get_rate_ms()

    def toggle_csv(self, checked: bool):
        if checked:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CSV", f"log_{ts}.csv", "CSV (*.csv)"
            )
            if path:
                self._csv_file = open(path, "w", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow([
                    "time_s", "v_in", "i_motor", "i_input", "rpm", "duty",
                    "temp_mos", "temp_mot", "id", "iq", "fault"
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
