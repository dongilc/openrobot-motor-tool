"""
Position tab — real-time encoder rotor position graph via COMM_SET_DETECT streaming.

Firmware streams COMM_ROTOR_POSITION (22) every ~10ms when display mode is active.
Start/stop via COMM_SET_DETECT (11) with DISP_POS_MODE_* constants.
"""

import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QComboBox, QLabel, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer

from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import (
    build_set_detect, decode_rotor_position,
    DISP_POS_MODE_NONE, DISP_POS_MODE_ENCODER,
    DISP_POS_MODE_OBSERVER, DISP_POS_MODE_PID_POS,
    DISP_POS_MODE_PID_POS_ERROR, DISP_POS_MODE_ENCODER_OBSERVER_ERROR,
    DISP_POS_MODE_ACCUM,
)
from .plot_style import style_plot, graph_pen, style_legend, set_curve_visible

RENDER_INTERVAL_MS = 33  # ~30 fps
BUF_CAP = 12000
BUF_COMPACT = 24000

# Display mode label → constant mapping
_MODE_MAP = [
    ("Encoder",                DISP_POS_MODE_ENCODER),
    ("Observer (FOC)",         DISP_POS_MODE_OBSERVER),
    ("PID Position",           DISP_POS_MODE_PID_POS),
    ("PID Pos Error",          DISP_POS_MODE_PID_POS_ERROR),
    ("Encoder-Observer Error", DISP_POS_MODE_ENCODER_OBSERVER_ERROR),
    ("Accum Position (MCU)",   DISP_POS_MODE_ACCUM),
]


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
        return self._buf[:self._len]

    def compact(self, keep: int):
        if self._len > keep:
            start = self._len - keep
            self._buf[:keep] = self._buf[start:self._len]
            self._len = keep


class PositionTab(QWidget):
    def __init__(self, transport: SerialTransport):
        super().__init__()
        self._transport = transport
        self._streaming = False
        self._current_mode = DISP_POS_MODE_NONE
        self._sample_count = 0
        self._dirty = False
        self._auto_range = True
        self._x_window = 10.0
        self._frame_count = 0
        self._programmatic_update = False

        # Multiturn tracking (SW wrap detection)
        self._prev_pos = None
        self._cumulative_turns = 0.0

        # Ring buffers
        self._rb_time = _GrowBuffer(BUF_CAP)
        self._rb_pos = _GrowBuffer(BUF_CAP)        # single-cycle position (deg)
        self._rb_multi = _GrowBuffer(BUF_CAP)       # primary multiturn (deg)
        self._rb_mcu_multi = _GrowBuffer(BUF_CAP)   # MCU accumulated (ACCUM mode)
        self._rb_sw_multi = _GrowBuffer(BUF_CAP)    # SW wrap-detected (ACCUM mode)
        self._rb_error = _GrowBuffer(BUF_CAP)       # MCU - SW error (ACCUM mode)

        self._all_rbs = [self._rb_time, self._rb_pos, self._rb_multi,
                         self._rb_mcu_multi, self._rb_sw_multi, self._rb_error]

        self._build_ui()

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._render_frame)
        self._render_timer.start()

        # Auto-resume timer
        self._auto_resume_timer = QTimer(self)
        self._auto_resume_timer.setSingleShot(True)
        self._auto_resume_timer.setInterval(2000)
        self._auto_resume_timer.timeout.connect(self._auto_resume)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Controls row ──
        ctrl = QHBoxLayout()
        layout.addLayout(ctrl)

        self.mode_combo = QComboBox()
        for label, _ in _MODE_MAP:
            self.mode_combo.addItem(label)
        self.mode_combo.setCurrentIndex(len(_MODE_MAP) - 1)  # Default: Accum Position (MCU)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        ctrl.addWidget(self.mode_combo)

        self.start_btn = QPushButton("Start Streaming")
        self.start_btn.setStyleSheet(
            "QPushButton { background: #1B5E20; color: white; font-weight: bold; padding: 4px 12px; }"
            "QPushButton:hover { background: #2E7D32; }"
        )
        self.start_btn.clicked.connect(self._toggle_streaming)
        ctrl.addWidget(self.start_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
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
        self.fps_combo.addItems(["10", "15", "20", "30", "60"])
        self.fps_combo.setCurrentText("30")
        self.fps_combo.currentTextChanged.connect(self._on_fps_changed)
        ctrl.addWidget(self.fps_combo)

        # ── Plots (vertical splitter for resizable heights) ──
        self.plot_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self.plot_splitter, stretch=1)

        # Plot 1: Rotor Position (degrees, single cycle 0~360)
        self.plot_pos = pg.PlotWidget()
        style_plot(self.plot_pos, title="Rotor Position",
                   left_label="Position", left_unit="deg")
        style_legend(self.plot_pos)
        self.curve_pos = self.plot_pos.plot(pen=graph_pen(0), name="rotor_pos")
        self.plot_splitter.addWidget(self.plot_pos)

        # Plot 2: Multiturn Angle — MCU vs SW comparison
        self.plot_multi = pg.PlotWidget()
        style_plot(self.plot_multi, title="Multiturn Angle",
                   left_label="Angle", left_unit="deg")
        self._legend_multi = style_legend(self.plot_multi)
        self.curve_mcu = self.plot_multi.plot(pen=graph_pen(2), name="MCU")
        self.curve_sw = self.plot_multi.plot(pen=graph_pen(3), name="SW")
        set_curve_visible(self._legend_multi, self.curve_sw, False)
        self.plot_splitter.addWidget(self.plot_multi)

        # Plot 3: Multiturn Error (MCU - SW), only in ACCUM mode
        self.plot_error = pg.PlotWidget()
        style_plot(self.plot_error, title="Multiturn Error (MCU - SW)",
                   left_label="Error", left_unit="deg")
        style_legend(self.plot_error)
        self.curve_error = self.plot_error.plot(pen=graph_pen(1), name="error")
        self.plot_splitter.addWidget(self.plot_error)

        # Initial splitter proportions (pos : multi : error)
        self.plot_splitter.setSizes([400, 200, 200])

        self._all_plots = [self.plot_pos, self.plot_multi, self.plot_error]
        for pw in self._all_plots:
            pw.disableAutoRange()
            pw.setXRange(0, 10, padding=0)

        for pw in self._all_plots:
            pw.getPlotItem().getViewBox().sigRangeChangedManually.connect(self._on_manual_zoom)

        # Status label
        self.status_label = QLabel("Select mode and press Start Streaming")
        self.status_label.setStyleSheet("font-family: monospace; font-size: 12px; padding: 4px;")
        layout.addWidget(self.status_label)

    # ── Streaming control ──

    def _toggle_streaming(self):
        if self._streaming:
            self._stop_streaming()
        else:
            self._start_streaming()

    def _start_streaming(self):
        if not self._transport.is_connected():
            self.status_label.setText("Not connected")
            return
        idx = self.mode_combo.currentIndex()
        _, mode = _MODE_MAP[idx]
        self._transport.send_packet(build_set_detect(mode))
        self._streaming = True
        self._current_mode = mode
        self._sample_count = 0

        # ACCUM mode: show Multiturn + Error plots; other modes: only Rotor Position
        is_accum = (mode == DISP_POS_MODE_ACCUM)
        self.plot_multi.setVisible(is_accum)
        self.plot_error.setVisible(is_accum)

        self.start_btn.setText("Stop Streaming")
        self.start_btn.setStyleSheet(
            "QPushButton { background: #B71C1C; color: white; font-weight: bold; padding: 4px 12px; }"
            "QPushButton:hover { background: #D32F2F; }"
        )
        self.status_label.setText(f"Streaming: {self.mode_combo.currentText()}")
        self.status_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; padding: 4px; color: #66ff66;")

    def _stop_streaming(self):
        if self._transport.is_connected():
            self._transport.send_packet(build_set_detect(DISP_POS_MODE_NONE))
        self._streaming = False
        self.start_btn.setText("Start Streaming")
        self.start_btn.setStyleSheet(
            "QPushButton { background: #1B5E20; color: white; font-weight: bold; padding: 4px 12px; }"
            "QPushButton:hover { background: #2E7D32; }"
        )
        # Keep the last values visible (don't overwrite status_label)

    def _on_mode_changed(self, index: int):
        """Auto-clear and restart streaming when mode changes."""
        was_streaming = self._streaming
        if was_streaming:
            self._stop_streaming()
        self._on_clear()
        _, mode = _MODE_MAP[index]
        is_accum = (mode == DISP_POS_MODE_ACCUM)
        self.plot_multi.setVisible(is_accum)
        self.plot_error.setVisible(is_accum)
        if was_streaming:
            self._start_streaming()

    # ── Data ingestion (called from main_window dispatch) ──

    def on_rotor_position(self, data: bytes):
        """Handle COMM_ROTOR_POSITION packet.

        Packet formats (after cmd_id byte is stripped):
          4 bytes: single int32 / 100000.0  (legacy single-value modes)
          8 bytes: int32[0] / 10000.0 (encoder_deg) + int32[1] / 1000.0 (accum_deg)
        """
        import struct

        # Uniform 10ms spacing (firmware sends at exactly 10ms intervals)
        t = self._sample_count * 0.01
        self._sample_count += 1

        if self._current_mode == DISP_POS_MODE_ACCUM and len(data) >= 8:
            # Dual-value packet: real encoder + MCU accumulated
            single = struct.unpack_from(">i", data, 0)[0] / 10000.0
            mcu_multi = struct.unpack_from(">i", data, 4)[0] / 1000.0

            # SW wrap-detection on real encoder position
            if self._prev_pos is not None:
                delta = single - self._prev_pos
                if delta > 180.0:
                    delta -= 360.0
                elif delta < -180.0:
                    delta += 360.0
                self._cumulative_turns += delta
            self._prev_pos = single
            sw_multi = self._cumulative_turns

            accum = mcu_multi
            error = mcu_multi - sw_multi
        elif self._current_mode == DISP_POS_MODE_ACCUM:
            # Fallback: old firmware sending single accum value
            mcu_multi = decode_rotor_position(data)
            single = mcu_multi % 360.0
            if single < 0:
                single += 360.0

            if self._prev_pos is not None:
                delta = single - self._prev_pos
                if delta > 180.0:
                    delta -= 360.0
                elif delta < -180.0:
                    delta += 360.0
                self._cumulative_turns += delta
            self._prev_pos = single
            sw_multi = self._cumulative_turns

            accum = mcu_multi
            error = mcu_multi - sw_multi
        else:
            # Software multiturn tracking via wrap detection
            pos_deg = decode_rotor_position(data)
            single = pos_deg
            if self._prev_pos is not None:
                delta = pos_deg - self._prev_pos
                if delta > 180.0:
                    delta -= 360.0
                elif delta < -180.0:
                    delta += 360.0
                self._cumulative_turns += delta
            self._prev_pos = pos_deg
            accum = self._cumulative_turns
            mcu_multi = 0.0
            sw_multi = accum
            error = 0.0

        self._rb_time.append(t)
        self._rb_pos.append(single)
        self._rb_multi.append(accum)
        self._rb_mcu_multi.append(mcu_multi)
        self._rb_sw_multi.append(sw_multi)
        self._rb_error.append(error)

        self._dirty = True

    # ── Rendering ──

    def _render_frame(self):
        if not self._dirty:
            return
        self._dirty = False
        self._frame_count += 1

        n = len(self._rb_time)
        if n == 0:
            return

        # Periodic compaction
        if n > BUF_COMPACT:
            keep = BUF_CAP
            for rb in self._all_rbs:
                rb.compact(keep)
            n = keep

        x_all = self._rb_time.array()
        t_now = x_all[n - 1]

        if self._x_window <= 0:
            t_min = x_all[0]
            s = 0
        else:
            t_min = max(x_all[0], t_now - self._x_window)
            s = int(np.searchsorted(x_all, t_min))

        x = x_all[s:]
        d_pos = self._rb_pos.array()[s:]
        d_multi = self._rb_multi.array()[s:]

        is_accum = (self._current_mode == DISP_POS_MODE_ACCUM)
        if is_accum:
            d_mcu = self._rb_mcu_multi.array()[s:]
            d_sw = self._rb_sw_multi.array()[s:]
            d_err = self._rb_error.array()[s:]

        self._programmatic_update = True

        self.curve_pos.setData(x, d_pos, skipFiniteCheck=True)
        if is_accum:
            self.curve_mcu.setData(x, d_mcu, skipFiniteCheck=True)
            self.curve_sw.setData(x, d_sw, skipFiniteCheck=True)
            self.curve_error.setData(x, d_err, skipFiniteCheck=True)

        if len(x) > 0:
            pad = 0.0 if self._x_window > 0 else 0.02
            for pw in self._all_plots:
                if pw.isVisible():
                    pw.setXRange(t_min, t_now, padding=pad)

        if self._auto_range and len(x) > 0:
            def _yr(a):
                lo, hi = float(a.min()), float(a.max())
                m = (hi - lo) * 0.05 if hi != lo else 1.0
                return lo - m, hi + m

            self.plot_pos.setYRange(*_yr(d_pos), padding=0)
            if is_accum:
                all_multi = np.concatenate([d_mcu, d_sw])
                self.plot_multi.setYRange(*_yr(all_multi), padding=0)
                self.plot_error.setYRange(*_yr(d_err), padding=0)

        self._programmatic_update = False

        # Update status bar every ~4 frames
        if len(x) > 0 and (self._frame_count & 3) == 0:
            cur_pos = d_pos[-1]
            if is_accum:
                cur_mcu = d_mcu[-1]
                cur_sw = d_sw[-1]
                cur_err = d_err[-1]
                mcu_turns = cur_mcu / 360.0
                sw_turns = cur_sw / 360.0
                self.status_label.setText(
                    f"pos={cur_pos:.2f}  |  MCU={cur_mcu:.1f} ({mcu_turns:.2f}T)  "
                    f"SW={cur_sw:.1f} ({sw_turns:.2f}T)  "
                    f"err={cur_err:.2f} deg  |  samples={n}"
                )
            else:
                self.status_label.setText(
                    f"pos={cur_pos:.2f} deg  |  samples={n}"
                )

    # ── Controls ──

    def _on_xrange_changed(self, text: str):
        if text == "All":
            self._x_window = 0.0
        else:
            self._x_window = float(text.replace(" s", ""))
        self._dirty = True

    def _on_autorange_toggled(self, checked: bool):
        self._auto_range = checked
        if checked:
            self._auto_resume_timer.stop()
            self._dirty = True

    def _on_manual_zoom(self):
        if getattr(self, '_programmatic_update', False):
            return
        if self._auto_range:
            self._auto_range = False
            self.autorange_chk.setChecked(False)
        self._auto_resume_timer.start()

    def _auto_resume(self):
        if not self._auto_range:
            self._auto_range = True
            self.autorange_chk.setChecked(True)
            self._dirty = True

    def _fit_all(self):
        old = self._x_window
        self._x_window = 0.0
        self._dirty = True
        self._render_frame()
        self._x_window = old

    def _on_fps_changed(self, text: str):
        fps = int(text)
        self._render_timer.setInterval(int(1000 / fps))

    def _on_clear(self):
        for rb in self._all_rbs:
            rb.clear()
        self._sample_count = 0
        self._prev_pos = None
        self._cumulative_turns = 0.0
        self._dirty = True
        self.status_label.setText("Cleared.")

    def showEvent(self, event):
        super().showEvent(event)

    def cleanup(self):
        if self._streaming:
            self._stop_streaming()
        self._render_timer.stop()
