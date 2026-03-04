"""
CAN Bus Health monitoring tab — real-time bus load, frame rate, and error graphs.
Pure read-only: polls transport cumulative counters + PCAN bus status, no CAN commands sent.

Layout:
  Left (graphs):           Right:
    Graph 1 — Bus Load       Per-Motor Traffic table
    Graph 2 — Frame Rate       (with Clear button)
    Graph 3 — Error Rate
"""

import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSplitter,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

from ..protocol.can_transport import (
    PcanTransport, PCAN_AVAILABLE,
)
from .plot_style import style_plot, graph_pen, COLORS, BG_DARK

# Import PCAN error constants (graceful if DLL unavailable)
_BUSOFF = 0x10
_BUSPASSIVE = 0x40000
_BUSHEAVY = 0x08
_BUSLIGHT = 0x04
if PCAN_AVAILABLE:
    try:
        from ..protocol.PCANBasic import (
            PCAN_ERROR_BUSOFF, PCAN_ERROR_BUSPASSIVE,
            PCAN_ERROR_BUSHEAVY, PCAN_ERROR_BUSLIGHT,
        )
        _BUSOFF = int(PCAN_ERROR_BUSOFF)
        _BUSPASSIVE = int(PCAN_ERROR_BUSPASSIVE)
        _BUSHEAVY = int(PCAN_ERROR_BUSHEAVY)
        _BUSLIGHT = int(PCAN_ERROR_BUSLIGHT)
    except Exception:
        pass


BUF_CAP = 3600  # ~60 min at 1 Hz


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


def _bus_state_label(status: int) -> tuple[str, str]:
    """Return (text, color_hex) for bus state."""
    if status & _BUSOFF:
        return "Bus-Off", COLORS["red"]
    if status & _BUSPASSIVE:
        return "Error Passive", COLORS["orange"]
    if status & _BUSHEAVY:
        return "Warning", COLORS["yellow"]
    if status & _BUSLIGHT:
        return "Bus Light", COLORS["yellow"]
    return "OK", COLORS["green"]


def _bus_state_y(status: int) -> float:
    """Map bus state to Y value for scatter plot."""
    if status & _BUSOFF:
        return 2.0
    if status & _BUSPASSIVE:
        return 1.0
    if status & (_BUSHEAVY | _BUSLIGHT):
        return 0.0
    return -1.0  # OK — no marker


class CanBusHealthTab(QWidget):
    def __init__(self, transport: PcanTransport):
        super().__init__()
        self._transport = transport
        self._polling = False
        self._t0 = 0.0

        # Previous poll state for delta calculation
        self._prev_rx = 0
        self._prev_tx = 0
        self._prev_errs = 0
        self._last_poll_time = 0.0

        # Time-series buffers
        self._time = _GrowBuffer(BUF_CAP)
        self._bus_load = _GrowBuffer(BUF_CAP)
        self._rx_rate = _GrowBuffer(BUF_CAP)
        self._tx_rate = _GrowBuffer(BUF_CAP)
        self._err_rate = _GrowBuffer(BUF_CAP)

        # Per-motor traffic tracking
        self._prev_per_motor: dict[int, tuple[int, int]] = {}

        # Bus state event markers (only non-OK states)
        self._evt_time = _GrowBuffer(BUF_CAP)
        self._evt_y = _GrowBuffer(BUF_CAP)
        self._evt_color = []  # list of QBrush for per-point color

        self._build_ui()

        # Poll timer
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_tick)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Controls ──
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
        self._combo_rate.addItems(["1 Hz", "2 Hz"])
        self._combo_rate.setCurrentIndex(0)
        self._combo_rate.setFixedWidth(80)
        ctrl.addWidget(self._combo_rate)

        ctrl.addWidget(QLabel("Window:"))
        self._combo_window = QComboBox()
        self._combo_window.addItems(["30 s", "60 s", "5 min", "All"])
        self._combo_window.setCurrentIndex(1)
        self._combo_window.setFixedWidth(80)
        ctrl.addWidget(self._combo_window)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedWidth(60)
        self._btn_clear.clicked.connect(self._on_clear)
        ctrl.addWidget(self._btn_clear)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ── Status labels ──
        status_group = QGroupBox("Current Values")
        sg = QHBoxLayout(status_group)
        sg.setContentsMargins(8, 4, 8, 4)

        label_font = QFont("Consolas", 11)
        sep_style = "color: #666666;"

        self._lbl_bus_state = QLabel("--")
        self._lbl_bus_state.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        self._lbl_bus_load = QLabel("--")
        self._lbl_bus_load.setFont(label_font)
        self._lbl_rx_rate = QLabel("--")
        self._lbl_rx_rate.setFont(label_font)
        self._lbl_tx_rate = QLabel("--")
        self._lbl_tx_rate.setFont(label_font)
        self._lbl_tx_errors = QLabel("--")
        self._lbl_tx_errors.setFont(label_font)
        self._lbl_err_rate = QLabel("--")
        self._lbl_err_rate.setFont(label_font)

        def _sep():
            s = QLabel("|")
            s.setFont(label_font)
            s.setStyleSheet(sep_style)
            return s

        sg.addWidget(QLabel("Bus State:"))
        sg.addWidget(self._lbl_bus_state)
        sg.addWidget(_sep())
        sg.addWidget(QLabel("Bus Load:"))
        sg.addWidget(self._lbl_bus_load)
        sg.addWidget(_sep())
        sg.addWidget(QLabel("RX Rate:"))
        sg.addWidget(self._lbl_rx_rate)
        sg.addWidget(_sep())
        sg.addWidget(QLabel("TX Rate:"))
        sg.addWidget(self._lbl_tx_rate)
        sg.addWidget(_sep())
        sg.addWidget(QLabel("TX Errors:"))
        sg.addWidget(self._lbl_tx_errors)
        sg.addWidget(_sep())
        sg.addWidget(QLabel("Error Rate:"))
        sg.addWidget(self._lbl_err_rate)
        sg.addStretch()

        root.addWidget(status_group)

        # ── Main area: Graphs (left) + Per-Motor Traffic (right) ──
        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.setChildrenCollapsible(False)

        # ── Graphs (left, vertical stack) ──
        graph_widget = QWidget()
        graph_lay = QVBoxLayout(graph_widget)
        graph_lay.setContentsMargins(0, 0, 0, 0)
        graph_lay.setSpacing(4)

        # Graph 1: Bus Load (%)
        self._pw_load = pg.PlotWidget()
        style_plot(self._pw_load, title="Bus Load",
                   left_label="Load", left_unit="%",
                   bottom_label="Time", bottom_unit="s")
        self._pw_load.setYRange(0, 100)
        self._pw_load.getPlotItem().getViewBox().setLimits(yMin=0, yMax=100)
        # Warning thresholds
        self._line_warn30 = pg.InfiniteLine(
            pos=30, angle=0, pen=pg.mkPen(COLORS["yellow"], width=1,
                                           style=Qt.PenStyle.DashLine))
        self._line_warn70 = pg.InfiniteLine(
            pos=70, angle=0, pen=pg.mkPen(COLORS["red"], width=1,
                                           style=Qt.PenStyle.DashLine))
        self._pw_load.addItem(self._line_warn30)
        self._pw_load.addItem(self._line_warn70)
        self._curve_load = self._pw_load.plot(
            pen=graph_pen(5, width=2.0), name="Bus Load")  # cyan
        graph_lay.addWidget(self._pw_load, stretch=3)

        # Graph 2: Frame Rate (fps)
        self._pw_fps = pg.PlotWidget()
        style_plot(self._pw_fps, title="Frame Rate",
                   left_label="Rate", left_unit="fps",
                   bottom_label="Time", bottom_unit="s")
        self._curve_rx = self._pw_fps.plot(
            pen=graph_pen(2, width=1.5), name="RX")  # green
        self._curve_tx = self._pw_fps.plot(
            pen=graph_pen(0, width=1.5), name="TX")  # blue
        # Add legend
        legend_fps = self._pw_fps.addLegend(offset=(10, 10))
        legend_fps.setBrush(pg.mkBrush(0, 0, 0, 100))
        legend_fps.setPen(pg.mkPen(COLORS["grey"]))
        graph_lay.addWidget(self._pw_fps, stretch=2)

        # Graph 3: Error Rate + Bus State events
        self._pw_err = pg.PlotWidget()
        style_plot(self._pw_err, title="Send Error Rate / Bus State",
                   left_label="Error Rate", left_unit="err/s",
                   bottom_label="Time", bottom_unit="s")
        self._curve_err = self._pw_err.plot(
            pen=graph_pen(3, width=1.5), name="Send Err/s")  # orange

        # Bus state scatter on right Y axis
        self._vb_state = pg.ViewBox()
        self._pw_err.getPlotItem().scene().addItem(self._vb_state)
        self._pw_err.getPlotItem().getAxis('right').linkToView(self._vb_state)
        self._vb_state.setXLink(self._pw_err)
        ax_right = self._pw_err.getPlotItem().getAxis('right')
        ax_right.setLabel("Bus State", color=COLORS["red"])
        ax_right.show()
        self._vb_state.setYRange(-0.5, 2.5)
        ytick = ax_right
        ytick.setTicks([[(0.0, 'Warning'), (1.0, 'ErrPassive'), (2.0, 'Bus-Off')]])

        self._scatter_state = pg.ScatterPlotItem(size=14, symbol='x')
        self._vb_state.addItem(self._scatter_state)
        self._pw_err.getPlotItem().vb.sigResized.connect(self._sync_viewbox)

        graph_lay.addWidget(self._pw_err, stretch=2)

        main_split.addWidget(graph_widget)

        # ── Per-Motor Traffic Table (right) ──
        motor_group = QGroupBox("Per-Motor Traffic")
        mg_lay = QVBoxLayout(motor_group)
        mg_lay.setContentsMargins(4, 4, 4, 4)

        btn_clear_motor = QPushButton("Clear")
        btn_clear_motor.setFixedWidth(60)
        btn_clear_motor.clicked.connect(self._clear_per_motor)
        motor_hdr = QHBoxLayout()
        motor_hdr.addStretch()
        motor_hdr.addWidget(btn_clear_motor)
        mg_lay.addLayout(motor_hdr)

        self._motor_table = QTableWidget(0, 6)
        self._motor_table.setHorizontalHeaderLabels(
            ["Motor ID", "RX", "TX", "RX/s", "TX/s", "Total"])
        self._motor_table.horizontalHeader().setStretchLastSection(True)
        self._motor_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._motor_table.verticalHeader().setVisible(False)
        self._motor_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._motor_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self._motor_table.setShowGrid(True)
        self._motor_table.setStyleSheet(
            "QTableWidget { gridline-color: #666666; }"
            "QHeaderView::section { border: 1px solid #666666; }")
        self._motor_table.setFont(QFont("Consolas", 10))
        mg_lay.addWidget(self._motor_table)

        motor_group.setMinimumWidth(280)
        main_split.addWidget(motor_group)

        # Equal stretch so resizing keeps 50:50 ratio
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 1)
        self._main_split = main_split

        root.addWidget(main_split, stretch=1)

    def showEvent(self, event):
        super().showEvent(event)
        # Force 50:50 split once the widget has a real width
        w = self._main_split.width()
        if w > 0:
            half = w // 2
            self._main_split.setSizes([half, half])

    def _sync_viewbox(self):
        self._vb_state.setGeometry(
            self._pw_err.getPlotItem().vb.sceneBoundingRect())

    # ── Controls ──────────────────────────────────────────────────────

    def _on_start(self):
        if self._polling:
            return
        self._polling = True
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)

        # Snapshot current cumulative counts as baseline
        rx, tx, errs = self._transport.get_cumulative_counts()
        self._prev_rx = rx
        self._prev_tx = tx
        self._prev_errs = errs
        self._last_poll_time = time.monotonic()
        self._t0 = time.monotonic()

        rate_text = self._combo_rate.currentText()
        hz = 2 if "2" in rate_text else 1
        self._poll_timer.start(int(1000 / hz))

    def _on_stop(self):
        self._polling = False
        self._poll_timer.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)

    def _on_clear(self):
        for buf in (self._time, self._bus_load, self._rx_rate,
                    self._tx_rate, self._err_rate,
                    self._evt_time, self._evt_y):
            buf.clear()
        self._evt_color.clear()
        self._curve_load.setData([], [])
        self._curve_rx.setData([], [])
        self._curve_tx.setData([], [])
        self._curve_err.setData([], [])
        self._scatter_state.setData([], [])
        for lbl in (self._lbl_bus_state, self._lbl_bus_load,
                    self._lbl_rx_rate, self._lbl_tx_rate,
                    self._lbl_tx_errors, self._lbl_err_rate):
            lbl.setText("--")
        self._lbl_bus_state.setStyleSheet("")
        self._motor_table.setRowCount(0)
        self._prev_per_motor.clear()

    def _clear_per_motor(self):
        self._transport.clear_per_motor_counts()
        self._motor_table.setRowCount(0)
        self._prev_per_motor.clear()

    # ── Polling ───────────────────────────────────────────────────────

    def _poll_tick(self):
        now = time.monotonic()
        dt = now - self._last_poll_time
        if dt < 0.01:
            return  # guard against zero division
        self._last_poll_time = now
        t = now - self._t0

        rx, tx, errs = self._transport.get_cumulative_counts()

        d_rx = rx - self._prev_rx
        d_tx = tx - self._prev_tx
        d_err = errs - self._prev_errs
        self._prev_rx = rx
        self._prev_tx = tx
        self._prev_errs = errs

        rx_fps = d_rx / dt
        tx_fps = d_tx / dt
        err_rate = d_err / dt

        # Bus load: (total_frames * 111 bits) / 1_000_000 bps * 100
        bus_load = ((rx_fps + tx_fps) * 111) / 1_000_000 * 100
        bus_load = min(bus_load, 100.0)

        bus_status = self._transport.get_bus_status()

        # Append to buffers
        self._time.append(t)
        self._bus_load.append(bus_load)
        self._rx_rate.append(rx_fps)
        self._tx_rate.append(tx_fps)
        self._err_rate.append(err_rate)

        # Compact if over capacity
        if len(self._time) > BUF_CAP:
            for buf in (self._time, self._bus_load, self._rx_rate,
                        self._tx_rate, self._err_rate):
                buf.compact(BUF_CAP)

        # Bus state event markers (only non-OK)
        state_y = _bus_state_y(bus_status)
        if state_y >= 0:
            self._evt_time.append(t)
            self._evt_y.append(state_y)
            if state_y >= 2.0:
                self._evt_color.append(pg.mkBrush(COLORS["red"]))
            elif state_y >= 1.0:
                self._evt_color.append(pg.mkBrush(COLORS["orange"]))
            else:
                self._evt_color.append(pg.mkBrush(COLORS["yellow"]))

        # Update status labels
        state_text, state_color = _bus_state_label(bus_status)
        self._lbl_bus_state.setText(state_text)
        self._lbl_bus_state.setStyleSheet(f"color: {state_color};")
        self._lbl_bus_load.setText(f"{bus_load:.1f} %")
        self._lbl_rx_rate.setText(f"{rx_fps:.0f} fps")
        self._lbl_tx_rate.setText(f"{tx_fps:.0f} fps")
        self._lbl_tx_errors.setText(f"{errs}")
        self._lbl_err_rate.setText(f"{err_rate:.1f} err/s")

        # Color bus load label
        if bus_load > 70:
            self._lbl_bus_load.setStyleSheet(f"color: {COLORS['red']};")
        elif bus_load > 30:
            self._lbl_bus_load.setStyleSheet(f"color: {COLORS['yellow']};")
        else:
            self._lbl_bus_load.setStyleSheet(f"color: {COLORS['green']};")

        # ── Update per-motor table ──
        per_motor = self._transport.get_per_motor_counts()
        motor_ids = sorted(per_motor.keys())
        self._motor_table.setRowCount(len(motor_ids))
        for row, mid in enumerate(motor_ids):
            rx_cnt, tx_cnt = per_motor[mid]
            prev_rx_m, prev_tx_m = self._prev_per_motor.get(mid, (rx_cnt, tx_cnt))
            rx_rate_m = (rx_cnt - prev_rx_m) / dt
            tx_rate_m = (tx_cnt - prev_tx_m) / dt

            items = [
                str(mid),
                str(rx_cnt),
                str(tx_cnt),
                f"{rx_rate_m:.1f}",
                f"{tx_rate_m:.1f}",
                str(rx_cnt + tx_cnt),
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._motor_table.setItem(row, col, item)
        self._prev_per_motor = {mid: per_motor[mid] for mid in motor_ids}

        # ── Update graphs ──
        ta = self._time.array()
        window = self._get_window_sec()
        if window > 0 and len(ta) > 0:
            t_min = ta[-1] - window
            mask = ta >= t_min
            tx_arr = ta[mask]
            self._curve_load.setData(tx_arr, self._bus_load.array()[mask])
            self._curve_rx.setData(tx_arr, self._rx_rate.array()[mask])
            self._curve_tx.setData(tx_arr, self._tx_rate.array()[mask])
            self._curve_err.setData(tx_arr, self._err_rate.array()[mask])
        else:
            self._curve_load.setData(ta, self._bus_load.array())
            self._curve_rx.setData(ta, self._rx_rate.array())
            self._curve_tx.setData(ta, self._tx_rate.array())
            self._curve_err.setData(ta, self._err_rate.array())

        # Bus state scatter
        if len(self._evt_time) > 0:
            et = self._evt_time.array()
            ey = self._evt_y.array()
            if window > 0 and len(et) > 0:
                t_min_evt = ta[-1] - window if len(ta) > 0 else 0
                evt_mask = et >= t_min_evt
                et = et[evt_mask]
                ey = ey[evt_mask]
                brushes = [b for b, m in zip(self._evt_color, evt_mask) if m]
            else:
                brushes = list(self._evt_color)
            if len(et) > 0:
                self._scatter_state.setData(
                    et, ey, brush=brushes,
                    pen=pg.mkPen(None))
            else:
                self._scatter_state.setData([], [])
        else:
            self._scatter_state.setData([], [])

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_window_sec(self) -> int:
        text = self._combo_window.currentText()
        if "All" in text:
            return 0
        if "min" in text:
            return int(text.split()[0]) * 60
        return int(text.split()[0])

    def cleanup(self):
        self._on_stop()
