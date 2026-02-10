"""
Experiment Data tab -- real-time graph from VESC COMM_PLOT_* commands.

Firmware sends:
  COMM_PLOT_INIT:       reset plot, set axis labels (nameX, nameY)
  COMM_PLOT_ADD_GRAPH:  add a named curve
  COMM_PLOT_SET_GRAPH:  select active curve index for subsequent data
  COMM_PLOT_DATA:       (x, y) float32 data point for the active curve
  COMM_EXPERIMENT_SAMPLE: array of int32/10000 values (legacy)
"""

import struct
import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QComboBox,
)
from PyQt6.QtCore import Qt, QTimer

from ..protocol.can_transport import PcanTransport
from ..protocol.commands import build_terminal_cmd
from .plot_style import style_plot, graph_pen, style_legend, TEXT_NORMAL

RENDER_INTERVAL_MS = 33  # ~30 fps


class ExperimentTab(QWidget):
    def __init__(self, transport: PcanTransport):
        super().__init__()
        self._transport = transport
        self._dirty = False
        self._auto_range = True
        self._frame_count = 0
        self._programmatic_update = False

        self._streaming = False

        # Graph state
        self._graphs = []       # list of {"name", "x", "y", "curve"}
        self._active_graph = 0
        self._x_label = "Sample"
        self._y_label = "Value"

        self._build_ui()

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(RENDER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._render_frame)
        self._render_timer.start()

        # Auto-resume timer
        self._auto_resume_timer = QTimer(self)
        self._auto_resume_timer.setSingleShot(True)
        self._auto_resume_timer.setInterval(3000)
        self._auto_resume_timer.timeout.connect(self._auto_resume)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Controls row
        ctrl = QHBoxLayout()
        layout.addLayout(ctrl)

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

        self.autorange_chk = QCheckBox("Auto Y")
        self.autorange_chk.setChecked(True)
        self.autorange_chk.toggled.connect(self._on_autorange_toggled)
        ctrl.addWidget(self.autorange_chk)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("Reset plot to full data range")
        self.fit_btn.clicked.connect(self._fit_all)
        ctrl.addWidget(self.fit_btn)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Graph FPS:"))
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["10", "15", "20", "30", "60"])
        self.fps_combo.setCurrentText("30")
        self.fps_combo.currentTextChanged.connect(self._on_fps_changed)
        ctrl.addWidget(self.fps_combo)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        style_plot(self.plot_widget, title="Experiment Data",
                   left_label="Value", left_unit="",
                   bottom_label="Sample", bottom_unit="")
        self._legend = style_legend(self.plot_widget)
        self.plot_widget.disableAutoRange()
        layout.addWidget(self.plot_widget, stretch=1)

        self.plot_widget.getPlotItem().getViewBox().sigRangeChangedManually.connect(
            self._on_manual_zoom)

        # Status label
        self.status_label = QLabel("Waiting for experiment data...")
        self.status_label.setStyleSheet("font-family: monospace; font-size: 12px; padding: 4px;")
        layout.addWidget(self.status_label)

    # ── Streaming control ──

    def _toggle_streaming(self):
        if not self._transport.is_connected():
            self.status_label.setText("Not connected")
            return
        # "or_rp" toggles streaming on/off in firmware
        self._transport.send_vesc_to_target(build_terminal_cmd("or_rp"))
        if self._streaming:
            self._set_streaming(False)
        else:
            self._set_streaming(True)

    def _set_streaming(self, active: bool):
        self._streaming = active
        if active:
            self.start_btn.setText("Stop Streaming")
            self.start_btn.setStyleSheet(
                "QPushButton { background: #B71C1C; color: white; font-weight: bold; padding: 4px 12px; }"
                "QPushButton:hover { background: #D32F2F; }"
            )
            pass  # status will be updated by render frame
            self.status_label.setStyleSheet(
                "font-family: monospace; font-size: 12px; padding: 4px; color: #66ff66;")
        else:
            self.start_btn.setText("Start Streaming")
            self.start_btn.setStyleSheet(
                "QPushButton { background: #1B5E20; color: white; font-weight: bold; padding: 4px 12px; }"
                "QPushButton:hover { background: #2E7D32; }"
            )
            self.status_label.setStyleSheet(
                "font-family: monospace; font-size: 12px; padding: 4px;")

    # ── Data handlers (called from main_window dispatch) ──

    def on_plot_init(self, data: bytes):
        """COMM_PLOT_INIT: reset plot and set axis names."""
        try:
            parts = data.split(b'\x00')
            self._x_label = parts[0].decode("utf-8", errors="ignore") if len(parts) > 0 else "X"
            self._y_label = parts[1].decode("utf-8", errors="ignore") if len(parts) > 1 else "Y"
        except Exception:
            self._x_label = "X"
            self._y_label = "Y"

        self._clear_graphs()

        plot = self.plot_widget.getPlotItem()
        plot.setLabel("bottom", self._x_label, "",
                      **{"color": TEXT_NORMAL, "font-size": "9pt"})
        plot.setLabel("left", self._y_label, "",
                      **{"color": TEXT_NORMAL, "font-size": "9pt"})
        plot.setTitle(f"Experiment Data ({self._x_label} vs {self._y_label})",
                      color="#DCDCDC", size="10pt")

        # Auto-detect streaming started (firmware sent PLOT_INIT)
        if not self._streaming:
            self._set_streaming(True)

        self.status_label.setText(f"Plot initialized: X={self._x_label}, Y={self._y_label}")

    def on_plot_add_graph(self, data: bytes):
        """COMM_PLOT_ADD_GRAPH: add a named curve."""
        name = data.decode("utf-8", errors="ignore").rstrip("\x00")
        idx = len(self._graphs)
        curve = self.plot_widget.plot(pen=graph_pen(idx), name=name)
        self._graphs.append({
            "name": name,
            "x": [],
            "y": [],
            "curve": curve,
        })
        self.status_label.setText(
            f"Graphs: {', '.join(g['name'] for g in self._graphs)}")

    def on_plot_set_graph(self, data: bytes):
        """COMM_PLOT_SET_GRAPH: set active graph index."""
        if len(data) >= 1:
            idx = data[0]
            # Auto-create graphs if PLOT_INIT/ADD_GRAPH were missed
            # (e.g., tool connected after firmware boot)
            while len(self._graphs) <= idx:
                i = len(self._graphs)
                curve = self.plot_widget.plot(pen=graph_pen(i), name=str(i + 1))
                self._graphs.append({"name": str(i + 1), "x": [], "y": [], "curve": curve})
            self._active_graph = idx

    def on_plot_data(self, data: bytes):
        """COMM_PLOT_DATA: add (x, y) point to active graph."""
        if len(data) < 8:
            return

        # Auto-detect streaming if PLOT_INIT was missed
        if not self._streaming:
            self._set_streaming(True)

        x = struct.unpack_from(">f", data, 0)[0]
        y = struct.unpack_from(">f", data, 4)[0]

        if 0 <= self._active_graph < len(self._graphs):
            g = self._graphs[self._active_graph]
            g["x"].append(x)
            g["y"].append(y)
            self._dirty = True

    def on_experiment_sample(self, data: bytes):
        """COMM_EXPERIMENT_SAMPLE: array of int32/10000 values."""
        if len(data) < 4:
            return
        n_vals = len(data) // 4

        # Auto-create graphs if not yet initialized
        if not self._graphs:
            for i in range(n_vals):
                curve = self.plot_widget.plot(pen=graph_pen(i), name=f"Ch {i}")
                self._graphs.append({"name": f"Ch {i}", "x": [], "y": [], "curve": curve})

        x_val = float(len(self._graphs[0]["x"])) if self._graphs else 0.0

        for i in range(min(n_vals, len(self._graphs))):
            val = struct.unpack_from(">i", data, i * 4)[0] / 10000.0
            self._graphs[i]["x"].append(x_val)
            self._graphs[i]["y"].append(val)

        self._dirty = True

    # ── Rendering ──

    def _render_frame(self):
        if not self._dirty:
            return
        self._dirty = False
        self._frame_count += 1

        if not self._graphs:
            return

        self._programmatic_update = True

        total_points = 0
        y_min, y_max = float('inf'), float('-inf')
        x_min, x_max = float('inf'), float('-inf')

        for g in self._graphs:
            n = len(g["x"])
            if n == 0:
                continue
            xa = np.array(g["x"], dtype=np.float64)
            ya = np.array(g["y"], dtype=np.float64)
            g["curve"].setData(xa, ya, skipFiniteCheck=True)
            total_points += n
            y_min = min(y_min, float(ya.min()))
            y_max = max(y_max, float(ya.max()))
            x_min = min(x_min, float(xa.min()))
            x_max = max(x_max, float(xa.max()))

        if total_points > 0:
            x_pad = (x_max - x_min) * 0.02 if x_max != x_min else 1.0
            self.plot_widget.setXRange(x_min - x_pad, x_max + x_pad, padding=0)

            if self._auto_range:
                y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
                self.plot_widget.setYRange(y_min - y_margin, y_max + y_margin, padding=0)

        self._programmatic_update = False

        # Status update every 4 frames
        if (self._frame_count & 3) == 0 and total_points > 0:
            parts = []
            for g in self._graphs:
                n = len(g["y"])
                last_y = f"{g['y'][-1]:.3f}" if n > 0 else "—"
                parts.append(f"{g['name']}={last_y}({n})")
            self.status_label.setText(
                f"Graphs: {len(self._graphs)}  |  {', '.join(parts)}  |  "
                f"Total: {total_points}")

    # ── Controls ──

    def _on_clear(self):
        self._clear_graphs()
        self.status_label.setText("Cleared.")
        self._dirty = True
        self._render_frame()

    def _clear_graphs(self):
        for g in self._graphs:
            self.plot_widget.removeItem(g["curve"])
        self._graphs.clear()
        self._active_graph = 0

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
        old_auto = self._auto_range
        self._auto_range = True
        self._dirty = True
        self._render_frame()
        self._auto_range = old_auto

    def _on_fps_changed(self, text: str):
        fps = int(text)
        self._render_timer.setInterval(int(1000 / fps))

    def cleanup(self):
        if self._streaming:
            self._set_streaming(False)
        self._render_timer.stop()
