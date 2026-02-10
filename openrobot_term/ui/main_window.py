"""
Main window with tab-based layout, docked motor control panel, and VESC packet dispatcher.
CAN-only: all VESC communication via PCAN-USB EID multi-frame protocol.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QDockWidget,
    QTextEdit, QSplitter, QLabel, QHBoxLayout, QPushButton,
)
from PyQt6.QtGui import QShortcut, QKeySequence, QFont
from PyQt6.QtCore import pyqtSlot, Qt, QTimer

from ..protocol.commands import CommPacketId, VescValues, McconfPid
from ..protocol.can_transport import PcanTransport
from .connection_bar import ConnectionBar
from .terminal_tab import TerminalTab
from .firmware_tab import FirmwareTab
from .realtime_tab import RealtimeTab
from .waveform_tab import WaveformTab
from .parameter_tab import ParameterTab
from .position_tab import PositionTab
from .experiment_tab import ExperimentTab
from .can_control_tab import CanControlTab
from .can_data_tab import CanDataTab
from .can_position_tuning_tab import CanPositionTuningTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenRobot Motor Analyzer")

        self.can_transport = PcanTransport()

        # Central widget: connection bar + tabs
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Connection bar (CAN only)
        self.conn_bar = ConnectionBar(self.can_transport)
        layout.addWidget(self.conn_bar)

        # Splitter for tabs + debug panel
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self.main_splitter)

        # Tab widget (center)
        self.tabs = QTabWidget()
        self.main_splitter.addWidget(self.tabs)

        # Debug panel (hidden by default)
        self._debug_panel = self._create_debug_panel()
        self.main_splitter.addWidget(self._debug_panel)
        self._debug_panel.hide()

        # Set splitter sizes (tabs take most space)
        self.main_splitter.setSizes([700, 200])

        # Create tabs (Parameter first)
        self.parameter_tab = ParameterTab(self.can_transport)
        self.realtime_tab = RealtimeTab(self.can_transport)
        self.can_data_tab = CanDataTab(self.can_transport)
        self.experiment_tab = ExperimentTab(self.can_transport)
        self.position_tab = PositionTab(self.can_transport)
        self.waveform_tab = WaveformTab(self.can_transport)
        self.analysis_tab = CanPositionTuningTab(self.can_transport)
        self.firmware_tab = FirmwareTab(self.can_transport)

        self.tabs.addTab(self.parameter_tab, "Parameter")
        self.tabs.addTab(self.realtime_tab, "Real-time Data")
        self.tabs.addTab(self.can_data_tab, "CAN Data")
        self.tabs.addTab(self.experiment_tab, "Experiment Data")
        self.tabs.addTab(self.position_tab, "Position")
        self.tabs.addTab(self.waveform_tab, "Waveform")
        self.tabs.addTab(self.analysis_tab, "AI Analysis")
        self.tabs.addTab(self.firmware_tab, "Firmware")

        # Right dock spans full height (top-right & bottom-right corners belong to right)
        self.setCorner(Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea, QTabWidget.TabPosition.North)

        # Unified Motor Control dock (RMD + VESC EID combined)
        self.can_control_tab = CanControlTab(self.can_transport)
        self.motor_dock = QDockWidget("Motor Control", self)
        self.motor_dock.setWidget(self.can_control_tab)
        self.motor_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.motor_dock)
        self.motor_dock.setMinimumWidth(320)

        # Terminal dock (bottom-left, below tabs)
        self.terminal_tab = TerminalTab(self.can_transport)
        self.terminal_dock = QDockWidget("Terminal", self)
        self.terminal_dock.setWidget(self.terminal_tab)
        self.terminal_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.terminal_dock)
        self.resizeDocks([self.terminal_dock], [250], Qt.Orientation.Vertical)
        self.resizeDocks([self.motor_dock], [340], Qt.Orientation.Horizontal)

        # Wire parameter tab log messages to terminal
        self.parameter_tab.log_message.connect(self.terminal_tab.log)

        # APPCONF CAN ID change → re-scan (firmware applies new ID immediately)
        self.parameter_tab.rescan_requested.connect(
            self.conn_bar.rescan_after_id_change
        )

        # CAN connection state updates
        self.conn_bar.can_connection_changed.connect(self._on_can_connection_changed)

        # CAN log → terminal
        self.can_transport.log_message.connect(self.terminal_tab.log)

        # CAN scan results → tabs with CAN target combos
        self.conn_bar.can_targets_updated.connect(self.can_data_tab.update_discovered_ids)
        self.conn_bar.can_targets_updated.connect(self.can_control_tab.update_discovered_ids)
        self.conn_bar.can_targets_updated.connect(self.firmware_tab.update_can_targets)
        self.conn_bar.can_targets_updated.connect(self._on_can_targets_found)

        # VESC EID response dispatch (CAN → all tabs)
        self.can_transport.vesc_response_received.connect(self._dispatch_vesc_response)

        # CAN Control torque/pos cmd → CAN Data graph
        self.can_control_tab.torque_cmd_sent.connect(self.can_data_tab._on_torque_cmd)
        self.can_control_tab.pos_cmd_sent.connect(self.can_data_tab._on_pos_cmd)

        # CAN frame → AI Analysis tab (for PID read responses)
        self.can_transport.frame_received.connect(self.analysis_tab.on_frame_received)

        # Keyboard shortcut: Ctrl+Shift+I to toggle debug panel
        self._debug_shortcut = QShortcut(QKeySequence("Ctrl+Shift+I"), self)
        self._debug_shortcut.activated.connect(self._toggle_debug_panel)

        # Debug buffer for recent data (keep last 100 entries even when panel is hidden)
        self._debug_buffer = []
        self._debug_buffer_max = 100

    @pyqtSlot(int, bytes)
    def _dispatch_vesc_response(self, sender_id: int, payload: bytes):
        """Route VESC EID responses (received via CAN) to appropriate tabs."""
        if not payload:
            return

        cmd_id = payload[0]
        data = payload[1:]

        if cmd_id == CommPacketId.COMM_GET_VALUES:
            values = VescValues.from_payload(data)
            self.realtime_tab.on_values(values)
            self.can_control_tab.on_values(values)

        elif cmd_id == CommPacketId.COMM_ROTOR_POSITION:
            self.position_tab.on_rotor_position(data)

        elif cmd_id == CommPacketId.COMM_SAMPLE_PRINT:
            self.waveform_tab.on_sample_data(data)

        elif cmd_id == CommPacketId.COMM_SET_MCCONF:
            self.terminal_tab.log(f"[MCCONF] Write OK (from ID {sender_id})")

        elif cmd_id == CommPacketId.COMM_GET_MCCONF:
            pid_values = McconfPid.from_mcconf_payload(data)
            self.analysis_tab.on_mcconf_received(pid_values, raw_data=data)
            self.parameter_tab.on_mcconf_received(data, is_default=False)
            self.terminal_tab.log(
                f"[MCCONF] Received {len(data)} bytes from ID {sender_id}"
            )
            # Sync foc_encoder_ratio → CAN Data & AI Analysis pole pairs
            mcvals = self.parameter_tab._mcconf_values
            if mcvals and "foc_encoder_ratio" in mcvals:
                pp = int(mcvals["foc_encoder_ratio"])
                self.can_data_tab.set_pole_pairs(pp)
                self.analysis_tab.set_pole_pairs(pp)

        elif cmd_id == CommPacketId.COMM_GET_MCCONF_DEFAULT:
            pid_values = McconfPid.from_mcconf_payload(data)
            self.analysis_tab.on_mcconf_received(pid_values, is_default=True)
            self.terminal_tab.log(
                f"[MCCONF DEFAULT] Received {len(data)} bytes from ID {sender_id}"
            )

        elif cmd_id == CommPacketId.COMM_SET_APPCONF:
            self.terminal_tab.log(f"[APPCONF] Write OK (from ID {sender_id})")

        elif cmd_id == CommPacketId.COMM_GET_APPCONF:
            self.parameter_tab.on_appconf_received(data, is_default=False)
            self.terminal_tab.log(
                f"[APPCONF] Received {len(data)} bytes from ID {sender_id}"
            )

        elif cmd_id == CommPacketId.COMM_GET_APPCONF_DEFAULT:
            self.parameter_tab.on_appconf_received(data, is_default=True)
            self.terminal_tab.log(
                f"[APPCONF DEFAULT] Received {len(data)} bytes from ID {sender_id}"
            )

        elif cmd_id == CommPacketId.COMM_FW_VERSION:
            if len(data) >= 2:
                major, minor = data[0], data[1]
                hw_name = ""
                if len(data) > 12:
                    try:
                        name_len = data[12]
                        if name_len > 0 and len(data) > 13 + name_len:
                            hw_name = data[13:13 + name_len].decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                msg = f"[FW] Version {major}.{minor}"
                if hw_name:
                    msg += f" ({hw_name})"
                msg += f" from ID {sender_id}"
                self.terminal_tab.log(msg)

        elif cmd_id == CommPacketId.COMM_TERMINAL_CMD or cmd_id == CommPacketId.COMM_TERMINAL_CMD_SYNC:
            text = data.decode("utf-8", errors="ignore").rstrip("\x00")
            if text:
                self.terminal_tab.log(text)

        elif cmd_id == CommPacketId.COMM_PRINT:
            text = data.decode("utf-8", errors="ignore").rstrip("\x00")
            if text:
                self.terminal_tab.log(text)

        elif cmd_id == CommPacketId.COMM_PLOT_INIT:
            self.experiment_tab.on_plot_init(data)

        elif cmd_id == CommPacketId.COMM_PLOT_ADD_GRAPH:
            self.experiment_tab.on_plot_add_graph(data)

        elif cmd_id == CommPacketId.COMM_PLOT_SET_GRAPH:
            self.experiment_tab.on_plot_set_graph(data)

        elif cmd_id == CommPacketId.COMM_PLOT_DATA:
            self.experiment_tab.on_plot_data(data)

        elif cmd_id == CommPacketId.COMM_EXPERIMENT_SAMPLE:
            self.experiment_tab.on_experiment_sample(data)

        elif cmd_id in (
            CommPacketId.COMM_ERASE_NEW_APP,
            CommPacketId.COMM_WRITE_NEW_APP_DATA,
            CommPacketId.COMM_ERASE_BOOTLOADER,
            CommPacketId.COMM_JUMP_TO_BOOTLOADER,
        ):
            # Firmware upload responses — handled by CanFirmwareUploader
            pass

        else:
            # Handle custom firmware responses or unknown packet types
            try:
                text = data.decode("utf-8", errors="ignore").rstrip("\x00")
                printable_ratio = sum(1 for c in text if c.isprintable() or c in '\n\r\t') / max(len(text), 1)
                if text and printable_ratio > 0.7:
                    self.terminal_tab.log(text)
                else:
                    self.terminal_tab.log(
                        f"[CAN EID] cmd={cmd_id}, {len(data)} bytes from ID {sender_id}"
                    )
            except Exception:
                self.terminal_tab.log(
                    f"[CAN EID] cmd={cmd_id}, {len(data)} bytes from ID {sender_id}"
                )

    def _on_can_targets_found(self, targets: list):
        """Auto-read MCCONF/APPCONF after CAN scan discovers devices."""
        if targets:
            self.terminal_tab.log("[INFO] Auto-reading MCCONF/APPCONF...")
            QTimer.singleShot(100, self.parameter_tab._on_read_mcconf)
            QTimer.singleShot(500, self.parameter_tab._on_read_appconf)

    def _on_can_connection_changed(self, connected: bool):
        if connected:
            self.terminal_tab.log("[INFO] PCAN Connected (1 Mbps)")
        else:
            self.terminal_tab.log("[INFO] PCAN Disconnected.")
            self.realtime_tab.stop_polling()
            self.experiment_tab.cleanup()
            self.position_tab.cleanup()
            self.can_control_tab.cleanup()

    def _create_debug_panel(self) -> QWidget:
        """Create the CAN debug panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header row
        header = QHBoxLayout()
        title = QLabel("CAN Debug (Ctrl+Shift+I to hide)")
        title.setStyleSheet("font-weight: bold; color: #aaa;")
        header.addWidget(title)
        header.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(60)
        clear_btn.clicked.connect(lambda: self._debug_text.clear())
        header.addWidget(clear_btn)
        layout.addLayout(header)

        # Debug text area
        self._debug_text = QTextEdit()
        self._debug_text.setReadOnly(True)
        self._debug_text.setFont(QFont("Consolas", 9))
        self._debug_text.setStyleSheet(
            "QTextEdit { background: #1a1a1a; color: #ddd; border: 1px solid #333; }"
        )
        self._debug_text.setMaximumHeight(180)
        layout.addWidget(self._debug_text)

        return panel

    def _toggle_debug_panel(self):
        """Toggle debug panel visibility."""
        if self._debug_panel.isVisible():
            self._debug_panel.hide()
        else:
            # Show buffered data when opening panel
            if self._debug_buffer:
                for entry in self._debug_buffer:
                    self._debug_text.append(entry)
                self._debug_buffer.clear()
                self._debug_text.verticalScrollBar().setValue(
                    self._debug_text.verticalScrollBar().maximum()
                )
            self._debug_panel.show()

    def closeEvent(self, event):
        self.realtime_tab.cleanup()
        self.experiment_tab.cleanup()
        self.position_tab.cleanup()
        self.analysis_tab.cleanup()
        self.can_control_tab.cleanup()
        self.can_data_tab.cleanup()
        self.can_transport.disconnect()
        event.accept()
