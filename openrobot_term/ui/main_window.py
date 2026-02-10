"""
Main window with tab-based layout, docked motor control panel, and VESC packet dispatcher.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QDockWidget,
    QTextEdit, QSplitter, QLabel, QHBoxLayout, QPushButton,
)
from PyQt6.QtGui import QIcon, QShortcut, QKeySequence, QFont
from PyQt6.QtCore import pyqtSlot, Qt, QTimer
from datetime import datetime

from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import CommPacketId, VescValues, WaveformSamples, McconfPid
from ..protocol.can_transport import PcanTransport
from .connection_bar import ConnectionBar
from .terminal_tab import TerminalTab
from .firmware_tab import FirmwareTab
from .realtime_tab import RealtimeTab
from .waveform_tab import WaveformTab
from .analysis_tab import AnalysisTab
from .motor_control_tab import MotorControlTab
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

        self.transport = SerialTransport()
        self.can_transport = PcanTransport()

        # Central widget: connection bar + tabs
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Connection bar (shared, top — serial + PCAN)
        self.conn_bar = ConnectionBar(self.transport, self.can_transport)
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
        self.parameter_tab = ParameterTab(self.transport)
        self.realtime_tab = RealtimeTab(self.transport)
        self.experiment_tab = ExperimentTab(self.transport)
        self.position_tab = PositionTab(self.transport)
        self.waveform_tab = WaveformTab(self.transport)
        self.analysis_tab = AnalysisTab(self.transport)
        self.firmware_tab = FirmwareTab(self.transport)

        # CAN tabs
        self.can_data_tab = CanDataTab(self.can_transport)
        self.can_pos_tuning_tab = CanPositionTuningTab(self.can_transport, self.transport)
        self.tabs.addTab(self.parameter_tab, "Parameter")
        self.tabs.addTab(self.realtime_tab, "Real-time Data")
        self.tabs.addTab(self.experiment_tab, "Experiment Data")
        self.tabs.addTab(self.position_tab, "Position")
        self.tabs.addTab(self.waveform_tab, "Waveform")
        self.tabs.addTab(self.analysis_tab, "AI Analysis")
        self.tabs.addTab(self.firmware_tab, "Firmware")

        # Separator before CAN tabs
        sep = QWidget()
        sep.setFixedWidth(0)
        sep.setEnabled(False)
        sep_idx = self.tabs.addTab(sep, "")
        self.tabs.setTabEnabled(sep_idx, False)
        self.tabs.tabBar().setTabButton(sep_idx, self.tabs.tabBar().ButtonPosition.RightSide, None)
        self.tabs.tabBar().setTabButton(sep_idx, self.tabs.tabBar().ButtonPosition.LeftSide, None)
        self.tabs.setStyleSheet("""
            QTabBar::tab:disabled {
                width: 1px;
                padding: 0px;
                margin-left: 6px;
                margin-right: 6px;
                background: #555;
                min-width: 1px;
                max-width: 1px;
            }
        """)

        self.tabs.addTab(self.can_data_tab, "CAN Data")
        self.tabs.addTab(self.can_pos_tuning_tab, "CAN AI Analysis")

        # Right dock spans full height (top-right & bottom-right corners belong to right)
        self.setCorner(Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setTabPosition(Qt.DockWidgetArea.RightDockWidgetArea, QTabWidget.TabPosition.North)

        # Motor control dock (right side, full height)
        self.motor_tab = MotorControlTab(self.transport)
        self.motor_dock = QDockWidget("Motor Control", self)
        self.motor_dock.setWidget(self.motor_tab)
        self.motor_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.motor_dock.setTitleBarWidget(QWidget())  # hide title bar (tabs suffice)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.motor_dock)
        self.motor_dock.setMinimumWidth(320)

        # CAN control dock (right side, tabified with Motor Control)
        self.can_control_tab = CanControlTab(self.can_transport)
        self.can_control_dock = QDockWidget("CAN Control", self)
        self.can_control_dock.setWidget(self.can_control_tab)
        self.can_control_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.can_control_dock.setTitleBarWidget(QWidget())  # hide title bar
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.can_control_dock)
        self.can_control_dock.setMinimumWidth(320)
        self.tabifyDockWidget(self.motor_dock, self.can_control_dock)
        self.motor_dock.raise_()  # Motor Control visible by default

        # Auto-switch dock tab when center tab changes
        self._can_tabs = {self.can_data_tab, self.can_pos_tuning_tab}
        self.tabs.currentChanged.connect(self._on_center_tab_changed)

        # Terminal dock (bottom-left, below tabs)
        self.terminal_tab = TerminalTab(self.transport)
        self.terminal_dock = QDockWidget("Terminal", self)
        self.terminal_dock.setWidget(self.terminal_tab)
        self.terminal_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.terminal_dock)
        self.resizeDocks([self.terminal_dock], [200], Qt.Orientation.Vertical)
        self.resizeDocks([self.motor_dock], [340], Qt.Orientation.Horizontal)

        # Wire up packet dispatcher
        self.transport.packet_received.connect(self._dispatch_packet)

        # Wire realtime values to analysis tab
        self.realtime_tab.values_received.connect(self.analysis_tab.on_values)

        # Wire parameter tab log messages to terminal
        self.parameter_tab.log_message.connect(self.terminal_tab.log)

        # Connection state updates
        self.conn_bar.connection_changed.connect(self._on_connection_changed)

        # CAN log → terminal
        self.can_transport.log_message.connect(self.terminal_tab.log)

        # CAN Control torque/pos cmd → CAN Data graph
        self.can_control_tab.torque_cmd_sent.connect(self.can_data_tab._on_torque_cmd)
        self.can_control_tab.pos_cmd_sent.connect(self.can_data_tab._on_pos_cmd)

        # CAN frame → position tuning tab (for PID read responses)
        self.can_transport.frame_received.connect(self.can_pos_tuning_tab.on_frame_received)

        # Debug panel signals
        self.transport.debug_tx.connect(self._on_debug_tx)
        self.transport.debug_rx.connect(self._on_debug_rx)

        # Keyboard shortcut: Ctrl+Shift+I to toggle debug panel
        self._debug_shortcut = QShortcut(QKeySequence("Ctrl+Shift+I"), self)
        self._debug_shortcut.activated.connect(self._toggle_debug_panel)

        # Debug buffer for recent data (keep last 100 entries even when panel is hidden)
        self._debug_buffer = []
        self._debug_buffer_max = 100

    @pyqtSlot(bytes)
    def _dispatch_packet(self, payload: bytes):
        """Route incoming VESC packets to the appropriate tab."""
        if not payload:
            return

        cmd_id = payload[0]
        data = payload[1:]

        if cmd_id == CommPacketId.COMM_GET_VALUES:
            values = VescValues.from_payload(data)
            self.realtime_tab.on_values(values)
            self.motor_tab.on_values(values)

        elif cmd_id == CommPacketId.COMM_ROTOR_POSITION:
            self.position_tab.on_rotor_position(data)

        elif cmd_id == CommPacketId.COMM_SAMPLE_PRINT:
            self.waveform_tab.on_sample_data(data)

        elif cmd_id == CommPacketId.COMM_SET_MCCONF:
            # ACK from VESC after writing MCCONF
            self.terminal_tab.log("[MCCONF] Write OK")

        elif cmd_id == CommPacketId.COMM_GET_MCCONF:
            pid_values = McconfPid.from_mcconf_payload(data)
            self.analysis_tab.on_mcconf_received(pid_values, raw_data=data)
            self.can_pos_tuning_tab.on_mcconf_received(pid_values, raw_data=data)
            self.parameter_tab.on_mcconf_received(data, is_default=False)
            self.terminal_tab.log(f"[MCCONF] Received {len(data)} bytes")
            # Sync foc_encoder_ratio → CAN Data pole pairs
            mcvals = self.parameter_tab._mcconf_values
            if mcvals and "foc_encoder_ratio" in mcvals:
                self.can_data_tab.set_pole_pairs(int(mcvals["foc_encoder_ratio"]))

        elif cmd_id == CommPacketId.COMM_GET_MCCONF_DEFAULT:
            pid_values = McconfPid.from_mcconf_payload(data)
            self.analysis_tab.on_mcconf_received(pid_values, raw_data=data)
            self.can_pos_tuning_tab.on_mcconf_received(pid_values)
            self.parameter_tab.on_mcconf_received(data, is_default=True)
            self.terminal_tab.log(f"[MCCONF DEFAULT] Received {len(data)} bytes")

        elif cmd_id == CommPacketId.COMM_SET_APPCONF:
            # ACK from VESC after writing APPCONF
            self.terminal_tab.log("[APPCONF] Write OK")

        elif cmd_id == CommPacketId.COMM_GET_APPCONF:
            self.parameter_tab.on_appconf_received(data, is_default=False)
            self.terminal_tab.log(f"[APPCONF] Received {len(data)} bytes")

        elif cmd_id == CommPacketId.COMM_GET_APPCONF_DEFAULT:
            self.parameter_tab.on_appconf_received(data, is_default=True)
            self.terminal_tab.log(f"[APPCONF DEFAULT] Received {len(data)} bytes")

        elif cmd_id == CommPacketId.COMM_FW_VERSION:
            if len(data) >= 2:
                major, minor = data[0], data[1]
                # Try to extract hardware name if available
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
                msg += " - ✓ Communication OK!"
                self.terminal_tab.log(msg)

        elif cmd_id == CommPacketId.COMM_TERMINAL_CMD or cmd_id == CommPacketId.COMM_TERMINAL_CMD_SYNC:
            text = data.decode("utf-8", errors="ignore").rstrip("\x00")
            if text:
                self.terminal_tab.log(text)
            else:
                # Log empty response for debugging
                self.terminal_tab.log(f"[DEBUG] Terminal packet received (cmd={cmd_id}, len={len(data)}, empty text)")

        elif cmd_id == CommPacketId.COMM_PRINT:
            # Terminal printf output
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
            # Firmware upload responses — handled by FirmwareUploader via DirectConnection
            pass

        else:
            # Handle custom firmware responses or unknown packet types
            # Try to decode as text if it looks like printable content
            if cmd_id != CommPacketId.COMM_GET_VALUES:
                try:
                    text = data.decode("utf-8", errors="ignore").rstrip("\x00")
                    # Check if mostly printable ASCII
                    printable_ratio = sum(1 for c in text if c.isprintable() or c in '\n\r\t') / max(len(text), 1)
                    if text and printable_ratio > 0.7:
                        # Likely terminal output from custom firmware
                        self.terminal_tab.log(text)
                    else:
                        self.terminal_tab.log(f"[DEBUG] Unknown packet: cmd_id={cmd_id}, len={len(data)}")
                except Exception:
                    self.terminal_tab.log(f"[DEBUG] Unknown packet: cmd_id={cmd_id}, len={len(data)}")

    def _on_center_tab_changed(self, index: int):
        """Auto-switch right dock panel based on center tab."""
        widget = self.tabs.widget(index)
        if widget in self._can_tabs:
            self.can_control_dock.raise_()
        else:
            self.motor_dock.raise_()

    def _on_connection_changed(self, connected: bool):
        if connected:
            self.terminal_tab.log(
                f"[INFO] Connected: {self.transport.port_name} @ {self.transport.baudrate}"
            )
            # Auto-read MCCONF, then APPCONF after a short delay
            QTimer.singleShot(200, self.parameter_tab._on_read_mcconf)
            QTimer.singleShot(600, self.parameter_tab._on_read_appconf)
        else:
            self.terminal_tab.log("[INFO] Disconnected.")
            self.realtime_tab.stop_polling()
            self.experiment_tab.cleanup()
            self.position_tab.cleanup()
            self.motor_tab.cleanup()

    def _create_debug_panel(self) -> QWidget:
        """Create the UART debug panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header row
        header = QHBoxLayout()
        title = QLabel("UART Debug (Ctrl+Shift+I to hide)")
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

    @pyqtSlot(bytes)
    def _on_debug_tx(self, data: bytes):
        """Display TX data in debug panel."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        hex_str = data.hex(' ').upper()
        entry = f"<span style='color:#66ff66;'>[{ts}] TX ({len(data)}): {hex_str}</span>"

        if self._debug_panel.isVisible():
            self._debug_text.append(entry)
            self._debug_text.verticalScrollBar().setValue(
                self._debug_text.verticalScrollBar().maximum()
            )
        else:
            # Buffer when panel is hidden
            self._debug_buffer.append(entry)
            if len(self._debug_buffer) > self._debug_buffer_max:
                self._debug_buffer.pop(0)

    @pyqtSlot(bytes)
    def _on_debug_rx(self, data: bytes):
        """Display RX data in debug panel."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        hex_str = data.hex(' ').upper()
        entry = f"<span style='color:#6699ff;'>[{ts}] RX ({len(data)}): {hex_str}</span>"

        if self._debug_panel.isVisible():
            self._debug_text.append(entry)
            self._debug_text.verticalScrollBar().setValue(
                self._debug_text.verticalScrollBar().maximum()
            )
        else:
            # Buffer when panel is hidden
            self._debug_buffer.append(entry)
            if len(self._debug_buffer) > self._debug_buffer_max:
                self._debug_buffer.pop(0)

    def closeEvent(self, event):
        self.motor_tab.cleanup()
        self.realtime_tab.cleanup()
        self.experiment_tab.cleanup()
        self.position_tab.cleanup()
        self.analysis_tab.cleanup()
        self.can_control_tab.cleanup()
        self.can_data_tab.cleanup()
        self.can_pos_tuning_tab.cleanup()
        self.can_transport.disconnect()
        self.transport.disconnect()
        event.accept()
