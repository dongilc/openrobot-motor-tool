"""
Shared connection bar widget: COM port, baud rate, connect/disconnect, auto-reconnect.
Includes optional PCAN connection row.
"""

import time

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QMessageBox, QSpinBox,
)
from PyQt6.QtCore import pyqtSignal, QTimer

from ..protocol.serial_transport import SerialTransport, list_serial_ports
from ..protocol.commands import build_reboot
from ..protocol.can_transport import PcanTransport, PCAN_AVAILABLE

DEFAULT_BAUD = 115200


class ConnectionBar(QWidget):
    connection_changed = pyqtSignal(bool)  # True=connected, False=disconnected
    can_connection_changed = pyqtSignal(bool)

    def __init__(self, transport: SerialTransport, can_transport: PcanTransport = None):
        super().__init__()
        self._transport = transport
        self._can_transport = can_transport
        self._transport.connection_lost.connect(self._on_connection_lost)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(2)

        # ── Row 1: Serial connection ──
        serial_row = QHBoxLayout()
        root_layout.addLayout(serial_row)

        serial_row.addWidget(QLabel("COM:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(120)
        serial_row.addWidget(self.port_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        serial_row.addWidget(self.refresh_btn)

        serial_row.addSpacing(12)

        serial_row.addWidget(QLabel("Baud:"))
        self.baud_combo = QComboBox()
        for b in [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]:
            self.baud_combo.addItem(str(b))
        self.baud_combo.setCurrentText(str(DEFAULT_BAUD))
        serial_row.addWidget(self.baud_combo)

        serial_row.addSpacing(12)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect)
        serial_row.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.on_disconnect)
        self.disconnect_btn.setEnabled(False)
        serial_row.addWidget(self.disconnect_btn)

        serial_row.addSpacing(12)
        self.auto_reconnect = QCheckBox("Auto-reconnect")
        self.auto_reconnect.setChecked(True)
        serial_row.addWidget(self.auto_reconnect)

        serial_row.addSpacing(16)
        self.bitrate_label = QLabel("")
        self.bitrate_label.setStyleSheet("color: #B4B4B4; font-size: 9pt;")
        serial_row.addWidget(self.bitrate_label)

        serial_row.addStretch()

        self.reboot_btn = QPushButton("REBOOT")
        self.reboot_btn.setMinimumHeight(28)
        self.reboot_btn.setStyleSheet(
            "QPushButton { background-color: #CC8800; color: white; "
            "font-weight: bold; font-size: 12px; padding: 2px 14px; }"
            "QPushButton:hover { background-color: #E09900; }"
        )
        self.reboot_btn.setToolTip("Reboot the VESC (COMM_REBOOT)")
        self.reboot_btn.clicked.connect(self._on_reboot)
        serial_row.addWidget(self.reboot_btn)

        # ── Row 2: PCAN connection ──
        can_row = QHBoxLayout()
        root_layout.addLayout(can_row)

        can_label = QLabel("PCAN:")
        can_label.setStyleSheet("font-weight: bold; color: #CE7D2C;")
        can_row.addWidget(can_label)

        can_row.addWidget(QLabel("Motor ID:"))
        self.can_motor_id_spin = QSpinBox()
        self.can_motor_id_spin.setRange(1, 32)
        self.can_motor_id_spin.setValue(1)
        self.can_motor_id_spin.setMinimumWidth(80)
        self.can_motor_id_spin.valueChanged.connect(self._on_can_motor_id_changed)
        can_row.addWidget(self.can_motor_id_spin)

        can_row.addSpacing(12)

        self.can_open_btn = QPushButton("Open")
        self.can_open_btn.clicked.connect(self._on_can_open)
        can_row.addWidget(self.can_open_btn)

        self.can_close_btn = QPushButton("Close")
        self.can_close_btn.clicked.connect(self._on_can_close)
        self.can_close_btn.setEnabled(False)
        can_row.addWidget(self.can_close_btn)

        can_row.addSpacing(12)

        self.can_periodic_btn = QPushButton("Periodic")
        self.can_periodic_btn.setCheckable(True)
        self.can_periodic_btn.clicked.connect(self._on_can_periodic)
        self.can_periodic_btn.setEnabled(False)
        can_row.addWidget(self.can_periodic_btn)

        can_row.addSpacing(12)
        self.can_status_label = QLabel("")
        self.can_status_label.setStyleSheet("color: #B4B4B4; font-size: 9pt;")
        can_row.addWidget(self.can_status_label)

        can_row.addSpacing(12)
        self.can_traffic_label = QLabel("")
        self.can_traffic_label.setStyleSheet("color: #B4B4B4; font-size: 9pt;")
        can_row.addWidget(self.can_traffic_label)

        can_row.addStretch()

        # Disable PCAN row if DLL not available
        if not PCAN_AVAILABLE or self._can_transport is None:
            for widget in [self.can_motor_id_spin, self.can_open_btn,
                           self.can_close_btn, self.can_periodic_btn]:
                widget.setEnabled(False)
            self.can_status_label.setText("PCAN DLL not found")
            self.can_status_label.setStyleSheet("color: #999; font-size: 9pt;")

        # Bitrate update timer (1 Hz)
        self._bitrate_timer = QTimer(self)
        self._bitrate_timer.setInterval(1000)
        self._bitrate_timer.timeout.connect(self._update_bitrate)

        # CAN traffic update timer (1 Hz)
        self._can_traffic_timer = QTimer(self)
        self._can_traffic_timer.setInterval(1000)
        self._can_traffic_timer.timeout.connect(self._update_can_traffic)

        self.refresh_ports()

    def refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = list_serial_ports()
        if not ports:
            ports = ["(none)"]
        self.port_combo.addItems(ports)
        if current in ports:
            self.port_combo.setCurrentText(current)

    def on_connect(self):
        port = self.port_combo.currentText().strip()
        if not port or port == "(none)":
            QMessageBox.warning(self, "No Port", "No available COM port.")
            return
        try:
            baud = int(self.baud_combo.currentText())
        except ValueError:
            QMessageBox.warning(self, "Baud Error", "Invalid baudrate.")
            return
        try:
            self._transport.connect(port, baud)
        except Exception as e:
            QMessageBox.critical(self, "Connect failed", str(e))
            return

        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self._bitrate_timer.start()
        self.connection_changed.emit(True)

    def _on_reboot(self):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to a COM port first.")
            return
        reply = QMessageBox.question(
            self, "Reboot", "Reboot the VESC?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._transport.send_packet(build_reboot())

    def on_disconnect(self):
        self._transport.disconnect()
        self._bitrate_timer.stop()
        self.bitrate_label.setText("")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.connection_changed.emit(False)

    def _update_bitrate(self):
        if not self._transport.is_connected():
            self.bitrate_label.setText("")
            return
        rx, tx = self._transport.take_byte_counts()
        total_bits = (rx + tx) * 8
        baud = self._transport.baudrate
        if baud > 0:
            load_pct = total_bits / baud * 100
        else:
            load_pct = 0.0

        # Format bitrate human-readable
        if total_bits >= 1_000_000:
            br_str = f"{total_bits / 1_000_000:.1f} Mbps"
        elif total_bits >= 1_000:
            br_str = f"{total_bits / 1_000:.1f} kbps"
        else:
            br_str = f"{total_bits} bps"

        rx_bits = rx * 8
        tx_bits = tx * 8
        if rx_bits >= 1000:
            rx_str = f"{rx_bits / 1000:.1f}k"
        else:
            rx_str = f"{rx_bits}"
        if tx_bits >= 1000:
            tx_str = f"{tx_bits / 1000:.1f}k"
        else:
            tx_str = f"{tx_bits}"

        self.bitrate_label.setText(
            f"{br_str}  (RX:{rx_str} TX:{tx_str})  Load: {load_pct:.1f}%"
        )

    def _on_connection_lost(self, reason: str):
        self._bitrate_timer.stop()
        self.bitrate_label.setText("")
        if self.auto_reconnect.isChecked():
            self._try_reconnect()
        else:
            self.on_disconnect()

    def _try_reconnect(self):
        port = self.port_combo.currentText().strip()
        try:
            baud = int(self.baud_combo.currentText())
        except ValueError:
            baud = DEFAULT_BAUD

        self._transport.disconnect()

        for i in range(1, 6):
            try:
                time.sleep(0.5)
                self._transport.connect(port, baud)
                self.connect_btn.setEnabled(False)
                self.disconnect_btn.setEnabled(True)
                self._bitrate_timer.start()
                self.connection_changed.emit(True)
                return
            except Exception:
                pass

        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.connection_changed.emit(False)

    # ── PCAN connection controls ──

    def _on_can_motor_id_changed(self, val: int):
        if self._can_transport:
            self._can_transport.set_motor_id(val)

    def _on_can_open(self):
        if not self._can_transport:
            return
        ok = self._can_transport.connect()
        if ok:
            self.can_open_btn.setEnabled(False)
            self.can_close_btn.setEnabled(True)
            self.can_periodic_btn.setEnabled(True)
            self.can_status_label.setText("Connected (1 Mbps)")
            self.can_status_label.setStyleSheet("color: #66ff66; font-size: 9pt;")
            self._can_traffic_timer.start()
            self.can_connection_changed.emit(True)
        else:
            self.can_status_label.setText("Connection failed")
            self.can_status_label.setStyleSheet("color: #ff4444; font-size: 9pt;")

    def _on_can_close(self):
        if not self._can_transport:
            return
        self._can_transport.disconnect()
        self._can_traffic_timer.stop()
        self.can_traffic_label.setText("")
        self.can_open_btn.setEnabled(True)
        self.can_close_btn.setEnabled(False)
        self.can_periodic_btn.setEnabled(False)
        self.can_periodic_btn.setChecked(False)
        self.can_status_label.setText("Disconnected")
        self.can_status_label.setStyleSheet("color: #ffaa00; font-size: 9pt;")
        self.can_connection_changed.emit(False)

    def _update_can_traffic(self):
        if not self._can_transport or not self._can_transport.is_connected():
            self.can_traffic_label.setText("")
            return
        rx, tx = self._can_transport.take_frame_counts()
        total = rx + tx
        # CAN 1 Mbps, standard frame ~111 bits (47 overhead + 64 data)
        bus_load = (total * 111) / 1_000_000 * 100
        self.can_traffic_label.setText(
            f"RX:{rx} TX:{tx} fps  Load: {bus_load:.1f}%"
        )

    def _on_can_periodic(self, checked: bool):
        if not self._can_transport:
            return
        if checked:
            self._can_transport.start_periodic()
            self.can_periodic_btn.setStyleSheet(
                "QPushButton { background-color: #338833; color: white; }"
            )
        else:
            self._can_transport.stop_periodic()
            self.can_periodic_btn.setStyleSheet("")
