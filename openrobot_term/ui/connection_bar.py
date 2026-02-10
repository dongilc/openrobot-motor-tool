"""
CAN-only connection bar widget: PCAN connection, scan, periodic, reboot.
"""

import time

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, QTimer

from ..protocol.can_transport import PcanTransport, PCAN_AVAILABLE
from ..protocol.commands import build_reboot


class ConnectionBar(QWidget):
    can_connection_changed = pyqtSignal(bool)
    can_targets_updated = pyqtSignal(list)  # [int, ...] — discovered motor IDs

    def __init__(self, can_transport: PcanTransport):
        super().__init__()
        self._can_transport = can_transport

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(2)

        # ── PCAN connection row ──
        can_row = QHBoxLayout()
        root_layout.addLayout(can_row)

        can_label = QLabel("PCAN:")
        can_label.setStyleSheet("font-weight: bold; color: #CE7D2C;")
        can_row.addWidget(can_label)

        can_row.addWidget(QLabel("CAN Target:"))
        self.can_target_combo = QComboBox()
        self.can_target_combo.addItem("(No devices)", 0)
        self.can_target_combo.setMinimumWidth(120)
        self.can_target_combo.currentIndexChanged.connect(self._on_can_target_changed)
        can_row.addWidget(self.can_target_combo)

        self.can_scan_btn = QPushButton("Scan")
        self.can_scan_btn.setToolTip("Scan CAN bus for active motor controllers")
        self.can_scan_btn.clicked.connect(self._on_can_scan)
        can_row.addWidget(self.can_scan_btn)

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

        # REBOOT button (sends COMM_REBOOT via CAN EID)
        self.reboot_btn = QPushButton("REBOOT")
        self.reboot_btn.setMinimumHeight(28)
        self.reboot_btn.setStyleSheet(
            "QPushButton { background-color: #CC8800; color: white; "
            "font-weight: bold; font-size: 12px; padding: 2px 14px; }"
            "QPushButton:hover { background-color: #E09900; }"
        )
        self.reboot_btn.setToolTip("Reboot the VESC (COMM_REBOOT via CAN)")
        self.reboot_btn.clicked.connect(self._on_reboot)
        can_row.addWidget(self.reboot_btn)

        # Disable PCAN row if DLL not available
        if not PCAN_AVAILABLE or self._can_transport is None:
            for widget in [self.can_target_combo, self.can_scan_btn,
                           self.can_open_btn, self.can_close_btn,
                           self.can_periodic_btn]:
                widget.setEnabled(False)
            self.can_status_label.setText("PCAN DLL not found")
            self.can_status_label.setStyleSheet("color: #999; font-size: 9pt;")

        # CAN traffic update timer (1 Hz)
        self._can_traffic_timer = QTimer(self)
        self._can_traffic_timer.setInterval(1000)
        self._can_traffic_timer.timeout.connect(self._update_can_traffic)

    # ── PCAN connection controls ──

    def _on_can_target_changed(self, index):
        motor_id = self.can_target_combo.currentData()
        if motor_id and motor_id > 0 and self._can_transport:
            self._can_transport.set_motor_id(motor_id)

    def _on_can_scan(self):
        if not self._can_transport or not self._can_transport.is_connected():
            return
        self._can_transport.start_scan()
        self.can_scan_btn.setEnabled(False)
        self.can_scan_btn.setText("Scanning...")
        QTimer.singleShot(1500, self._on_scan_done)

    def _on_scan_done(self):
        if not self._can_transport:
            return
        found = self._can_transport.finish_scan()
        self.can_target_combo.blockSignals(True)
        self.can_target_combo.clear()
        if found:
            for mid in found:
                self.can_target_combo.addItem(f"ID {mid}", mid)
            self._can_transport.set_motor_id(found[0])
            self.can_status_label.setText(
                f"Scan: {len(found)} device(s) found")
            self.can_status_label.setStyleSheet("color: #66ff66; font-size: 9pt;")
        else:
            self.can_target_combo.addItem("(No devices)", 0)
            self.can_status_label.setText("Scan: no devices found")
            self.can_status_label.setStyleSheet("color: #ffaa00; font-size: 9pt;")
        self.can_target_combo.blockSignals(False)
        self.can_scan_btn.setEnabled(True)
        self.can_scan_btn.setText("Scan")
        self.reboot_btn.setEnabled(True)
        self.can_targets_updated.emit(found)

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
            # Auto-scan after successful open
            QTimer.singleShot(100, self._on_can_scan)
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

    def rescan_after_id_change(self):
        """Re-scan CAN bus after controller_id change (applied immediately by firmware)."""
        if not self._can_transport or not self._can_transport.is_connected():
            return
        self.can_status_label.setText("ID changed, re-scanning ...")
        self.can_status_label.setStyleSheet("color: #ffaa00; font-size: 9pt;")
        # Disable reboot during re-scan (old ID is stale, command would be ignored)
        self.reboot_btn.setEnabled(False)
        QTimer.singleShot(500, self._on_can_scan)

    def _on_reboot(self):
        if not self._can_transport or not self._can_transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Open PCAN connection first.")
            return
        reply = QMessageBox.question(
            self, "Reboot", "Reboot the VESC via CAN?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._can_transport.send_vesc_to_target(build_reboot())
            self.can_status_label.setText("Rebooting, re-scanning in 3s ...")
            self.can_status_label.setStyleSheet("color: #ffaa00; font-size: 9pt;")
            self.reboot_btn.setEnabled(False)
            # Wait 3s for MCU to finish IWDG reset + ChibiOS boot, then re-scan
            QTimer.singleShot(3000, self._on_can_scan)
