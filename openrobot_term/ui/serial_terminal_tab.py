"""
Serial Terminal tab: COM port connection, debug_printf log view, command input.
"""

import time

import serial
import serial.tools.list_ports

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QComboBox, QLabel, QCheckBox,
    QSizePolicy, QMessageBox,
)
from PyQt6.QtGui import QTextCursor


DEFAULT_BAUD = 115200


def list_serial_ports():
    return [p.device for p in serial.tools.list_ports.comports()]


class SerialReader(QThread):
    received = pyqtSignal(str)
    disconnected = pyqtSignal(str)

    def __init__(self, ser: serial.Serial):
        super().__init__()
        self.ser = ser
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        buffer = ""
        while self._running:
            try:
                if self.ser is None or not self.ser.is_open:
                    time.sleep(0.05)
                    continue
                n = self.ser.in_waiting
                if n:
                    incoming = self.ser.read(n).decode("utf-8", errors="ignore")
                    buffer += incoming
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        self.received.emit(line + "\n")
                time.sleep(0.005)
            except (serial.SerialException, OSError, PermissionError) as e:
                self.disconnected.emit(f"{type(e).__name__}: {e}")
                return
            except Exception as e:
                self.disconnected.emit(f"Unexpected error: {e}")
                return


class SerialTerminalTab(QWidget):
    def __init__(self):
        super().__init__()

        self.ser = None
        self.reader = None

        layout = QVBoxLayout(self)

        # Row 1: COM port + Baud + Connect
        row1 = QHBoxLayout()
        layout.addLayout(row1)

        row1.addWidget(QLabel("COM:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(100)
        row1.addWidget(self.port_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        row1.addWidget(self.refresh_btn)

        row1.addSpacing(8)
        row1.addWidget(QLabel("Baud:"))
        self.baud_combo = QComboBox()
        for b in [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]:
            self.baud_combo.addItem(str(b))
        self.baud_combo.setCurrentText(str(DEFAULT_BAUD))
        row1.addWidget(self.baud_combo)

        row1.addSpacing(8)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect)
        row1.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.on_disconnect)
        self.disconnect_btn.setEnabled(False)
        row1.addWidget(self.disconnect_btn)

        row1.addSpacing(8)
        self.auto_reconnect = QCheckBox("Auto-reconnect")
        self.auto_reconnect.setChecked(True)
        row1.addWidget(self.auto_reconnect)

        row1.addStretch()

        # Log view
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_view)

        # Row 2: Command input + Send + Clear
        row2 = QHBoxLayout()
        layout.addLayout(row2)

        self.cmd_edit = QLineEdit()
        self.cmd_edit.setPlaceholderText("Type command and press Enter (or Send).")
        self.cmd_edit.returnPressed.connect(self.on_send)
        row2.addWidget(self.cmd_edit)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send)
        row2.addWidget(self.send_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.on_clear)
        row2.addWidget(self.clear_btn)

        # Init ports
        self.refresh_ports()

    # --- Utilities ---

    def log(self, s: str):
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(s)
        if not s.endswith("\n"):
            self.log_view.insertPlainText("\n")
        self.log_view.ensureCursorVisible()

    def refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = list_serial_ports()
        if not ports:
            ports = ["(none)"]
        self.port_combo.addItems(ports)
        if current in ports:
            self.port_combo.setCurrentText(current)

    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    # --- Serial Connect/Disconnect ---

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
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.log(f"[INFO] Connected: {port} @ {baud}")
        except Exception as e:
            QMessageBox.critical(self, "Connect failed", str(e))
            return

        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self._start_reader()

    def on_disconnect(self):
        self._stop_reader()
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.log("[INFO] Disconnected.")

    def _start_reader(self):
        self._stop_reader()
        if not self.is_connected():
            return
        self.reader = SerialReader(self.ser)
        self.reader.received.connect(self._on_serial_received)
        self.reader.disconnected.connect(self._on_serial_disconnected)
        self.reader.start()

    def _stop_reader(self):
        if self.reader:
            try:
                self.reader.stop()
                self.reader.wait(500)
            except Exception:
                pass
        self.reader = None

    def _on_serial_received(self, text: str):
        self.log(text.rstrip("\0"))

    def _on_serial_disconnected(self, reason: str):
        self.log(f"[INFO] Disconnected: {reason}")
        if self.auto_reconnect.isChecked():
            self.log("[INFO] Auto-reconnect enabled. Trying to reconnect...")
            self._try_reconnect()
        else:
            self.on_disconnect()

    def _try_reconnect(self):
        port = self.port_combo.currentText().strip()
        try:
            baud = int(self.baud_combo.currentText())
        except ValueError:
            baud = DEFAULT_BAUD

        self._stop_reader()
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

        for i in range(1, 6):
            try:
                time.sleep(0.5)
                self.ser = serial.Serial(port, baud, timeout=0.1)
                self.log(f"[INFO] Reconnected: {port} @ {baud}")
                self._start_reader()
                return
            except Exception as e:
                self.log(f"[WARN] Reconnect {i}/5 failed: {e}")

        self.log("[ERROR] Reconnect failed. Please connect manually.")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)

    # --- Command Send ---

    def on_send(self):
        if not self.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to a COM port first.")
            return
        cmd = self.cmd_edit.text()
        self.cmd_edit.clear()
        try:
            if cmd == "":
                self.ser.write(b"\r")
                self.ser.flush()
            else:
                self.ser.write((cmd + "\r\n").encode("utf-8"))
                self.ser.flush()
                self.log(f"> {cmd}")
        except Exception as e:
            self.log(f"[ERROR] Send failed: {e}")

    def on_clear(self):
        self.log_view.clear()
        self.cmd_edit.setFocus()

    # --- Cleanup ---

    def cleanup(self):
        self._stop_reader()
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
