"""
Terminal tab: log view + command input (extracted from original MainWindow).
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QSizePolicy, QMessageBox,
)
from PyQt6.QtGui import QTextCursor

from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import build_terminal_cmd, build_get_fw_version


class TerminalTab(QWidget):
    def __init__(self, transport: SerialTransport):
        super().__init__()
        self._transport = transport

        layout = QVBoxLayout(self)

        # Log view
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_view)

        # Bottom row: input + send + options
        row = QHBoxLayout()
        layout.addLayout(row)

        self.cmd_edit = QLineEdit()
        self.cmd_edit.setPlaceholderText("Type command and press Enter (or Send).")
        self.cmd_edit.returnPressed.connect(self.on_send)
        row.addWidget(self.cmd_edit)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send)
        row.addWidget(self.send_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.on_clear)
        row.addWidget(self.clear_btn)

        self.test_btn = QPushButton("Test Connection")
        self.test_btn.setToolTip("Send COMM_FW_VERSION to test basic communication")
        self.test_btn.clicked.connect(self.on_test_connection)
        row.addWidget(self.test_btn)

        # Connect transport text signal
        self._transport.text_received.connect(self.on_text_received)

    def log(self, s: str):
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(s)
        if not s.endswith("\n"):
            self.log_view.insertPlainText("\n")
        self.log_view.ensureCursorVisible()

    def on_text_received(self, text: str):
        self.log(text.rstrip("\0"))

    def on_send(self):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to a COM port first.")
            return

        cmd = self.cmd_edit.text()
        self.cmd_edit.clear()

        try:
            self._transport.send_packet(build_terminal_cmd(cmd))
            if cmd:
                self.log(f"> {cmd}")
        except Exception as e:
            self.log(f"[ERROR] Send failed: {e}")

    def on_clear(self):
        self.log_view.clear()
        self.cmd_edit.setFocus()

    def on_test_connection(self):
        """Send COMM_FW_VERSION to test basic communication."""
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to a COM port first.")
            return

        try:
            self._transport.send_packet(build_get_fw_version())
            self.log("[TEST] Sent COMM_FW_VERSION request... waiting for response")
        except Exception as e:
            self.log(f"[ERROR] Test failed: {e}")
