"""
Firmware upload tab — CAN-only via PCAN-USB EID multi-frame protocol.
"""

import threading
import time
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QProgressBar, QFileDialog, QMessageBox, QTextEdit,
    QComboBox, QGroupBox, QFrame,
)
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import Qt

from ..protocol.can_transport import PcanTransport
from ..workers.firmware_uploader import CanFirmwareUploader

DEFAULT_FW_PATH = "build/ch.bin"
DEFAULT_BL_PATH = "Openrobot_Bootloader.bin"


class FirmwareTab(QWidget):
    def __init__(self, transport: PcanTransport):
        super().__init__()
        self._transport = transport
        self.uploader = None
        self.upload_error_flag = False

        layout = QVBoxLayout(self)

        # Upload mode selection
        mode_group = QGroupBox("Upload Mode")
        mode_layout = QHBoxLayout(mode_group)
        layout.addWidget(mode_group)

        # Upload type selection
        mode_layout.addWidget(QLabel("Type:"))
        self.upload_type_combo = QComboBox()
        self.upload_type_combo.addItem("Firmware", "firmware")
        self.upload_type_combo.addItem("Bootloader", "bootloader")
        self.upload_type_combo.currentIndexChanged.connect(self._on_upload_type_changed)
        mode_layout.addWidget(self.upload_type_combo)

        mode_layout.addSpacing(20)

        # CAN target ID
        mode_layout.addWidget(QLabel("Target:"))
        self.can_target_combo = QComboBox()
        self.can_target_combo.addItem("Broadcast (ALL)", 255)
        mode_layout.addWidget(self.can_target_combo)

        mode_layout.addStretch()

        self.warning_label = QLabel(
            "<span style='color: green;'>CAN Firmware upload mode</span>"
        )
        self.warning_label.setWordWrap(False)
        mode_layout.addWidget(self.warning_label)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        layout.addWidget(file_group)

        # BIN path row
        row1 = QHBoxLayout()
        file_layout.addLayout(row1)

        row1.addWidget(QLabel("BIN File:"))
        self.bin_edit = QLineEdit(DEFAULT_FW_PATH)
        self.bin_edit.setPlaceholderText("Select firmware binary file...")
        row1.addWidget(self.bin_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.on_browse)
        row1.addWidget(self.browse_btn)

        # File info
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: gray;")
        file_layout.addWidget(self.file_info_label)
        self.bin_edit.textChanged.connect(self._update_file_info)
        self._update_file_info()

        # Log view for upload messages
        log_group = QGroupBox("Upload Log")
        log_layout = QVBoxLayout(log_group)
        layout.addWidget(log_group)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_view.setMinimumHeight(150)
        log_layout.addWidget(self.log_view)

        # Upload controls
        control_group = QGroupBox("Upload Control")
        control_layout = QVBoxLayout(control_group)
        layout.addWidget(control_group)

        row2 = QHBoxLayout()
        control_layout.addLayout(row2)

        self.flash_btn = QPushButton("Start Upload")
        self.flash_btn.clicked.connect(self.on_flash_upload)
        self.flash_btn.setMinimumWidth(120)
        row2.addWidget(self.flash_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.cancel_btn.setEnabled(False)
        row2.addWidget(self.cancel_btn)

        row2.addSpacing(20)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumWidth(200)
        row2.addWidget(self.progress_bar, 1)

        # Status label
        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)

        layout.addStretch()

        # Protocol info
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QVBoxLayout(info_frame)
        layout.addWidget(info_frame)

        info_text = QLabel(
            "<b>CAN Firmware Upload:</b> COMM_ERASE_NEW_APP(2) + COMM_WRITE_NEW_APP_DATA(3) + COMM_JUMP_TO_BOOTLOADER(1)<br>"
            "<b>Transport:</b> PCAN-USB EID multi-frame (send_mode=2, fire-and-forget)<br>"
            "<b>Chunk size:</b> 384 bytes, <b>Erase timeout:</b> 30s, <b>Chunk delay:</b> 50ms<br>"
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

    def _update_file_info(self):
        path = self.bin_edit.text().strip()
        if not path:
            self.file_info_label.setText("No file specified")
            self.file_info_label.setStyleSheet("color: gray;")
            return

        p = Path(path)
        if p.exists():
            size = p.stat().st_size
            self.file_info_label.setText(f"File: {p.name} ({size:,} bytes)")
            self.file_info_label.setStyleSheet("color: green;")
        else:
            self.file_info_label.setText(f"File not found: {p.name}")
            self.file_info_label.setStyleSheet("color: red;")

    def log(self, s: str):
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(s)
        if not s.endswith("\n"):
            self.log_view.insertPlainText("\n")
        self.log_view.ensureCursorVisible()

    def _on_upload_type_changed(self, index):
        is_bootloader = self.upload_type_combo.currentData() == "bootloader"
        if is_bootloader:
            self.warning_label.setText(
                "<span style='color: orange;'>CAN Bootloader upload mode</span>")
            self._set_default_bin_path(DEFAULT_BL_PATH)
        else:
            self.warning_label.setText(
                "<span style='color: green;'>CAN Firmware upload mode</span>")
            self._set_default_bin_path(DEFAULT_FW_PATH)

    def _set_default_bin_path(self, relative_path: str):
        """Set bin_edit to the default path relative to the tool's root directory."""
        tool_dir = Path(__file__).resolve().parent.parent.parent
        default = tool_dir / relative_path
        self.bin_edit.setText(str(default))

    def on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Firmware BIN file", "", "Binary (*.bin);;All (*.*)"
        )
        if path:
            self.bin_edit.setText(path)

    def on_flash_upload(self):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Open PCAN connection first.")
            return

        bin_path = self.bin_edit.text().strip()
        if not bin_path:
            QMessageBox.warning(self, "BIN missing", "Select BIN file path.")
            return
        if not Path(bin_path).exists():
            QMessageBox.warning(self, "BIN not found", f"File not found:\n{bin_path}")
            return

        is_bootloader = self.upload_type_combo.currentData() == "bootloader"
        type_str = "bootloader" if is_bootloader else "firmware"
        target_id = self.can_target_combo.currentData()
        target_str = "ALL controllers (broadcast)" if target_id == 255 else f"controller ID {target_id}"

        if is_bootloader:
            reply = QMessageBox.warning(
                self,
                "Bootloader Upload Warning",
                f"You are about to upload a BOOTLOADER to {target_str}.\n\n"
                f"File: {Path(bin_path).name}\n\n"
                "WARNING: A corrupted bootloader may brick the device!\n"
                "Recovery requires SWD/JTAG programmer.\n\n"
                "Are you absolutely sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
        elif target_id == 255:
            reply = QMessageBox.warning(
                self,
                "CAN Broadcast Upload Warning",
                f"You are about to broadcast firmware to ALL controllers on the CAN bus.\n\n"
                f"File: {Path(bin_path).name}\n\n"
                "ALL controllers will be updated simultaneously.\n"
                "Are you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
        else:
            reply = QMessageBox.question(
                self,
                "Confirm CAN Upload",
                f"Upload {type_str} via CAN to {target_str}?\n\nFile: {Path(bin_path).name}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Clear log and reset
        self.log_view.clear()
        self.upload_error_flag = False
        self.progress_bar.setValue(0)

        self.status_label.setText(f"Starting {type_str} upload (CAN)...")

        self.log(f"[INFO] Starting {type_str} upload (CAN)...")
        self.log(f"[INFO] File: {bin_path}")
        self.log(f"[INFO] Target: {target_str}")

        self.cancel_btn.setEnabled(True)
        self.flash_btn.setEnabled(False)

        # Create and start CAN uploader
        self.uploader = CanFirmwareUploader(
            self._transport, bin_path, target_id=target_id,
            bootloader_mode=is_bootloader)
        self.uploader.log.connect(self.log)
        self.uploader.progress.connect(self._on_progress)
        self.uploader.finished_ok.connect(self._on_finished)
        self.uploader.aborted.connect(self._on_aborted)
        self.uploader.start()

    def _on_progress(self, percent: int):
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"Uploading... {percent}%")

    def on_cancel(self):
        if self.uploader and self.uploader.isRunning():
            self.uploader.cancel()
            self.log("[INFO] Cancel requested...")
            self.status_label.setText("Cancelling...")

    def _on_finished(self):
        is_bootloader = self.upload_type_combo.currentData() == "bootloader"
        type_str = "Bootloader" if is_bootloader else "Firmware"

        self.log(f"[INFO] {type_str} upload finished successfully!")
        self.status_label.setText(f"{type_str} upload complete!")
        self.status_label.setStyleSheet("color: green;")
        self.cancel_btn.setEnabled(False)
        self.flash_btn.setEnabled(True)

        QMessageBox.information(
            self,
            "Upload Complete",
            f"{type_str} upload completed successfully.\n\n"
            "The device may need to reboot to apply the update."
        )

    def update_can_targets(self, ids: list[int]):
        """Update CAN target dropdown with discovered IDs from scan."""
        current_data = self.can_target_combo.currentData()
        self.can_target_combo.blockSignals(True)
        self.can_target_combo.clear()
        self.can_target_combo.addItem("Broadcast (ALL)", 255)
        for mid in ids:
            self.can_target_combo.addItem(f"ID {mid}", mid)
        # Restore previous selection if still available
        for i in range(self.can_target_combo.count()):
            if self.can_target_combo.itemData(i) == current_data:
                self.can_target_combo.setCurrentIndex(i)
                break
        self.can_target_combo.blockSignals(False)

    def _on_aborted(self, reason: str):
        self.log(reason)
        self.status_label.setText("Upload failed")
        self.status_label.setStyleSheet("color: red;")
        self.cancel_btn.setEnabled(False)
        self.flash_btn.setEnabled(True)
