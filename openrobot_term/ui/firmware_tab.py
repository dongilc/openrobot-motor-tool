"""
Firmware/Bootloader upload tab using standard VESC protocol.
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

from ..protocol.serial_transport import SerialTransport
from ..workers.firmware_uploader import FirmwareUploader, UploadMode

DEFAULT_FW_PATH = "build/ch.bin"
DEFAULT_BL_PATH = "build/bootloader.bin"


class FirmwareTab(QWidget):
    def __init__(self, transport: SerialTransport):
        super().__init__()
        self._transport = transport
        self.uploader = None
        self.upload_error_flag = False

        layout = QVBoxLayout(self)

        # Upload mode selection
        mode_group = QGroupBox("Upload Mode")
        mode_layout = QHBoxLayout(mode_group)
        layout.addWidget(mode_group)

        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Firmware", UploadMode.FIRMWARE)
        self.mode_combo.addItem("Bootloader", UploadMode.BOOTLOADER)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()

        # Warning label
        self.warning_label = QLabel()
        self.warning_label.setWordWrap(False)
        self._update_warning_label()
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
        self.bin_edit.setPlaceholderText("Select firmware or bootloader binary file...")
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
            "<b>Firmware:</b> COMM_ERASE_NEW_APP(2) + COMM_WRITE_NEW_APP_DATA(3) + COMM_JUMP_TO_BOOTLOADER(1)<br>"
            "<b>Bootloader:</b> COMM_ERASE_BOOTLOADER(73) + COMM_WRITE_NEW_APP_DATA(3)<br>"
            "<b>Chunk size:</b> 384 bytes, <b>Erase timeout:</b> 20s, <b>Write timeout:</b> 3s<br>"
            "<br>"
            "<span style='color: red;'><b>Warning:</b> Bootloader upload is risky! "
            "A failed bootloader upload may brick the device.</span>"
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        # Listen for MCU abort in text
        self._transport.text_received.connect(self._check_abort)

    def _update_warning_label(self):
        mode = self.mode_combo.currentData()
        if mode == UploadMode.BOOTLOADER:
            self.warning_label.setText(
                "<span style='color: red; font-weight: bold;'>"
                "BOOTLOADER MODE - Use with caution!</span>"
            )
        else:
            self.warning_label.setText(
                "<span style='color: green;'>Firmware upload mode</span>"
            )

    def _on_mode_changed(self, index):
        self._update_warning_label()
        mode = self.mode_combo.currentData()
        if mode == UploadMode.BOOTLOADER:
            if self.bin_edit.text() == DEFAULT_FW_PATH:
                self.bin_edit.setText(DEFAULT_BL_PATH)
        else:
            if self.bin_edit.text() == DEFAULT_BL_PATH:
                self.bin_edit.setText(DEFAULT_FW_PATH)
        self._update_file_info()

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

    def _check_abort(self, text: str):
        if "[ABORT]" in text:
            self.upload_error_flag = True
            if self.uploader and self.uploader.isRunning():
                self.uploader.notify_mcu_abort()

    def on_browse(self):
        mode = self.mode_combo.currentData()
        if mode == UploadMode.BOOTLOADER:
            title = "Select Bootloader BIN file"
        else:
            title = "Select Firmware BIN file"

        path, _ = QFileDialog.getOpenFileName(
            self, title, "", "Binary (*.bin);;All (*.*)"
        )
        if path:
            self.bin_edit.setText(path)

    def on_flash_upload(self):
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not connected", "Connect to device first.")
            return

        bin_path = self.bin_edit.text().strip()
        if not bin_path:
            QMessageBox.warning(self, "BIN missing", "Select BIN file path.")
            return
        if not Path(bin_path).exists():
            QMessageBox.warning(self, "BIN not found", f"File not found:\n{bin_path}")
            return

        mode = self.mode_combo.currentData()
        mode_str = "bootloader" if mode == UploadMode.BOOTLOADER else "firmware"

        # Confirmation dialog (especially important for bootloader)
        if mode == UploadMode.BOOTLOADER:
            reply = QMessageBox.warning(
                self,
                "Bootloader Upload Warning",
                "You are about to upload a BOOTLOADER.\n\n"
                "This is a RISKY operation!\n"
                "A failed bootloader upload may brick the device permanently.\n\n"
                "Are you absolutely sure you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        else:
            reply = QMessageBox.question(
                self,
                "Confirm Upload",
                f"Upload firmware to device?\n\nFile: {Path(bin_path).name}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Clear log and reset
        self.log_view.clear()
        self.upload_error_flag = False
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Starting {mode_str} upload...")

        self.log(f"[INFO] Starting {mode_str} upload...")
        self.log(f"[INFO] File: {bin_path}")

        self.cancel_btn.setEnabled(True)
        self.flash_btn.setEnabled(False)
        self.mode_combo.setEnabled(False)

        # Create and start uploader
        self.uploader = FirmwareUploader(self._transport, bin_path, mode)
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
        self.log("[INFO] Upload finished successfully!")
        self.status_label.setText("Upload complete!")
        self.status_label.setStyleSheet("color: green;")
        self.cancel_btn.setEnabled(False)
        self.flash_btn.setEnabled(True)
        self.mode_combo.setEnabled(True)

        QMessageBox.information(
            self,
            "Upload Complete",
            "Firmware upload completed successfully.\n\n"
            "The device may need to reboot to use the new firmware."
        )

    def _on_aborted(self, reason: str):
        self.log(reason)
        self.status_label.setText("Upload failed")
        self.status_label.setStyleSheet("color: red;")
        self.cancel_btn.setEnabled(False)
        self.flash_btn.setEnabled(True)
        self.mode_combo.setEnabled(True)
