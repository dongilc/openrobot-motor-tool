"""
VESC firmware upload worker using standard VESC protocol.
Based on VESC Tool implementation (vescinterface.cpp).

Supports:
- Firmware upload (COMM_ERASE_NEW_APP + COMM_WRITE_NEW_APP_DATA + COMM_JUMP_TO_BOOTLOADER)
- Bootloader upload (COMM_ERASE_BOOTLOADER + COMM_WRITE_NEW_APP_DATA)
"""

import struct
import threading
import time
from binascii import crc_hqx
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal, Qt

from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import (
    CommPacketId,
    build_erase_new_app,
    build_write_new_app_data,
    build_erase_bootloader,
    build_jump_to_bootloader,
)

# VESC firmware upload parameters (from VESC Tool)
CHUNK_SIZE = 384              # Bytes per write chunk (VESC Tool uses 384)
ERASE_TIMEOUT_S = 20.0        # Timeout waiting for erase response
WRITE_TIMEOUT_S = 3.0         # Timeout waiting for write response
RETRY_COUNT = 3               # Number of retries for failed writes
JUMP_DELAY_S = 0.5            # Delay after jump to bootloader

# Bootloader flash offset: bootloader lives in sector 11 (0x080E0000),
# but COMM_WRITE_NEW_APP_DATA always writes to NEW_APP_BASE (sector 8, 0x08080000).
# To reach sector 11, add 3 * 128KB = 393216 to offset.
# This matches VESC Tool's Commands::writeNewAppDataBootloader().
BOOTLOADER_FLASH_OFFSET = 3 * 128 * 1024  # 393216 = 0x60000


class UploadMode(Enum):
    FIRMWARE = "firmware"
    BOOTLOADER = "bootloader"


def prepare_firmware_with_header(firmware: bytes) -> bytes:
    """
    Prepare firmware data with VESC header.
    Header format: [size:4][CRC16:2]

    This header is prepended to the firmware before upload.
    The bootloader uses this to verify the firmware integrity.
    """
    size = len(firmware)
    crc = crc_hqx(firmware, 0)
    header = struct.pack(">I", size) + struct.pack(">H", crc)
    return header + firmware


def is_chunk_empty(data: bytes) -> bool:
    """Check if chunk is all 0xFF (erased flash state)."""
    return all(b == 0xFF for b in data)


class FirmwareUploader(QThread):
    """
    VESC firmware/bootloader upload worker.

    Protocol (based on VESC Tool):
    1. For firmware:
       - COMM_ERASE_NEW_APP(size) - erase flash
       - COMM_WRITE_NEW_APP_DATA(offset, data) - write chunks (384 bytes each)
       - COMM_JUMP_TO_BOOTLOADER - reboot to apply firmware

    2. For bootloader:
       - COMM_ERASE_BOOTLOADER - erase bootloader area
       - COMM_WRITE_NEW_APP_DATA(offset, data) - write chunks
       - (device resets automatically)
    """
    log = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100
    finished_ok = pyqtSignal()
    aborted = pyqtSignal(str)

    def __init__(self, transport: SerialTransport, bin_path: str,
                 mode: UploadMode = UploadMode.FIRMWARE,
                 chunk_size: int = CHUNK_SIZE):
        super().__init__()
        self.transport = transport
        self.bin_path = bin_path
        self.mode = mode
        self.chunk_size = chunk_size
        self._cancel = False
        # Thread-safe response handling
        self._response_event = threading.Event()
        self._response_lock = threading.Lock()
        self._response_success = False
        self._response_offset = 0

    def cancel(self):
        """Request cancellation of the upload."""
        self._cancel = True

    def notify_mcu_abort(self):
        """Called when MCU sends [ABORT] message."""
        self._cancel = True
        self._response_success = False

    def _on_packet_received(self, payload: bytes):
        """Handle VESC packet response. Called from reader thread."""
        if len(payload) < 1:
            return

        cmd_id = payload[0]

        if cmd_id == CommPacketId.COMM_ERASE_NEW_APP:
            # Response: [COMM_ERASE_NEW_APP][success:1]
            if len(payload) >= 2:
                with self._response_lock:
                    self._response_success = (payload[1] != 0)
                self._response_event.set()

        elif cmd_id == CommPacketId.COMM_ERASE_BOOTLOADER:
            # Response: [COMM_ERASE_BOOTLOADER][success:1]
            if len(payload) >= 2:
                with self._response_lock:
                    self._response_success = (payload[1] != 0)
                self._response_event.set()

        elif cmd_id == CommPacketId.COMM_WRITE_NEW_APP_DATA:
            # Response: [COMM_WRITE_NEW_APP_DATA][success:1][offset:4 (optional)]
            if len(payload) >= 2:
                with self._response_lock:
                    self._response_success = (payload[1] != 0)
                    if len(payload) >= 6:
                        self._response_offset = struct.unpack_from(">I", payload, 2)[0]
                self._response_event.set()

    def _prepare_for_response(self):
        """Clear event and reset state. Must be called BEFORE send_packet()."""
        self._response_event.clear()
        with self._response_lock:
            self._response_success = False
            self._response_offset = 0

    def _wait_for_response(self, timeout_s: float) -> bool:
        """Wait for response. _prepare_for_response() must be called before sending."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._cancel:
                return False
            if self._response_event.wait(timeout=0.1):
                with self._response_lock:
                    return self._response_success
        return False

    def run(self):
        """Main upload thread."""
        # Connect packet handler with DirectConnection for immediate response handling
        # This ensures the slot is called in the reader thread context immediately
        self.transport.packet_received.connect(
            self._on_packet_received,
            Qt.ConnectionType.DirectConnection
        )

        try:
            self._do_upload()
        finally:
            # Disconnect handler
            try:
                self.transport.packet_received.disconnect(self._on_packet_received)
            except Exception:
                pass

    def _do_upload(self):
        """Perform the actual upload."""
        if not self.transport.is_connected():
            self.aborted.emit("[ERROR] Serial not connected.")
            return

        # Read firmware file
        p = Path(self.bin_path)
        if not p.exists():
            self.aborted.emit(f"[ERROR] File not found: {p}")
            return

        raw_firmware = p.read_bytes()
        mode_str = "Bootloader" if self.mode == UploadMode.BOOTLOADER else "Firmware"

        self.log.emit(f"[UPLOAD] {mode_str}: {p.name} ({len(raw_firmware)} bytes)")

        # Prepare firmware data
        if self.mode == UploadMode.FIRMWARE:
            # Add header for firmware (size + CRC16)
            firmware = prepare_firmware_with_header(raw_firmware)
            self.log.emit(f"[UPLOAD] With header: {len(firmware)} bytes (header: 6 bytes)")
            crc = crc_hqx(raw_firmware, 0)
            self.log.emit(f"[UPLOAD] CRC16: 0x{crc:04X}")
        else:
            # Bootloader doesn't need header
            firmware = raw_firmware
            self.log.emit(
                f"[UPLOAD] Bootloader flash offset: 0x{BOOTLOADER_FLASH_OFFSET:06X} "
                f"(sector 11 @ 0x080E0000)"
            )

        total_size = len(firmware)

        # Step 1: Erase flash
        self.log.emit("[UPLOAD] Erasing flash... (this may take up to 20 seconds)")
        self.progress.emit(0)

        if self.mode == UploadMode.BOOTLOADER:
            erase_cmd = build_erase_bootloader()
        else:
            erase_cmd = build_erase_new_app(total_size)

        self._prepare_for_response()
        self.transport.send_packet(erase_cmd)

        if not self._wait_for_response(ERASE_TIMEOUT_S):
            if self._cancel:
                self.aborted.emit("[CANCELLED] Upload cancelled by user.")
            else:
                self.aborted.emit("[ERROR] Erase timed out. Device may not support this operation or is not ready.")
            return

        self.log.emit("[UPLOAD] Flash erased successfully.")

        # Step 2: Write firmware chunks
        offset = 0
        chunks_written = 0
        chunks_skipped = 0

        while offset < total_size:
            if self._cancel:
                self.aborted.emit("[CANCELLED] Upload cancelled by user.")
                return

            # Get next chunk
            chunk_end = min(offset + self.chunk_size, total_size)
            chunk = firmware[offset:chunk_end]

            # Skip empty chunks (all 0xFF) - VESC Tool optimization
            if is_chunk_empty(chunk):
                chunks_skipped += 1
                offset = chunk_end
                percent = int((offset * 100) / total_size)
                self.progress.emit(percent)
                continue

            # Try to write with retries
            success = False
            for retry in range(RETRY_COUNT):
                if self._cancel:
                    self.aborted.emit("[CANCELLED] Upload cancelled by user.")
                    return

                # For bootloader upload, shift offset to reach sector 11
                flash_offset = offset
                if self.mode == UploadMode.BOOTLOADER:
                    flash_offset += BOOTLOADER_FLASH_OFFSET
                write_cmd = build_write_new_app_data(flash_offset, chunk)
                self._prepare_for_response()
                self.transport.send_packet(write_cmd)

                if self._wait_for_response(WRITE_TIMEOUT_S):
                    success = True
                    break
                else:
                    self.log.emit(f"[RETRY] Write at offset {offset} (attempt {retry + 1}/{RETRY_COUNT})")

            if not success:
                self.aborted.emit(f"[ERROR] Write failed at offset {offset} after {RETRY_COUNT} retries.")
                return

            chunks_written += 1
            offset = chunk_end

            # Update progress
            percent = int((offset * 100) / total_size)
            self.progress.emit(percent)

        self.log.emit(f"[UPLOAD] Written {chunks_written} chunks, skipped {chunks_skipped} empty chunks.")

        # Step 3: Jump to bootloader (firmware only)
        if self.mode == UploadMode.FIRMWARE:
            self.log.emit("[UPLOAD] Sending jump to bootloader command...")
            jump_cmd = build_jump_to_bootloader()
            self.transport.send_packet(jump_cmd)
            time.sleep(JUMP_DELAY_S)
            self.log.emit("[UPLOAD] Device should reboot now.")

        # Upload complete
        self.progress.emit(100)
        self.log.emit(f"[SUCCESS] {mode_str} upload complete ({total_size} bytes).")
        self.finished_ok.emit()
