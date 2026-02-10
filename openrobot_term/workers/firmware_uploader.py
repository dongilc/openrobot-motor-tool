"""
VESC CAN firmware upload worker using EID multi-frame protocol.
No Serial connection required — uses PCAN-USB directly.
"""

import struct
import time
from binascii import crc_hqx
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from ..protocol.can_transport import PcanTransport
from ..protocol.commands import (
    build_erase_new_app,
    build_write_new_app_data,
    build_jump_to_bootloader,
    build_erase_bootloader,
)

# VESC firmware upload parameters (from VESC Tool)
CHUNK_SIZE = 384              # Bytes per write chunk (VESC Tool uses 384)
JUMP_DELAY_S = 0.5            # Delay after jump to bootloader

# Bootloader flash offset: sector 11 (0x080E0000) relative to NEW_APP_BASE (0x08080000)
BOOTLOADER_FLASH_OFFSET = 0x080E0000 - 0x08080000  # = 0x60000 = 393216
BOOTLOADER_ERASE_TIMEOUT_S = 10.0


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


# ── CAN firmware upload parameters ──
CAN_CHUNK_DELAY_MS = 50  # ms delay between write chunks (CAN propagation + flash write)
CAN_ERASE_TIMEOUT_S = 30.0


class CanFirmwareUploader(QThread):
    """
    CAN-only firmware upload worker using VESC EID multi-frame protocol.
    No Serial connection required — uses PCAN-USB directly.

    Protocol:
    1. COMM_ERASE_NEW_APP(size) via EID → send_mode=2 (no response expected)
    2. COMM_WRITE_NEW_APP_DATA(offset, data) via EID → send_mode=2
    3. COMM_JUMP_TO_BOOTLOADER via EID → send_mode=2

    target_id=255 for broadcast (all controllers) or specific ID for single.
    """
    log = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100
    finished_ok = pyqtSignal()
    aborted = pyqtSignal(str)

    def __init__(self, can_transport: PcanTransport, bin_path: str,
                 target_id: int = 255, chunk_size: int = CHUNK_SIZE,
                 bootloader_mode: bool = False):
        super().__init__()
        self._can = can_transport
        self.bin_path = bin_path
        self.target_id = target_id
        self.chunk_size = chunk_size
        self.bootloader_mode = bootloader_mode
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self._do_upload()
        except Exception as ex:
            self.aborted.emit(f"[ERROR] {ex}")

    def _do_upload(self):
        if not self._can.is_connected():
            self.aborted.emit("[ERROR] PCAN not connected.")
            return

        p = Path(self.bin_path)
        if not p.exists():
            self.aborted.emit(f"[ERROR] File not found: {p}")
            return

        raw_data = p.read_bytes()
        target_str = "ALL (broadcast)" if self.target_id == 255 else f"ID {self.target_id}"

        if self.bootloader_mode:
            self._do_bootloader_upload(raw_data, p.name, target_str)
        else:
            self._do_firmware_upload(raw_data, p.name, target_str)

    def _do_firmware_upload(self, raw_firmware: bytes, filename: str, target_str: str):
        """
        Firmware upload protocol:
          1. COMM_ERASE_NEW_APP(size)  — erase staging area (sectors 8-10)
          2. COMM_WRITE_NEW_APP_DATA(offset, data) — write [header+fw] to staging
          3. COMM_JUMP_TO_BOOTLOADER — bootloader copies staging → app (sectors 0-7)
        """
        firmware = prepare_firmware_with_header(raw_firmware)
        total_size = len(firmware)

        self.log.emit(f"[CAN UPLOAD] Firmware: {filename} ({len(raw_firmware)} bytes)")
        self.log.emit(f"[CAN UPLOAD] With header: {total_size} bytes, Target: {target_str}")
        crc = crc_hqx(raw_firmware, 0)
        self.log.emit(f"[CAN UPLOAD] CRC16: 0x{crc:04X}")

        # Step 1: Erase staging area (sectors 8-10)
        self.log.emit("[CAN UPLOAD] Erasing staging area (sectors 8-10)...")
        self.progress.emit(0)

        erase_payload = build_erase_new_app(total_size)
        self._can.send_vesc_command(self.target_id, erase_payload, send_mode=2)

        self.log.emit(f"[CAN UPLOAD] Waiting {CAN_ERASE_TIMEOUT_S}s for erase...")
        deadline = time.time() + CAN_ERASE_TIMEOUT_S
        while time.time() < deadline:
            if self._cancel:
                self.aborted.emit("[CANCELLED] Upload cancelled by user.")
                return
            time.sleep(0.5)

        self.log.emit("[CAN UPLOAD] Erase complete. Starting write...")

        # Step 2: Write chunks to staging (offset 0 = 0x08080000)
        chunks_written, chunks_skipped = self._write_chunks(firmware, total_size, base_offset=0)

        self.log.emit(
            f"[CAN UPLOAD] Written {chunks_written} chunks, "
            f"skipped {chunks_skipped} empty chunks."
        )

        # Step 3: Jump to bootloader (bootloader copies staging → app)
        self.log.emit("[CAN UPLOAD] Sending jump to bootloader...")
        jump_payload = build_jump_to_bootloader()
        self._can.send_vesc_command(self.target_id, jump_payload, send_mode=2)
        time.sleep(JUMP_DELAY_S)

        self.progress.emit(100)
        self.log.emit(f"[CAN UPLOAD] Firmware upload complete ({total_size} bytes).")
        self.finished_ok.emit()

    def _do_bootloader_upload(self, raw_bootloader: bytes, filename: str, target_str: str):
        """
        Bootloader upload protocol — writes directly to sector 11 (0x080E0000):
          1. COMM_ERASE_BOOTLOADER — erase sector 11
          2. COMM_WRITE_NEW_APP_DATA(offset=0x60000, data) — write raw .bin to sector 11
          No header, no jump. The bootloader is written in-place.

        flash_helper_write_new_app_data() writes to (0x08080000 + offset),
        so offset 0x60000 targets 0x080E0000 (sector 11) directly.
        """
        total_size = len(raw_bootloader)

        self.log.emit(f"[CAN UPLOAD] Bootloader: {filename} ({total_size} bytes)")
        self.log.emit(f"[CAN UPLOAD] Target: {target_str}")
        self.log.emit(f"[CAN UPLOAD] Write offset: 0x{BOOTLOADER_FLASH_OFFSET:X} "
                       f"(sector 11 = 0x080E0000)")

        # Step 1: Erase bootloader sector (sector 11 only)
        self.log.emit("[CAN UPLOAD] Erasing bootloader sector (sector 11)...")
        self.progress.emit(0)

        erase_bl_payload = build_erase_bootloader()
        self._can.send_vesc_command(self.target_id, erase_bl_payload, send_mode=2)

        self.log.emit(f"[CAN UPLOAD] Waiting {BOOTLOADER_ERASE_TIMEOUT_S}s for erase...")
        deadline = time.time() + BOOTLOADER_ERASE_TIMEOUT_S
        while time.time() < deadline:
            if self._cancel:
                self.aborted.emit("[CANCELLED] Upload cancelled by user.")
                return
            time.sleep(0.5)

        self.log.emit("[CAN UPLOAD] Sector 11 erased. Writing bootloader...")

        # Step 2: Write raw bootloader to sector 11 (no header!)
        chunks_written, chunks_skipped = self._write_chunks(
            raw_bootloader, total_size, base_offset=BOOTLOADER_FLASH_OFFSET)

        self.log.emit(
            f"[CAN UPLOAD] Written {chunks_written} chunks, "
            f"skipped {chunks_skipped} empty chunks."
        )

        # No jump needed — bootloader is already in-place at sector 11
        self.progress.emit(100)
        self.log.emit(f"[CAN UPLOAD] Bootloader upload complete ({total_size} bytes).")
        self.finished_ok.emit()

    def _write_chunks(self, data: bytes, total_size: int,
                      base_offset: int = 0) -> tuple[int, int]:
        """Write data in chunks via COMM_WRITE_NEW_APP_DATA. Returns (written, skipped)."""
        offset = 0
        chunks_written = 0
        chunks_skipped = 0

        while offset < total_size:
            if self._cancel:
                self.aborted.emit("[CANCELLED] Upload cancelled by user.")
                return chunks_written, chunks_skipped

            chunk_end = min(offset + self.chunk_size, total_size)
            chunk = data[offset:chunk_end]

            if is_chunk_empty(chunk):
                chunks_skipped += 1
                offset = chunk_end
                percent = int((offset * 100) / total_size)
                self.progress.emit(percent)
                continue

            write_payload = build_write_new_app_data(base_offset + offset, chunk)
            self._can.send_vesc_command(self.target_id, write_payload, send_mode=2)

            chunks_written += 1
            offset = chunk_end

            percent = int((offset * 100) / total_size)
            self.progress.emit(percent)

            time.sleep(CAN_CHUNK_DELAY_MS / 1000.0)

        return chunks_written, chunks_skipped
