"""
VESC CAN firmware upload worker using EID multi-frame protocol.
No Serial connection required — uses PCAN-USB directly.

v2.2: Combined FW+BL upload mode, config backup/restore around firmware update.
v2.3: Multi-device broadcast config backup/restore via UUID-based CAN ID prediction.
"""

import struct
import time
import threading
from dataclasses import dataclass, field
from binascii import crc_hqx
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal, Qt

from ..protocol.can_transport import PcanTransport
from ..protocol.commands import (
    CommPacketId,
    build_erase_new_app,
    build_write_new_app_data,
    build_jump_to_bootloader,
    build_erase_bootloader,
    build_get_mcconf,
    build_get_appconf,
    build_get_fw_version,
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


def crc32c(data: bytes) -> int:
    """CRC32-Castagnoli — matches VESC firmware utils_crc32c()."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            mask = -(crc & 1) & 0xFFFFFFFF
            crc = (crc >> 1) ^ (0x82F63B78 & mask)
    return (~crc) & 0xFFFFFFFF


def uuid_to_default_can_id(uuid_bytes: bytes) -> int:
    """Compute VESC default CAN ID from 12-byte STM32 UUID.
    Matches firmware: hw_id_from_uuid() = utils_crc32c(UUID, 12) & 0x7F"""
    return crc32c(uuid_bytes) & 0x7F


def parse_uuid_from_fw_version(fw_data: bytes) -> bytes | None:
    """Extract 12-byte STM32 UUID from COMM_FW_VERSION response data.
    Format: [major:1][minor:1][hw_name:N+1 (null-terminated)][uuid:12][...]
    """
    if len(fw_data) < 4:
        return None
    # Find null terminator of HW_NAME starting at offset 2
    try:
        null_pos = fw_data.index(0x00, 2)
    except ValueError:
        return None
    uuid_start = null_pos + 1
    if len(fw_data) < uuid_start + 12:
        return None
    return bytes(fw_data[uuid_start:uuid_start + 12])


# ── CAN firmware upload parameters ──
CAN_CHUNK_DELAY_MS = 50  # ms delay between write chunks (CAN propagation + flash write)
CAN_ERASE_TIMEOUT_S = 30.0

# ── Config backup/restore parameters (v2.2+) ──
CONFIG_RESPONSE_TIMEOUT_S = 5.0   # Max wait for GET_MCCONF/GET_APPCONF response
REBOOT_PROBE_MIN_WAIT_S = 3.0    # Minimum wait after jump before probing
REBOOT_PROBE_TIMEOUT_S = 10.0    # Max time to wait for device to come back
REBOOT_PROBE_INTERVAL_S = 1.0    # Probe interval (COMM_FW_VERSION)


@dataclass
class _DeviceBackup:
    """Per-device config backup for multi-device restore."""
    original_id: int
    uuid: bytes | None = None
    expected_default_id: int | None = None
    mcconf: bytes | None = None
    appconf: bytes | None = None


class CanFirmwareUploader(QThread):
    """
    CAN-only firmware upload worker using VESC EID multi-frame protocol.
    No Serial connection required — uses PCAN-USB directly.

    Protocol:
    1. COMM_ERASE_NEW_APP(size) via EID → send_mode=2 (no response expected)
    2. COMM_WRITE_NEW_APP_DATA(offset, data) via EID → send_mode=2
    3. COMM_JUMP_TO_BOOTLOADER via EID → send_mode=2

    target_id=255 for broadcast (all controllers) or specific ID for single.

    Combined mode (bootloader + firmware):
    1. Erase sector 11 → write bootloader in-place
    2. Erase sectors 8-10 → write firmware to staging
    3. Jump to bootloader → new bootloader copies staging → app
    """
    log = pyqtSignal(str)
    progress = pyqtSignal(int)  # 0-100
    finished_ok = pyqtSignal()
    aborted = pyqtSignal(str)
    config_restore_skipped = pyqtSignal(str)  # reason (signature mismatch, timeout, etc.)

    def __init__(self, can_transport: PcanTransport, bin_path: str,
                 target_id: int = 255, chunk_size: int = CHUNK_SIZE,
                 bootloader_mode: bool = False,
                 combined_mode: bool = False, bl_bin_path: str = "",
                 config_backup: bool = False,
                 discovered_ids: list[int] | None = None):
        super().__init__()
        self._can = can_transport
        self.bin_path = bin_path
        self.target_id = target_id
        self.chunk_size = chunk_size
        self.bootloader_mode = bootloader_mode
        self.combined_mode = combined_mode
        self.bl_bin_path = bl_bin_path
        self.config_backup = config_backup
        self._cancel = False
        self._discovered_ids = list(discovered_ids) if discovered_ids else []

        # Config backup/restore state (v2.3 — multi-device)
        self._device_backups: list[_DeviceBackup] = []
        self._response_event = threading.Event()
        self._response_cmd: int = 0
        self._response_data: bytes = b""
        self._scan_active: bool = False
        self._scan_results: set[int] = set()

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            # Connect CAN response signal for config backup/restore
            if self.config_backup:
                self._can.vesc_response_received.connect(
                    self._on_vesc_response, Qt.ConnectionType.DirectConnection)
            try:
                self._do_upload()
            finally:
                if self.config_backup:
                    try:
                        self._can.vesc_response_received.disconnect(self._on_vesc_response)
                    except TypeError:
                        pass
        except Exception as ex:
            self.aborted.emit(f"[ERROR] {ex}")

    def _on_vesc_response(self, sender_id: int, payload: bytes):
        """CAN response handler (called from reader thread via DirectConnection)."""
        if not payload:
            return
        cmd_id = payload[0]
        # Collect scan results during post-reboot CAN scan
        if self._scan_active and cmd_id == CommPacketId.COMM_FW_VERSION:
            self._scan_results.add(sender_id)
        if cmd_id == self._response_cmd:
            self._response_data = payload[1:]  # strip cmd byte
            self._response_event.set()

    def _wait_for_response(self, cmd_id: int, timeout: float = CONFIG_RESPONSE_TIMEOUT_S) -> bytes | None:
        """Send-and-wait helper: wait for a specific COMM response. Returns data or None on timeout."""
        self._response_cmd = cmd_id
        self._response_data = b""
        self._response_event.clear()
        if self._response_event.wait(timeout):
            return self._response_data
        return None

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

        # ── Config backup (before upload) ──
        if self.config_backup and not self.bootloader_mode:
            if not self._backup_configs():
                return  # backup failed and user was notified

        # ── Upload ──
        if self.combined_mode:
            bl_p = Path(self.bl_bin_path)
            if not bl_p.exists():
                self.aborted.emit(f"[ERROR] Bootloader file not found: {bl_p}")
                return
            bl_data = bl_p.read_bytes()
            self._do_combined_upload(raw_data, p.name, bl_data, bl_p.name, target_str)
        elif self.bootloader_mode:
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

        # Config restore after reboot (if backup was performed)
        if self.config_backup and self._device_backups:
            self.log.emit("")
            self._restore_configs()

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

    def _do_combined_upload(self, raw_firmware: bytes, fw_filename: str,
                            raw_bootloader: bytes, bl_filename: str, target_str: str):
        """
        Combined upload: bootloader first, then firmware.
          Phase 1 (0-30%): Upload bootloader to sector 11 in-place
          Phase 2 (30-100%): Upload firmware to staging + jump to bootloader

        Bootloader MUST be uploaded first so the new bootloader is ready
        when JUMP_TO_BOOTLOADER is called at the end of firmware upload.
        """
        bl_size = len(raw_bootloader)
        firmware = prepare_firmware_with_header(raw_firmware)
        fw_total = len(firmware)
        fw_crc = crc_hqx(raw_firmware, 0)

        self.log.emit("=" * 60)
        self.log.emit("[COMBINED] Firmware + Bootloader upload")
        self.log.emit(f"[COMBINED] Bootloader: {bl_filename} ({bl_size} bytes)")
        self.log.emit(f"[COMBINED] Firmware:   {fw_filename} ({len(raw_firmware)} bytes, "
                       f"with header: {fw_total} bytes)")
        self.log.emit(f"[COMBINED] FW CRC16: 0x{fw_crc:04X}")
        self.log.emit(f"[COMBINED] Target: {target_str}")
        self.log.emit("=" * 60)

        # ── Phase 1: Bootloader upload (progress 0-30%) ──
        self.log.emit("")
        self.log.emit("[PHASE 1/2] Bootloader upload to sector 11...")
        self.log.emit(f"[PHASE 1/2] Write offset: 0x{BOOTLOADER_FLASH_OFFSET:X} "
                       f"(sector 11 = 0x080E0000)")
        self.progress.emit(0)

        # Erase sector 11
        self.log.emit("[PHASE 1/2] Erasing bootloader sector (sector 11)...")
        erase_bl_payload = build_erase_bootloader()
        self._can.send_vesc_command(self.target_id, erase_bl_payload, send_mode=2)

        self.log.emit(f"[PHASE 1/2] Waiting {BOOTLOADER_ERASE_TIMEOUT_S}s for erase...")
        deadline = time.time() + BOOTLOADER_ERASE_TIMEOUT_S
        while time.time() < deadline:
            if self._cancel:
                self.aborted.emit("[CANCELLED] Upload cancelled by user.")
                return
            time.sleep(0.5)

        self.log.emit("[PHASE 1/2] Sector 11 erased. Writing bootloader...")

        # Write bootloader chunks (progress 5-28%)
        chunks_written, chunks_skipped = self._write_chunks(
            raw_bootloader, bl_size,
            base_offset=BOOTLOADER_FLASH_OFFSET,
            progress_start=5, progress_end=28)

        self.log.emit(
            f"[PHASE 1/2] Bootloader done — {chunks_written} chunks written, "
            f"{chunks_skipped} skipped."
        )
        self.progress.emit(30)

        if self._cancel:
            return

        # ── Phase 2: Firmware upload (progress 30-100%) ──
        self.log.emit("")
        self.log.emit("[PHASE 2/2] Firmware upload to staging area...")

        # Erase staging (sectors 8-10)
        self.log.emit("[PHASE 2/2] Erasing staging area (sectors 8-10)...")
        erase_payload = build_erase_new_app(fw_total)
        self._can.send_vesc_command(self.target_id, erase_payload, send_mode=2)

        self.log.emit(f"[PHASE 2/2] Waiting {CAN_ERASE_TIMEOUT_S}s for erase...")
        deadline = time.time() + CAN_ERASE_TIMEOUT_S
        while time.time() < deadline:
            if self._cancel:
                self.aborted.emit("[CANCELLED] Upload cancelled by user.")
                return
            time.sleep(0.5)

        self.log.emit("[PHASE 2/2] Erase complete. Writing firmware...")

        # Write firmware chunks (progress 35-95%)
        chunks_written, chunks_skipped = self._write_chunks(
            firmware, fw_total,
            base_offset=0,
            progress_start=35, progress_end=95)

        self.log.emit(
            f"[PHASE 2/2] Firmware written — {chunks_written} chunks, "
            f"{chunks_skipped} skipped."
        )

        if self._cancel:
            return

        # Jump to bootloader
        self.log.emit("[PHASE 2/2] Sending jump to bootloader...")
        jump_payload = build_jump_to_bootloader()
        self._can.send_vesc_command(self.target_id, jump_payload, send_mode=2)
        time.sleep(JUMP_DELAY_S)

        # Config restore after reboot (if backup was performed)
        if self.config_backup and self._device_backups:
            self.log.emit("")
            self._restore_configs()

        self.progress.emit(100)
        self.log.emit("")
        self.log.emit("=" * 60)
        self.log.emit("[COMBINED] Upload complete!")
        self.log.emit(f"[COMBINED] Bootloader: {bl_size} bytes → sector 11")
        self.log.emit(f"[COMBINED] Firmware:   {fw_total} bytes → staging → app")
        self.log.emit("=" * 60)
        self.finished_ok.emit()

    # ── Config backup/restore (v2.3 — multi-device) ─────────────

    def _backup_configs(self) -> bool:
        """
        Read UUID, MCCONF, APPCONF from each target device before firmware upload.
        Broadcast mode: backs up all discovered devices.
        Single target: backs up just that one device.
        Returns True to proceed with upload, False to abort.
        """
        # Determine which IDs to back up
        if self.target_id == 255:
            backup_ids = list(self._discovered_ids)
            if not backup_ids:
                self.log.emit("[CONFIG BACKUP] Broadcast mode but no discovered devices — skipped.")
                self.config_backup = False
                return True
            self.log.emit(f"[CONFIG BACKUP] Broadcast mode — backing up {len(backup_ids)} "
                          f"devices: {backup_ids}")
        else:
            backup_ids = [self.target_id]
            self.log.emit("[CONFIG BACKUP] Reading current configuration...")

        for dev_id in backup_ids:
            if self._cancel:
                self.aborted.emit("[CANCELLED] Upload cancelled by user.")
                return False

            backup = _DeviceBackup(original_id=dev_id)
            self.log.emit(f"[CONFIG BACKUP] ── Device ID {dev_id} ──")

            # Read UUID via COMM_FW_VERSION
            self._can.send_vesc_command(dev_id, build_get_fw_version(), send_mode=0)
            fw_data = self._wait_for_response(CommPacketId.COMM_FW_VERSION, timeout=3.0)
            if fw_data is not None:
                uuid_bytes = parse_uuid_from_fw_version(fw_data)
                if uuid_bytes is not None:
                    backup.uuid = uuid_bytes
                    backup.expected_default_id = uuid_to_default_can_id(uuid_bytes)
                    uuid_hex = uuid_bytes.hex().upper()
                    self.log.emit(
                        f"[CONFIG BACKUP]   UUID: {uuid_hex} → "
                        f"default ID after reset: {backup.expected_default_id}")
                else:
                    self.log.emit(f"[CONFIG BACKUP]   UUID parse failed.")
            else:
                self.log.emit(f"[CONFIG BACKUP]   FW version timeout — UUID unknown.")

            # Read MCCONF
            self._can.send_vesc_command(dev_id, build_get_mcconf(), send_mode=0)
            mc_data = self._wait_for_response(CommPacketId.COMM_GET_MCCONF)
            if mc_data is not None and len(mc_data) >= 8:
                backup.mcconf = mc_data
                mc_sig = struct.unpack(">I", mc_data[:4])[0]
                self.log.emit(f"[CONFIG BACKUP]   MCCONF: {len(mc_data)} bytes, "
                              f"sig=0x{mc_sig:08X}")
            else:
                self.log.emit(f"[CONFIG BACKUP]   MCCONF read timeout.")

            # Read APPCONF
            self._can.send_vesc_command(dev_id, build_get_appconf(), send_mode=0)
            app_data = self._wait_for_response(CommPacketId.COMM_GET_APPCONF)
            if app_data is not None and len(app_data) >= 8:
                backup.appconf = app_data
                app_sig = struct.unpack(">I", app_data[:4])[0]
                self.log.emit(f"[CONFIG BACKUP]   APPCONF: {len(app_data)} bytes, "
                              f"sig=0x{app_sig:08X}")
            else:
                self.log.emit(f"[CONFIG BACKUP]   APPCONF read timeout.")

            if backup.mcconf is not None:
                self._device_backups.append(backup)

        if not self._device_backups:
            self.log.emit("[CONFIG BACKUP] No devices backed up — config restore disabled.")
            self.config_restore_skipped.emit("No device configs backed up (all timeouts)")
            self.config_backup = False
            return True

        self.log.emit(f"[CONFIG BACKUP] {len(self._device_backups)} device(s) backed up. "
                      f"Proceeding with upload...")
        self.log.emit("")
        return True

    def _find_device(self, backup: _DeviceBackup) -> int | None:
        """
        Find a specific device after reboot using UUID-predicted CAN ID.
        Probe order: expected_default_id → original_id → full scan (fallback).
        Verifies UUID match to avoid confusion with other devices.
        """
        fw_probe = build_get_fw_version()

        # Build probe list: expected default first (most likely after reset)
        probe_ids = []
        if backup.expected_default_id is not None and backup.expected_default_id > 0:
            probe_ids.append(backup.expected_default_id)
        if backup.original_id not in probe_ids and backup.original_id > 0:
            probe_ids.append(backup.original_id)

        self.log.emit(f"[CONFIG RESTORE]   Probing IDs {probe_ids} for device "
                      f"(was ID {backup.original_id})...")

        probe_deadline = time.time() + REBOOT_PROBE_TIMEOUT_S
        while time.time() < probe_deadline:
            if self._cancel:
                return None
            for pid in probe_ids:
                self._can.send_vesc_command(pid, fw_probe, send_mode=0)
                fw_data = self._wait_for_response(CommPacketId.COMM_FW_VERSION,
                                                  timeout=REBOOT_PROBE_INTERVAL_S)
                if fw_data is None:
                    continue
                # Verify UUID to ensure this is the right device
                if backup.uuid is not None:
                    resp_uuid = parse_uuid_from_fw_version(fw_data)
                    if resp_uuid != backup.uuid:
                        continue  # Different device on this ID
                ver = ""
                if len(fw_data) >= 2:
                    ver = f" (FW v{fw_data[0]}.{fw_data[1]:02d})"
                if pid == backup.original_id:
                    self.log.emit(f"[CONFIG RESTORE]   Found on original ID {pid}{ver}")
                else:
                    self.log.emit(f"[CONFIG RESTORE]   Found on UUID-default ID {pid}"
                                  f" (was {backup.original_id}){ver}")
                return pid

        self.log.emit(f"[CONFIG RESTORE]   Device (was ID {backup.original_id}) not found.")
        return None

    def _restore_single_device(self, restore_id: int, backup: _DeviceBackup):
        """Restore MCCONF and APPCONF to a single device."""
        # Read new MCCONF for signature check
        self._can.send_vesc_command(restore_id, build_get_mcconf(), send_mode=0)
        new_mc = self._wait_for_response(CommPacketId.COMM_GET_MCCONF)

        if new_mc is None or len(new_mc) < 4:
            self.log.emit(f"[CONFIG RESTORE]   Cannot read MCCONF from ID {restore_id} — skipped.")
            return

        old_mc_sig = struct.unpack(">I", backup.mcconf[:4])[0]
        new_mc_sig = struct.unpack(">I", new_mc[:4])[0]

        # Restore MCCONF
        if old_mc_sig == new_mc_sig:
            self.log.emit(f"[CONFIG RESTORE]   MCCONF sig match (0x{new_mc_sig:08X}) — restoring...")
            set_mc = bytes([CommPacketId.COMM_SET_MCCONF]) + backup.mcconf
            self._can.send_vesc_command(restore_id, set_mc, send_mode=0)
            ack = self._wait_for_response(CommPacketId.COMM_SET_MCCONF)
            if ack is not None:
                self.log.emit(f"[CONFIG RESTORE]   MCCONF restored.")
            else:
                self.log.emit(f"[CONFIG RESTORE]   MCCONF write sent (no ACK).")
            time.sleep(0.3)
        else:
            self.log.emit(f"[CONFIG RESTORE]   MCCONF sig mismatch "
                          f"(old=0x{old_mc_sig:08X}, new=0x{new_mc_sig:08X}) — skipped.")
            self.config_restore_skipped.emit(
                f"MCCONF signature mismatch for device {backup.original_id}")

        # Restore APPCONF (contains CAN ID)
        if backup.appconf is not None:
            self._can.send_vesc_command(restore_id, build_get_appconf(), send_mode=0)
            new_app = self._wait_for_response(CommPacketId.COMM_GET_APPCONF)

            if new_app is not None and len(new_app) >= 4:
                old_app_sig = struct.unpack(">I", backup.appconf[:4])[0]
                new_app_sig = struct.unpack(">I", new_app[:4])[0]

                if old_app_sig == new_app_sig:
                    self.log.emit(f"[CONFIG RESTORE]   APPCONF sig match — restoring "
                                  f"(CAN ID → {backup.original_id})...")
                    set_app = bytes([CommPacketId.COMM_SET_APPCONF]) + backup.appconf
                    self._can.send_vesc_command(restore_id, set_app, send_mode=0)
                    ack = self._wait_for_response(CommPacketId.COMM_SET_APPCONF)
                    if ack is not None:
                        self.log.emit(f"[CONFIG RESTORE]   APPCONF restored.")
                    else:
                        self.log.emit(f"[CONFIG RESTORE]   APPCONF write sent (no ACK).")
                else:
                    self.log.emit(f"[CONFIG RESTORE]   APPCONF sig mismatch — skipped.")
                    self.config_restore_skipped.emit(
                        f"APPCONF signature mismatch for device {backup.original_id}")
            else:
                self.log.emit(f"[CONFIG RESTORE]   Cannot read new APPCONF — skipped.")

    def _restore_configs(self):
        """
        After firmware upload + reboot, find each backed-up device
        (by UUID-predicted CAN ID) and restore its config.
        """
        n = len(self._device_backups)
        self.log.emit(f"[CONFIG RESTORE] Restoring {n} device(s)...")

        # Wait for bootloader to copy firmware + boot
        self.log.emit(f"[CONFIG RESTORE] Minimum wait {REBOOT_PROBE_MIN_WAIT_S}s...")
        wait_end = time.time() + REBOOT_PROBE_MIN_WAIT_S
        while time.time() < wait_end:
            if self._cancel:
                return
            time.sleep(0.5)

        restored = 0
        failed = 0

        for i, backup in enumerate(self._device_backups):
            if self._cancel:
                break
            self.log.emit(f"[CONFIG RESTORE] ── Device {i+1}/{n} "
                          f"(was ID {backup.original_id}) ──")

            restore_id = self._find_device(backup)
            if restore_id is None:
                self.log.emit(f"[CONFIG RESTORE]   NOT FOUND — skipped.")
                self.config_restore_skipped.emit(
                    f"Device (was ID {backup.original_id}) not found after reboot")
                failed += 1
                continue

            time.sleep(0.3)
            self._restore_single_device(restore_id, backup)
            restored += 1
            time.sleep(0.3)

        self.log.emit(f"[CONFIG RESTORE] Complete: {restored} restored, {failed} not found.")

    def _write_chunks(self, data: bytes, total_size: int,
                      base_offset: int = 0,
                      progress_start: int = 0,
                      progress_end: int = 100) -> tuple[int, int]:
        """
        Write data in chunks via COMM_WRITE_NEW_APP_DATA. Returns (written, skipped).

        progress_start/progress_end: map chunk progress to a sub-range of the
        overall progress bar (e.g. 5-28% for bootloader phase in combined mode).
        Default 0-100 for standalone uploads.
        """
        offset = 0
        chunks_written = 0
        chunks_skipped = 0
        progress_range = progress_end - progress_start

        while offset < total_size:
            if self._cancel:
                self.aborted.emit("[CANCELLED] Upload cancelled by user.")
                return chunks_written, chunks_skipped

            chunk_end = min(offset + self.chunk_size, total_size)
            chunk = data[offset:chunk_end]

            if is_chunk_empty(chunk):
                chunks_skipped += 1
                offset = chunk_end
                percent = progress_start + int((offset * progress_range) / total_size)
                self.progress.emit(percent)
                continue

            write_payload = build_write_new_app_data(base_offset + offset, chunk)
            self._can.send_vesc_command(self.target_id, write_payload, send_mode=2)

            chunks_written += 1
            offset = chunk_end

            percent = progress_start + int((offset * progress_range) / total_size)
            self.progress.emit(percent)

            time.sleep(CAN_CHUNK_DELAY_MS / 1000.0)

        return chunks_written, chunks_skipped
