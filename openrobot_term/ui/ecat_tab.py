"""
EtherCAT tab: LAN9252 EtherCAT slave monitor ported from tkinter to PyQt6.
Consolidates 8 original tabs into 4 sub-tabs:
  1. Status & Control  (slave overview + AL state/control + diagnose)
  2. PDO Test          (motor control via FPWR/FPRD)
  3. Diagnostics       (DL status + error counters + register viewer)
  4. EEPROM            (SII programming)
"""

import struct
import sys
import time
import json
import os
import ctypes
import threading

import pysoem

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QGroupBox,
    QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit, QCheckBox,
    QProgressBar, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QMessageBox, QFileDialog, QSizePolicy, QSpinBox, QGridLayout,
    QSplitter, QScrollArea,
)
from PyQt6.QtGui import QTextCursor, QFont
import pyqtgraph as pg


# ═══════════════════════════════════════════════════════════════
# Constants (from lan9252_monitor.py)
# ═══════════════════════════════════════════════════════════════

LAN9252_REGISTERS = {
    0x0000: ("Type", 1, "ESC type"), 0x0001: ("Revision", 1, "ESC revision"),
    0x0002: ("Build", 2, "Build number"), 0x0004: ("FMMUs", 1, "FMMU count"),
    0x0005: ("SyncManagers", 1, "SM count"), 0x0006: ("RAM Size", 1, "RAM (KB)"),
    0x0007: ("Port Descriptor", 1, "Port desc"), 0x0008: ("ESC Features", 2, "Features"),
    0x0010: ("Station Address", 2, "Configured addr"), 0x0012: ("Station Alias", 2, "Alias"),
    0x0100: ("DL Control", 4, "Data link control"), 0x0110: ("DL Status", 2, "Data link status"),
    0x0120: ("AL Control", 2, "AL control"), 0x0130: ("AL Status", 2, "AL status"),
    0x0134: ("AL Status Code", 2, "AL status code"),
    0x0140: ("PDI Control", 1, "PDI control"), 0x0141: ("ESC Config", 1, "ESC config"),
    0x0200: ("ECAT Event Mask", 2, "Event mask"), 0x0220: ("ECAT Event Req", 2, "Event req"),
    0x0300: ("RX Err Port 0", 4, "Port0 RX error"), 0x0308: ("RX Err Port 1", 4, "Port1 RX error"),
    0x030C: ("Fwd RX Err", 4, "Forwarded RX error"),
    0x0400: ("WD Divider", 2, "Watchdog divider"),
    0x0410: ("WD Time PDI", 2, "PDI WD time"), 0x0420: ("WD Time PD", 2, "PD WD time"),
    0x0440: ("WD Status PD", 2, "PD WD status"),
    0x0442: ("WD Counter PD", 1, "PD WD counter"), 0x0443: ("WD Counter PDI", 1, "PDI WD counter"),
    0x0800: ("SM0 Addr", 2, "SM0 start"), 0x0802: ("SM0 Len", 2, "SM0 length"),
    0x0804: ("SM0 Ctrl", 1, "SM0 control"), 0x0805: ("SM0 Status", 1, "SM0 status"),
    0x0806: ("SM0 Activate", 1, "SM0 activate"),
    0x0808: ("SM1 Addr", 2, "SM1 start"), 0x080A: ("SM1 Len", 2, "SM1 length"),
    0x080C: ("SM1 Ctrl", 1, "SM1 control"), 0x080D: ("SM1 Status", 1, "SM1 status"),
    0x080E: ("SM1 Activate", 1, "SM1 activate"),
    0x0810: ("SM2 Addr", 2, "SM2 start"), 0x0812: ("SM2 Len", 2, "SM2 length"),
    0x0814: ("SM2 Ctrl", 1, "SM2 control"), 0x0815: ("SM2 Status", 1, "SM2 status"),
    0x0816: ("SM2 Activate", 1, "SM2 activate"),
    0x0818: ("SM3 Addr", 2, "SM3 start"), 0x081A: ("SM3 Len", 2, "SM3 length"),
    0x081C: ("SM3 Ctrl", 1, "SM3 control"), 0x081D: ("SM3 Status", 1, "SM3 status"),
    0x081E: ("SM3 Activate", 1, "SM3 activate"),
    0x0600: ("FMMU0 Logical Start", 4, "FMMU0 logical"), 0x0604: ("FMMU0 Length", 2, "FMMU0 len"),
    0x0502: ("SII EEPROM Ctrl", 2, "EEPROM ctrl"), 0x0504: ("SII EEPROM Addr", 2, "EEPROM addr"),
    0x0508: ("SII EEPROM Data", 4, "EEPROM data"),
}

AL_STATE_MAP = {0x01: "INIT", 0x02: "PRE-OP", 0x03: "BOOT", 0x04: "SAFE-OP", 0x08: "OP"}

AL_STATUS_CODES = {
    0x0000: "No error", 0x0001: "Unspecified error",
    0x0011: "Invalid requested state change", 0x0012: "Unknown requested state",
    0x0016: "Invalid mailbox configuration (PRE-OP)",
    0x0017: "Invalid sync manager configuration",
    0x001B: "Sync manager watchdog",
}

DL_STATUS_BITS = {
    0: ("PDI operational", "ON", "OFF"), 1: ("DL user watchdog", "ON", "OFF"),
    4: ("Link Port 0 (ECAT IN)", "Connected", "No cable"),
    5: ("Link Port 1 (ECAT OUT)", "Connected", "No cable"),
    8: ("Loop Port 0", "Loopback", "Closed"), 9: ("Loop Port 1", "Loopback", "Closed"),
    12: ("Signal Port 0", "Signal", "None"), 13: ("Signal Port 1", "Signal", "None"),
}

PDI_TYPES = {
    "SPI Host (0x80)": 0x80, "SPI Slave (0x05)": 0x05,
    "HBI 8-bit": 0x08, "HBI 16-bit": 0x09, "Digital I/O": 0x04, "SQI": 0x85,
}

MODE_NAMES = {
    0x80: "MOTOR_OFF", 0x81: "MOTOR_STOP", 0x88: "MOTOR_START",
    0xA1: "TORQUE", 0xA2: "SPEED", 0xA4: "MULTITURN_POS",
    0xA5: "DUTY", 0xC0: "MIT_CONTROL",
}

MODE_FIELD_DEFS = {
    0x80: [], 0x81: [], 0x88: [],
    0xA1: [("Current", "target_torque", 10000, "A"), ("Damping", "kp", 100, "")],
    0xA2: [("Velocity", "target_velocity", 10000, "dps")],
    0xA4: [("Position", "target_position", 10000, "deg"), ("Speed limit", "target_velocity", 10000, "dps")],
    0xA5: [("Duty", "target_torque", 10000, "-1 ~ +1")],
    0xC0: [("Position", "target_position", 10000, "rad"), ("Velocity", "target_velocity", 10000, "rad/s"),
           ("Torque", "target_torque", 10000, "Nm"), ("Kp", "kp", 100, "Nm/rad"), ("Kd", "kd", 1000, "Nm·s/rad")],
}


# ═══════════════════════════════════════════════════════════════
# SII Image Builder
# ═══════════════════════════════════════════════════════════════

def sii_crc8(data_bytes):
    crc = 0xFF
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def build_sii_image(vendor_id=0, product_code=1, revision=1, serial_number=1,
                    pdi_type=0x80, device_name="LAN9252 Slave", mailbox=True, compact=False):
    image = []
    image.append(pdi_type & 0xFF)
    for _ in range(6):
        image.append(0x0000)
    config_bytes = b""
    for w in image[:7]:
        config_bytes += struct.pack("<H", w)
    image.append(sii_crc8(config_bytes))
    for val in [vendor_id & 0xFFFF, (vendor_id >> 16) & 0xFFFF,
                product_code & 0xFFFF, (product_code >> 16) & 0xFFFF,
                revision & 0xFFFF, (revision >> 16) & 0xFFFF,
                serial_number & 0xFFFF, (serial_number >> 16) & 0xFFFF]:
        image.append(val)
    for _ in range(4):
        image.append(0x0000)
    mbx = mailbox and not compact
    for start, sz in [(0x1000, 128), (0x1080, 128)]:
        image.append(start if mbx else 0)
        image.append(sz if mbx else 0)
    for start, sz in [(0x1000, 128), (0x1080, 128)]:
        image.append(start if mbx else 0)
        image.append(sz if mbx else 0)
    image.append(0x0004 if mbx else 0)
    image.append(0x0000)
    image.append(0x0000)
    image.append(0x0001)

    name_bytes = device_name.encode("ascii", errors="replace")[:63]
    string_data = bytes([1, len(name_bytes)]) + name_bytes
    if len(string_data) % 2:
        string_data += b'\x00'
    image.append(0x000A)
    image.append(len(string_data) // 2)
    for i in range(0, len(string_data), 2):
        image.append(struct.unpack_from("<H", string_data, i)[0])

    if compact:
        image.append(0x001E); image.append(0x0004)
        image.extend([0x0001, 0x0001, 0x0001, 0x0001])
        image.append(0x0028); image.append(0x0001); image.append(0x0201)
        image.append(0x0029); image.append(0x0008)
        image.extend([0x1100, 17, 0x0024, 0x0301, 0x1180, 20, 0x0020, 0x0401])
    else:
        image.append(0x001E); image.append(0x0010)
        image.extend([0x0001]*4 + [0x0000, 0x0023] + [0x0000]*10)
        image.append(0x0028); image.append(0x0001); image.append(0x0201)
        image.append(0x0029)
        sm_data = []
        if mailbox:
            sm_data += [0x1000, 128, 0x0026, 0x0101, 0x1080, 128, 0x0022, 0x0201]
        else:
            sm_data += [0]*8
        sm_data += [0x1100, 17, 0x0024, 0x0301, 0x1180, 20, 0x0020, 0x0401]
        image.append(len(sm_data))
        image.extend(sm_data)

    image.append(0x7FFF)
    image.append(0x0000)
    return image


# ═══════════════════════════════════════════════════════════════
# EtherCAT Monitor Backend (pysoem wrapper)
# ═══════════════════════════════════════════════════════════════

class EtherCATMonitor:
    KNOWN_ESC_TYPES = {0xC0: "LAN9252/9253", 0x95: "LAN9252 (legacy)"}
    SM_CONFIG = {
        0: {"psa": 0x1000, "len": 128, "ctrl": 0x26, "act": 0x01},
        1: {"psa": 0x1080, "len": 128, "ctrl": 0x22, "act": 0x01},
        2: {"psa": 0x1100, "len": 17,  "ctrl": 0x24, "act": 0x01},
        3: {"psa": 0x1180, "len": 20,  "ctrl": 0x20, "act": 0x01},
    }
    SM2_ADDR = 0x1100; SM2_SIZE = 17; SM3_ADDR = 0x1180; SM3_SIZE = 20

    def __init__(self):
        self.master = None
        self.connected = False
        self.adapter_name = None

    @staticmethod
    def list_adapters():
        return pysoem.find_adapters()

    def connect(self, adapter_name):
        self.master = pysoem.Master()
        self.master.open(adapter_name)
        self.adapter_name = adapter_name
        try:
            count = self.master.config_init()
        except Exception as e:
            self.master.close(); self.master = None
            raise RuntimeError(f"Scan failed: {e}")
        if count == 0:
            self.master.close(); self.master = None
            raise RuntimeError("No slaves found.")
        self.master.read_state()
        self.connected = True
        for i in range(count):
            data = self.read_register(i, 0x0000, 1)
            if data and struct.unpack_from("<B", data)[0] in self.KNOWN_ESC_TYPES:
                return count
        self.master.close(); self.master = None; self.connected = False
        raise RuntimeError("No LAN9252 found.")

    def disconnect(self):
        if self.master:
            try: self.master.close()
            except: pass
        self.master = None; self.connected = False

    def get_slave_count(self):
        return len(self.master.slaves) if self.connected else 0

    def get_slave_info(self, idx=0):
        if not self.connected or idx >= len(self.master.slaves): return None
        s = self.master.slaves[idx]
        info = {"name": s.name, "man": f"0x{s.man:08X}", "id": f"0x{s.id:08X}",
                "rev": f"0x{s.rev:08X}"}
        # Read CoE strings (requires PRE-OP or higher for mailbox)
        for key, obj_idx in [("device_name", 0x1008), ("hw_version", 0x1009),
                              ("sw_version", 0x100A), ("group", None)]:
            if obj_idx is None:
                # Group: try 0x1001 or leave from SII name
                info[key] = ""
                continue
            try:
                data = s.sdo_read(obj_idx, 0)
                info[key] = data.decode("utf-8", errors="ignore").rstrip("\x00")
            except Exception:
                info[key] = ""
        return info

    def read_register(self, slave_idx, reg_addr, size):
        if not self.connected: return None
        try: return self.master.slaves[slave_idx]._fprd(reg_addr, size)
        except: return None

    def get_al_state(self, slave_idx=0):
        data = self.read_register(slave_idx, 0x0130, 2)
        if data:
            val = struct.unpack_from("<H", data)[0]
            return val & 0x0F, bool(val & 0x10)
        return None, None

    def get_al_status_code(self, slave_idx=0):
        data = self.read_register(slave_idx, 0x0134, 2)
        return struct.unpack_from("<H", data)[0] if data else None

    def get_dl_status(self, slave_idx=0):
        data = self.read_register(slave_idx, 0x0110, 2)
        return struct.unpack_from("<H", data)[0] if data else None

    def request_state_bwr(self, target_state, log_fn=None):
        if not self.connected: return False, None, None
        def log(m):
            if log_fn: log_fn(m)
        slave = self.master.slaves[0]
        def write_reg(addr, data):
            try: slave._fpwr(addr, data); return True
            except: pass
            try:
                if hasattr(slave, '_apwr'): slave._apwr(addr, data); return True
            except: pass
            return False
        def read_u16(addr):
            try: return struct.unpack_from("<H", slave._fprd(addr, 2))[0]
            except: return None
        if target_state >= 0x02:
            for i in [0, 1]:
                c = self.SM_CONFIG[i]
                d = struct.pack("<HHBBBx", c["psa"], c["len"], c["ctrl"], 0x00, c["act"])
                ok = write_reg(0x0800 + i*8, d); log(f"  SM{i}: {'OK' if ok else 'FAIL'}")
        if target_state >= 0x04:
            for i in [2, 3]:
                c = self.SM_CONFIG[i]
                d = struct.pack("<HHBBBx", c["psa"], c["len"], c["ctrl"], 0x00, c["act"])
                ok = write_reg(0x0800 + i*8, d); log(f"  SM{i}: {'OK' if ok else 'FAIL'}")
        ok = write_reg(0x0120, struct.pack("<H", target_state))
        log(f"  AL Control=0x{target_state:04X}: {'OK' if ok else 'FAIL'}")
        final = None
        for i in range(30):
            time.sleep(0.1)
            stat = read_u16(0x0130)
            if stat is not None:
                if (stat & 0x0F) >= (target_state & 0x0F):
                    final = stat; log(f"  OK ({(i+1)*100}ms) AL=0x{stat:04X}"); break
                if stat & 0x10: final = stat; break
        if final is None: final = read_u16(0x0130)
        al_code = read_u16(0x0134)
        return (final is not None and (final & 0x0F) >= (target_state & 0x0F)), final, al_code

    def write_rxpdo(self, mode, pos=0, vel=0, torque=0, kp=0, kd=0, slave_idx=0):
        if not self.connected: return False
        data = struct.pack("<BiiiHH", mode & 0xFF, int(pos), int(vel), int(torque),
                           int(kp) & 0xFFFF, int(kd) & 0xFFFF)
        try: self.master.slaves[slave_idx]._fpwr(self.SM2_ADDR, data); return True
        except: return False

    def read_txpdo(self, slave_idx=0):
        if not self.connected: return None
        try:
            data = self.master.slaves[slave_idx]._fprd(self.SM3_ADDR, self.SM3_SIZE)
            if not data or len(data) < self.SM3_SIZE: return None
            v = struct.unpack_from("<BbiiiHbbH", data)
            return {"control_mode": v[0], "fault_code": v[1], "actual_position": v[2],
                    "actual_velocity": v[3], "actual_torque": v[4], "bus_voltage": v[5],
                    "temp_motor": v[6], "temp_mosfet": v[7], "encoder_pos": v[8]}
        except: return None

    def read_all_registers(self, slave_idx=0):
        results = {}
        for addr, (name, size, desc) in LAN9252_REGISTERS.items():
            data = self.read_register(slave_idx, addr, size)
            if data:
                fmt = {1: "<B", 2: "<H", 4: "<I"}.get(size)
                val = struct.unpack_from(fmt, data)[0] if fmt else data.hex()
            else:
                val = None
            results[addr] = (name, val, size, desc)
        return results

    def eeprom_read_word(self, slave_idx, word_addr):
        if not self.connected: return None
        try: return struct.unpack_from("<H", self.master.slaves[slave_idx].eeprom_read(word_addr))[0]
        except: return None

    def eeprom_write_word(self, slave_idx, word_addr, value):
        if not self.connected: return False
        self.master.slaves[slave_idx].eeprom_write(word_addr, struct.pack("<H", value & 0xFFFF))
        time.sleep(0.01); return True

    def eeprom_read_range(self, slave_idx, start, count):
        return [self.eeprom_read_word(slave_idx, start + i) or 0xFFFF for i in range(count)]

    def diagnose_connection(self, slave_idx=0):
        if not self.connected: return ["Not connected"]
        lines = []; s = self.master.slaves[slave_idx]
        lines.append(f"Slave: '{s.name}' man=0x{s.man:08X} id=0x{s.id:08X}")
        d = self.read_register(slave_idx, 0x0010, 2)
        if d: lines.append(f"Station Addr: 0x{struct.unpack_from('<H', d)[0]:04X}")
        lines.append("--- Communication Test (reference only) ---")
        try: s._fprd(0x0000, 1); lines.append("FPRD (read) test: OK")
        except Exception as e: lines.append(f"FPRD (read) test: FAIL - {e}")
        try: s._fpwr(0x0012, struct.pack("<H", 0)); lines.append("FPWR (write) test: OK")
        except Exception as e:
            lines.append(f"FPWR (write) test: FAIL (Station Alias reg, may be read-only)")
        lines.append("Note: FPWR FAIL on 0x0012 is normal for some ESCs.")
        lines.append("      State transitions use a separate path (BWR) and are not affected.")
        return lines


# ═══════════════════════════════════════════════════════════════
# Main EtherCAT Tab Widget
# ═══════════════════════════════════════════════════════════════

class EcatTab(QWidget):
    def __init__(self):
        super().__init__()
        self.monitor = EtherCATMonitor()
        self._adapters = []
        self._pdo_applied = {"mode": 0x80, "target_position": 0, "target_velocity": 0,
                             "target_torque": 0, "kp": 0, "kd": 0}
        self._pdo_cycling = False
        self._pdo_timer = QTimer()
        self._pdo_timer.timeout.connect(self._pdo_cycle_tick)
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_cycle)
        self._sii_image = None
        self._pdo_field_widgets = []
        self._pdo_tick_count = 0
        self._graph_max = 200  # ~10 seconds at 50ms
        self._graph_data = {
            "position": [], "velocity": [], "torque": [],
            "cmd_position": [], "cmd_velocity": [], "cmd_torque": [],
        }

        self._build_ui()
        self._refresh_adapters()
        # Auto-connect if restarted with --ecat-adapter (after UAC elevation)
        if self._get_ecat_adapter_arg():
            QTimer.singleShot(1000, self._auto_connect_if_requested)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Connection bar
        conn = QHBoxLayout()
        layout.addLayout(conn)
        conn.addWidget(QLabel("Adapter:"))
        self.adapter_combo = QComboBox(); self.adapter_combo.setMinimumWidth(250)
        conn.addWidget(self.adapter_combo)
        b = QPushButton("Refresh"); b.clicked.connect(self._refresh_adapters); conn.addWidget(b)
        self.connect_btn = QPushButton("Connect"); self.connect_btn.clicked.connect(self._toggle_connect)
        conn.addWidget(self.connect_btn)
        self.status_label = QLabel("Disconnected"); self.status_label.setStyleSheet("color: red;")
        conn.addWidget(self.status_label)
        conn.addStretch()
        conn.addWidget(QLabel("Poll(ms):"))
        self.poll_spin = QSpinBox(); self.poll_spin.setRange(200, 10000); self.poll_spin.setValue(1000)
        conn.addWidget(self.poll_spin)
        self.poll_btn = QPushButton("Auto Monitor"); self.poll_btn.clicked.connect(self._toggle_polling)
        conn.addWidget(self.poll_btn)

        # Sub-tabs
        self.sub_tabs = QTabWidget()
        layout.addWidget(self.sub_tabs)
        self._build_status_tab()
        self._build_pdo_tab()
        self._build_diag_tab()
        self._build_eeprom_tab()

    # ───────────────────────────────────────────────────────────
    # Sub-tab 1: Status & Control
    # ───────────────────────────────────────────────────────────

    def _build_status_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.sub_tabs.addTab(w, "Status && Control")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: slave info + ESC info
        left = QWidget(); ll = QVBoxLayout(left)
        info_grp = QGroupBox("Slave Info"); ig = QGridLayout(info_grp)
        self._info_labels = {}
        for i, (k, label) in enumerate([("count", "Slaves:"), ("name", "Name:"),
                ("device_name", "Device Name:"), ("hw_version", "HW Version:"),
                ("sw_version", "SW Version:"), ("group", "Group:"),
                ("man", "Manufacturer:"), ("id", "Product:"), ("rev", "Revision:")]):
            ig.addWidget(QLabel(label), i, 0)
            v = QLabel("-"); v.setFont(QFont("Consolas", 10)); ig.addWidget(v, i, 1)
            self._info_labels[k] = v
        ll.addWidget(info_grp)

        esc_grp = QGroupBox("ESC Info"); eg = QGridLayout(esc_grp)
        self._esc_labels = {}
        for i, (k, label) in enumerate([("type", "ESC Type:"), ("rev", "Revision:"),
                ("build", "Build:"), ("fmmu", "FMMUs:"), ("sm", "SyncManagers:"),
                ("ram", "RAM:"), ("feat", "Features:")]):
            eg.addWidget(QLabel(label), i, 0)
            v = QLabel("-"); v.setFont(QFont("Consolas", 10)); eg.addWidget(v, i, 1)
            self._esc_labels[k] = v
        ll.addWidget(esc_grp)
        ll.addStretch()
        splitter.addWidget(left)

        # Right: AL state + control + log
        right = QWidget(); rl = QVBoxLayout(right)
        al_grp = QGroupBox("AL State")
        al_l = QVBoxLayout(al_grp)
        self.al_state_label = QLabel("-"); self.al_state_label.setFont(QFont("Consolas", 24, QFont.Weight.Bold))
        self.al_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        al_l.addWidget(self.al_state_label)
        self.al_err_label = QLabel(""); self.al_err_label.setStyleSheet("color: red;")
        self.al_err_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        al_l.addWidget(self.al_err_label)
        self.al_code_label = QLabel(""); self.al_code_label.setFont(QFont("Consolas", 9))
        self.al_code_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        al_l.addWidget(self.al_code_label)
        rl.addWidget(al_grp)

        btn_row = QHBoxLayout()
        for text, state in [("INIT", 0x01), ("PRE-OP", 0x02), ("SAFE-OP", 0x04), ("OP", 0x08)]:
            b = QPushButton(text); b.clicked.connect(lambda _, s=state: self._request_state(s))
            btn_row.addWidget(b)
        diag_btn = QPushButton("Diagnose"); diag_btn.clicked.connect(self._run_diagnose)
        btn_row.addWidget(diag_btn)
        rl.addLayout(btn_row)

        self.state_log = QTextEdit(); self.state_log.setReadOnly(True)
        self.state_log.setFont(QFont("Consolas", 9))
        rl.addWidget(self.state_log)
        splitter.addWidget(right)
        splitter.setSizes([350, 550])

    # ───────────────────────────────────────────────────────────
    # Sub-tab 2: PDO Test
    # ───────────────────────────────────────────────────────────

    def _build_pdo_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.sub_tabs.addTab(w, "PDO Test")

        # Top row: state control + mode selector (horizontal)
        top_row = QHBoxLayout()
        b = QPushButton("Go OP + Start Cyclic"); b.clicked.connect(self._pdo_go_op); top_row.addWidget(b)
        b = QPushButton("Stop + INIT"); b.clicked.connect(self._pdo_stop_all); top_row.addWidget(b)
        self.pdo_state_label = QLabel("State: ?"); self.pdo_state_label.setStyleSheet("color: gray;")
        top_row.addWidget(self.pdo_state_label)
        top_row.addSpacing(16)
        top_row.addWidget(QLabel("Mode:"))
        self.pdo_mode_combo = QComboBox()
        for code, name in MODE_NAMES.items():
            self.pdo_mode_combo.addItem(f"0x{code:02X} {name}", code)
        self.pdo_mode_combo.currentIndexChanged.connect(self._pdo_mode_changed)
        top_row.addWidget(self.pdo_mode_combo)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Horizontal splitter: left (RxPDO + TxPDO) | right (Graph)
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(h_splitter)

        # Left: RxPDO controls (top) + TxPDO text (bottom), vertically split
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # RxPDO controls
        rx_grp = QGroupBox("RxPDO - Motor Control (17B)")
        rx_l = QVBoxLayout(rx_grp)
        self.pdo_fields_layout = QHBoxLayout()
        rx_l.addLayout(self.pdo_fields_layout)
        self._pdo_rebuild_fields(0x80)
        btn_row = QHBoxLayout()
        self.pdo_apply_btn = QPushButton("Apply"); self.pdo_apply_btn.clicked.connect(self._pdo_apply_values)
        btn_row.addWidget(self.pdo_apply_btn)
        b = QPushButton("Send Once"); b.clicked.connect(self._pdo_send_once); btn_row.addWidget(b)
        self.pdo_cycle_btn = QPushButton("Start Cyclic (50ms)"); self.pdo_cycle_btn.clicked.connect(self._pdo_toggle_cyclic)
        btn_row.addWidget(self.pdo_cycle_btn)
        b = QPushButton("MOTOR_OFF"); b.setStyleSheet("background-color: #c0392b; color: white;")
        b.clicked.connect(self._pdo_emergency_off); btn_row.addWidget(b)
        btn_row.addStretch()
        rx_l.addLayout(btn_row)
        left_splitter.addWidget(rx_grp)

        # TxPDO text (below RxPDO)
        tx_grp = QGroupBox("TxPDO - Motor Status (20B)")
        tx_l = QVBoxLayout(tx_grp)
        self.pdo_status_text = QTextEdit(); self.pdo_status_text.setReadOnly(True)
        self.pdo_status_text.setFont(QFont("Consolas", 10))
        tx_l.addWidget(self.pdo_status_text)
        b_row = QHBoxLayout()
        b = QPushButton("Read TxPDO"); b.clicked.connect(self._pdo_read_status); b_row.addWidget(b)
        b_row.addStretch()
        tx_l.addLayout(b_row)
        left_splitter.addWidget(tx_grp)
        left_splitter.setSizes([120, 300])
        h_splitter.addWidget(left_splitter)

        # Graph (right, full height)
        graph_w = QWidget()
        graph_l = QVBoxLayout(graph_w); graph_l.setContentsMargins(0, 0, 0, 0)
        g_toolbar = QHBoxLayout()
        g_toolbar.addWidget(QLabel("PDO Graph"))
        self.graph_clear_btn = QPushButton("Clear"); self.graph_clear_btn.clicked.connect(self._graph_clear)
        g_toolbar.addWidget(self.graph_clear_btn)
        g_toolbar.addStretch()
        graph_l.addLayout(g_toolbar)

        self._graph_plots = {}
        self._graph_curves = {}
        for key, title, unit, color in [
            ("pos", "Position", "deg", "#2ecc71"),
            ("vel", "Velocity", "dps", "#3498db"),
            ("torq", "Torque", "A", "#e74c3c"),
        ]:
            p = pg.PlotWidget(title=f"{title} ({unit})")
            p.setLabel("left", unit)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.addLegend(offset=(60, 5))
            p.showGrid(x=True, y=True, alpha=0.3)
            cmd_key = f"cmd_{'position' if key == 'pos' else 'velocity' if key == 'vel' else 'torque'}"
            act_key = "position" if key == "pos" else "velocity" if key == "vel" else "torque"
            self._graph_curves[cmd_key] = p.plot(pen=pg.mkPen("gray", width=1, style=Qt.PenStyle.DashLine), name="cmd")
            self._graph_curves[act_key] = p.plot(pen=pg.mkPen(color, width=2), name="actual")
            self._graph_plots[key] = p
            graph_l.addWidget(p)
        self._graph_plots["vel"].setXLink(self._graph_plots["pos"])
        self._graph_plots["torq"].setXLink(self._graph_plots["pos"])
        h_splitter.addWidget(graph_w)
        h_splitter.setSizes([350, 550])

    # ───────────────────────────────────────────────────────────
    # Sub-tab 3: Diagnostics (DL + Error + Register)
    # ───────────────────────────────────────────────────────────

    def _build_diag_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.sub_tabs.addTab(w, "Diagnostics")

        diag_tabs = QTabWidget()
        layout.addWidget(diag_tabs)

        # DL Status
        dl_w = QWidget(); dl_l = QVBoxLayout(dl_w)
        diag_tabs.addTab(dl_w, "DL Status")
        self.dl_raw_label = QLabel("-"); self.dl_raw_label.setFont(QFont("Consolas", 11))
        dl_l.addWidget(self.dl_raw_label)
        self.dl_text = QTextEdit(); self.dl_text.setReadOnly(True); self.dl_text.setFont(QFont("Consolas", 9))
        self.dl_text.setMaximumHeight(300)
        dl_l.addWidget(self.dl_text)
        dl_l.addStretch()

        # Error Counters
        err_w = QWidget(); err_l = QVBoxLayout(err_w)
        diag_tabs.addTab(err_w, "Error Counters")
        btn_row = QHBoxLayout()
        b = QPushButton("Read Errors"); b.clicked.connect(self._read_error_counters); btn_row.addWidget(b)
        b = QPushButton("Reset Errors"); b.clicked.connect(self._reset_error_counters); btn_row.addWidget(b)
        btn_row.addStretch()
        err_l.addLayout(btn_row)
        self._error_regs = [
            (0x0300, "RX Error Port 0", 4), (0x0304, "RX Error Port 0 (Fwd)", 4),
            (0x0308, "RX Error Port 1", 4), (0x030C, "Forwarded RX Error", 4),
            (0x0310, "ECAT Processing Unit Error", 1), (0x0311, "PDI Error", 1),
            (0x0442, "WD Counter PD", 1), (0x0443, "WD Counter PDI", 1),
        ]
        g = QGridLayout()
        self._error_labels = {}
        for i, (addr, name, _) in enumerate(self._error_regs):
            g.addWidget(QLabel(f"0x{addr:04X} - {name}:"), i, 0)
            v = QLabel("-"); v.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            g.addWidget(v, i, 1); self._error_labels[addr] = v
        err_l.addLayout(g)
        self.error_status_label = QLabel("")
        err_l.addWidget(self.error_status_label)
        err_l.addStretch()

        # Register Viewer
        reg_w = QWidget(); reg_l = QVBoxLayout(reg_w)
        diag_tabs.addTab(reg_w, "Registers")
        tb = QHBoxLayout()
        b = QPushButton("Read All"); b.clicked.connect(self._read_all_regs); tb.addWidget(b)
        tb.addWidget(QLabel("  Addr(hex):"))
        self.custom_addr_edit = QLineEdit("0130"); self.custom_addr_edit.setMaximumWidth(80)
        tb.addWidget(self.custom_addr_edit)
        tb.addWidget(QLabel("Size:"))
        self.custom_size_combo = QComboBox(); self.custom_size_combo.addItems(["1", "2", "4"])
        self.custom_size_combo.setCurrentText("2"); tb.addWidget(self.custom_size_combo)
        b = QPushButton("Read"); b.clicked.connect(self._read_custom_reg); tb.addWidget(b)
        self.custom_result_label = QLabel(""); self.custom_result_label.setFont(QFont("Consolas", 10))
        tb.addWidget(self.custom_result_label)
        tb.addStretch()
        reg_l.addLayout(tb)

        self.reg_tree = QTreeWidget()
        self.reg_tree.setHeaderLabels(["Address", "Name", "HEX", "DEC", "Description"])
        self.reg_tree.setFont(QFont("Consolas", 9))
        h = self.reg_tree.header()
        h.resizeSection(0, 80); h.resizeSection(1, 200); h.resizeSection(2, 120)
        h.resizeSection(3, 80); h.resizeSection(4, 200)
        reg_l.addWidget(self.reg_tree)

    # ───────────────────────────────────────────────────────────
    # Sub-tab 4: EEPROM
    # ───────────────────────────────────────────────────────────

    def _build_eeprom_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        self.sub_tabs.addTab(w, "EEPROM")

        cfg_grp = QGroupBox("SII Configuration")
        g = QGridLayout(cfg_grp)
        row = 0
        g.addWidget(QLabel("PDI Type:"), row, 0)
        self.pdi_combo = QComboBox(); self.pdi_combo.addItems(list(PDI_TYPES.keys()))
        self.pdi_combo.setCurrentText("SPI Host (0x80)"); g.addWidget(self.pdi_combo, row, 1)
        for label, default, attr in [("Vendor ID (hex):", "00524F42", "vendor_edit"),
                                      ("Product Code (hex):", "00000001", "product_edit"),
                                      ("Revision (hex):", "00000001", "revision_edit"),
                                      ("Serial Number (hex):", "00000001", "serial_edit")]:
            row += 1; g.addWidget(QLabel(label), row, 0)
            e = QLineEdit(default); g.addWidget(e, row, 1); setattr(self, attr, e)
        row += 1; g.addWidget(QLabel("Device Name:"), row, 0)
        self.devname_edit = QLineEdit("OpenRobot MC"); g.addWidget(self.devname_edit, row, 1, 1, 2)
        row += 1
        self.mailbox_check = QCheckBox("CoE Mailbox"); self.mailbox_check.setChecked(True)
        g.addWidget(self.mailbox_check, row, 0)
        self.compact_check = QCheckBox("Compact (64 words)"); self.compact_check.setChecked(True)
        g.addWidget(self.compact_check, row, 1)
        layout.addWidget(cfg_grp)

        btn_row = QHBoxLayout()
        for text, fn in [("Preview", self._preview_sii), ("Read EEPROM", self._read_eeprom),
                          ("Write EEPROM", self._write_eeprom), ("Verify", self._verify_eeprom),
                          ("Load JSON", self._load_config), ("Save JSON", self._save_config)]:
            b = QPushButton(text); b.clicked.connect(fn); btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.eeprom_progress = QProgressBar(); self.eeprom_progress.setRange(0, 100)
        layout.addWidget(self.eeprom_progress)
        self.eeprom_status_label = QLabel(""); self.eeprom_status_label.setFont(QFont("Consolas", 9))
        layout.addWidget(self.eeprom_status_label)
        self.eeprom_text = QTextEdit(); self.eeprom_text.setReadOnly(True)
        self.eeprom_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.eeprom_text)

    # ═══════════════════════════════════════════════════════════
    # Connection
    # ═══════════════════════════════════════════════════════════

    def _refresh_adapters(self):
        try:
            self._adapters = EtherCATMonitor.list_adapters()
            self.adapter_combo.clear()
            for a in self._adapters:
                self.adapter_combo.addItem(f"{a.desc} [{a.name}]")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Adapter scan failed:\n{e}\nIs Npcap installed?")

    def _toggle_connect(self):
        if self.monitor.connected:
            self._disconnect()
        else:
            self._do_connect()

    @staticmethod
    def _is_admin():
        if sys.platform != "win32":
            return os.getuid() == 0
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False

    @staticmethod
    def _restart_as_admin(adapter_name=""):
        """Re-launch the current process with UAC elevation (Windows)."""
        if sys.platform != "win32":
            return False
        try:
            if getattr(sys, 'frozen', False):
                exe = sys.executable
                args = [a for a in sys.argv[1:] if not a.startswith("--ecat-adapter=")]
            else:
                exe = sys.executable
                args = [a for a in sys.argv if not a.startswith("--ecat-adapter=")]
            if adapter_name:
                args.append(f"--ecat-adapter={adapter_name}")
            params = " ".join(f'"{a}"' for a in args)
            ret = ctypes.windll.shell32.ShellExecuteW(
                None, "runas", exe, params, None, 1  # SW_SHOWNORMAL
            )
            return ret > 32  # >32 means success
        except Exception:
            return False

    @staticmethod
    def _get_ecat_adapter_arg():
        """Return adapter name from --ecat-adapter=... CLI arg, or None."""
        for arg in sys.argv:
            if arg.startswith("--ecat-adapter="):
                return arg.split("=", 1)[1]
        return None

    def _auto_connect_if_requested(self):
        """Called after UI init: auto-connect if launched with --ecat-adapter."""
        adapter_desc = self._get_ecat_adapter_arg()
        if not adapter_desc:
            return
        # Match by desc (name has special chars that break shell args)
        for i, a in enumerate(self._adapters):
            d = a.desc.decode("utf-8", errors="ignore") if isinstance(a.desc, bytes) else str(a.desc)
            if d == adapter_desc:
                self.adapter_combo.setCurrentIndex(i)
                self._do_connect()
                return
        self.status_label.setText(f"Adapter not found: {adapter_desc}")
        self.status_label.setStyleSheet("color: orange;")

    def _do_connect(self):
        if not self._is_admin():
            reply = QMessageBox.question(
                self, "Admin Required",
                "EtherCAT (Npcap) requires Administrator privileges.\n\n"
                "Restart the application as Administrator?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Pass current adapter desc (not name — name has backslashes/braces that break shell)
                idx = self.adapter_combo.currentIndex()
                adapter_desc = ""
                if 0 <= idx < len(self._adapters):
                    d = self._adapters[idx].desc
                    adapter_desc = d.decode("utf-8", errors="ignore") if isinstance(d, bytes) else str(d)
                if self._restart_as_admin(adapter_desc):
                    QApplication.instance().quit()
                else:
                    QMessageBox.critical(self, "Error", "Failed to elevate. Please run as Administrator manually.")
            return
        idx = self.adapter_combo.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "Warning", "Select an adapter."); return
        try:
            count = self.monitor.connect(self._adapters[idx].name)
            self.status_label.setText(f"Connected ({count} slave)"); self.status_label.setStyleSheet("color: green;")
            self.connect_btn.setText("Disconnect")
            self._update_slave_info(); self._update_esc_info(); self._update_al_state()
            # Auto transition to SAFE-OP
            self._auto_go_safeop()
        except Exception as e:
            QMessageBox.critical(self, "Connect Failed", str(e))

    def _disconnect(self):
        self._poll_timer.stop(); self._pdo_timer.stop(); self._pdo_cycling = False
        self.pdo_cycle_btn.setText("Start Cyclic (50ms)")
        self.monitor.disconnect()
        self.status_label.setText("Disconnected"); self.status_label.setStyleSheet("color: red;")
        self.connect_btn.setText("Connect")

    # ═══════════════════════════════════════════════════════════
    # Status & Control
    # ═══════════════════════════════════════════════════════════

    def _update_slave_info(self):
        self._info_labels["count"].setText(str(self.monitor.get_slave_count()))
        info = self.monitor.get_slave_info(0)
        if info:
            for k in ("name", "man", "id", "rev", "device_name", "hw_version", "sw_version", "group"):
                val = info.get(k, "")
                self._info_labels[k].setText(val if val else "N/A")

    def _update_esc_info(self):
        esc_regs = {"type": (0x0000, 1), "rev": (0x0001, 1), "build": (0x0002, 2),
                    "fmmu": (0x0004, 1), "sm": (0x0005, 1), "ram": (0x0006, 1), "feat": (0x0008, 2)}
        for key, (addr, size) in esc_regs.items():
            data = self.monitor.read_register(0, addr, size)
            if data:
                fmt = {1: "<B", 2: "<H", 4: "<I"}[size]
                val = struct.unpack_from(fmt, data)[0]
                if key == "type":
                    name = EtherCATMonitor.KNOWN_ESC_TYPES.get(val, "")
                    self._esc_labels[key].setText(f"0x{val:02X} ({name})" if name else f"0x{val:02X}")
                elif key == "ram":
                    self._esc_labels[key].setText(f"{val} KB")
                else:
                    f = f"0x{val:02X}" if size == 1 else f"0x{val:04X}"
                    self._esc_labels[key].setText(f)
            else:
                self._esc_labels[key].setText("read fail")

    def _update_al_state(self):
        state, err = self.monitor.get_al_state(0)
        if state is not None:
            name = AL_STATE_MAP.get(state, f"0x{state:02X}")
            self.al_state_label.setText(name)
            color = "green" if state == 8 else ("orange" if state >= 4 else "gray")
            self.al_state_label.setStyleSheet(f"color: {color};")
            if err:
                self.al_err_label.setText("ERROR FLAG SET")
                code = self.monitor.get_al_status_code(0)
                if code is not None:
                    self.al_code_label.setText(f"0x{code:04X} - {AL_STATUS_CODES.get(code, 'Unknown')}")
            else:
                self.al_err_label.setText(""); self.al_code_label.setText("")

    def _auto_go_safeop(self):
        """Auto transition INIT -> PRE-OP -> SAFE-OP after connect."""
        for target, name in [(0x01, "INIT"), (0x02, "PRE-OP"), (0x04, "SAFE-OP")]:
            self._state_log(f"Auto: -> {name}")
            ok, final, al_code = self.monitor.request_state_bwr(target, log_fn=self._state_log)
            if not ok:
                self._state_log(f"  Auto {name} FAILED (0x{al_code or 0:04X})")
                break
            self._state_log(f"  -> {name} OK")
        self._update_al_state()
        self._update_slave_info()

    def _request_state(self, state):
        if not self.monitor.connected:
            QMessageBox.warning(self, "Warning", "Connect first."); return
        target_name = AL_STATE_MAP.get(state, f"0x{state:02X}")
        self._state_log(f"Request: -> {target_name}")
        ok, final, al_code = self.monitor.request_state_bwr(state, log_fn=self._state_log)
        if ok:
            self._state_log(f"  Result: OK")
        else:
            self._state_log(f"  Result: FAILED (AL code: 0x{al_code or 0:04X})")
        self._update_al_state()
        self._update_slave_info()

    def _run_diagnose(self):
        if not self.monitor.connected:
            QMessageBox.warning(self, "Warning", "Connect first."); return
        self._state_log("== Diagnose ==")
        for line in self.monitor.diagnose_connection(0):
            self._state_log(f"  {line}")
        self._update_al_state()

    def _state_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.state_log.moveCursor(QTextCursor.MoveOperation.End)
        self.state_log.insertPlainText(f"[{ts}] {msg}\n")
        self.state_log.ensureCursorVisible()

    # ═══════════════════════════════════════════════════════════
    # PDO Test
    # ═══════════════════════════════════════════════════════════

    def _pdo_go_op(self):
        if not self.monitor.connected:
            self._pdo_log("ERROR: Connect first"); return
        self._pdo_log("-- Go OP --")
        for target, name in [(0x01, "INIT"), (0x02, "PRE-OP"), (0x04, "SAFE-OP"), (0x08, "OP")]:
            ok, final, code = self.monitor.request_state_bwr(target, log_fn=lambda m: self._pdo_log(f"  {m}"))
            if not ok:
                self._pdo_log(f"ERROR: {name} failed (0x{code or 0:04X})")
                self._pdo_update_state(); return
            self._pdo_log(f"  -> {name} OK")
        self._pdo_update_state()
        self._update_slave_info()
        if not self._pdo_cycling:
            self._pdo_toggle_cyclic()
        self._pdo_log("-- OP + Cyclic started --")

    def _pdo_stop_all(self):
        if self._pdo_cycling: self._pdo_toggle_cyclic()
        self.monitor.write_rxpdo(0x80)
        self._pdo_log("MOTOR_OFF sent")
        self.monitor.request_state_bwr(0x01)
        self._pdo_log("-> INIT"); self._pdo_update_state()

    def _pdo_update_state(self):
        state, err = self.monitor.get_al_state(0)
        if state is not None:
            name = AL_STATE_MAP.get(state, f"0x{state:02X}")
            color = "green" if state == 8 else ("orange" if state >= 4 else "gray")
            err_txt = " (ERROR)" if err else ""
            self.pdo_state_label.setText(f"State: {name}{err_txt}")
            self.pdo_state_label.setStyleSheet(f"color: {color};")

    def _pdo_mode_changed(self):
        mode = self.pdo_mode_combo.currentData()
        self._pdo_rebuild_fields(mode)
        self._pdo_applied = {"mode": mode, "target_position": 0, "target_velocity": 0,
                             "target_torque": 0, "kp": 0, "kd": 0}
        self._pdo_log(f"Mode -> 0x{mode:02X} {MODE_NAMES.get(mode, '?')}")

    def _pdo_rebuild_fields(self, mode):
        for w in self._pdo_field_widgets:
            w.setParent(None)
        self._pdo_field_widgets = []
        self._pdo_field_entries = {}
        defs = MODE_FIELD_DEFS.get(mode, [])
        if not defs:
            lbl = QLabel("(no parameters for this mode)"); lbl.setStyleSheet("color: gray;")
            self.pdo_fields_layout.addWidget(lbl); self._pdo_field_widgets.append(lbl)
            return
        for label, pdo_field, scale, unit in defs:
            frame = QWidget(); fl = QVBoxLayout(frame); fl.setContentsMargins(0, 0, 0, 0)
            fl.addWidget(QLabel(label))
            input_row = QHBoxLayout(); input_row.setContentsMargins(0, 0, 0, 0)
            e = QLineEdit("0"); e.setMaximumWidth(120); input_row.addWidget(e)
            u = QLabel(unit); u.setStyleSheet("color: gray;"); input_row.addWidget(u)
            input_row.addStretch()
            fl.addLayout(input_row)
            self.pdo_fields_layout.addWidget(frame)
            self._pdo_field_widgets.append(frame)
            self._pdo_field_entries[label] = (e, pdo_field, scale)

    def _pdo_apply_values(self):
        mode = self.pdo_mode_combo.currentData()
        pdo = {"mode": mode, "target_position": 0, "target_velocity": 0,
               "target_torque": 0, "kp": 0, "kd": 0}
        parts = [f"mode=0x{mode:02X}"]
        for label, (edit, field, scale) in self._pdo_field_entries.items():
            try: val = float(edit.text())
            except ValueError:
                self._pdo_log(f"ERROR: invalid value for '{label}'"); return
            pdo[field] = int(val * scale); parts.append(f"{label}={val}")
        self._pdo_applied = pdo
        self._pdo_log(f"Applied: {', '.join(parts)}")

    def _pdo_send_once(self):
        self._pdo_apply_values()
        a = self._pdo_applied
        ok = self.monitor.write_rxpdo(a["mode"], a["target_position"], a["target_velocity"],
                                      a["target_torque"], a["kp"], a["kd"])
        self._pdo_log(f"Send Once -> {'OK' if ok else 'FAIL'}")
        status = self.monitor.read_txpdo()
        if status: self._pdo_display_status(status)

    def _pdo_emergency_off(self):
        self._pdo_applied = {"mode": 0x80, "target_position": 0, "target_velocity": 0,
                             "target_torque": 0, "kp": 0, "kd": 0}
        self.pdo_mode_combo.setCurrentIndex(0)
        ok = self.monitor.write_rxpdo(0x80)
        self._pdo_log(f"Emergency MOTOR_OFF -> {'OK' if ok else 'FAIL'}")
        status = self.monitor.read_txpdo()
        if status: self._pdo_display_status(status)

    def _pdo_toggle_cyclic(self):
        if self._pdo_cycling:
            self._pdo_cycling = False; self._pdo_timer.stop()
            self.pdo_cycle_btn.setText("Start Cyclic (50ms)")
        else:
            self._pdo_cycling = True; self._pdo_timer.start(50)
            self.pdo_cycle_btn.setText("Stop Cyclic")

    def _pdo_cycle_tick(self):
        a = self._pdo_applied
        self.monitor.write_rxpdo(a["mode"], a["target_position"], a["target_velocity"],
                                 a["target_torque"], a["kp"], a["kd"])
        status = self.monitor.read_txpdo()
        if status: self._pdo_display_status(status)
        # Check AL state every ~1s (20 ticks × 50ms) to detect state drops
        self._pdo_tick_count += 1
        if self._pdo_tick_count >= 20:
            self._pdo_tick_count = 0
            self._pdo_update_state()
            self._update_al_state()

    def _pdo_read_status(self):
        status = self.monitor.read_txpdo()
        if status:
            self._pdo_display_status(status)
        else:
            self._pdo_log("TxPDO read failed")

    def _pdo_display_status(self, status):
        a = self._pdo_applied
        rx_name = MODE_NAMES.get(a["mode"], f"0x{a['mode']:02X}")
        # Scale to physical units (same values plotted on graph)
        pos  = status["actual_position"] / 10000.0
        vel  = status["actual_velocity"] / 10000.0
        torq = status["actual_torque"]   / 10000.0
        vbus = status["bus_voltage"]     / 100.0
        lines = [
            "-- RxPDO (Master -> Slave) --",
            f"  control_mode:    0x{a['mode']:02X} ({rx_name})",
            f"  target_position: {a['target_position']/10000:.4f} deg  (raw {a['target_position']})",
            f"  target_velocity: {a['target_velocity']/10000:.4f} dps  (raw {a['target_velocity']})",
            f"  target_torque:   {a['target_torque']/10000:.4f} A  (raw {a['target_torque']})",
            f"  kp:              {a['kp']/100:.2f}  (raw {a['kp']})",
            f"  kd:              {a['kd']/1000:.3f}  (raw {a['kd']})", "",
            "-- TxPDO (Slave -> Master) --",
            f"  control_mode:    0x{status['control_mode']:02X} ({MODE_NAMES.get(status['control_mode'], '?')})",
            f"  fault_code:      {status['fault_code']}",
            f"  actual_position: {pos:+.4f} deg  (raw {status['actual_position']})",
            f"  actual_velocity: {vel:+.4f} dps  (raw {status['actual_velocity']})",
            f"  actual_torque:   {torq:+.4f} A  (raw {status['actual_torque']})",
            f"  bus_voltage:     {vbus:.2f} V",
            f"  temp_motor:      {status['temp_motor']} C",
            f"  temp_mosfet:     {status['temp_mosfet']} C",
            f"  encoder_pos:     {status['encoder_pos']}",
        ]
        self.pdo_status_text.setPlainText("\n".join(lines))
        self._graph_push(a, status)

    def _graph_push(self, applied, status):
        """Append one sample to graph buffers and update curves."""
        gd = self._graph_data
        mx = self._graph_max
        gd["position"].append(status["actual_position"] / 10000.0)
        gd["velocity"].append(status["actual_velocity"] / 10000.0)
        gd["torque"].append(status["actual_torque"] / 10000.0)
        gd["cmd_position"].append(applied["target_position"] / 10000.0)
        gd["cmd_velocity"].append(applied["target_velocity"] / 10000.0)
        gd["cmd_torque"].append(applied["target_torque"] / 10000.0)
        for k in gd:
            if len(gd[k]) > mx:
                gd[k] = gd[k][-mx:]
        for k, curve in self._graph_curves.items():
            curve.setData(gd[k])

    def _graph_clear(self):
        for k in self._graph_data:
            self._graph_data[k] = []
        for curve in self._graph_curves.values():
            curve.setData([])

    def _pdo_log(self, msg):
        self.pdo_status_text.moveCursor(QTextCursor.MoveOperation.End)
        self.pdo_status_text.insertPlainText("\n" + msg)
        self.pdo_status_text.ensureCursorVisible()

    # ═══════════════════════════════════════════════════════════
    # Diagnostics
    # ═══════════════════════════════════════════════════════════

    def _update_dl_status(self):
        dl = self.monitor.get_dl_status(0)
        if dl is None: self.dl_raw_label.setText("read fail"); return
        self.dl_raw_label.setText(f"DL Status: 0x{dl:04X} ({dl:016b})")
        lines = []
        for bit, (name, on, off) in sorted(DL_STATUS_BITS.items()):
            val = (dl >> bit) & 1
            lines.append(f"  Bit {bit:2d} {name:30s} {'ON' if val else 'OFF':5s} ({on if val else off})")
        # Port summary
        lines.append("")
        for p in range(2):
            link = (dl >> (4+p)) & 1; loop = (dl >> (8+p)) & 1; sig = (dl >> (12+p)) & 1
            pname = f"Port {p} ({'ECAT IN' if p==0 else 'ECAT OUT'})"
            if link and sig: summary = "Connected"
            elif loop and not link: summary = "No cable (loopback)"
            else: summary = "No connection"
            lines.append(f"  {pname}: Link={'ON' if link else 'OFF'} Loop={'ON' if loop else 'OFF'} Signal={'ON' if sig else 'OFF'} -> {summary}")
        self.dl_text.setPlainText("\n".join(lines))

    def _read_error_counters(self):
        if not self.monitor.connected:
            QMessageBox.warning(self, "Warning", "Connect first."); return
        for addr, name, size in self._error_regs:
            data = self.monitor.read_register(0, addr, size)
            if data:
                fmt = {1: "<B", 2: "<H", 4: "<I"}[size]
                self._error_labels[addr].setText(str(struct.unpack_from(fmt, data)[0]))
            else:
                self._error_labels[addr].setText("fail")
        self.error_status_label.setText(f"[{time.strftime('%H:%M:%S')}] Read done")

    def _reset_error_counters(self):
        if not self.monitor.connected:
            QMessageBox.warning(self, "Warning", "Connect first."); return
        s = self.monitor.master.slaves[0]
        for addr, _, size in self._error_regs:
            try: s._fpwr(addr, b'\x00' * size)
            except: pass
        self._read_error_counters()
        self.error_status_label.setText(f"[{time.strftime('%H:%M:%S')}] Reset done")

    def _read_all_regs(self):
        if not self.monitor.connected:
            QMessageBox.warning(self, "Warning", "Connect first."); return
        self.reg_tree.clear()
        results = self.monitor.read_all_registers(0)
        for addr in sorted(results.keys()):
            name, val, size, desc = results[addr]
            if val is not None and isinstance(val, int):
                h = f"0x{val:02X}" if size == 1 else (f"0x{val:04X}" if size == 2 else f"0x{val:08X}")
                d = str(val)
            else:
                h = "ERR"; d = "-"
            QTreeWidgetItem(self.reg_tree, [f"0x{addr:04X}", name, h, d, desc])

    def _read_custom_reg(self):
        if not self.monitor.connected:
            QMessageBox.warning(self, "Warning", "Connect first."); return
        try:
            addr = int(self.custom_addr_edit.text(), 16)
            size = int(self.custom_size_combo.currentText())
        except ValueError:
            self.custom_result_label.setText("Invalid input"); return
        data = self.monitor.read_register(0, addr, size)
        if data:
            fmt = {1: "<B", 2: "<H", 4: "<I"}[size]
            val = struct.unpack_from(fmt, data)[0]
            if size <= 2:
                bits = f"{val:08b}" if size == 1 else f"{val:016b}"
                self.custom_result_label.setText(f"0x{val:0{size*2}X} ({val}) bin:{bits}")
            else:
                self.custom_result_label.setText(f"0x{val:08X} ({val})")
        else:
            self.custom_result_label.setText("Read failed")

    # ═══════════════════════════════════════════════════════════
    # EEPROM
    # ═══════════════════════════════════════════════════════════

    def _get_sii_params(self):
        try:
            return {
                "vendor_id": int(self.vendor_edit.text(), 16),
                "product_code": int(self.product_edit.text(), 16),
                "revision": int(self.revision_edit.text(), 16),
                "serial_number": int(self.serial_edit.text(), 16),
                "pdi_type": PDI_TYPES.get(self.pdi_combo.currentText(), 0x80),
                "device_name": self.devname_edit.text(),
                "mailbox": self.mailbox_check.isChecked(),
                "compact": self.compact_check.isChecked(),
            }
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid hex value."); return None

    def _format_hex_dump(self, words, label=""):
        lines = []
        if label: lines.append(f"-- {label} ({len(words)} words, {len(words)*2} bytes) --")
        for i in range(0, len(words), 8):
            addr = f"0x{i:04X}"
            vals = "  ".join(f"{w:04X}" if w is not None else "????" for w in words[i:i+8])
            lines.append(f"  {addr}: {vals}")
        return "\n".join(lines)

    def _preview_sii(self):
        params = self._get_sii_params()
        if not params: return
        self._sii_image = build_sii_image(**params)
        dump = self._format_hex_dump(self._sii_image, "Generated SII Image")
        info = [dump, "", "-- Field Interpretation --",
                f"  PDI:      0x{self._sii_image[0]:04X} ({self.pdi_combo.currentText()})",
                f"  CRC:      0x{self._sii_image[7]:04X}",
                f"  Vendor:   0x{params['vendor_id']:08X}",
                f"  Product:  0x{params['product_code']:08X}",
                f"  Name:     {params['device_name']}",
                f"  Size:     {len(self._sii_image)} words"]
        self.eeprom_text.setPlainText("\n".join(info))
        self.eeprom_status_label.setText(f"SII image generated: {len(self._sii_image)} words")

    def _read_eeprom(self):
        if not self.monitor.connected:
            QMessageBox.warning(self, "Warning", "Connect first."); return
        self.eeprom_status_label.setText("Reading...")
        words = []; max_w = 256
        for addr in range(max_w):
            val = self.monitor.eeprom_read_word(0, addr)
            words.append(val if val is not None else 0xFFFF)
            if addr >= 32 and val == 0x7FFF:
                nxt = self.monitor.eeprom_read_word(0, addr+1)
                words.append(nxt if nxt is not None else 0); break
            if addr % 16 == 0:
                self.eeprom_progress.setValue(int(addr/max_w*100))
        self.eeprom_progress.setValue(100)
        dump = self._format_hex_dump(words, f"EEPROM Content (0x0000~0x{len(words)-1:04X})")
        vendor = (words[9]<<16)|words[8]; product = (words[11]<<16)|words[10]
        info = [dump, "", "-- Interpretation --",
                f"  PDI:     0x{words[0]:04X}", f"  CRC:     0x{words[7]:04X}",
                f"  Vendor:  0x{vendor:08X}", f"  Product: 0x{product:08X}"]
        self.eeprom_text.setPlainText("\n".join(info))
        self.eeprom_status_label.setText(f"Read done: {len(words)} words")

    def _write_eeprom(self):
        if not self.monitor.connected:
            QMessageBox.warning(self, "Warning", "Connect first."); return
        if not self._sii_image:
            QMessageBox.warning(self, "Warning", "Generate SII image first (Preview)."); return
        if QMessageBox.question(self, "Confirm", f"Write {len(self._sii_image)} words to EEPROM?") != QMessageBox.StandardButton.Yes:
            return
        self.eeprom_status_label.setText("Writing...")
        try:
            for i, w in enumerate(self._sii_image):
                self.monitor.eeprom_write_word(0, i, w)
                if i % 2 == 0:
                    self.eeprom_progress.setValue(int((i+1)/len(self._sii_image)*100))
            self.eeprom_progress.setValue(100)
            self.eeprom_status_label.setText(f"Write done: {len(self._sii_image)} words")
        except Exception as e:
            QMessageBox.critical(self, "Write Failed", str(e))
            self.eeprom_status_label.setText(f"Write failed: {e}")

    def _verify_eeprom(self):
        if not self.monitor.connected:
            QMessageBox.warning(self, "Warning", "Connect first."); return
        if not self._sii_image:
            QMessageBox.warning(self, "Warning", "Generate SII image first (Preview)."); return
        read = self.monitor.eeprom_read_range(0, 0, len(self._sii_image))
        mismatches = [(i, exp, act) for i, (exp, act) in enumerate(zip(self._sii_image, read)) if exp != act]
        if not mismatches:
            self.eeprom_text.setPlainText(f"Verify OK! All {len(self._sii_image)} words match.")
            self.eeprom_status_label.setText("Verify: OK")
        else:
            lines = [f"Verify FAILED: {len(mismatches)} mismatches", "", "  Addr     Expected  Actual", "  " + "-"*35]
            for addr, exp, act in mismatches[:30]:
                lines.append(f"  0x{addr:04X}  0x{exp:04X}    0x{act:04X}")
            self.eeprom_text.setPlainText("\n".join(lines))
            self.eeprom_status_label.setText(f"Verify FAILED: {len(mismatches)} mismatches")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f: config = json.load(f)
            ident = config.get("identity", {})
            dev = config.get("device", {})
            self.vendor_edit.setText(ident.get("vendor_id", "0").replace("0x", ""))
            self.product_edit.setText(ident.get("product_code", "1").replace("0x", ""))
            self.revision_edit.setText(ident.get("revision", "1").replace("0x", ""))
            self.serial_edit.setText(ident.get("serial_number", "1").replace("0x", ""))
            self.devname_edit.setText(dev.get("name", ""))
            pdi = dev.get("pdi_type", "")
            idx = self.pdi_combo.findText(pdi)
            if idx >= 0: self.pdi_combo.setCurrentIndex(idx)
            self.mailbox_check.setChecked(config.get("mailbox", {}).get("enabled", True))
            self.compact_check.setChecked(config.get("eeprom", {}).get("compact", False))
            self.eeprom_status_label.setText(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {e}")

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "JSON (*.json)")
        if not path: return
        params = self._get_sii_params()
        if not params: return
        config = {
            "identity": {k: f"0x{params[k]:08X}" for k in ["vendor_id", "product_code", "revision", "serial_number"]},
            "device": {"name": params["device_name"], "pdi_type": self.pdi_combo.currentText()},
            "mailbox": {"enabled": params["mailbox"]},
            "process_data": {"sm2": {"start": "0x1100", "size": 17}, "sm3": {"start": "0x1180", "size": 20}},
            "eeprom": {"compact": params["compact"]},
        }
        try:
            with open(path, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4, ensure_ascii=False)
            self.eeprom_status_label.setText(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    # ═══════════════════════════════════════════════════════════
    # Auto Polling
    # ═══════════════════════════════════════════════════════════

    def _toggle_polling(self):
        if self._poll_timer.isActive():
            self._poll_timer.stop(); self.poll_btn.setText("Auto Monitor")
        else:
            if not self.monitor.connected:
                QMessageBox.warning(self, "Warning", "Connect first."); return
            self._poll_timer.start(self.poll_spin.value())
            self.poll_btn.setText("Stop Monitor")

    def _poll_cycle(self):
        if not self.monitor.connected:
            self._poll_timer.stop(); self.poll_btn.setText("Auto Monitor"); return
        try:
            self._update_al_state(); self._update_dl_status(); self._update_slave_info()
        except:
            self._poll_timer.stop(); self.poll_btn.setText("Auto Monitor")

    # ═══════════════════════════════════════════════════════════
    # Cleanup
    # ═══════════════════════════════════════════════════════════

    def cleanup(self):
        self._pdo_timer.stop(); self._poll_timer.stop()
        self.monitor.disconnect()
