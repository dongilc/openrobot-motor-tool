"""
Parameter tab — read/display/edit MCCONF & APPCONF, write to VESC, save/load XML.
"""

import os
import struct
from collections import OrderedDict
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QFileDialog, QMessageBox,
    QSpinBox, QMenu, QLineEdit, QComboBox, QCheckBox, QSizePolicy,
)
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt, pyqtSignal

from ..protocol.can_transport import PcanTransport
from ..protocol.commands import (
    CommPacketId,
    build_get_mcconf, build_get_mcconf_default,
    build_get_appconf, build_get_appconf_default,
    build_terminal_cmd,
)
from ..protocol.confparser import (
    MCCONF_FIELDS, APPCONF_FIELDS, MCCONF_SIGNATURE, APPCONF_SIGNATURE,
    FIELD_DESCRIPTIONS, parse_conf, serialize_conf, conf_to_xml, xml_to_conf,
)

# Color for edited (dirty) values
_DIRTY_COLOR = QColor("#FFDD57")  # yellow-ish


class ParameterTab(QWidget):
    log_message = pyqtSignal(str)  # emitted to show messages in Terminal dock
    rescan_requested = pyqtSignal()  # CAN ID changed — trigger re-scan

    def __init__(self, transport: PcanTransport):
        super().__init__()
        self._transport = transport
        self._mcconf_values: OrderedDict | None = None
        self._appconf_values: OrderedDict | None = None
        self._mcconf_original: OrderedDict | None = None   # snapshot at read time
        self._appconf_original: OrderedDict | None = None
        self._mcconf_defaults: OrderedDict | None = None  # factory defaults
        self._appconf_defaults: OrderedDict | None = None
        self._mc_dirty: set = set()   # field names that were edited
        self._app_dirty: set = set()
        self._populating = False      # guard against itemChanged during populate
        # Pending default load: (field_name, tree_item, is_mc) — auto-read defaults then apply
        self._pending_default: tuple | None = None

        layout = QVBoxLayout(self)

        sep_style = "color: #666; font-size: 14px;"

        # ── Row 1: Detect Motor ──────────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.setContentsMargins(0, 0, 0, 0)

        self.btn_detect = QPushButton("Detect Motor FOC Params")
        self.btn_detect.setStyleSheet(
            "QPushButton { background: #1B5E20; color: white; font-weight: bold; padding: 4px 12px; }"
            "QPushButton:hover { background: #2E7D32; }"
        )
        self.btn_detect.setToolTip("foc_detect_apply_all — Detect R, L, flux linkage, sensor and apply")
        self.btn_detect.clicked.connect(self._on_detect_motor)
        row1.addWidget(self.btn_detect)

        motor_watt_label = QLabel("Motor:")
        motor_watt_label.setStyleSheet("color: #ccc;")
        row1.addWidget(motor_watt_label)

        self.motor_watt_spin = QSpinBox()
        self.motor_watt_spin.setRange(50, 2000)
        self.motor_watt_spin.setValue(200)
        self.motor_watt_spin.setSuffix(" W")
        self.motor_watt_spin.setSingleStep(50)
        self.motor_watt_spin.setToolTip("Motor rated power (watts)")
        self.motor_watt_spin.setFixedWidth(120)
        self.motor_watt_spin.valueChanged.connect(self._update_power_loss_label)
        row1.addWidget(self.motor_watt_spin)

        arrow_label = QLabel("→")
        arrow_label.setStyleSheet("color: #888; font-size: 14px;")
        row1.addWidget(arrow_label)

        ploss_title = QLabel("max_power_loss:")
        ploss_title.setStyleSheet("color: #ccc;")
        row1.addWidget(ploss_title)

        self.power_loss_label = QLabel("")
        self.power_loss_label.setStyleSheet(
            "color: #FFD54F; font-weight: bold; padding: 2px 6px;"
        )
        self.power_loss_label.setToolTip("= Motor W × 25%  (I²R heat budget for detection)")
        row1.addWidget(self.power_loss_label)

        self._update_power_loss_label()

        sep_status = QLabel("|"); sep_status.setStyleSheet(sep_style)
        row1.addWidget(sep_status)

        self.status_label = QLabel("Last read: never")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt;")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status_label.setMinimumWidth(80)
        row1.addWidget(self.status_label, 1)  # stretch=1 so it fills available space

        search_label = QLabel("Search:")
        search_label.setStyleSheet("color: #ccc;")
        row1.addWidget(search_label)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("parameter name...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedWidth(200)
        self.search_edit.textChanged.connect(self._on_search)
        row1.addWidget(self.search_edit)

        self.search_keep_chk = QCheckBox("Keep")
        self.search_keep_chk.setToolTip("Keep search filter after Read MCCONF/APPCONF or Load XML")
        self.search_keep_chk.setChecked(True)
        row1.addWidget(self.search_keep_chk)

        self.search_result_label = QLabel("")
        self.search_result_label.setStyleSheet("color: #888; font-size: 9pt;")
        row1.addWidget(self.search_result_label)

        layout.addLayout(row1)

        # ── Row 2: Read / Write / Tree / XML ─────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        self.btn_read_mc = QPushButton("Read MCCONF")
        self.btn_read_mc.clicked.connect(self._on_read_mcconf)
        row2.addWidget(self.btn_read_mc)

        self.btn_read_app = QPushButton("Read APPCONF")
        self.btn_read_app.clicked.connect(self._on_read_appconf)
        row2.addWidget(self.btn_read_app)

        sep0 = QLabel("|"); sep0.setStyleSheet(sep_style)
        row2.addWidget(sep0)

        self.btn_read_mc_def = QPushButton("Read MC Default")
        self.btn_read_mc_def.clicked.connect(self._on_read_mcconf_default)
        row2.addWidget(self.btn_read_mc_def)

        self.btn_read_app_def = QPushButton("Read APP Default")
        self.btn_read_app_def.clicked.connect(self._on_read_appconf_default)
        row2.addWidget(self.btn_read_app_def)

        sep1 = QLabel("|"); sep1.setStyleSheet(sep_style)
        row2.addWidget(sep1)

        self.btn_load_xml = QPushButton("Load XML")
        self.btn_load_xml.clicked.connect(self._on_load_xml)
        row2.addWidget(self.btn_load_xml)

        self.btn_save_xml = QPushButton("Save XML")
        self.btn_save_xml.clicked.connect(self._on_save_xml)
        row2.addWidget(self.btn_save_xml)

        row2.addStretch()

        self.btn_write_mc = QPushButton("Write MCCONF")
        self.btn_write_mc.setStyleSheet(
            "QPushButton { background: #8B4513; color: white; font-weight: bold; padding: 4px 12px; }"
            "QPushButton:hover { background: #A0522D; }"
        )
        self.btn_write_mc.clicked.connect(self._on_write_mcconf)
        row2.addWidget(self.btn_write_mc)

        self.btn_write_app = QPushButton("Write APPCONF")
        self.btn_write_app.setStyleSheet(
            "QPushButton { background: #8B4513; color: white; font-weight: bold; padding: 4px 12px; }"
            "QPushButton:hover { background: #A0522D; }"
        )
        self.btn_write_app.clicked.connect(self._on_write_appconf)
        row2.addWidget(self.btn_write_app)
        layout.addLayout(row2)

        # ── Sub-tabs: MCCONF / APPCONF ────────────────────────────────
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setStyleSheet(
            "QTabBar::tab { background: #3a3a3a; color: #888; "
            "padding: 6px 16px; border: 1px solid #555; "
            "border-bottom: none; margin-right: 2px; }"
            "QTabBar::tab:selected { background: #505050; color: #fff; "
            "font-weight: bold; border-bottom: 2px solid #4FC3F7; }"
            "QTabBar::tab:hover:!selected { background: #444; color: #bbb; }"
        )
        layout.addWidget(self.sub_tabs)

        self.mc_tree = self._create_tree()
        self.app_tree = self._create_tree()
        self.sub_tabs.addTab(self.mc_tree, "MCCONF")
        self.sub_tabs.addTab(self.app_tree, "APPCONF")

        # Expand/Collapse buttons in tab bar corner
        corner_widget = QWidget()
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 4, 0)
        corner_layout.setSpacing(4)

        self.btn_expand_all = QPushButton("Expand All")
        self.btn_expand_all.clicked.connect(self._on_expand_all)
        corner_layout.addWidget(self.btn_expand_all)

        self.btn_collapse_all = QPushButton("Collapse All")
        self.btn_collapse_all.clicked.connect(self._on_collapse_all)
        corner_layout.addWidget(self.btn_collapse_all)

        self.sub_tabs.setCornerWidget(corner_widget)

        # Connect edit signals
        self.mc_tree.itemChanged.connect(self._on_mc_item_changed)
        self.app_tree.itemChanged.connect(self._on_app_item_changed)

        # Context menu (right-click)
        self.mc_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.mc_tree.customContextMenuRequested.connect(
            lambda pos: self._show_context_menu(self.mc_tree, pos))
        self.app_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.app_tree.customContextMenuRequested.connect(
            lambda pos: self._show_context_menu(self.app_tree, pos))

    # ── Tree creation ─────────────────────────────────────────────────

    def _create_tree(self) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setHeaderLabels(["Parameter", "MCU", "Tool", "Description", "Type"])
        tree.setColumnWidth(0, 350)
        tree.setColumnWidth(1, 140)
        tree.setColumnWidth(2, 140)
        tree.setColumnWidth(3, 500)
        tree.setColumnWidth(4, 50)
        tree.setAlternatingRowColors(True)
        tree.setFont(QFont("Consolas", 9))
        tree.setStyleSheet(
            "QTreeWidget { background: #272727; color: #dcdcdc; "
            "alternate-background-color: #303030; border: 1px solid #505050; }"
            "QTreeWidget::item { padding: 2px 0; }"
            "QHeaderView::section { background: #424242; color: #dcdcdc; "
            "border: 1px solid #505050; padding: 3px; }"
        )
        return tree

    # ── Tree expand/collapse state ────────────────────────────────────

    def _save_expanded_state(self, tree: QTreeWidget) -> set:
        """Return set of expanded category names."""
        expanded = set()
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            if item.isExpanded():
                expanded.add(item.text(0))
        return expanded

    def _restore_expanded_state(self, tree: QTreeWidget, expanded: set):
        """Restore expanded state by category name."""
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            item.setExpanded(item.text(0) in expanded)

    def _on_expand_all(self):
        tree = self._current_tree()
        tree.expandAll()

    def _on_collapse_all(self):
        tree = self._current_tree()
        tree.collapseAll()

    def _current_tree(self) -> QTreeWidget:
        return self.mc_tree if self.sub_tabs.currentIndex() == 0 else self.app_tree

    # ── Search ─────────────────────────────────────────────────────────

    def _on_search(self, text: str):
        """Find and scroll to matching parameters in current tree."""
        tree = self._current_tree()
        self._apply_search_to_tree(tree, text)

    def _apply_search_to_tree(self, tree: QTreeWidget, text: str):
        """Apply search filter to a specific tree. Reusable for Keep re-apply."""
        text = text.strip().lower()
        if not text:
            self.search_result_label.setText("")
            # Unhide all items
            for i in range(tree.topLevelItemCount()):
                cat = tree.topLevelItem(i)
                cat.setHidden(False)
                for j in range(cat.childCount()):
                    cat.child(j).setHidden(False)
            return

        matches = []
        for i in range(tree.topLevelItemCount()):
            cat = tree.topLevelItem(i)
            cat_has_match = False
            for j in range(cat.childCount()):
                child = cat.child(j)
                if text in child.text(0).lower():
                    child.setHidden(False)
                    cat_has_match = True
                    matches.append(child)
                else:
                    child.setHidden(True)
            cat.setHidden(not cat_has_match)
            if cat_has_match:
                cat.setExpanded(True)

        self.search_result_label.setText(f"{len(matches)} hit")

        # Scroll to first match
        if matches:
            tree.scrollToItem(matches[0])

    def _maybe_reapply_search(self, tree: QTreeWidget):
        """Re-apply search filter if Keep checkbox is checked."""
        if self.search_keep_chk.isChecked():
            text = self.search_edit.text()
            if text.strip():
                self._apply_search_to_tree(tree, text)

    # ── Context menu ───────────────────────────────────────────────────

    def _show_context_menu(self, tree: QTreeWidget, pos):
        item = tree.itemAt(pos)
        if item is None or item.parent() is None:
            return  # only for leaf items (not category headers)

        is_mc = (tree is self.mc_tree)
        defaults = self._mcconf_defaults if is_mc else self._appconf_defaults
        values = self._mcconf_values if is_mc else self._appconf_values

        menu = QMenu(self)

        # Load Default Value (factory default)
        act_default = menu.addAction("Load Default Value")

        # Revert to MCU Value (what was last read from VESC)
        original = self._mcconf_original if is_mc else self._appconf_original
        act_original = menu.addAction("Revert to MCU Value")
        if original is None:
            act_original.setEnabled(False)

        chosen = menu.exec(tree.viewport().mapToGlobal(pos))
        if chosen is None or values is None:
            return

        name = item.text(0)

        if chosen == act_default:
            if defaults is not None and name in defaults:
                # Defaults already loaded — apply immediately
                self._set_item_value(item, name, defaults[name], values, is_mc)
            elif self._transport.is_connected():
                # Auto-read defaults, then apply this field
                self._pending_default = (name, item, is_mc)
                if is_mc:
                    self._transport.send_vesc_to_target(build_get_mcconf_default())
                else:
                    self._transport.send_vesc_to_target(build_get_appconf_default())
                self.status_label.setText("Reading defaults from VESC...")
        elif chosen == act_original and original is not None and name in original:
            self._set_item_value(item, name, original[name], values, is_mc)

    def _set_item_value(self, item: QTreeWidgetItem, name: str,
                        new_val, values: OrderedDict, is_mc: bool):
        """Set a parameter value from context menu (default/original)."""
        values[name] = new_val
        dirty_set = self._mc_dirty if is_mc else self._app_dirty
        dirty_set.add(name)

        if isinstance(new_val, float):
            val_str = f"{new_val:.6f}"
        else:
            val_str = str(new_val)

        self._populating = True
        item.setText(2, val_str)
        self._populating = False

        # Highlight if differs from MCU
        orig_str = item.text(1)
        if orig_str != val_str:
            item.setForeground(2, _DIRTY_COLOR)
        else:
            item.setForeground(2, QColor("#dcdcdc"))
            dirty_set.discard(name)

    # ── Populate tree ─────────────────────────────────────────────────

    def _populate_tree(self, tree: QTreeWidget, values: OrderedDict,
                       fields, original: OrderedDict | None = None):
        self._populating = True

        # Save current expanded state before clearing
        expanded = self._save_expanded_state(tree)

        tree.clear()
        bold_font = QFont("Consolas", 9)
        bold_font.setBold(True)

        # Group by category
        categories = OrderedDict()
        for f in fields:
            if f.category not in categories:
                categories[f.category] = []
            categories[f.category].append(f)

        for cat_name, cat_fields in categories.items():
            cat_item = QTreeWidgetItem([cat_name, "", "", "", ""])
            cat_item.setFont(0, bold_font)
            cat_item.setExpanded(False)
            tree.addTopLevelItem(cat_item)

            for f in cat_fields:
                val = values.get(f.name, 0)
                if isinstance(val, float):
                    val_str = f"{val:.6f}"
                else:
                    val_str = str(val)

                # MCU column: show MCU value if available
                if original is not None:
                    orig_val = original.get(f.name, 0)
                    if isinstance(orig_val, float):
                        orig_str = f"{orig_val:.6f}"
                    else:
                        orig_str = str(orig_val)
                else:
                    orig_str = val_str

                desc_str = FIELD_DESCRIPTIONS.get(f.name, "")

                dtype_str = f.dtype
                if f.dtype == "float16":
                    dtype_str = f"float16 (/{f.scale})"

                child = QTreeWidgetItem([f.name, orig_str, val_str, desc_str, dtype_str])
                # Make only Tool column (2) editable
                child.setFlags(
                    child.flags() | Qt.ItemFlag.ItemIsEditable
                )

                # Highlight if Tool differs from MCU
                if orig_str != val_str:
                    child.setForeground(2, _DIRTY_COLOR)

                cat_item.addChild(child)

        # Annotate sensor_mode / foc_sensor_mode based on motor_type
        motor_type = values.get("motor_type")
        if motor_type is not None:
            self._annotate_sensor_mode(tree, values, motor_type)

        # Restore expanded state
        if expanded:
            self._restore_expanded_state(tree, expanded)

        self._populating = False

    # ── Sensor-mode annotation ─────────────────────────────────────────

    _SENSOR_MODE_NAMES = {0: "Sensorless", 1: "Sensored", 2: "Hybrid"}
    _FOC_SENSOR_MODE_NAMES = {0: "Sensorless", 1: "Encoder", 2: "Hall", 3: "HFI"}
    _MOTOR_TYPE_NAMES = {0: "BLDC", 1: "DC", 2: "FOC", 3: "GPD"}

    def _annotate_sensor_mode(self, tree: QTreeWidget, values: OrderedDict,
                              motor_type: int):
        """Add contextual notes to sensor_mode / foc_sensor_mode descriptions."""
        is_foc = (motor_type == 2)
        inactive_color = QColor("#666")
        active_color = QColor("#4FC3F7")  # light blue

        for i in range(tree.topLevelItemCount()):
            cat = tree.topLevelItem(i)
            for j in range(cat.childCount()):
                child = cat.child(j)
                name = child.text(0)

                if name == "sensor_mode":
                    val = int(values.get("sensor_mode", 0))
                    mode_name = self._SENSOR_MODE_NAMES.get(val, str(val))
                    if is_foc:
                        child.setText(3, f"[Inactive in FOC] {mode_name} — see foc_sensor_mode")
                        child.setForeground(0, inactive_color)
                        child.setForeground(3, inactive_color)
                    else:
                        child.setText(3, f"[Active] {mode_name}")
                        child.setForeground(3, active_color)

                elif name == "foc_sensor_mode":
                    val = int(values.get("foc_sensor_mode", 0))
                    mode_name = self._FOC_SENSOR_MODE_NAMES.get(val, str(val))
                    if is_foc:
                        child.setText(3, f"[Active] {mode_name}")
                        child.setForeground(3, active_color)
                    else:
                        child.setText(3, f"[Inactive — motor_type is not FOC] {mode_name}")
                        child.setForeground(0, inactive_color)
                        child.setForeground(3, inactive_color)

                elif name == "motor_type":
                    val = int(values.get("motor_type", 0))
                    mt_name = self._MOTOR_TYPE_NAMES.get(val, str(val))
                    child.setText(3, f"{mt_name}")

    # ── Item changed handlers ─────────────────────────────────────────

    def _on_mc_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._populating or column != 2:
            return
        self._apply_edit(item, self._mcconf_values, MCCONF_FIELDS, self._mc_dirty)

    def _on_app_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._populating or column != 2:
            return
        self._apply_edit(item, self._appconf_values, APPCONF_FIELDS, self._app_dirty)

    def _apply_edit(self, item: QTreeWidgetItem, values: OrderedDict,
                    fields, dirty_set: set):
        """Parse edited text in Tool column (2) back into the values dict."""
        if values is None:
            return

        name = item.text(0)
        new_text = item.text(2).strip()

        # Find field dtype
        dtype = None
        for f in fields:
            if f.name == name:
                dtype = f.dtype
                break
        if dtype is None:
            return

        try:
            if dtype in ("float32_auto", "float16"):
                new_val = float(new_text)
            else:
                new_val = int(float(new_text))
        except ValueError:
            # Revert to old value
            old_val = values.get(name, 0)
            self._populating = True
            if isinstance(old_val, float):
                item.setText(2, f"{old_val:.6f}")
            else:
                item.setText(2, str(old_val))
            self._populating = False
            return

        values[name] = new_val
        dirty_set.add(name)

        # Highlight Tool column if it differs from MCU
        orig_str = item.text(1)
        cur_str = item.text(2).strip()
        if orig_str != cur_str:
            item.setForeground(2, _DIRTY_COLOR)
        else:
            item.setForeground(2, QColor("#dcdcdc"))

    # ── Button handlers: Read ─────────────────────────────────────────

    def _send_via_can(self, payload: bytes):
        """Send a VESC command via CAN EID to current target."""
        if self._transport.is_connected():
            self._transport.send_vesc_to_target(payload)
            return True
        return False

    def _on_read_mcconf(self):
        self._send_via_can(build_get_mcconf())

    def _on_read_appconf(self):
        self._send_via_can(build_get_appconf())

    def _on_read_mcconf_default(self):
        self._send_via_can(build_get_mcconf_default())

    def _on_read_appconf_default(self):
        self._send_via_can(build_get_appconf_default())

    # ── Button handlers: Write ────────────────────────────────────────

    def _on_write_mcconf(self):
        if self._mcconf_values is None:
            QMessageBox.warning(self, "No Data",
                                "No MCCONF data. Read from VESC first.")
            return

        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not Connected",
                                "Open PCAN connection first.")
            return

        n_dirty = len(self._mc_dirty)
        msg = f"Write MCCONF to VESC (CAN)?\n\n{n_dirty} parameter(s) modified."
        if n_dirty > 0:
            sample = list(self._mc_dirty)[:10]
            msg += "\n\nChanged: " + ", ".join(sample)
            if n_dirty > 10:
                msg += f" ... (+{n_dirty - 10} more)"

        ret = QMessageBox.question(
            self, "Write MCCONF", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        data = serialize_conf(self._mcconf_values, MCCONF_FIELDS, MCCONF_SIGNATURE)
        packet = bytes([CommPacketId.COMM_SET_MCCONF]) + data
        self._send_via_can(packet)

        self._mc_dirty.clear()
        self._clear_dirty_highlight(self.mc_tree)
        self.status_label.setText(
            f"Wrote MCCONF ({len(data)} bytes, CAN) at "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )

    def _on_write_appconf(self):
        if self._appconf_values is None:
            QMessageBox.warning(self, "No Data",
                                "No APPCONF data. Read from VESC first.")
            return

        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not Connected",
                                "Open PCAN connection first.")
            return

        # Detect controller_id change before writing
        id_changed = False
        old_id = self._transport.motor_id
        if (self._appconf_original is not None
                and 'controller_id' in self._app_dirty):
            new_id = self._appconf_values.get('controller_id')
            orig_id = self._appconf_original.get('controller_id')
            if new_id is not None and orig_id is not None and new_id != orig_id:
                id_changed = True

        n_dirty = len(self._app_dirty)
        msg = f"Write APPCONF to VESC (CAN)?\n\n{n_dirty} parameter(s) modified."
        if n_dirty > 0:
            sample = list(self._app_dirty)[:10]
            msg += "\n\nChanged: " + ", ".join(sample)
            if n_dirty > 10:
                msg += f" ... (+{n_dirty - 10} more)"
        if id_changed:
            msg += (f"\n\nCAN ID changed ({orig_id} -> {new_id})."
                    f"\nWill re-scan to find device at new ID.")

        ret = QMessageBox.question(
            self, "Write APPCONF", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        data = serialize_conf(self._appconf_values, APPCONF_FIELDS, APPCONF_SIGNATURE)
        packet = bytes([CommPacketId.COMM_SET_APPCONF]) + data
        self._send_via_can(packet)

        self._app_dirty.clear()
        self._clear_dirty_highlight(self.app_tree)
        self.status_label.setText(
            f"Wrote APPCONF ({len(data)} bytes, CAN) at "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )

        # Re-scan if controller_id was changed (firmware applies new ID immediately)
        if id_changed:
            self.log_message.emit(
                f"[APPCONF] CAN ID changed ({orig_id} -> {new_id}), re-scanning ..."
            )
            self.rescan_requested.emit()

    def _calc_power_loss(self) -> float:
        """Calculate max_power_loss from motor wattage (25% of rated power)."""
        return self.motor_watt_spin.value() * 0.25

    def _update_power_loss_label(self):
        """Update the displayed max_power_loss value."""
        ploss = self._calc_power_loss()
        self.power_loss_label.setText(f"{ploss:.1f} W")

    def _on_detect_motor(self):
        """Send foc_detect_apply_all terminal command."""
        if not self._transport.is_connected():
            QMessageBox.warning(self, "Not Connected",
                                "Open PCAN connection first.")
            return

        power = self._calc_power_loss()
        motor_w = self.motor_watt_spin.value()
        ret = QMessageBox.question(
            self, "Detect Motor Parameters",
            f"Motor: {motor_w} W → max_power_loss: {power:.1f} W\n\n"
            f"Run foc_detect_apply_all {power:.1f}?\n\n"
            "This will spin the motor to measure R, L, flux linkage\n"
            "and auto-detect sensors (Hall/Encoder/Sensorless).\n\n"
            "Results will appear in the Terminal window.\n"
            "After completion, press Read MCCONF to see updated parameters.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        cmd = f"foc_detect_apply_all {power:.1f}"
        self._transport.send_vesc_to_target(build_terminal_cmd(cmd))
        self.log_message.emit(f"[CMD] > {cmd}")
        self.status_label.setText(
            f"Sent: {cmd} at {datetime.now().strftime('%H:%M:%S')} — check Terminal for results"
        )

    def _clear_dirty_highlight(self, tree: QTreeWidget):
        """Reset Tool column cells to normal color."""
        normal = QColor("#dcdcdc")
        for i in range(tree.topLevelItemCount()):
            cat = tree.topLevelItem(i)
            for j in range(cat.childCount()):
                cat.child(j).setForeground(2, normal)

    # ── Packet handlers (called from main_window dispatcher) ──────────

    def on_mcconf_received(self, data: bytes, is_default: bool = False):
        """Parse MCCONF binary payload and populate tree."""
        if len(data) < 8:
            return
        sig = struct.unpack_from(">I", data, 0)[0]
        if sig != MCCONF_SIGNATURE:
            return

        new_values = parse_conf(data, MCCONF_FIELDS)

        if is_default:
            self._mcconf_defaults = OrderedDict(new_values)
            # Check for pending single-field default load
            if self._pending_default is not None:
                pname, pitem, pis_mc = self._pending_default
                if pis_mc and pname in new_values:
                    self._pending_default = None
                    self._set_item_value(
                        pitem, pname, new_values[pname],
                        self._mcconf_values, True)
                    self.status_label.setText(
                        f"Loaded default for {pname}")
            return  # don't overwrite current tree with defaults

        # MCU = what MCU has now (fixed), Tool = working copy (editable)
        self._mcconf_original = OrderedDict(new_values)
        self._mcconf_values = OrderedDict(new_values)
        self._mc_dirty.clear()
        self._populate_tree(self.mc_tree, self._mcconf_values, MCCONF_FIELDS,
                            original=self._mcconf_original)
        self.mc_tree.expandAll()
        self._maybe_reapply_search(self.mc_tree)
        self.sub_tabs.setCurrentIndex(0)

        label = "MCCONF Default" if is_default else "MCCONF"
        ts = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(
            f"Last read: {label} at {ts} ({len(self._mcconf_values)} params, "
            f"{len(data)} bytes)"
        )

    def on_appconf_received(self, data: bytes, is_default: bool = False):
        """Parse APPCONF binary payload and populate tree."""
        if len(data) < 8:
            return
        sig = struct.unpack_from(">I", data, 0)[0]
        if sig != APPCONF_SIGNATURE:
            return

        new_values = parse_conf(data, APPCONF_FIELDS)

        if is_default:
            self._appconf_defaults = OrderedDict(new_values)
            # Check for pending single-field default load
            if self._pending_default is not None:
                pname, pitem, pis_mc = self._pending_default
                if not pis_mc and pname in new_values:
                    self._pending_default = None
                    self._set_item_value(
                        pitem, pname, new_values[pname],
                        self._appconf_values, False)
                    self.status_label.setText(
                        f"Loaded default for {pname}")
                    return  # don't overwrite current tree

        # MCU = what MCU has now (fixed), Tool = working copy (editable)
        self._appconf_original = OrderedDict(new_values)
        self._appconf_values = OrderedDict(new_values)
        self._app_dirty.clear()
        self._populate_tree(self.app_tree, self._appconf_values, APPCONF_FIELDS,
                            original=self._appconf_original)
        self.app_tree.expandAll()
        self._maybe_reapply_search(self.app_tree)
        self.sub_tabs.setCurrentIndex(1)

        label = "APPCONF Default" if is_default else "APPCONF"
        ts = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(
            f"Last read: {label} at {ts} ({len(self._appconf_values)} params, "
            f"{len(data)} bytes)"
        )

    # ── XML save / load ───────────────────────────────────────────────

    def _on_save_xml(self):
        """Save the currently displayed config to XML."""
        current_idx = self.sub_tabs.currentIndex()
        if current_idx == 0:
            values = self._mcconf_values
            root_tag = "MCConfiguration"
            default_name = "mcconf.xml"
        else:
            values = self._appconf_values
            root_tag = "APPConfiguration"
            default_name = "appconf.xml"

        if values is None:
            QMessageBox.warning(self, "No Data",
                                "No configuration data to save. Read from VESC first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration XML", default_name,
            "XML files (*.xml);;All files (*.*)"
        )
        if not path:
            return

        xml_str = conf_to_xml(values, root_tag)
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml_str)

        fname = os.path.basename(path)
        self.status_label.setText(f"Saved {root_tag} to {fname}")
        self.status_label.setToolTip(path)

    def _on_load_xml(self):
        """Load a VESC-Tool XML config and display in tree."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration XML", "",
            "XML files (*.xml);;All files (*.*)"
        )
        if not path:
            return

        try:
            root_tag, values = xml_to_conf(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

        if root_tag == "MCConfiguration":
            self._mcconf_values = values
            self._mc_dirty.clear()
            # MCU column stays = MCU value, Tool = loaded XML
            self._populate_tree(self.mc_tree, values, MCCONF_FIELDS,
                                original=self._mcconf_original)
            self._maybe_reapply_search(self.mc_tree)
            self.sub_tabs.setCurrentIndex(0)
        elif root_tag == "APPConfiguration":
            self._appconf_values = values
            self._app_dirty.clear()
            self._populate_tree(self.app_tree, values, APPCONF_FIELDS,
                                original=self._appconf_original)
            self._maybe_reapply_search(self.app_tree)
            self.sub_tabs.setCurrentIndex(1)

        ts = datetime.now().strftime("%H:%M:%S")
        self.status_label.setText(
            f"Loaded {root_tag} from XML at {ts} ({len(values)} params)"
        )
