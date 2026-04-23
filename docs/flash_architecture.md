# Flash Architecture: F405 vs G474

## F405 (STM32F405, 1MB Flash, Sector-based)

```
0x08000000 ┌─────────────────────────────┐
           │ Sector 0-3  (16KB x4)       │  App (384KB)
0x08010000 │ Sector 4    (64KB)          │  - CPU boots here directly
0x08020000 │ Sector 5-7  (128KB x3)      │  - No first-stage bootloader
0x08080000 ├─────────────────────────────┤
           │ Sector 8-10 (128KB x3)      │  Staging (384KB) = NEW_APP_BASE
           │                             │  - FW upload landing zone
0x080E0000 ├─────────────────────────────┤
           │ Sector 11   (128KB)         │  Bootloader (FW updater utility)
           │                             │  - Called via software jump only
0x08100000 └─────────────────────────────┘
```

### F405 Boot / FW Update Flow
1. Power ON → CPU starts at 0x08000000 → **App runs directly**
2. FW update request via CAN:
   - `COMM_ERASE_NEW_APP` → erase sectors 8-10 (staging)
   - `COMM_WRITE_NEW_APP_DATA(offset=0, data)` → write to 0x08080000 + offset
   - `COMM_JUMP_TO_BOOTLOADER` → jump to sector 11
3. Bootloader (sector 11) copies staging → app (sectors 0-7) → reset

### F405 Bootloader Upload (CAN)
- `COMM_ERASE_BOOTLOADER` → erase sector 11
- `COMM_WRITE_NEW_APP_DATA(offset=0x60000, data)` → 0x08080000 + 0x60000 = **0x080E0000** (sector 11)
- Same write function, different offset → works because BL is ABOVE staging in address space

---

## G474 (STM32G474, 512KB Flash, Page-based 2KB uniform)

```
0x08000000 ┌─────────────────────────────┐
           │ Pages 0-7   (16KB)          │  Bootloader (first-stage)
           │                             │  - CPU boots here, jumps to App
0x08004000 ├─────────────────────────────┤
           │ Pages 8-135 (256KB)         │  App = APP_BASE
           │                             │  - VTOR at 0x08004000
0x08044000 ├─────────────────────────────┤
           │ Pages 136-247 (224KB)       │  Staging = NEW_APP_BASE
           │                             │  - FW upload landing zone
0x0807C000 ├─────────────────────────────┤
           │ Pages 248-251 (8KB)         │  (unused gap)
0x0807E000 ├─────────────────────────────┤
           │ Pages 252-255 (8KB)         │  EEPROM
0x08080000 └─────────────────────────────┘
```

### G474 Boot / FW Update Flow
1. Power ON → CPU starts at 0x08000000 → **Bootloader runs first**
2. Bootloader checks valid app at 0x08004000 → jumps to App
3. FW update request via CAN:
   - `COMM_ERASE_NEW_APP` → erase staging pages
   - `COMM_WRITE_NEW_APP_DATA(offset=0, data)` → write to 0x08044000 + offset
   - `COMM_JUMP_TO_BOOTLOADER` → reset to bootloader
4. Bootloader copies staging → app (0x08004000) → jumps to app

### G474 Bootloader Upload Problem
- `write_new_app_data(offset)` → 0x08044000 + offset
- Bootloader at 0x08000000 is BELOW staging → **negative offset needed → impossible**
- Cannot place BL at 0x080E0000 (beyond 512KB flash boundary 0x0807FFFF)
- Cannot move BL to end of flash (CPU must boot from 0x08000000)

### G474 Bootloader Upload Solution: Flag Mechanism
```
erase_bootloader()        → _bl_write_mode = true, erase pages 0-7
write_new_app_data(off):
  if _bl_write_mode       → write to 0x08000000 + offset (BL area)
  else                    → write to 0x08044000 + offset (staging)
erase_new_app()           → _bl_write_mode = false
```
- Motor Tool sends offset=0 for G474 bootloader writes
- FW-side flag auto-redirects to correct flash region
- F405 unchanged (still uses offset=0x60000, no flag needed)

---

## Key Differences Summary

| | F405 | G474 |
|---|---|---|
| Flash size | 1MB | 512KB |
| Erase unit | Sector (variable 16-128KB) | Page (uniform 2KB) |
| Write unit | Byte | Double-word (8 bytes) |
| Boot entry | 0x08000000 = App | 0x08000000 = Bootloader |
| App base | 0x08000000 | 0x08004000 |
| Staging base | 0x08080000 | 0x08044000 |
| Bootloader addr | 0x080E0000 (end) | 0x08000000 (start) |
| BL via offset? | Yes (offset=0x60000) | No (flag mechanism) |
| BL upload offset | 0x60000 | 0 (with flag) |

## Motor Tool MCU Detection
- HW_NAME from `COMM_FW_VERSION` response identifies the board
- G474 boards: HW_NAME contains "G474" or known G474 board names
- Motor Tool uses offset=0 for G474, offset=0x60000 for F405
