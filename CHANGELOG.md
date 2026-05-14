# Changelog

## v3.5 (2026-05-15)

### Added
- **python-can backend** — CANable 2.0 (slcan) support, in addition to PCAN-USB
- **Backend selector** in the Connection bar (PCAN-USB / CANable 2.0 slcan), last choice persisted
- **Experiment plot — X Window selector** (50 / 100 / 200 / 500 / 1000 / 2000 / All) with sliding view
- **Read MC/App Default** now populates the parameter tree for review before Write (previously the values were silently kept in memory)

### Improved
- **slcan stream throttling** — while Experiment is streaming, CanControl skips its automatic GET_VALUES feedback poll. Fixes CRC mismatch / fragmented EID overrun on slcan UART buffers
- **Auto-Y autorange** — re-fits to currently visible curves over the visible X window only; legend visibility toggles trigger an immediate refit
- **Buffer trimming** — per-curve buffers capped at ~3× the X window (or 100k for "All") to prevent memory growth over long sessions
- Cleaner connect/disconnect log output (no duplicate lines)

## v3.4 (2026-04-21)

### Added
- 0x20A0 Output Encoder SDO group (AksIM-4 support)
- signed int32 format support for SDO

### EtherCAT
- pysoem integration (EtherCAT slave control)

### CAN
- PCAN-USB CAN-Only architecture (existing)
- VESC EID MCCONF/APPCONF read/write (existing)

## Previous Versions

Earlier releases (v2.3 ~ v3.3) are no longer maintained.
For latest features and bug fixes, please use **v3.5**.
