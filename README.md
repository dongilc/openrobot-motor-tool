# OpenRobot Motor Tool

> English | **[한국어](README_ko.md)**

GUI tool for OpenRobot motor controllers. **CAN-Only architecture** — communicates exclusively via PCAN-USB.

**[CAN Protocol Documentation](https://dongilc.github.io/openrobot-motor-tool/)** — OpenRobot MC CAN communication protocol specification (SID/EID)

## Download

Download the latest version from **[Releases](https://github.com/dongilc/openrobot-motor-tool/releases/latest)**:

- **Windows**: `OpenRobot_Motor_Tool_v3.4.exe`

No installation required — single executable.

## Features

### CAN Control & Monitoring (PCAN-USB)
- **CAN Control** — Motor control (position/speed/torque), Motor Off/Stop/Start
- **CAN Data** — Real-time monitoring (current, speed, position, temperature), data logging
- **Parameter** — MCCONF/APPCONF parameter read/write (VESC EID)
- **Real-time Data** — RPM, current, voltage real-time graphs (VESC EID)
- **Experiment Data** — Custom plot data visualization (COMM_PLOT)
- **Position** — Position control and monitoring
- **Waveform** — Sampling waveform analysis

### AI Analysis & Auto-Tuning
- **Position PID Tuning** — MCCONF Position PID (Kp/Ki/Kd/Kd Filter) read/write
- **Speed eRPM Tuning** — MCCONF Speed PID (Kp/Ki/Kd/Kd Filter/Ramp) read/write
- **Step Response Analysis** — Position step, Speed eRPM step response analysis
- **FFT Analysis** — Frequency response analysis, quality score calculation
- **LLM Auto-Tune** — OpenAI GPT-based PID auto-tuning recommendations

### Firmware Management
- **Firmware Upload** — Firmware upload via CAN EID
- **Bootloader Upload** — Bootloader upload via CAN EID

## System Requirements

- **OS**: Windows 10/11 (64-bit)
- **PCAN driver** installed (PCAN-USB compatible)

### Hardware
- **OpenRobot Motor Controller** (SPN-MC1 V1R2)
- **USB-to-CAN Adapter** (required) — one of the following:
  - [PCAN-USB](https://www.peak-system.com/PCAN-USB.199.0.html) (PEAK-System)
  - [PCAN-USB FD](https://www.peak-system.com/PCAN-USB-FD.365.0.html) (PEAK-System, CAN FD support)
  - [Pibiger USB to CAN](https://www.pibiger-tech.com/) (PCAN-compatible)

> **Note**: This tool uses a CAN-Only architecture and cannot operate without a USB-to-CAN adapter. The PCAN driver must be installed separately.

## Quick Start

1. Download `OpenRobot_Motor_Tool_v3.4.exe` from [Releases](https://github.com/dongilc/openrobot-motor-tool/releases/latest)
2. Install PCAN driver from PEAK-System
3. Connect PCAN-USB adapter to PC
4. Run the executable
5. Click **Open** to start automatic CAN bus scan
6. MCCONF/APPCONF are automatically read after scan

## Target Hardware

- **MCU**: STM32F405 (ARM Cortex-M4, 168MHz)
- **Motor Controller**: OpenRobot MC series (SPN-MC1 V1R2)
- **Encoder**: AS5047 (14-bit), MT6835 (21-bit) supported
- **CAN**: [OpenRobot Motor CAN Protocol v12](https://dongilc.github.io/openrobot-motor-tool/) (SID + VESC EID)

## Documentation

- **[CAN Protocol Specification](https://dongilc.github.io/openrobot-motor-tool/)** — Full SID/EID protocol reference

## Support

For issues or questions, please open an [issue](https://github.com/dongilc/openrobot-motor-tool/issues).

For commercial licensing inquiries, contact OpenRobot.

## License

Proprietary. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 OpenRobot, Inc. All rights reserved.
