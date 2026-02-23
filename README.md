# OpenRobot Motor Tool v2.1

> English | **[한국어](README_ko.md)**

PyQt6-based GUI tool for OpenRobot motor controllers. **CAN-Only architecture** — communicates exclusively via PCAN-USB.

**[CAN Protocol Documentation](https://dongilc.github.io/openrobot-motor-tool/)** — OpenRobot MC CAN communication protocol specification (SID/EID)

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
- **Step Response Analysis** — Position (0xA3/0xA4), Speed eRPM (0xA2) step response analysis
- **FFT Analysis** — Frequency response analysis, quality score calculation
- **LLM Auto-Tune** — OpenAI GPT-based PID auto-tuning recommendations

### Firmware Management
- **Firmware Upload** — Firmware upload via CAN EID
- **Bootloader Upload** — Bootloader upload via CAN EID

## v2 Changes (from v1)

- **CAN-Only Architecture** — Removed all Serial code, PCAN-USB only
- **VESC EID Integration** — Read/write Position/Speed PID directly via MCCONF
- **Auto-MCCONF Read** — Automatically read MCCONF/APPCONF after CAN scan
- **Pole Pair Sync** — Auto-sync pole pairs from MCCONF `foc_encoder_ratio`
- **Bootloader Upload** — Added bootloader upload via CAN
- **CAN Sender ID Fix** — Show actual controller_id in VESC EID responses

## Requirements

- Python 3.10+
- Windows (PCAN driver required)

### Hardware
- **OpenRobot Motor Controller** (SPN-MC1 V1R2 60A)
- **USB-to-CAN Adapter** (required) — one of the following:
  - [PCAN-USB](https://www.peak-system.com/PCAN-USB.199.0.html) (PEAK-System)
  - [PCAN-USB FD](https://www.peak-system.com/PCAN-USB-FD.365.0.html) (PEAK-System, CAN FD support)
  - [Pibiger USB to CAN](https://www.pibiger-tech.com/) (PCAN-compatible)

> **Note**: This program uses a CAN-Only architecture and cannot operate without a USB-to-CAN adapter. The PCAN driver must be installed.

## Installation

```bash
# Clone repository
git clone https://github.com/dongilc/openrobot-motor-tool.git
cd openrobot-motor-tool

# Virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyQt6 | >= 6.5 | GUI framework |
| pyqtgraph | >= 0.13 | Real-time graphs |
| numpy | >= 1.24 | Signal processing |
| scipy | >= 1.11 | FFT, filtering |
| openai | >= 1.0 | LLM-based AI analysis |
| python-dotenv | >= 1.0 | Environment variable (.env) loading |
| PyOpenGL | >= 3.1 | pyqtgraph OpenGL rendering |

### AI Analysis Setup (Optional)

To use AI-based PID tuning recommendations, create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

```bash
# Option 1: Run as module
python -m openrobot_term

# Option 2: Entry point
python openrobot_term.py
```

### Connection

1. Connect **PCAN-USB** and click **Open**
2. Automatic CAN bus scan discovers devices
3. MCCONF/APPCONF are automatically read after scan
4. Pole pairs and other settings auto-sync

## Project Structure

```
openrobot_motor_tool/
├── openrobot_term.py          # Entry point
├── requirements.txt
├── openrobot_term/
│   ├── main.py                # App initialization
│   ├── protocol/              # Communication protocol
│   │   ├── can_transport.py      # PCAN-USB CAN communication (SID + EID)
│   │   ├── can_commands.py       # OpenRobot CAN SID commands
│   │   ├── commands.py           # VESC EID commands (MCCONF/APPCONF)
│   │   ├── confparser.py         # MCCONF/APPCONF parser
│   │   └── PCANBasic.py          # PCAN API wrapper
│   ├── ui/                    # GUI tabs
│   │   ├── main_window.py        # Main window + packet dispatcher
│   │   ├── connection_bar.py     # PCAN connection bar
│   │   ├── parameter_tab.py      # MCCONF/APPCONF parameters
│   │   ├── realtime_tab.py       # Real-time data graphs
│   │   ├── can_position_tuning_tab.py  # AI analysis + PID tuning
│   │   ├── can_control_tab.py    # CAN motor control
│   │   ├── can_data_tab.py       # CAN data logging
│   │   ├── firmware_tab.py       # Firmware/bootloader upload
│   │   └── ...
│   ├── analysis/              # Signal analysis + AI tuning
│   │   ├── signal_metrics.py     # FFT, ripple, settling time, etc.
│   │   ├── llm_advisor.py        # LLM-based PID recommendations
│   │   ├── can_auto_tuner.py     # CAN position auto-tuning
│   │   ├── can_speed_auto_tuner.py  # CAN speed eRPM auto-tuning
│   │   └── ...
│   └── workers/               # Background threads
│       ├── can_poller.py         # CAN data polling
│       └── firmware_uploader.py  # Firmware/bootloader upload
```

## Target Hardware

- **MCU**: STM32F405 (ARM Cortex-M4, 168MHz)
- **RTOS**: ChibiOS 3.0.5
- **Motor Controller**: OpenRobot MC series (SPN-MC1 V1R2 60A)
- **Encoder**: AS5047 (14-bit), MT6835 (21-bit) supported
- **CAN**: [OpenRobot Motor CAN Protocol v11](https://dongilc.github.io/openrobot-motor-tool/) (SID + VESC EID)

## License

TBD
