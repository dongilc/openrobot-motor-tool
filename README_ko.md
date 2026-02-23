# OpenRobot Motor Tool v2.2

> **[English](README.md)** | 한국어

PyQt6 기반 OpenRobot 모터 컨트롤러 GUI 툴. **CAN-Only 아키텍처** — PCAN-USB를 통한 CAN 통신 전용.

**[CAN Protocol Documentation](https://dongilc.github.io/openrobot-motor-tool/)** — OpenRobot MC CAN 통신 프로토콜 사양서 (SID/EID)

## Features

### CAN Control & Monitoring (PCAN-USB)
- **CAN Control** — 모터 제어 (위치/속도/토크), Motor Off/Stop/Start
- **CAN Data** — 실시간 모니터링 (전류, 속도, 위치, 온도), 데이터 로깅
- **Parameter** — MCCONF/APPCONF 파라미터 읽기/쓰기 (VESC EID)
- **Real-time Data** — RPM, 전류, 전압 등 실시간 그래프 (VESC EID)
- **Experiment Data** — 커스텀 plot 데이터 시각화 (COMM_PLOT)
- **Position** — 위치 제어 및 모니터링
- **Waveform** — 샘플링 파형 분석

### AI Analysis & Auto-Tuning
- **Position PID Tuning** — MCCONF Position PID (Kp/Ki/Kd/Kd Filter) 읽기/쓰기
- **Speed eRPM Tuning** — MCCONF Speed PID (Kp/Ki/Kd/Kd Filter/Ramp) 읽기/쓰기
- **Step Response Analysis** — Position (0xA3/0xA4), Speed eRPM (0xA2) 스텝 응답 분석
- **FFT Analysis** — 주파수 응답 분석, 품질 점수 계산
- **LLM Auto-Tune** — OpenAI GPT 기반 PID 자동 튜닝 추천

### Firmware Management
- **Firmware Upload** — CAN EID를 통한 펌웨어 업로드
- **Bootloader Upload** — CAN EID를 통한 부트로더 업로드

## v2 Changes (from v1)

- **CAN-Only Architecture** — Serial 코드 전면 제거, PCAN-USB 전용
- **VESC EID Integration** — Position/Speed PID를 MCCONF로 직접 읽기/쓰기
- **Auto-MCCONF Read** — CAN scan 완료 후 자동으로 MCCONF/APPCONF 읽기
- **Pole Pair Sync** — MCCONF `foc_encoder_ratio`에서 pole pair 자동 동기화
- **Bootloader Upload** — CAN을 통한 부트로더 업로드 기능 추가
- **CAN Sender ID Fix** — VESC EID 응답에서 실제 controller_id 표시

## Requirements

- Python 3.10+
- Windows (PCAN 드라이버 필요)

### Hardware
- **OpenRobot 모터 컨트롤러** (SPN-MC1 V1R2 60A)
- **USB-to-CAN 어댑터** (필수) — 아래 중 하나 필요:
  - [PCAN-USB](https://www.peak-system.com/PCAN-USB.199.0.html) (PEAK-System)
  - [PCAN-USB FD](https://www.peak-system.com/PCAN-USB-FD.365.0.html) (PEAK-System, CAN FD 지원)
  - [Pibiger USB to CAN](https://www.pibiger-tech.com/) (PCAN 호환)

> **Note**: 이 프로그램은 CAN-Only 아키텍처로 동작하며, USB-to-CAN 어댑터 없이는 사용할 수 없습니다. PCAN 드라이버가 설치되어 있어야 합니다.

## Installation

```bash
# 저장소 클론
git clone https://github.com/dongilc/openrobot-motor-tool.git
cd openrobot-motor-tool

# 가상환경 (선택)
python -m venv venv
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### Dependencies

| 패키지 | 버전 | 용도 |
|--------|------|------|
| PyQt6 | >= 6.5 | GUI 프레임워크 |
| pyqtgraph | >= 0.13 | 실시간 그래프 |
| numpy | >= 1.24 | 신호 처리 |
| scipy | >= 1.11 | FFT, 필터링 |
| openai | >= 1.0 | LLM 기반 AI 분석 |
| python-dotenv | >= 1.0 | 환경변수 (.env) 로딩 |
| PyOpenGL | >= 3.1 | pyqtgraph OpenGL 렌더링 |

### AI 분석 설정 (선택)

AI 기반 PID 튜닝 추천 기능을 사용하려면 프로젝트 루트에 `.env` 파일을 생성합니다:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

```bash
# 방법 1: 모듈로 실행
python -m openrobot_term

# 방법 2: 엔트리 포인트
python openrobot_term.py
```

### 연결

1. **PCAN-USB** 연결 후 **Open** 클릭
2. 자동으로 CAN bus scan → 디바이스 검색
3. scan 완료 후 자동으로 MCCONF/APPCONF 읽기
4. Pole pair 등 설정값 자동 동기화

## Project Structure

```
openrobot_motor_tool/
├── openrobot_term.py          # 엔트리 포인트
├── requirements.txt
├── openrobot_term/
│   ├── main.py                # 앱 초기화
│   ├── protocol/              # 통신 프로토콜
│   │   ├── can_transport.py      # PCAN-USB CAN 통신 (SID + EID)
│   │   ├── can_commands.py       # OpenRobot CAN SID 커맨드
│   │   ├── commands.py           # VESC EID 커맨드 (MCCONF/APPCONF)
│   │   ├── confparser.py         # MCCONF/APPCONF 파서
│   │   └── PCANBasic.py          # PCAN API 래퍼
│   ├── ui/                    # GUI 탭들
│   │   ├── main_window.py        # 메인 윈도우 + 패킷 디스패처
│   │   ├── connection_bar.py     # PCAN 연결 바
│   │   ├── parameter_tab.py      # MCCONF/APPCONF 파라미터
│   │   ├── realtime_tab.py       # 실시간 데이터 그래프
│   │   ├── can_position_tuning_tab.py  # AI 분석 + PID 튜닝
│   │   ├── can_control_tab.py    # CAN 모터 제어
│   │   ├── can_data_tab.py       # CAN 데이터 로깅
│   │   ├── firmware_tab.py       # 펌웨어/부트로더 업로드
│   │   └── ...
│   ├── analysis/              # 신호 분석 + AI 튜닝
│   │   ├── signal_metrics.py     # FFT, 리플, 정착시간 등
│   │   ├── llm_advisor.py        # LLM 기반 PID 추천
│   │   ├── can_auto_tuner.py     # CAN Position 자동 튜닝
│   │   ├── can_speed_auto_tuner.py  # CAN Speed eRPM 자동 튜닝
│   │   └── ...
│   └── workers/               # 백그라운드 스레드
│       ├── can_poller.py         # CAN 데이터 폴링
│       └── firmware_uploader.py  # 펌웨어/부트로더 업로드
```

## Target Hardware

- **MCU**: STM32F405 (ARM Cortex-M4, 168MHz)
- **RTOS**: ChibiOS 3.0.5
- **Motor Controller**: OpenRobot MC 시리즈 (SPN-MC1 V1R2 60A)
- **Encoder**: AS5047 (14-bit), MT6835 (21-bit) 지원
- **CAN**: [OpenRobot Motor CAN Protocol v11](https://dongilc.github.io/openrobot-motor-tool/) (SID + VESC EID)

## License

TBD
