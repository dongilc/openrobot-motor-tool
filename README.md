# OpenRobot Motor Tool

PyQt6 기반 OpenRobot 모터 컨트롤러 GUI 툴. Serial/CAN 통신, 실시간 모니터링, AI 기반 PID 자동 튜닝을 지원합니다.

## Features

### Serial
- **Parameter** — MCCONF/APPCONF 파라미터 읽기/쓰기
- **Real-time Data** — RPM, 전류, 전압 등 실시간 그래프
- **Experiment Data** — 커스텀 plot 데이터 시각화 (COMM_PLOT)
- **Position** — 위치 제어 및 모니터링
- **Waveform** — 샘플링 파형 분석
- **AI Analysis** — 속도/위치/전류 PID 스텝 응답 분석 + LLM 기반 튜닝 추천
- **Firmware** — 펌웨어 업로드
- **Motor Control** — 전류/속도/위치/Duty 수동 제어 (도킹 패널)

### CAN (PCAN-USB / OpenRobot Protocol)
- **CAN Control** — OpenRobot CAN 프로토콜 모터 제어 (위치, 속도, 토크)
- **CAN Data** — CAN 프레임 실시간 로깅 및 모니터링
- **CAN AI Analysis** — CAN 기반 스텝 응답 분석
  - Position 모드 (0xA3, 0xA4) — CAN PID 자동 튜닝
  - Speed eRPM 모드 (0xA2) — Speed PID 자동 튜닝 (serial MCCONF 경유)
  - FFT 분석, 품질 점수, LLM 기반 추천

## Requirements

- Python 3.10+
- Windows (PCAN 드라이버는 Windows 전용, Serial은 크로스플랫폼)

### Hardware
- **OpenRobot 모터 컨트롤러** (Serial 연결)
- **PCAN-USB** (CAN 기능 사용 시, 선택사항)

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
| pyserial | >= 3.5 | Serial 통신 |
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

1. **Serial** — 상단 연결 바에서 COM 포트 선택 후 Connect
2. **PCAN** — PCAN-USB 연결 후 Bitrate 설정하고 Connect (기본값 1Mbps, PCAN 드라이버 미설치 시 CAN 기능 비활성화)

## Project Structure

```
openrobot_motor_tool/
├── openrobot_term.py          # 엔트리 포인트
├── requirements.txt
├── openrobot_term/
│   ├── main.py                # 앱 초기화
│   ├── protocol/              # 통신 프로토콜
│   │   ├── serial_transport.py   # Serial 통신
│   │   ├── can_transport.py      # PCAN-USB CAN 통신
│   │   ├── can_commands.py       # OpenRobot CAN 프로토콜 커맨드
│   │   ├── commands.py           # Serial 프로토콜 커맨드
│   │   ├── packet.py             # 패킷 프레이밍
│   │   ├── confparser.py         # MCCONF/APPCONF 파서
│   │   └── PCANBasic.py          # PCAN API 래퍼
│   ├── ui/                    # GUI 탭들
│   │   ├── main_window.py        # 메인 윈도우 + 패킷 디스패처
│   │   ├── connection_bar.py     # Serial + PCAN 연결 바
│   │   ├── parameter_tab.py      # MCCONF/APPCONF 파라미터
│   │   ├── realtime_tab.py       # 실시간 데이터 그래프
│   │   ├── analysis_tab.py       # AI 분석 (Serial)
│   │   ├── can_position_tuning_tab.py  # CAN AI 분석
│   │   ├── can_control_tab.py    # CAN 모터 제어
│   │   ├── can_data_tab.py       # CAN 데이터 로깅
│   │   ├── can_realtime_tab.py   # CAN 실시간 모니터링
│   │   └── ...
│   ├── analysis/              # 신호 분석 + AI 튜닝
│   │   ├── signal_metrics.py     # FFT, 리플, 정착시간 등
│   │   ├── llm_advisor.py        # LLM 기반 PID 추천
│   │   ├── auto_tuner.py         # Serial 속도/위치 자동 튜닝
│   │   ├── can_auto_tuner.py     # CAN Position 자동 튜닝
│   │   ├── can_speed_auto_tuner.py  # CAN Speed eRPM 자동 튜닝
│   │   └── ...
│   └── workers/               # 백그라운드 스레드
│       ├── data_poller.py        # Serial 데이터 폴링
│       ├── can_poller.py         # CAN 데이터 폴링
│       └── firmware_uploader.py  # 펌웨어 업로드
```

## Target Hardware

- **MCU**: STM32F405 (ARM Cortex-M4, 168MHz)
- **RTOS**: ChibiOS 3.0.5
- **Motor Controller**: OpenRobot MC 시리즈 (SPN-MC1 V1R2 60A)
- **Encoder**: AS5047 (14-bit), MT6835 (21-bit) 지원
- **CAN**: OpenRobot Motor CAN Protocol 호환

## License

TBD
