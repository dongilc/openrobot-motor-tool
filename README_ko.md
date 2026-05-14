# OpenRobot Motor Tool

> [English](README.md) | **한국어**

OpenRobot 모터 컨트롤러용 GUI 도구입니다. **CAN-Only 아키텍처** — PCAN-USB 또는 CANable 2.0 (slcan)을 통해 통신합니다.

**[CAN 프로토콜 문서](https://dongilc.github.io/openrobot-motor-tool/)** — OpenRobot MC CAN 통신 프로토콜 명세 (SID/EID)

## 다운로드

최신 버전은 **[Releases](https://github.com/dongilc/openrobot-motor-tool/releases/latest)** 에서 다운로드:

- **Windows**: `OpenRobot_Motor_Tool_v3.5.exe`

설치 불필요 — 단일 실행 파일.

## 주요 기능

### CAN 제어 및 모니터링 (PCAN-USB 또는 CANable 2.0/slcan)
- **CAN Control** — 모터 제어 (위치/속도/토크), Motor Off/Stop/Start
- **CAN Data** — 실시간 모니터링 (전류, 속도, 위치, 온도), 데이터 로깅
- **Parameter** — MCCONF/APPCONF 파라미터 읽기/쓰기 (VESC EID)
- **Real-time Data** — RPM, 전류, 전압 실시간 그래프 (VESC EID)
- **Experiment Data** — 사용자 정의 plot 데이터 시각화 (COMM_PLOT)
- **Position** — 위치 제어 및 모니터링
- **Waveform** — 샘플링 파형 분석

### AI 분석 및 자동 튜닝
- **Position PID Tuning** — MCCONF Position PID (Kp/Ki/Kd/Kd Filter) 읽기/쓰기
- **Speed eRPM Tuning** — MCCONF Speed PID (Kp/Ki/Kd/Kd Filter/Ramp) 읽기/쓰기
- **Step Response Analysis** — Position step, Speed eRPM step response 분석
- **FFT Analysis** — 주파수 응답 분석, 품질 점수 계산
- **LLM Auto-Tune** — OpenAI GPT 기반 PID 자동 튜닝 추천

### 펌웨어 관리
- **Firmware Upload** — CAN EID를 통한 펌웨어 업로드
- **Bootloader Upload** — CAN EID를 통한 부트로더 업로드

## 시스템 요구사항

- **OS**: Windows 10/11 (64-bit)
- **CAN 드라이버** — PCAN 드라이버 (PCAN-USB) **또는** slcan 펌웨어가 적재된 CANable 2.0 (Win10/11 에서 별도 드라이버 불필요)

### 하드웨어
- **OpenRobot Motor Controller** — 다음 중 하나:
  - **SPN-MC1 V1R2** (60V class)
  - **SPN-MC V2** (100V class, DRV8350 + INA241A3)
- **USB-to-CAN 어댑터** (필수) — 다음 중 하나:
  - [PCAN-USB](https://www.peak-system.com/PCAN-USB.199.0.html) (PEAK-System)
  - [PCAN-USB FD](https://www.peak-system.com/PCAN-USB-FD.365.0.html) (PEAK-System, CAN FD 지원)
  - [Pibiger USB to CAN](https://www.pibiger-tech.com/) (PCAN 호환)
  - **[CANable 2.0](https://canable.io/) (slcan 펌웨어)** — 저가형 USB CAN ($25 급, v3.5 신규 지원)

> **참고**: 본 도구는 CAN-Only 아키텍처를 사용하며, USB-to-CAN 어댑터 없이는 동작하지 않습니다. PCAN 계열은 PEAK-System 의 PCAN 드라이버를, CANable 2.0 은 slcan 펌웨어를 사용합니다 (Win10/11 에서 별도 드라이버 설치 불필요).

## 빠른 시작

1. [Releases](https://github.com/dongilc/openrobot-motor-tool/releases/latest) 에서 `OpenRobot_Motor_Tool_v3.5.exe` 다운로드
2. 어댑터별 드라이버 준비:
   - **PCAN-USB**: PEAK-System 에서 PCAN 드라이버 설치
   - **CANable 2.0**: slcan 펌웨어 적재 (Win10/11 별도 드라이버 불필요)
3. USB-to-CAN 어댑터를 PC에 연결
4. 실행 파일 실행
5. 백엔드 선택 (**PCAN-USB** 또는 **CANable 2.0 (slcan)**) → **Open** 클릭 → 자동 CAN 버스 스캔 시작
6. 스캔 후 MCCONF/APPCONF 자동 읽기

## 대상 하드웨어

- **MCU**: STM32F405 (ARM Cortex-M4, 168MHz)
- **Motor Controller**: OpenRobot MC 시리즈 (SPN-MC1 V1R2, SPN-MC V2)
- **Encoder**: AS5047 (14-bit), MT6835 (21-bit) 지원
- **CAN**: [OpenRobot Motor CAN Protocol v12](https://dongilc.github.io/openrobot-motor-tool/) (SID + VESC EID)

## 문서

- **[CAN 프로토콜 명세](https://dongilc.github.io/openrobot-motor-tool/)** — 전체 SID/EID 프로토콜 레퍼런스

## 지원

문의 또는 이슈는 [GitHub Issues](https://github.com/dongilc/openrobot-motor-tool/issues) 에 등록해주세요.

상용 라이선스 문의는 OpenRobot으로 연락 바랍니다.

## 라이선스

Proprietary. 상세 내용은 [LICENSE](LICENSE) 참조.

Copyright (c) 2026 OpenRobot, Inc. All rights reserved.
