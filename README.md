# SKKU AI Autonomous Driving

성균관대 AI 자율주행 경진대회 SW 부문을 위한 Python/Arduino 프로젝트입니다.

현재 목표는 빠르게 실험하되, 대회 직전에 코드가 섞이지 않도록 센서, 인지, 판단, 제어, 미션 로직을 분리하는 것입니다. 기존 `lanedetect.py`는 초기 차선 검출 프로토타입으로 그대로 두고, 새 코드는 `src/skku_autocar/` 아래에 추가했습니다.

## Structure

- `configs/`: 카메라, 라이다, 아두이노 포트, 제어 상수
- `src/skku_autocar/sensors/`: 카메라와 라이다 입력
- `src/skku_autocar/perception/`: 차선, 신호등, 장애물 인식
- `src/skku_autocar/planning/`: 미션별 주행 판단
- `src/skku_autocar/control/`: 아두이노 시리얼 프로토콜과 제어 명령
- `firmware/arduino/`: 아두이노 차량 제어 스케치
- `docs/`: 대회 규칙 요약, 아키텍처, 체크리스트
- `tests/`: 하드웨어 없이 돌릴 수 있는 순수 Python 테스트

## Setup

대회 자료 기준 개발환경은 Python 3.9 계열을 전제로 잡았습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

하드웨어 없이 구조와 명령 포맷만 확인:

```bash
PYTHONPATH=src python3 -m skku_autocar --config configs/default.json dry-run
```

테스트:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Immediate Milestones

1. `scripts/camera_check.py`로 카메라 번호, 해상도, ROI 확인
2. `scripts/list_serial_ports.py`로 아두이노와 라이다 포트 고정
3. `firmware/arduino/vehicle_controller/vehicle_controller.ino`의 핀 번호를 실제 배선에 맞게 수정 후 업로드
4. `src/skku_autocar/perception/lane.py`의 차선 중심 추정값을 실제 트랙 영상으로 보정
5. 시간측정 주행부터 안정화한 뒤 장애물, 신호등, 수직주차 모드를 추가
