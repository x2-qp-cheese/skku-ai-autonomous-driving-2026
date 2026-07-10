# SKKU AI Autonomous Driving

성균관대 AI 자율주행 경진대회 SW 부문을 위한 Python/Arduino 프로젝트입니다.

현재 브랜치의 주행 런타임은 `trained_model/` 안의 YOLOv8 segmentation 모델을 사용합니다. OpenCV는 카메라 캡처와 디버그 화면 표시 용도로만 남기고, 차선/주행 가능 영역 인식은 YOLO 모델 출력 mask를 기준으로 처리합니다.

## Structure

- `configs/`: 카메라, 라이다, 아두이노 포트, 제어 상수
- `src/skku_autocar/sensors/`: 카메라와 라이다 입력
- `src/skku_autocar/perception/`: YOLO segmentation 모델 실행
- `src/skku_autocar/estimation/`: YOLO mask에서 주행 중심과 차량 중심 오차 계산
- `src/skku_autocar/planning/`: 중심 오차 기반 속도/조향 결정
- `src/skku_autocar/control/`: 아두이노 시리얼 프로토콜과 제어 명령
- `firmware/arduino/`: 아두이노 차량 제어 스케치
- `docs/`: 대회 규칙 요약, 아키텍처, 체크리스트
- `scripts/`: 실험, 장치 확인, 프로토타입 실행 스크립트
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

YOLO 주행 실행:

```bash
python3 scripts/drive.py
```

macOS Apple Silicon에서는 기본값 `--device auto`가 PyTorch MPS를 감지하면 `mps`를 사용합니다. Windows에서 NVIDIA CUDA가 가능하면 CUDA 장치를 사용하고, 둘 다 아니면 CPU로 실행합니다.

주행하면서 학습용 raw 카메라 영상 저장:

```bash
python3 scripts/drive.py --device mps --record on
```

저장 위치는 기본적으로 `data/raw/drive_recordings/<timestamp>/<timestamp>_raw.mp4`입니다. YOLO overlay가 들어간 확인용 영상까지 같이 저장하려면 `--record-debug on`을 추가합니다.

## Immediate Milestones

1. `scripts/camera_check.py`로 카메라 번호, 해상도, ROI 확인
2. `scripts/list_serial_ports.py`로 아두이노와 라이다 포트 고정
3. `firmware/arduino/vehicle_controller/vehicle_controller.ino`는 유지하고, 필요하면 Arduino IDE에서 그대로 다시 업로드
4. 학습 완료된 모델을 `trained_model/best.pt`로 두거나, `trained_model/` 안에 `.pt` 파일 하나만 둔 뒤 `python3 scripts/drive.py --no-serial --show-mask`로 mask 품질 확인
5. 시간측정 주행부터 안정화한 뒤 장애물, 신호등, 수직주차 모드를 추가
