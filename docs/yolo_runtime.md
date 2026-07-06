# YOLO Segmentation Runtime

이 브랜치의 주행 경로는 `trained_model/` 안의 YOLOv8 segmentation 모델을 기준으로 한다. 기본 경로는 `trained_model/best.pt`이고, 그 파일이 없더라도 `trained_model/` 안에 `.pt` 파일이 하나뿐이면 자동으로 그 파일을 사용한다.

```text
camera frame
  -> YOLO segmentation mask
  -> mask row sampling
  -> vehicle center vs target center error
  -> speed / steering command
  -> Arduino serial protocol
```

## 실행

```bash
python3 scripts/drive.py
```

하드웨어 없이 화면과 로그만 확인:

```bash
python3 scripts/drive.py --no-serial --show-mask
```

주행과 동시에 학습용 raw 카메라 영상을 저장:

```bash
python3 scripts/drive.py --device mps --record on
```

디버그 overlay 영상도 함께 저장:

```bash
python3 scripts/drive.py --device mps --record on --record-debug on
```

학습용으로는 `_raw.mp4`를 사용한다. `_debug.mp4`는 판단이 어떻게 되었는지 사람이 확인하는 용도다.

## 장치 선택

기본값은 `--device auto`다.

- macOS Apple Silicon: PyTorch MPS가 가능하면 `mps`
- Windows/Linux NVIDIA: CUDA가 가능하면 `0`
- 그 외: `cpu`

강제로 지정하려면 다음처럼 실행한다.

```bash
python3 scripts/drive.py --device mps
python3 scripts/drive.py --device cpu
```

## 아두이노

`firmware/arduino/vehicle_controller/vehicle_controller.ino`는 수정하지 않는다. Python 쪽은 기존 프로토콜 그대로 보낸다.

```text
DRIVE <speed> <steer>
STOP
```

macOS는 `/dev/cu.usbmodem...`, Windows는 `COM3` 같은 포트를 자동 후보로 찾는다. 자동 선택이 이상하면 명시한다.

```bash
python3 scripts/drive.py --serial-port COM3
python3 scripts/drive.py --serial-port /dev/cu.usbmodem113301
```

## 튜닝 우선순위

1. `--no-serial --show-mask`로 YOLO mask가 도로/차선을 제대로 덮는지 확인한다.
2. mask가 안정적이면 `--lookahead`, `--sample-top`, `--sample-bottom`으로 어느 깊이의 mask를 중심 계산에 쓸지 조절한다.
3. 조향이 늦으면 `kp_lateral`, `kp_heading`을 코드 config에서 올리고, 떨리면 낮춘다.
4. 커브에서 너무 멈칫하면 `--min-curve-speed`를 올리고, 직선 속도는 `--speed`로 올린다.
