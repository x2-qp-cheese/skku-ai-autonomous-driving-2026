# YOLO Segmentation Runtime

이 브랜치의 주행 경로는 `trained_model/` 안의 YOLOv8 segmentation 모델을 기준으로 한다. 기본 모델은 `trained_model/skku_merged_yolov8n_seg_aug_best.pt`이며, `--model` 옵션으로 다른 모델을 명시할 수 있다.

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
3. 직진에서 계속 한쪽으로 붙으면 `--vehicle-center-offset`으로 차량 기준 중심을 보정한다. 오른쪽으로 붙으면 양수, 왼쪽으로 붙으면 음수부터 시도한다.
4. 조향각이 부족하거나 커브 진입이 늦으면 먼저 `--kp-lateral`, `--curve-steering-scale`, `--steering-rate-limit`을 올린다. 중심에서 멀어졌을 때 빨리 복귀시키려면 `--center-recovery-*` 옵션을 조절한다. `--kp-heading`은 보조값으로만 작게 조절한다.
5. 커브에 너무 빨리 들어가면 `--min-curve-speed`를 낮추거나 `--speed-curve-slowdown`을 올린다. 직선 속도는 `--speed`로 올린다.

조향이 부족할 때 시작점:

```bash
python3 scripts/drive.py --device mps --record on \
  --speed 55 --min-curve-speed 38 --max-speed 80 \
  --kp-lateral 190 --kd-lateral 45 \
  --kp-heading 12 --kd-heading 4 \
  --speed-curve-slowdown 70 \
  --min-steering-rate-limit 40 --steering-rate-limit 110 \
  --steering-release-rate-limit 22 \
  --straight-steering-scale 0.45 --curve-steering-scale 1.45 \
  --center-recovery-error-threshold 0.14 \
  --center-recovery-steering-boost 2.0 \
  --center-recovery-min-steering 85 \
  --center-recovery-rate-limit 120 \
  --center-recovery-max-speed 50 \
  --vehicle-center-offset 0.03 \
  --lookahead 0.82
```

중심 복귀가 너무 과하면 `--center-recovery-min-steering`, `--center-recovery-steering-boost`, `--center-recovery-rate-limit`을 낮춘다. 복귀가 늦으면 세 값을 올리거나 `--center-recovery-error-threshold`를 낮춘다.
코너 탈출에서 바퀴가 너무 빨리 풀리면 `--steering-release-rate-limit`을 낮추고, 너무 오래 꺾여 있으면 올린다.
직진에서 계속 오른쪽으로 붙으면 `--vehicle-center-offset 0.02`부터 `0.06`까지 올려본다. 계속 왼쪽으로 붙으면 `-0.02`부터 `-0.06`까지 낮춘다. 이 값은 화면 폭에 대한 비율이므로 `0.03`은 화면 폭의 3%만큼 차량 중심선을 오른쪽으로 옮긴다는 뜻이다.

`firmware/arduino/vehicle_controller/vehicle_controller.ino`의 `STEER_INPUT_MAX`가 120이므로, `.ino`를 바꾸지 않는 한 `--max-steering`은 120을 넘겨도 실제 조향 입력이 더 커지지 않는다.
