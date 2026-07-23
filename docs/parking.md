# T자 주차 런타임

`scripts/parking.py`는 주차 전용 런타임이다. 현재 브랜치의 기본 전략은
단순하다.

1. 차량은 조향 보정값 `straight_steering_trim`만 적용하고 계속 직진한다.
2. LiDAR가 오른쪽의 주차 차량/슬롯 후보를 찾는다.
3. 후보가 충분히 보이면 확정 완료를 기다리지 않고 전진 좌회전으로 주차 각을 만든다.
4. LiDAR 슬롯이 확정되고 후방 카메라/BEV 기하가 준비되면 후진 목표를 만든다.
5. 잠긴 LiDAR 슬롯과 후방 카메라 BEV 선을 융합해 후진 목표를 갱신한다.
6. 후진 경로가 생성되면 경로 곡률을 조향값으로 바꿔 슬롯 안으로 들어간다.
7. 차량이 슬롯 안에 완전히 들어가고, 뒷선 여유가 목표에 닿으면 정지한다.

이전 실험용 로직인 첫 차량 기준 선제 좌조향, `PREALIGN_LEFT`, 후륜 위치 맞춤,
전진-후진 보정 왕복은 기본 주행 경로에서 사용하지 않는다.

## 핵심 상태

```text
IDLE
  -> SEARCH_CARS          # LiDAR를 보며 직진
  -> TRACK_GAP            # 약한 슬롯 후보는 짧게 확인
  -> ENTRY_SETUP          # 후보/첫 차량 trigger가 보이면 즉시 전진 좌회전
  -> VERIFY_SLOT_BOX      # 이미 각이 충분한 경우의 확정/기하 대기 fallback
  -> PLAN_REVERSE_PATH    # 현재 슬롯 중심으로 짧은 후진 목표 생성
  -> FOLLOW_ENTRY_CURVE   # 경로 곡률 기반 후진 진입
  -> FOLLOW_SLOT_CENTER   # 슬롯 중심선 후진
  -> PARKED
```

LiDAR가 없거나 오래된 scan이면 움직이지 않는다. 빈 공간을 찾기 전에는 직진이
원칙이지만, 센서 없이 움직이는 것은 금지한다.

## 설정값

기본 설정은 `configs/parking.json`에 있다.

- `search_speed`: 슬롯을 찾을 때 직진 속도
- `gap_tracking_speed`: 후보 슬롯을 연속 확인할 때 직진 속도
- `start_forward_s`: 시작 직후 LiDAR 판단을 무시하는 강제 직진 시간. 기본값은 `0.0`
- `gap_confirm_scans`: LiDAR 슬롯 확정 scan 수. 기본값은 `2`
- `first_car_confirm_scans`: 첫 차량 기반 조기 좌회전 trigger 확인 scan 수. 기본값은 `3`
- `straight_steering_trim`: 실제 차가 직진하지 않을 때 넣는 조향 trim
- `early_entry_setup_enabled`: 후보 슬롯 또는 첫 차량 trigger에서 조기 좌회전을 시작할지 여부
- `entry_setup_speed`: 슬롯 후보 이후 주차 각을 만들 때의 전진 속도
- `entry_setup_steering`: 슬롯 후보 이후 각을 만들 때의 좌회전 조향값
- `entry_setup_min_s`: 후진 시작 전 최소로 전진 좌회전을 유지할 시간
- `entry_setup_max_s`: 각이 끝내 만들어지지 않을 때 abort할 최대 시간
- `entry_setup_target_heading_deg`: 이 heading 오차 이하가 되어야 후진 시작 가능
- `path_confirm_frames`: 후진 경로 확인 프레임 수. 기본값은 빠른 진입을 위해 `1`
- `reverse_entry_speed`: 곡선 후진 속도
- `reverse_center_speed`: 중심선 후진 속도
- `reverse_entry_min_steering`: 진입 곡선에서 너무 작은 조향이 나오지 않게 하는 하한
- `reverse_entry_release_heading_deg`: 이 각도 안에 들어오면 중심선 추종으로 전환
- `stop_depth_margin_px`: 뒷선 목표에 도달했다고 볼 BEV pixel margin

명시적으로 꺼진 값:

- `first_car_preemptive_turn_enabled=false`
- `prealign_enabled=false`
- `correction_enabled=false`

## 실행

실시간 실행 예시:

```bash
python3 scripts/parking.py \
  --source 1 \
  --front-source 0 \
  --device mps \
  --lidar-port /dev/tty.usbserial-LIDAR \
  --serial
```

직진 offset이나 진입 전 좌회전 시간을 실험할 때는 실행 인자로 바로 바꿀 수 있다.

```bash
python3 scripts/parking.py \
  --source 1 \
  --front-source 0 \
  --device mps \
  --lidar-port /dev/cu.usbserial-1130 \
  --serial \
  --straight-steering-trim -30 \
  --entry-setup-min-s 1.0 \
  --entry-setup-max-s 3.0
```

현재 설정 파일의 카메라 index:

- `rear_camera.index=1`
- `front_camera.index=0`
- 현재 아두이노 포트: `/dev/cu.usbmodem11101`

전방 카메라는 라이브 대시보드의 후방 화면 위에 작은 inset으로 표시된다.
장치 순서가 바뀌면 `--front-source 2`처럼 실행 인자만 바꾸면 된다. 필요 없을
때는 `--no-front-camera`를 사용한다.
아두이노 자동 탐색이 실패할 때만 `--serial-port /dev/cu.usbmodem11101`을
명시한다.

녹화 ZIP 재생:

```bash
PYTHONPATH=src python3 scripts/parking.py \
  --recording-zip "recording.zip" \
  --device cpu \
  --imgsz 512 \
  --frame-stride 2 \
  --auto-start
```

조작키:

- `Space`: 미션 시작
- `R`: 상태 머신, LiDAR 추정기, 카메라 추정기 초기화
- `Q` 또는 `Esc`: 종료

## 디버그에서 볼 것

대시보드의 핵심 줄은 다음이다.

- `STATE`: 현재 상태와 drive/steer 명령
- `LiDAR`: `pair=Y` 후보가 보이면 조기 좌회전이 시작될 수 있고,
  `gap=CONFIRMED`부터 후진 경로 생성이 허용된다.
- `FRONT CAMERA`: 전방 카메라 확인용 inset이다. 주차 제어에는 사용하지 않는다.
- `SLOT`: `pair=Y`는 두 차량으로 슬롯을 새로 본 상태, `coast=Y`는 잠긴 슬롯을
  잠시 추적만 하는 상태다.
- `LOCKED SLOT`: `lat`, `head`, `depth`, `full`이 후진 제어의 판단값이다.

라이다 화면:

- 주황 사각형: 잠긴 `950 x 1500 mm` 주차 슬롯
- 초록 선: 슬롯 중심선
- 청록/노랑 경로: 현재 후진 look-ahead 목표
- 파랑 박스: 감지된 주차 차량 군집
- 빨강 사각형: 후방 안전 ROI

## 센서와 경로

LiDAR는 페인트 선이 아니라 주차 차량 표면을 본다. 그래서 실제 주차칸 폭
`parking_space_width_mm=950`과 관측되는 차량 표면 간격
`expected_observed_gap_mm=1375`는 별도 값이다.

후방 카메라는 YOLO segmentation으로 주차선 mask를 만들고 BEV로 변환한다.
LiDAR 슬롯이 안정적이면 LiDAR 박스가 슬롯 정체성을 잡고, 카메라 선은 뒷선과
세부 정렬 보정에 사용된다. 카메라가 불안정하면 LiDAR 슬롯 박스로만 경로를
만든다.

후진 경로는 전체 궤적을 한 번에 고정하지 않는다. 매 프레임 현재 잠긴 슬롯에서
짧은 look-ahead 목표를 다시 만들고, 그 곡률을 조향값으로 변환한다. 이 방식은
차량이 실제로 조금씩 다른 자세로 들어가도 다음 프레임에서 경로가 다시 잡힌다.

## 안전 조건

- `SEARCH_CARS`와 `TRACK_GAP`에서는 전방 초음파 긴급 조건을 본다.
- 조기 좌회전 이후에는 LiDAR가 사라지면 전진 각도 만들기나 후진을 하지 않는다.
- `ENTRY_SETUP`은 `gap=CONFIRMED` 전에도 좌회전할 수 있지만, 후진은 확정 뒤에만 시작한다.
- `ENTRY_SETUP`에서 heading/lateral 조건이 좋아지지 않으면 후진하지 않고 abort한다.
- `emergency_stop_enabled=true`일 때 LiDAR 안전 ROI나 초음파가 임계값 이하이면
  `EMERGENCY_STOP`으로 latch된다.
- 주차 완료 판정은 `depth_remaining_px <= stop_depth_margin_px`,
  `vehicle_fully_inside=true`, 정렬 오차 허용 범위 충족이 모두 필요하다.

## 보정 순서

실차에서 먼저 맞출 값은 다음 순서가 좋다.

1. `straight_steering_trim`: 직진 중 차가 한쪽으로 흐르지 않게 맞춘다.
2. `entry_setup_min_s`: 칸을 찾은 뒤 좌회전으로 주차 각을 만들 시간을 맞춘다.
3. `entry_setup_steering`: 좌회전 방향이 실제로 맞는지 확인한다. 반대로 꺾이면 부호를 바꾼다.
4. `angle_offset_deg`: LiDAR 화면에서 차량 오른쪽 물체가 오른쪽에 보이게 한다.
5. `sensor_to_rear_axle_y_back_mm`: LiDAR 원점에서 후축까지의 부호 있는 거리.
6. `expected_slot_width_px`: BEV에서 주차칸 폭이 실제 `950 mm`와 맞게 보정한다.
7. `reverse_steering_sign`: 후진 경로가 오른쪽이면 실제 바퀴도 오른쪽으로 꺾이는지 확인한다.
