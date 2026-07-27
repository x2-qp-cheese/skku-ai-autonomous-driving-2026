# LiDAR-only T자 주차

실차 진입점은 `scripts/parking.py`이고 기본 설정은
`configs/parking.json`이다. 카메라와 초음파는 주차 판단 및 조향에 사용하지
않는다. 주차칸 선택, 차체 자세, 경로 생성, 후진 조향, 완료 판정은 모두 후방
2D LiDAR의 metric 좌표를 사용한다.

## 논문에서 채택한 부분

기준 논문은 *Implementation of Autonomous Parking System Using
LiDAR-based Triangulation Method* (ACK 2023)이다.

논문의 핵심은 LiDAR 원점, 주차칸 위쪽 차량 1의 안쪽 모서리, 아래쪽 차량 2의
안쪽 모서리가 만드는 결정삼각형이다. 두 모서리를 `P1`, `P2`, LiDAR를 `L`이라
하면 다음 값을 매 fresh scan에서 계산한다.

```text
a = |L - P1|
b = |L - P2|
c = |P2 - P1|
decision_angle = acos((a² + c² - b²) / (2ac))
```

`P1-P2` 축의 수직방향 중 빈 주차칸 안쪽을 향하는 방향이 주차칸 깊이축이다.
이 깊이축과 차체 후방축의 차이가 `correction_angle`이다.

논문 원형은 `0.5초 대기`, `3.5초 경로 보정`을 사용한다. 이 구현은 그 시간
루틴을 사용하지 않는다. 결정삼각형을 반복 관측해 만든 현재 slot pose에서
최종 주차 pose까지 실제 차량이 실행 가능한 전체 경로가 생겼을 때만 다음
상태로 넘어간다.

## 후방 LiDAR 좌표 변환

논문 차량은 LiDAR가 전방 하단에 있지만 이 차량은 후방에 있다. raw 각도
임계값 `269~271도`, `280도`를 그대로 복사하면 안 된다.

이 구현의 LiDAR 좌표는 다음과 같다.

```text
x_right > 0 : 차체 오른쪽
y_back  > 0 : 차체 뒤쪽
```

논문의 전방 기준 보정각 대신 `+y_back`을 후진 기준축으로 사용한다.
planning 좌표로 옮길 때는 다음 변환을 한 번만 적용한다.

```text
x_plan    = x_right
y_forward = sensor_to_rear_axle_y_back_mm - y_back
```

현재 후방 LiDAR에서 후축은 앞쪽에 있으므로
`sensor_to_rear_axle_y_back_mm=-300`이다. 디버그 화면 회전값은 perception
좌표를 바꾸지 않는다.

## 상태 흐름과 전환 근거

```text
IDLE
  -> SEARCH_CARS
  -> TRACK_GAP
  -> ENTRY_SETUP / acquire_slot
  -> ENTRY_SETUP / align_reverse
  -> FOLLOW_ENTRY_CURVE
  -> FOLLOW_SLOT_CENTER
  -> PARKED
```

### 1. SEARCH_CARS

- 속도 `+90`, 직진 조향 `-33`
- 오른쪽 `x=350~1100 mm`, 거리 `3 m` 이내의 밀도 높은 첫 차량 표면을 찾는다.
- 동일 물체가 서로 다른 LiDAR timestamp에서 3회 확인돼야 첫 차량으로 인정한다.
- 한 스캔을 control loop가 여러 번 읽어도 카운터는 한 번만 증가한다.

디버그:

```text
lidar_search_first_car:n/3
```

### 2. TRACK_GAP

- 첫 차량을 확인하면 속도를 `+50`으로 낮춘다.
- 첫 차량 표면이 사라진 fresh scan 6개가 연속될 때 빈칸 시작을 확정한다.
- 단발 dropout이나 같은 timestamp 반복은 좌조향을 시작시키지 않는다.

디버그:

```text
lidar_first_car_seen_wait_open:n/6
```

### 3. ENTRY_SETUP - acquire_slot

- 빈칸이 확정되면 속도 `+90`, 조향 `-150`으로 좌측 확보 동작을 시작한다.
- 이 순간 이전 LiDAR cluster와 잠긴 polygon을 초기화한다.
- 좌조향 후 새 관측에서 차량 1과 차량 2의 안쪽 모서리를 찾는다.
- 두 모서리 간격, cluster 크기, 연속 관측, 결정삼각형이 모두 유효해야
  공식 크기 `950 x 1500 mm` slot polygon을 잠근다.

디버그:

```text
lidar_open_gap_confirmed_start_left_setup
entry_acquire_lidar_triangle_after_reset:<gap>/<triangle>
```

### 4. ENTRY_SETUP - align_reverse

- 잠긴 slot pose를 후방 LiDAR fresh pair로 재고정하고, 일시 가림은 제한된
  LiDAR scan matching으로 추적한다.
- 좌조향 속도를 `+50`으로 낮춘다.
- 매 새 pose에서 `우조향 arc -> 직선 -> 반대조향 arc -> 직선`으로 구성된
  reverse-only S 경로를 탐색한다.
- 경로의 모든 pose에서 폭 `600 mm`, 전장 `1000 mm`의 차체와 `40 mm`
  계획 여유를 적용해 두 인접 차량 및 뒷경계와 충돌하는지 검사한다.
- 최종 차체 네 모서리가 slot 안에 들어가는 전체 경로만 허용한다.
- 전체 경로가 처음 생기면 즉시 전진을 멈춘다. 정지 상태에서 동일한
  slot-relative 경로를 서로 다른 3개 LiDAR scan에 재투영해 다시 안전한 경우에만
  후진 기어를 넣는다.

디버그:

```text
entry_align_left_until_reverse_reachable:<reason>
entry_reverse_path_stationary_confirm:1/3
entry_reverse_path_stationary_confirm:2/3
reverse_only_path_armed_after_entry_setup
```

### 5. FOLLOW_ENTRY_CURVE / FOLLOW_SLOT_CENTER

- 후진 속도는 기본 `-90`, 목표 근처는 `-60`이다.
- 경로 각 구간의 곡률을 feed-forward로 주고, live LiDAR 경로오차를 20% 섞어
  조향한다.
- 후진 중 매 scan에서 잠긴 slot 기준 남은 경로와 swept footprint를 재검사한다.
- 이 차량의 기계적 직진값은 `-33`이므로 수학적 조향 `0`을 물리 명령 `-33`에
  대응시키는 비대칭 affine mapping을 사용한다. 좌우 최대 명령 `-150/+150`은
  그대로 보존한다.
- 카메라 보정과 좌우 초음파 보정은 LiDAR-only 모드에서 입력하지 않는다.

디버그:

```text
gear_change_stationary_settle:-1
follow_parking_path_reverse ...
follow_parking_path_final_reverse ...
```

### 6. PARKED

다음 조건을 모두 3회 연속 만족하면 정지한다.

- 후축 목표 위치 오차 `115 mm` 이하
- slot 방향 오차 `7도` 이하
- LiDAR slot polygon 안의 차체 점유율 `96%` 이상
- 차체 네 모서리가 모두 polygon 내부

이동시간이나 후진시간으로 완료를 판정하지 않는다. 규정상 완료 후 정지만
`3.4초`를 사용한다. 현재 실험 기본값은 자동 출차가 꺼져 있다.

## 마지막 기록의 예상 상태 전환

`20260727_201225_lidar.csv`를 현재 설정으로 입력했을 때:

```text
11.369  TRACK_GAP       첫 차량 3 scan 확정
12.280  ENTRY_SETUP     첫 차량 lost 6 scan, 좌조향 시작
16.371  align_reverse   두 차량 결정삼각형 및 slot 확정
18.077  STOP            완전한 3.00 m S 경로 최초 생성
18.188  STOP            같은 경로 2번째 확인
18.303  FOLLOW_ENTRY    같은 경로 3번째 확인, 후진 arm
18.643  REVERSE         speed=-90, 첫 우조향 arc 시작
```

이 시간들은 녹화 로그에서 센서 조건이 성립한 시각일 뿐 제어 임계값이 아니다.

## 실행

모터 없이 LiDAR와 상태 흐름만 확인:

```bash
PYTHONPATH=src .venv/bin/python scripts/parking.py \
  --lidar-port /dev/cu.usbserial-1130 \
  --manual-start \
  --no-auto-exit
```

실차 모터 출력:

```bash
PYTHONPATH=src .venv/bin/python scripts/parking.py \
  --lidar-port /dev/cu.usbserial-1130 \
  --serial \
  --manual-start \
  --no-auto-exit
```

Space를 누르면 시작한다. 카메라가 기본 비활성화되어 있으므로 `--source`나
YOLO 모델 인자는 필요 없다. 실제 장치명이 다르면 먼저 확인한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/list_serial_ports.py
```

마지막 CSV를 모터 없이 재생:

```bash
PYTHONPATH=src .venv/bin/python scripts/parking.py \
  --lidar-csv data/parking/20260727_201225_lidar.csv \
  --auto-start \
  --no-auto-exit
```

## 실차 전에 반드시 확인할 값

1. 물체를 차량 오른쪽에 놓았을 때 LiDAR debug의 `RIGHT`에 찍히는지 확인한다.
2. 물체를 차량 바로 뒤에 놓았을 때 `REAR`에 찍히는지 확인한다.
3. LiDAR 중심에서 후축까지 거리를 실측해 `sensor_to_rear_axle_y_back_mm`를
   수정한다.
4. 2 m 직선 주행으로 `straight_steering_trim=-33`을 다시 확인한다.
5. 좌우 최대 조향에서 실제 road-wheel 각도를 재고
   `max_steering_angle_deg`를 수정한다.
6. 차체 폭, 전장, 후축-범퍼 거리를 실제 가장 바깥 돌출부 기준으로 잰다.
7. 첫 시험은 차량 사이에 사람 없이 `--no-auto-exit`으로 수행한다.

주요 설정:

- `lidar.max_distance_mm=3500`
- `lidar.stale_after_s=0.45`
- `runtime.camera_enabled=false`
- `model_planner.lidar_only_enabled=true`
- `model_planner.right_ultrasonic_slot_confirm_enabled=false`
- `vehicle.collision_clearance_mm=40`
- `runtime.locked_slot_max_translation_per_scan_mm=120`
- `runtime.locked_slot_max_rotation_per_scan_deg=8`
