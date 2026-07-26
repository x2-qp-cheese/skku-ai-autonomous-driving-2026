# T자 주차 전용 런타임

이 브랜치의 실차 런타임은 `scripts/parking.py`이며, 일반 자율주행 로직과
분리되어 있다. 현재 기본 제어기는 `ModelBasedTParkingPlanner`이다.

## 규정에서 코드로 옮긴 조건

규정집 ver.2.0의 수직 주차 조건은 다음과 같다.

- 주차 공간: `950 x 1500 mm`
- 당일 두 주차 공간 중 한 곳과 네 출발 위치 중 한 곳을 선정
- 두 주차 차량 사이에 후진으로 진입
- 앞바퀴와 뒷바퀴가 모두 라인 안에 있어야 성공
- 조향으로 주차해야 하며 차동 회전은 금지
- 주차 완료 후 `3~5초` 정지
- 회색 영역을 네 바퀴 모두 벗어나면 실격
- 출차 후 차량 앞바퀴가 OUT 라인을 지나야 전체 미션 성공

새 제어기는 시작 위치나 모터 구동 시간을 주차 조건으로 사용하지 않는다.
시간을 사용하는 곳은 규정상 필요한 정지 `3.4초`와 센서/제어 이상을 잡는
watchdog뿐이다.

## 검토한 방식

1. 카메라 주차선만 추종
   - 구조는 단순하지만 차량에 가려진 선, 반사, BEV 오차에 취약하다.
2. 초음파/LiDAR 임계 거리에서 최대 조향
   - 시간 하드코딩보다는 낫지만 시작 위치가 바뀌면 진입 곡률을 보장하지 못한다.
3. LiDAR 슬롯 지도 + 차량 모델 경로 계획 + 카메라/초음파 안전 보정
   - 상대 위치가 바뀔 때마다 경로를 다시 만들 수 있고 차량 외곽 충돌 검사도
     가능하다.

기본 구현은 3번이다. 후방 LiDAR가 두 주차 차량의 안쪽 면을 찾아 공식 크기의
주차 사각형을 잠근다. 연속 scan 사이에서는 ICP로 슬롯을 현재 차량 좌표계에
옮긴다. Hybrid A*가 실제 휠베이스와 최대 조향각을 사용하는 bicycle model로
전진 준비 구간과 후진 진입 구간을 함께 계산한다. 계산된 경로는 슬롯
좌표계에 고정하고, 매 LiDAR 갱신마다 현재 차량 좌표계로 다시 투영한다.
차량이 경로에서 설정 거리 이상 벗어났을 때만 Hybrid A*를 다시 실행한다.
따라서 gear 선택이 프레임마다 흔들리지 않으면서도 휠 엔코더나 누적 모터
시간 적분 없이 오차를 닫을 수 있다.

후방 카메라는 주차선과 뒷선을 확인해 LiDAR 슬롯과 일치할 때만 세부 정렬에
참여한다. 카메라가 불안정하면 LiDAR 슬롯이 기준을 유지한다. 좌우 초음파는
차체가 슬롯에 들어온 뒤 중앙 오차를 작게 보정하며, 전방 3개와 측면 센서는
독립 비상 정지층으로 사용한다.

## 상태 흐름

```text
IDLE
  -> SEARCH_CARS          저속 직진, 두 경계 차량 탐색
  -> TRACK_GAP            슬롯 후보를 연속 scan으로 확인
  -> VERIFY_SLOT_BOX      잠긴 950 x 1500 mm 사각형 검사
  -> FOLLOW_ENTRY_CURVE   Hybrid A* 전진/후진 구간 폐루프 추종
  -> FOLLOW_SLOT_CENTER   슬롯 내부 저속 중앙 정렬
  -> PARKED               전체 차체 내부 + 목표 pose 확인, 3.4초 정지
  -> EXIT_RIGHT
  -> EXIT_STRAIGHT        시간 대신 출차 목표 pose까지 전진 경로 추종
  -> EXIT_DONE
```

각 path pose는 `x=차량 오른쪽`, `y=차량 전방`, `heading=현재 차체 대비
회전각`으로 표현한다. 차량 외곽은 폭, 전장, 후축 위치와
`collision_clearance_mm`를 포함한 사각형이다. 양옆 주차 차량과 뒷선은
장애물 polygon으로 만들어 모든 경로 primitive에서 SAT 충돌 검사를 한다.

## 센서 장착 권장값

사진의 삼각대형 카메라 지지는 진동과 roll 변화가 생기기 쉬우며, 규정의 센서
높이 `75 cm` 제한도 반드시 실측해야 한다. 아래 값은 시작점이며 최종값은
캘리브레이션으로 확정한다.

### 후방 LiDAR

- 차체 좌우 중심에서 `±10 mm` 안에 설치
- scan 면을 지면과 평행하게 맞추고 roll/pitch 오차 `0.5도` 이내 권장
- 높이 `350~500 mm` 권장
- 브래킷, 바구니, 전선이 두 주차 차량 방향 beam을 가리지 않게 설치
- LiDAR 중심에서 후축 중심까지의 부호 있는 거리를 실측해
  `sensor_to_rear_axle_y_back_mm`에 입력

### 후방 카메라

- 차체 중심, 높이 `600~700 mm`
- 아래 방향 pitch `35~40도`, roll `0.5도` 이내 권장
- 영상 아래쪽에 후방 범퍼 근처 `0.3 m`, 위쪽에 슬롯 뒷선 약 `2.5 m`가
  동시에 보이게 조정
- 클램프 두 점 이상으로 고정하고 삼각대 관절은 테이프가 아니라 기계적으로 잠금

현재 영상은 수평선 비중이 높고 가까운 주차선이 BEV 아래에서 빨리 사라진다.
카메라를 규정 높이 안에서 낮추고 조금 더 하향하면 뒷선과 좌우 선의 동시 관측
시간이 늘어난다.

### 전방 카메라

- 차체 중심, 높이 `550~700 mm`, 아래 pitch `8~15도`
- 주차 경로의 주 센서는 아니며 전방 상황과 OUT 라인 확인용
- 출차 완료를 OUT 라인까지 자동화할 때는 전방 영상의 흰 선 검출을 별도 상태로
  연결해야 한다. 현재 `EXIT_DONE`은 주차칸을 빠져나와 차로 방향으로 정렬된
  상대 pose를 뜻한다.

### 초음파

- 좌우 센서는 후축 근처의 같은 높이 `250~400 mm`에 좌우 대칭 설치
- 지면이나 바퀴 대신 인접 차량 차체를 보도록 수평 설치
- 전방 좌/중앙/우 센서는 `-20/0/+20도` 정도로 벌려 사각을 줄임
- Arduino 보고 형식:

```text
US FL=... FC=... FR=... SL=... SR=...
```

중앙 센서가 `F=`를 보내는 기존 펌웨어도 host가 인식한다. 값 단위는 mm이며
echo 실패는 `0`, host에서는 `None`으로 취급한다.

## 반드시 먼저 실측할 값

`configs/parking.json`의 다음 값은 사진으로 정할 수 없다.

1. `vehicle.wheelbase_mm`
2. `vehicle.width_mm`, `vehicle.length_mm`
3. `vehicle.rear_axle_to_rear_bumper_mm`
4. `vehicle.max_steering_angle_deg`
5. `lidar.sensor_to_rear_axle_y_back_mm`
6. LiDAR의 `angle_offset_deg`
7. 직진 시 `model_planner.straight_steering_trim`

최대 조향각은 바퀴를 들어 눈으로 추정하지 말고, 바닥에서 일정 조향으로 원을
그린 뒤 후축 궤적 반경 `R`을 재어 `atan(wheelbase/R)`로 계산하는 것이 좋다.
좌우 반경이 다르면 작은 쪽만 쓰지 말고 조향 command-to-angle 표를 좌/우
각각 기록해야 한다. 현재 코드는 대칭 선형 mapping을 가정하므로 차이가 크면
lookup table로 바꿔야 한다.

## 캘리브레이션 순서

1. 차체 치수와 센서 원점을 mm 단위로 측정한다.
2. 바퀴를 띄운 상태에서 speed/steering 부호와 비상 정지를 확인한다.
3. 바닥 2 m 직선에서 `straight_steering_trim`만 맞춘다.
4. 평평한 벽을 LiDAR 오른쪽에 두고 debug 화면의 오른쪽에 수직으로 나타나도록
   `angle_offset_deg`를 맞춘다.
5. 정지 상태에서 목표 주차칸을 스캔해 주황 사각형이 두 차량 사이에 있고,
   초록 깊이축이 슬롯 안쪽을 향하는지 확인한다.
6. 바닥에 실제 좌표를 잰 네 점을 놓고 후방 카메라 BEV homography를 다시
   구한다. 정규화된 사다리꼴을 눈대중으로 조절하지 않는다.
7. `--no-auto-exit`으로 주차만 저속 시험한다.
8. 시작 표식 네 위치와 두 슬롯을 조합해 최소 8개 조건을 각각 반복한다.
9. 마지막에만 `auto_exit_enabled=true`로 출차를 연결한다.

## 실행

실차 전 dry run:

```bash
PYTHONPATH=src venv/bin/python scripts/parking.py \
  --source 1 \
  --front-source 0 \
  --lidar-port /dev/cu.usbserial-1130 \
  --no-auto-exit
```

모터 출력까지 활성화:

```bash
PYTHONPATH=src venv/bin/python scripts/parking.py \
  --source 1 \
  --front-source 0 \
  --lidar-port /dev/cu.usbserial-1130 \
  --serial \
  --no-auto-exit
```

치수는 JSON을 수정하지 않고도 임시 검증할 수 있다.

```bash
PYTHONPATH=src venv/bin/python scripts/parking.py \
  --wheelbase-mm 620 \
  --max-steering-angle-deg 30 \
  --rear-axle-to-rear-bumper-mm 200 \
  --parking-back-clearance-mm 120 \
  --no-auto-exit
```

조작키:

- `Space`: 미션 시작
- `R`: 즉시 정지하고 슬롯 추적기와 상태 머신 초기화
- `Q` 또는 `Esc`: 종료

## 대시보드에서 합격시킬 항목

- LiDAR `gap=CONFIRMED`, `pair=Y`
- 주황 슬롯 사각형이 인접 차량과 겹치지 않음
- 흰색 Hybrid A* 목표와 청록 경로가 슬롯 중앙으로 들어감
- 전진에서 후진으로 바뀔 때 정지 command가 먼저 출력됨
- 후진 중 rear safety ROI에 물체가 들어오면 즉시 latch 정지
- `LOCKED SLOT full=Y`, inside가 기준 이상
- PARKED 상태에서 motor command가 0인 채 3.4초 유지

## 주요 설정

- `vehicle.*`: 충돌 검사와 bicycle model의 실제 치수
- `hybrid_path.*`: 경로 primitive, 해상도, 목표 오차, gear-change 비용
- `model_planner.*`: 속도, lookahead, 완료 기준, 비상 정지, 출차 정책
- `lidar.*`: 슬롯 검출과 LiDAR-후축 외부 파라미터
- `geometry.*`, `fusion.*`: 카메라 BEV와 LiDAR 일치 판정

기존 `planner`와 `path` section은 녹화 replay 및 이전 단위 테스트 호환용이다.
실차 `scripts/parking.py`의 이동 제어에는 사용되지 않는다.
