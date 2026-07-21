# T자 주차 런타임

`scripts/parking.py`는 일반 차선 주행과 분리된 주차 전용 런타임이다. 다음
순서를 그대로 상태 머신으로 실행한다.

1. 라이다로 주차칸 양옆의 주차 차량 두 대를 찾는다.
2. 두 차량 표면 사이 간격의 중심에 차량 후축을 맞춘다.
3. 정차한 뒤 후방카메라 YOLO segmentation으로 ㄷ자 주차선 세 개를 찾는다.
4. 주차칸 중심과 뒷선을 향하는 후진 경로를 만들고 추종한다.
5. 카메라에서 계산한 뒷선 여유 거리에 도달하면 정지한다.

대회 도면상 주차칸 크기는 `950 × 1500 mm`이다. 도로 폭 `850 mm`는 주차
제어에 사용하지 않는다. 라이다는 페인트 선을 보는 센서가 아니므로,
`parking_space_width_mm=950`과 실제 로그에서 측정한 이웃 차량 표면 간격
`expected_observed_gap_mm=1375`를 별도 설정값으로 유지한다.

## 모델

주차 모델은 일반 주행 모델과 분리한다.

```text
trained_model/parking_best.pt
```

Roboflow COCO segmentation 데이터로 `yolov8n-seg.pt`를 미세조정한 모델이며,
ㄷ자의 세 직선은 각각 별도 `line` 인스턴스여야 한다.

## 녹화 ZIP 재생

ZIP 안에 MP4 한 개와 `*_lidar.csv` 한 개가 있으면 직접 재생할 수 있다.
녹화 재생에서는 시리얼 출력이 강제로 금지된다.

```powershell
..\venv\Scripts\python.exe scripts/parking.py `
  --recording-zip "첫번째 라이다 데이터.zip" `
  --device cpu `
  --imgsz 512 `
  --frame-stride 2 `
  --auto-start
```

CPU에서 mask가 불안정하면 `--imgsz 640 --frame-stride 1`로 되돌린다. 녹화
재생의 상태 전이와 라이다 조회는 벽시계 시간이 아니라 영상 시간으로
동기화된다.

조작키:

- `Space`: 미션 시작/취소
- `R`: 라이다·카메라 추정기와 상태 머신 초기화
- `Q` 또는 `Esc`: 종료

## 실시간 통합 화면과 자동 녹화

숫자 카메라 소스로 실행하면 녹화 재생과 동일한 `1280 × 720` 통합 화면 하나를
표시한다. 왼쪽은 후방카메라와 YOLO mask, 오른쪽 위는 BEV, 오른쪽 아래는
LiDAR이며 아래쪽에는 상태 머신·출력 명령·초음파 값이 표시된다. 화면에 표시한
동일 프레임은 별도 옵션 없이 자동으로 다음 경로에 저장된다.

```text
data/parking/YYYYMMDD_HHMMSS.mp4
```

macOS 실시간 실행 예시(포트명은 실제 장치명으로 교체):

```bash
python3 scripts/parking.py \
  --source 1 \
  --device mps \
  --lidar-port /dev/tty.usbserial-LIDAR \
  --serial \
  --serial-port /dev/tty.usbmodem-ARDUINO
```

`--record-dashboard off`로 자동 녹화를 끌 수 있고, 녹화 영상 재생에서도 새
대시보드 영상을 만들려면 `--record-dashboard on`을 사용한다. 저장 폴더와 FPS는
각각 `--parking-record-dir`, `--dashboard-record-fps`로 바꾼다. `Q` 또는 `Esc`로
종료하면 모터 정지 명령을 보낸 뒤 MP4를 정상적으로 닫는다.

## 디버그 화면

후방카메라/BEV mask 색상:

- 청록: 왼쪽 주차선
- 초록: 오른쪽 주차선
- 빨강: 주차 뒷선
- 자홍: 선택된 주차칸에 포함되지 않은 line 인스턴스
- 주황 곡선: 생성된 후진 경로
- 노랑 점: 현재 look-ahead 목표
- 흰 원: 뒷선 앞 최종 정지 목표

라이다 화면:

- 주황 사각형: 감지된 두 이웃 차량의 라이다 방향 표면에서 시작해 주차장
  안쪽으로 1500mm만 확장되는 주차 공간. 장애물 중심을 기준으로 앞뒤 절반씩
  확장하지 않는다.
- 빨강 사각형: 긴급 정지용 좁은 직후방 ROI
- 파랑 사각형: 주차 차량 군집. 주황 공간의 양쪽 기준점이 된다.
- 공간 확정 후 차량이 회전하면 처음 오른쪽에 있던 차량이 왼쪽 좌표로 넘어갈
  수 있으므로 차량 군집 ROI는 좌우 `-1800~1800mm`를 사용한다. 공간 폭은
  고정 y축 차이가 아니라 두 군집을 잇는 실제 방향의 거리로 계산한다.
- 최초 확정 뒤에는 같은 두 차량과 주차칸 깊이 방향을 추적한다. 중심과 각도는
  새 관측에 따라 계속 갱신하지만 이전 방향과 반대인 180도 후보 및 35도보다
  큰 단일 스캔 점프는 거부한다. 두 차량을 모두 놓친 동안에는 마지막 박스를
  `HOLD`로 유지하며, 이 값으로 POSITIONING 상태를 진행하지는 않는다.
- 초록 선: 두 차량을 기준으로 만든 동적 주차 공간의 중앙선
- 청록 짧은 선: 라이다보다 앞쪽에 있는 차량 후축 기준선
- 연두 점: 라이다 측정점. 안전 ROI 안에서도 빨간 점으로 바꾸지 않는다.
- 흰색 박스와 화살표: `550 × 1000mm` 차량 외곽선과 차량 앞 방향. 라이다를
  뒤 범퍼보다 100mm 뒤에 둔 임시 장착값을 사용하므로 박스 전체가 라이다
  원점보다 앞에 표시된다.
- 자홍 십자: 차량 뒤에 장착된 라이다 원점

기본 디버그 화면은 차량 전진 방향이 위를 향하도록 `0°`로 표시한다. 이
설정은 표시 전용이며 검출 좌표를 바꾸지 않는다. 표시 방향만 바꾸려면
`--lidar-display-rotation`, 실제 센서 각도 보정은 `--lidar-angle-offset`을
사용한다.

LiDAR 장착 위치는 기본적으로 뒤 범퍼보다 10cm 뒤, 뒤 차축보다 30cm 뒤로
가정한다. 측정 후 JSON을 수정하거나 재생 명령에서
`--lidar-behind-vehicle-rear-cm 10 --lidar-to-rear-axle-cm -30`으로 바로
덮어쓸 수 있다. 뒤쪽 LiDAR 기준에서 차량 앞 방향은 음수이다.

```powershell
..\venv\Scripts\python.exe scripts/parking.py `
  --recording-zip "첫번째 라이다 데이터.zip" `
  --lidar-display-rotation 0 `
  --lidar-angle-offset -90
```

주차칸 내부가 비어 있다는 대회 조건 때문에 내부 점군의 유무를 주차칸
판정에 사용하지 않는다. 빨강 ROI는 예상 밖 물체가 차량 바로 뒤에 들어온
경우만 정지시키는 별도 안전장치다.

## 상태 순서

```text
IDLE
  -> SEARCH_CARS
  -> TRACK_GAP (첫 차량 감지 즉시 속도 10으로 감속)
  -> PREALIGN_LEFT
  -> VERIFY_PARKING_LINES
  -> PLAN_REVERSE_PATH
  -> FOLLOW_ENTRY_CURVE
  -> FOLLOW_SLOT_CENTER
  -> PARKED
```

대회 조건상 첫 번째 주차 차량 바로 옆이 빈 주차칸으로 보장되므로 두 번째
차량 검출을 기다리지 않는다. 첫 차량의 주차칸 인접 모서리를 2개 scan에서
확인하고 그 모서리가 기본 `yBack=-65cm`에 도달하면 최대 좌조향을 시작한다.
두 번째 차량은 좌조향 중에 주차칸 폭과 방향을 확정하는 데 사용한다.
`POSITION_REAR_AXLE`은 선제 좌조향 기능을 끈 경우에만 사용하는 fallback이며,
그 경우에도 오차 부호에 따라 한 방향으로 보정하고 목표에 들어오면 즉시
다음 상태로 넘어가므로 앞뒤 왕복을 반복하지 않는다.

`PREALIGN_LEFT`에서는 먼저 정지한 채 설정된 최대 좌조향까지 0.4초 동안
조향한 뒤 저속으로 전진한다. 단순 타이머 회전이 아니라 매 LiDAR scan의
주차칸 깊이 방향, 뒤축에서 입구 중심까지의 방향과 거리를 확인한다. 방향과
위치가 연속 3회 허용 범위에 들어오면 바로 후진 준비로 넘어간다. 6초 안에
정렬되지 않거나 목표 각도를 지나치면 즉시 정지하고 후방카메라로 만든 곡선
경로를 따라 우회전 후진하는 fallback을 사용한다.

녹화 영상은 이미 정해진 차량 움직임을 바꿀 수 없으므로 명령의 상태 전이와
가상 속도/조향만 검증할 수 있다. 실제 차량에서는 아래 값을 바퀴를 띄운
상태와 넓은 빈 공간에서 먼저 보정한다.

- `--prealign-speed`: 좌회전 전진 구간 속도. 기본값 35
- `--prealign-steering`: 선회 준비 전용 좌회전 명령. 기본값 -150. 후진 경로
  추종 상한 `max_steering=110`과는 독립적이다.
- `--prealign-timeout-s`: 정렬 실패 후 camera-curve fallback까지 시간. 기본값 6초
- `--first-car-turn-target-cm`: 첫 차량 모서리의 좌조향 시작 좌표. 기본값 -65cm

`scripts/arduino_parking_replay.py`는 더 이상 별도의 Python 상태 머신으로
속도와 조향을 재계산하지 않는다. 실시간 `scripts/parking.py`와 동일한
`TParkingPlanner`를 직접 호출하며, CSV의 `planner_state`, `drive_speed`,
`steer_deg`, `event`는 실시간 실행에서 생성될 값과 동일하다. 단, 녹화된
차량 궤적은 가상 명령에 반응하지 않으므로 상태가 timeout으로 끝날 수 있다.

실시간 시리얼 연결에서는 `parking.py`가 아두이노에 `USON`을 보내 측면 초음파
`SL/SR`을 약 220ms 주기로 받는다. 0mm는 echo 측정 실패로 보고 사용하지
않으며, 마지막 측정이 0.8초보다 오래되면 폐기한다. 후진 중에는
`0.23 steering/mm * (right-left)` P보정을 최대 ±35까지 카메라 경로 조향에
더한다. 유효한 어느 한쪽 거리가 100mm 이하이면 `EMERGENCY_STOP`을 latch한다.
초음파 측정이 다시 정상이어도 자동 재출발하지 않으며 운전자가 `R`로 원인을
확인하고 다시 시작해야 한다.

후축이 목표를 지나친 경우, 라이다나 카메라가 사라진 경우, 경로 곡률이
한계를 넘는 경우에는 움직이지 않는다. 후진 중 빨강 안전 ROI에 물체가
들어오면 `EMERGENCY_STOP`이 걸리고 운전자 reset 전까지 해제되지 않는다.

## 후진 경로

BEV에서 현재 후축 위치, 뒷선 앞 정지 목표, 주차선 방향을 이용해 3차
베지어 곡선을 만든다. 시작 접선은 차량의 직후방, 끝 접선은 양옆 주차선의
중심 방향이다. 매 프레임 경로를 다시 만들고 look-ahead 곡률을 조향값으로
변환하므로 mask 위치 변화에 따라 경로가 갱신된다.

## 현재 녹화 라이다 검증 결과

제공된 첫 로그는 456 scan, 52.44초, 약 8.7 Hz이다. 보정 전 화면에서는
오른쪽 축이 실제 차량 앞, 위쪽 축이 실제 차량 왼쪽으로 나타났다. raw 90°를
차량 앞 0°로 맞추기 위해 `angle_offset_deg=-90`,
`clockwise_angles=false`를 사용한다. 디버그 화면은 차량 앞=위, 오른쪽=오른쪽,
뒤=아래, 왼쪽=왼쪽으로 고정한다. 현재 보정값에서:

- 차량 오른쪽의 두 군집 간격이 처음 연속 확인되는 시점: 약 22.5초
- 최초 확인 간격: 약 1523 mm
- 최초 확인 시 임시 후축 보정값 기준 오차: 약 +26 mm

이 수치는 녹화 로그의 상대 검증값이다. 실제 차량 구동 전 아래 차량 치수를
반드시 측정해 `configs/parking.json`에 반영해야 한다.

1. 라이다 원점에서 후축 중심까지의 부호 있는 전후 거리(현재 임시값 `-300mm`)
2. 후방카메라에서 후축 중심까지의 전후 거리, 카메라 높이와 아래보기 각도
3. 차량 전체 폭, 축거, 후방 오버행
4. 최대 조향각과 조향 명령 부호

## BEV 조정

현재 시작값은 원거리 과확대를 줄이기 위해 far edge를 낮고 넓게 잡았다.

```powershell
python scripts/bev_tune.py `
  --source 녹화파일.mp4 `
  --no-centerline `
  --out-width 600 `
  --out-height 600 `
  --dst-margin 0.15 `
  --top-y 0.56 `
  --top-left-x 0.18 `
  --top-right-x 0.82 `
  --bottom-y 1.0 `
  --bottom-left-x 0.0 `
  --bottom-right-x 1.0
```

`p`를 눌러 출력된 비율을 `configs/parking.json`에 복사한다. 실제 평행한
주차선이 BEV에서도 평행하고, 주차칸 폭이 화면 상단과 하단에서 거의 같아야
한다.

`parking.py`에서도 설정 파일을 수정하지 않고 같은 항목을 덮어쓸 수 있다.
제공 영상에서 먼 주차선을 더 포함시키기 위한 시작 예시는 다음과 같다.

```powershell
..\venv\Scripts\python.exe scripts/parking.py `
  --recording-zip "첫번째 라이다 데이터.zip" `
  --device cpu `
  --imgsz 512 `
  --bev-top-y 0.25 `
  --bev-top-left-x 0.00 `
  --bev-top-right-x 1.00 `
  --bev-bottom-y 1.00 `
  --bev-bottom-left-x 0.00 `
  --bev-bottom-right-x 1.00 `
  --bev-dst-margin 0.15
```

- `--bev-top-y`를 낮추면 더 먼 영역이 보이지만 상단 확대가 강해질 수 있다.
- top-left를 낮추고 top-right를 높여 상단 폭을 넓히면 상단 확대가 줄어든다.
- `--bev-dst-margin`을 줄이면 BEV 좌우 검은 여백이 줄어든다.
- `--bev-out-width`, `--bev-out-height`로 출력 화면 크기도 바꿀 수 있다.

x 좌표는 `-1.0~2.0`까지 입력할 수 있다. `left < 0`, `right > 1`은 영상보다
넓은 영역을 source로 잡기 때문에 좌우 시야는 넓어지지만 영상 밖 검은 영역이
생길 수 있다. 반대로 BEV 속 물체 자체를 가로로 더 크게 보이게 하려면
`left > 0`, `right < 1`처럼 source 폭을 좁혀야 한다. `bev_tune.py`의 x
트랙바도 동일하게 `-1.0~2.0` 범위를 사용한다.

## 실차 실행

Apple Silicon에서는 `--device auto`가 MPS를 선택한다. 모터 출력은 녹화
검증과 바퀴를 띄운 방향 확인 후에만 명시적으로 켠다.

```powershell
..\venv\Scripts\python.exe scripts/parking.py `
  --source 1 `
  --lidar-port COM5 `
  --model trained_model/parking_best.pt `
  --device mps `
  --serial `
  --serial-port COM6
```

macOS 실차 실행 예시는 다음과 같다. `--prealign-steering -150`은 현재
`vehicle_controller.ino`의 좌측 끝 pot 보정값으로 이동하므로, 먼저 바퀴를
띄운 상태에서 실제 기구 끝과 일치하는지 확인한다.

```bash
python3 scripts/parking.py \
  --source 1 \
  --lidar-port /dev/cu.usbserial-LIDAR \
  --model trained_model/parking_best.pt \
  --device mps \
  --serial \
  --serial-port /dev/cu.usbmodem-ARDUINO \
  --prealign-speed 35 \
  --prealign-steering -150 \
  --prealign-timeout-s 6
```
