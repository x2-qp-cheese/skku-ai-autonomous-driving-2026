# 주행 실행 모드

현재 경기 런타임은 항상 YOLO 클래스별 mask를 BEV로 변환한 뒤 2차선 corridor를 계산합니다.
별도의 `--bev-corridor` 옵션은 필요하지 않습니다.

## 속도 측정 미션

```powershell
..\venv\Scripts\python.exe scripts\drive.py --camera 0 --traffic-light off
```

## 신호등 미션

```powershell
..\venv\Scripts\python.exe scripts\drive.py --camera 0 --traffic-light on --crosswalk-halt off
```

두 실행 모두 시작 직후에는 PAUSE이며 스페이스바를 눌러 주행을 시작합니다.
Arduino 자동 검색이 실패하면 `--serial-port COM3`처럼 포트를 지정합니다.
macOS에서는 카메라 인덱스에 AVFoundation을 사용하고 Arduino는
`/dev/cu.usbmodem*` 또는 `/dev/cu.usbserial*` 포트를 우선 선택합니다.

LiDAR 입력과 미션 타입은 향후 확장을 위해 보존되어 있습니다.
현재 `drive.py`의 장애물 회피는 YOLO segmentation과 초음파 센서를 융합합니다.

## 키보드 장애물 회피 테스트

YOLO 장애물 감지 없이 차선 변경을 연습할 때는 주행 중 `l` 키를 누릅니다.
첫 `l`은 장애물 감지처럼 차선 2에서 차선 1로 이동을 요청하고, 차선 1에 있거나 이동 중일 때 다시 `l`을 누르면 장애물 해제처럼 차선 2 복귀를 요청합니다.
기본값 `--lane-change-mode external`은 센서 융합과 `l` 키 요청을 모두 받습니다.

```powershell
..\venv\Scripts\python.exe scripts\drive.py --camera 0 --traffic-light off --lane-change-mode external --obstacle-avoidance off
```

차선 변경이 느리거나 조향이 작으면 아래 값부터 조정합니다.

```powershell
..\venv\Scripts\python.exe scripts\drive.py --camera 0 --traffic-light off --lane-change-mode external --obstacle-avoidance off --lane-change-transition-seconds 1.0 --lane-change-steering-min 110 --lane-change-steering-boost 35
```

`--lane-change-transition-seconds`는 목표 차선으로 넘어가는 시간이고, `--lane-change-steering-min`은 변경 중 강제로 보장할 최소 조향값입니다.
`--lane-change-steering-boost`는 기존 차선 추종 조향에 추가로 더하는 값입니다.
`--lane-change-steering-override on`은 차선 변경 구간에서 차선 추종 조향을 섞지 않고 변경 방향 조향을 우선합니다.
현재 차선 변경 완료 판정은 시간만 보지 않고 `stabilizing_lane1/stabilizing_lane2` 상태에서 새 차선 중심 안정도를 확인합니다.
기본적으로 실제(non-virtual) 차선에서 `abs(err) <= 0.12`, `abs(head) <= 0.18`이 5프레임 연속 유지되어야 다음 차선 상태로 확정됩니다.
속도 제한은 차선 변경 중에만 적용되고, 차선 1에 도착한 뒤에는 기존 `--fixed-speed on --speed 255` 설정대로 다시 255가 나갑니다.
기본값 `--light-stop-during-lane-change off`는 차선 변경/차선 1 유지 중 먼 빨간불 contact 오검출로 정지하지 않게 합니다.

## YOLO + 초음파 장애물 회피

학습 데이터에서 장애물 외곽을 polygon으로 지정하고 클래스 이름을 정확히 `obstacle`로 사용합니다.
차선 모델과 같은 YOLO segmentation 모델에 이 클래스를 함께 학습해야 합니다.

판단 순서는 다음과 같습니다.

1. 카메라 obstacle mask를 BEV Bayesian 점유 지도에 누적하고 connected component별 장애물로 분리합니다. 기본값에서는 이 단계에서 감속하거나 차선을 변경하지 않습니다.
2. 현재 차선과 반대 차선 후보 경로를 점유 지도와 실선 mask로 미리 평가해 목표 차선을 `plan=L1/L2`로 저장합니다. 목적 차선 점유가 바뀌면 저장된 계획도 즉시 취소하거나 다시 계산합니다.
3. 신뢰도 0.75 이상의 현재 경로 점유가 유지된 상태에서 전방 초음파가 2600mm 이내로 들어오면, 저장된 계획을 측면 초음파와 최신 점유 지도로 재검증한 뒤 같은 프레임에서 감속과 회피를 시작합니다. 기본 TTC 선행 트리거는 꺼져 있으므로 2600mm보다 먼 거리에서는 실행하지 않습니다.
4. 목적 경로의 YOLO 점유, 목적 방향 측면 초음파, 두 차선 중심 사이의 `lane-side` 실선 검사를 모두 통과해야 변경을 시작합니다. 목적 차선 바깥의 정상 외곽 실선은 변경을 차단하지 않습니다.
5. 목적 경로가 막힌 상태로 650mm 이내까지 접근하면 정지합니다. 변경 전에는 전방 센서 2개가 300mm 이내여도 독립 비상 정지합니다.
6. 변경을 시작하면 시간 보간 없이 목적 차선 전체 중심선을 목표로 설정합니다. 근거리 목표 오차가 0.32보다 크면 변경 방향 우선 조향을 유지하고, 0.32 이내에서는 차선 추종 피드백을 허용해 횡방향 오버슈트를 줄입니다. 0.20 이내로 2프레임 들어와야 안정화로 전환하며, 근거리와 원거리 목표 오차가 함께 안정될 때만 정상 주행으로 복귀합니다.
7. 변경 중 원래 차선 장애물의 초음파 잔여 에코와 큰 근접 mask는 중간 제동을 만들지 않습니다. 완료 직후 동일 mask가 두 경로에 걸치면 `CLEARING_SOURCE`로 통과한 뒤 새 장애물을 판단합니다.
8. 한 장애물 이벤트는 차선 변경과 안정화가 끝날 때까지 한 번만 소비됩니다. 원래 차선에 남은 같은 component는 재요청하지 않지만, 새 차선이 안정된 뒤 새 현재 경로에서 별도 장애물이 잡히고 반대 경로가 비어 있으면 clear 구간 없이 다음 목표 차선을 미리 계획합니다.

라이브 카메라는 BEV를 보정한 `1280x720` 프레임과 정확히 일치해야 합니다. 카메라가 `640x480` 같은 다른 모드로 열리면 아두이노 연결 전에 실행을 거부합니다. `--camera-resolution-policy allow`는 `--no-serial` 보정 작업에서만 사용할 수 있습니다.

초음파 단독 신호는 경로와 물체 종류를 알 수 없으므로 차선 변경을 시작하지 않습니다.
신뢰도 0.75 미만의 obstacle mask는 추적 정보에만 사용하며 선제 감속이나 차선 변경 요청에는 사용하지 않습니다. 가까운 저신뢰도 물체는 비상 정지 대상으로는 유지합니다.

```powershell
..\venv\Scripts\python.exe scripts\drive.py --camera 1 --width 1280 --height 720 --fourcc MJPG --camera-resolution-policy strict --traffic-light on --obstacle-avoidance on --obstacle-local-map on --obstacle-visual-slowdown off --obstacle-fusion-mode fused --obstacle-action-confidence 0.75 --obstacle-frame-visual-trigger-y 0.12 --obstacle-visual-trigger-y 0.05 --obstacle-trigger-mm 2600 --obstacle-clear-mm 2900 --ultrasonic-max-valid-mm 3200 --obstacle-min-front-sensors 2 --obstacle-range-confirm-frames 1 --obstacle-rearm-clear-frames 3 --obstacle-ttc-seconds 0 --obstacle-stop-mm 300 --obstacle-blocked-stop-mm 650 --obstacle-side-clearance-mm 300 --obstacle-speed-cap 135 --obstacle-solid-crossing-margin-px 8 --lane-change-transition-seconds 1.0 --lane-change-speed-cap 135 --lane-change-steering-min 150 --lane-change-steering-cap 150 --lane-change-unreliable-speed-cap 70 --lane-change-unreliable-steering-cap 90 --lane-change-stabilizing-steering-min 70 --lane-change-target-approach-error 0.32 --lane-change-target-capture-error 0.20 --lane-change-target-capture-frames 2 --lane-change-stable-lateral-error 0.12 --lane-change-stable-near-error 0.18 --lane-change-stable-heading-error 0.18 --lane-change-stable-frames 5
```

`--obstacle-path-half-width-px`는 차량이 점유할 BEV 충돌 경로의 반폭입니다.
`--obstacle-min-overlap`은 obstacle 접지 mask가 경로와 겹쳐야 하는 최소 비율입니다.
`--obstacle-solid-crossing-margin-px`는 두 차선 중심 사이에 실선이 있는지 판정할 때 쓰는 BEV 허용 오차입니다.
현재 모델에 `obstacle` 클래스가 없으면 경고만 출력하고 기존 차선 주행은 계속합니다.
아두이노 없이 녹화 영상을 재생할 때만 `--no-serial --obstacle-fusion-mode yolo`를 사용합니다.

## 키보드 수동 조종 + 라벨링 사진 저장

`debug_drive.py`는 YOLO 없이 카메라 화면을 보면서 키보드로 아두이노에 직접 `DRIVE/STOP` 명령을 보냅니다.
동시에 기본 1초 간격으로 라벨링용 원본 이미지를 `data/raw/debug_drive/<timestamp>/images/`에 저장하고, 당시 속도/조향은 `metadata.csv`에 기록합니다.

```powershell
..\venv\Scripts\python.exe scripts\debug_drive.py --camera 0 --start-speed 100 --max-speed 100 --max-steering 150 --capture-interval 1.0
```

기본값은 `--hold-to-run on`이라 `W/S/A/D` 입력이 `--key-timeout` 시간 안에 반복되지 않으면 자동으로 `STOP`을 보냅니다.
키를 누르고 있는데도 너무 자주 끊기면 `--key-timeout 0.5`처럼 조금 늘립니다.
기본 조작은 `--control-style direct`라 `W`를 누르면 속도 100, `A/D`를 누르면 지정 조향값이 즉시 들어갑니다.
키는 `W/S` 속도, `A/D` 조향, `Z` 조향 중앙, `X` 속도 0, `Space` 즉시 STOP, `C` 수동 사진 저장, `T` 자동 저장 on/off, `Q` 종료입니다.
아두이노 없이 카메라와 저장만 확인하려면 `--no-serial`을 추가합니다.
전방/후방 카메라를 동시에 저장하려면 전방을 `--camera`, 후방을 `--rear-camera`로 지정합니다.
이때 이미지는 `images/front/`, `images/rear/`에 같은 번호와 시간으로 저장됩니다.

```powershell
..\venv\Scripts\python.exe scripts\debug_drive.py --camera 0 --rear-camera 1 --start-speed 100 --max-speed 100 --max-steering 150 --capture-interval 1.0
```

## 초음파 센서 확인

아두이노 MEGA 펌웨어는 아래 핀의 HC-SR04 계열 초음파 센서를 읽습니다.

| 위치 | TRIG | ECHO |
| --- | ---: | ---: |
| 전방 우 | D22 | D23 |
| 전방 좌 | D24 | D25 |
| 전방 중앙 | D30 | D31 |
| 옆 우 | D26 | D27 |
| 옆 좌 | D28 | D29 |

펌웨어를 다시 업로드한 뒤 PC에서 값을 확인합니다.

```powershell
..\venv\Scripts\python.exe scripts\ultrasonic_check.py --interval 0.2
```

출력은 mm 단위이며 `0mm`은 echo 타임아웃 또는 미검출입니다.
한쪽 센서만 따로 확인하려면 `--sensor front_left`, `--sensor side_left`처럼 지정합니다.

```powershell
..\venv\Scripts\python.exe scripts\ultrasonic_check.py --sensor front_left --interval 0.2
..\venv\Scripts\python.exe scripts\ultrasonic_check.py --sensor side_left --interval 0.2
```

왼쪽 센서만 `0mm` 또는 `9mm`처럼 고정되면 왼쪽 센서의 `VCC/GND`, `TRIG/ECHO 반대 연결`, 핀 번호(D24/D25, D28/D29), 센서 불량을 우선 확인합니다.
펌웨어는 30ms마다 센서 하나를 순환 측정합니다. `US`와 `USF`는 캐시된 최신값을 즉시 출력하고, `USFC/USFR/USFL/USSR/USSL`은 고장 진단용 단일 센서 측정입니다. `USON`과 `USOFF`는 캐시값 스트리밍을 켜고 끕니다.
