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
현재 `drive.py`의 장애물 차선 변경은 아두이노 초음파 센서값을 사용합니다.

## 키보드 장애물 회피 테스트

초음파 센서나 LiDAR 없이 장애물 차선 변경을 연습할 때는 주행 중 `l` 키를 누릅니다.
첫 `l`은 장애물 감지처럼 차선 2에서 차선 1로 이동을 요청하고, 차선 1에 있거나 이동 중일 때 다시 `l`을 누르면 장애물 해제처럼 차선 2 복귀를 요청합니다.
기본값은 키보드 테스트가 켜진 `--lane-change-test keyboard`이며, 키 입력 전에는 차선 변경하지 않습니다.

```powershell
..\venv\Scripts\python.exe scripts\drive.py --camera 0 --traffic-light off --lane-change-test keyboard
```

차선 변경이 느리거나 조향이 작으면 아래 값부터 조정합니다.

```powershell
..\venv\Scripts\python.exe scripts\drive.py --camera 0 --traffic-light off --lane-change-test keyboard --lane-change-transition-seconds 1.0 --lane-change-steering-min 110 --lane-change-steering-boost 35
```

`--lane-change-transition-seconds`는 목표 차선으로 넘어가는 시간이고, `--lane-change-steering-min`은 변경 중 강제로 보장할 최소 조향값입니다.
`--lane-change-steering-boost`는 기존 차선 추종 조향에 추가로 더하는 값입니다.
`--lane-change-steering-override on`은 차선 변경/settle 구간에서 기존 차선 추종 조향을 섞지 않고 차선 변경 방향 조향을 우선합니다.
`--lane-change-settle-seconds`는 시간상 차선 변경이 끝난 뒤에도 같은 방향 강제 조향과 속도 제한을 유지하는 시간입니다.
현재 차선 변경 완료 판정은 시간만 보지 않고 `stabilizing_lane1/stabilizing_lane2` 상태에서 새 차선 중심 안정도를 확인합니다.
기본적으로 실제(non-virtual) 차선에서 `abs(err) <= 0.12`, `abs(head) <= 0.18`이 5프레임 연속 유지되어야 다음 차선 상태로 확정됩니다.
`--lane-change-recenter-steering on`은 왼쪽으로 피한 뒤 안정화 중 heading/lateral 조건이 맞으면 오른쪽 counter-steer를, 오른쪽으로 복귀한 뒤에는 왼쪽 counter-steer를 강하게 넣어 차체를 빨리 세웁니다.
속도 제한은 차선 변경 중에만 적용되고, 차선 1에 도착한 뒤에는 기존 `--fixed-speed on --speed 255` 설정대로 다시 255가 나갑니다.
기본값 `--light-stop-during-lane-change off`는 차선 변경/차선 1 유지 중 먼 빨간불 contact 오검출로 정지하지 않게 합니다.

## 초음파 자동 장애물 차선 변경

아두이노 펌웨어를 다시 업로드하면 `drive.py`가 전방 초음파 센서에 `USF`를 주기적으로 보내 `FR/FL` 값을 읽습니다.
기본값은 `--obstacle-lane-change on`이며, `FR` 또는 `FL` 중 하나가 `50..850mm` 범위에 들어오면 장애물로 보고 다음 차선 변경을 요청합니다.
차선을 바꾼 뒤에는 기존 차선 추종 로직으로 계속 주행하고, 값이 `950mm` 이상 또는 `0mm`/`50mm` 미만으로 clear된 뒤 다시 장애물을 만나면 반대 차선으로 다시 변경합니다.
장애물 감지 직후와 차선 변경 예약/전환 중에는 `--obstacle-speed-cap`으로 속도를 낮추고, `--obstacle-stop-mm` 이하로 너무 가까우면 정지합니다.

```powershell
..\venv\Scripts\python.exe scripts\drive.py --camera 0 --traffic-light on --obstacle-lane-change on --obstacle-trigger-mm 850 --obstacle-clear-mm 950 --obstacle-min-mm 50 --obstacle-speed-cap 90 --obstacle-stop-mm 300
```

왼쪽 센서가 아직 `9mm`처럼 튀는 상태라면 기본 `--obstacle-min-mm 50` 때문에 그 값만으로는 차선 변경하지 않습니다.
양쪽 전방 센서가 모두 감지할 때만 바꾸고 싶으면 `--obstacle-front-mode both`를 추가합니다.

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
아두이노 시리얼 명령을 직접 보낼 때는 `US`가 전체 1회 측정, `USFR/USFL/USSR/USSL`이 단일 센서 측정, `USON`이 주기 출력 켜기, `USOFF`가 주기 출력 끄기입니다.
