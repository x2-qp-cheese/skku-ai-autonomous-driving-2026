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

LiDAR 입력과 미션 타입은 향후 장애물 미션 연결을 위해 보존되어 있지만,
현재 `drive.py`에는 아직 장애물 회피 판단이 연결되어 있지 않습니다.
