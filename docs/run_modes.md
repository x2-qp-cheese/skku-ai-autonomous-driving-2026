# 기본 주행

현재 런타임은 YOLO segmentation mask를 이용한 기본 차선 추종만 수행합니다.
차선 변경, 장애물 회피, 신호등 감지 및 정지 기능은 포함하지 않습니다.

```powershell
..\venv\Scripts\python.exe scripts\drive.py --camera 0
```

횡단보도에서는 차선이 가려지거나 폭 측정이 불안정해질 수 있으므로, 횡단보도 mask를
감지하면 고정 폭 가상 중심선을 구성해 계속 주행합니다. 관련 조정 옵션은
`--corridor-crosswalk-lane-width-px`, `--corridor-crosswalk-center-smooth`,
`--corridor-crosswalk-max-center-jump`, `--corridor-crosswalk-option`입니다.

주행 중 `Space`로 시작/정지를 전환하고, `Q` 또는 `Esc`로 종료합니다.
