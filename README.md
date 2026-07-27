# Paper-Only Rear 2D LiDAR Parking

이 브랜치는 기존 주차 인식·경로 계획 코드를 사용하지 않는다. Arduino
차량 제어와 RPLidar 수신 코드만 유지하고 다음 논문의 수직 후진 주차
흐름을 새로 구현한다.

> H. K. Hong et al., “Automatic Reverse Parking Algorithm for
> Inter-Vehicle Spaces Using Rear 2D Lidar,” ICCE-Asia 2023.

카메라, 초음파, YOLO, 주차선, polygon, 경로 planner, 차량 형상 필터,
다중 scan 확인, 추적, 평활화는 제어 판단에 사용하지 않는다.

## 논문 흐름

1. 속도 `+80`, 조향 `0`으로 전진하면서 오른쪽 차량 수를 센다.
2. `차량 → 빈 구간 → 차량`이 관측되면 두 차량이 검출된 것으로 본다.
3. `is_near`가 참이면 속도 `+80`, 조향 `-7`로 전진한다.
4. `is_near`가 거짓이면 식 (2)~(5)로 조향을 매 scan 계산하며
   속도 `-80`으로 후진한다.
5. 후진 중 `is_near`가 참이면 잠시 정지하고 `Dist_C-Dist_D`를 계산한다.
6. 절댓값이 `250 mm` 미만이면 조향 `0`으로 후진하고, `Dist_C` 또는
   `Dist_D`가 `None`이 되는 순간 종료한다.
7. 절댓값이 `250 mm` 이상이거나 C와 D가 모두 `None`이면 조향 `0`,
   속도 `+80`으로 3초 전진한 뒤 후진 계산을 반복한다.

논문 상수는 그대로 사용한다.

- 후방 기준 각도: `0°`, 왼쪽 `-90°`, 오른쪽 `+90°`
- LiDAR 시야: `-110°..+110°`
- `near_distance`: `600 mm`
- C 구간: `-100°..-70°`
- D 구간: `+70°..+100°`
- C/D 유효 거리: `2,000 mm` 미만
- 논문 조향: `-7..+7`

논문 조향 `-7..+7`은 기존 Arduino 프로토콜의 `-150..+150`에 대칭
선형 변환한다. Arduino 펌웨어를 바꾸지 않고 실차의 직진 오차만
보정할 수 있도록 `actuator_steering_offset`을 최종 명령에 더한다.
기본값은 `0`이다.

주차공간 검출의 상세 구현은 논문에 없다. 별도 숫자를 추가하지 않기
위해 논문에 정의된 D 구간에 2,000 mm 미만 점이 있으면 오른쪽 차량이
있다고 해석한다. A/B도 논문 그림대로 후방 0°를 사이에 둔 양쪽 원시
점 중 빈 부채꼴에 닿는 두 점을 직접 사용한다.

## 실행

```bash
PYTHONPATH=src python3 scripts/parking.py --config configs/parking.json
```

디버그 창에서 `SPACE`로 시작/정지, `R`로 초기화, `Q`로 종료한다.
실행할 때마다 `data/parking_v2/`에 원시 LiDAR CSV와 논문의 A/B,
Angle_Bisector, C/D, 두 bias, 조향 계산값을 담은 telemetry CSV가
저장된다.

실차 시험은 [`docs/paper_parking_test_checklist.md`](docs/paper_parking_test_checklist.md)
순서대로 진행한다.
