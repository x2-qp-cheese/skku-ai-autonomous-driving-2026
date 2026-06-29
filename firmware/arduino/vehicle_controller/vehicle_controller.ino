// =================== SKKU AutoCar - Vehicle Controller (Arduino MEGA) ===================
// 배선 (사용자 제공):
//   오른쪽 뒷바퀴 모터 : IN1=D3,  IN2=D4
//   왼쪽   뒷바퀴 모터 : IN1=D7,  IN2=D8
//   조향 모터          : IN1=D11, IN2=D12
//   조향각 가변저항     : A4
//   드라이버 PWM(Enable)핀 -> 아두이노 5V 고정  (항상 활성화)
//   드라이버 5V / COM(GND) -> 아두이노 5V / GND (공통 접지 필수!)
//
//   * Enable핀이 5V에 고정되어 있으므로 속도는 IN1/IN2 핀에 PWM(analogWrite)을 걸어 제어합니다.
//   * D3,D4,D7,D8,D11,D12 는 모두 메가의 PWM 가능 핀입니다. (별도 속도핀 불필요)
//
// 시리얼 프로토콜 (PC -> 아두이노), protocol.py 와 1:1 호환:
//   "DRIVE <speed> <steer>\n"   speed: -255..255 (양수=전진), steer: -120..120 (0=직진, 양수=우측 목표)
//   "STOP\n"   "PING\n"   "POT\n"(조향 캘리브레이션용: 현재 A4값 출력)

const long BAUD_RATE = 115200;
const unsigned long SAFETY_TIMEOUT_MS = 500;   // 이 시간 동안 명령 없으면 자동 정지(안전)

// ---- 구동 모터 핀 (IN핀에 PWM을 건다) ----
const int RIGHT_IN1 = 3,  RIGHT_IN2 = 4;
const int LEFT_IN1  = 7,  LEFT_IN2  = 8;

// ---- 조향 모터 핀 ----
const int STEER_IN1 = 11, STEER_IN2 = 12;

// ---- 조향 피드백(가변저항) ----
const int STEER_POT = A4;

// ================= 조향 캘리브레이션 (반드시 실측 후 수정!) =================
// 시동 끄고 바퀴를 손으로 [정중앙/왼쪽끝/오른쪽끝] 에 두고 "POT" 명령으로 A4값을 읽어 채우세요.
int   STEER_CENTER_POT = 472;   // 바퀴가 정확히 직진일 때의 A4값 (실측)
int   STEER_LEFT_POT   = 572;   // 바퀴가 왼쪽 끝일 때의 A4값 (실측)
int   STEER_RIGHT_POT  = 399;   // 바퀴가 오른쪽 끝일 때의 A4값 (실측)
const int   STEER_INPUT_MAX = 120;  // PC가 보내는 steer 절대 최대값 (= config의 max_steering)
const int   STEER_DEADBAND  = 8;    // 이 오차 이내면 정지 (조향 떨림 방지)
const float STEER_KP        = 2.5;  // 조향 위치 P 게인 (떨면 낮추고, 느리면 올리기)
const int   STEER_MAX_PWM   = 255;  // 조향 모터 최대 출력 제한
const int   STEER_MIN_PWM   = 90;   // 데드존 밖에서 모터를 움직이기 위한 최소 출력 (끝까지 못 가는 문제 보정)
// 조향 모터가 목표 반대로 폭주하면(끝으로 밀림) 아래를 false 로 바꾸세요.
const bool  STEER_DIR_NORMAL = true;

int targetSteer = 0;   // 목표 조향 (-INPUT_MAX..INPUT_MAX)
int driveSpeed  = 0;   // 구동 속도 (-255..255)
unsigned long lastCommandAt = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  int outs[] = {RIGHT_IN1, RIGHT_IN2, LEFT_IN1, LEFT_IN2, STEER_IN1, STEER_IN2};
  for (int i = 0; i < 6; i++) pinMode(outs[i], OUTPUT);
  stopAll();
  lastCommandAt = millis();
  Serial.println("OK READY");
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      handleCommand(line);
      lastCommandAt = millis();
    }
  }

  // 통신이 끊기면(타임아웃) 안전 정지
  if (millis() - lastCommandAt > SAFETY_TIMEOUT_MS) {
    driveSpeed  = 0;
    targetSteer = 0;
  }

  applyDrive(driveSpeed);   // 구동 모터 (좌우 동시)
  updateSteering();         // 조향 위치제어 (매 루프 실행)
}

void handleCommand(const String& line) {
  if (line == "PING") { Serial.println("OK PONG"); return; }

  if (line == "POT") {                 // 캘리브레이션용: 현재 조향각 센서값 출력
    Serial.print("POT ");
    Serial.println(analogRead(STEER_POT));
    return;
  }

  if (line == "STOP") {
    driveSpeed = 0; targetSteer = 0;
    Serial.println("OK STOP");
    return;
  }

  if (line.startsWith("DRIVE ")) {
    int s1 = line.indexOf(' ');
    int s2 = line.indexOf(' ', s1 + 1);
    if (s2 < 0) { Serial.println("ERR BAD_DRIVE"); return; }
    driveSpeed  = constrain(line.substring(s1 + 1, s2).toInt(), -255, 255);
    targetSteer = constrain(line.substring(s2 + 1).toInt(), -STEER_INPUT_MAX, STEER_INPUT_MAX);
    Serial.println("OK DRIVE");
    return;
  }

  Serial.println("ERR UNKNOWN");
}

// 두 뒷바퀴를 같은 속도/방향으로 구동 (조향은 앞쪽 조향모터가 담당)
void applyDrive(int value) {
  motor(RIGHT_IN1, RIGHT_IN2, value);
  motor(LEFT_IN1,  LEFT_IN2,  value);
}

// 조향: 목표 steer -> 목표 pot값 -> A4 읽어서 P제어로 그 위치까지만 이동
void updateSteering() {
  long target;
  if (targetSteer == 0)      target = STEER_CENTER_POT;
  else if (targetSteer > 0)  target = map(targetSteer, 0, STEER_INPUT_MAX, STEER_CENTER_POT, STEER_RIGHT_POT);
  else                       target = map(targetSteer, -STEER_INPUT_MAX, 0, STEER_LEFT_POT, STEER_CENTER_POT);

  int pos = analogRead(STEER_POT);
  int err = (int)target - pos;

  if (abs(err) <= STEER_DEADBAND) {           // 목표 도달 -> 조향모터 정지
    motor(STEER_IN1, STEER_IN2, 0);
    return;
  }
  int cmd = (int)(STEER_KP * err);
  cmd = constrain(cmd, -STEER_MAX_PWM, STEER_MAX_PWM);
  // 오차가 작아져도 출력이 너무 약해 모터가 못 움직이는 걸 방지 (최소 출력 보장)
  if (cmd > 0 && cmd < STEER_MIN_PWM) cmd = STEER_MIN_PWM;
  if (cmd < 0 && cmd > -STEER_MIN_PWM) cmd = -STEER_MIN_PWM;
  if (!STEER_DIR_NORMAL) cmd = -cmd;
  motor(STEER_IN1, STEER_IN2, cmd);
}

// Enable핀이 5V에 고정된 드라이버: IN1/IN2에 PWM을 걸어 속도+방향 제어
//   value > 0 : IN1에 PWM, IN2=0   (전진)
//   value < 0 : IN1=0,   IN2에 PWM (후진)
//   value = 0 : 둘 다 0  (정지)
void motor(int in1, int in2, int value) {
  int pwm = constrain(abs(value), 0, 255);
  if (value > 0)      { analogWrite(in1, pwm); analogWrite(in2, 0);   }
  else if (value < 0) { analogWrite(in1, 0);   analogWrite(in2, pwm); }
  else                { analogWrite(in1, 0);   analogWrite(in2, 0);   }
}

void stopAll() {
  driveSpeed = 0; targetSteer = 0;
  motor(RIGHT_IN1, RIGHT_IN2, 0);
  motor(LEFT_IN1,  LEFT_IN2,  0);
  motor(STEER_IN1, STEER_IN2, 0);
}
