// =================== SKKU AutoCar - Vehicle Controller (Arduino MEGA) ===================
// 배선 (사용자 제공):
//   오른쪽 뒷바퀴 모터 : IN1=D3,  IN2=D4
//   왼쪽   뒷바퀴 모터 : IN1=D7,  IN2=D8
//   조향 모터          : IN1=D11, IN2=D12
//   조향각 가변저항     : A4
//   초음파 센서:
//     전방 우: TRIG=D22, ECHO=D23
//     전방 좌: TRIG=D24, ECHO=D25
//     전방 중앙: TRIG=D30, ECHO=D31
//     옆   우: TRIG=D26, ECHO=D27
//     옆   좌: TRIG=D28, ECHO=D29
//   드라이버 PWM(Enable)핀 -> 아두이노 5V 고정  (항상 활성화)
//   드라이버 5V / COM(GND) -> 아두이노 5V / GND (공통 접지 필수!)
//
//   * Enable핀이 5V에 고정되어 있으므로 속도는 IN1/IN2 핀에 PWM(analogWrite)을 걸어 제어합니다.
//   * D3,D4,D7,D8,D11,D12 는 모두 메가의 PWM 가능 핀입니다. (별도 속도핀 불필요)
//
// 시리얼 프로토콜 (PC -> 아두이노), protocol.py 와 1:1 호환:
//   "DRIVE <speed> <steer>\n"   speed: -255..255 (양수=전진), steer: -150..150 (0=직진, 양수=우측 목표)
//   "STOP\n"   "PING\n"   "POT\n"(조향 캘리브레이션용: 현재 A4값 출력)
//   "US\n"     캐시된 최신값 출력: US FC=<mm> FR=<mm> FL=<mm> SR=<mm> SL=<mm> (0=미검출/타임아웃)
//   "USF\n"    캐시된 전방 최신값 출력: US FC=<mm> FR=<mm> FL=<mm>
//   "USFC\n"   전방 중앙 1개만 측정, "USFR\n" 전방 우, "USFL\n" 전방 좌
//   "USSR\n"   옆 우 1개만 측정, "USSL\n" 옆 좌
//   "USON\n"   캐시된 초음파 값 스트리밍 켜기, "USOFF\n" 끄기

const long BAUD_RATE = 115200;
const unsigned long SAFETY_TIMEOUT_MS = 500;   // 이 시간 동안 명령 없으면 자동 정지(안전)

// ---- 구동 모터 핀 (IN핀에 PWM을 건다) ----
const int RIGHT_IN1 = 3,  RIGHT_IN2 = 4;
const int LEFT_IN1  = 7,  LEFT_IN2  = 8;

// ---- 조향 모터 핀 ----
const int STEER_IN1 = 11, STEER_IN2 = 12;

// ---- 조향 피드백(가변저항) ----
const int STEER_POT = A4;

// ---- 초음파 센서 핀 ----
const int US_FRONT_RIGHT_TRIG  = 22, US_FRONT_RIGHT_ECHO  = 23;
const int US_FRONT_LEFT_TRIG   = 24, US_FRONT_LEFT_ECHO   = 25;
const int US_FRONT_CENTER_TRIG = 30, US_FRONT_CENTER_ECHO = 31;
const int US_SIDE_RIGHT_TRIG   = 26, US_SIDE_RIGHT_ECHO   = 27;
const int US_SIDE_LEFT_TRIG    = 28, US_SIDE_LEFT_ECHO    = 29;

const unsigned long US_ECHO_TIMEOUT_US = 18000UL;  // 약 3.1m. YOLO 조기 인식 뒤 원거리 회피 승인을 허용.
const unsigned long US_SAMPLE_INTERVAL_MS = 30;
const unsigned long US_STREAM_INTERVAL_MS = 150;
const int US_SENSOR_COUNT = 5;
const int US_TRIG_PINS[US_SENSOR_COUNT] = {
  US_FRONT_CENTER_TRIG, US_FRONT_RIGHT_TRIG, US_FRONT_LEFT_TRIG,
  US_SIDE_RIGHT_TRIG, US_SIDE_LEFT_TRIG
};
const int US_ECHO_PINS[US_SENSOR_COUNT] = {
  US_FRONT_CENTER_ECHO, US_FRONT_RIGHT_ECHO, US_FRONT_LEFT_ECHO,
  US_SIDE_RIGHT_ECHO, US_SIDE_LEFT_ECHO
};
int ultrasonicValues[US_SENSOR_COUNT] = {0, 0, 0, 0, 0};
int nextUltrasonicSensor = 0;
bool ultrasonicCycleReady = false;
bool ultrasonicStream = false;
unsigned long lastUltrasonicSampleAt = 0;
unsigned long lastUltrasonicAt = 0;

// ================= 조향 캘리브레이션 (반드시 실측 후 수정!) =================
// 시동 끄고 바퀴를 손으로 [정중앙/왼쪽끝/오른쪽끝] 에 두고 "POT" 명령으로 A4값을 읽어 채우세요.
int   STEER_CENTER_POT = 484;   // 바퀴가 정확히 직진일 때의 A4값 (실측)
int   STEER_LEFT_POT   = 572;   // 바퀴가 왼쪽 끝일 때의 A4값 (실측)
int   STEER_RIGHT_POT  = 399;   // 바퀴가 오른쪽 끝일 때의 A4값 (실측)
const int   STEER_INPUT_MAX = 150;  // PC가 보내는 steer 절대 최대값 (= config의 max_steering)
const int   STEER_DEADBAND  = 4;    // 이 오차 이내면 정지 (조향 떨림 방지)
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
  setupUltrasonicPins();
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
  updateUltrasonicSensors();
  updateUltrasonicStream();
}

void handleCommand(const String& line) {
  if (line == "PING") { Serial.println("OK PONG"); return; }

  if (line == "POT") {                 // 캘리브레이션용: 현재 조향각 센서값 출력
    Serial.print("POT ");
    Serial.println(analogRead(STEER_POT));
    return;
  }

  if (line == "US") {
    printCachedUltrasonicReadings();
    return;
  }

  if (line == "USF") {
    printCachedFrontUltrasonicReadings();
    return;
  }

  if (line == "USFC") { printSingleUltrasonic("FC", US_FRONT_CENTER_TRIG, US_FRONT_CENTER_ECHO); return; }
  if (line == "USFR") { printSingleUltrasonic("FR", US_FRONT_RIGHT_TRIG, US_FRONT_RIGHT_ECHO); return; }
  if (line == "USFL") { printSingleUltrasonic("FL", US_FRONT_LEFT_TRIG, US_FRONT_LEFT_ECHO); return; }
  if (line == "USSR") { printSingleUltrasonic("SR", US_SIDE_RIGHT_TRIG, US_SIDE_RIGHT_ECHO); return; }
  if (line == "USSL") { printSingleUltrasonic("SL", US_SIDE_LEFT_TRIG, US_SIDE_LEFT_ECHO); return; }

  if (line == "USON") {
    ultrasonicStream = true;
    lastUltrasonicAt = 0;
    Serial.println("OK USON");
    return;
  }

  if (line == "USOFF") {
    ultrasonicStream = false;
    Serial.println("OK USOFF");
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
// 배선상 전진/후진이 반대로 연결되어 있어 부호를 반전합니다.
void applyDrive(int value) {
  motor(RIGHT_IN1, RIGHT_IN2, -value);
  motor(LEFT_IN1,  LEFT_IN2,  -value);
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

void setupUltrasonicPins() {
  pinMode(US_FRONT_CENTER_TRIG, OUTPUT);
  pinMode(US_FRONT_RIGHT_TRIG, OUTPUT);
  pinMode(US_FRONT_LEFT_TRIG, OUTPUT);
  pinMode(US_SIDE_RIGHT_TRIG, OUTPUT);
  pinMode(US_SIDE_LEFT_TRIG, OUTPUT);
  pinMode(US_FRONT_CENTER_ECHO, INPUT);
  pinMode(US_FRONT_RIGHT_ECHO, INPUT);
  pinMode(US_FRONT_LEFT_ECHO, INPUT);
  pinMode(US_SIDE_RIGHT_ECHO, INPUT);
  pinMode(US_SIDE_LEFT_ECHO, INPUT);
  digitalWrite(US_FRONT_CENTER_TRIG, LOW);
  digitalWrite(US_FRONT_RIGHT_TRIG, LOW);
  digitalWrite(US_FRONT_LEFT_TRIG, LOW);
  digitalWrite(US_SIDE_RIGHT_TRIG, LOW);
  digitalWrite(US_SIDE_LEFT_TRIG, LOW);
}

void updateUltrasonicSensors() {
  unsigned long now = millis();
  if (now - lastUltrasonicSampleAt < US_SAMPLE_INTERVAL_MS) return;
  lastUltrasonicSampleAt = now;
  ultrasonicValues[nextUltrasonicSensor] = readUltrasonicMm(
    US_TRIG_PINS[nextUltrasonicSensor],
    US_ECHO_PINS[nextUltrasonicSensor]
  );
  nextUltrasonicSensor++;
  if (nextUltrasonicSensor >= US_SENSOR_COUNT) {
    nextUltrasonicSensor = 0;
    ultrasonicCycleReady = true;
  }
}

void updateUltrasonicStream() {
  if (!ultrasonicStream || !ultrasonicCycleReady) return;
  unsigned long now = millis();
  if (now - lastUltrasonicAt < US_STREAM_INTERVAL_MS) return;
  lastUltrasonicAt = now;
  printCachedUltrasonicReadings();
}

int readUltrasonicMm(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  unsigned long duration = pulseIn(echoPin, HIGH, US_ECHO_TIMEOUT_US);
  if (duration == 0) return 0;
  // 왕복 시간(us) * 음속(mm/us) / 2. 20도 기준 음속은 약 0.343 mm/us.
  return (int)((duration * 343UL) / 2000UL);
}

void printCachedUltrasonicReadings() {
  Serial.print("US FC=");
  Serial.print(ultrasonicValues[0]);
  Serial.print(" FR=");
  Serial.print(ultrasonicValues[1]);
  Serial.print(" FL=");
  Serial.print(ultrasonicValues[2]);
  Serial.print(" SR=");
  Serial.print(ultrasonicValues[3]);
  Serial.print(" SL=");
  Serial.println(ultrasonicValues[4]);
}

void printCachedFrontUltrasonicReadings() {
  Serial.print("US FC=");
  Serial.print(ultrasonicValues[0]);
  Serial.print(" FR=");
  Serial.print(ultrasonicValues[1]);
  Serial.print(" FL=");
  Serial.println(ultrasonicValues[2]);
}

void printSingleUltrasonic(const char* name, int trigPin, int echoPin) {
  int value = readUltrasonicMm(trigPin, echoPin);
  Serial.print("US ");
  Serial.print(name);
  Serial.print("=");
  Serial.println(value);
}
