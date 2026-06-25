const long BAUD_RATE = 115200;
const unsigned long SAFETY_TIMEOUT_MS = 500;

// Update these pins to match the actual inspected wiring before upload.
const int DRIVE_IN1_PIN = 3;
const int DRIVE_IN2_PIN = 4;
const int STEER_IN1_PIN = 5;
const int STEER_IN2_PIN = 6;

unsigned long lastCommandAt = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(DRIVE_IN1_PIN, OUTPUT);
  pinMode(DRIVE_IN2_PIN, OUTPUT);
  pinMode(STEER_IN1_PIN, OUTPUT);
  pinMode(STEER_IN2_PIN, OUTPUT);
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

  if (millis() - lastCommandAt > SAFETY_TIMEOUT_MS) {
    stopAll();
  }
}

void handleCommand(const String& line) {
  if (line == "PING") {
    Serial.println("OK PONG");
    return;
  }

  if (line == "STOP") {
    stopAll();
    Serial.println("OK STOP");
    return;
  }

  if (line.startsWith("DRIVE ")) {
    int firstSpace = line.indexOf(' ');
    int secondSpace = line.indexOf(' ', firstSpace + 1);
    if (secondSpace < 0) {
      Serial.println("ERR BAD_DRIVE");
      return;
    }
    int speed = line.substring(firstSpace + 1, secondSpace).toInt();
    int steering = line.substring(secondSpace + 1).toInt();
    applyMotor(DRIVE_IN1_PIN, DRIVE_IN2_PIN, constrain(speed, -255, 255));
    applyMotor(STEER_IN1_PIN, STEER_IN2_PIN, constrain(steering, -255, 255));
    Serial.println("OK DRIVE");
    return;
  }

  Serial.println("ERR UNKNOWN");
}

void applyMotor(int in1, int in2, int value) {
  int pwm = abs(value);
  if (value > 0) {
    analogWrite(in1, pwm);
    analogWrite(in2, 0);
  } else if (value < 0) {
    analogWrite(in1, 0);
    analogWrite(in2, pwm);
  } else {
    analogWrite(in1, 0);
    analogWrite(in2, 0);
  }
}

void stopAll() {
  applyMotor(DRIVE_IN1_PIN, DRIVE_IN2_PIN, 0);
  applyMotor(STEER_IN1_PIN, STEER_IN2_PIN, 0);
}
