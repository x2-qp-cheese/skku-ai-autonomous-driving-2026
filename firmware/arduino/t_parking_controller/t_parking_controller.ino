// ================= LEGACY REFERENCE - DO NOT FLASH FOR LIVE PARKING =====
//
// scripts/parking.py now runs the model-based closed-loop planner on the host
// and sends DRIVE/STOP commands through SerialVehicleClient. This older
// standalone state machine contains fixed-duration fallback manoeuvres and is
// intentionally not used by the live runtime.
//
// ================= Arduino Mega - Autonomous T Parking =================
//
// Required hardware-layer functions (provide them in another Arduino tab or
// linked library):
//   float get_lidar_distance(int angleDeg);          // -1 means invalid
//   float get_line_error();                          // px: left -, right +; NAN if lost
//   float get_ultrasonic_distance(const char* side); // cm, -1 means invalid
//   void  steer(int angleDeg);                       // -45 left .. +45 right
//   void  drive(int speed);                          // -100 reverse .. +100 forward
//
// LiDAR convention assumed by this mission:
//   front=0 deg, left=90 deg, rear=180 deg, right=270 deg.
//
// The implementation intentionally starts stopped. Send 'S' over Serial to
// start, 'R' to reset, or 'X' to latch an emergency stop.

#include <Arduino.h>
#include <math.h>


// ---------------- Hardware API supplied by the vehicle project ----------------
float get_lidar_distance(int angleDeg);
float get_line_error();
float get_ultrasonic_distance(const char* side);
void steer(int angleDeg);
void drive(int speed);


enum ParkingState {
  SEARCHING,
  POSITIONING,
  REVERSING,
  FINISHED
};

enum PositioningPhase {
  ALIGN_REAR_AXLE,
  PREALIGN_LEFT
};

struct LidarCartesianPoint {
  float xRightCm;
  float yBackCm;
};

struct CarCluster {
  int pointCount;
  float xMinCm;
  float xMaxCm;
  float yMinCm;
  float yMaxCm;
};

struct GapObservation {
  bool found;
  float centerAngleDeg;
  float centerXRightCm;
  float centerYBackCm;
  float widthCm;
  float depthX;
  float depthY;
  int carCount;
  bool firstCarFound;
  float firstCarEdgeYBackCm;
};

struct SensorSnapshot {
  float leftCm;
  float rightCm;
  float rearCm;
  float lineErrorPx;
};


// ---------------- Calibration: verify before wheels-on-ground testing ----------
const bool AUTO_START = false;
const float INVALID_DISTANCE = -1.0f;

// Set to 1.0 when get_lidar_distance() returns cm, or 0.1 when it returns mm.
const float LIDAR_TO_CM = 1.0f;

const int FRONT_LIDAR_ANGLE_DEG = 0;
const int REAR_LIDAR_ANGLE_DEG = 180;
const int RIGHT_LIDAR_ANGLE_DEG = 270;

// Search the full vehicle frame so a bordering car remains trackable after the
// ego vehicle rotates and that car crosses into the left half-plane. Only gap
// centers on the vehicle-right are accepted as mission candidates.
const int GAP_SCAN_START_DEG = 0;
const int GAP_SCAN_END_DEG = 358;
const int GAP_SCAN_STEP_DEG = 2;
const int GAP_SAMPLE_COUNT =
    ((GAP_SCAN_END_DEG - GAP_SCAN_START_DEG) / GAP_SCAN_STEP_DEG) + 1;
const int MAX_CAR_CLUSTERS = 8;
const float CAR_ROI_X_MIN_CM = -180.0f;
const float CAR_ROI_X_MAX_CM = 180.0f;
const float CAR_ROI_Y_MIN_CM = -250.0f;
const float CAR_ROI_Y_MAX_CM = 250.0f;
const float CAR_CLUSTER_RADIUS_CM = 25.0f;
const int MIN_CAR_CLUSTER_POINTS = 3;
const float EXPECTED_OBSERVED_GAP_CM = 137.5f;
const float MIN_OBSERVED_GAP_CM = 110.0f;
const float MAX_OBSERVED_GAP_CM = 165.0f;
const float GAP_CENTER_X_MIN_CM = 0.0f;
const int GAP_CONFIRM_CYCLES = 3;
const float GAP_CENTER_MAX_JUMP_CM = 40.0f;
const float GAP_MIN_ORIENTATION_ALIGNMENT = 0.8192f;  // cos(35 deg)

// The gap is centered when its longitudinal vehicle coordinate yBack is zero.
// Negative yBack is ahead of the LiDAR, so the car moves forward. Change this
// sign only if the real drive() direction is wired oppositely.
const int POSITION_DIRECTION_SIGN = 1;
// Align the gap center with the vehicle rear axle before reversing. Negative
// yBack is in front of the LiDAR. The LiDAR is provisionally modeled 30 cm
// behind the rear axle; replace this signed value after measuring the car.
const float POSITION_CENTER_Y_TARGET_CM = -30.0f;
const float POSITION_CENTER_TOLERANCE_CM = 10.0f;
const int POSITION_CONFIRM_CYCLES = 5;

// The first parked car is guaranteed to be followed immediately by the mission
// bay. Slow on its first observation and start the left arc after two scans,
// when its slot-adjacent surface reaches this vehicle-frame yBack coordinate.
const int FIRST_CAR_CONFIRM_CYCLES = 2;
const int FIRST_CAR_APPROACH_SPEED = 10;
const float FIRST_CAR_MIN_X_RIGHT_CM = 50.0f;
const float FIRST_CAR_TURN_TARGET_Y_BACK_CM = -65.0f;

// After the rear axle reaches the longitudinal gap center, move slowly forward
// at maximum left steering. The maneuver stops from LiDAR geometry rather than
// from an assumed motor time: both the slot-depth direction and the entrance
// bearing must align with the vehicle rear direction.
const bool PREALIGN_ENABLED = true;
const int PREALIGN_SPEED = 14;
const int PREALIGN_STEER_DEG = -45;
const unsigned long PREALIGN_STEER_SETTLE_MS = 400;
const unsigned long PREALIGN_TIMEOUT_MS = 6000;
const unsigned long PREALIGN_GAP_ACQUIRE_TIMEOUT_MS = 12000;
const float PREALIGN_SLOT_HEADING_TOLERANCE_DEG = 18.0f;
const float PREALIGN_ENTRY_BEARING_TOLERANCE_DEG = 25.0f;
const float PREALIGN_TARGET_DISTANCE_MIN_CM = 25.0f;
const float PREALIGN_TARGET_DISTANCE_MAX_CM = 220.0f;
const int PREALIGN_CONFIRM_CYCLES = 3;
const float PREALIGN_HEADING_OVERSHOOT_DEG = 25.0f;

const int SEARCH_SPEED = 28;
const int POSITION_SPEED = 16;
const int REVERSE_SPEED = -22;

// Camera is the primary reverse reference. With conventional vehicle steering,
// a positive wheel angle moves the rear of a reversing car toward the right.
const float CAMERA_KP_DEG_PER_PIXEL = 0.12f;
const float ULTRASONIC_KP_DEG_PER_CM = 0.70f;
const float STEERING_FILTER_ALPHA = 0.35f;
const float MAX_CAMERA_ERROR_PX = 250.0f;
const float MAX_SIDE_DISTANCE_CM = 250.0f;
const int MAX_STEER_DEG = 45;
const float STEER_DEADBAND_DEG = 1.0f;

// With no rear ultrasonic sensor, rear LiDAR is used for both completion and
// emergency clearance. A valid distance is non-negative; -1 means failure.
const float PARKING_COMPLETE_REAR_CM = 20.0f;
const int PARKING_COMPLETE_CONFIRM_CYCLES = 3;
const float EMERGENCY_DISTANCE_CM = 10.0f;

const unsigned long CONTROL_INTERVAL_MS = 50;
const unsigned long DEBUG_INTERVAL_MS = 250;
const unsigned long GAP_LOST_STOP_MS = 350;


ParkingState state = SEARCHING;
PositioningPhase positioningPhase = ALIGN_REAR_AXLE;
bool missionEnabled = AUTO_START;
bool emergencyStopLatched = false;

unsigned long lastControlAtMs = 0;
unsigned long lastDebugAtMs = 0;
unsigned long lastGapSeenAtMs = 0;

int gapConfirmCount = 0;
int positionConfirmCount = 0;
int finishConfirmCount = 0;
int prealignConfirmCount = 0;
int firstCarConfirmCount = 0;
unsigned long positioningPhaseStartedAtMs = 0;
unsigned long prealignGapAcquiredAtMs = 0;
bool prealignGapAcquired = false;
float trackedGapCenterDeg = RIGHT_LIDAR_ANGLE_DEG;
float trackedGapCenterXRightCm = 0.0f;
float trackedGapCenterYBackCm = 0.0f;
float trackedGapDepthX = 0.0f;
float trackedGapDepthY = 0.0f;
bool gapDepthInitialized = false;
float filteredSteeringDeg = 0.0f;


// Explicit declarations keep this file valid C++ even without Arduino's
// automatic sketch-prototype generation.
void updateSearching(unsigned long now);
void updatePositioning(unsigned long now);
void updatePrealign(unsigned long now, const GapObservation& gap,
                    bool gapConfirmed);
void updateReversing(const SensorSnapshot& sensors);
GapObservation detectRightParkingGap();
bool confirmGap(const GapObservation& gap, unsigned long now);
void updateTrackedGapDepth(const GapObservation& gap);
SensorSnapshot readSensors();
bool hasEmergencyObstacle(const SensorSnapshot& sensors);
bool isEmergencyDistance(float distanceCm);
void latchEmergencyStop(const __FlashStringHelper* reason);
void stopVehicle();
float readLidarCm(int angleDeg);
int collectCarPoints(LidarCartesianPoint* points);
int clusterCarPoints(const LidarCartesianPoint* points, int pointCount,
                     CarCluster* clusters);
float vehicleAngleFromPoint(float xRightCm, float yBackCm);
bool isValidDistance(float distanceCm);
bool isUsableSideDistance(float distanceCm);
float cameraSteeringCorrection(float lineErrorPx);
float constrainFloat(float value, float minimum, float maximum);
int normalizeAngle(int angleDeg);
float signedAngleDifference(float angleDeg, float referenceDeg);
float lowPassAngle(float previousDeg, float currentDeg, float alpha);
void changeState(ParkingState nextState);
void resetTracking();
void resetMission(bool startImmediately);
void handleSerialCommands();
const __FlashStringHelper* stateName(ParkingState value);
void printDebug(unsigned long now, const SensorSnapshot& sensors);


void setup() {
  Serial.begin(115200);
  stopVehicle();
  resetTracking();
  Serial.println(F("T_PARKING_READY: S=start R=reset X=emergency-stop"));
}

void loop() {
  handleSerialCommands();

  const unsigned long now = millis();
  if (now - lastControlAtMs < CONTROL_INTERVAL_MS) {
    return;
  }
  lastControlAtMs = now;

  if (!missionEnabled || emergencyStopLatched) {
    stopVehicle();
    return;
  }

  SensorSnapshot sensors = readSensors();
  if (hasEmergencyObstacle(sensors)) {
    latchEmergencyStop(F("distance<=10cm"));
    return;
  }

  switch (state) {
    case SEARCHING:
      updateSearching(now);
      break;
    case POSITIONING:
      updatePositioning(now);
      break;
    case REVERSING:
      updateReversing(sensors);
      break;
    case FINISHED:
      stopVehicle();
      break;
  }

  printDebug(now, sensors);
}


// =============================== State machine ================================

void updateSearching(unsigned long now) {
  steer(0);
  drive(SEARCH_SPEED);

  GapObservation gap = detectRightParkingGap();
  const bool gapConfirmed = confirmGap(gap, now);
  if (gap.firstCarFound) {
    firstCarConfirmCount++;
    drive(FIRST_CAR_APPROACH_SPEED);
    const bool turnReached =
        gap.firstCarEdgeYBackCm >= FIRST_CAR_TURN_TARGET_Y_BACK_CM;
    if (PREALIGN_ENABLED &&
        firstCarConfirmCount >= FIRST_CAR_CONFIRM_CYCLES &&
        (turnReached || gapConfirmed)) {
      changeState(POSITIONING);
      positioningPhase = PREALIGN_LEFT;
      positioningPhaseStartedAtMs = now;
      prealignGapAcquired = false;
      prealignConfirmCount = 0;
      steer(PREALIGN_STEER_DEG);
      drive(0);
    }
    return;
  }
  firstCarConfirmCount = 0;
  if (!gapConfirmed) return;

  changeState(POSITIONING);
  positioningPhase = ALIGN_REAR_AXLE;
  positioningPhaseStartedAtMs = now;
  drive(0);
}

void updatePositioning(unsigned long now) {
  GapObservation gap = detectRightParkingGap();
  const bool gapConfirmed = confirmGap(gap, now);
  if (positioningPhase == PREALIGN_LEFT) {
    updatePrealign(now, gap, gapConfirmed);
    return;
  }
  if (!gap.found) {
    positionConfirmCount = 0;
    drive(0);
    steer(0);
    if (now - lastGapSeenAtMs > GAP_LOST_STOP_MS) {
      Serial.println(F("POSITIONING: gap lost, holding stop"));
    }
    return;
  }

  if (hypot(gap.centerXRightCm - trackedGapCenterXRightCm,
            gap.centerYBackCm - trackedGapCenterYBackCm) >
      GAP_CENTER_MAX_JUMP_CM) {
    // Do not replace the confirmed physical slot with another cluster pair.
    positionConfirmCount = 0;
    drive(0);
    steer(0);
    return;
  }

  lastGapSeenAtMs = now;
  updateTrackedGapDepth(gap);
  trackedGapCenterDeg = lowPassAngle(trackedGapCenterDeg, gap.centerAngleDeg, 0.45f);
  trackedGapCenterXRightCm =
      0.45f * gap.centerXRightCm + 0.55f * trackedGapCenterXRightCm;
  trackedGapCenterYBackCm =
      0.45f * gap.centerYBackCm + 0.55f * trackedGapCenterYBackCm;
  const float positionErrorCm =
      trackedGapCenterYBackCm - POSITION_CENTER_Y_TARGET_CM;

  steer(0);
  if (fabs(positionErrorCm) <= POSITION_CENTER_TOLERANCE_CM) {
    drive(0);
    positionConfirmCount++;
    if (positionConfirmCount >= POSITION_CONFIRM_CYCLES) {
      if (PREALIGN_ENABLED) {
        positioningPhase = PREALIGN_LEFT;
        positioningPhaseStartedAtMs = now;
        prealignConfirmCount = 0;
        steer(PREALIGN_STEER_DEG);
      } else {
        changeState(REVERSING);
        steer(0);
      }
      drive(0);
    }
    return;
  }

  positionConfirmCount = 0;
  // yBack < 0 means that the gap is still ahead of the LiDAR.
  const int direction = positionErrorCm < 0.0f ? 1 : -1;
  drive(POSITION_DIRECTION_SIGN * direction * POSITION_SPEED);
}

void updatePrealign(unsigned long now, const GapObservation& gap,
                    bool gapConfirmed) {
  const unsigned long phaseElapsedMs = now - positioningPhaseStartedAtMs;
  if (!gapConfirmed) {
    prealignConfirmCount = 0;
    if (phaseElapsedMs >= PREALIGN_GAP_ACQUIRE_TIMEOUT_MS) {
      Serial.println(F("PREALIGN_ABORT_SECOND_CAR_TIMEOUT"));
      missionEnabled = false;
      stopVehicle();
      return;
    }
    steer(PREALIGN_STEER_DEG);
    drive(phaseElapsedMs < PREALIGN_STEER_SETTLE_MS ? 0 : PREALIGN_SPEED);
    return;
  }
  if (!prealignGapAcquired) {
    prealignGapAcquired = true;
    prealignGapAcquiredAtMs = now;
  }
  const float targetX = gap.centerXRightCm;
  const float targetY = gap.centerYBackCm - POSITION_CENTER_Y_TARGET_CM;
  const float targetDistanceCm = hypot(targetX, targetY);
  const float slotHeadingDeg = atan2(gap.depthX, gap.depthY) * RAD_TO_DEG;
  const float entryBearingDeg = atan2(targetX, targetY) * RAD_TO_DEG;
  const bool directReady =
      fabs(slotHeadingDeg) <= PREALIGN_SLOT_HEADING_TOLERANCE_DEG &&
      fabs(entryBearingDeg) <= PREALIGN_ENTRY_BEARING_TOLERANCE_DEG &&
      targetDistanceCm >= PREALIGN_TARGET_DISTANCE_MIN_CM &&
      targetDistanceCm <= PREALIGN_TARGET_DISTANCE_MAX_CM;
  prealignConfirmCount = directReady ? prealignConfirmCount + 1 : 0;

  if (prealignConfirmCount >= PREALIGN_CONFIRM_CYCLES) {
    Serial.println(F("PREALIGN_DIRECT_REVERSE_READY"));
    changeState(REVERSING);
    steer(0);
    drive(0);
    return;
  }

  const bool overshot = slotHeadingDeg < -PREALIGN_HEADING_OVERSHOOT_DEG;
  if (overshot || now - prealignGapAcquiredAtMs >= PREALIGN_TIMEOUT_MS) {
    Serial.println(
        overshot ? F("PREALIGN_FALLBACK_HEADING_OVERSHOOT")
                 : F("PREALIGN_FALLBACK_TIMEOUT"));
    changeState(REVERSING);
    steer(0);
    drive(0);
    return;
  }

  steer(PREALIGN_STEER_DEG);
  drive(phaseElapsedMs < PREALIGN_STEER_SETTLE_MS ? 0 : PREALIGN_SPEED);
}

void updateReversing(const SensorSnapshot& sensors) {
  // Rear LiDAR replaces the unavailable rear ultrasonic sensor.
  if (isValidDistance(sensors.rearCm) &&
      sensors.rearCm <= PARKING_COMPLETE_REAR_CM) {
    finishConfirmCount++;
  } else {
    finishConfirmCount = 0;
  }

  if (finishConfirmCount >= PARKING_COMPLETE_CONFIRM_CYCLES) {
    changeState(FINISHED);
    stopVehicle();
    return;
  }

  float steeringDeg = cameraSteeringCorrection(sensors.lineErrorPx);

  // Only apply side centering when both side measurements are valid. A -1
  // return is never interpreted as either an obstacle or a large error.
  if (isUsableSideDistance(sensors.leftCm) &&
      isUsableSideDistance(sensors.rightCm)) {
    // Example: right distance is smaller -> negative correction -> steer left.
    const float sideErrorCm = sensors.rightCm - sensors.leftCm;
    steeringDeg += ULTRASONIC_KP_DEG_PER_CM * sideErrorCm;
  }

  steeringDeg = constrainFloat(steeringDeg, -MAX_STEER_DEG, MAX_STEER_DEG);
  filteredSteeringDeg =
      STEERING_FILTER_ALPHA * steeringDeg +
      (1.0f - STEERING_FILTER_ALPHA) * filteredSteeringDeg;

  if (fabs(filteredSteeringDeg) < STEER_DEADBAND_DEG) {
    filteredSteeringDeg = 0.0f;
  }

  steer((int)round(filteredSteeringDeg));
  drive(REVERSE_SPEED);
}


// ============================== Gap detection =================================

GapObservation detectRightParkingGap() {
  LidarCartesianPoint points[GAP_SAMPLE_COUNT];
  CarCluster clusters[MAX_CAR_CLUSTERS];
  const int pointCount = collectCarPoints(points);
  const int clusterCount = clusterCarPoints(points, pointCount, clusters);

  GapObservation best = {
      false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      clusterCount, false, 0.0f};
  float bestWidthError = 1000000.0f;
  float bestCenterDistance = 1000000.0f;

  float firstCarBestCenterY = -1000000.0f;
  for (int index = 0; index < clusterCount; index++) {
    const CarCluster& cluster = clusters[index];
    const float centerX = (cluster.xMinCm + cluster.xMaxCm) * 0.5f;
    const float centerY = (cluster.yMinCm + cluster.yMaxCm) * 0.5f;
    if (centerX >= FIRST_CAR_MIN_X_RIGHT_CM &&
        centerY > firstCarBestCenterY) {
      firstCarBestCenterY = centerY;
      best.firstCarFound = true;
      best.firstCarEdgeYBackCm = cluster.yMinCm;
    }
  }

  for (int firstIndex = 0; firstIndex < clusterCount - 1; firstIndex++) {
    const CarCluster& first = clusters[firstIndex];
    const float firstCenterX = (first.xMinCm + first.xMaxCm) * 0.5f;
    const float firstCenterY = (first.yMinCm + first.yMaxCm) * 0.5f;
    for (int secondIndex = firstIndex + 1;
         secondIndex < clusterCount;
         secondIndex++) {
      const CarCluster& second = clusters[secondIndex];
      const float secondCenterX = (second.xMinCm + second.xMaxCm) * 0.5f;
      const float secondCenterY = (second.yMinCm + second.yMaxCm) * 0.5f;
      const float axisX = secondCenterX - firstCenterX;
      const float axisY = secondCenterY - firstCenterY;
      const float centerDistance = sqrt(axisX * axisX + axisY * axisY);
      if (centerDistance <= 0.001f) {
        continue;
      }
      const float unitX = axisX / centerDistance;
      const float unitY = axisY / centerDistance;
      const float firstRadius =
          fabs(unitX) * (first.xMaxCm - first.xMinCm) * 0.5f +
          fabs(unitY) * (first.yMaxCm - first.yMinCm) * 0.5f;
      const float secondRadius =
          fabs(unitX) * (second.xMaxCm - second.xMinCm) * 0.5f +
          fabs(unitY) * (second.yMaxCm - second.yMinCm) * 0.5f;
      const float gapWidthCm = centerDistance - firstRadius - secondRadius;
      if (gapWidthCm < MIN_OBSERVED_GAP_CM ||
          gapWidthCm > MAX_OBSERVED_GAP_CM) {
        continue;
      }

      const float firstEdgeX = firstCenterX + unitX * firstRadius;
      const float firstEdgeY = firstCenterY + unitY * firstRadius;
      const float secondEdgeX = secondCenterX - unitX * secondRadius;
      const float secondEdgeY = secondCenterY - unitY * secondRadius;
      float centerXRightCm = (firstEdgeX + secondEdgeX) * 0.5f;
      float centerYBackCm = (firstEdgeY + secondEdgeY) * 0.5f;
      if (centerXRightCm < GAP_CENTER_X_MIN_CM) {
        continue;
      }

      // Move the gap reference from the middle of the parked-car returns to
      // their LiDAR-facing surfaces. This is the entrance boundary used by the
      // Python debug view and keeps the inferred bay out of the obstacles.
      float depthX = -unitY;
      float depthY = unitX;
      const float depthAlignment =
          depthX * trackedGapDepthX + depthY * trackedGapDepthY;
      if ((gapDepthInitialized && depthAlignment < 0.0f) ||
          (!gapDepthInitialized &&
           centerXRightCm * depthX + centerYBackCm * depthY < 0.0f)) {
        depthX = -depthX;
        depthY = -depthY;
      }
      if (gapDepthInitialized &&
          depthX * trackedGapDepthX + depthY * trackedGapDepthY <
              GAP_MIN_ORIENTATION_ALIGNMENT) {
        // Normal driving changes the slot angle gradually. A larger one-scan
        // turn indicates another cluster pair, so retain the existing slot.
        continue;
      }
      const float firstDepthRadius =
          fabs(depthX) * (first.xMaxCm - first.xMinCm) * 0.5f +
          fabs(depthY) * (first.yMaxCm - first.yMinCm) * 0.5f;
      const float secondDepthRadius =
          fabs(depthX) * (second.xMaxCm - second.xMinCm) * 0.5f +
          fabs(depthY) * (second.yMaxCm - second.yMinCm) * 0.5f;
      centerXRightCm -= depthX * (firstDepthRadius + secondDepthRadius) * 0.5f;
      centerYBackCm -= depthY * (firstDepthRadius + secondDepthRadius) * 0.5f;

      const float widthError = fabs(gapWidthCm - EXPECTED_OBSERVED_GAP_CM);
      const float centerTrackingDistance =
          hypot(centerXRightCm - trackedGapCenterXRightCm,
                centerYBackCm - trackedGapCenterYBackCm);
      const bool betterCandidate =
          gapDepthInitialized
              ? (centerTrackingDistance < bestCenterDistance ||
                 (fabs(centerTrackingDistance - bestCenterDistance) < 0.1f &&
                  widthError < bestWidthError))
              : widthError < bestWidthError;
      if (betterCandidate) {
        best.found = true;
        best.centerAngleDeg =
            vehicleAngleFromPoint(centerXRightCm, centerYBackCm);
        best.centerXRightCm = centerXRightCm;
        best.centerYBackCm = centerYBackCm;
        best.widthCm = gapWidthCm;
        best.depthX = depthX;
        best.depthY = depthY;
        bestWidthError = widthError;
        bestCenterDistance = centerTrackingDistance;
      }
    }
  }

  return best;
}

int collectCarPoints(LidarCartesianPoint* points) {
  int pointCount = 0;
  for (int angleDeg = GAP_SCAN_START_DEG;
       angleDeg <= GAP_SCAN_END_DEG && pointCount < GAP_SAMPLE_COUNT;
       angleDeg += GAP_SCAN_STEP_DEG) {
    const float distanceCm = readLidarCm(angleDeg);
    if (!isValidDistance(distanceCm)) {
      continue;
    }

    const float bearingRad = angleDeg * DEG_TO_RAD;
    const float xRightCm = -distanceCm * sin(bearingRad);
    const float yBackCm = -distanceCm * cos(bearingRad);
    if (xRightCm < CAR_ROI_X_MIN_CM || xRightCm > CAR_ROI_X_MAX_CM ||
        yBackCm < CAR_ROI_Y_MIN_CM || yBackCm > CAR_ROI_Y_MAX_CM) {
      continue;
    }

    points[pointCount].xRightCm = xRightCm;
    points[pointCount].yBackCm = yBackCm;
    pointCount++;
  }
  return pointCount;
}

int clusterCarPoints(const LidarCartesianPoint* points, int pointCount,
                     CarCluster* clusters) {
  bool visited[GAP_SAMPLE_COUNT] = {false};
  int pending[GAP_SAMPLE_COUNT];
  int clusterCount = 0;
  const float radiusSquared =
      CAR_CLUSTER_RADIUS_CM * CAR_CLUSTER_RADIUS_CM;

  for (int seed = 0;
       seed < pointCount && clusterCount < MAX_CAR_CLUSTERS;
       seed++) {
    if (visited[seed]) {
      continue;
    }

    int pendingRead = 0;
    int pendingWrite = 0;
    pending[pendingWrite++] = seed;
    visited[seed] = true;

    CarCluster cluster;
    cluster.pointCount = 0;
    cluster.xMinCm = cluster.xMaxCm = points[seed].xRightCm;
    cluster.yMinCm = cluster.yMaxCm = points[seed].yBackCm;

    while (pendingRead < pendingWrite) {
      const int current = pending[pendingRead++];
      const float currentX = points[current].xRightCm;
      const float currentY = points[current].yBackCm;
      cluster.pointCount++;
      if (currentX < cluster.xMinCm) cluster.xMinCm = currentX;
      if (currentX > cluster.xMaxCm) cluster.xMaxCm = currentX;
      if (currentY < cluster.yMinCm) cluster.yMinCm = currentY;
      if (currentY > cluster.yMaxCm) cluster.yMaxCm = currentY;

      for (int candidate = 0; candidate < pointCount; candidate++) {
        if (visited[candidate]) {
          continue;
        }
        const float dx = points[candidate].xRightCm - currentX;
        const float dy = points[candidate].yBackCm - currentY;
        if (dx * dx + dy * dy <= radiusSquared) {
          visited[candidate] = true;
          pending[pendingWrite++] = candidate;
        }
      }
    }

    if (cluster.pointCount >= MIN_CAR_CLUSTER_POINTS) {
      clusters[clusterCount++] = cluster;
    }
  }
  return clusterCount;
}

float vehicleAngleFromPoint(float xRightCm, float yBackCm) {
  float angleDeg = atan2(-xRightCm, -yBackCm) * RAD_TO_DEG;
  if (angleDeg < 0.0f) {
    angleDeg += 360.0f;
  }
  return angleDeg;
}

bool confirmGap(const GapObservation& gap, unsigned long now) {
  if (!gap.found) {
    gapConfirmCount = 0;
    return false;
  }

  if (gapConfirmCount > 0 &&
      hypot(gap.centerXRightCm - trackedGapCenterXRightCm,
            gap.centerYBackCm - trackedGapCenterYBackCm) >
          GAP_CENTER_MAX_JUMP_CM) {
    gapConfirmCount = 1;
    trackedGapCenterDeg = gap.centerAngleDeg;
    trackedGapCenterXRightCm = gap.centerXRightCm;
    trackedGapCenterYBackCm = gap.centerYBackCm;
    updateTrackedGapDepth(gap);
  } else {
    trackedGapCenterDeg =
        lowPassAngle(trackedGapCenterDeg, gap.centerAngleDeg, 0.40f);
    trackedGapCenterXRightCm =
        0.40f * gap.centerXRightCm + 0.60f * trackedGapCenterXRightCm;
    trackedGapCenterYBackCm =
        0.40f * gap.centerYBackCm + 0.60f * trackedGapCenterYBackCm;
    updateTrackedGapDepth(gap);
    gapConfirmCount++;
  }

  lastGapSeenAtMs = now;
  return gapConfirmCount >= GAP_CONFIRM_CYCLES;
}

void updateTrackedGapDepth(const GapObservation& gap) {
  float currentX = gap.depthX;
  float currentY = gap.depthY;
  if (gapDepthInitialized &&
      currentX * trackedGapDepthX + currentY * trackedGapDepthY < 0.0f) {
    currentX = -currentX;
    currentY = -currentY;
  }
  if (!gapDepthInitialized) {
    trackedGapDepthX = currentX;
    trackedGapDepthY = currentY;
    gapDepthInitialized = true;
    return;
  }
  trackedGapDepthX = 0.25f * currentX + 0.75f * trackedGapDepthX;
  trackedGapDepthY = 0.25f * currentY + 0.75f * trackedGapDepthY;
  const float length = hypot(trackedGapDepthX, trackedGapDepthY);
  if (length > 0.001f) {
    trackedGapDepthX /= length;
    trackedGapDepthY /= length;
  }
}


// ================================ Safety =======================================

SensorSnapshot readSensors() {
  SensorSnapshot sensors;
  sensors.leftCm = get_ultrasonic_distance("LEFT");
  sensors.rightCm = get_ultrasonic_distance("RIGHT");
  sensors.rearCm = readLidarCm(REAR_LIDAR_ANGLE_DEG);
  sensors.lineErrorPx = get_line_error();
  return sensors;
}

bool hasEmergencyObstacle(const SensorSnapshot& sensors) {
  return isEmergencyDistance(sensors.leftCm) ||
         isEmergencyDistance(sensors.rightCm) ||
         isEmergencyDistance(sensors.rearCm);
}

bool isEmergencyDistance(float distanceCm) {
  return isValidDistance(distanceCm) && distanceCm <= EMERGENCY_DISTANCE_CM;
}

void latchEmergencyStop(const __FlashStringHelper* reason) {
  emergencyStopLatched = true;
  missionEnabled = false;
  stopVehicle();
  Serial.print(F("EMERGENCY_STOP: "));
  Serial.println(reason);
}

void stopVehicle() {
  drive(0);
  steer(0);
}


// ================================ Utilities =====================================

float readLidarCm(int angleDeg) {
  const float raw = get_lidar_distance(normalizeAngle(angleDeg));
  if (raw < 0.0f || !isfinite(raw)) {
    return INVALID_DISTANCE;
  }
  return raw * LIDAR_TO_CM;
}

bool isValidDistance(float distanceCm) {
  return distanceCm >= 0.0f && isfinite(distanceCm);
}

bool isUsableSideDistance(float distanceCm) {
  return isValidDistance(distanceCm) && distanceCm <= MAX_SIDE_DISTANCE_CM;
}

float cameraSteeringCorrection(float lineErrorPx) {
  // -1 px is a valid left error, so camera loss must use NAN rather than -1.
  if (!isfinite(lineErrorPx)) {
    return 0.0f;
  }
  const float clampedError =
      constrainFloat(lineErrorPx, -MAX_CAMERA_ERROR_PX, MAX_CAMERA_ERROR_PX);
  return CAMERA_KP_DEG_PER_PIXEL * clampedError;
}

float constrainFloat(float value, float minimum, float maximum) {
  if (value < minimum) return minimum;
  if (value > maximum) return maximum;
  return value;
}

int normalizeAngle(int angleDeg) {
  int result = angleDeg % 360;
  if (result < 0) result += 360;
  return result;
}

float signedAngleDifference(float angleDeg, float referenceDeg) {
  float difference = fmod(angleDeg - referenceDeg + 540.0f, 360.0f) - 180.0f;
  return difference;
}

float lowPassAngle(float previousDeg, float currentDeg, float alpha) {
  return previousDeg + alpha * signedAngleDifference(currentDeg, previousDeg);
}

void changeState(ParkingState nextState) {
  if (state == nextState) return;
  state = nextState;
  gapConfirmCount = 0;
  positionConfirmCount = 0;
  finishConfirmCount = 0;
  prealignConfirmCount = 0;
  firstCarConfirmCount = 0;
  prealignGapAcquired = false;
  filteredSteeringDeg = 0.0f;
  positioningPhase = ALIGN_REAR_AXLE;
  positioningPhaseStartedAtMs = millis();

  Serial.print(F("STATE -> "));
  Serial.println(stateName(state));
}

void resetTracking() {
  gapConfirmCount = 0;
  positionConfirmCount = 0;
  finishConfirmCount = 0;
  prealignConfirmCount = 0;
  firstCarConfirmCount = 0;
  prealignGapAcquired = false;
  prealignGapAcquiredAtMs = 0;
  trackedGapCenterDeg = RIGHT_LIDAR_ANGLE_DEG;
  trackedGapCenterXRightCm = 0.0f;
  trackedGapCenterYBackCm = 0.0f;
  trackedGapDepthX = 0.0f;
  trackedGapDepthY = 0.0f;
  gapDepthInitialized = false;
  filteredSteeringDeg = 0.0f;
  lastGapSeenAtMs = millis();
}

void resetMission(bool startImmediately) {
  stopVehicle();
  state = SEARCHING;
  emergencyStopLatched = false;
  missionEnabled = startImmediately;
  resetTracking();
  Serial.println(startImmediately ? F("MISSION_STARTED") : F("MISSION_RESET"));
}

void handleSerialCommands() {
  while (Serial.available() > 0) {
    const char command = (char)Serial.read();
    if (command == 'S' || command == 's') {
      resetMission(true);
    } else if (command == 'R' || command == 'r') {
      resetMission(false);
    } else if (command == 'X' || command == 'x') {
      latchEmergencyStop(F("operator"));
    }
  }
}

const __FlashStringHelper* stateName(ParkingState value) {
  switch (value) {
    case SEARCHING: return F("SEARCHING");
    case POSITIONING: return F("POSITIONING");
    case REVERSING: return F("REVERSING");
    case FINISHED: return F("FINISHED");
  }
  return F("UNKNOWN");
}

void printDebug(unsigned long now, const SensorSnapshot& sensors) {
  if (now - lastDebugAtMs < DEBUG_INTERVAL_MS) return;
  lastDebugAtMs = now;

  Serial.print(F("state="));
  Serial.print(stateName(state));
  Serial.print(F(" gapCenter="));
  Serial.print(trackedGapCenterDeg, 1);
  Serial.print(F(" posPhase="));
  Serial.print(positioningPhase == ALIGN_REAR_AXLE ? F("ALIGN") : F("PREALIGN_LEFT"));
  Serial.print(F(" gapYBackCm="));
  Serial.print(trackedGapCenterYBackCm, 1);
  Serial.print(F(" gapXRightCm="));
  Serial.print(trackedGapCenterXRightCm, 1);
  Serial.print(F(" linePx="));
  Serial.print(sensors.lineErrorPx, 1);
  Serial.print(F(" usL="));
  Serial.print(sensors.leftCm, 1);
  Serial.print(F(" usR="));
  Serial.print(sensors.rightCm, 1);
  Serial.print(F(" rearLidar="));
  Serial.print(sensors.rearCm, 1);
  Serial.print(F(" steer="));
  Serial.println(filteredSteeringDeg, 1);
}
