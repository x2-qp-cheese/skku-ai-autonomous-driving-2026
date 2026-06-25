from typing import Optional

from ..config import ControlConfig, LidarConfig
from ..types import ControlCommand, LaneEstimate, ObstacleEstimate


class LaneFollower:
    def __init__(self, control: ControlConfig, lidar: Optional[LidarConfig] = None):
        self.control = control
        self.lidar = lidar

    def plan(
        self,
        lane: LaneEstimate,
        obstacle: Optional[ObstacleEstimate] = None,
    ) -> ControlCommand:
        if self._must_stop_for_obstacle(obstacle):
            return ControlCommand.stop("front_obstacle")
        if lane.confidence <= 0.05 and self.control.lane_lost_brake:
            return ControlCommand.stop("lane_lost")

        steering = int(round(-self.control.lane_kp * lane.center_offset_norm))
        command = ControlCommand(
            speed=self.control.base_speed,
            steering=steering,
            brake=False,
            reason="lane_follow",
        )
        return command.clipped(self.control.max_speed, self.control.max_steering)

    def _must_stop_for_obstacle(self, obstacle: Optional[ObstacleEstimate]) -> bool:
        if obstacle is None or obstacle.front_mm is None or self.lidar is None:
            return False
        return obstacle.front_mm <= self.lidar.stop_distance_mm
