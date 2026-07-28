from dataclasses import replace
import unittest

from skku_autocar.config import PaperControllerConfig
from skku_autocar.perception.rear_lidar import (
    RearLidarObservation,
    TangentPair,
)
from skku_autocar.planning.paper_controller import PaperParkingController
from skku_autocar.types import ParkingState


def observation(
    timestamp: float,
    *,
    right_vehicle: bool = True,
    right_raw: bool = True,
    track_id: int = 1,
    min_y_back_mm=None,
    pair: TangentPair = TangentPair(),
) -> RearLidarObservation:
    return RearLidarObservation(
        timestamp=timestamp,
        valid=True,
        right_vehicle_present=right_vehicle,
        right_vehicle_raw=right_raw,
        right_vehicle_track_id=track_id,
        right_vehicle_min_y_back_mm=min_y_back_mm,
        pair=pair,
    )


def pair_at(angle_deg: float) -> TangentPair:
    return TangentPair(
        valid=True,
        angle_a_deg=angle_deg - 3.0,
        angle_b_deg=angle_deg + 3.0,
        dist_a_mm=1000.0,
        dist_b_mm=1000.0,
        angle_bisector_deg=angle_deg,
        reason="test_pair",
    )


class FirstVehicleEndEntryTest(unittest.TestCase):
    def make_controller(self) -> PaperParkingController:
        config = replace(
            PaperControllerConfig(),
            first_vehicle_end_trigger_y_mm=30.0,
            forward_speed=50,
            parallel_forward_speed=35,
            actuator_steering_offset=-28,
            pair_filter_scans=3,
            entry_start_angle_deg=45.0,
        )
        controller = PaperParkingController(config)
        controller.start()
        return controller

    def test_configured_y_crossing_starts_slow_full_left_immediately(self):
        controller = self.make_controller()

        controller.update(
            observation(1.0, min_y_back_mm=-20.0),
            1.0,
        )
        command = controller.update(
            observation(2.0, min_y_back_mm=-1.3),
            2.0,
        )
        self.assertEqual(controller.state, ParkingState.SEARCH_OPEN_SPACE)
        self.assertEqual(command.steering, -28)

        command = controller.update(
            observation(
                3.0,
                min_y_back_mm=16.9,
                pair=pair_at(58.0),
            ),
            3.0,
        )

        self.assertEqual(controller.state, ParkingState.SEARCH_OPEN_SPACE)
        self.assertEqual(command.steering, -28)

        command = controller.update(
            observation(
                4.0,
                min_y_back_mm=31.0,
                pair=pair_at(58.0),
            ),
            4.0,
        )

        self.assertEqual(controller.state, ParkingState.PREALIGN_LEFT)
        self.assertEqual((command.speed, command.steering), (35, -150))
        self.assertEqual(
            command.reason,
            "first_vehicle_end_left_acquiring_ab",
        )

    def test_three_fresh_pairs_hand_off_to_existing_45_degree_logic(self):
        controller = self.make_controller()
        controller.update(
            observation(1.0, min_y_back_mm=-20.0),
            1.0,
        )
        controller.update(
            observation(2.0, min_y_back_mm=-1.0),
            2.0,
        )
        controller.update(
            observation(
                3.0,
                min_y_back_mm=31.0,
                pair=pair_at(58.0),
            ),
            3.0,
        )

        for timestamp in (4.0, 5.0):
            command = controller.update(
                observation(timestamp, pair=pair_at(58.0)),
                timestamp,
            )
            self.assertEqual((command.speed, command.steering), (35, -150))

        command = controller.update(
            observation(6.0, pair=pair_at(58.0)),
            6.0,
        )
        self.assertEqual((command.speed, command.steering), (50, -150))
        self.assertEqual(controller.state, ParkingState.PREALIGN_LEFT)

        controller.update(
            observation(7.0, pair=pair_at(44.0)),
            7.0,
        )
        controller.update(
            observation(8.0, pair=pair_at(44.0)),
            8.0,
        )
        self.assertEqual(controller.state, ParkingState.REVERSE_ALIGN)

    def test_crossing_requires_same_confirmed_raw_track(self):
        controller = self.make_controller()
        controller.update(
            observation(1.0, min_y_back_mm=-20.0),
            1.0,
        )
        controller.update(
            observation(2.0, min_y_back_mm=-1.0),
            2.0,
        )
        command = controller.update(
            observation(
                3.0,
                track_id=2,
                min_y_back_mm=31.0,
            ),
            3.0,
        )

        self.assertEqual(controller.state, ParkingState.SEARCH_OPEN_SPACE)
        self.assertEqual(command.steering, -28)


if __name__ == "__main__":
    unittest.main()
