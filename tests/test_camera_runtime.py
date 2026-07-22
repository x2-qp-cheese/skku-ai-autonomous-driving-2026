import unittest

from skku_autocar.runtime.yolo_drive_app import (
    enforce_camera_contract,
    parse_args,
)


class CameraRuntimeContractTest(unittest.TestCase):
    def test_live_camera_must_match_calibrated_resolution(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "requested 1280x720 but received 640x480",
        ):
            enforce_camera_contract(
                (480, 640, 3),
                1280,
                720,
                live_camera=True,
                policy="strict",
            )

    def test_matching_live_camera_is_accepted(self):
        enforce_camera_contract(
            (720, 1280, 3),
            1280,
            720,
            live_camera=True,
            policy="strict",
        )

    def test_video_replay_can_use_its_native_resolution(self):
        enforce_camera_contract(
            (480, 640, 3),
            1280,
            720,
            live_camera=False,
            policy="strict",
        )

    def test_allow_policy_is_explicitly_available_for_calibration(self):
        enforce_camera_contract(
            (480, 640, 3),
            1280,
            720,
            live_camera=True,
            policy="allow",
        )

    def test_competition_default_is_strict(self):
        self.assertEqual(parse_args([]).camera_resolution_policy, "strict")

    def test_competition_default_uses_external_front_camera(self):
        self.assertEqual(parse_args([]).camera, "1")


if __name__ == "__main__":
    unittest.main()
