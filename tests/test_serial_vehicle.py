import unittest

from skku_autocar.control.serial_vehicle import parse_ultrasonic_line
from skku_autocar.runtime.parking_app import newest_ultrasonic_sample


class SerialVehicleUltrasonicTest(unittest.TestCase):
    def test_parses_full_ultrasonic_stream_line_and_rejects_zero_echo(self):
        sample = parse_ultrasonic_line("US FR=410 FL=0 SR=725 SL=680")

        self.assertIsNotNone(sample)
        self.assertEqual(sample.front_right_mm, 410.0)
        self.assertIsNone(sample.front_left_mm)
        self.assertEqual(sample.side_right_mm, 725.0)
        self.assertEqual(sample.side_left_mm, 680.0)

    def test_non_ultrasonic_line_is_ignored(self):
        self.assertIsNone(parse_ultrasonic_line("OK DRIVE"))

    def test_newest_sample_wins(self):
        sample = newest_ultrasonic_sample([
            "US FR=1 FL=2 SR=300 SL=400",
            "OK DRIVE",
            "US FR=1 FL=2 SR=500 SL=600",
        ])

        self.assertEqual(sample.side_right_mm, 500.0)
        self.assertEqual(sample.side_left_mm, 600.0)


if __name__ == "__main__":
    unittest.main()
