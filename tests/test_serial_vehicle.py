import unittest
from types import SimpleNamespace

from skku_autocar.control.serial_vehicle import (
    find_arduino_port,
    is_ready_line,
    parse_ultrasonic_line,
)
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

    def test_ultrasonic_stream_counts_as_ready_serial_output(self):
        self.assertTrue(is_ready_line("T_PARKING_READY: S=start"))
        self.assertTrue(is_ready_line("PONG"))
        self.assertTrue(is_ready_line("US FC=0 FR=0 FL=0 SR=0 SL=0"))
        self.assertFalse(is_ready_line("OK DRIVE"))

    def test_newest_sample_wins(self):
        sample = newest_ultrasonic_sample([
            "US FR=1 FL=2 SR=300 SL=400",
            "OK DRIVE",
            "US FR=1 FL=2 SR=500 SL=600",
        ])

        self.assertEqual(sample.side_right_mm, 500.0)
        self.assertEqual(sample.side_left_mm, 600.0)

    def test_arduino_auto_detect_prefers_usbmodem_over_lidar_usbserial(self):
        ports = (
            SimpleNamespace(
                device="/dev/cu.usbserial-1130",
                description="USB Serial",
                hwid="VID:PID=1A86:7523",
                manufacturer="wch.cn",
            ),
            SimpleNamespace(
                device="/dev/cu.usbmodem11101",
                description="Arduino USB Modem",
                hwid="VID:PID=2341:0043",
                manufacturer="Arduino",
            ),
        )

        selected = find_arduino_port("auto", ports=ports)

        self.assertEqual(selected, "/dev/cu.usbmodem11101")

    def test_missing_dev_placeholder_falls_back_to_detected_arduino(self):
        ports = (
            SimpleNamespace(
                device="/dev/cu.usbmodem11101",
                description="Arduino USB Modem",
                hwid="VID:PID=2341:0043",
                manufacturer="Arduino",
            ),
        )

        selected = find_arduino_port(
            "/dev/tty.usbmodem-ARDUINO",
            ports=ports,
            exists=lambda _: False,
        )

        self.assertEqual(selected, "/dev/cu.usbmodem11101")


if __name__ == "__main__":
    unittest.main()
