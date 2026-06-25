import unittest

from skku_autocar.control.protocol import decode_status, encode_command
from skku_autocar.sensors.lidar import angle_in_window, nearest_distance_mm
from skku_autocar.types import ControlCommand


class ProtocolTest(unittest.TestCase):
    def test_stop_command(self):
        self.assertEqual(encode_command(ControlCommand.stop()), "STOP\n")

    def test_drive_command_is_clipped(self):
        command = ControlCommand(speed=999, steering=-999)
        self.assertEqual(
            encode_command(command, max_speed=160, max_steering=120),
            "DRIVE 160 -120\n",
        )

    def test_status_decode(self):
        self.assertEqual(decode_status("OK READY"), {"type": "OK", "message": "READY"})


class LidarMathTest(unittest.TestCase):
    def test_angle_window_wraps_zero_degrees(self):
        self.assertTrue(angle_in_window(350, 330, 30))
        self.assertTrue(angle_in_window(10, 330, 30))
        self.assertFalse(angle_in_window(180, 330, 30))

    def test_nearest_distance(self):
        points = [(350, 500), (5, 220), (180, 100)]
        self.assertEqual(nearest_distance_mm(points, 330, 30), 220)


if __name__ == "__main__":
    unittest.main()
