import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from skku_autocar.runtime.parking_app import (
    DashboardVideoRecorder,
    compose_parking_dashboard,
    dashboard_recording_enabled,
    parse_args,
    timestamped_dashboard_path,
)


class ParkingDashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest("OpenCV/NumPy unavailable") from exc
        cls.cv2 = cv2
        cls.np = np

    def test_live_camera_records_by_default(self):
        args = parse_args([])
        self.assertEqual(args.record_dashboard, "auto")
        self.assertEqual(args.parking_record_dir, "data/parking")
        self.assertEqual(args.dashboard_record_fps, 10.0)
        self.assertTrue(dashboard_recording_enabled("auto", is_video=False))
        self.assertFalse(dashboard_recording_enabled("auto", is_video=True))
        self.assertTrue(dashboard_recording_enabled("on", is_video=True))
        self.assertFalse(dashboard_recording_enabled("off", is_video=False))

    def test_timestamp_path_avoids_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 7, 21, 14, 30, 45)
            first = timestamped_dashboard_path(directory, now)
            self.assertEqual(first.name, "20260721_143045.mp4")
            first.touch()
            second = timestamped_dashboard_path(directory, now)
            self.assertEqual(second.name, "20260721_143045_01.mp4")

    def test_shared_dashboard_is_1280_by_720(self):
        rear = self.np.zeros((480, 640, 3), dtype=self.np.uint8)
        bev = self.np.zeros((640, 640, 3), dtype=self.np.uint8)
        lidar = self.np.zeros((600, 600, 3), dtype=self.np.uint8)
        dashboard = compose_parking_dashboard(
            self.cv2,
            self.np,
            rear,
            bev,
            lidar,
            "LIVE idle",
            (255, 255, 255),
            ("REC=test.mp4",),
        )
        self.assertEqual(dashboard.shape, (720, 1280, 3))

    def test_recorder_writes_readable_dashboard_mp4(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dashboard.mp4"
            recorder = DashboardVideoRecorder(self.cv2, output, fps=10.0)
            frame = self.np.zeros((720, 1280, 3), dtype=self.np.uint8)
            recorder.write(frame, 0.0)
            recorder.write(frame, 0.25)
            recorder.close()

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
            capture = self.cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(self.cv2.CAP_PROP_FRAME_WIDTH)), 1280)
                self.assertEqual(int(capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT)), 720)
                self.assertGreaterEqual(
                    int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT)),
                    3,
                )
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
