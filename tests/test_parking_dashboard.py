import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from skku_autocar.estimation.parking_lidar import LidarParkingObservation
from skku_autocar.planning.t_parking_planner import ParkingState
from skku_autocar.runtime.parking_app import (
    DashboardVideoRecorder,
    camera_source_candidates,
    compose_parking_dashboard,
    dashboard_recording_enabled,
    draw_camera_alignment_line,
    is_auto_camera_source,
    parse_args,
    slot_lock_requested,
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
        self.assertIsNone(args.source)
        self.assertIsNone(args.front_source)
        self.assertEqual(args.record_dashboard, "auto")
        self.assertEqual(args.parking_record_dir, "data/parking")
        self.assertEqual(args.dashboard_record_fps, 10.0)
        self.assertTrue(dashboard_recording_enabled("auto", is_video=False))
        self.assertFalse(dashboard_recording_enabled("auto", is_video=True))
        self.assertTrue(dashboard_recording_enabled("on", is_video=True))
        self.assertFalse(dashboard_recording_enabled("off", is_video=False))

    def test_source_auto_must_be_explicit(self):
        args = parse_args(["--source", "auto", "--front-source", "auto"])
        self.assertEqual(args.source, "auto")
        self.assertEqual(args.front_source, "auto")

    def test_slot_lock_starts_on_pair_candidate_before_confirmation(self):
        candidate = LidarParkingObservation(
            valid=True,
            gap_found=True,
            gap_pair_observed=True,
            gap_confirmed=False,
        )

        self.assertTrue(slot_lock_requested(ParkingState.SEARCH_CARS, candidate))

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
        front = self.np.full((360, 640, 3), 255, dtype=self.np.uint8)
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
            front_display=front,
        )
        self.assertEqual(dashboard.shape, (720, 1280, 3))
        self.assertGreater(int(dashboard[330:450, 580:840].sum()), 0)

    def test_camera_alignment_line_marks_center_column(self):
        image = self.np.zeros((20, 30, 3), dtype=self.np.uint8)

        draw_camera_alignment_line(self.cv2, image)

        center = image[:, 15]
        left = image[:, 10]
        self.assertGreater(int(center.sum()), 0)
        self.assertEqual(int(left.sum()), 0)

    def test_auto_camera_source_candidates_prioritize_configured_rear_index(self):
        self.assertTrue(is_auto_camera_source("auto"))
        self.assertTrue(is_auto_camera_source(" AUTO "))
        self.assertFalse(is_auto_camera_source("1"))
        self.assertEqual(camera_source_candidates(1, max_index=3), [1, 0, 2, 3])
        self.assertEqual(camera_source_candidates("auto", max_index=2), [0, 1, 2])

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
