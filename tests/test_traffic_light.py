import unittest

import numpy as np

from skku_autocar.perception.traffic_light import TrafficLightConfig, TrafficLightController
from skku_autocar.perception.yolo_lane import YoloLaneConfig, YoloLaneSegmenter
from skku_autocar.types import ControlCommand


def light_frame(bgr, top=10, bottom=30):
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[top:bottom, 10:30] = bgr
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[top:bottom, 10:30] = 255
    return frame, (mask,)


class TrafficLightControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = TrafficLightController(
            TrafficLightConfig(confirm_frames=2, min_color_pixels=5, stop_line_y_ratio=0.70)
        )
        self.drive = ControlCommand(speed=100, steering=12, brake=False, reason="lane")

    def test_light_label_is_preserved_as_traffic_light_class(self):
        segmenter = object.__new__(YoloLaneSegmenter)
        segmenter.config = YoloLaneConfig()

        self.assertEqual(segmenter._class_kind("light"), "light")

    def test_red_stops_after_confirmation_and_stays_latched_when_lost(self):
        red, masks = light_frame((0, 0, 255))

        first = self.controller.update(red, masks)
        second = self.controller.update(red, masks)
        lost = self.controller.update(red, ())

        self.assertEqual(first.state, "unknown")
        self.assertEqual(second.state, "red")
        self.assertEqual(lost.state, "red")
        stopped = self.controller.apply(self.drive, running=True)
        self.assertTrue(stopped.brake)
        self.assertEqual(stopped.speed, 0)
        self.assertEqual(stopped.reason, "traffic_light:red_contact")

    def test_confirmed_red_keeps_driving_until_mask_touches_contact_line(self):
        controller = TrafficLightController(
            TrafficLightConfig(confirm_frames=2, min_color_pixels=5, stop_line_y_ratio=0.82)
        )
        far_red, far_masks = light_frame((0, 0, 255), top=4, bottom=18)
        near_red, near_masks = light_frame((0, 0, 255), top=18, bottom=36)

        controller.update(far_red, far_masks)
        far = controller.update(far_red, far_masks)

        self.assertEqual(far.state, "red")
        self.assertFalse(far.contact)
        self.assertEqual(controller.apply(self.drive, running=True), self.drive)

        near = controller.update(near_red, near_masks)
        self.assertTrue(near.contact)
        self.assertTrue(near.stop_latched)
        self.assertTrue(controller.apply(self.drive, running=True).brake)

    def test_red_contact_stop_stays_latched_if_mask_is_lost(self):
        red, masks = light_frame((0, 0, 255), top=18, bottom=36)
        self.controller.update(red, masks)
        self.controller.update(red, masks)

        lost = self.controller.update(red, ())

        self.assertTrue(lost.stop_latched)
        self.assertTrue(self.controller.apply(self.drive, running=True).brake)

    def test_confirmed_green_releases_red_latch(self):
        red, masks = light_frame((0, 0, 255))
        green, green_masks = light_frame((0, 255, 0))
        self.controller.update(red, masks)
        self.controller.update(red, masks)

        first_green = self.controller.update(green, green_masks)
        second_green = self.controller.update(green, green_masks)

        self.assertEqual(first_green.state, "red")
        self.assertEqual(second_green.state, "green")
        self.assertEqual(self.controller.apply(self.drive, running=True), self.drive)

    def test_ambiguous_color_cannot_release_red(self):
        red, masks = light_frame((0, 0, 255))
        yellow, yellow_masks = light_frame((0, 255, 255))
        self.controller.update(red, masks)
        self.controller.update(red, masks)

        observation = self.controller.update(yellow, yellow_masks)

        self.assertEqual(observation.candidate, "unknown")
        self.assertEqual(observation.state, "red")


if __name__ == "__main__":
    unittest.main()
