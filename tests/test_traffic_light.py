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


def mask_band(top, bottom):
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[top:bottom, :] = 255
    return (mask,)


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
        far_crosswalk = mask_band(4, 18)
        near_crosswalk = mask_band(28, 39)

        first = self.controller.update(red, masks, far_crosswalk)
        self.controller.update(red, masks, near_crosswalk)
        second = self.controller.update(red, masks, near_crosswalk)
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

        controller.update(near_red, near_masks)
        near = controller.update(near_red, near_masks)
        self.assertTrue(near.contact)
        self.assertTrue(near.stop_latched)
        self.assertTrue(controller.apply(self.drive, running=True).brake)

    def test_crosswalk_mask_controls_contact_independently_of_light_mask(self):
        red, near_light_masks = light_frame((0, 0, 255), top=18, bottom=38)
        far_crosswalk = mask_band(4, 18)
        near_crosswalk = mask_band(18, 36)

        self.controller.update(red, near_light_masks, far_crosswalk)
        far = self.controller.update(red, near_light_masks, far_crosswalk)

        self.assertEqual(far.state, "red")
        self.assertFalse(far.contact)
        self.assertEqual(self.controller.apply(self.drive, running=True), self.drive)

        self.controller.update(red, (), near_crosswalk)
        contact = self.controller.update(red, (), near_crosswalk)

        self.assertFalse(contact.detected)
        self.assertTrue(contact.contact)
        self.assertTrue(contact.stop_latched)
        self.assertTrue(self.controller.apply(self.drive, running=True).brake)

    def test_red_can_confirm_before_green_with_separate_frame_count(self):
        controller = TrafficLightController(
            TrafficLightConfig(
                confirm_frames=3,
                red_confirm_frames=1,
                min_color_pixels=5,
                stop_line_y_ratio=0.70,
            )
        )
        red, red_masks = light_frame((0, 0, 255))
        green, green_masks = light_frame((0, 255, 0))
        far_crosswalk = mask_band(4, 18)
        crosswalk_contact = mask_band(28, 39)

        controller.update(red, red_masks, far_crosswalk)
        controller.update(red, red_masks, crosswalk_contact)
        red_observation = controller.update(red, red_masks, crosswalk_contact)
        first_green = controller.update(green, green_masks, crosswalk_contact)
        second_green = controller.update(green, green_masks, crosswalk_contact)
        third_green = controller.update(green, green_masks, crosswalk_contact)

        self.assertEqual(red_observation.state, "red")
        self.assertTrue(red_observation.stop_latched)
        self.assertEqual(first_green.state, "red")
        self.assertEqual(second_green.state, "red")
        self.assertEqual(third_green.state, "green")
        self.assertFalse(third_green.stop_latched)

    def test_mask_first_seen_below_line_does_not_create_contact(self):
        controller = TrafficLightController(
            TrafficLightConfig(confirm_frames=1, min_color_pixels=5, stop_line_y_ratio=0.70)
        )
        red, masks = light_frame((0, 0, 255))
        appeared_below_line = mask_band(28, 39)

        controller.update(red, masks, appeared_below_line)
        observation = controller.update(red, masks, appeared_below_line)

        self.assertEqual(observation.state, "red")
        self.assertFalse(observation.contact)
        self.assertFalse(observation.stop_latched)
        self.assertEqual(controller.apply(self.drive, running=True), self.drive)

    def test_one_frame_bottom_jump_does_not_create_contact(self):
        controller = TrafficLightController(
            TrafficLightConfig(confirm_frames=1, min_color_pixels=5, stop_line_y_ratio=0.70)
        )
        red, masks = light_frame((0, 0, 255))

        controller.update(red, masks, mask_band(4, 18))
        jumped = controller.update(red, masks, mask_band(28, 39))
        recovered = controller.update(red, masks, mask_band(4, 18))

        self.assertFalse(jumped.contact)
        self.assertFalse(recovered.contact)
        self.assertFalse(recovered.stop_latched)

    def test_red_contact_stop_stays_latched_if_mask_is_lost(self):
        red, masks = light_frame((0, 0, 255), top=18, bottom=36)
        self.controller.update(red, masks, mask_band(4, 18))
        self.controller.update(red, masks, mask_band(28, 39))
        self.controller.update(red, masks, mask_band(28, 39))

        lost = self.controller.update(red, ())

        self.assertTrue(lost.stop_latched)
        self.assertTrue(self.controller.apply(self.drive, running=True).brake)

    def test_confirmed_green_releases_red_latch(self):
        red, masks = light_frame((0, 0, 255))
        green, green_masks = light_frame((0, 255, 0))
        crosswalk_contact = mask_band(28, 39)
        self.controller.update(red, masks, mask_band(4, 18))
        self.controller.update(red, masks, crosswalk_contact)
        self.controller.update(red, masks, crosswalk_contact)

        first_green = self.controller.update(green, green_masks, crosswalk_contact)
        second_green = self.controller.update(green, green_masks, crosswalk_contact)

        self.assertEqual(first_green.state, "red")
        self.assertEqual(second_green.state, "green")
        self.assertTrue(second_green.contact)
        self.assertFalse(second_green.stop_latched)
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
