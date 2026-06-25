import unittest

from skku_autocar.config import config_from_dict


class ConfigTest(unittest.TestCase):
    def test_defaults_are_applied_for_missing_sections(self):
        config = config_from_dict({})
        self.assertEqual(config.camera.width, 1280)
        self.assertEqual(config.mission.mode, "time_trial")

    def test_rejects_non_object_section(self):
        with self.assertRaises(ValueError):
            config_from_dict({"camera": 1})


if __name__ == "__main__":
    unittest.main()
