import unittest

from skku_autocar.sensors.ultrasonic import (
    UltrasonicConfig,
    UltrasonicFilter,
    UltrasonicSnapshot,
    parse_ultrasonic_line,
)


class UltrasonicFilterTest(unittest.TestCase):
    def test_parser_reads_full_and_partial_sensor_lines(self):
        self.assertEqual(
            parse_ultrasonic_line("US FC=800 FR=810 FL=820 SR=0 SL=500"),
            {"FC": 800, "FR": 810, "FL": 820, "SR": 0, "SL": 500},
        )
        self.assertEqual(parse_ultrasonic_line("US SL=300"), {"SL": 300})
        self.assertEqual(parse_ultrasonic_line("OK DRIVE"), {})

    def test_median_filter_rejects_single_distance_spike(self):
        sensor = UltrasonicFilter(UltrasonicConfig(median_window=3))
        sensor.update_lines(["US FC=900 FR=900 FL=900 SR=0 SL=500"], 1.0)
        sensor.update_lines(["US FC=100 FR=900 FL=900 SR=0 SL=500"], 1.1)
        sensor.update_lines(["US FC=900 FR=900 FL=900 SR=0 SL=500"], 1.2)

        self.assertEqual(sensor.snapshot(1.2).fc, 900)

    def test_first_positive_echo_bypasses_zero_filled_median_window(self):
        sensor = UltrasonicFilter(UltrasonicConfig(median_window=3))
        sensor.update_lines(["US FC=0"], 1.0)
        sensor.update_lines(["US FC=0"], 1.1)

        sensor.update_lines(["US FC=1900"], 1.2)

        self.assertEqual(sensor.snapshot(1.2).fc, 1900)

    def test_closing_distance_is_reported_without_median_lag(self):
        sensor = UltrasonicFilter(UltrasonicConfig(median_window=3))
        sensor.update_lines(["US FC=2100"], 1.0)
        sensor.update_lines(["US FC=1800"], 1.1)

        self.assertEqual(sensor.snapshot(1.1).fc, 1800)

    def test_invalid_short_reading_does_not_refresh_sensor(self):
        sensor = UltrasonicFilter(UltrasonicConfig(max_age_seconds=0.5))
        sensor.update_lines(["US FC=800"], 1.0)
        sensor.update_lines(["US FC=9"], 1.4)

        snapshot = sensor.snapshot(1.6)

        self.assertNotIn("FC", snapshot.fresh_keys)

    def test_zero_is_a_fresh_no_echo_value(self):
        sensor = UltrasonicFilter(UltrasonicConfig(median_window=1))
        sensor.update_lines(["US FC=0 FR=0 FL=0 SR=0 SL=0"], 1.0)

        snapshot = sensor.snapshot(1.1)

        self.assertTrue(snapshot.front_fresh)
        self.assertIsNone(snapshot.front_min_mm)
        self.assertTrue(snapshot.side_clear(-1, 300))
        self.assertTrue(snapshot.side_clear(1, 300))

    def test_front_quorum_uses_only_fresh_sensor_values(self):
        snapshot = UltrasonicSnapshot(
            fc=900,
            fr=1100,
            fl=100,
            fresh_keys=("FC", "FR"),
        )

        self.assertFalse(snapshot.front_fresh)
        self.assertTrue(snapshot.front_ready(2))
        self.assertEqual(snapshot.front_fresh_count, 2)
        self.assertEqual(snapshot.front_min_mm, 900)
        self.assertEqual(snapshot.front_close_count(1000), 1)


if __name__ == "__main__":
    unittest.main()
