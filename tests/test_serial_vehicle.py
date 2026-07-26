import unittest
from types import SimpleNamespace

from skku_autocar.control.serial_vehicle import _score_port


def port(device, description, hwid, vid):
    return SimpleNamespace(
        device=device,
        description=description,
        hwid=hwid,
        manufacturer="",
        vid=vid,
    )


class SerialPortSelectionTest(unittest.TestCase):
    def test_arduino_usbmodem_outranks_ch340_usbserial(self):
        arduino = port(
            "/dev/cu.usbmodem21101",
            "IOUSBHostDevice",
            "USB VID:PID=2341:0042",
            0x2341,
        )
        ch340 = port(
            "/dev/cu.usbserial-2130",
            "USB Serial",
            "USB VID:PID=1A86:7523",
            0x1A86,
        )

        self.assertGreater(_score_port(arduino), _score_port(ch340))

    def test_debug_and_bluetooth_ports_are_rejected(self):
        debug = port("/dev/cu.debug-console", "debug console", "n/a", None)
        bluetooth = port(
            "/dev/cu.Bluetooth-Incoming-Port",
            "Bluetooth",
            "n/a",
            None,
        )

        self.assertEqual(_score_port(debug), 0)
        self.assertEqual(_score_port(bluetooth), 0)


if __name__ == "__main__":
    unittest.main()
