import argparse
import sys
import time

import serial
from serial.tools import list_ports


def find_serial_port() -> str:
    preferred_ports = []
    fallback_ports = []
    for port in list_ports.comports():
        device = port.device
        if sys.platform == "darwin" and not device.startswith("/dev/cu."):
            continue

        fallback_ports.append(device)
        details = "%s %s %s" % (device, port.description, port.hwid)
        details = details.lower()
        if any(token in details for token in ("arduino", "usbmodem", "usbserial", "ch340")):
            preferred_ports.append(device)

    if preferred_ports:
        return preferred_ports[0]
    if fallback_ports:
        return fallback_ports[0]
    raise RuntimeError("serial port not found")


def main() -> int:
    parser = argparse.ArgumentParser(description="Arduino serial stability test using PING only.")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    port = args.port or find_serial_port()
    print("opening:", port)

    with serial.Serial(port, args.baud, timeout=0.2) as conn:
        time.sleep(2.0)
        start = time.time()
        next_ping = start
        count = 0
        while time.time() - start < args.duration:
            now = time.time()
            if now >= next_ping:
                conn.write(b"PING\n")
                count += 1
                next_ping = now + args.interval

            while conn.in_waiting > 0:
                line = conn.readline().decode(errors="ignore").strip()
                if line:
                    print("[ARDUINO]", line)

            time.sleep(0.01)

    print("done: sent %d PING commands" % count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
