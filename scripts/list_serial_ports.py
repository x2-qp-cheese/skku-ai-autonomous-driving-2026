def main() -> int:
    try:
        from serial.tools import list_ports
    except ImportError:
        print("pyserial is not installed. Run: pip install pyserial")
        return 1

    ports = list(list_ports.comports())
    if not ports:
        print("no serial ports found")
        return 0
    for port in ports:
        print("%s\t%s\t%s" % (port.device, port.description, port.hwid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
