import argparse

from skku_autocar.config import load_config
from skku_autocar.perception.bev import BevConfig
from skku_autocar.sensors.camera import Camera


def main() -> int:
    parser = argparse.ArgumentParser(description="Check camera index, resolution and BEV source ROI")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    config = load_config(args.config)
    bev = BevConfig()

    import cv2
    import numpy as np

    with Camera(config.camera) as camera:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("failed to read frame")
                return 1
            height, width = frame.shape[:2]
            points = np.array(
                [
                    (bev.src_top_left[0] * width, bev.src_top_left[1] * height),
                    (bev.src_top_right[0] * width, bev.src_top_right[1] * height),
                    (bev.src_bottom_right[0] * width, bev.src_bottom_right[1] * height),
                    (bev.src_bottom_left[0] * width, bev.src_bottom_left[1] * height),
                ],
                dtype=np.int32,
            )
            cv2.polylines(frame, [points], True, (0, 255, 255), 2)
            cv2.putText(
                frame, "%dx%d camera=%s" % (width, height, config.camera.index),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
            )
            cv2.imshow("Camera Check (q=quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
