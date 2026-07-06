import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


DEFAULT_OUTPUT_DIR = Path("data/raw/recordings")
WINDOW_NAME = "Camera Recorder"


def open_camera(index, width, height, backend):
    if backend == "auto":
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        elif sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(index)
    elif backend == "default":
        cap = cv2.VideoCapture(index)
    elif backend == "dshow":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif backend == "avfoundation":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        raise ValueError("unknown backend: %s" % backend)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def make_session_dir(output_dir):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / session_id
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir, images_dir


def draw_status(frame, recording, written_frames, saved_images, fps):
    status = "REC" if recording else "PAUSE"
    color = (0, 0, 255) if recording else (0, 255, 255)
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(
        frame,
        "frames=%d images=%d fps=%.1f" % (written_frames, saved_images, fps),
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "space: pause/resume | s: snapshot | q: quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--backend", choices=("auto", "default", "dshow", "avfoundation"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--frame-interval", type=int, default=10)
    parser.add_argument("--start-paused", action="store_true")
    args = parser.parse_args()

    cap = open_camera(args.camera_index, args.width, args.height, args.backend)
    if not cap.isOpened():
        print("camera index %s could not be opened" % args.camera_index)
        return 1

    session_id, session_dir, images_dir = make_session_dir(args.output_dir)
    video_path = session_dir / ("%s.mp4" % session_id)

    ok, first_frame = cap.read()
    if not ok:
        print("failed to read first frame")
        cap.release()
        return 1

    frame_h, frame_w = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*args.fourcc),
        args.fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        print("failed to open video writer: %s" % video_path)
        cap.release()
        return 1

    recording = not args.start_paused
    written_frames = 0
    saved_images = 0
    prev_time = time.time()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, frame_w, frame_h)

    print("recording session:", session_dir)
    print("video:", video_path)
    print("images:", images_dir)
    print("space: pause/resume | s: snapshot | q: quit")

    pending_frame = first_frame
    try:
        while True:
            if pending_frame is None:
                ok, frame = cap.read()
                if not ok:
                    print("failed to read frame")
                    break
            else:
                frame = pending_frame
                pending_frame = None

            raw_frame = frame.copy()
            if recording:
                writer.write(raw_frame)
                written_frames += 1
                if args.frame_interval > 0 and written_frames % args.frame_interval == 0:
                    image_path = images_dir / ("frame_%06d.jpg" % written_frames)
                    cv2.imwrite(str(image_path), raw_frame)
                    saved_images += 1

            now = time.time()
            measured_fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            preview = frame.copy()
            draw_status(preview, recording, written_frames, saved_images, measured_fps)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
                print("recording:", "ON" if recording else "OFF")
            elif key == ord("s"):
                image_path = images_dir / ("snapshot_%06d.jpg" % saved_images)
                cv2.imwrite(str(image_path), raw_frame)
                saved_images += 1
                print("saved snapshot:", image_path)
    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("saved video:", video_path)
        print("saved images:", saved_images)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
