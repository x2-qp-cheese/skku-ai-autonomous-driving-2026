from typing import Any, Optional, Tuple

from ..config import CameraConfig


class Camera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap: Optional[Any] = None

    def open(self) -> None:
        cv2 = _load_cv2()
        cap = cv2.VideoCapture(self.config.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        if not cap.isOpened():
            raise RuntimeError("camera index %s could not be opened" % self.config.index)
        self._cap = cap

    def read(self) -> Tuple[bool, Any]:
        cap = self._require_open()
        return cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _require_open(self):
        if self._cap is None:
            raise RuntimeError("camera is not open")
        return self._cap

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for camera access") from exc
    return cv2
