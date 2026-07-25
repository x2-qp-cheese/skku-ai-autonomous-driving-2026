import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Install competition behavior fixes before the runtime imports and constructs
# obstacle/lane-change controllers.
from skku_autocar.competition_patch import install_competition_patch

install_competition_patch()

from skku_autocar.runtime.yolo_drive_app import main


if __name__ == "__main__":
    main()
