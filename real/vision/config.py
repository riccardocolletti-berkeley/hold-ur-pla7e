"""Vision-side configuration: thin loader on top of ``config.yaml``.

The static, user-tunable values (camera, ArUco markers, HSV bands, Kalman
noise) live in :file:`config.yaml` next to this module. Plate, ball, and
physics constants instead come from the shared hardware spec at
``ball_on_plate/config/hardware.yaml``, loaded through
:mod:`ballplate.hardware`.

Everything is exposed as plain module-level names so the rest of the vision
package imports them without any boilerplate::

    from vision.config import CAMERA_INDEX, MARKER_POSITIONS_TABLE
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ballplate.hardware import default_path as _hardware_default_path
from ballplate.hardware import load as _load_hardware

_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _THIS_DIR / "config.yaml"
_data = yaml.safe_load(_CONFIG_PATH.read_text())


# --- Camera ----------------------------------------------------------------
CAMERA_INDEX = int(_data["camera"]["index"])
CAMERA_WIDTH = int(_data["camera"]["width"])
CAMERA_HEIGHT = int(_data["camera"]["height"])
CAMERA_FPS = int(_data["camera"]["fps"])


# --- ArUco markers ---------------------------------------------------------
ARUCO_DICT_NAME = str(_data["aruco"]["dictionary"])
MARKER_IDS = [int(m["id"]) for m in _data["aruco"]["markers"]]

#: Marker centres in the table frame (metres). Missing markers are dropped
#: from the homography fit silently.
MARKER_POSITIONS_TABLE = {
    int(m["id"]): np.array([float(m["x"]), float(m["y"])]) for m in _data["aruco"]["markers"]
}


# --- Ball detection (HSV) --------------------------------------------------
# uint8 dtype matches the HSV image depth, which `cv2.inRange` requires.
BALL_HSV_LOWER = np.array(_data["ball_detection"]["hsv_lower"], dtype=np.uint8)
BALL_HSV_UPPER = np.array(_data["ball_detection"]["hsv_upper"], dtype=np.uint8)
BALL_MIN_RADIUS_PX = int(_data["ball_detection"]["min_radius_px"])
BALL_MAX_RADIUS_PX = int(_data["ball_detection"]["max_radius_px"])


# --- Kalman filter ---------------------------------------------------------
KF_PROCESS_NOISE = float(_data["kalman"]["process_noise"])
KF_MEASUREMENT_NOISE = float(_data["kalman"]["measurement_noise"])


# --- Shared hardware / physics ---------------------------------------------
_HW = _load_hardware(_hardware_default_path())

GRAVITY_M_S2 = _HW.physics.gravity
#: 5/7 for a uniform solid sphere, 1.0 for pure sliding.
ROLLING_FACTOR = _HW.physics.rolling_factor
#: Physical ball radius (metres). Useful for parallax compensation.
BALL_RADIUS_M = _HW.ball.radius
