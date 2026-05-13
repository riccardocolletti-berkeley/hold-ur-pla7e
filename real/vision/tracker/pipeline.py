"""Real-time tracking pipeline: camera -> markers -> detector -> Kalman -> callback.

Owns the camera handle, the per-frame detection components, the filter,
and the loop driver. Callers integrate by passing a callback to
:meth:`TrackingPipeline.run`; the callback receives a fresh
:class:`ballplate.state.BallState` every frame.

The only direct dependency on ``ballplate`` is the shared ``BallState``
dataclass; the tracking math itself is independent of the rest of the
monorepo.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np

from ballplate.state import BallState
from vision.config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
)
from vision.tracker.ball_detector import BallDetector
from vision.tracker.kalman_filter import BallKalmanFilter, tilt_to_accel
from vision.tracker.table_frame import TableFrame

#: Number of warm-up frames discarded at startup so the camera's auto-exposure
#: settles before the first measurement is fed to the Kalman filter.
_CAMERA_WARMUP_FRAMES = 30


Callback = Callable[[BallState, dict], None]


class TrackingPipeline:
    """Drives the camera loop and produces filtered ball states each frame."""

    def __init__(
        self,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> None:
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera index {CAMERA_INDEX}. "
                "Check the device is connected and not held by another process."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        for _ in range(_CAMERA_WARMUP_FRAMES):
            self.cap.read()

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.table = TableFrame()
        self.ball = BallDetector()
        self.kf = BallKalmanFilter(dt=1.0 / CAMERA_FPS)

        # Latest commanded plate tilt (radians). Updated by the controller via
        # :meth:`set_plate_angles`; consumed by the KF as a control input.
        self._plate_angles: tuple[float, float] = (0.0, 0.0)

        # Lazily set on the first :meth:`process_frame` call so the first dt
        # reflects the real camera period, not the gap between construction
        # and the first frame (which can be seconds during ROS spin-up etc.).
        self._prev_time: float | None = None

    # ----------------------------------------------------------- controller --

    def set_plate_angles(self, theta_x: float, theta_y: float) -> None:
        """Push the latest commanded plate tilt (radians) into the filter input."""
        self._plate_angles = (float(theta_x), float(theta_y))

    # ---------------------------------------------------------------- frame --

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_time: float | None = None,
    ) -> tuple[BallState, dict, np.ndarray]:
        """Process one frame.

        Returns the filtered :class:`BallState`, a debug dict, and the
        (possibly undistorted) frame that the detection actually ran on. The
        caller can hand that frame to :meth:`_draw_debug` so the marker
        overlay matches the corners fed into the homography fit.

        ``frame_time`` should be the wall-clock time at which ``frame`` was
        captured. If omitted, ``time.time()`` is used.
        """
        now = frame_time if frame_time is not None else time.time()
        if self._prev_time is None:
            # First call: we have no reference for dt yet, so use the nominal
            # camera period rather than a wild number.
            dt = 1.0 / CAMERA_FPS
        else:
            dt = now - self._prev_time
        self._prev_time = now

        a_x, a_y = tilt_to_accel(*self._plate_angles)
        self.kf.set_control(a_x, a_y)

        frame = self._undistort(frame)

        markers_ok = self.table.update(frame)
        detection = self.ball.detect(frame)

        debug: dict = {
            "markers_found": self.table.markers_found,
            "dt": dt,
            "ball_pixel": None,
            "ball_radius_px": None,
        }

        # Convert the pixel detection to the table frame (if both available).
        table_pos: tuple[float, float] | None = None
        if detection is not None:
            u, v, radius = detection
            debug["ball_pixel"] = (u, v)
            debug["ball_radius_px"] = radius
            if markers_ok:
                table_pos = self.table.pixel_to_table(u, v)

        # Single decision tree so the KF always advances when initialised,
        # even if the homography returned None this frame for some reason.
        valid = False
        if table_pos is not None:
            mx, my = table_pos
            if not self.kf.initialized:
                self.kf.reset(mx, my)
            else:
                self.kf.predict(dt=dt)
                self.kf.update(mx, my)
            valid = True
        elif self.kf.initialized:
            # Coast on prediction. The control input already carries the latest
            # commanded tilt so the prediction is gravity-aware.
            self.kf.predict(dt=dt)

        if self.kf.initialized:
            x, y = self.kf.get_position()
            vx, vy = self.kf.get_velocity()
        else:
            x = y = vx = vy = 0.0

        ball = BallState(x=x, y=y, vx=vx, vy=vy, timestamp=now, valid=valid)
        return ball, debug, frame

    # -------------------------------------------------------------- loop ---

    def run(self, callback: Callback | None = None, show_debug: bool = True) -> None:
        """Spin the camera loop until the user presses ``q`` in the debug window."""
        print("Starting tracking pipeline. Press 'q' in the debug window to quit.")
        while True:
            ret, frame = self.cap.read()
            # Sample the timestamp as close to the buffer pull as possible so
            # downstream consumers (controller D-term, predictors) see the
            # frame's true age, not the post-processing wall clock.
            frame_time = time.time()
            if not ret:
                print("Camera read failed.")
                break

            ball, debug, processed = self.process_frame(frame, frame_time=frame_time)

            if callback is not None:
                callback(ball, debug)

            if show_debug:
                view = self._draw_debug(processed, ball, debug)
                cv2.imshow("Ball Tracker", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------- viewer --

    def _draw_debug(self, frame: np.ndarray, ball: BallState, debug: dict) -> np.ndarray:
        frame = self.table.draw_debug(frame)

        if debug["ball_pixel"] is not None:
            u, v = debug["ball_pixel"]
            r = int(debug.get("ball_radius_px") or 10)
            cv2.circle(frame, (int(u), int(v)), r, (0, 255, 0), 2)

        info = [
            f"Markers: {debug['markers_found']}",
            f"dt: {debug['dt'] * 1000.0:.1f} ms",
        ]
        if ball.valid or self.kf.initialized:
            info.append(f"Pos: ({ball.x * 1000.0:+6.1f}, {ball.y * 1000.0:+6.1f}) mm")
            info.append(f"Vel: ({ball.vx * 1000.0:+6.1f}, {ball.vy * 1000.0:+6.1f}) mm/s")

        for i, line in enumerate(info):
            cv2.putText(
                frame,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return frame
