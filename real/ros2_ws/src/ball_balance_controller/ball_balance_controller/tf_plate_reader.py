"""TF lookup helper for the world-to-plate orientation.

The controller needs the rotation matrix ``R_world_plate`` every step to
project the PID's plate-frame command into a world-frame angular velocity.
ROS already maintains this transform via ``robot_state_publisher`` plus the
launch-file static TF that anchors the plate to the wrist; this helper just
performs the lookup and returns the 3×3 matrix.
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


def _quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion to a 3×3 rotation matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


class PlateTfReader:
    """Subscribes to TF and exposes the latest world-frame plate orientation."""

    def __init__(
        self,
        node: Node,
        world_frame: str = "base_link",
        plate_frame: str = "plate_center",
        buffer_seconds: float = 5.0,
    ):
        self._node = node
        self.world_frame = world_frame
        self.plate_frame = plate_frame
        self._buffer = Buffer(cache_time=Duration(seconds=buffer_seconds))
        # tf2_ros humble's TransformListener does not accept callback_group;
        # it falls back to the node's default group, which is a separate
        # MutuallyExclusiveCallbackGroup from the controller's _cb_control,
        # so it still runs on its own executor thread.
        self._listener = TransformListener(self._buffer, node)

    def latest_rotation(self, timeout_s: float = 0.0) -> np.ndarray | None:
        """Return the latest 3×3 ``R_world_plate``, or ``None`` if not available."""
        try:
            transform: TransformStamped = self._buffer.lookup_transform(
                self.world_frame,
                self.plate_frame,
                Time(),  # latest available
                timeout=Duration(seconds=timeout_s),
            )
        except Exception:
            return None

        q = transform.transform.rotation
        return _quat_to_rotation(q.x, q.y, q.z, q.w)
