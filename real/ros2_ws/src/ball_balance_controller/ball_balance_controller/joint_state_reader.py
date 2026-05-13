"""Cache the most recent ``/joint_states`` snapshot in canonical UR order.

The UR driver publishes ``/joint_states`` asynchronously; the control
loop wants a deterministic snapshot indexed in the canonical joint
order (shoulder_pan first, wrist_3 last). The subscription QoS matches
the publisher's ``qos_profile_sensor_data`` (BEST_EFFORT, KEEP_LAST 5)
so DDS endpoint compatibility holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock
from typing import Optional

import numpy as np
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState

#: Canonical UR joint order. ``/joint_states`` may publish in any order;
#: the controller's command stream must use this sequence (it is also
#: what ros2_control's trajectory controller expects on the command topic).
DEFAULT_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class JointStateReader:
    """Subscribes to ``/joint_states`` and exposes the latest snapshot atomically."""

    def __init__(
        self,
        node: Node,
        joint_names: Sequence[str] = DEFAULT_JOINT_NAMES,
        topic: str = "/joint_states",
        qos: QoSProfile = qos_profile_sensor_data,
        callback_group: CallbackGroup | None = None,
    ):
        self._node = node
        self._joint_names = tuple(joint_names)
        self._lock = Lock()
        self._latest: np.ndarray | None = None
        self._latest_stamp = None

        self._sub = node.create_subscription(
            JointState,
            topic,
            self._on_msg,
            qos,
            callback_group=callback_group,
        )

    # ------------------------------------------------------------- ingress --

    def _on_msg(self, msg: JointState) -> None:
        if not msg.position:
            return
        positions = np.zeros(len(self._joint_names), dtype=float)
        name_to_pos = dict(zip(msg.name, msg.position, strict=False))
        for i, name in enumerate(self._joint_names):
            if name not in name_to_pos:
                # Drop any /joint_states message that does not cover the full
                # canonical UR set so the controller never operates on a
                # partial snapshot.
                return
            positions[i] = float(name_to_pos[name])
        with self._lock:
            self._latest = positions
            self._latest_stamp = msg.header.stamp

    # ---------------------------------------------------------- accessors --

    def latest(self) -> np.ndarray | None:
        """Return a copy of the most recent snapshot, or ``None`` if none yet."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def latest_stamp(self):
        """Return the header stamp of the most recent snapshot (or ``None``)."""
        with self._lock:
            return self._latest_stamp
