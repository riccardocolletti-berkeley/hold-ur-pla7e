"""Streaming joint-trajectory publisher for the UR ros2_control bridge.

Publishes ``trajectory_msgs/JointTrajectory`` to
``/scaled_joint_trajectory_controller/joint_trajectory``. Each ``send``
builds a single-waypoint trajectory; the scaled trajectory controller
blends the live joint state to the new target over ``trajectory_time_s``,
which is what the UR firmware's URScript loop is designed to consume.
In this lab the ``forward_position_controller`` raw-position path wedges
the URScript loop under hard PID flips; the scaled-trajectory path does
not, so we use it.

The class name and the public ``send`` signature are kept stable so
callers don't need to know which path is active.

Switch the active UR controller once per driver session::

    ros2 control switch_controllers \\
        --activate scaled_joint_trajectory_controller \\
        --deactivate forward_position_controller
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from builtin_interfaces.msg import Duration
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Seconds after construction at which we warn if no subscriber has connected
# to the trajectory topic. A subscriber count of zero means the active ros2
# controller is not the one we expect (scaled_joint_trajectory_controller).
SUBSCRIBER_CHECK_DELAY_S: float = 1.5

# A trajectory point with ``time_from_start = 0`` is rejected by the scaled
# controller; clamp very-small horizons to this lower bound.
_MIN_TIME_FROM_START_S: float = 0.002


class JointTrajectoryClient:
    """Publish ``JointTrajectory`` to the UR scaled trajectory controller.

    The class name is kept for source compatibility with the previous
    Float64MultiArray-based implementation.
    """

    def __init__(
        self,
        node: Node,
        joint_names: Sequence[str],
        joint_state_reader=None,  # accepted for API parity; unused
        topic: str = "/scaled_joint_trajectory_controller/joint_trajectory",
        queue_size: int = 1,
        callback_group: CallbackGroup | None = None,
    ) -> None:
        del joint_state_reader  # the trajectory controller reads /joint_states itself
        self._node = node
        self._joint_names = list(joint_names)
        self._n_joints = len(self._joint_names)
        self._topic = topic

        self._publisher = node.create_publisher(JointTrajectory, topic, queue_size)

        # Sanity check: warn if no one is subscribed after a short delay.
        self._subscriber_check_timer = node.create_timer(
            SUBSCRIBER_CHECK_DELAY_S,
            self._check_subscribers,
            callback_group=callback_group,
        )

        node.get_logger().info(f"Streaming JointTrajectory on {topic}.")

    # ----------------------------------------------------------- send --

    def send(
        self,
        joint_positions: Sequence[float],
        trajectory_time_s: float = 0.0,
        goal_time_tolerance_s: float = 0.0,  # accepted for API parity; unused
    ) -> None:
        """Command the arm to reach ``joint_positions`` in ``trajectory_time_s``.

        Builds a one-point ``JointTrajectory`` with ``time_from_start``
        equal to the requested horizon (clamped to a minimum). The scaled
        controller interpolates internally from the live joint state to
        the target; no software interpolation here, no per-tick velocity
        clamp, no busy-state bookkeeping.
        """
        del goal_time_tolerance_s

        if len(joint_positions) != self._n_joints:
            raise ValueError(
                f"Expected {self._n_joints} joint positions, got {len(joint_positions)}"
            )

        horizon = max(float(trajectory_time_s), _MIN_TIME_FROM_START_S)
        sec = int(horizon)
        nanosec = int((horizon - sec) * 1.0e9)

        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in joint_positions]
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)

        msg = JointTrajectory()
        msg.joint_names = self._joint_names
        msg.points = [point]

        self._publisher.publish(msg)

    # ------------------------------------------------------------- status --

    def is_busy(self) -> bool:
        """The scaled trajectory controller manages its own queue; we never gate on it."""
        return False

    # ---------------------------------------------------- subscriber check --

    def _check_subscribers(self) -> None:
        # Run once.
        timer = self._subscriber_check_timer
        if timer is not None:
            self._subscriber_check_timer = None
            timer.cancel()

        count = self._publisher.get_subscription_count()
        if count == 0:
            self._node.get_logger().error(
                f"No subscribers on {self._topic} after "
                f"{SUBSCRIBER_CHECK_DELAY_S:.1f}s. The robot will not move. "
                f"Activate scaled_joint_trajectory_controller with: "
                f"`ros2 control switch_controllers "
                f"--activate scaled_joint_trajectory_controller "
                f"--deactivate forward_position_controller`."
            )
        else:
            self._node.get_logger().info(
                f"{count} subscriber(s) on {self._topic}; driver looks active."
            )
