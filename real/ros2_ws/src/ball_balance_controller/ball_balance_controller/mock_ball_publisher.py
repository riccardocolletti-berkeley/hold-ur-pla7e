"""Synthetic ``/ball_state`` publisher for offline controller tests.

Publishes a circling ball with the analytically-correct velocity, so the
PID controller can run without a real camera or robot. Position is in the
table frame (metres); velocity is the analytical derivative. Optional
Gaussian noise on position and velocity is available for stress testing.
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from ball_tracker_msgs.msg import BallState
from rclpy.node import Node


class MockBallPublisher(Node):
    """Drives a virtual ball around a circle at a configurable radius and period."""

    def __init__(self) -> None:
        super().__init__("mock_ball_publisher")

        self.declare_parameter("radius_m", 0.010)
        self.declare_parameter("period_s", 8.0)
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("add_noise", False)
        self.declare_parameter("noise_pos_m", 0.001)
        self.declare_parameter("noise_vel_mps", 0.005)

        self._pub = self.create_publisher(BallState, "/ball_state", 10)
        self._t0 = time.time()

        hz = float(self.get_parameter("publish_hz").value)
        self.create_timer(1.0 / hz, self._publish)
        self.get_logger().info(
            f"MockBallPublisher: r={self.get_parameter('radius_m').value} m, "
            f"T={self.get_parameter('period_s').value} s, rate={hz} Hz"
        )

    def _publish(self) -> None:
        t = time.time() - self._t0
        r = float(self.get_parameter("radius_m").value)
        period = float(self.get_parameter("period_s").value)
        omega = 2.0 * math.pi / period

        x = r * math.cos(omega * t)
        y = r * math.sin(omega * t)
        vx = -r * omega * math.sin(omega * t)
        vy = r * omega * math.cos(omega * t)

        if self.get_parameter("add_noise").value:
            sp = float(self.get_parameter("noise_pos_m").value)
            sv = float(self.get_parameter("noise_vel_mps").value)
            x += np.random.normal(0.0, sp)
            y += np.random.normal(0.0, sp)
            vx += np.random.normal(0.0, sv)
            vy += np.random.normal(0.0, sv)

        msg = BallState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "plate_center"
        msg.ball_found = True
        msg.tracking_valid = True
        msg.markers_found = 4
        msg.x = float(x)
        msg.y = float(y)
        msg.vx = float(vx)
        msg.vy = float(vy)

        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockBallPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
