"""Offline launch for the PID ball-balance controller.

Runs the controller alongside ``mock_ball_publisher``, which feeds a
synthetic ``/ball_state`` describing a ball circling at fixed radius.
Useful for exercising the PID logic without a real camera. Joint
trajectories are still published on
``/scaled_joint_trajectory_controller/joint_trajectory``; without a
ros2_control consumer (real driver or a fake) the messages just go
nowhere and the controller still comes up.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("ball_balance_controller")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([pkg, "config", "pid_params.yaml"]),
        description="Path to the controller params YAML.",
    )

    # The controller refuses to start without a URDF, so we expose the
    # same CLI override as the real-robot launch. Users can point at any
    # URDF (or override the empty default in pid_params.yaml) without
    # editing the YAML for an offline test.
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value="",
        description="Absolute path to the UR URDF (overrides params YAML when non-empty).",
    )

    mock_ball = Node(
        package="ball_balance_controller",
        executable="mock_ball_publisher",
        name="mock_ball_publisher",
        output="screen",
        parameters=[
            {
                "radius_m": 0.010,
                "period_s": 8.0,
                "publish_hz": 30.0,
            }
        ],
    )

    controller = Node(
        package="ball_balance_controller",
        executable="ball_balance_controller",
        name="ball_balance_controller",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"urdf_path": LaunchConfiguration("urdf_path")},
        ],
    )

    return LaunchDescription(
        [
            params_file_arg,
            urdf_path_arg,
            mock_ball,
            controller,
        ]
    )
