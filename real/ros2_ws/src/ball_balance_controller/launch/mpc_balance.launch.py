"""Real-robot launch file for the ball-balance MPC controller.

Sibling of ``balance.launch.py`` (the PID launch). Spawns the same set of
support nodes (static TF, optional RViz + visualizer) but drives the arm
from the MPC node instead of the PID node.

Run alongside the existing UR driver bring-up and ball tracker:

    ros2 launch ball_balance_controller mpc_balance.launch.py
    ros2 launch ball_balance_controller mpc_balance.launch.py rviz:=true
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from ballplate.hardware import default_path as _hardware_default_path
from ballplate.hardware import load as _load_hardware


def _source_config_dir() -> str:
    """Source config/ for ball_balance_controller. See balance.launch.py."""
    prefix = os.environ.get("COLCON_PREFIX_PATH", "")
    install_dir = prefix.split(os.pathsep, 1)[0] if prefix else ""
    workspace = os.path.dirname(install_dir) if install_dir else os.getcwd()
    return os.path.join(workspace, "src", "ball_balance_controller", "config")


def _adapter_static_tf_args() -> list[str]:
    hw = _load_hardware(_hardware_default_path())
    px, py, pz = (str(v) for v in hw.adapter.position)
    rx, ry, rz = (str(v) for v in hw.adapter.orientation)  # roll, pitch, yaw
    return [px, py, pz, rz, ry, rx, "tool0", "plate_center"]


def _speed_slider_fraction() -> str:
    return str(_load_hardware(_hardware_default_path()).arm.speed_slider_fraction)


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("ball_balance_controller")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([pkg, "config", "mpc_params.yaml"]),
        description="Path to the MPC controller params YAML.",
    )
    urdf_path_arg = DeclareLaunchArgument(
        "urdf_path",
        default_value=PathJoinSubstitution([pkg, "urdf", "ur7e.urdf"]),
        description="Path to the UR URDF used for FK / Jacobian (defaults to the bundled UR7e).",
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Logging level: debug | info | warn | error",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Spawn ball_balance_visualizer + rviz2 alongside the controller.",
    )

    set_speed_slider = ExecuteProcess(
        cmd=[
            "ros2",
            "service",
            "call",
            "/io_and_status_controller/set_speed_slider",
            "ur_msgs/srv/SetSpeedSliderFraction",
            "{speed_slider_fraction: " + _speed_slider_fraction() + "}",
        ],
        output="screen",
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="adapter_static_tf",
        arguments=_adapter_static_tf_args(),
        output="screen",
    )

    reference_file = PathJoinSubstitution([pkg, "config", "reference.yaml"])

    # Watch the *source* YAMLs (not the install/share copies, which
    # ament_python freezes at build time even with --symlink-install).
    # See ``balance.launch.py`` for the full rationale. Built before the
    # node so the MPC controller can pick it up too.
    src_config = _source_config_dir()
    live_reload_params = {
        "params_file_path": os.path.join(src_config, "mpc_params.yaml"),
        "reference_file_path": os.path.join(src_config, "reference.yaml"),
    }

    mpc_controller = Node(
        package="ball_balance_controller",
        executable="ball_balance_mpc",
        name="ball_balance_mpc",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            reference_file,
            {"urdf_path": LaunchConfiguration("urdf_path")},
            live_reload_params,
        ],
        arguments=[
            "--ros-args",
            "--log-level",
            LaunchConfiguration("log_level"),
        ],
    )

    rviz_condition = IfCondition(LaunchConfiguration("rviz"))

    visualizer = Node(
        package="ball_balance_controller",
        executable="ball_balance_visualizer",
        name="ball_balance_visualizer",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            reference_file,
            live_reload_params,
        ],
        condition=rviz_condition,
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", PathJoinSubstitution([pkg, "rviz", "balance.rviz"])],
        output="screen",
        condition=rviz_condition,
    )

    return LaunchDescription(
        [
            params_file_arg,
            urdf_path_arg,
            log_level_arg,
            rviz_arg,
            set_speed_slider,
            static_tf,
            mpc_controller,
            visualizer,
            rviz,
        ]
    )
