"""Launch file for the ball-tracker ROS 2 node.

Loads two parameter files:

  - ``config/tracker_params.yaml``     (this package, tracker-specific knobs)
  - ``config/reference.yaml``          (ball_balance_controller, shared reference
                                        trajectory used by the controller, the
                                        RViz visualizer, and this tracker's
                                        OpenCV overlay)

Override either per-run:

    ros2 launch ball_tracker_ros tracker_launch.py \\
        params_file:=/path/to/your/tracker_params.yaml \\
        reference_file:=/path/to/your/reference.yaml

The node logs every loaded value at startup so a stale ``install/share`` copy
of either YAML is visible immediately.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _workspace_root() -> str:
    """Return the colcon workspace root (parent of ``install/``)."""
    prefix = os.environ.get("COLCON_PREFIX_PATH", "")
    install_dir = prefix.split(os.pathsep, 1)[0] if prefix else ""
    return os.path.dirname(install_dir) if install_dir else os.getcwd()


def _source_config(pkg: str, fname: str) -> str:
    """Source ``config/<fname>`` for a package in this workspace.

    The live YAML watcher in ``tracker_node`` polls this path. We resolve
    to source (not install/share) because ``colcon build --symlink-install``
    physically copies ``data_files`` into ``install/share`` for ament_python
    packages, freezing the install copy's mtime at build time.
    """
    return os.path.join(_workspace_root(), "src", pkg, "config", fname)


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("ball_tracker_ros")
    controller_pkg = FindPackageShare("ball_balance_controller")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([pkg, "config", "tracker_params.yaml"]),
        description="Path to the tracker params YAML.",
    )
    reference_file_arg = DeclareLaunchArgument(
        "reference_file",
        default_value=PathJoinSubstitution([controller_pkg, "config", "reference.yaml"]),
        description="Path to the shared reference-trajectory YAML.",
    )

    live_reload_params = {
        "params_file_path": _source_config("ball_tracker_ros", "tracker_params.yaml"),
        "reference_file_path": _source_config("ball_balance_controller", "reference.yaml"),
    }

    tracker = Node(
        package="ball_tracker_ros",
        executable="tracker_node",
        name="ball_tracker",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            LaunchConfiguration("reference_file"),
            live_reload_params,
        ],
    )

    return LaunchDescription(
        [
            params_file_arg,
            reference_file_arg,
            tracker,
        ]
    )
