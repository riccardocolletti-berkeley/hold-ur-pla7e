"""Real-robot launch for the PID ball-balance controller.

Spawns up to five nodes (the last two are opt-in):

* ``ball_balance_controller``: the closed-loop PID node.
* ``static_transform_publisher``: anchors the plate frame to the wrist
  using the adapter geometry from ``ball_on_plate/config/hardware.yaml``
  (shared with the simulator).
* ``ball_balance_visualizer``: republishes ``/ball_state`` and the active
  reference trajectory as ``visualization_msgs/MarkerArray`` for RViz.
  Only when ``rviz:=true``.
* ``rviz2``: bundled RViz config with every display pre-wired. Only when
  ``rviz:=true``.
* (out of this file's scope) ``robot_state_publisher`` must be running so
  the rest of the TF tree (base_link -> tool0) is available; the UR
  driver launch usually starts it.
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
    """Return the absolute path of ``ball_balance_controller``'s source config dir.

    ``colcon build --symlink-install`` symlinks files under ``build/`` but
    *copies* ``data_files`` into ``install/share/`` for ament_python packages,
    so the install share path's mtime is frozen at build time and the live
    YAML watcher cannot see source edits if pointed there. We resolve the
    source dir from ``COLCON_PREFIX_PATH`` (set by ``install/setup.bash``)
    so the watcher polls a path whose mtime advances on every editor save.
    """
    prefix = os.environ.get("COLCON_PREFIX_PATH", "")
    install_dir = prefix.split(os.pathsep, 1)[0] if prefix else ""
    workspace = os.path.dirname(install_dir) if install_dir else os.getcwd()
    return os.path.join(workspace, "src", "ball_balance_controller", "config")


def _adapter_static_tf_args() -> list[str]:
    """Return CLI args for ``static_transform_publisher`` from the shared hardware spec."""
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
        default_value=PathJoinSubstitution([pkg, "config", "pid_params.yaml"]),
        description="Path to the controller params YAML.",
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

    # Re-assert the UR speed slider to ``hardware.yaml::arm.speed_slider_fraction``.
    # The lab `enable_comms` bringup sets it to 0.2 during boot and ramps
    # to 1.0 once the robot reaches RUNNING; anything below 1.0 silently
    # throttles every motion command (including ours). The service comes
    # from the io_and_status_controller in the UR ros2_control driver.
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

    # Pass the *source* YAML paths (not the install/share copies) so the
    # live-reload watcher in each node sees mtime advance whenever the
    # operator saves a YAML in their editor. The ``params_file`` argument
    # above still controls the YAML rclpy parses for *initial* parameter
    # values; the watcher path is independent so a user override of
    # ``params_file:=`` does not mistarget the watcher to a different file.
    src_config = _source_config_dir()
    live_reload_params = {
        "params_file_path": os.path.join(src_config, "pid_params.yaml"),
        "reference_file_path": os.path.join(src_config, "reference.yaml"),
    }

    controller = Node(
        package="ball_balance_controller",
        executable="ball_balance_controller",
        name="ball_balance_controller",
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
            controller,
            visualizer,
            rviz,
        ]
    )
