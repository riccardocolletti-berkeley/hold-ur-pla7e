"""Visualization-only launch: see what the system *thinks* is going on.

Brings up everything needed to inspect the ball-balance stack in RViz
without firing up the real UR driver:

* ``robot_state_publisher``: broadcasts the bundled UR7e URDF.
* ``joint_state_publisher_gui``: manual joint sliders so you can pose
  the arm in RViz to sanity-check the plate / wrist frames at any joint
  configuration.
* ``static_transform_publisher``: anchors ``plate_center`` to ``tool0``
  using the adapter geometry from ``ballplate.hardware``.
* ``ball_balance_visualizer``: republishes ``/ball_state`` and the active
  reference trajectory as ``visualization_msgs/MarkerArray``.
* ``rviz2``: opens the bundled config with the displays pre-wired.

When the real bring-up is running, the UR driver supplies
``robot_state_publisher`` and ``/joint_states`` already; use
``balance.launch.py rviz:=true`` instead. That variant skips the
duplicates and only adds the visualizer + RViz on top of the live system.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from ballplate.hardware import default_path as _hardware_default_path
from ballplate.hardware import load as _load_hardware


def _adapter_static_tf_args() -> list[str]:
    """CLI args for static_transform_publisher (tool0 -> plate_center)."""
    hw = _load_hardware(_hardware_default_path())
    px, py, pz = (str(v) for v in hw.adapter.position)
    rx, ry, rz = (str(v) for v in hw.adapter.orientation)  # roll, pitch, yaw
    return [px, py, pz, rz, ry, rx, "tool0", "plate_center"]


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("ball_balance_controller")
    urdf_path = PathJoinSubstitution([pkg, "urdf", "ur7e.urdf"])
    rviz_config = PathJoinSubstitution([pkg, "rviz", "balance.rviz"])

    use_jsp_gui_arg = DeclareLaunchArgument(
        "use_joint_state_publisher_gui",
        default_value="true",
        description="Run joint_state_publisher_gui so you can pose the arm with sliders.",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Whether to launch rviz2 with the bundled config.",
    )
    reference_shape_arg = DeclareLaunchArgument(
        "reference_shape",
        default_value="circle",
        description="Reference shape: stationary | circle | figure8 | drawn",
    )
    reference_radius_arg = DeclareLaunchArgument(
        "reference_radius_m",
        default_value="0.05",
    )
    reference_period_arg = DeclareLaunchArgument(
        "reference_period_s",
        default_value="20.0",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        # `xacro` would normally process the URDF; this one is plain so a
        # `cat` substitution is enough.
        parameters=[{"robot_description": Command(["cat ", urdf_path])}],
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
        condition=_if_true("use_joint_state_publisher_gui"),
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="adapter_static_tf",
        arguments=_adapter_static_tf_args(),
        output="screen",
    )

    visualizer = Node(
        package="ball_balance_controller",
        executable="ball_balance_visualizer",
        name="ball_balance_visualizer",
        output="screen",
        parameters=[
            {
                "reference_shape": LaunchConfiguration("reference_shape"),
                "reference_radius_m": LaunchConfiguration("reference_radius_m"),
                "reference_period_s": LaunchConfiguration("reference_period_s"),
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
        condition=_if_true("rviz"),
    )

    return LaunchDescription(
        [
            use_jsp_gui_arg,
            use_rviz_arg,
            reference_shape_arg,
            reference_radius_arg,
            reference_period_arg,
            robot_state_publisher,
            joint_state_publisher_gui,
            static_tf,
            visualizer,
            rviz,
        ]
    )


def _if_true(arg_name: str):
    """Helper: only spawn the node when LaunchConfiguration(arg_name) == 'true'."""
    from launch.conditions import IfCondition

    return IfCondition(LaunchConfiguration(arg_name))
