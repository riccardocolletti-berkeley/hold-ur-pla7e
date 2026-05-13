import os
from glob import glob

from setuptools import find_packages, setup

package_name = "ball_balance_controller"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "urdf"),
            glob("urdf/*.urdf"),
        ),
        (
            os.path.join("share", package_name, "rviz"),
            glob("rviz/*.rviz"),
        ),
    ],
    install_requires=["setuptools", "ikpy"],
    zip_safe=True,
    maintainer="Riccardo Colletti",
    maintainer_email="riccardo_colletti@berkeley.edu",
    description="Closed-loop ball-on-plate controllers (PID and MPC) for the UR arm.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "ball_balance_controller = ball_balance_controller.controller:main",
            "ball_balance_mpc        = ball_balance_controller.mpc_node:main",
            "ball_balance_visualizer = ball_balance_controller.visualizer:main",
            "mock_ball_publisher     = ball_balance_controller.mock_ball_publisher:main",
        ],
    },
)
