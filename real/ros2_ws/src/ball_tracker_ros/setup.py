from setuptools import find_packages, setup

package_name = "ball_tracker_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/tracker_launch.py"]),
        (f"share/{package_name}/config", ["config/tracker_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Riccardo Colletti",
    maintainer_email="riccardo_colletti@berkeley.edu",
    description="ROS 2 wrapper around the standalone vision tracker.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "tracker_node = ball_tracker_ros.tracker_node:main",
        ],
    },
)
