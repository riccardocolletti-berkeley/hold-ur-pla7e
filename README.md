# Hold UR Pla7e: ball-on-plate control with a UR7e arm

### University of California, Berkeley · EECS 206A Introduction to Robotics

- Riccardo Colletti · EECS MEng · [riccardo_colletti@berkeley.edu](mailto:riccardo_colletti@berkeley.edu)
- Alexander Remmerie · EECS MEng
- Xutao Ma · EECS PhD
- Alex Gasca Rosas · EECS MS

## Overview

A Universal Robots UR7e arm balances a 40 mm ping-pong ball on a 45.7
cm acrylic plate. The same domain library is shared between a MuJoCo
simulator (training and benchmarking) and the real-arm bring-up.

Three plate-frame controllers are implemented and interchangeable:

- **PID**. One decoupled loop per plate axis maps the tracking error to
  a scalar tilt command.
- **MPC**. A linear receding-horizon controller, one per axis, plans
  the next `N` tilt commands at each tick and applies only the first.
- **Residual PPO** (PID + RL). A bounded neural correction trained with
  Proximal Policy Optimization is summed on top of the deployed PID
  command, so it fine-tunes the baseline without replacing it.
  Training runs in MuJoCo with seven physical and sensing parameters
  randomised per episode.

All three controllers emit a 2-D virtual tilt command in the plate
frame, then go through the same downstream wiring: rotational Jacobian
of the wrist, joint-trajectory smoothing, and dispatch. On the real
arm the controller drives the `ros2_control` trajectory bridge against
an overhead RealSense tracker that estimates the ball pose with a
tilt-aware Kalman filter. A browser tool lets the operator sketch
reference trajectories on an iPad and dispatch them to either backend.

The repository is split into four Python packages plus a ROS 2 colcon
workspace. The dependency direction is one-way inward: the pure-Python
`ballplate` package knows nothing about MuJoCo, ROS 2, or OpenCV;
`sim`, `real`, and `designer` import from it and never the other way
around. Plate, ball, and physics constants live once in
`config/hardware.yaml`, loaded into typed dataclasses by every backend.

## Layout

```
.
├── pyproject.toml         uv workspace + ruff + pytest configuration
├── uv.lock                workspace-wide lockfile
├── config/
│   ├── hardware.yaml      plate / ball / physics, loaded by sim, real, designer
│   └── control.yaml       cross-backend control settings (joint locking)
│
├── ballplate/             pure-Python domain library
│   └── src/ballplate/     state, trajectories, PID, MPC, RL inference, safety
│
├── sim/                   MuJoCo simulator (depends on ballplate)
│   └── src/sim/           scene, adapters, controllers, RL training
│
├── real/                  real-robot deployment (depends on ballplate)
│   ├── vision/            standalone OpenCV tracker (no ROS)
│   └── ros2_ws/           ball_tracker_ros, ball_balance_controller, msgs
│
└── designer/              Flask + vanilla-JS trajectory designer
    └── static/            iPad-friendly drawing canvas + run dispatcher
```

## Installation

The four Python packages are members of a single
[uv](https://docs.astral.sh/uv/) workspace. From the repository root:

```bash
uv sync                       # baseline (ballplate, sim, real-vision, designer)
uv sync --extra rl            # adds stable-baselines3 + torch + tensorboard
uv sync --extra video         # adds imageio[ffmpeg] for the MuJoCo runner mp4 recorder
```

`uv sync` resolves the whole workspace into one shared `.venv/`, so
cross-package edits are picked up without a manual reinstall. Every
subsequent command runs through `uv run`:

```bash
uv run pytest                 # full suite across ballplate/ and sim/
uv run ruff check .           # lint
```

## Running the pieces

```bash
# Trajectory designer (browser UI on port 5050)
uv run python -m designer.run

# Simulator. Use mjpython on macOS, plain python on Linux.
uv run mjpython -m sim.runner --controller pid --shape circle
uv run mjpython -m sim.runner --controller rl  --rl-run-dir sim/policies/<run>

# Standalone real-robot tracker (no ROS)
uv run python -m vision.main

# RL training (requires the [rl] extra)
uv run python -m sim.learning.train --config sim/config/learning/train.yaml
```

## ROS 2 workspace

The real-robot bring-up lives in `real/ros2_ws/`. It is built by colcon
rather than uv, because `ament_python` and `ament_cmake` packages are
not pip-installable. The ROS nodes import from `ballplate`, so the
colcon workspace expects `ballplate` to be importable from the active
Python environment: typically the system ROS Python with `ballplate`
installed in editable mode, or a venv that exposes `ballplate` on
`PYTHONPATH`.

```bash
cd real/ros2_ws
colcon build
source install/setup.bash
ros2 launch ball_tracker_ros          tracker_launch.py
ros2 launch ball_balance_controller   balance.launch.py        # PID
ros2 launch ball_balance_controller   mpc_balance.launch.py    # MPC
```

`rviz:=true` on either balance launch brings up the visualizer and
RViz next to the live controller; `view.launch.py` boots only the URDF
and the plate frame for offline inspection. The full RViz walk-through
lives in [real/README.md](real/README.md#rviz). `real/ros2_ws/install/`,
`build/`, and `log/` are gitignored and regenerated on every colcon
build.

## Package documentation

| Package                                  | Contents                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------ |
| [`ballplate`](ballplate/README.md)       | Pure-Python core: state, trajectories, controllers, safety. Single dependency for every other package. |
| [`sim`](sim/README.md)                   | MuJoCo simulator, controller wrappers, PPO training pipeline.                  |
| [`real`](real/README.md)                 | Standalone vision tracker and ROS 2 workspace.                                 |
| [`designer`](designer/README.md)         | Browser-based trajectory designer with sim and real backends.                  |

## Tooling

The project uses `uv` for dependency resolution and venv management,
`ruff` for lint and import sort, `mypy` (strict mode, scoped to
`ballplate/`) for type-checking, and `pytest` with `pytest-cov` for
tests and line/branch coverage. Pre-commit hooks pin formatting,
linting, and type-checking at commit time; enable them once per
checkout with

```bash
uv run pre-commit install
```

`uv run pre-commit run --all-files` runs every hook against the whole
tree on demand. The hook list is in `.pre-commit-config.yaml`; bump the
pinned revisions with `uv run pre-commit autoupdate`.

Coverage is collected on every test run:

```bash
uv run pytest --cov              # full suite + line/branch coverage
uv run pytest --cov --cov-report=html   # writes htmlcov/index.html
```

CI enforces a minimum line/branch coverage of 85 %.
