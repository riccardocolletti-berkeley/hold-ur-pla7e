# ballplate

`ballplate` is the pure-Python core of the EECS 206A ball-on-plate
project. It carries the state types, reference-trajectory primitives,
classical and learned controllers, and the safety helpers that every
other package in the workspace (`sim`, `real`, `designer`) imports from.
Nothing in this package depends on MuJoCo, ROS 2, or OpenCV; the
controller math is therefore testable in isolation and the same code
runs unchanged on the simulator and on the real UR arm.

## Overview

The library is organised around two axes. On the data side, frozen
dataclasses describe the physical setup and every snapshot of the ball
or the plate: `BallState`, `PlateGeometry`, the `*Spec` family loaded
from `config/hardware.yaml`, the PID and MPC gain bundles, and the
parametric reference shapes. On the control side, plain classes own the
state that has to evolve in time: the PID integrator, the MPC scratch
buffers, the lazily-loaded RL policy. Each controller exposes a `step`
method that consumes a `BallState` plus a reference sample and returns
a two-dimensional plate-tilt command in radians. The platform-specific
wrappers in `sim` and `real` turn that command into joint torques or a
ROS trajectory message.

Reference trajectories implement a single `Reference` protocol:
`period` and `evaluate(t) -> (x, y, vx, vy)`. Concrete implementations
include closed-form shapes (`Stationary`, `Circle`, `Figure8`),
JSON-backed polylines drawn in the browser designer (`DrawnPath`), a
random factory used during RL training, and `SmoothApproach`, a wrapper
that blends from a start pose into any inner reference. All values are
in SI units (metres, metres per second, radians, seconds), in a
right-handed plate-local frame with the origin at the plate centre.

## Layout

```
src/ballplate/
  state.py                BallState, PlateGeometry
  hardware.py             loader for config/hardware.yaml
  control.py              loader for config/control.yaml + joint-lock helpers
  safety.py               deadband, velocity_clip
  controllers/
    pid/                  PidController, PidGains, preset helpers
    mpc/                  linear receding-horizon MPC (closed-form gains)
    rl/                   Stable-Baselines 3 PPO wrapper (opt-in)
  trajectories/
    base.py               Reference protocol
    parametric.py         Stationary, Circle, Figure8
    drawn.py              DrawnPath (JSON polyline)
    wrappers.py           SmoothApproach
    random.py             random reference factory
```

The pytest suite under `tests/` exercises every public class. None of
these tests pull in MuJoCo, ROS, or PyTorch, so they run on a fresh
checkout under `uv run pytest` in well under a second.

## Installation

The package is one member of the repo-root uv workspace. From the
monorepo root:

```sh
uv sync              # installs ballplate plus sim, real-vision, designer
uv sync --extra rl   # adds stable-baselines3 + torch for RLPolicy
```

The `[rl]` extra is opt-in because nothing else in the library depends
on PyTorch. The RL wrapper is also lazy-imported, so a process that
only references `PidController` never touches the SB3 import path.

## Usage

A minimal closed-loop tick on a synthetic ball state:

```python
from ballplate import BallState, PidController, PidGains, Circle
from ballplate.controllers.pid import presets

gains = presets.from_dict({"kp": 0.5, "ki": 1.0, "kd": 1.2})
pid = PidController(gains)
reference = Circle(radius=0.05, period=10.0)

ball = BallState(x=0.02, y=0.0, vx=0.0, vy=0.0, timestamp=0.0)
x_ref, y_ref, vx_ref, vy_ref = reference.evaluate(t=0.0)

ux, uy = pid.step(
    ball=ball,
    target_pos=(x_ref, y_ref),
    target_vel=(vx_ref, vy_ref),
    dt=1.0 / 60.0,
)
```

The MPC and RL controllers share the same call surface.
`MpcController.step_with_reference(ball, reference, now)` is the
typical entry point for trajectory tracking, because the MPC plans
against the reference sampled along its horizon. The RL wrapper takes
a pre-built observation vector produced by
`ballplate.controllers.rl.build_observation`; the observation layout
is the contract between training and deploy and is defined once in
`controllers/rl/observation.py`.

## Public API

| Module                                | Surface                                                                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `ballplate.state`                     | `BallState`, `PlateGeometry`                                                                                              |
| `ballplate.hardware`                  | `HardwareSpec`, `PlateSpec`, `BallSpec`, `PhysicsSpec`, `AdapterSpec`, `ArmSpec`, `load`, `default_path`                   |
| `ballplate.control`                   | `ControlSpec`, `load`, `default_path`, `parse_lock_spec`, `freeze`                                                        |
| `ballplate.safety`                    | `deadband`, `velocity_clip`                                                                                               |
| `ballplate.controllers`               | `PidController`, `PidGains`, `MpcController`, `MpcParams`                                                                 |
| `ballplate.controllers.pid.presets`   | `from_dict`, `from_named_dict`                                                                                            |
| `ballplate.controllers.rl`            | `RLPolicy` (opt-in), `build_observation`, `normalize_observation`                                                         |
| `ballplate.controllers.rl.loader`     | `pick_checkpoint`, `obs_rms_path_for_checkpoint`, `load_meta`, `load_obs_rms`                                             |
| `ballplate.trajectories`              | `Reference`, `Stationary`, `Circle`, `Figure8`, `DrawnPath`, `SmoothApproach`, `sample`                                   |

## Testing

```sh
uv run pytest ballplate/tests/ -q
```

The suite is the contract for every public symbol above: any change
that breaks a test breaks a downstream package, since the simulator,
the ROS controllers, and the trajectory designer all import from
`ballplate` directly.
