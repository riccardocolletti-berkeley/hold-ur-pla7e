# sim

`sim` is the MuJoCo wiring for the ball-on-plate stack. Every module
that depends on MuJoCo's Python API lives here: the scene builder, the
adapters that translate `mujoco.MjData` into `ballplate.BallState`, the
controller wrappers, the PPO training pipeline, and the runtime entry
point that drives the interactive viewer. The package never
reimplements PID/MPC/RL math: that logic lives in the sibling
`ballplate` package and is imported as-is, which keeps the controllers
testable in isolation and lets the same code run on the real arm
without modification.

## Layout

```
sim/
├── pyproject.toml
├── README.md
└── src/sim/
    ├── __init__.py
    ├── runner.py          interactive viewer entry point
    ├── scene.py           MuJoCo XML scene builder (UR5e + plate + ball)
    ├── adapters/          MuJoCo <-> ballplate bridges
    ├── controllers/       MuJoCo wrappers around ballplate.controllers
    ├── arm/               arm joint waypoints and lock helpers
    ├── viz/               MuJoCo viewer overlays (trail, target, recording)
    ├── benchmark/         PID vs PID+RL comparison across DR domains
    ├── dual/              side-by-side viewer of two arms in one scene
    └── learning/          PPO training, evaluation, callbacks, configs
```

A pytest suite under `tests/` exercises the adapters, the arm helpers,
and the reward and curriculum shaping.

## Installation

The package is one member of the repo-root uv workspace. From the
monorepo root:

```sh
uv sync                       # baseline (PID, viewer, tests)
uv sync --extra rl            # adds stable-baselines3 + torch + tensorboard
uv sync --extra video         # adds imageio[ffmpeg] for the offscreen recorder
```

Use `uv run mjpython ...` on macOS (or `uv run python ...` on Linux) so
the MuJoCo-aware interpreter ends up on `PATH`.

## Running

The interactive viewer is the typical entry point:

```sh
uv run mjpython -m sim.runner --controller pid --shape circle
uv run mjpython -m sim.runner --controller pid --drawn-file path/to/draw.json
uv run mjpython -m sim.runner --controller rl  --rl-run-dir runs/ppo_dr_v1
```

`--lock-joints all` freezes the upper-arm chain at home so only the
wrist actuates the plate. Useful when tuning the PID gains or
verifying an RL policy in distribution.

PPO training reads the YAML under `config/learning/` and writes the
artefacts (checkpoints, observation statistics, metrics) under
`policies/<run>/`:

```sh
uv run python -m sim.learning.train --config sim/config/learning/train.yaml
```

The benchmark module replays a trained policy against the deployed PID
on the eight perturbed domains plus the nominal baseline:

```sh
uv run python -m sim.benchmark --rl-run-dir sim/policies/<run>
```

The dual viewer renders two arms side-by-side in the same scene: one
running PID only, one running PID plus the residual policy. Useful for
visual demos because both arms see the same reference and the same
random seed at every step.

## Public API

| Module                | Surface                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `sim.adapters`        | `BallStateReader`, `JointActuator`, `BallDropMonitor`                                                              |
| `sim.scene`           | `SceneConfig`, `build_scene` (assembles the UR5e + plate + ball scene from MuJoCo Menagerie)                       |
| `sim.arm`             | `JointTrajectory`, `parse_lock_spec`, `freeze`                                                                     |
| `sim.viz`             | `BallTrail`, `TargetOverlay`                                                                                       |
| `sim.viz.recorder`    | `Recorder` (offscreen MuJoCo to mp4 writer; needs `[video]` extra)                                                 |
| `sim.controllers`     | `PidController`, `MpcController` (eager); `RLController` via `sim.controllers.rl` (needs `[rl]` extra)             |
| `sim.runner`          | `main` (CLI entry point that wires the world and drives the viewer)                                                |
| `sim.benchmark`       | Domain definitions and rollout helpers; `python -m sim.benchmark` runs the comparison table                        |
| `sim.dual`            | Dual-arm scene builder and runner                                                                                  |
| `sim.learning.env`    | `BallPlateEnv` (Gymnasium env: 2-d action, 12-d observation)                                                       |
| `sim.learning.randomize` | `load_ranges`, `sample`, `apply` (DR sampling + in-place model mutation)                                        |
| `sim.learning.reward` | `load_weights`, `step_reward`                                                                                      |
| `sim.learning.curriculum` | `shapes_for_step`                                                                                              |
| `sim.learning.callbacks`  | `JSONScalarLogger`, `CurriculumCallback`, `LinearScheduleCallback`, `BestModelCallback`, `CheckpointWithStatsCallback` |
| `sim.learning.train`  | `main` (PPO training entry point)                                                                                  |

## Testing

```sh
uv run pytest sim/tests/ -q
```

The sim tests cover the actuator filter, the arm lock helpers, and the
reward and curriculum shaping. The full repository suite combines them
with `ballplate/tests/`:

```sh
uv run pytest
```
