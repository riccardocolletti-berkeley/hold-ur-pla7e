# ball_balance_controller

Closed-loop ball-on-plate controller for the UR arm, in two flavours.

```
/ball_state    ->  PID  ->  (ux, uy)        (PID node, controller.py)
                                |
TF base->plate ->  R_WP(q)      |
                                v
                  omega_des  =  -ux * y_plate + uy * x_plate     (world frame)
                  Delta_q    =  J_rot^+ * omega_des              (free joints)
                  q_cmd      =  q_current + Delta_q_filtered
                                |
                                v
        trajectory_msgs/JointTrajectory  ->  /scaled_joint_trajectory_controller/joint_trajectory
```

`mpc_node.py` is the MPC variant of the same node and uses the same
projection-plus-IK stage; the only difference is the position-level
controller producing `(ux, uy)`.

## Layout

```
ball_balance_controller/
  ball_balance_controller/
    controller.py            PID ROS 2 node, control loop, safety guards
    mpc_node.py              MPC ROS 2 node, same projection/IK stage
    visualizer.py            RViz marker publisher (plate / ball / reference)
    jacobian.py              ikpy chain wrapper, rotational J(q)
    joint_state_reader.py    /joint_states cache in canonical UR order
    tf_plate_reader.py       base_link -> plate_center rotation lookup
    trajectory_client.py     JointTrajectory publisher
    mock_ball_publisher.py   synthetic /ball_state for offline tests
  config/
    pid_params.yaml          tunable parameters for controller.py
    mpc_params.yaml          tunable parameters for mpc_node.py
    reference.yaml           shared reference-trajectory block
    tuning_presets.yaml      named gain bundles (soft / stiff / ...)
  launch/
    balance.launch.py        PID on the real arm
    mpc_balance.launch.py    MPC on the real arm
    balance_sim.launch.py    PID + mock_ball_publisher (offline)
    view.launch.py           RViz only (URDF + frames sanity check)
  rviz/
    balance.rviz             shared RViz config for both launches
  urdf/
    ur7e.urdf                bundled UR7e description
  package.xml
  setup.py
```

The PID core, frames, trajectories, and safety helpers come from the
pure-Python `ballplate` package. Plate / ball / physics constants and
the wrist-to-plate adapter pose come from `config/hardware.yaml` at the
monorepo root.

## Build and run

`ballplate` is a pure-Python workspace member that the controller imports
at runtime. `uv sync` from the monorepo root installs it into the
shared `.venv/` that the ROS environment must see; without it the
`from ballplate...` imports fail at launch.

```bash
# 1. Install Python workspace deps once
cd ball_on_plate
uv sync

# 2. Build the ROS 2 packages
cd real/ros2_ws
colcon build --packages-select \
    ball_tracker_msgs ball_tracker_ros ball_balance_controller
source install/setup.bash

# 3. Run on the real arm
ros2 launch ball_balance_controller balance.launch.py            # PID
ros2 launch ball_balance_controller mpc_balance.launch.py        # MPC

# 4. Run offline (mock ball publisher + no real arm)
ros2 launch ball_balance_controller balance_sim.launch.py
```

Add `rviz:=true` to either of the real-arm launches to bring up RViz
and the visualizer alongside the controller. Use `view.launch.py` to
inspect the URDF + plate frame without the arm connected.

## Live YAML reload

The controller, visualizer, and tracker nodes register an on-set-parameters
callback that rebuilds their frozen config dataclass when a YAML file
changes, validates it, and only swaps state in if validation passes.
Editing `config/pid_params.yaml` or `config/reference.yaml` therefore
updates the running nodes within ~1 s, no colcon rebuild, no relaunch.
URDF path, home pose, frames, locked joints, control timer rate, and
camera resolution/FPS/calibration are init-only and reject live edits
with a "needs relaunch" reason in the log.

## Tuning sequence

1. Sign check. With small gains and `reference_shape: stationary`,
   confirm pitch and roll move the ball back toward the centre.
2. Proportional. Raise `kp` until the ball oscillates gently.
3. Derivative. Raise `kd` until the oscillation damps within one or
   two swings.
4. Integral. Enable `ki` only when a residual steady-state offset shows.
5. Tracking. Switch `reference_shape` to `circle` and start with a
   small radius and a long period.
