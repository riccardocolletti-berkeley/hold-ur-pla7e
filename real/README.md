# real

Real-robot deployment for the ball-on-plate stack.

| Folder      | Contents                                                            |
| ----------- | ------------------------------------------------------------------- |
| `vision/`   | Standalone Python tracker (camera, ArUco, tilt-aware Kalman). No ROS dependency. |
| `ros2_ws/`  | ROS 2 colcon workspace: tracker node, message types, PID and MPC balance controllers, visualizer. |

The tracker node in `ros2_ws/src/ball_tracker_ros` imports from
`real.vision`, so the detection and filtering code is shared with the
standalone path.

## Conventions

SI units throughout: metres, metres/second, radians, kilograms. Gravity
9.81 m/s^2; rolling factor 5/7 for a uniform solid sphere. The table
frame is right-handed, origin at the table centre, +x to the camera
right and +y up as seen from the overhead camera.

## Quick start

```bash
# 1. Install workspace (real-vision, ballplate, sim, designer).
uv sync                                       # from the repo root

# 2. Print and place ArUco markers.
uv run python -m vision.tools.generate_markers

# 3. (Optional) calibrate the camera.
uv run python -m vision.tools.calibrate_camera

# 4. Run the standalone tracker.
uv run python -m vision.main
```

## ROS 2 usage

```bash
cd real/ros2_ws
colcon build
source install/setup.bash
ros2 launch ball_tracker_ros          tracker_launch.py
ros2 launch ball_balance_controller   balance.launch.py        # PID
ros2 launch ball_balance_controller   mpc_balance.launch.py    # MPC
```

## RViz

Two launch entry points share the same RViz config at
`ros2_ws/src/ball_balance_controller/rviz/balance.rviz`:

- `balance.launch.py rviz:=true` brings up the visualizer and RViz next
  to the live controller. Expects `robot_state_publisher` and
  `/joint_states` from the UR driver, plus `/ball_state` from the
  tracker. Without the tracker, only the plate slab and the static
  reference line are visible.
- `view.launch.py` launches `robot_state_publisher` against the bundled
  URDF, `joint_state_publisher_gui` with sliders for every revolute
  joint, the static `tool0 -> plate_center` TF, the visualizer node,
  and RViz. Useful for inspecting the plate frame at an arbitrary joint
  configuration without the UR connected.

### Display contents

Fixed frame: `base_link`. Bundled displays:

| Display               | Source                                         | Notes                                                                    |
| --------------------- | ---------------------------------------------- | ------------------------------------------------------------------------ |
| `RobotModel`          | `/robot_description` (transient-local)         | UR7e from the bundled URDF.                                              |
| `TF`                  | `/tf`, `/tf_static`                            | `base_link`, `tool0`, `plate_center`. `plate_center` is anchored to `tool0` by a static TF. |
| `BallBalance Markers` | `/ball_balance_visualization`                  | Plate slab, ball, and reference trajectory. See marker table below.      |
| `Grid`                | local                                          | 0.1 m grid in `base_link` XY.                                            |

All markers in the `MarkerArray` live in the `plate_center` frame so the
overlay tracks the wrist as the arm moves:

| Marker (id)             | Source                                                 | Behaviour                                                                                                              |
| ----------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `plate` (0)             | `ballplate.hardware.PlateSpec`                         | Translucent blue slab; top face flush with `plate_center` z = 0.                                                       |
| `ball` (1)              | `/ball_state` (`ball_tracker_msgs/BallState`)          | Orange sphere at `(x, y, ball_radius)`. Hidden when `ball_found = false`, so a stale Kalman prediction is not confused with a real detection. |
| `reference` (2)         | `reference_*` parameters via `ballplate.trajectories`  | Green line strip of one period. Closed for `circle` / `figure8`, open for `drawn`. Hidden for `stationary`.            |
| `reference_target` (3)  | `/reference_target` from the active controller         | Small green sphere at the controller's current setpoint.                                                               |

### View-launch flags

```bash
ros2 launch ball_balance_controller view.launch.py rviz:=false                       # headless
ros2 launch ball_balance_controller view.launch.py use_joint_state_publisher_gui:=false
ros2 launch ball_balance_controller view.launch.py reference_shape:=figure8 reference_radius_m:=0.06
```

### Diagnostics

| Symptom in RViz                                         | Cause                                                                                                                                                |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RobotModel`: "Failed to receive description"           | `robot_state_publisher` is not running, or the UR driver has not published `/robot_description`. Check the launch graph.                             |
| `BallBalance Markers` shows the plate but no ball       | `markers_found < 4`, or the ball is outside the HSV mask. Cross-check against the OpenCV debug window in the tracker.                                |
| Ball marker visibly off the plate slab                  | Bad homography (check ArUco corners in the tracker overlay) or a mismatch between `config/hardware.yaml::adapter` and the physical adapter geometry. |
| Reference path floats above the plate surface           | Intentional: the line is drawn at z = `ball_radius` so it does not z-fight with the plate marker.                                                    |
| "Lookup would require extrapolation" spam               | The tracker's `/ball_state` timestamps come from `time.time()` (wall clock); RViz uses the ROS clock. Keep `Frame Timeout: 15` on the TF display.    |

## Visualizer topics and parameters

```text
ball_balance_visualizer
  subscriptions
    /ball_state         ball_tracker_msgs/BallState   (sensor data QoS)
    /reference_target   geometry_msgs/PointStamped     (target marker)
  publications
    /ball_balance_visualization   visualization_msgs/MarkerArray
  parameters
    reference_shape           string   stationary | circle | figure8 | drawn
    reference_radius_m        double   default 0.020
    reference_period_s        double   default 20.0
    hold_x_m, hold_y_m        double   default 0.0  (used when shape=stationary)
    drawn_file                string   required when shape=drawn
    reference_overlay_samples int      default 240
    publish_rate_hz           double   default 10.0
```

`balance.launch.py rviz:=true` and `mpc_balance.launch.py rviz:=true`
pass `params_file` and `reference.yaml` to both the controller and the
visualizer, so the static green path and the controller's actual
reference cannot drift.
