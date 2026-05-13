std_msgs/Header header

# True when the ball was detected this frame; false when the value is a
# Kalman prediction only (no fresh measurement). Use this for visualization
# (e.g. hide a stale prediction marker), NOT for gating control.
bool ball_found

# True when the tracker is providing a usable position/velocity estimate,
# either from a fresh detection (``ball_found=true``) or from a recent
# Kalman coast within the dropout window. Downstream control should gate
# on this field rather than ``ball_found`` so brief detection failures
# (motion blur during fast rolls) don't freeze the PID step.
bool tracking_valid

# Number of ArUco markers detected this frame. The homography needs all
# four during bootstrap; afterwards a single characterised marker (4
# corner pairs) is enough to refit.
int32 markers_found

# Position of the ball centre in the table frame, in metres, with the
# origin at the table centre.
float64 x
float64 y

# Velocity in the table frame, in metres per second.
float64 vx
float64 vy

# Raw pixel coordinates of the detection (debugging only).
float64 pixel_u
float64 pixel_v
