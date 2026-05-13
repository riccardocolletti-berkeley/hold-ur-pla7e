"""Read ball-on-plate state from MuJoCo data into `ballplate` types."""

import numpy as np

from ballplate.state import BallState


class BallStateReader:
    """Project the MuJoCo world-frame ball pose into the plate-local frame.

    Holds the body / site IDs once at startup; every `read` is then a small
    fixed-cost dot-product against the plate's rotation columns. The reader is
    deliberately stateless aside from the cached IDs so callers can use it
    safely from inside vectorised environments.
    """

    def __init__(self, model, plate_site_id: int, ball_body_id: int):
        self.model = model
        self.plate_site_id = int(plate_site_id)
        self.ball_body_id = int(ball_body_id)

    def read(self, data, timestamp: float = 0.0) -> tuple[BallState, np.ndarray]:
        """Return ``(ball_state, plate_orientation_3x3)``.

        The orientation matrix is the plate frame expressed in world
        coordinates; consumers that only need the ball state can ignore it,
        but actuation and RL observation building both need it (for the
        Jacobian step and for plate pitch / roll respectively).
        """
        ball_pos = data.xpos[self.ball_body_id]
        plate_pos = data.site_xpos[self.plate_site_id]
        plate_mat = data.site_xmat[self.plate_site_id].reshape(3, 3)

        offset = ball_pos - plate_pos
        x_hat = plate_mat[:, 0]
        y_hat = plate_mat[:, 1]

        # Project world-frame position and linear velocity onto plate axes.
        bx = float(np.dot(offset, x_hat))
        by = float(np.dot(offset, y_hat))
        ball_vel = data.cvel[self.ball_body_id][3:]  # cvel layout: [angular; linear]
        vx = float(np.dot(ball_vel, x_hat))
        vy = float(np.dot(ball_vel, y_hat))

        return (
            BallState(x=bx, y=by, vx=vx, vy=vy, timestamp=float(timestamp)),
            plate_mat,
        )
