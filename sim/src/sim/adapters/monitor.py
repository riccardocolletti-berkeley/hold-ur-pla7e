"""Detect when the ball has fallen off the plate (MuJoCo-specific)."""

import numpy as np


class BallDropMonitor:
    """Compare the ball's world-frame z to the plate site's z.

    A drop is declared when the ball is more than `drop_below_m` below the
    plate site, which sits a few millimetres above the plate top. The
    threshold is conservative enough to ignore a small bounce on placement
    while still firing well before the ball escapes the simulator volume.
    """

    def __init__(
        self,
        model,
        ball_body_id: int,
        plate_site_id: int,
        drop_below_m: float = 0.05,
    ):
        self.model = model
        self.ball_body_id = int(ball_body_id)
        self.plate_site_id = int(plate_site_id)
        self.drop_below_m = float(drop_below_m)

    def is_dropped(self, data) -> bool:
        """True when the ball is below the plate by more than the threshold."""
        ball_z = float(data.xpos[self.ball_body_id, 2])
        plate_z = float(data.site_xpos[self.plate_site_id, 2])
        return ball_z < plate_z - self.drop_below_m

    def reset_ball(self, data, settle_height_m: float = 0.015) -> None:
        """Reseat the ball at the plate centre with zero velocity.

        The ball's free joint is assumed to start immediately after the
        ``nu`` arm joints in ``data.qpos`` (the convention the scene builder
        produces). ``settle_height_m`` lets the ball drop a couple of
        millimetres onto the plate so contact is established cleanly.
        """
        plate_pos = data.site_xpos[self.plate_site_id]
        ball_qpos_start = int(self.model.nu)

        # Position (3) + quaternion (4); velocity slot is 6 wide for a free joint.
        data.qpos[ball_qpos_start : ball_qpos_start + 3] = plate_pos + np.array(
            [0.0, 0.0, settle_height_m]
        )
        data.qpos[ball_qpos_start + 3] = 1.0
        data.qpos[ball_qpos_start + 4 : ball_qpos_start + 7] = 0.0
        data.qvel[ball_qpos_start : ball_qpos_start + 6] = 0.0
