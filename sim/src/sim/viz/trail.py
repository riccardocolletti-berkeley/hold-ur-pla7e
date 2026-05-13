"""Fading visual trail of the ball over the plate, rendered into the viewer.

Points are stored in the *plate-local* frame so the trail stays glued to the
plate surface as it tilts, instead of being left behind in world coordinates.
Each point fades linearly with elapsed simulation time and is removed once
its alpha hits zero.
"""

from collections import deque

import mujoco
import numpy as np


class BallTrail:
    """Append-only ring buffer of recently-visited plate-local positions."""

    def __init__(
        self,
        ball_body_id: int,
        plate_body_id: int,
        plate_half_thickness: float,
        max_points: int = 10,
        fade_seconds: float = 2.5,
        color: tuple[float, float, float] = (0.0, 0.9, 0.2),
        point_size: float = 0.0035,
        lift: float = 0.001,
        min_dt: float = 0.25,
    ):
        self.ball_body_id = int(ball_body_id)
        self.plate_body_id = int(plate_body_id)
        # Dots sit just above the plate surface so they do not z-fight with the slab.
        self.surface_z_local = float(plate_half_thickness + lift)
        self.fade_seconds = float(fade_seconds)
        self.color = color
        self.point_size = float(point_size)
        self.min_dt = float(min_dt)
        self._points: deque = deque(maxlen=int(max_points))

    def record(self, data) -> None:
        """Append the current ball position (in plate-local coords) to the trail."""
        if self._points and data.time - self._points[-1][1] < self.min_dt:
            return
        rot = data.xmat[self.plate_body_id].reshape(3, 3)
        # World to plate-local: rotate the world-frame ball offset by R^T.
        ball_local = rot.T @ (data.xpos[self.ball_body_id] - data.xpos[self.plate_body_id])
        self._points.append(
            (
                np.array([ball_local[0], ball_local[1], self.surface_z_local]),
                float(data.time),
            )
        )

    def clear(self) -> None:
        """Drop all points; call between independent runs."""
        self._points.clear()

    def render(self, viewer, data) -> None:
        """Draw the trail into the viewer's user scene (overwrites previous geoms)."""
        scn = viewer.user_scn
        scn.ngeom = 0
        if not self._points:
            return

        rot = data.xmat[self.plate_body_id].reshape(3, 3)
        plate_pos = data.xpos[self.plate_body_id]
        now = self._points[-1][1]
        identity = np.eye(3).flatten()
        size = np.array([self.point_size, 0.0, 0.0])

        for local, t in self._points:
            if scn.ngeom >= scn.maxgeom:
                break
            alpha = 1.0 - (now - t) / self.fade_seconds
            if alpha <= 0.0:
                continue
            world_pos = plate_pos + rot @ local
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size,
                pos=world_pos,
                mat=identity,
                rgba=np.array([*self.color, alpha], dtype=np.float32),
            )
            scn.ngeom += 1
