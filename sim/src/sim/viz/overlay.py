"""Static overlay of the target reference, rendered on the plate surface.

Samples the reference once at construction in plate-local coordinates; each
frame re-projects those samples through the plate's current pose so the
overlay stays glued to the surface as it tilts.
"""

import mujoco
import numpy as np

from ballplate.trajectories import Reference


class TargetOverlay:
    """A polyline of yellow dots showing the desired ball path on the plate."""

    def __init__(
        self,
        reference: Reference,
        plate_body_id: int,
        plate_half_thickness: float,
        n_samples: int = 120,
        period: float = 10.0,
        color: tuple[float, float, float] = (1.0, 0.85, 0.0),
        point_size: float = 0.0015,
        lift: float = 0.0015,
    ):
        self.plate_body_id = int(plate_body_id)
        self.surface_z_local = float(plate_half_thickness + lift)
        self.color = color
        self.point_size = float(point_size)

        # Pre-sample the reference at uniformly-spaced times across one period
        # so add_to_scene only has to project the cached points each frame.
        ts = np.linspace(0.0, float(period), int(n_samples), endpoint=False)
        self.points_local = np.array([reference.evaluate(float(t))[:2] for t in ts], dtype=float)

    def add_to_scene(self, scn, data) -> None:
        """Append the overlay dots to the viewer's user scene `scn`."""
        rot = data.xmat[self.plate_body_id].reshape(3, 3)
        plate_pos = data.xpos[self.plate_body_id]
        identity = np.eye(3).flatten()
        size = np.array([self.point_size, 0.0, 0.0])
        rgba = np.array([*self.color, 1.0], dtype=np.float32)

        for lx, ly in self.points_local:
            if scn.ngeom >= scn.maxgeom:
                break
            local = np.array([lx, ly, self.surface_z_local])
            world_pos = plate_pos + rot @ local
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size,
                pos=world_pos,
                mat=identity,
                rgba=rgba,
            )
            scn.ngeom += 1
