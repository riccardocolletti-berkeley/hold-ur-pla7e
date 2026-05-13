"""Observation vector for the RL policy: 12 floats, plate-local SI units.

Single source of truth shared between training (in the simulator) and
deploy. Any caller that feeds an ``RLPolicy`` must build the observation
through ``build_observation`` so ordering and units match the training
distribution.

Layout::

    [0]  bx           ball position x [m]
    [1]  by           ball position y [m]
    [2]  bvx          ball velocity x [m/s]
    [3]  bvy          ball velocity y [m/s]
    [4]  tx           target position x [m]
    [5]  ty           target position y [m]
    [6]  tvx          target velocity x [m/s]
    [7]  tvy          target velocity y [m/s]
    [8]  tx_la        target lookahead position x [m]
    [9]  ty_la        target lookahead position y [m]
    [10] plate_pitch  rotation of the plate normal about world x [rad]
    [11] plate_roll   rotation of the plate normal about world y [rad]
"""

import numpy as np

from ballplate.state import BallState

#: Length of the observation vector; tied to the layout above.
OBSERVATION_DIM: int = 12


def build_observation(
    ball: BallState,
    target_now: tuple[float, float, float, float],
    target_lookahead_pos: tuple[float, float],
    plate_pitch: float,
    plate_roll: float,
) -> np.ndarray:
    """Pack inputs into the canonical 12-element float32 vector.

    ``target_now`` is ``(tx, ty, tvx, tvy)``; ``target_lookahead_pos`` is
    the target position one lookahead horizon ahead.
    """
    tx, ty, tvx, tvy = target_now
    tx_la, ty_la = target_lookahead_pos
    return np.array(
        [
            ball.x,
            ball.y,
            ball.vx,
            ball.vy,
            tx,
            ty,
            tvx,
            tvy,
            tx_la,
            ty_la,
            plate_pitch,
            plate_roll,
        ],
        dtype=np.float32,
    )


def normalize_observation(
    obs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    clip: float,
) -> np.ndarray:
    """VecNormalize-style standardisation: ``clip((obs - mean) / std, +/-clip)``.

    ``mean`` and ``std`` are the running statistics from
    ``stable_baselines3.common.vec_env.VecNormalize`` saved at training
    time so deploy reproduces the training distribution.
    """
    out: np.ndarray = np.clip((obs - mean) / std, -clip, clip).astype(np.float32)
    return out
