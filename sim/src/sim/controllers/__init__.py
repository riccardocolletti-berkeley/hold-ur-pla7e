"""Controller wrappers that wire `ballplate.controllers` into the MuJoCo runtime.

PID and MPC are pure-NumPy/SciPy and are re-exported here. The RL wrapper
loads a Stable-Baselines 3 policy and is therefore imported explicitly so
deployments that don't need it do not pay the SB3 cost::

    from sim.controllers import PidController                # always works
    from sim.controllers import MpcController                # always works
    from sim.controllers.rl import RLController              # needs sim[rl]
"""

from sim.controllers.mpc import MpcController
from sim.controllers.pid import PidController

__all__ = ["MpcController", "PidController"]
