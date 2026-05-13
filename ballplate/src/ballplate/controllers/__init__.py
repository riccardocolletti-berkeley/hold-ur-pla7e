"""Controllers package: PID and MPC eagerly imported; RL opt-in.

PID and MPC are pure NumPy/SciPy and always importable. The RL wrapper
pulls Stable-Baselines 3 and is exposed only via the submodule import,
so PID/MPC-only callers don't pay that cost.

    from ballplate.controllers import PidController, PidGains   # always
    from ballplate.controllers import MpcController, MpcParams  # always
    from ballplate.controllers.rl import RLPolicy               # [rl] extra
"""

from ballplate.controllers.mpc import MpcController, MpcParams
from ballplate.controllers.pid import PidController, PidGains

__all__ = ["MpcController", "MpcParams", "PidController", "PidGains"]
