"""Joint-locking helpers re-exported from :mod:`ballplate.control`.

The implementation lives in the shared ``ballplate`` package so the
simulator and the real-robot controller parse the same lock spec from
the same ``config/control.yaml``. This module is a backwards-compatible
alias for callers that historically imported from ``sim.arm.lock``.
"""

from ballplate.control import freeze, parse_lock_spec

__all__ = ["freeze", "parse_lock_spec"]
