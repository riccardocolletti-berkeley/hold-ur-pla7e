"""Reference trajectory protocol."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reference(Protocol):
    """Time-dependent ball-position target in the plate-local frame.

    ``period`` is one full pass of the path, in seconds; time-invariant
    references (e.g. ``Stationary``) report any positive value.
    ``evaluate(t)`` returns ``(x, y, vx, vy)`` in plate-local metres and
    metres per second.
    """

    @property
    def period(self) -> float: ...

    def evaluate(self, t: float) -> tuple[float, float, float, float]: ...
