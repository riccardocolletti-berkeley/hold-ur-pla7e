"""Stateless safety primitives shared by every controller (sim and real)."""


def deadband(error: float, threshold: float) -> float:
    """Return 0 when ``|error| <= threshold``, else ``error`` unchanged.

    Kills chatter when the position estimate has noise of order ``threshold``.
    ``threshold`` is taken absolute.
    """
    return 0.0 if abs(error) <= abs(threshold) else error


def velocity_clip(velocity_error: float, limit: float) -> float:
    """Saturate ``velocity_error`` to ``[-|limit|, +|limit|]``.

    Caps the derivative-term contribution when the velocity estimate is
    transiently noisy.
    """
    lim = abs(limit)
    return max(-lim, min(lim, velocity_error))
