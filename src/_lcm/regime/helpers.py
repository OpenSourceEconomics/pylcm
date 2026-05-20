"""Default helpers used by the user-facing `Regime`."""

from lcm.typing import FloatND


def _default_H(
    utility: FloatND, E_next_V: FloatND, discount_factor: FloatND
) -> FloatND:
    """Default Bellman aggregator: `U + β · E[V_next]`."""
    return utility + discount_factor * E_next_V
