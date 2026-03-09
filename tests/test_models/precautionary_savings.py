"""Precautionary savings model — thin wrapper around lcm_examples."""

from lcm_examples.precautionary_savings import (
    RegimeId,
    ShockType,
    get_model,
    get_params,
    next_regime,
    next_wealth,
    utility,
    wealth_constraint,
)

__all__ = [
    "RegimeId",
    "ShockType",
    "get_model",
    "get_params",
    "next_regime",
    "next_wealth",
    "utility",
    "wealth_constraint",
]
