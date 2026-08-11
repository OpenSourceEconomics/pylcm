"""Unit-free consumption Euler errors — a brute-free solution-accuracy metric.

A solver's accuracy can be read off the Euler error: at an interior
(unconstrained) consumption--saving optimum the Euler equation
`u'(c) = beta*(1+r)*u'(c_next)` holds exactly, so the relative gap between the chosen
consumption and the consumption the equation implies measures how well a method nulls
the first-order condition — independent of any reference solve. It is reported as the
base-10 logarithm of the relative consumption error, so `-3` reads as a 0.1% error.

The metric is meaningful only on the unconstrained interior: where the borrowing
constraint binds the Euler equation holds with a positive multiplier, and the residual
there reflects the constraint, not solution error.
"""

import jax.numpy as jnp

from _lcm.egm.preferences import Preferences
from lcm.typing import Float1D


def consumption_euler_error_log10(
    *,
    liquid_grid: Float1D,
    consumption: Float1D,
    next_consumption: Float1D,
    discount_factor: float,
    preferences: Preferences,
    return_liquid: float,
    income: float,
) -> Float1D:
    """Compute the log10 unit-free consumption Euler error at each liquid grid point.

    For chosen consumption `c` the next-period liquid state is
    `(1 + r)*(liquid - c) + income`, and the Euler equation implies
    `c_euler = (u')^-1(beta*(1+r)*u'(c_next))`, with `c_next` the next-period
    consumption policy interpolated at the next-period liquid state. The error is
    `log10(|c_euler / c - 1|)`.

    Args:
        liquid_grid: Regular liquid-state grid (ascending) the policy is defined on.
        consumption: Chosen consumption policy on `liquid_grid`.
        next_consumption: Next period's consumption policy on `liquid_grid`. Pass the
            identity `liquid_grid` for a terminal bequest (all wealth consumed).
        discount_factor: Discount factor `beta`.
        preferences: The regime's felicity `u`, its marginal `u'`, and its
            inverse marginal `(u')^-1`, bound to this solve's parameters.
        return_liquid: Liquid net return `r`.
        income: Deterministic income added to next-period liquid.

    Returns:
        The base-10 log relative consumption error at each liquid grid point, shape
        `(len(liquid_grid),)`.

    """
    next_liquid = (1.0 + return_liquid) * (liquid_grid - consumption) + income
    consumption_next = jnp.interp(next_liquid, liquid_grid, next_consumption)
    marginal_next = preferences.marginal_utility(consumption_next)
    consumption_euler = preferences.inverse_marginal_utility(
        discount_factor * (1.0 + return_liquid) * marginal_next
    )
    relative_error = jnp.abs(consumption_euler / consumption - 1.0)
    return jnp.log10(relative_error)
