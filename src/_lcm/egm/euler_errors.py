"""Unit-free consumption Euler errors — a brute-free solution-accuracy metric.

A solver's accuracy can be read off the Euler error: at an interior
(unconstrained) consumption--saving optimum the Euler equation
`u'(c) = beta*g'(A)*u'(c_next)` holds exactly, where `g` is the regime's own law of
motion and `A = liquid - c` the savings it is evaluated at. The relative gap between
the chosen consumption and the consumption the equation implies measures how well a
method nulls the first-order condition — independent of any reference solve. It is
reported as the base-10 logarithm of the relative consumption error, so `-3` reads as
a 0.1% error.

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
    next_liquid: Float1D,
    marginal_return: Float1D,
) -> Float1D:
    """Compute the log10 unit-free consumption Euler error at each liquid grid point.

    For chosen consumption `c` the Euler equation implies
    `c_euler = (u')^-1(beta*g'(A)*u'(c_next))`, with `c_next` the next-period
    consumption policy interpolated at the landing point `g(A)` the regime's own law
    of motion gives for the savings `A = liquid - c`. The error is
    `log10(|c_euler / c - 1|)`.

    Both readings of the law are taken as arguments rather than rebuilt here from an
    assumed functional form: a metric that rebuilds `(1 + r)*A + income` reports the
    accuracy of a model the regime does not declare whenever the law carries anything
    else, and the departure cannot register as an error anywhere.

    Args:
        liquid_grid: Regular liquid-state grid (ascending); the abscissae
            `next_consumption` is read at.
        consumption: Chosen consumption policy on `liquid_grid`.
        next_consumption: Next period's consumption policy on `liquid_grid`. Pass the
            identity `liquid_grid` for a terminal bequest (all wealth consumed).
        discount_factor: Discount factor `beta`.
        preferences: The regime's felicity `u`, its marginal `u'`, and its
            inverse marginal `(u')^-1`, bound to this solve's parameters.
        next_liquid: The regime's own law of motion evaluated at each point's savings
            `liquid_grid - consumption`, same shape. Where that saving lands next
            period.
        marginal_return: That law's derivative with respect to savings, same shape.
            How the landing point moves when savings move, which is the factor the
            Euler equation discounts the continuation marginal by. For the
            conventional law this is the gross return at every point.

    Returns:
        The base-10 log relative consumption error at each liquid grid point, shape
        `(len(liquid_grid),)`.

    """
    consumption_next = jnp.interp(next_liquid, liquid_grid, next_consumption)
    marginal_next = preferences.marginal_utility(consumption_next)
    consumption_euler = preferences.inverse_marginal_utility(
        discount_factor * marginal_return * marginal_next
    )
    relative_error = jnp.abs(consumption_euler / consumption - 1.0)
    return jnp.log10(relative_error)
