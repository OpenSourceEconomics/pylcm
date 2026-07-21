"""One backward EGM step for a 1-D consumption--saving problem.

The retired phase of the DS pension model is a plain consumption--saving problem: a
single continuous state `liquid`, a single continuous action `consumption`, and no
discrete choice. With one continuous state the endogenous grid method needs no upper
envelope — invert the consumption Euler equation

$$u'(c) = \\beta (1+r) V'_{\\text{next}}\\big((1+r)(\\text{liquid}-c)+y\\big)$$

on a post-decision savings grid $s = \\text{liquid} - c \\ge 0$, read the next-period
value and its marginal off the prior arrays, and map the resulting endogenous liquid
back onto the regular liquid grid. Below the natural borrowing limit the constraint
binds and the agent consumes all liquid.

The step carries the **marginal value of liquid** $V'(\\text{liquid})$ between periods
rather than finite-differencing the value array: by the envelope theorem
$V'(\\text{liquid}) = u'(c^*)$, which is exact, whereas a finite difference of the value
array is badly inaccurate where the value is steep (low liquid). Each step therefore
both consumes `next_marginal` and publishes this period's marginal.
"""

from typing import NamedTuple

import jax.numpy as jnp

from _lcm.egm.crra import crra_utility
from lcm.typing import Float1D, ScalarFloat


class RetiredEGMResult(NamedTuple):
    """A retired EGM step's value, marginal value of liquid, and consumption policy."""

    value: Float1D
    """This period's value on `liquid_grid`."""
    marginal: Float1D
    """This period's marginal value of liquid `V' = u'(c*)` on `liquid_grid`."""
    consumption: Float1D
    """This period's optimal consumption on `liquid_grid`."""


def egm_one_asset_step(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    crra: ScalarFloat | float,
    return_liquid: ScalarFloat | float,
    income: ScalarFloat | float,
) -> RetiredEGMResult:
    """Solve one period of the 1-D consumption--saving problem by EGM.

    Args:
        next_value: Next period's value on `liquid_grid`, shape `(n_liquid,)`.
        next_marginal: Next period's marginal value of liquid `V'` on `liquid_grid`,
            shape `(n_liquid,)`. For a terminal bequest `u(liquid)` this is
            `liquid ** (-crra)`.
        liquid_grid: Regular liquid-state grid (ascending).
        savings_grid: Post-decision savings grid `s = liquid - consumption` (ascending,
            starting at 0), shape `(n_savings,)`.
        discount_factor: Discount factor `beta`.
        crra: Coefficient of relative risk aversion `rho`.
        return_liquid: Liquid net return `r`.
        income: Deterministic income added to next-period liquid.

    Returns:
        This period's value, marginal value of liquid, and consumption policy on
        `liquid_grid`.

    """
    gross_return = 1.0 + return_liquid
    next_liquid = gross_return * savings_grid + income
    value_next = jnp.interp(next_liquid, liquid_grid, next_value)
    marginal_next = jnp.interp(next_liquid, liquid_grid, next_marginal)

    consumption = (discount_factor * gross_return * marginal_next) ** (-1.0 / crra)
    liquid_endog = consumption + savings_grid
    value_endog = crra_utility(consumption, crra) + discount_factor * value_next

    # Interior: interpolate the endogenous value and consumption onto the regular grid.
    interior_value = jnp.interp(liquid_grid, liquid_endog, value_endog)
    interior_consumption = jnp.interp(liquid_grid, liquid_endog, consumption)

    # Constrained: below the smallest endogenous liquid the borrowing constraint binds,
    # so the agent consumes all liquid and saves nothing (`next_liquid = income`).
    constrained = liquid_grid < liquid_endog[0]
    value_at_zero_savings = jnp.interp(income, liquid_grid, next_value)
    constrained_value = (
        crra_utility(liquid_grid, crra) + discount_factor * value_at_zero_savings
    )
    consumption_on_grid = jnp.where(constrained, liquid_grid, interior_consumption)
    value = jnp.where(constrained, constrained_value, interior_value)
    # Envelope theorem: the marginal value of liquid is the marginal utility of the
    # optimal consumption.
    marginal = consumption_on_grid ** (-crra)
    return RetiredEGMResult(
        value=value, marginal=marginal, consumption=consumption_on_grid
    )
