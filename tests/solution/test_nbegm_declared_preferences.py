"""The case-piece EGM step solves the felicity the regime declares.

`nbegm_one_asset_step` takes the three preference maps as a `Preferences` bundle —
felicity, its marginal, and its inverse marginal — exactly as the plain EGM step
does, and evaluates them. It assumes no preference family of its own, so a regime
whose felicity is not CRRA is solved as declared rather than silently replaced by
the CRRA form its parameters happen to resemble.

The witness is a subsistence floor, `u(c) = log(c - floor)`. No coefficient of
relative risk aversion expresses it, so a step that reached for a CRRA formula
would publish a different policy from the one the bundle describes.

The expectation is built from the candidate set the step documents. A boundary
below `income` makes both boundary-targeting branches infeasible and equal
subsidies put every node on one side of the case, so what remains is the Euler
interior path and the hard borrowing corner, and the larger value owns the node.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_step import nbegm_one_asset_step
from _lcm.egm.preferences import Preferences
from tests.conftest import DECIMAL_PRECISION

_CURRENT_GRID = (1.0, 2.0, 3.0)
_SAVINGS_GRID = (0.0, 1.0, 2.0)
_NEXT_GRID = (1.0, 2.5, 6.0)
_DISCOUNT = 0.95
_RETURN = 0.0
_INCOME = 1.0
_ASSET_LIMIT = 0.5
# Subsistence level of the felicity `log(c - floor)`; every grid node exceeds it.
_FLOOR = 0.5


def _subsistence_preferences() -> Preferences:
    """Felicity `log(c - floor)` with its marginal and inverse marginal."""
    return Preferences(
        utility=lambda consumption: jnp.log(consumption - _FLOOR),
        marginal_utility=lambda consumption: 1.0 / (consumption - _FLOOR),
        inverse_marginal_utility=lambda marginal: 1.0 / marginal + _FLOOR,
    )


def _expected_consumption() -> np.ndarray:
    """Optimal consumption over the step's candidate set, by hand.

    The Euler equation at savings `s` reads `1 / (c - floor) = beta * (1 + r) *
    V'(m')` with `m' = (1 + r) * s + income`, so `c = floor + 1 / (beta * (1 + r)
    * V'(m'))`. `V` and `V'` are linear interpolants on the next period's nodes.
    The borrowing corner consumes all cash-on-hand and lands next-period liquid at
    `income`.
    """
    next_grid = np.asarray(_NEXT_GRID, dtype=float)
    next_value = np.log(next_grid)
    next_marginal = 1.0 / next_grid
    savings = np.asarray(_SAVINGS_GRID, dtype=float)
    current = np.asarray(_CURRENT_GRID, dtype=float)

    gross = 1.0 + _RETURN
    next_wealth = gross * savings + _INCOME
    marginal_next = np.interp(next_wealth, next_grid, next_marginal)
    value_next = np.interp(next_wealth, next_grid, next_value)

    consumption = _FLOOR + 1.0 / (_DISCOUNT * gross * marginal_next)
    endogenous_wealth = consumption + savings
    node_value = np.log(consumption - _FLOOR) + _DISCOUNT * value_next

    bracketed = (current >= endogenous_wealth[0]) & (current <= endogenous_wealth[-1])
    interior_consumption = np.interp(current, endogenous_wealth, consumption)
    interior_value = np.where(
        bracketed, np.interp(current, endogenous_wealth, node_value), -np.inf
    )

    corner_value = np.log(current - _FLOOR) + _DISCOUNT * np.interp(
        _INCOME, next_grid, next_value
    )

    return np.where(interior_value >= corner_value, interior_consumption, current)


def test_the_step_solves_the_declared_felicity():
    """A subsistence-floor felicity yields its own optimum, not a CRRA one."""
    next_grid = jnp.asarray(_NEXT_GRID)
    _value, _marginal, consumption = nbegm_one_asset_step(
        next_value=jnp.log(next_grid),
        next_marginal=1.0 / next_grid,
        liquid_grid=jnp.asarray(_CURRENT_GRID),
        next_liquid_grid=next_grid,
        savings_grid=jnp.asarray(_SAVINGS_GRID),
        discount_factor=_DISCOUNT,
        preferences=_subsistence_preferences(),
        return_liquid=_RETURN,
        income=_INCOME,
        subsidy_when=0.0,
        subsidy_otherwise=0.0,
        asset_limit=_ASSET_LIMIT,
        equality_owner="otherwise",
    )

    np.testing.assert_array_almost_equal(
        np.asarray(consumption),
        _expected_consumption(),
        decimal=DECIMAL_PRECISION,
    )


def test_a_subsistence_floor_is_not_a_crra_policy():
    """The witness separates the two families, so the test above can discriminate.

    Without this, a step that ignored the bundle and solved CRRA could still match
    if the two happened to agree on this grid.
    """
    next_grid = np.asarray(_NEXT_GRID, dtype=float)
    savings = np.asarray(_SAVINGS_GRID, dtype=float)
    marginal_next = np.interp(savings + _INCOME, next_grid, 1.0 / next_grid)
    subsistence = _FLOOR + 1.0 / (_DISCOUNT * marginal_next)
    crra = 1.0 / (_DISCOUNT * marginal_next)
    assert not np.allclose(subsistence, crra)
