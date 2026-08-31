"""The corner is the savings grid's lower bound, not zero savings.

A household that cannot borrow saves at least nothing, and when the constraint
binds it consumes everything on hand. A household that *can* borrow, up to a
limit, saves at least minus that limit, and when the constraint binds it
consumes everything on hand plus the limit. Both are the same statement about
the lowest attainable savings, so the corner reads it off the grid rather than
assuming it is zero.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from tests.conftest import DECIMAL_PRECISION
from tests.solution._crra_preferences import crra_preferences

_BORROWING_LIMIT = 2.0
_SAVINGS_GRID = jnp.linspace(-_BORROWING_LIMIT, 10.0, 25)
_LIQUID_GRID = jnp.linspace(0.5, 4.0, 8)
_NEXT_LIQUID_GRID = jnp.linspace(-5.0, 30.0, 60)
_DISCOUNT_FACTOR = 0.95
_RETURN = 0.03
_INCOME = 1.0
_NEXT_VALUE_LEVEL = -0.25


def _step():
    """One step whose continuation is flat, so saving carries no marginal value."""
    gross_return = 1.0 + _RETURN
    return egm_one_asset_step(
        next_value=jnp.full_like(_NEXT_LIQUID_GRID, _NEXT_VALUE_LEVEL),
        next_marginal=jnp.zeros_like(_NEXT_LIQUID_GRID),
        liquid_grid=_LIQUID_GRID,
        next_liquid_grid=_NEXT_LIQUID_GRID,
        savings_grid=_SAVINGS_GRID,
        discount_factor=_DISCOUNT_FACTOR,
        preferences=crra_preferences(crra=2.0),
        next_liquid=gross_return * _SAVINGS_GRID + _INCOME,
        marginal_return=jnp.full_like(_SAVINGS_GRID, gross_return),
    )


def test_a_borrowing_household_consumes_its_wealth_plus_the_limit():
    """With no marginal value of saving the household borrows to the limit.

    Saving nothing is not the corner where borrowing is allowed: the household
    takes the lowest savings the grid permits, so consumption is wealth plus the
    borrowing limit rather than wealth alone.
    """
    np.testing.assert_array_almost_equal(
        np.asarray(_step().consumption),
        np.asarray(_LIQUID_GRID) + _BORROWING_LIMIT,
        decimal=DECIMAL_PRECISION,
    )


def test_the_constrained_value_discounts_the_landing_point_at_the_limit():
    """The continuation is reached from the limit, not from zero savings."""
    preferences = crra_preferences(crra=2.0)
    expected = (
        preferences.utility(_LIQUID_GRID + _BORROWING_LIMIT)
        + _DISCOUNT_FACTOR * _NEXT_VALUE_LEVEL
    )

    np.testing.assert_array_almost_equal(
        np.asarray(_step().value), np.asarray(expected), decimal=DECIMAL_PRECISION
    )
