"""The 1-D EGM step reads next-period arrays on the next period's own grid.

`egm_one_asset_step` handles two grids in distinct roles: it publishes this period's
value, marginal and consumption on `liquid_grid`, and it reads `next_value` and
`next_marginal` — both tabulated on the *next* period's nodes — on
`next_liquid_grid`. The two coincide unless the liquid state is an
`AgeSpecializedGrid`, so using one for both roles is invisible until a grid moves,
and then it evaluates the continuation at wealth levels the next period's grid never
covers.

The expectations below come from solving the same problem in closed form, so they are
independent of how the production step happens to arrange its interpolations.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from tests.conftest import DECIMAL_PRECISION
from tests.solution._crra_preferences import crra_preferences

_CURRENT_GRID = (1.0, 2.0, 3.0)
_SAVINGS_GRID = (0.0, 1.0, 2.0)
_DISCOUNT = 0.95
_RETURN = 0.0
_INCOME = 1.0


def _closed_form_consumption(next_nodes: tuple[float, ...]) -> np.ndarray:
    """Optimal consumption for log utility, by hand.

    With `crra = 1` the continuation is `log`, so `u'(c) = 1/c` and the Euler
    equation at savings `s` reads `1/c = beta * (1 + r) * V'(m')`, where
    `m' = (1 + r) * s + income`. `V'` is the linear interpolant of `next_marginal`
    on `next_grid` — the only place the next period's nodes enter. Inverting gives
    the endogenous wealth `m = c + s`, and the policy on the current grid is that
    correspondence interpolated back, clamped by the borrowing constraint.
    """
    next_grid = np.asarray(next_nodes, dtype=float)
    next_marginal = 1.0 / next_grid
    savings = np.asarray(_SAVINGS_GRID, dtype=float)

    gross = 1.0 + _RETURN
    next_wealth = gross * savings + _INCOME
    marginal_next = np.interp(next_wealth, next_grid, next_marginal)
    consumption = 1.0 / (_DISCOUNT * gross * marginal_next)
    endogenous_wealth = consumption + savings

    current = np.asarray(_CURRENT_GRID, dtype=float)
    interior = np.interp(current, endogenous_wealth, consumption)
    # Below the smallest endogenous wealth the constraint binds: consume everything.
    return np.where(current < endogenous_wealth[0], current, interior)


@pytest.mark.parametrize(
    "next_nodes",
    [
        (1.0, 2.0, 3.0),  # unmoved — the age-invariant case must not shift
        (1.0, 3.0, 5.0),
        (0.5, 2.5, 4.5),
        (1.0, 2.5, 6.0),
    ],
)
def test_next_period_arrays_are_read_on_the_next_period_grid(next_nodes):
    """Consumption matches the closed form for any next-period grid."""
    next_grid = jnp.asarray(next_nodes)
    step = egm_one_asset_step(
        next_value=jnp.log(next_grid),
        next_marginal=1.0 / next_grid,
        liquid_grid=jnp.asarray(_CURRENT_GRID),
        next_liquid_grid=next_grid,
        savings_grid=jnp.asarray(_SAVINGS_GRID),
        discount_factor=_DISCOUNT,
        preferences=crra_preferences(1.0),
        return_liquid=_RETURN,
        income=_INCOME,
    )

    np.testing.assert_array_almost_equal(
        np.asarray(step.consumption),
        _closed_form_consumption(next_nodes),
        decimal=DECIMAL_PRECISION,
    )


def test_a_moved_next_period_grid_changes_the_policy():
    """A different next-period grid yields a different policy, all else equal.

    Without this the test above would still pass if `next_liquid_grid` were ignored
    and the current grid silently used for both roles.
    """
    moved = _closed_form_consumption((1.0, 3.0, 5.0))
    unmoved = _closed_form_consumption(_CURRENT_GRID)
    assert not np.allclose(moved, unmoved)
