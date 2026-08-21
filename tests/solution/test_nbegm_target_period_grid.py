"""The case-piece EGM step reads next-period arrays on the next period's own grid.

`nbegm_one_asset_step` handles two grids in distinct roles, exactly as
`egm_one_asset_step` does: it publishes this period's value, marginal and
consumption on `liquid_grid`, and it reads `next_value` and `next_marginal` — both
tabulated on the *next* period's nodes — on `next_liquid_grid`. The two coincide
unless the liquid state is an `AgeSpecializedGrid`, so using one for both roles is
invisible until a grid moves, and then it evaluates the continuation at wealth
levels the next period's grid never covers.

The expectation is built here from the candidate set the step documents, so it is
independent of how production arranges its interpolations. Two choices keep that
candidate set small enough to write down: equal subsidies put every node on one
side of the case boundary, and a boundary below `income` makes both
boundary-targeting branches infeasible (their `s_kink` is negative). What remains
is the Euler interior path and the hard borrowing corner, and the published value
is the larger of the two.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_step import nbegm_one_asset_step
from tests.conftest import DECIMAL_PRECISION
from tests.solution._crra_preferences import crra_preferences

_CURRENT_GRID = (1.0, 2.0, 3.0)
_SAVINGS_GRID = (0.0, 1.0, 2.0)
_DISCOUNT = 0.95
_RETURN = 0.0
_INCOME = 1.0
# Below every current node and below `income`, so one case owns the whole grid and
# neither boundary-targeting branch is feasible.
_ASSET_LIMIT = 0.5


def _expected_consumption(next_nodes: tuple[float, ...]) -> np.ndarray:
    """Optimal consumption over the step's candidate set, by hand.

    With `crra = 1` the felicity is `log`, so `u'(c) = 1/c` and the Euler equation
    at savings `s` reads `1/c = beta * (1 + r) * V'(m')`, where
    `m' = (1 + r) * s + income`. Both `V` and `V'` are linear interpolants on the
    *next* period's nodes — the only place those nodes enter. The interior path is
    the resulting `(endogenous wealth, consumption, value)` correspondence read at
    the current grid; the borrowing corner consumes all cash-on-hand and lands
    next-period liquid at `income`. Whichever carries the larger value owns the
    node.
    """
    next_grid = np.asarray(next_nodes, dtype=float)
    next_value = np.log(next_grid)
    next_marginal = 1.0 / next_grid
    savings = np.asarray(_SAVINGS_GRID, dtype=float)
    current = np.asarray(_CURRENT_GRID, dtype=float)

    gross = 1.0 + _RETURN
    next_wealth = gross * savings + _INCOME
    marginal_next = np.interp(next_wealth, next_grid, next_marginal)
    value_next = np.interp(next_wealth, next_grid, next_value)

    consumption = 1.0 / (_DISCOUNT * gross * marginal_next)
    endogenous_wealth = consumption + savings
    node_value = np.log(consumption) + _DISCOUNT * value_next

    # EGM publishes value and policy as linear interpolants over the endogenous
    # nodes, so the interior candidate is read the same way -- and only where the
    # correspondence actually brackets the query.
    bracketed = (current >= endogenous_wealth[0]) & (current <= endogenous_wealth[-1])
    interior_consumption = np.interp(current, endogenous_wealth, consumption)
    interior_value = np.where(
        bracketed, np.interp(current, endogenous_wealth, node_value), -np.inf
    )

    corner_consumption = current
    corner_value = np.log(current) + _DISCOUNT * np.interp(
        _INCOME, next_grid, next_value
    )

    return np.where(
        interior_value >= corner_value, interior_consumption, corner_consumption
    )


@pytest.mark.parametrize(
    "next_nodes",
    [
        (1.0, 2.0, 3.0),  # unmoved — the age-invariant case must not shift
        (1.0, 3.0, 5.0),
        (0.5, 2.5, 4.5),
        (1.0, 2.5, 6.0),
        (1.0, 6.0),  # coarser than this period's grid
        (0.5, 1.2, 2.0, 3.1, 4.4, 6.0),  # finer than this period's grid
    ],
)
def test_next_period_arrays_are_read_on_the_next_period_grid(next_nodes):
    """Consumption matches the hand calculation for any next-period grid."""
    next_grid = jnp.asarray(next_nodes)
    _value, _marginal, consumption = nbegm_one_asset_step(
        next_value=jnp.log(next_grid),
        next_marginal=1.0 / next_grid,
        liquid_grid=jnp.asarray(_CURRENT_GRID),
        next_liquid_grid=next_grid,
        savings_grid=jnp.asarray(_SAVINGS_GRID),
        discount_factor=_DISCOUNT,
        preferences=crra_preferences(1.0),
        next_liquid=(1.0 + _RETURN) * jnp.asarray(_SAVINGS_GRID) + _INCOME,
        marginal_return=jnp.full_like(jnp.asarray(_SAVINGS_GRID), 1.0 + _RETURN),
        subsidy_when=0.0,
        subsidy_otherwise=0.0,
        asset_limit=_ASSET_LIMIT,
        equality_owner="otherwise",
    )

    np.testing.assert_array_almost_equal(
        np.asarray(consumption),
        _expected_consumption(next_nodes),
        decimal=DECIMAL_PRECISION,
    )


def test_a_moved_next_period_grid_changes_the_policy():
    """A different next-period grid yields a different policy, all else equal.

    Without this the test above would still pass if `next_liquid_grid` were ignored
    and the current grid silently used for both roles.
    """
    moved = _expected_consumption((1.0, 3.0, 5.0))
    unmoved = _expected_consumption(_CURRENT_GRID)
    assert not np.allclose(moved, unmoved)
