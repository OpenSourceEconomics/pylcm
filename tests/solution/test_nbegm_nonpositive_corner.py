"""No NB-EGM step publishes a corner at a non-positive cash-on-hand.

Consuming the whole budget is an action only where the budget is positive. A
CRRA utility evaluated at a non-positive budget still returns a number — at an
even-integer `crra` a *larger* number than `u(c)` for any feasible `c`, and at
exactly zero an infinity — so an unguarded corner wins the upper envelope
wherever it brackets and reports a positive value, a negative consumption
policy, or a magnitude no feasible action can reach.

Every step therefore NaN-deads its corner where cash-on-hand is non-positive:
the liquid points with no feasible action publish NaN, and the points that do
have one are untouched by the corner that could not exist beside them.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_step import (
    _case_step,
    _interval_corner_candidates,
    _recurring_jump_case,
    nbegm_multi_interval_step,
)
from tests.solution._nbegm_step_helpers import crra_utility, dense_brute_value

CRRA = 2.0
DISCOUNT_FACTOR = 0.95
RETURN_LIQUID = 0.05
GROSS_RETURN = 1.0 + RETURN_LIQUID
INCOME = 0.0

# Starts strictly below zero, so the first three nodes carry a non-positive
# cash-on-hand under an identity budget and the rest a positive one.
LIQUID_GRID = jnp.linspace(-1.0, 5.0, 13)
SAVINGS_GRID = jnp.linspace(0.0, 5.0, 21)
NEXT_VALUE = -1.0 / (LIQUID_GRID + 2.0)
NEXT_MARGINAL = 1.0 / (LIQUID_GRID + 2.0) ** 2

INFEASIBLE = np.asarray(LIQUID_GRID) <= 0.0


def _next_value_of_liquid(liquid: jnp.ndarray) -> jnp.ndarray:
    return jnp.interp(liquid, LIQUID_GRID, NEXT_VALUE)


def _brute_where_feasible() -> np.ndarray:
    """Dense Bellman max on the positive-budget nodes of the identity budget."""
    feasible_grid = LIQUID_GRID[~INFEASIBLE]
    return np.asarray(
        dense_brute_value(
            liquid_grid=feasible_grid,
            coh_of_liquid=lambda liquid: liquid,
            next_value_of_liquid=_next_value_of_liquid,
            crra=CRRA,
            discount_factor=DISCOUNT_FACTOR,
            gross_return=GROSS_RETURN,
            income=INCOME,
        )
    )


def test_case_step_publishes_nan_where_cash_on_hand_is_non_positive():
    """The per-case step has no candidate below a zero budget, so it reports NaN."""
    value, _marginal, _policy = _case_step(
        next_value=NEXT_VALUE,
        next_marginal=NEXT_MARGINAL,
        liquid_grid=LIQUID_GRID,
        savings_grid=SAVINGS_GRID,
        discount_factor=DISCOUNT_FACTOR,
        crra=CRRA,
        return_liquid=RETURN_LIQUID,
        income=INCOME,
        subsidy=0.0,
        asset_limit=3.0,
        equality_owner="otherwise",
    )
    assert np.isnan(np.asarray(value)[INFEASIBLE]).all()


def test_case_step_matches_the_dense_brute_on_the_feasible_nodes():
    """A dropped infeasible corner leaves the feasible nodes at the true optimum."""
    value, _marginal, _policy = _case_step(
        next_value=NEXT_VALUE,
        next_marginal=NEXT_MARGINAL,
        liquid_grid=LIQUID_GRID,
        savings_grid=SAVINGS_GRID,
        discount_factor=DISCOUNT_FACTOR,
        crra=CRRA,
        return_liquid=RETURN_LIQUID,
        income=INCOME,
        subsidy=0.0,
        # Above the grid, so the case is an ordinary smooth-continuation solve
        # and the dense brute is a like-for-like oracle.
        asset_limit=99.0,
        equality_owner="otherwise",
    )
    np.testing.assert_allclose(
        np.asarray(value)[~INFEASIBLE], _brute_where_feasible(), rtol=2e-3
    )


def test_multi_interval_step_publishes_nan_where_cash_on_hand_is_non_positive():
    """The piecewise-affine step drops its borrowing corner below a zero budget."""
    value, _marginal, _policy = nbegm_multi_interval_step(
        next_value=NEXT_VALUE,
        next_marginal=NEXT_MARGINAL,
        liquid_grid=LIQUID_GRID,
        savings_grid=SAVINGS_GRID,
        discount_factor=DISCOUNT_FACTOR,
        crra=CRRA,
        gross_return=GROSS_RETURN,
        income=INCOME,
        coh_slopes=jnp.asarray([1.0]),
        coh_intercepts=jnp.asarray([0.0]),
        breakpoints=jnp.zeros((0,)),
    )
    assert np.isnan(np.asarray(value)[INFEASIBLE]).all()


def test_recurring_jump_case_marks_its_non_positive_corner_dead():
    """The recurring-cliff case emits no live corner at a non-positive budget."""
    _value, _policy, _marginal, endog, _segment = _recurring_jump_case(
        next_value=NEXT_VALUE,
        next_marginal=NEXT_MARGINAL,
        liquid_grid=LIQUID_GRID,
        savings_grid=SAVINGS_GRID,
        discount_factor=DISCOUNT_FACTOR,
        crra=CRRA,
        gross_return=GROSS_RETURN,
        income=INCOME,
        subsidy=0.0,
        jump_breakpoints=jnp.asarray([2.0]),
        equality_owner="otherwise",
    )
    # The corner is the trailing block of `liquid_grid`-length columns.
    corner_endog = np.asarray(endog)[-LIQUID_GRID.shape[0] :]
    assert np.isnan(corner_endog[INFEASIBLE]).all()


def test_interval_corner_is_dead_when_the_whole_floor_is_infeasible():
    """A flat interval whose budget is non-positive publishes no corner at all.

    On a flat (hard-constraint) interval the budget is the constant intercept.
    A non-positive intercept makes every savings node infeasible, so the corner
    must be dead rather than carry the infeasibility sentinel into the
    envelope, where it evaluates to NaN across the whole interval.
    """
    assert np.isnan(_infeasible_interval_corner_value(flat=True)).all()


def test_interval_corner_is_dead_on_a_sloped_non_positive_budget():
    """A sloped interval whose budget is non-positive publishes no corner either."""
    assert np.isnan(_infeasible_interval_corner_value(flat=False)).all()


def _infeasible_interval_corner_value(*, flat: bool) -> np.ndarray:
    """The no-save corner's value channel on an entirely infeasible interval."""
    channels = _interval_corner_candidates(
        corner_coh_grid=jnp.full_like(LIQUID_GRID, -0.5),
        liquid_grid=LIQUID_GRID,
        savings_grid=SAVINGS_GRID,
        lower=jnp.asarray(-jnp.inf),
        upper=jnp.asarray(jnp.inf),
        flat=jnp.asarray(flat),
        value_at_no_save=jnp.asarray(0.0),
        interval_value=jnp.zeros_like(SAVINGS_GRID),
        coh_slope=jnp.asarray(0.0 if flat else 1.0),
        coh_intercept=jnp.asarray(-0.5),
        discount_factor=jnp.asarray(DISCOUNT_FACTOR),
        utility_of_action=lambda consumption: crra_utility(consumption, CRRA),
        marginal_utility=lambda consumption: consumption ** (-CRRA),
        base=jnp.asarray(0.0),
        next_segment=jnp.asarray(0.0),
    )
    return np.asarray(channels[1])
