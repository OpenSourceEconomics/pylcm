"""The per-interval NBEGM merge dominates the savings-node Bellman floor.

At every liquid grid point, choosing any feasible post-decision node directly —
consume that point's cash-on-hand minus the node, earn the node's own-interval
continuation — is an available action. The merged envelope value must weakly
dominate the best such action. The Euler interior path alone cannot guarantee
this: at a continuation kink between savings nodes (the marginal continuation
drops sharply, e.g. across a child-value interpolation node), the Euler
inversion has no root, so the optimum there must be carried by an explicit
savings-node candidate. Where the continuation is smooth, the floor must not
displace the interior Euler optimum between two nodes.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from _lcm.egm.comparison_arithmetic import ComparisonArithmetic
from _lcm.egm.nbegm_step import (
    nbegm_multi_interval_step_savings,
    nbegm_per_interval_continuation_step_savings,
)
from _lcm.egm.preferences import Preferences
from lcm.typing import FloatND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution._crra_preferences import crra_preferences

_CRRA = 3.84
_DISCOUNT = 0.95
_GROSS_RETURN = 1.03
_INCOME = 1.0


def _utility_of_action(consumption):
    return consumption ** (1.0 - _CRRA) / (1.0 - _CRRA)


def _inverse_marginal_utility(marginal_continuation):
    return marginal_continuation ** (-1.0 / _CRRA)


def test_merged_value_dominates_every_feasible_savings_node_action():
    """Each grid point's value is at least the best direct savings-node action.

    The continuation is a piecewise-linear child value read on a coarse child
    grid: its marginal drops by orders of magnitude at the savings node where
    next-period liquid crosses the child's interior grid node, so for high
    liquid the optimum sits at that kink — between Euler roots.
    """
    liquid_grid = jnp.array([0.1, 15.05, 30.0])
    savings_grid = jnp.linspace(0.0, 28.0, 180)
    breakpoints = jnp.array([12.0])
    coh_slopes = jnp.array([1.0, 0.7])
    coh_intercepts = jnp.array([1.0, 4.6])

    child_grid = jnp.array([0.1, 15.05, 30.0])
    child_value = child_grid ** (1.0 - 2.0) / (1.0 - 2.0)
    next_liquid = _GROSS_RETURN * savings_grid + _INCOME
    cont_row = jnp.interp(next_liquid, child_grid, child_value)
    segment_slopes = jnp.diff(child_value) / jnp.diff(child_grid)
    marginal_row = _GROSS_RETURN * jnp.where(
        next_liquid < child_grid[1], segment_slopes[0], segment_slopes[1]
    )
    cont_value = jnp.stack([cont_row, cont_row])
    cont_marginal = jnp.stack([marginal_row, marginal_row])

    value, _, _ = nbegm_per_interval_continuation_step_savings(
        cont_value=cont_value,
        cont_marginal=cont_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=jnp.asarray(_DISCOUNT),
        preferences=crra_preferences(crra=_CRRA),
        coh_slopes=coh_slopes,
        coh_intercepts=coh_intercepts,
        breakpoints=breakpoints,
    )

    interval_of_grid = np.searchsorted(np.asarray(breakpoints), liquid_grid, "right")
    coh = (
        np.asarray(coh_slopes)[interval_of_grid] * np.asarray(liquid_grid)
        + (np.asarray(coh_intercepts)[interval_of_grid])
    )
    consumption = coh[:, None] - np.asarray(savings_grid)[None, :]
    feasible = consumption > 0.0
    action_value = np.where(
        feasible,
        np.asarray(_utility_of_action(jnp.where(feasible, consumption, 1.0)))
        + _DISCOUNT * np.asarray(cont_value)[interval_of_grid],
        -np.inf,
    )
    floor = action_value.max(axis=1)
    np.testing.assert_array_less(
        floor - 1e-6,
        np.asarray(value),
        err_msg="merged value falls below the savings-node Bellman floor",
    )


def _taste_log_preferences(*, taste: float) -> Preferences:
    """Build the bundle `u(c) = taste * log(c)` with its marginal and inverse."""

    def utility(consumption: FloatND) -> FloatND:
        return taste * jnp.log(consumption)

    def marginal_utility(consumption: FloatND) -> FloatND:
        return taste / consumption

    def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
        return taste / marginal_continuation

    return Preferences(
        utility=utility,
        marginal_utility=marginal_utility,
        inverse_marginal_utility=inverse_marginal_utility,
    )


_HALF_COST_NODE = 0.5 * float(np.sqrt(3.0))


@pytest.mark.parametrize(
    "arithmetic",
    [
        "ordinary",
        pytest.param(
            "certified",
            marks=pytest.mark.requires_exact_affine_kernel(
                reason=EXACT_KERNEL_SKIP_REASON
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("coh_intercept", "taste"),
    [
        (2.0 + _HALF_COST_NODE, 1.0),
        (2.0 - _HALF_COST_NODE, 1.0),
        (3.3, 1.0),
        (2.0 - _HALF_COST_NODE, 1.5),
    ],
)
@pytest.mark.parametrize("per_interval", [False, True])
def test_off_node_query_publishes_the_interior_euler_policy(
    *,
    coh_intercept: float,
    taste: float,
    arithmetic: ComparisonArithmetic,
    per_interval: bool,
) -> None:
    """Between savings nodes the published policy is the interior Euler solution.

    Felicity `taste * log(c)` against the continuation `log(s)` with no discounting
    has its optimum at `c = taste * R / (1 + taste)` for cash-on-hand `R`. The
    savings grid steps by 1/2 and these cash-on-hand levels put every optimal saving
    strictly between two nodes, where saving the nearest node is a feasible but
    worse action: the published value stays at or above that node's value, and the
    published policy is the interior optimum. The per-interval step reads the same
    continuation row in each of two intervals of one smooth budget.
    """
    liquid_grid = jnp.linspace(1.0, 5.0, 5)
    savings_grid = jnp.linspace(0.5, 6.0, 12)
    preferences = _taste_log_preferences(taste=taste)
    if per_interval:
        value, _, policy = nbegm_per_interval_continuation_step_savings(
            cont_value=jnp.tile(jnp.log(savings_grid), (2, 1)),
            cont_marginal=jnp.tile(1.0 / savings_grid, (2, 1)),
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            discount_factor=jnp.asarray(1.0),
            preferences=preferences,
            coh_slopes=jnp.ones(2),
            coh_intercepts=jnp.full(2, coh_intercept),
            breakpoints=jnp.array([2.5]),
            arithmetic=arithmetic,
        )
    else:
        value, _, policy = nbegm_multi_interval_step_savings(
            cont_value=jnp.log(savings_grid),
            cont_marginal=1.0 / savings_grid,
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            discount_factor=jnp.asarray(1.0),
            preferences=preferences,
            coh_slopes=jnp.ones(1),
            coh_intercepts=jnp.full(1, coh_intercept),
            breakpoints=jnp.zeros(0),
            arithmetic=arithmetic,
        )

    coh = np.asarray(liquid_grid) + coh_intercept
    nodes = np.asarray(savings_grid)
    assert not np.isin(coh / (1.0 + taste), nodes).any()
    assert_array_almost_equal(
        policy, taste * coh / (1.0 + taste), decimal=DECIMAL_PRECISION
    )
    node_consumption = coh[:, None] - nodes[None, :]
    node_value = np.where(
        node_consumption > 0.0,
        taste * np.log(np.where(node_consumption > 0.0, node_consumption, 1.0))
        + np.log(nodes)[None, :],
        -np.inf,
    )
    np.testing.assert_array_less(
        node_value.max(axis=1) - 10.0**-DECIMAL_PRECISION, np.asarray(value)
    )
