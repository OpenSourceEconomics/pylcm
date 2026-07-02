"""The per-interval BQSEGM merge dominates the savings-node Bellman floor.

At every liquid grid point, choosing any feasible post-decision node directly —
consume that point's cash-on-hand minus the node, earn the node's own-interval
continuation — is an available action. The merged envelope value must weakly
dominate the best such action. The Euler interior path alone cannot guarantee
this: at a continuation kink between savings nodes (the marginal continuation
drops sharply, e.g. across a child-value interpolation node), the Euler
inversion has no root, so the optimum there must be carried by an explicit
savings-node candidate.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.bqsegm_step import bqsegm_per_interval_continuation_step_savings

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

    value, _, _ = bqsegm_per_interval_continuation_step_savings(
        cont_value=cont_value,
        cont_marginal=cont_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=jnp.asarray(_DISCOUNT),
        utility_of_action=_utility_of_action,
        inverse_marginal_utility=_inverse_marginal_utility,
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
