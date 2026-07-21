"""A savings-node action feasible at a single liquid point stays on the envelope.

At the top liquid grid point a high-savings post-decision node can be the only
feasible one — its consumption is positive nowhere else — so its candidate is a
lone point. The link-based query envelope brackets a candidate only through a
consecutive same-segment pair, so a lone point needs its own zero-width segment
to stay visible. The merged value must weakly dominate every feasible
savings-node action at every liquid point, never collapsing a single-point
winner to a lower multi-point branch.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_step import nbegm_multi_interval_step_savings

_CRRA = 2.0
_DISCOUNT = 0.95


def _utility_of_action(consumption):
    return consumption ** (1.0 - _CRRA) / (1.0 - _CRRA)


def _inverse_marginal_utility(marginal_continuation):
    return marginal_continuation ** (-1.0 / _CRRA)


def test_single_point_savings_node_action_stays_on_the_envelope():
    """Each liquid point's value is at least its best feasible savings-node action.

    A high-savings node feasible only at the top liquid point is a lone
    candidate; the envelope brackets it at its own abscissa rather than dropping
    it to a multi-point branch below. Here the top point's winner is the
    `savings = 3.0` node (value `6.6`), feasible at no lower liquid point.
    """
    liquid_grid = jnp.array([0.5, 1.5, 2.5, 4.0])
    savings_grid = jnp.array([0.0, 1.0, 2.0, 3.0, 3.9])
    breakpoints = jnp.array([], dtype=jnp.float32)
    coh_slopes = jnp.array([1.0])
    coh_intercepts = jnp.array([0.0])
    cont_marginal = jnp.array([2.0, 1.0, 0.5, 0.2, 0.1])
    cont_value = jnp.array([1.0, 3.0, 5.0, 8.0, 12.0])

    value, _, _ = nbegm_multi_interval_step_savings(
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

    coh = (
        np.asarray(coh_slopes)[0] * np.asarray(liquid_grid)
        + np.asarray(coh_intercepts)[0]
    )
    consumption = coh[:, None] - np.asarray(savings_grid)[None, :]
    feasible = consumption > 0.0
    action_value = np.where(
        feasible,
        np.asarray(_utility_of_action(jnp.where(feasible, consumption, 1.0)))
        + _DISCOUNT * np.asarray(cont_value)[None, :],
        -np.inf,
    )
    floor = action_value.max(axis=1)

    np.testing.assert_array_less(
        floor - 1e-5,
        np.asarray(value),
        err_msg="single-point savings-node winner dropped by the link-only envelope",
    )
