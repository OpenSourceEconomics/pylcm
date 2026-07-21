"""A flat budget interval (consumption floor binds) yields a finite value.

Where `resources = max(cash_on_hand, floor)` is flat in the liquid state — the
consumption floor binds at low/negative assets — the interval's cash-on-hand is
constant, so the EGM endogenous grid of coh-values is degenerate and the Euler inversion
cannot recover a liquid-varying policy. The solved value there must still be finite and
equal the single-point Bellman max over savings at the constant budget (what brute-force
computes), constant across the flat interval's liquid range.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_step import nbegm_per_interval_continuation_step_savings

_CRRA = 2.0
_DISCOUNT = 0.95
_N_SAVINGS = 120
_N_LIQUID = 60
_FLOOR_COH = 5.0


def _utility_of_action(consumption):
    return consumption ** (1.0 - _CRRA) / (1.0 - _CRRA)


def _inverse_marginal_utility(marginal_continuation):
    return marginal_continuation ** (-1.0 / _CRRA)


def test_flat_budget_interval_value_matches_dense_savings_max():
    """A flat (slope-0) budget interval yields the constant dense-savings-max value."""
    liquid_grid = jnp.linspace(-5.0, 20.0, _N_LIQUID)
    savings_grid = jnp.linspace(0.0, 18.0, _N_SAVINGS)
    cont_value = jnp.linspace(1.0, 5.0, _N_SAVINGS)[None, :]
    cont_marginal = jnp.linspace(1.0, 0.2, _N_SAVINGS)[None, :]

    value, _, _ = nbegm_per_interval_continuation_step_savings(
        cont_value=cont_value,
        cont_marginal=cont_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=jnp.asarray(_DISCOUNT),
        utility_of_action=_utility_of_action,
        inverse_marginal_utility=_inverse_marginal_utility,
        coh_slopes=jnp.asarray([0.0]),
        coh_intercepts=jnp.asarray([_FLOOR_COH]),
        breakpoints=jnp.asarray([], dtype=liquid_grid.dtype),
    )
    value = np.asarray(value)

    consumption = _FLOOR_COH - np.asarray(savings_grid)
    feasible = consumption > 0.0
    node_value = np.where(
        feasible,
        _utility_of_action(np.where(feasible, consumption, 1.0))
        + _DISCOUNT * np.asarray(cont_value[0]),
        -np.inf,
    )
    expected = node_value.max()

    assert np.isfinite(value).all()
    np.testing.assert_allclose(value, expected, atol=1e-4, rtol=1e-4)
