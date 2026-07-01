"""Corner candidates cover high liquid and stay visible on a coarse liquid grid.

Two failure modes a piecewise-affine-budget EGM must avoid that a brute grid search does
not:

- **Upper-savings corner.** With a finite savings grid the Bellman optimum at high
  cash-on-hand can be to save at the top of the grid (`s = savings_grid[-1]`), not at an
  interior Euler point. Without that corner the recovered value is NaN or wrong where
  the Euler endogenous grid stops below the query.
- **Coarse-grid visibility.** When an interval contains only one (or zero) liquid grid
  point, a corner masked to grid nodes is a singleton the link-only upper envelope
  cannot bracket, so its value is lost. Every feasible corner must remain visible
  regardless of how many grid points fall inside its interval.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.bqsegm_step import bqsegm_per_interval_continuation_step_savings

_CRRA = 2.0
_DISCOUNT = 0.95


def _utility_of_action(consumption):
    return consumption ** (1.0 - _CRRA) / (1.0 - _CRRA)


def _inverse_marginal_utility(marginal_continuation):
    return marginal_continuation ** (-1.0 / _CRRA)


def test_high_liquid_value_uses_the_upper_savings_corner():
    """At high liquid where saving the grid maximum dominates, the value is finite.

    A continuation that rises steeply in savings makes `s = savings_grid[-1]` optimal at
    the top of the liquid grid, above where the interior Euler grid reaches. The solved
    value there must equal `u(coh - s_max) + beta * cont_value[-1]` and be finite.
    """
    n_liquid, n_savings = 40, 60
    liquid_grid = jnp.linspace(0.5, 50.0, n_liquid)
    savings_grid = jnp.linspace(0.0, 40.0, n_savings)
    # Continuation rises steeply in savings, so saving to the top is optimal at high
    # cash-on-hand.
    cont_value = jnp.linspace(0.0, 30.0, n_savings)[None, :]
    cont_marginal = jnp.linspace(0.5, 0.02, n_savings)[None, :]
    coh_slopes = jnp.asarray([1.0])
    coh_intercepts = jnp.asarray([2.0])

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
        breakpoints=jnp.asarray([], dtype=liquid_grid.dtype),
    )
    value = np.asarray(value)
    assert np.isfinite(value).all()

    # At the top liquid point the finite-domain optimum is checked against the exact
    # per-node Bellman max over the savings grid (what brute would compute).
    coh_top = 1.0 * float(liquid_grid[-1]) + 2.0
    consumption = coh_top - np.asarray(savings_grid)
    feasible = consumption > 0.0
    node_value = np.where(
        feasible,
        _utility_of_action(np.where(feasible, consumption, 1.0))
        + _DISCOUNT * np.asarray(cont_value[0]),
        -np.inf,
    )
    np.testing.assert_allclose(value[-1], node_value.max(), atol=1e-2, rtol=1e-2)


def test_corner_is_visible_when_an_interval_holds_one_grid_point():
    """A corner in a singleton interval is not lost by the link-only envelope.

    With a coarse liquid grid and breakpoints that put a single grid point inside an
    interval, that point's feasible corner must still be recovered as a finite value.
    """
    liquid_grid = jnp.asarray([1.0, 5.0, 20.0])
    savings_grid = jnp.linspace(0.0, 4.0, 40)
    # Three intervals via two breakpoints; each interval holds exactly one grid point.
    breakpoints = jnp.asarray([3.0, 10.0])
    coh_slopes = jnp.asarray([1.0, 1.0, 1.0])
    coh_intercepts = jnp.asarray([1.0, 1.0, 1.0])
    cont_value = jnp.broadcast_to(jnp.linspace(0.0, 2.0, 40), (3, 40))
    cont_marginal = jnp.broadcast_to(jnp.linspace(1.0, 0.1, 40), (3, 40))

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
    assert np.isfinite(np.asarray(value)).all()
