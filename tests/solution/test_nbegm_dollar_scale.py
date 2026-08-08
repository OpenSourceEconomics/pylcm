"""The Euler branch survives a dollar-denominated budget.

Marginal utility carries the units of `c**(-crra)`, so at dollar magnitudes and
a moderate risk aversion it is a very small number while the inversion itself
stays perfectly well conditioned. The degeneracy test therefore looks at what
the inversion produced — a non-positive marginal, or a consumption that is not
finite — rather than at the marginal's magnitude, so re-denominating a model
from ten-thousands to dollars cannot silently discard its whole Euler branch
and leave a corner-only policy the solve still reports as successful.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_step import nbegm_multi_interval_step
from tests.solution._nbegm_step_helpers import dense_brute_value

CRRA = 3.0
DISCOUNT_FACTOR = 0.95
GROSS_RETURN = 1.05
N_LIQUID = 61
N_SAVINGS = 181


def _solve_at_scale(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the same CRRA problem denominated in units of `scale`.

    Returns the liquid grid, the published value, and the published consumption
    policy. Under CRRA the problem is homothetic, so every money quantity
    scales with `scale` and the value scales by `scale ** (1 - crra)`.
    """
    liquid_grid = jnp.linspace(1.0, 10.0, N_LIQUID) * scale
    savings_grid = jnp.linspace(0.0, 9.0, N_SAVINGS) * scale
    next_value = liquid_grid ** (1.0 - CRRA) / (1.0 - CRRA)
    next_marginal = liquid_grid ** (-CRRA)
    value, _marginal, policy = nbegm_multi_interval_step(
        next_value=next_value,
        next_marginal=next_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=DISCOUNT_FACTOR,
        crra=CRRA,
        gross_return=GROSS_RETURN,
        income=0.5 * scale,
        coh_slopes=jnp.asarray([1.0]),
        coh_intercepts=jnp.asarray([0.0]),
        breakpoints=jnp.zeros((0,)),
    )
    return np.asarray(liquid_grid), np.asarray(value), np.asarray(policy)


def test_dollar_scaled_marginals_match_the_dense_oracle():
    """At dollar magnitudes the step still tracks a dense consumption search.

    Every node's marginal continuation is around `1e-13` here, far below any
    fixed absolute floor, yet each inversion returns a finite consumption — so
    the interior path competes in the envelope exactly as at any other
    denomination and the remaining gap is grid discretization.
    """
    liquid_grid, value, _policy = _solve_at_scale(1.0e4)
    expected = dense_brute_value(
        liquid_grid=jnp.asarray(liquid_grid),
        coh_of_liquid=lambda liquid: liquid,
        next_value_of_liquid=lambda liquid: jnp.interp(
            liquid,
            jnp.asarray(liquid_grid),
            jnp.asarray(liquid_grid) ** (1.0 - CRRA) / (1.0 - CRRA),
        ),
        crra=CRRA,
        discount_factor=DISCOUNT_FACTOR,
        gross_return=GROSS_RETURN,
        income=0.5e4,
    )
    np.testing.assert_allclose(value, np.asarray(expected), rtol=5e-3)


def test_the_policy_is_invariant_to_the_money_denomination():
    """Re-denominating a CRRA model rescales its policy and nothing else.

    The same problem stated in dollars and in ten-thousands of dollars differs
    only by the unit, so the dollar policy is exactly `1e4` times the
    ten-thousands one.
    """
    _grid_units, _value_units, policy_units = _solve_at_scale(1.0)
    _grid_dollars, _value_dollars, policy_dollars = _solve_at_scale(1.0e4)
    np.testing.assert_allclose(policy_dollars, 1.0e4 * policy_units, rtol=1e-6)
