"""A dead Euler path does not take the corners down with it.

Where the continuation is flat the Euler inversion sends consumption to
infinity, so every interior candidate is dropped and the whole interior block
carries a NaN segment id. The corners that remain — the constant-budget floor
optimum and the hard borrowing corner — are still well defined, so the step
must publish their envelope rather than an all-NaN row.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_step import nbegm_multi_interval_step

CRRA = 2.0
DISCOUNT_FACTOR = 0.95
GROSS_RETURN = 1.05
INCOME = 0.5
FLOOR_COH = 1.0
KINK = 2.0

LIQUID_GRID = jnp.linspace(0.1, 5.0, 13)
SAVINGS_GRID = jnp.linspace(0.0, 5.0, 21)
# A constant continuation: zero marginal value of liquid everywhere, so saving
# buys nothing and the Euler inversion is degenerate at every savings node.
NEXT_VALUE = jnp.full_like(LIQUID_GRID, -0.5)
NEXT_MARGINAL = jnp.zeros_like(LIQUID_GRID)


def test_flat_continuation_publishes_the_corner_envelope_not_an_all_nan_row():
    """With no interior candidate left, the value is `u(coh) + beta * V'`.

    Below the kink a hard-constraint floor pins cash-on-hand at a constant; above
    it the budget slopes. A constant continuation makes consuming everything
    optimal in both, so the value is the no-save corner's throughout.
    """
    value, _marginal, policy = nbegm_multi_interval_step(
        next_value=NEXT_VALUE,
        next_marginal=NEXT_MARGINAL,
        liquid_grid=LIQUID_GRID,
        savings_grid=SAVINGS_GRID,
        discount_factor=DISCOUNT_FACTOR,
        crra=CRRA,
        gross_return=GROSS_RETURN,
        income=INCOME,
        coh_slopes=jnp.asarray([0.0, 1.0]),
        coh_intercepts=jnp.asarray([FLOOR_COH, -1.0]),
        breakpoints=jnp.asarray([KINK]),
        flat_interval_mask=(True, False),
    )
    liquid = np.asarray(LIQUID_GRID)
    coh = np.where(liquid < KINK, FLOOR_COH, liquid - 1.0)
    expected = coh ** (1.0 - CRRA) / (1.0 - CRRA) + DISCOUNT_FACTOR * (-0.5)
    np.testing.assert_allclose(np.asarray(value), expected, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(policy), coh, rtol=1e-6)
