"""The save-to-cliff candidate publishes the marginal value of liquid.

A boundary-targeting candidate saves to land next-period liquid just inside a
cliff and consumes the case's cash-on-hand. On an affine cash-on-hand interval
`coh(a) = sigma * a + iota`, the saving that targets the fixed cliff is
independent of the liquid state, so consumption moves one-for-`sigma` with the
state and the marginal value of liquid is `sigma * c**(-crra)`. The candidate
must publish that slope-scaled marginal, matching the interior and corner
candidates, so parent-period Euler inversion reads the correct marginal when the
save-to-cliff candidate wins.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.bqsegm_step import _boundary_targeting_coh


def test_boundary_targeting_marginal_scales_with_the_cash_on_hand_slope():
    """The published marginal is `coh_slope * c**(-crra)`, not `c**(-crra)`."""
    crra = 2.0
    coh_slope_value = 2.0
    liquid_grid = jnp.linspace(1.0, 10.0, 8)
    coh_case_grid = coh_slope_value * liquid_grid + 4.0
    coh_slope = jnp.full_like(liquid_grid, coh_slope_value)
    next_value = jnp.linspace(-5.0, -1.0, 8)

    _endog, _value, policy, marginal = _boundary_targeting_coh(
        liquid_grid=liquid_grid,
        coh_case_grid=coh_case_grid,
        next_value=next_value,
        discount_factor=0.95,
        crra=crra,
        gross_return=1.0,
        income=0.0,
        asset_limit=3.0,
        prev_limit=liquid_grid[0] - 1.0,
        coh_slope=coh_slope,
        valid=jnp.ones_like(liquid_grid, dtype=bool),
    )
    live = np.isfinite(np.asarray(marginal))
    expected = coh_slope_value * np.asarray(policy)[live] ** (-crra)
    np.testing.assert_allclose(np.asarray(marginal)[live], expected, rtol=1e-6)
