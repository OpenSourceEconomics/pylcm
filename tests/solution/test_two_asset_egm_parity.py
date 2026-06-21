"""One 2-D EGM step reproduces the brute-force value on the unconstrained interior.

The terminal value weights pension below liquid (`pension_bequest_weight < 1`), which
puts the marginal-value ratio into the interior-deposit band, so the unconstrained
region is non-empty and the EGM inversion has something to reproduce. On that interior
the assembled step — post-decision value and gradients, closed-form Euler inversion,
and the inverse-bilinear locator deposit — must agree with a dense grid-search solve.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.two_asset_step import egm_step
from tests.test_models.deterministic.two_asset import get_model, get_params

_P = {
    "discount_factor": 0.95,
    "crra": 2.0,
    "match_rate": 1.0,
    "return_liquid": 0.02,
    "return_pension": 0.06,
    "wage": 10.0,
}


def _egm_and_brute():
    model = get_model(n_periods=2)
    params = get_params(n_periods=2, pension_bequest_weight=0.5)
    brute = model.solve(params=params, log_level="off")
    next_value = jnp.asarray(brute[1]["dead"])

    m_grid = jnp.linspace(1.0, 100.0, 12)
    n_grid = jnp.linspace(0.0, 50.0, 10)
    # Keep m'(a) = 1.02a + 10 <= 100 and n'(b) = 1.06b <= 50 so the carried-forward
    # states stay on the next grid (off-grid clamps the gradient -> NaN inverse).
    a_grid = jnp.linspace(1.0, 85.0, 18)
    b_grid = jnp.linspace(0.5, 46.0, 16)

    egm = np.asarray(
        egm_step(
            next_value=next_value,
            m_grid=m_grid,
            n_grid=n_grid,
            a_grid=a_grid,
            b_grid=b_grid,
            **_P,
        )
    )
    return egm, np.asarray(brute[0]["working"])


def test_egm_step_matches_brute_on_unconstrained_interior():
    """EGM value matches the dense grid-search solve where the constraints are slack."""
    egm, brute_working = _egm_and_brute()
    rel = np.abs(egm - brute_working) / np.abs(brute_working)
    # Unconstrained interior: liquid large relative to pension, so the borrowing and
    # deposit constraints are slack and the endogenous cloud covers the targets.
    interior = rel[6:, :5]
    assert np.median(interior) < 0.01
    assert interior.max() < 0.03
