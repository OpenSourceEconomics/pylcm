"""The 1-D retirement EGM step reproduces the brute solve of the retired sub-problem.

The retired agent solves a plain consumption--saving problem: one continuous state
(`liquid`), one continuous action (`consumption`), no discrete choice, so the endogenous
grid method needs no upper envelope — invert the consumption Euler equation on a
post-decision savings grid and map back to the liquid grid. Run against the DS pension
brute solve, the step matches the dense grid-search retired value where the grid covers.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from tests.test_models.deterministic.ds_pension import get_model, get_params

_LIQUID_GRID = jnp.linspace(0.1, 20.0, 12)
_SAVINGS_GRID = jnp.linspace(0.0, 20.0, 40)
_P = {
    "discount_factor": 0.98,
    "crra": 2.0,
    "return_liquid": 0.02,
    "income": 0.50,
}
# The two lowest liquid points sit in the borrowing-constrained boundary layer, where
# the value has a kink and a coarse linear grid cannot resolve the steep low-wealth
# value; both EGM and grid search carry boundary error there. The unconstrained
# interior is what the parity assertion covers.
_INTERIOR = np.s_[2:]


def _brute_retired(*, n_consumption=200):
    """Brute retired value, solved with a fine consumption grid as the fair reference.

    The default oracle's 14-point consumption grid is too coarse to be a faithful
    reference at low liquid (the optimal consumption is small and falls between grid
    points); a fine grid converges to the true value the EGM step targets.
    """
    model = get_model(n_periods=5, n_consumption=n_consumption)
    brute = model.solve(params=get_params(), log_level="off")
    return jnp.asarray(brute[4]["dead"]), np.asarray(brute[3]["retired"])


def _bequest_marginal():
    """Marginal value of liquid for the terminal CRRA bequest `u(liquid)`."""
    return _LIQUID_GRID ** (-_P["crra"])


def test_retired_egm_matches_brute_on_the_liquid_interior():
    """One retired EGM step from the terminal bequest matches the brute solve."""
    v_dead, brute_retired = _brute_retired()
    v_retired, _marginal = egm_one_asset_step(
        next_value=v_dead,
        next_marginal=_bequest_marginal(),
        liquid_grid=_LIQUID_GRID,
        savings_grid=_SAVINGS_GRID,
        **_P,
    )
    v_retired = np.asarray(v_retired)
    assert np.isfinite(v_retired).all()
    rel = np.abs(v_retired[_INTERIOR] - brute_retired[_INTERIOR]) / np.abs(
        brute_retired[_INTERIOR]
    )
    assert np.median(rel) < 0.01
    assert np.max(rel) < 0.05


def test_retired_egm_value_is_increasing_in_liquid():
    """More liquid wealth is weakly more valuable in retirement."""
    v_dead, _brute = _brute_retired()
    v_retired, marginal = egm_one_asset_step(
        next_value=v_dead,
        next_marginal=_bequest_marginal(),
        liquid_grid=_LIQUID_GRID,
        savings_grid=_SAVINGS_GRID,
        **_P,
    )
    assert np.all(np.diff(np.asarray(v_retired)) >= -1e-6)
    # The marginal value of liquid is positive and decreasing (concave value).
    assert np.all(np.asarray(marginal) > 0)
    assert np.all(np.diff(np.asarray(marginal)) <= 1e-6)
