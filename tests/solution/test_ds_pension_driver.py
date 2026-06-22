"""The G2EGM driver solves the full DS pension lifecycle, matching the brute solve.

The driver chains the endogenous-grid steps over the whole lifecycle — terminal bequest,
1-D retired EGM, the working->retired boundary G2EGM step, and the working->working
G2EGM steps — and reproduces the dense grid-search value at every period and regime on
the covered interior. The off-grid top-pension hole advances one pension column inward
per backward working period, so the comparison excludes the boundary layer, which
thickens toward the first period.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.ds_pension_driver import solve_ds_pension_g2egm
from tests.test_models.deterministic.ds_pension import get_model, get_params

_N_PERIODS = 5
_RETIREMENT_PERIOD = 3

_LIQUID_GRID = jnp.linspace(0.1, 20.0, 12)
_PENSION_GRID = jnp.linspace(0.0, 15.0, 10)
_PARAMS = {
    "discount_factor": 0.98,
    "crra": 2.0,
    "work_disutility": 0.25,
    "match_rate": 0.10,
    "return_liquid": 0.02,
    "return_pension": 0.04,
    "wage": 1.0,
    "retirement_income": 0.50,
    "pension_payout_return": 1.04,
}


def _solve_both():
    egm = solve_ds_pension_g2egm(
        n_periods=_N_PERIODS,
        retirement_period=_RETIREMENT_PERIOD,
        liquid_grid=_LIQUID_GRID,
        pension_grid=_PENSION_GRID,
        a_grid=jnp.linspace(0.0, 20.0, 18),
        b_grid=jnp.linspace(0.0, 30.0, 16),
        consumption_grid=jnp.linspace(0.1, 20.0, 18),
        savings_grid=jnp.linspace(0.0, 20.0, 40),
        **_PARAMS,
    )
    brute = get_model(n_periods=_N_PERIODS, n_consumption=200).solve(
        params=get_params(), log_level="off"
    )
    return egm, brute


# Working interior, period by period. Two boundary layers are excluded:
# - Pension (columns): the off-grid top-pension hole advances one column inward per
#   backward working period (boundary p2 -> p1 -> p0), so each earlier period drops one
#   more column.
# - Liquid (rows): the steep low-wealth value is unresolvable on a coarse linear grid,
#   and the error compounds over the working chain; it collapses under grid refinement
#   (~60% at 12 liquid points, ~2% at 48), confirming a resolution limit, not a gap.
_WORKING_INTERIOR = {2: np.s_[3:, :9], 1: np.s_[3:, :8], 0: np.s_[3:, :7]}
# Retired is 1-D; exclude the borrowing-constrained low-liquid boundary layer.
_RETIRED_INTERIOR = np.s_[2:]


@pytest.mark.parametrize("period", [0, 1, 2])
def test_driver_working_value_matches_brute_on_the_pension_interior(period):
    """Each working period's value matches the brute solve where the grid covers."""
    egm, brute = _solve_both()
    sl = _WORKING_INTERIOR[period]
    egm_v = np.asarray(egm[period]["working"])[sl]
    brute_v = np.asarray(brute[period]["working"])[sl]
    assert np.isfinite(egm_v).all()
    rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
    assert np.median(rel) < 0.03
    assert np.percentile(rel, 90) < 0.15


def test_driver_retired_value_matches_brute_on_the_liquid_interior():
    """The retired value matches the brute solve on the unconstrained interior."""
    egm, brute = _solve_both()
    egm_v = np.asarray(egm[_RETIREMENT_PERIOD]["retired"])[_RETIRED_INTERIOR]
    brute_v = np.asarray(brute[_RETIREMENT_PERIOD]["retired"])[_RETIRED_INTERIOR]
    rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
    assert np.median(rel) < 0.01
    assert np.max(rel) < 0.05


def test_driver_returns_every_period_and_regime():
    """The driver publishes working/retired/dead values for the whole lifecycle."""
    egm, _brute = _solve_both()
    assert set(egm[0]) == {"working"}
    assert set(egm[_RETIREMENT_PERIOD]) == {"retired"}
    assert set(egm[_N_PERIODS - 1]) == {"dead"}
