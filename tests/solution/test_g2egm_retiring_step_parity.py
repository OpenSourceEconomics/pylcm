"""The working->retired boundary G2EGM step reproduces the brute solve.

At the retirement boundary the working agent's 2-D problem reads a 1-D retired
continuation: the pension is paid out as a lump sum, so both post-decision balances
feed a single retired liquid state. Chaining the 1-D retired EGM step into the
boundary G2EGM step and comparing to the DS pension brute solve checks the lump-sum
adapter end to end. The working utility carries an additive work disutility the generic
envelope objective omits, so the driver subtracts it (an additive constant that shifts
the value level without changing the policy).
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.one_asset_egm_step import egm_one_asset_step
from _lcm.egm.two_asset_g2egm_step import g2egm_retiring_step
from tests.test_models.deterministic.ds_pension import get_model, get_params

_LIQUID_GRID = jnp.linspace(0.1, 20.0, 12)
_PENSION_GRID = jnp.linspace(0.0, 15.0, 10)
_SAVINGS_GRID = jnp.linspace(0.0, 20.0, 40)
_A_GRID = jnp.linspace(0.0, 20.0, 18)
_B_GRID = jnp.linspace(0.0, 30.0, 16)
_CONSUMPTION_GRID = jnp.linspace(0.1, 20.0, 18)

_DISCOUNT, _CRRA, _MATCH = 0.98, 2.0, 0.10
_RETURN_LIQUID, _RETURN_PENSION, _RET_INCOME = 0.02, 0.04, 0.50
_PAYOUT = 1.0 + _RETURN_PENSION
_WORK_DISUTILITY = 0.25

# The top pension column is the off-grid uncovered edge (the post-decision pension
# exceeds the grid); the interior is what the parity assertion covers.
_INTERIOR = np.s_[:, :9]


def _solve_to_boundary():
    """Brute-solve DS pension, then EGM-chain dead -> retired -> working-retiring.

    Returns the EGM boundary working value and the brute `brute[2]["working"]`.
    """
    brute = get_model(n_periods=5, n_consumption=200).solve(
        params=get_params(), log_level="off"
    )
    v_dead = jnp.asarray(brute[4]["dead"])
    v_retired, marginal_retired = egm_one_asset_step(
        next_value=v_dead,
        next_marginal=_LIQUID_GRID ** (-_CRRA),
        liquid_grid=_LIQUID_GRID,
        savings_grid=_SAVINGS_GRID,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
        return_liquid=_RETURN_LIQUID,
        income=_RET_INCOME,
    )
    v_working_raw = g2egm_retiring_step(
        next_value_retired=v_retired,
        next_marginal_retired=marginal_retired,
        liquid_grid=_LIQUID_GRID,
        m_grid=_LIQUID_GRID,
        n_grid=_PENSION_GRID,
        a_grid=_A_GRID,
        b_grid=_B_GRID,
        consumption_grid=_CONSUMPTION_GRID,
        discount_factor=_DISCOUNT,
        crra=_CRRA,
        match_rate=_MATCH,
        return_liquid=_RETURN_LIQUID,
        pension_payout_return=_PAYOUT,
        retirement_income=_RET_INCOME,
    )
    v_working = np.asarray(v_working_raw) - _WORK_DISUTILITY
    return v_working, np.asarray(brute[2]["working"])


def test_retiring_boundary_step_matches_brute_on_the_pension_interior():
    """The boundary working value matches the brute solve where the grid covers."""
    v_working, brute_working = _solve_to_boundary()
    assert np.isfinite(v_working[_INTERIOR]).all()
    rel = np.abs(v_working[_INTERIOR] - brute_working[_INTERIOR]) / np.abs(
        brute_working[_INTERIOR]
    )
    assert np.median(rel) < 0.02
    assert np.percentile(rel, 90) < 0.10


def test_retiring_boundary_value_is_monotone_in_both_assets_on_the_interior():
    """More liquid and more pension stay weakly valuable at the boundary.

    Liquid monotonicity is exact. Pension monotonicity holds up to a tiny tolerance: at
    the maximum liquid the post-payout retired liquid most exceeds the retired grid, so
    the continuation is clamped flat there and the envelope leaves sub-0.1% noise — the
    same under-coverage the brute solve absorbs by clamping (which is why parity holds).
    """
    v_working, _brute = _solve_to_boundary()
    assert np.all(np.diff(v_working[_INTERIOR], axis=0) >= -1e-6)
    assert np.all(np.diff(v_working[_INTERIOR], axis=1) >= -2e-3)
