"""`model.solve(solver=TwoDimEGM(...))` drives the DS pension G2EGM solve.

The two-asset G2EGM kernel is a prime-time pylcm `Solver`: naming it on the working
regime makes `model.solve()` reproduce the standalone driver's lifecycle solve through
the engine's backward induction. The working regime uses `TwoDimEGM`, the retired
regime the 1-D `OneAssetEGM`, and the dead regime stays terminal. The solved value
matches the dense grid-search brute on the same covered interior the standalone driver
matches: the working pension interior (excluding the off-grid top-pension boundary
layer that thickens backward, and the steep low-liquid rows) and the retired
unconstrained liquid interior.
"""

import numpy as np
import pytest

from lcm import LinSpacedGrid
from lcm.solvers import OneAssetEGM, TwoDimEGM
from tests.test_models.deterministic.ds_pension import get_model, get_params

_N_PERIODS = 5
_RETIREMENT_PERIOD = 3

# Same post-decision grids the standalone driver test uses.
_A_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=18)
_B_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=16)
_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=18)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)

# Same interior masks the standalone driver test asserts on.
_WORKING_INTERIOR = {2: np.s_[3:, :9], 1: np.s_[3:, :8], 0: np.s_[3:, :7]}
_RETIRED_INTERIOR = np.s_[2:]


def _solver_model():
    """The DS pension model with the G2EGM/1-D-EGM prime-time solvers."""
    return get_model(
        n_periods=_N_PERIODS,
        solvers={
            "working": TwoDimEGM(
                a_grid=_A_GRID,
                b_grid=_B_GRID,
                consumption_grid=_CONSUMPTION_GRID,
            ),
            "retired": OneAssetEGM(savings_grid=_SAVINGS_GRID),
        },
    )


def _solve_both():
    egm = _solver_model().solve(params=get_params(), log_level="off")
    brute = get_model(n_periods=_N_PERIODS, n_consumption=200).solve(
        params=get_params(), log_level="off"
    )
    return egm, brute


@pytest.mark.parametrize("period", [0, 1, 2])
def test_solver_working_value_matches_brute_on_the_pension_interior(period):
    """Each working period's solved value matches brute where the grid covers."""
    egm, brute = _solve_both()
    sl = _WORKING_INTERIOR[period]
    egm_v = np.asarray(egm[period]["working"])[sl]
    brute_v = np.asarray(brute[period]["working"])[sl]
    assert np.isfinite(egm_v).all()
    rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
    assert np.median(rel) < 0.03
    assert np.percentile(rel, 90) < 0.15


def test_solver_retired_value_matches_brute_on_the_liquid_interior():
    """The retired value matches brute on the unconstrained liquid interior."""
    egm, brute = _solve_both()
    egm_v = np.asarray(egm[_RETIREMENT_PERIOD]["retired"])[_RETIRED_INTERIOR]
    brute_v = np.asarray(brute[_RETIREMENT_PERIOD]["retired"])[_RETIRED_INTERIOR]
    rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
    assert np.median(rel) < 0.01
    assert np.max(rel) < 0.05


def test_solver_publishes_every_period_and_regime():
    """The solver-driven solve publishes working/retired/dead over the lifecycle."""
    egm, _brute = _solve_both()
    assert "working" in egm[0]
    assert "retired" in egm[_RETIREMENT_PERIOD]
    assert "dead" in egm[_N_PERIODS - 1]
