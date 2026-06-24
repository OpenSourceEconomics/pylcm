"""`TwoDimEGM(upper_envelope="rfc")` drives the DS pension RFC lifecycle solve.

The combined-cloud rooftop-cut is selected as the working regime's two-asset upper
envelope by naming `TwoDimEGM(upper_envelope="rfc")` on it. The engine's backward
induction then chains the RFC step across the lifecycle: the working regime uses the
RFC two-asset step (the retirement-boundary period falls back to the G2EGM retiring
step), the retired regime the 1-D `OneAssetEGM`, and the dead regime stays terminal.
The solve publishes every period and regime, and the working value tracks the dense
grid-search brute on the pension interior away from the low-liquid corner (a known RFC
corner-accuracy gap).
"""

import numpy as np
import pytest

from lcm import LinSpacedGrid
from lcm.solvers import OneAssetEGM, TwoDimEGM
from tests.test_models.deterministic.ds_pension import get_model, get_params

_N_PERIODS = 5
_RETIREMENT_PERIOD = 3

_A_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=18)
_B_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=16)
_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=18)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)

# Pension interior, away from the top-pension boundary and the low-liquid corner. The
# corner-accuracy gap propagates one liquid row inward per backward period (as the
# top-pension hole does), so the interior floor rises one row earlier each period back.
_WORKING_INTERIOR = {2: np.s_[3:, :9], 1: np.s_[4:, :8], 0: np.s_[5:, :7]}


def _rfc_model():
    return get_model(
        n_periods=_N_PERIODS,
        solvers={
            "working": TwoDimEGM(
                a_grid=_A_GRID,
                b_grid=_B_GRID,
                consumption_grid=_CONSUMPTION_GRID,
                upper_envelope="rfc",
            ),
            "retired": OneAssetEGM(savings_grid=_SAVINGS_GRID),
        },
    )


def test_rfc_solver_publishes_every_period_and_regime():
    """The RFC-driven lifecycle solve publishes working/retired/dead end-to-end."""
    egm = _rfc_model().solve(params=get_params(), log_level="off")
    assert "working" in egm[0]
    assert "retired" in egm[_RETIREMENT_PERIOD]
    assert "dead" in egm[_N_PERIODS - 1]


@pytest.mark.parametrize("period", [0, 1, 2])
def test_rfc_solver_working_value_tracks_brute_on_the_pension_interior(period):
    """Each working period's RFC value tracks brute on the covered pension interior."""
    egm = _rfc_model().solve(params=get_params(), log_level="off")
    brute = get_model(n_periods=_N_PERIODS, n_consumption=200).solve(
        params=get_params(), log_level="off"
    )
    sl = _WORKING_INTERIOR[period]
    egm_v = np.asarray(egm[period]["working"])[sl]
    brute_v = np.asarray(brute[period]["working"])[sl]
    assert np.isfinite(egm_v).all()
    rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
    assert np.median(rel) < 0.08
    assert np.percentile(rel, 90) < 0.20
