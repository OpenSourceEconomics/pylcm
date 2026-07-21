"""NBEGM on a budget with no declared breakpoints.

A fully smooth consumption-saving regime — affine budget, no declared kinks,
jumps, or case pieces — is the degenerate NB-EGM problem: one interval covering
the whole liquid axis, no topology rows. NBEGM solves it as plain EGM, so a
model author can adopt the solver before declaring any breakpoints and add them
incrementally.
"""

import numpy as np

from lcm import LinSpacedGrid, Model
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    resolve_solver,
    utility,
)

_PARAMS = {
    "alive": {
        "utility": {"crra": 2.0},
        "H": {"discount_factor": 0.95},
        "resources": {"base_income": 2.0},
        "alive": {
            "next_liquid": {"return_liquid": 0.03, "income": 1.0},
            "next_regime": {"final_age_alive": 3.0},
        },
        "dead": {
            "next_liquid": {"return_liquid": 0.03, "income": 1.0},
            "next_regime": {"final_age_alive": 3.0},
        },
    },
    "dead": {"utility": {"crra": 2.0}},
}


def _resources(liquid: ContinuousState, base_income: float) -> FloatND:
    """Cash-on-hand: liquid wealth plus base income — affine, no declarations."""
    return liquid + base_income


def _build_model(*, variant: str) -> Model:
    return make_alive_dead_model(
        n_periods=4,
        n_liquid=40,
        liquid_max=30.0,
        n_consumption=120,
        alive_functions={"utility": utility, "resources": _resources},
        liquid_law=next_liquid,
        alive_solver=resolve_solver(
            variant,
            savings_grid=LinSpacedGrid(start=0.0, stop=28.0, n_points=100),
        ),
        constraints={"feasible": feasible},
    )


def test_nbegm_solves_a_budget_without_declared_breakpoints() -> None:
    """A declaration-free affine budget solves and weakly dominates dense brute.

    With no breakpoints the NB-EGM partition is a single interval and the solve
    is plain EGM: off-grid consumption, so at every state the value is at least
    the dense grid search's, up to interpolation noise on brute's grid.
    """
    nbegm = _build_model(variant="nbegm").solve(params=_PARAMS, log_level="off")
    brute = _build_model(variant="brute").solve(params=_PARAMS, log_level="off")
    for period in (0, 1, 2):
        nbegm_V = np.asarray(nbegm[period]["alive"])
        brute_V = np.asarray(brute[period]["alive"])
        assert not np.isnan(nbegm_V).any(), f"period {period}"
        # Off-grid consumption keeps NBEGM at or above brute up to the two
        # families' continuation-interpolation difference.
        assert np.all(nbegm_V >= brute_V - 0.01), f"period {period}"
        np.testing.assert_allclose(nbegm_V, brute_V, atol=0.05)
