"""NBEGM with a completely flat (zero-marginal) continuation.

The last alive period of a model whose terminal target has no bequest motive
reads a continuation whose marginal value of liquid wealth is identically
zero. Every interior Euler candidate is then degenerate (dropped as NaN by
design), and the whole solution must come from the savings-node corner
chains — which requires the segment-id bookkeeping to survive an all-NaN
interior segment (`_next_segment_id`). Before that guard, `nanmax` of the
all-NaN interior ids poisoned the corner chains and the published value
was all-NaN.

The consume-everything solution is analytic — `V(liquid) = u(liquid +
base_income)` with zero savings — so the test pins values, not just
finiteness.
"""

import jax.numpy as jnp
import numpy as np

from lcm import LinSpacedGrid
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    resolve_solver,
    utility,
)

_CRRA = 2.0
_BASE_INCOME = 2.0
_N_LIQUID = 40

_PARAMS = {
    "alive": {
        "utility": {"crra": _CRRA},
        "H": {"discount_factor": 0.95},
        "resources": {"base_income": _BASE_INCOME},
        "alive": {
            "next_liquid": {"return_liquid": 0.03, "income": 1.0},
            "next_regime": {"final_age_alive": 1.0},
        },
        "dead": {
            "next_liquid": {"return_liquid": 0.03, "income": 1.0},
            "next_regime": {"final_age_alive": 1.0},
        },
    },
    "dead": {},
}


def _resources(liquid: ContinuousState, base_income: float) -> FloatND:
    """Cash-on-hand: liquid wealth plus base income — affine, no declarations."""
    return liquid + base_income


def _flat_utility(liquid: ContinuousState) -> FloatND:
    """No bequest motive: constant zero value, zero marginal — the flat case."""
    return jnp.zeros_like(liquid)


def test_nbegm_survives_a_zero_marginal_continuation() -> None:
    """A flat continuation solves to the analytic consume-everything value."""
    model = make_alive_dead_model(
        n_periods=2,
        n_liquid=_N_LIQUID,
        liquid_max=30.0,
        n_consumption=120,
        alive_functions={"utility": utility, "resources": _resources},
        liquid_law=next_liquid,
        alive_solver=resolve_solver(
            "nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=28.0, n_points=100),
        ),
        constraints={"feasible": feasible},
        dead_functions={"utility": _flat_utility},
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    v = np.asarray(solution[0]["alive"])
    assert np.all(np.isfinite(v)), "flat continuation produced non-finite values"
    liquid = np.linspace(0.1, 30.0, _N_LIQUID)
    cash = liquid + _BASE_INCOME
    expected = cash ** (1.0 - _CRRA) / (1.0 - _CRRA)
    np.testing.assert_allclose(v, expected, rtol=1e-6)
