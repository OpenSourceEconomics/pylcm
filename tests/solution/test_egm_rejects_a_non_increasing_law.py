"""A declared law the endogenous grid cannot be read back through is refused.

The method solves on a savings grid and maps the result onto the regular wealth
grid by interpolation, whose abscissae must ascend strictly. A law whose landing
points fall as savings rise leaves them unsorted, and one that is flat over a
band leaves them tied. Neither is detected by the interpolation itself — it
returns quietly wrong numbers — so the solver checks the law it was handed
before inverting anything through it.
"""

import pytest

from lcm.exceptions import RegimeInitializationError
from lcm.solvers import EGM
from lcm.typing import ContinuousState, FloatND
from tests.solution.test_egm_honours_the_declared_law import _model
from tests.solution.test_egm_solver import (
    _CRRA,
    _DISCOUNT_FACTOR,
    _RETURN,
    _SAVINGS_GRID,
)


def next_wealth_falling_in_savings(
    savings: FloatND,
    return_liquid: float,
    retirement_income: float,
) -> ContinuousState:
    """A law that punishes a saver: every extra unit saved lowers next wealth."""
    return retirement_income - (1.0 + return_liquid) * savings


def test_a_law_falling_in_savings_is_refused_when_the_model_is_solved():
    """Solving reports the ordering requirement, not a wrong number."""
    model = _model(
        solver=EGM(savings_grid=_SAVINGS_GRID),
        law=next_wealth_falling_in_savings,
    )

    law = {"return_liquid": _RETURN, "retirement_income": 5.0}
    params = {
        "saving": {
            "utility": {"crra": _CRRA},
            "koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR},
            "saving": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
            "done": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
        },
        "done": {"utility": {"crra": _CRRA}},
    }

    with pytest.raises(RegimeInitializationError, match="falls as savings rise"):
        model.solve(params=params, log_level="debug")
