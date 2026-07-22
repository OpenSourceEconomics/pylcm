"""The DS pension benchmark solves as a faithful three-regime lifecycle oracle.

The model is the dense-grid brute-force reference the 2-D EGM kernel is validated
against, so its own solve must be correct: the lifecycle regimes appear in the right
periods, the terminal bequest is the closed-form CRRA value, the working value rises in
both assets, and the comparative statics (a more generous employer match, higher
retirement income) move value in the right direction.
"""

import numpy as np
import pytest

from tests.conftest import X64_ENABLED
from tests.test_models.deterministic.ds_pension import get_model, get_params

# The closed-form comparison is float-eps-limited at the active precision.
_RTOL = 1e-12 if X64_ENABLED else 1e-5


def _solve(**param_overrides):
    model = get_model()
    params = get_params(**param_overrides)
    return model.solve(params=params, log_level="off")


def test_lifecycle_regimes_appear_in_the_right_periods():
    """Working is 2-D for the working periods, retired is 1-D, then the agent dies."""
    solution = _solve()
    assert np.asarray(solution[0]["working"]).shape == (12, 10)
    assert np.asarray(solution[2]["working"]).shape == (12, 10)
    assert np.asarray(solution[3]["retired"]).shape == (12,)
    assert "working" not in solution[3]
    assert "retired" not in solution[4]


def test_terminal_bequest_is_the_closed_form_crra_value():
    """The dead regime carries the closed-form CRRA value of liquid, no optimization."""
    solution = _solve(crra=2.0)
    liquid_grid = np.linspace(0.1, 20.0, 12)
    # CRRA with rho = 2: u(liquid) = liquid**(-1) / (-1) = -1 / liquid.
    expected = -1.0 / liquid_grid
    np.testing.assert_allclose(np.asarray(solution[4]["dead"]), expected, rtol=_RTOL)


def test_working_value_increases_in_both_assets():
    """More liquid wealth and more pension wealth are both weakly valuable."""
    working = np.asarray(_solve()[0]["working"])
    assert np.all(np.diff(working, axis=0) >= -1e-9)
    assert np.all(np.diff(working, axis=1) >= -1e-9)


def test_more_generous_employer_match_does_not_lower_working_value():
    """A larger match `chi` is a free subsidy on deposits, so value weakly rises."""
    low = np.asarray(_solve(match_rate=0.10)[0]["working"])
    high = np.asarray(_solve(match_rate=0.30)[0]["working"])
    assert np.all(high >= low - 1e-9)


@pytest.mark.parametrize(("low_income", "high_income"), [(0.5, 1.0)])
def test_higher_retirement_income_raises_retired_value(low_income, high_income):
    """A richer retirement income strictly raises the retired value function."""
    low = np.asarray(_solve(retirement_income=low_income)[3]["retired"])
    high = np.asarray(_solve(retirement_income=high_income)[3]["retired"])
    assert np.all(high > low)
