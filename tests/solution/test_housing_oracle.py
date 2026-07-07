"""The DS housing benchmark solves as a faithful adjust/keep brute oracle.

The model is the dense-grid reference the 2-D EGM / NEGM kernels are validated against,
so its own solve must be correct: the regimes appear in the right periods, the terminal
bequest is the closed-form CRRA value, the working value rises in both assets, and the
comparative statics (a costlier house trade, a richer income) move value the right way.
"""

import numpy as np

from tests.conftest import X64_ENABLED
from tests.test_models.deterministic.housing import get_model, get_params

# The closed-form comparison is float-eps-limited at the active precision.
_RTOL = 1e-12 if X64_ENABLED else 1e-5


def _solve(**param_overrides):
    model = get_model()
    params = get_params(**param_overrides)
    return model.solve(params=params, log_level="off")


def test_lifecycle_regimes_appear_in_the_right_periods():
    """The working regime is 2-D (liquid, housing) until the terminal dead period."""
    solution = _solve()
    assert np.asarray(solution[0]["working"]).shape == (12, 8)
    assert np.asarray(solution[2]["working"]).shape == (12, 8)
    assert "working" not in solution[3]
    assert np.asarray(solution[3]["dead"]).shape == (12, 8)


def test_terminal_bequest_is_the_closed_form_crra_value():
    """The dead regime carries the closed-form CRRA value of liquid plus housing."""
    crra = 1.458
    solution = _solve(crra=crra)
    liquid = np.linspace(0.01, 50.0, 12)[:, None]
    housing = np.linspace(0.01, 50.0, 8)[None, :]
    expected = (liquid + housing) ** (1.0 - crra) / (1.0 - crra)
    np.testing.assert_allclose(np.asarray(solution[3]["dead"]), expected, rtol=_RTOL)


def test_working_value_increases_in_both_assets():
    """More liquid wealth and a bigger house are both weakly valuable."""
    working = np.asarray(_solve()[0]["working"])
    assert np.all(np.diff(working, axis=0) >= -1e-9)
    assert np.all(np.diff(working, axis=1) >= -1e-9)


def test_higher_transaction_cost_does_not_raise_working_value():
    """A costlier house trade can only shrink the agent's opportunity set."""
    low = np.asarray(_solve(transaction_cost=0.20)[0]["working"])
    high = np.asarray(_solve(transaction_cost=0.60)[0]["working"])
    assert np.all(high <= low + 1e-9)


def test_higher_income_raises_working_value():
    """A richer income strictly raises the working value function."""
    low = np.asarray(_solve(income=1.0)[0]["working"])
    high = np.asarray(_solve(income=2.0)[0]["working"])
    assert np.all(high > low)
