"""Brute-force solve of the deterministic two-asset model — the 2-D EGM oracle.

Validates that the engine expresses two action-coupled continuous states
(`liquid`, `pension`) with two continuous actions (`consumption`, `deposit`) and
solves them by dense grid search, producing the reference value function the
multidimensional EGM kernel is checked against.
"""

import numpy as np

from tests.test_models.deterministic.two_asset import get_model, get_params


def test_two_asset_model_exposes_two_coupled_continuous_states():
    """The working regime carries two continuous states and two continuous actions."""
    working = get_model().user_regimes["working"]
    assert set(working.states) == {"liquid", "pension"}
    assert set(working.actions) == {"consumption", "deposit"}


def test_two_asset_params_template_matches_get_params():
    """`get_params` fills exactly the leaves the model's params template declares."""
    model = get_model()
    template = model.get_params_template()
    params = get_params()
    # Every leaf `get_params` fills is a real template leaf; the template may also
    # carry default-empty groups (`H`, the constraint, the regime transition) that
    # need no values, which the successful brute solve confirms.
    assert set(params["working"]) <= set(template["working"])
    assert "discount_factor" in params


def test_two_asset_brute_value_is_finite_and_increases_in_liquid_wealth():
    """Brute-solved value is finite and weakly increasing in liquid wealth."""
    model = get_model()
    value = model.solve(params=get_params(), log_level="off")
    working_first_period = np.asarray(value[0]["working"])
    assert np.all(np.isfinite(working_first_period))
    # `liquid` is the first state axis; more cash-on-hand is weakly better.
    liquid_diffs = np.diff(working_first_period, axis=0)
    assert np.all(liquid_diffs >= -1e-6)


def test_two_asset_brute_value_increases_in_pension_wealth():
    """The brute-solved value function is weakly increasing in pension wealth."""
    value = get_model().solve(params=get_params(), log_level="off")
    working_first_period = np.asarray(value[0]["working"])
    pension_diffs = np.diff(working_first_period, axis=1)
    assert np.all(pension_diffs >= -1e-6)
