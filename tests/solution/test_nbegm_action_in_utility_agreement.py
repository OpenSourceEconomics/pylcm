"""NBEGM binds the discrete action into period utility per branch.

A binary discrete choice enters period utility directly (a leisure/effort-like term
reading `buy_private`), so each branch has a different utility level and marginal.
The discrete envelope must solve each branch against its own utility, not one shared
per-cell utility. NBEGM's value function must match a dense brute solve across the
liquid interior, over the income ride nodes.
"""

from collections.abc import Mapping

import numpy as np
import pytest

from _lcm.solution.preconditions import check_solver_params
from lcm.exceptions import RegimeInitializationError
from tests.test_models import nbegm_ride_discrete_toy as toy

_ALIVE = "alive"
_TAX_EXEMPTION = 12.0
_LIQUID = np.linspace(0.1, 30.0, 100)
_AWAY_FROM_KINK = (
    (_LIQUID > 1.5) & (_LIQUID < 27.0) & (np.abs(_LIQUID - _TAX_EXEMPTION) > 0.4)
)


def _solve(*, variant: str, n_consumption: int) -> Mapping[int, Mapping]:
    model = toy.build_model(
        variant=variant,
        n_liquid=100,
        liquid_max=30.0,
        n_savings=150,
        savings_max=28.0,
        n_consumption=n_consumption,
        action_in_utility=True,
    )
    return model.solve(params=toy.build_params(), log_level="debug").values


def test_discrete_route_checks_affinity_across_the_liquid_domain() -> None:
    """A discrete branch cannot hide upper-domain budget curvature."""
    model = toy.build_model(
        variant="nbegm",
        include_income=False,
        nonlinear_budget_above_ten=True,
        n_liquid=40,
        liquid_max=20.0,
        n_savings=40,
    )
    params = toy.build_params(include_income=False, nonlinear_budget_above_ten=True)

    with pytest.raises(RegimeInitializationError, match=r"affine|second derivative"):
        check_solver_params(
            regimes=model._regimes, flat_params=model._process_params(params)
        )


def test_single_liquid_nbegm_binds_action_before_building_preferences() -> None:
    """A single-liquid branch evaluates utility with its own discrete action."""
    params = toy.build_params(include_income=False, final_age_alive=2.0)
    nbegm = (
        toy.build_model(
            variant="nbegm",
            n_periods=3,
            n_liquid=45,
            liquid_max=20.0,
            n_savings=80,
            savings_max=18.0,
            n_consumption=50,
            action_in_utility=True,
            include_income=False,
        )
        .solve(params=params, log_level="debug")
        .values
    )
    brute = (
        toy.build_model(
            variant="brute",
            n_periods=3,
            n_liquid=45,
            liquid_max=20.0,
            n_savings=80,
            savings_max=18.0,
            n_consumption=1000,
            action_in_utility=True,
            include_income=False,
        )
        .solve(params=params, log_level="debug")
        .values
    )

    period = max(p for p in brute if _ALIVE in brute[p])
    np.testing.assert_allclose(
        np.asarray(nbegm[period][_ALIVE]),
        np.asarray(brute[period][_ALIVE]),
        rtol=6e-3,
        atol=6e-3,
    )


def test_nbegm_action_in_utility_matches_brute() -> None:
    """`V` agrees with a 1200-point brute across the liquid interior when the
    discrete action enters period utility, over the income nodes — the
    per-branch-utility case."""
    nbegm = _solve(variant="nbegm", n_consumption=100)
    brute = _solve(variant="brute", n_consumption=1200)
    period = max(p for p in brute if _ALIVE in brute[p])
    bq_v = np.asarray(nbegm[period][_ALIVE])
    brute_v = np.asarray(brute[period][_ALIVE])
    assert bq_v.shape == brute_v.shape
    (liquid_axis,) = (
        axis for axis, size in enumerate(bq_v.shape) if size == _LIQUID.shape[0]
    )
    bq_interior = np.take(bq_v, np.flatnonzero(_AWAY_FROM_KINK), axis=liquid_axis)
    brute_interior = np.take(brute_v, np.flatnonzero(_AWAY_FROM_KINK), axis=liquid_axis)
    np.testing.assert_allclose(bq_interior, brute_interior, rtol=5e-3, atol=5e-3)
