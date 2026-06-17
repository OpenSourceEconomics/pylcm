"""The solver-selection seam.

A regime carries a `solver` configuration selecting its backward-induction
algorithm. `BruteForce()` is the default and routes the existing grid search
through the builder registry; `DCEGM(...)` is a published configuration whose
engine is not yet available, so requesting it is rejected at model build.
"""

import pytest
from numpy.testing import assert_array_equal

from lcm import DCEGM, AgeGrid, BruteForce, Model
from lcm.exceptions import RegimeInitializationError
from lcm_examples.iskhakov_et_al_2017 import (
    WEALTH_GRID,
    RegimeId,
    dead,
    get_params,
    retirement,
    working_life,
)

_N_PERIODS = 4
_PARAMS = get_params(
    n_periods=_N_PERIODS,
    discount_factor=0.98,
    disutility_of_work=1.0,
    interest_rate=0.0,
    wage=20.0,
)


def _build_model(*, working_solver: object | None = None) -> Model:
    """Build the retirement model, optionally overriding `working_life`'s solver."""
    ages = AgeGrid(start=40, stop=40 + (_N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    overrides = {} if working_solver is None else {"solver": working_solver}
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age, la=last_age: age < la, **overrides
            ),
            "retirement": retirement.replace(active=lambda age, la=last_age: age < la),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )


def _valid_dcegm() -> DCEGM:
    return DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=WEALTH_GRID,
    )


def test_regime_solver_defaults_to_brute_force():
    """A regime with no solver specified uses `BruteForce()`."""
    assert working_life.solver == BruteForce()


def test_explicit_brute_force_matches_default_solution():
    """Setting `solver=BruteForce()` explicitly yields the same value function
    as leaving the solver at its default — the dispatch through the registry
    changes no numerics."""
    default = _build_model().solve(log_level="debug", params=_PARAMS)
    explicit = _build_model(working_solver=BruteForce()).solve(
        log_level="debug", params=_PARAMS
    )
    for period, regime_to_V_arr in default.items():
        for regime_name, V_arr in regime_to_V_arr.items():
            assert_array_equal(V_arr, explicit[period][regime_name])


def test_dcegm_config_constructs():
    """A `DCEGM` config with valid fields constructs and exposes its defaults."""
    cfg = _valid_dcegm()
    assert cfg.continuous_state == "wealth"
    assert cfg.upper_envelope == "fues"


def test_dcegm_config_rejects_refined_grid_factor_at_or_below_one():
    """`refined_grid_factor <= 1.0` leaves no headroom and is rejected."""
    with pytest.raises(RegimeInitializationError, match="refined_grid_factor"):
        DCEGM(
            continuous_state="wealth",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            savings_grid=WEALTH_GRID,
            refined_grid_factor=1.0,
        )


def test_model_with_dcegm_solver_raises_not_implemented():
    """Requesting the DC-EGM solver is rejected at model build: the public
    configuration exists, but the solver engine is not yet available."""
    with pytest.raises(NotImplementedError, match="DC-EGM"):
        _build_model(working_solver=_valid_dcegm())
