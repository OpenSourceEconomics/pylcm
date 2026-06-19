"""The solver-selection seam.

A regime carries a `solver` configuration selecting its backward-induction
algorithm. `GridSearch()` is the default and runs the existing grid search; the
engine dispatches polymorphically on the solver instance
(`solver.build_period_kernels`), not on its type. `DCEGM(...)` selects the
discrete-continuous endogenous grid method; its configuration is validated at
model build.
"""

from dataclasses import replace

import pytest
from numpy.testing import assert_array_equal

from lcm import DCEGM, AgeGrid, GridSearch, LinSpacedGrid, Model, NormalIIDProcess
from lcm.exceptions import RegimeInitializationError
from lcm_examples.iskhakov_et_al_2017 import (
    WEALTH_GRID,
    RegimeId,
    dead,
    get_params,
    retirement,
    working_life,
)
from tests.test_models.dcegm_paper_twin import build_dcegm_model

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


def test_regime_solver_defaults_to_grid_search():
    """A regime with no solver specified uses `GridSearch()`."""
    assert working_life.solver == GridSearch()


def test_explicit_grid_search_matches_default_solution():
    """Setting `solver=GridSearch()` explicitly yields the same value function
    as leaving the solver at its default — the polymorphic dispatch changes no
    numerics."""
    default = _build_model().solve(log_level="debug", params=_PARAMS)
    explicit = _build_model(working_solver=GridSearch()).solve(
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


def test_model_with_dcegm_solver_builds():
    """Selecting the DC-EGM solver builds the model: the engine wires the
    solver's kernels in rather than rejecting the configuration.

    Uses a regime that satisfies the DC-EGM contract (resources,
    post-decision, and inverse-marginal-utility functions); a stock
    grid-search regime would be rejected for missing them.
    """
    model = build_dcegm_model()
    assert isinstance(model.user_regimes["working_life"].solver, DCEGM)


_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=50)

_BASE_DCEGM = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=_SAVINGS_GRID,
)


def _dcegm(**overrides) -> DCEGM:
    return replace(_BASE_DCEGM, **overrides)


def test_dcegm_defaults_construct_without_error():
    """The documented defaults pass validation."""
    solver = _dcegm()
    assert solver.savings_grid is _SAVINGS_GRID


def test_dcegm_stochastic_savings_grid_is_rejected():
    """A stochastic process cannot serve as the deterministic savings grid."""
    process = NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
    with pytest.raises(RegimeInitializationError, match="stochastic process"):
        _dcegm(savings_grid=process)


@pytest.mark.parametrize("refined_grid_factor", [1.0, 0.9, float("nan"), float("inf")])
def test_dcegm_non_finite_or_too_small_refined_grid_factor_is_rejected(
    refined_grid_factor,
):
    """The headroom factor must be finite and exceed 1.0 (NaN/inf rejected too)."""
    with pytest.raises(RegimeInitializationError, match="refined_grid_factor"):
        _dcegm(refined_grid_factor=refined_grid_factor)


@pytest.mark.parametrize("fues_jump_thresh", [0.0, -1.0, float("nan"), float("inf")])
def test_dcegm_non_finite_or_non_positive_fues_jump_thresh_is_rejected(
    fues_jump_thresh,
):
    """The segment-switch threshold must be finite and positive (NaN/inf rejected)."""
    with pytest.raises(RegimeInitializationError, match="fues_jump_thresh"):
        _dcegm(fues_jump_thresh=fues_jump_thresh)


@pytest.mark.parametrize("n_constrained_points", [1, 0])
def test_dcegm_too_few_constrained_points_is_rejected(n_constrained_points):
    """The constrained segment needs at least two closed-form points."""
    with pytest.raises(RegimeInitializationError, match="n_constrained_points"):
        _dcegm(n_constrained_points=n_constrained_points)


def test_dcegm_zero_points_to_scan_is_rejected():
    """The FUES forward scan must inspect at least one point."""
    with pytest.raises(RegimeInitializationError, match="fues_n_points_to_scan"):
        _dcegm(fues_n_points_to_scan=0)
