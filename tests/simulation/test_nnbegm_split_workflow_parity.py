"""Solving and simulating separately must match solving inside `simulate()`.

`solve(return_simulation_policy=True)` publishes the replay artifacts that
simulation needs where the value functions alone do not determine the decision.
Feeding that exact `(values, policies)` pair back into `simulate()` is the
documented split workflow, and it must reproduce what the automatic
solve-and-simulate route produces — for the finite outer search and for the
adaptive continuous-outer mesh alike.

Which artifact counts as matching follows the configured outer search, so the
guards below also pin that the check still refuses a dropped mapping and a
policy published by the other route.
"""

import jax.numpy as jnp
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from _lcm.typing import (
    PeriodToRegimeToSimulationPolicy,
    PeriodToRegimeToVArr,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.model import Model
from lcm.solvers import AdaptiveOuterMesh
from tests.test_models import n_nbegm_toy as toy

_PARAMS = {"discount_factor": 0.95}
_SEED = 42
_N_PERIODS = 3

_MESH = AdaptiveOuterMesh(
    initial_grid=toy.OUTER_GRID,
    max_nodes=513,
    max_refinement_rounds=10,
    value_atol=1e-4,
    value_rtol=1e-4,
    golden_iterations=40,
)

_ROUTES = {"finite": None, "adaptive": _MESH}

_INITIAL = {
    "wealth": jnp.array([4.3, 11.7, 19.9, 8.1]),
    "illiquid": jnp.array([1.37, 6.6, 13.2, 17.5]),
    "age": jnp.full(4, 20.0),
    "regime_id": jnp.zeros(4, dtype=jnp.int32),
}


def _build(route: str) -> Model:
    return toy.build_model(
        variant="n_nbegm",
        n_periods=_N_PERIODS,
        outer_search=_ROUTES[route],
    )


def _simulate(
    model: Model,
    *,
    period_to_regime_to_V_arr: PeriodToRegimeToVArr | None,
    policies: PeriodToRegimeToSimulationPolicy | None = None,
) -> pd.DataFrame:
    return model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        policies=policies,
        log_level="debug",
        seed=_SEED,
    ).to_dataframe()


@pytest.fixture(scope="module", params=["finite", "adaptive"])
def route(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def solved(
    route: str,
) -> tuple[Model, PeriodToRegimeToVArr, PeriodToRegimeToSimulationPolicy]:
    """The model plus the `(values, policies)` pair its own `solve()` returned."""
    model = _build(route)
    values, policies = model.solve(
        params=_PARAMS,
        log_level="debug",
        return_simulation_policy=True,
    )
    return model, values, policies


def test_separate_solve_and_simulate_matches_the_automatic_route(
    solved: tuple[Model, PeriodToRegimeToVArr, PeriodToRegimeToSimulationPolicy],
) -> None:
    """The documented split workflow reproduces automatic solve-and-simulate."""
    model, values, policies = solved
    assert_frame_equal(
        _simulate(model, period_to_regime_to_V_arr=values, policies=policies),
        _simulate(model, period_to_regime_to_V_arr=None),
    )


def test_supplied_values_without_any_replay_policies_are_refused(
    solved: tuple[Model, PeriodToRegimeToVArr, PeriodToRegimeToSimulationPolicy],
) -> None:
    """Dropping the replay mapping is refused rather than silently re-optimized."""
    model, values, _ = solved
    with pytest.raises(InvalidSimulationInputError, match="replay policy"):
        _simulate(model, period_to_regime_to_V_arr=values, policies=None)


def test_replay_policies_published_by_the_other_outer_search_are_refused() -> None:
    """A replay policy of the wrong route cannot stand in for this one's."""
    finite_model = _build("finite")
    _, finite_policies = finite_model.solve(
        params=_PARAMS,
        log_level="debug",
        return_simulation_policy=True,
    )
    adaptive_model = _build("adaptive")
    adaptive_values = adaptive_model.solve(params=_PARAMS, log_level="debug")

    with pytest.raises(InvalidSimulationInputError, match="replay policy"):
        _simulate(
            adaptive_model,
            period_to_regime_to_V_arr=adaptive_values,
            policies=finite_policies,
        )
