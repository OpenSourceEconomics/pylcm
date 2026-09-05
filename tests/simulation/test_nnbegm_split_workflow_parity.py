"""A retained SolutionResult must replay exactly like automatic simulation.

The addressed replay store is model-authoritative: dropping it or substituting a policy
from another outer-search route fails before forward execution.
"""

from dataclasses import replace

import jax.numpy as jnp
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from _lcm.egm import outer_affine_structure, outer_inversion
from lcm import LinSpacedGrid
from lcm.exceptions import (
    InvalidSimulationInputError,
    UnrepresentableOuterCandidateError,
)
from lcm.model import Model
from lcm.solver_api import ArtifactStore, SolutionResult
from lcm.solvers import AdaptiveOuterMesh, FiniteOuterGrid
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

# A signed outer domain: the inverse recovers the action as `target - at_zero`
# and adds it back, which crosses zero and so is not exact there. Some declared
# nodes are then reached only to a stock outside the domain, and the solve
# refuses to publish a bank missing them.
_REFUSING_GRID = LinSpacedGrid(start=-1.0, stop=10.0, n_points=13)

_REFUSING_INITIAL = {
    "wealth": jnp.array([4.3, 11.7, 19.9, 8.1]),
    "illiquid": jnp.array([-0.4, 2.2, 5.9, 9.3]),
    "age": jnp.full(4, 20.0),
    "regime_id": jnp.zeros(4, dtype=jnp.int32),
}


def _build(route: str) -> Model:
    return toy.build_model(
        variant="n_nbegm",
        n_periods=_N_PERIODS,
        outer_search=_ROUTES[route],
    )


def _simulate(*, model: Model, solution: SolutionResult | None) -> pd.DataFrame:
    return model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="debug",
        seed=_SEED,
    ).to_dataframe()


@pytest.fixture(scope="module", params=["finite", "adaptive"])
def route(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def solved(route: str) -> tuple[Model, SolutionResult]:
    """Return the model and its complete labelled solution."""
    model = _build(route)
    return model, model.solve(params=_PARAMS, log_level="debug")


def test_separate_solve_and_simulate_matches_the_automatic_route(
    solved: tuple[Model, SolutionResult],
) -> None:
    """The documented split workflow reproduces automatic solve-and-simulate."""
    model, solution = solved
    assert_frame_equal(
        _simulate(model=model, solution=solution),
        _simulate(model=model, solution=None),
    )


def test_result_without_replay_policies_is_refused(
    solved: tuple[Model, SolutionResult],
) -> None:
    """Dropping replay artifacts is refused rather than silently re-optimized."""
    model, solution = solved
    without_replay = replace(solution, replay_artifacts=ArtifactStore())
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"unrecorded|missing accounting",
    ):
        _simulate(model=model, solution=without_replay)


def test_replay_policies_published_by_the_other_outer_search_are_refused() -> None:
    """A replay policy of the wrong route cannot stand in for this one's."""
    finite_solution = _build("finite").solve(params=_PARAMS, log_level="debug")
    adaptive_model = _build("adaptive")
    adaptive_solution = adaptive_model.solve(params=_PARAMS, log_level="debug")
    wrong_route = replace(
        adaptive_solution, replay_artifacts=finite_solution.replay_artifacts
    )

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        _simulate(model=adaptive_model, solution=wrong_route)


def test_a_solve_time_refusal_reaches_both_routes_identically() -> None:
    """A solve that refuses to publish refuses the same way through either route.

    Both workflows funnel through the same solve: the split one calls it
    directly, and the automatic one calls it from `simulate()` when no solution is
    supplied. A refusal must therefore surface identically, rather than one
    route reporting it and the other proceeding on a bank the solve declined to
    publish.
    """
    model = _build_refusing_model()

    with pytest.raises(UnrepresentableOuterCandidateError) as split:
        model.solve(params=_PARAMS, log_level="off")

    with pytest.raises(UnrepresentableOuterCandidateError) as automatic:
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_REFUSING_INITIAL),
            log_level="off",
            seed=_SEED,
        )

    assert str(split.value) == str(automatic.value)


def _build_refusing_model() -> Model:
    """Return a model whose solve cannot reconstruct every declared node."""
    return toy.build_model(
        variant="n_nbegm",
        n_periods=_N_PERIODS,
        illiquid_grid=_REFUSING_GRID,
        outer_search=FiniteOuterGrid(grid=_REFUSING_GRID),
    )


def test_split_replay_does_not_certify_the_declared_outer_map_itself() -> None:
    """Split replay consumes the published inversion verdict instead of re-deriving it.

    The solve settles the declared map's coefficient and the stock domain once,
    before publishing, and the replay policy carries that answer. A replay that
    certified the map again could admit a stock the solve refused, so making
    every certifier fatal for the duration of the replay establishes that none
    is reached — and the replay it produces is unchanged.
    """
    model = _build("finite")
    solution = model.solve(params=_PARAMS, log_level="debug")
    expected = _simulate(model=model, solution=solution)

    def refuse(**_kwargs):
        raise AssertionError("split replay re-certified the declared outer map")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(outer_affine_structure, "certify_outer_coefficient", refuse)
        patch.setattr(outer_inversion, "certify_declared_outer_inverse", refuse)
        got = _simulate(model=model, solution=solution)

    assert_frame_equal(got, expected)
