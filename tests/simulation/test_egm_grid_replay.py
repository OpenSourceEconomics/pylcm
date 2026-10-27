"""Explicit Cartesian replay is assessed separately from original EGM candidates."""

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY, CaseUReplaySpec
from lcm import (
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    Model,
    categorical,
)
from lcm.exceptions import InvalidSimulationInputError, UnsupportedOperationError
from lcm.persistence import load_solution
from lcm.result import SimulationResult
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactRef, ReplayMode, SolutionResult
from lcm.solvers import DCEGM
from lcm.typing import ContinuousAction, DiscreteAction, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations
from tests.egm.test_current_iid_parent_demands import _resources
from tests.simulation.test_case_u_reconstruction import _params, _ReplayRegime

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _flow(*, consumption: ContinuousAction, eps: FloatND) -> FloatND:
    return 0.5 * jnp.log(consumption) + eps


def _inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 0.5 / marginal_continuation


def _terminal(*, wealth: FloatND) -> FloatND:
    return 0.5 * jnp.log(wealth)


@categorical(ordered=False)
class _Branch:
    a: ScalarInt
    b: ScalarInt


def _branch_flow(
    *, consumption: ContinuousAction, eps: FloatND, branch: DiscreteAction
) -> FloatND:
    return _flow(consumption=consumption, eps=eps) + jnp.where(
        branch == _Branch.b, 0.5 * math.log(7 / 2), 0.0
    )


def _branch_resources(*, wealth: FloatND, branch: DiscreteAction) -> FloatND:
    return wealth * jnp.where(branch == _Branch.b, 0.5, 1.0)


def _model(
    *,
    replay: Literal["candidate", "grid"],
    matched: bool = False,
    subject_width: int = 1,
    branches: bool = False,
) -> Model:
    declarations = _actual_declarations()
    child = declarations.regimes["child"]
    done = declarations.regimes["done"]
    assert isinstance(child, ConsumptionSavingsRegime)
    assert isinstance(child.solver, DCEGM)
    savings = IrregSpacedGrid(points=np.asarray([0.125, 0.25, 0.5, 0.75, 1.0, 2.0]))
    child = replace(
        child,
        states={"wealth": IrregSpacedGrid(points=np.asarray([0.5, 1.0, 1.5]))},
        actions={
            "consumption": IrregSpacedGrid(
                points=np.asarray([0.25, 0.5, 0.75] if matched else [0.25, 0.75])
            )
        },
        functions={
            **child.functions,
            "utility": _flow,
            "inverse_marginal_utility": _inverse,
            "resources": _resources,
        },
        solver=replace(
            child.solver,
            savings_grid=savings,
            current_iid_projection_capacity=8,
            replay=replay,
            accuracy="balanced",
        ),
    )
    if branches:
        child = replace(
            child,
            actions={**child.actions, "branch": DiscreteGrid(_Branch)},
            functions={
                **child.functions,
                "utility": _branch_flow,
                "resources": _branch_resources,
            },
        )
    return Model._from_call_owned_params(
        params=_params(),
        declarations=replace(
            declarations,
            regimes={
                "child": child,
                "done": replace(
                    done, states={"wealth": savings}, functions={"utility": _terminal}
                ),
            },
            initial_nodes=InitialNodes(by_age={1: "child"}),
            edges={"child": {"done": (1,)}},
            regime_id_class=_ReplayRegime,
            execution_config=ExecutionConfig(
                device_memory_bytes=None, axis_widths={"subject": subject_width}
            ),
            shocks={
                "income": ShockGroup(
                    shocks={"eps": Uniform(start=0.0, stop=1.0)},
                    nodes=UserNodes(
                        nodes=np.asarray([[0.0], [1.0]]), weights=np.asarray([0.5, 0.5])
                    ),
                )
            },
        ),
    )


def _simulate(
    *,
    model: Model,
    solution: SolutionResult,
    physical: PhysicalValues | None = None,
) -> SimulationResult:
    return model.simulate(
        params=_params(),
        solution=solution,
        initial_conditions={
            "wealth": jnp.ones((3,)),
            "age": jnp.ones((3,)),
            "regime_id": jnp.full(
                (3,), model.regime_names_to_ids["child"], dtype=jnp.int32
            ),
            "eps": (
                PhysicalValues(values=np.asarray([0.0, 0.5, 1.0]))
                if physical is None
                else physical
            ),
        },
        seed=17,
        log_level="off",
    )


def test_grid_replay_refuses_certified_at_declaration() -> None:
    """Certified replay refuses Cartesian omission before a Model is prepared."""
    with pytest.raises(UnsupportedOperationError, match=r"certified.*grid"):
        DCEGM(
            savings_grid=IrregSpacedGrid(points=np.asarray([0.125, 0.5, 1.0])),
            replay="grid",
            accuracy="certified",
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_grid_replay_exposes_literal_candidate_omission() -> None:
    """The declared grid omits the original half-log maximiser and fails its budgets."""
    candidate = _model(replay="candidate")
    grid = _model(replay="grid")
    exact = candidate.solve(params=_params(), log_level="off")
    approximate = grid.solve(params=_params(), log_level="off")
    # Publication and Cartesian omission are different errors: both solve
    # the identical original candidate problem on these represented nodes.
    expected_values = np.log(np.asarray([0.5, 1.0, 1.5]) / 2) + 0.5
    for solved in (exact, approximate):
        np.testing.assert_array_almost_equal(
            solved.values[1]["child"], expected_values, decimal=DECIMAL_PRECISION
        )
    # Zero additive frames preserve an exact represented tie between the two
    # Cartesian choices; varied frames remain covered by the matched-grid test.
    physical = PhysicalValues(values=np.zeros((3,)))
    actual_candidate = _simulate(
        model=candidate, solution=exact, physical=physical
    ).raw_results["child"][1]
    actual_grid = _simulate(
        model=grid, solution=approximate, physical=physical
    ).raw_results["child"][1]
    np.testing.assert_array_almost_equal(
        actual_candidate.actions["consumption"],
        np.full(3, 0.5),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(actual_grid.actions["consumption"], np.full(3, 0.25))
    np.testing.assert_array_equal(
        actual_grid.shocks["eps"], actual_candidate.shocks["eps"]
    )
    np.testing.assert_array_equal(actual_grid.shocks["eps"], np.zeros((3,)))
    reference = np.asarray(actual_candidate.actions["consumption"])
    actual = np.asarray(actual_grid.actions["consumption"])
    policy_error = np.max(
        np.abs(actual - reference) / np.maximum(0.5, np.abs(reference))
    )
    moment_error = abs(np.mean(actual) - np.mean(reference)) / max(
        1.0, abs(np.mean(reference))
    )
    eps = np.finfo(actual.dtype).eps
    assert policy_error > max(1e-4, 8 * math.sqrt(eps))
    assert moment_error > max(1e-4, 64 * eps)
    # Same represented canonical objective at exact terminal support nodes;
    # integrated solved V is deliberately not used as a simulation Q statistic.
    z = actual_grid.shocks["eps"]
    regret = (
        _flow(consumption=jnp.asarray(reference), eps=z)
        + _terminal(wealth=jnp.asarray(1 - reference))
        - _flow(consumption=jnp.asarray(actual), eps=z)
        - _terminal(wealth=jnp.asarray(1 - actual))
    )
    np.testing.assert_array_almost_equal(
        regret, np.full(3, 0.5 * math.log(4 / 3)), decimal=DECIMAL_PRECISION
    )
    ref = ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    assert ref in exact.replay_artifacts
    assert ref not in approximate.replay_artifacts


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("subject_width", [1, 2])
def test_matched_grid_replay_passes_sampled_budgets_and_preserves_route_archive(
    *,
    subject_width: int,
    tmp_path: Path,
) -> None:
    """A candidate-containing grid meets separate budgets on the declared population."""
    candidate = _model(replay="candidate", matched=True, subject_width=subject_width)
    grid = _model(replay="grid", matched=True, subject_width=subject_width)
    exact = candidate.solve(params=_params(), log_level="off")
    approximate = grid.solve(params=_params(), log_level="off")
    actual_candidate = _simulate(model=candidate, solution=exact).raw_results["child"][
        1
    ]
    actual_grid = _simulate(model=grid, solution=approximate).raw_results["child"][1]
    reference = np.asarray(actual_candidate.actions["consumption"])
    actual = np.asarray(actual_grid.actions["consumption"])
    np.testing.assert_array_almost_equal(
        actual, np.full(3, 0.5), decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(
        actual_grid.shocks["eps"], actual_candidate.shocks["eps"]
    )
    eps = np.finfo(actual.dtype).eps
    assert np.max(
        np.abs(actual - reference) / np.maximum(0.5, np.abs(reference))
    ) <= max(1e-4, 8 * math.sqrt(eps))
    assert abs(np.mean(actual) - np.mean(reference)) / max(
        1.0, abs(np.mean(reference))
    ) <= max(1e-4, 64 * eps)
    candidate_route = candidate._regimes["child"].simulation.replay_route
    grid_route = grid._regimes["child"].simulation.replay_route
    assert isinstance(candidate_route, CaseUReplaySpec)
    assert isinstance(grid_route, CaseUReplaySpec)
    assert candidate_route.replay_mode is ReplayMode.EXACT_REPLAY
    assert grid_route.replay_mode is ReplayMode.VALID_RECOMPUTATION
    assert candidate_route.identity != grid_route.identity
    path = tmp_path / "grid-solution.zip"
    approximate.save(path=path)
    loaded = load_solution(path=path)
    fresh = _model(replay="grid", matched=True, subject_width=subject_width)
    restored = _simulate(model=fresh, solution=loaded).raw_results["child"][1]
    np.testing.assert_array_equal(
        restored.actions["consumption"], actual_grid.actions["consumption"]
    )
    with pytest.raises(InvalidSimulationInputError):
        _simulate(model=candidate, solution=loaded)


def test_literal_branch_flip_requires_original_discrete_case_u_installation() -> None:
    """The scalar prototype does not claim the required branch witness is installed."""
    # Original candidates: A has resources1 and c=.5; B has resources.5
    # and c=.25 plus its specified bonus. Cartesian {.25,.75} omits A's
    # maximiser, while B's maximiser is represented. This mathematical control
    # cannot replace the still-required genuine two-branch Model witness.
    candidate_a = math.log(0.5)
    candidate_b = math.log(0.25) + 0.5 * math.log(7 / 2)
    grid_a = 0.5 * math.log(0.25) + 0.5 * math.log(0.75)
    grid_b = candidate_b
    assert candidate_a > candidate_b
    assert grid_a < grid_b
    # The genuine declaration reaches the original scalar-only owner refusal;
    # it is not claimed to execute the required two-branch forward gate.
    with pytest.raises(UnsupportedOperationError, match="passive/choice"):
        _model(replay="grid", branches=True)
