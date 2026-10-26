"""Actual NB ride cells select complete branch candidates before current-node CE."""

import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY, CaseUReplayChildren
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.current_iid_case_u_native import CaseUExactArguments
from _lcm.solution.current_iid_nbegm import NativeNBCaseUPeriodKernel
from _lcm.solution.nbegm import _NBEGMBranchSolver, _NBEGMCellContinuation
from lcm import (
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    Model,
    categorical,
)
from lcm.persistence import load_solution
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactRef, ResultRetention, SolutionResult
from lcm.solvers import NBEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations
from tests.simulation.test_case_u_reconstruction import _ReplayRegime

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _Work:
    no: ScalarInt
    yes: ScalarInt


def _flow(
    *,
    consumption: ContinuousAction,
    eps: FloatND,
    work: DiscreteAction,
    work_penalty: FloatND,
) -> FloatND:
    return (1 + eps) * jnp.log(consumption) - work_penalty * work


def _resources(
    *, wealth: ContinuousState, work: DiscreteAction, work_gain: FloatND
) -> FloatND:
    return wealth + work_gain * work


def _inverse(*, marginal_continuation: FloatND, eps: FloatND) -> FloatND:
    return (1 + eps) / marginal_continuation


def _next_q(
    *, q: ContinuousState, work: DiscreteAction, ride_bonus: FloatND
) -> ContinuousState:
    return q + ride_bonus * work


def _terminal(*, wealth: ContinuousState, q: ContinuousState) -> FloatND:
    return jnp.log(wealth) + q


def _params(*, tie: bool = False) -> UserParams:
    return {
        "discount_factor": 1.0,
        "child": {
            "utility": {"work_penalty": 0.0 if tie else 1.25},
            "resources": {"work_gain": 0.0 if tie else 1.5},
            "next_wealth": {"slope": 1.0},
            "next_q": {"ride_bonus": 0.0 if tie else 0.25},
        },
    }


def _model(*, tie: bool = False) -> Model:
    declarations = _actual_declarations()
    original = declarations.regimes["child"]
    assert isinstance(original, ConsumptionSavingsRegime)
    savings = IrregSpacedGrid(
        points=np.asarray([0.25, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0, 4.5])
    )
    child = replace(
        original,
        states={
            "wealth": IrregSpacedGrid(points=np.asarray([1.5, 3.0])),
            "q": IrregSpacedGrid(points=np.asarray([0.0, 1.0])),
        },
        actions={
            "consumption": IrregSpacedGrid(points=np.asarray([0.1, 4.0])),
            "work": DiscreteGrid(_Work),
        },
        functions={
            "utility": _flow,
            "inverse_marginal_utility": _inverse,
            "resources": _resources,
            "savings": original.functions["savings"],
        },
        state_transitions={
            **original.state_transitions,
            "q": {"done": _next_q},
        },
        solver=NBEGM(savings_grid=savings),
    )
    done = replace(
        declarations.regimes["done"],
        states={
            "wealth": savings,
            "q": IrregSpacedGrid(points=np.asarray([0.0, 0.25, 1.0, 1.25])),
        },
        functions={"utility": _terminal},
    )
    return Model._from_call_owned_params(
        declarations=replace(
            declarations,
            regimes={"child": child, "done": done},
            initial_nodes=InitialNodes(by_age={1: "child"}),
            edges={"child": {"done": (1,)}},
            regime_id_class=_ReplayRegime,
            execution_config=ExecutionConfig(device_memory_bytes=None),
            shocks={
                "income": ShockGroup(
                    shocks={"eps": Uniform(start=0.0, stop=1.0)},
                    nodes=UserNodes(
                        nodes=np.asarray([[0.0], [1.0]]),
                        weights=np.asarray([0.5, 0.5]),
                    ),
                )
            },
        ),
        params=_params(tie=tie),
    )


def _oracle(
    *, wealth: float, q: float, shock: float, tie: bool = False
) -> tuple[float, float, float, float, int]:
    """Enumerate the original discrete order on represented Euler/corner nodes."""
    candidates = []
    for work in (0, 1):
        resources = wealth + (0.0 if tie else 1.5) * work
        saving = max(0.25, resources / (2 + shock))
        consumption = resources - saving
        value = (
            (1 + shock) * math.log(consumption)
            + math.log(saving)
            + q
            + (0.0 if tie else 0.25) * work
            - (0.0 if tie else 1.25) * work
        )
        candidates.append((value, (1 + shock) / consumption, consumption, saving, work))
    return max(candidates, key=lambda candidate: candidate[0])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("inspect_projection", [False, True])
def test_nbegm_ride_u_model_selects_each_branch_before_current_expectation(
    *,
    tmp_path: Path,
    inspect_projection: bool,
) -> None:
    """Per-branch q landings distinguish the actual continuation reader classes."""
    model = _model()
    solution = (
        model.solve(
            params=_params(),
            retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
            log_level="off",
        )
        if inspect_projection
        else model.solve(params=_params(), log_level="off")
    )
    ref = ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    if not inspect_projection:
        assert solution.metadata.retention is ResultRetention.VALUES_AND_REPLAY
        assert ref not in solution.retained_continuations
    assert ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY) in (
        solution.replay_artifacts
    )
    expected_values = np.asarray(
        [
            [
                np.mean([_oracle(wealth=w, q=q, shock=z)[0] for z in (0.0, 1.0)])
                for q in (0.0, 1.0)
            ]
            for w in (1.5, 3.0)
        ]
    )
    np.testing.assert_array_almost_equal(
        solution.values[1]["child"], expected_values, decimal=DECIMAL_PRECISION
    )
    if inspect_projection:
        stored = solution.retained_continuations[ref]
        assert isinstance(stored, ProjectedEGMCarry)
        for row, q in enumerate((0.0, 1.0)):
            for index, wealth in enumerate((1.5, 3.0)):
                assert float(stored.grid[row, index]) == wealth
                np.testing.assert_almost_equal(
                    stored.economic_marginal[row, index],
                    np.mean(
                        [_oracle(wealth=wealth, q=q, shock=z)[1] for z in (0.0, 1.0)]
                    ),
                    decimal=DECIMAL_PRECISION,
                )
    wealth = np.asarray([1.5, 1.5, 3.0, 3.0, 3.0])
    q = np.asarray([0.0, 1.0, 0.25, 1.0, 0.25])
    shocks = np.asarray([0.0, 1.0, 0.0, 1.0, 0.5])
    initial = {
        "wealth": jnp.asarray(wealth),
        "q": jnp.asarray(q),
        "age": jnp.ones(wealth.shape),
        "regime_id": jnp.full(
            wealth.shape, model.regime_names_to_ids["child"], jnp.int32
        ),
        "eps": PhysicalValues(values=shocks),
    }
    result = model.simulate(
        params=_params(),
        initial_conditions=initial,
        solution=solution,
        seed=17,
        log_level="off",
    )
    original = result.raw_results["child"][1]
    expected = np.asarray(
        [
            _bank_oracle(wealth=w, q=a, shock=z)
            for w, a, z in zip(wealth, q, shocks, strict=True)
        ]
    )
    np.testing.assert_array_equal(original.actions["work"], expected[:, 4])
    np.testing.assert_array_almost_equal(
        original.actions["consumption"], expected[:, 2], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["wealth"],
        expected[:, 3],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["q"],
        q + 0.25 * expected[:, 4],
        decimal=DECIMAL_PRECISION,
    )
    _assert_original_native_tuple(model=model, solution=solution)
    path = tmp_path / "ride-u.lcm"
    solution.save(path=path)
    restored = _model().simulate(
        params=_params(),
        initial_conditions=initial,
        solution=load_solution(path=path),
        seed=17,
        log_level="off",
    )
    np.testing.assert_array_equal(
        restored.raw_results["child"][1].actions["work"], original.actions["work"]
    )
    np.testing.assert_array_equal(
        restored.raw_results["child"][1].actions["consumption"],
        original.actions["consumption"],
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_ride_u_exact_branch_tie_keeps_original_first_action() -> None:
    model = _model(tie=True)
    solution = model.solve(params=_params(tie=True), log_level="off")
    result = model.simulate(
        params=_params(tie=True),
        solution=solution,
        initial_conditions={
            "wealth": jnp.asarray([1.5, 3.0]),
            "q": jnp.asarray([0.0, 1.0]),
            "age": jnp.ones((2,)),
            "regime_id": jnp.full((2,), model.regime_names_to_ids["child"], jnp.int32),
            "eps": PhysicalValues(values=np.asarray([0.0, 1.0])),
        },
        seed=17,
        log_level="off",
    )
    np.testing.assert_array_equal(
        result.raw_results["child"][1].actions["work"], [0, 0]
    )


def _assert_original_native_tuple(*, model: Model, solution: SolutionResult) -> None:
    """The branch chosen by the literal oracle owns every native selected channel."""
    from _lcm.egm.current_iid_nbegm import (  # noqa: PLC0415 - baseline admits first
        NBCaseURidePoint,
    )

    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    point = kernel.point
    assert isinstance(point, NBCaseURidePoint)
    future = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    ]
    assert isinstance(future, CaseUReplayChildren)
    arguments = kernel.program.argument_builder
    assert isinstance(arguments, CaseUExactArguments)
    flat, _generation = model._prepare_call(params=_params())
    q_value = jnp.asarray(0.25)
    pool = {
        **union_fixed_params(
            fixed_flat_params=arguments.point.fixed_flat_params,
            regime_name="child",
            transition_target_names=("done",),
        ),
        **union_free_params(
            flat_params=flat, regime_name="child", transition_target_names=("done",)
        ),
        "period": jnp.int32(1),
        "age": jnp.asarray(1.0),
        "q": q_value,
    }
    physical = {"eps": jnp.asarray(1.0)}
    liquid = jnp.asarray(1.5)
    actual = point.select(
        liquid=liquid, physical=physical, pool=pool, children=future.children
    )
    expected = _oracle(wealth=1.5, q=0.25, shock=1.0)
    np.testing.assert_array_equal(actual.output.action_codes, [expected[-1]])
    kwargs = {
        **pool,
        **physical,
        "wealth": point.source_grid,
        "work": jnp.asarray([0, 1], dtype=jnp.int32),
    }
    core = point.core
    rows = _NBEGMCellContinuation(
        kwargs=kwargs,
        carry=MappingProxyType(dict(future.children)),
        comap_bindings=MappingProxyType({}),
        savings_grid=point.savings_nodes,
        continuation_plan=core.continuation_plan,
        statics=core.statics,
        regime_name="child",
        cliff_candidates=False,
        schedule_spec=core.schedule_spec,
    )((q_value,))
    assert isinstance(rows, tuple)
    value, marginal = rows
    assert isinstance(value, jax.Array)
    assert isinstance(marginal, jax.Array)
    solver = core.envelope_build.bind_cell(kwargs)
    original = _NBEGMBranchSolver(
        cell_solver=solver,
        cell=MappingProxyType({"q": pool["q"]}),
        query_grid=point.source_grid,
        feasibility_partition=None,
        feasible_interval_mask=None,
        interval_continuation=None,
    ).solve(
        action_binding={"work": jnp.int32(expected[-1])},
        branch_cont_value=value[expected[-1]],
        branch_cont_marginal=marginal[expected[-1]],
        branch_extra_cont_value=None,
        branch_cliff_savings=None,
        source_query=liquid[None],
        return_owner=True,
    )
    np.testing.assert_array_equal(
        actual.output.original.native_owner, original.owner[0]
    )
    np.testing.assert_array_equal(actual.output.original.policy, original.policy[0])
    np.testing.assert_array_equal(
        actual.output.original.savings, original.query_data.savings[0]
    )
    np.testing.assert_array_equal(actual.output.original.marginal, original.marginal[0])
    np.testing.assert_array_equal(
        actual.output.original.corner,
        original.query_data.savings[0] == point.savings_nodes[0],
    )
    assert bool(actual.output.original.valid)
    streamed_statics = replace(core.statics, branch_width=1)
    streamed_point = replace(
        point,
        core=replace(
            core,
            statics=streamed_statics,
            envelope_build=replace(
                core.envelope_build,
                bind_cell=replace(
                    core.envelope_build.bind_cell, statics=streamed_statics
                ),
            ),
        ),
    )
    streamed = jax.jit(streamed_point.select)(
        liquid=liquid, physical=physical, pool=pool, children=future.children
    )
    np.testing.assert_array_equal(
        streamed.output.action_codes, actual.output.action_codes
    )
    for channel in ("native_owner", "policy", "savings", "corner", "feasible", "valid"):
        np.testing.assert_array_equal(
            getattr(streamed.output.original, channel),
            getattr(actual.output.original, channel),
        )
    np.testing.assert_array_almost_equal(
        streamed.output.original.value,
        actual.output.original.value,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        streamed.output.original.marginal,
        actual.output.original.marginal,
        decimal=DECIMAL_PRECISION,
    )


def _bank_oracle(
    *, wealth: float, q: float, shock: float
) -> tuple[float, float, float, float, int]:
    """Enumerate the original finite Euler links and source-node saving corners.

    Off-quadrature value is the original candidate's affine value, not a fresh
    continuous optimum or an integrated stored value. These smooth affine-budget
    source links have no folds; every dense corner is a self-pair at a source node.
    """
    savings = (0.25, 0.5, 0.75, 1.0, 1.5, 2.25, 3.0, 4.5)
    branches = []
    for work in (0, 1):
        consumption = tuple((1 + shock) * saving for saving in savings)
        liquid = tuple(
            c + saving - 1.5 * work
            for c, saving in zip(consumption, savings, strict=True)
        )
        values = tuple(
            (1 + shock) * math.log(c) + math.log(saving) + q - work
            for c, saving in zip(consumption, savings, strict=True)
        )
        candidates = []
        for index in range(len(savings) - 1):
            if liquid[index] <= wealth <= liquid[index + 1]:
                weight = (wealth - liquid[index]) / (liquid[index + 1] - liquid[index])
                candidates.append(
                    (
                        values[index] + weight * (values[index + 1] - values[index]),
                        1 / savings[index]
                        + weight * (1 / savings[index + 1] - 1 / savings[index]),
                        consumption[index]
                        + weight * (consumption[index + 1] - consumption[index]),
                        savings[index] + weight * (savings[index + 1] - savings[index]),
                        work,
                    )
                )
        if wealth in (1.5, 3.0):
            for saving in savings:
                c = wealth + 1.5 * work - saving
                if c > 0:
                    candidates.append(
                        (
                            (1 + shock) * math.log(c) + math.log(saving) + q - work,
                            (1 + shock) / c,
                            c,
                            saving,
                            work,
                        )
                    )
        branches.append(max(candidates, key=lambda row: row[0]))
    return max(branches, key=lambda row: row[0])
