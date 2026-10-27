"""Genuine Model-installed intermediate Case-U exact publication and parent reads."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import ProjectedContinuationSpec, ProjectedProducerDate
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import _ChildEulerState
from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY, CaseUReplayChildren
from _lcm.egm.current_iid_demands import ParentResourcesDemand
from _lcm.egm.projected_child import ProjectedChildReader
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.execution.core_program import (
    CoreBuildContext,
    core_program_graph,
    materialize_core_program,
    retained_artifact_payload_type,
    select_programs,
)
from _lcm.execution.output_layout import StateAxesLeading
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.current_iid_case_u_native import (
    NativeCaseUPeriodKernel,
    PreparedCaseUDate,
    acquire_installed_case_u_date,
    prepare_case_u_date,
)
from _lcm.solution.dcegm import EGMStepBuild, _BoundDCEGM
from _lcm.solution.feasibility import BoundedResult
from _lcm.solution.generation import (
    CanonicalModelGeneration,
    GenerationSolverPreparation,
)
from _lcm.solution.model_authority import build_solution_authority
from _lcm.typing import ArtifactPayload, EconFunctionKwargs, FlatParams
from lcm import ConsumptionSavingsRegime, Model
from lcm.exceptions import ExecutionPlanningError, UnsupportedOperationError
from lcm.solver_api import (
    EGM_CONTINUATION,
    ArtifactRef,
    PersistencePolicy,
    ResultRetention,
)
from lcm.solvers import DCEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations, _params
from tests.egm.test_current_iid_generation_materialization import _owner
from tests.egm.test_current_iid_parent_demands import _resources

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _utility_u(
    *, consumption: ContinuousAction, wealth: ContinuousState, eps: FloatND
) -> FloatND:
    return jnp.log(consumption) + eps * wealth


def _model_u() -> Model:
    """Build genuine utility-shock U declarations through the actual constructor."""
    declarations = _actual_declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    assert isinstance(child.solver, DCEGM)
    child = replace(
        child,
        solver=replace(child.solver, current_iid_projection_capacity=8),
        functions={**child.functions, "utility": _utility_u, "resources": _resources},
    )
    declarations = replace(
        declarations, regimes={**declarations.regimes, "child": child}
    )
    return Model._from_call_owned_params(declarations=declarations, params=_params())


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_genuine_installed_case_u_utility_shock_closes_parent_and_refuses_holes() -> (
    None
):
    """Actual source utility shock reaches original candidates and the real parent."""
    model = _model_u()
    params = _params()
    flat, generation = model._prepare_call(params=params)
    prepared = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=1
    )
    owner = generation.solver_preparation(regime_name="child", period=1)
    assert isinstance(owner.installed_kernel, NativeCaseUPeriodKernel)
    assert owner.installed_kernel.prepared is prepared
    assert prepared.context is owner.context
    assert prepared.build is owner.build
    assert prepared.context.current_iid_plans[1] is owner.plan
    _assert_original_u_template(prepared=prepared)
    _assert_original_u_retention(generation=generation, flat=flat)
    # This executes the actual installed terminal -> U child -> parent graph.
    solution = model.solve(
        params=params,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    carry = solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(carry, ProjectedEGMCarry)
    assert bool(carry.valid)
    assert not bool(jnp.any(carry.link_valid))
    parent_values = solution.values[0]["parent"]
    assert isinstance(parent_values, jax.Array)
    assert bool(jnp.all(jnp.isfinite(parent_values)))
    parent = generation.solver_preparation(regime_name="parent", period=0)
    read = parent.step.pieces.continuation_plan.child_reads["child"]
    assert read.projected_date is prepared.date
    done = _assert_original_u_future_producer(
        kernel=owner.installed_kernel,
        payload=solution.replay_artifacts[
            ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
        ],
        done=solution.retained_continuations[
            ArtifactRef(period=2, regime="done", key=EGM_CONTINUATION)
        ],
    )
    arguments = prepared.program.argument_builder(
        CoreBuildContext(
            state_action_space=prepared.context.state_action_space,
            next_regime_to_V_arr={},
            next_regime_to_continuation={"done": done},
            flat_params=flat,
            period=1,
            ages=prepared.context.ages,
        )
    )
    pools = arguments["_lcm_parent_demand_pools"]
    assert isinstance(pools, tuple)
    assert len(pools) == 1
    raw_pool = pools[0]
    assert isinstance(raw_pool, Mapping)
    exact_pool = {}
    for name, value in raw_pool.items():
        assert isinstance(name, str)
        assert isinstance(value, jax.Array)
        exact_pool[name] = value
    pool: EconFunctionKwargs = MappingProxyType(exact_pool)
    law = _ChildEulerState(
        next_state_func=read.euler_state_func,
        combo_pool=pool,
        post_decision_name=parent.step.pieces.continuation_plan.post_decision_name,
        next_state_key=read.next_state_key,
    )
    reader = ProjectedChildReader.bind_date(
        date=prepared.date, target="child", carry=carry, next_liquid=law
    )
    actual = reader(jnp.asarray(0.5))
    assert actual.query_status is not None
    assert not bool(actual.query_status[0].coverage_missing)
    # m=2*.5+.5, beta=1. Original savings grid contains m/2=.75.
    # Current eps in {0,1} is equally weighted and adds eps*m to utility.
    expected_value = 2 * jnp.log(jnp.asarray(0.75)) + 0.75
    expected_parent_marginal = (2 / jnp.asarray(1.5) + 0.5) * 2
    np.testing.assert_almost_equal(
        actual.output,
        (expected_value, expected_parent_marginal),
        decimal=DECIMAL_PRECISION,
    )
    # The uncertainty hole is read by the same actual dated parent law.
    hole = reader(jnp.asarray(0.625))
    assert hole.query_status is not None
    assert bool(hole.query_status[0].coverage_missing)
    _reject_hole_after_zero_discount(parent=parent, carry=carry, flat=flat)


def _reject_hole_after_zero_discount(
    *, parent: GenerationSolverPreparation, carry: ProjectedEGMCarry, flat: FlatParams
) -> None:
    """Use the actual installed parent and its original date/params/child reads."""
    # Losing exact source support cannot disappear behind a zero parent discount.
    corrupted = replace(carry, valid_length=jnp.int32(1))
    parent_call = CoreBuildContext(
        state_action_space=parent.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation={"child": corrupted},
        flat_params=flat,
        period=0,
        ages=parent.context.ages,
    )
    parent_program = materialize_core_program(
        program=core_program_graph(kernel=parent.installed_kernel)["main"],
        context=parent_call,
    )
    native = dict(parent_program.arguments)
    before_pool: EconFunctionKwargs = {
        name: value for name, value in native.items() if isinstance(value, jax.Array)
    }
    (original_discount,) = tuple(
        parent.step.pieces.build_W_kwargs(before_pool).values()
    )
    assert isinstance(original_discount, jax.Array)
    np.testing.assert_array_equal(original_discount, 1.0)
    discount_keys = tuple(name for name in native if name.endswith("__discount_factor"))
    assert discount_keys
    for name in discount_keys:
        value = native[name]
        assert isinstance(value, jax.Array)
        native[name] = jnp.zeros_like(value)
    after_pool: EconFunctionKwargs = {
        name: value for name, value in native.items() if isinstance(value, jax.Array)
    }
    (consumed_discount,) = tuple(parent.step.pieces.build_W_kwargs(after_pool).values())
    assert isinstance(consumed_discount, jax.Array)
    np.testing.assert_array_equal(consumed_discount, 0.0)
    rejected = parent_program.function(**native)
    assert isinstance(rejected, BoundedResult)
    assert rejected.query_status is not None
    assert any(bool(status.coverage_missing) for status in rejected.query_status)


def _use_actual_u_template_owner(
    *, monkeypatch: pytest.MonkeyPatch, owner: SimulationEntryAllocations
) -> list[PreparedCaseUDate]:
    """Admit the real U factory subphase inside the genuine private Model build."""
    from _lcm.solution import (  # noqa: PLC0415
        current_iid_case_u_parents,
        current_iid_dated,
    )

    completed: list[PreparedCaseUDate] = []

    def prepare(
        *,
        context: SolverBuildContext,
        solver: _BoundDCEGM,
        build: EGMStepBuild,
        period: int,
        date: ProjectedProducerDate,
        call: CoreBuildContext,
        fixed_flat_params: FlatParams,
        demands: tuple[ParentResourcesDemand, ...] = (),
        materializer: DatedGenerationMaterializer | None = None,
    ) -> PreparedCaseUDate:
        assert materializer is None
        result = prepare_case_u_date(
            context=context,
            solver=solver,
            build=build,
            period=period,
            date=date,
            call=call,
            fixed_flat_params=fixed_flat_params,
            demands=demands,
            materializer=owner,
        )
        completed.append(result)
        return result

    # Only this admitted factory subphase borrows the owner. Earlier constructor,
    # validators/step geometry and cold Model admission remain separate required work.
    monkeypatch.setattr(current_iid_dated, "prepare_case_u_date", prepare)
    monkeypatch.setattr(current_iid_case_u_parents, "prepare_case_u_date", prepare)
    return completed


def test_genuine_u_template_factory_refuses_before_output_allocation() -> None:
    """The actual Model-derived U schema requires its complete explicit budget."""
    from _lcm.solution.current_iid_case_u_build import (  # noqa: PLC0415
        materialize_case_u_schema,
    )

    model = _model_u()
    _, generation = model._prepare_call(params=_params())
    prepared = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=1
    )
    owner = _owner(roots=generation.roots)
    owner.budget_bytes = 0
    try:
        with pytest.raises(ExecutionPlanningError):
            materialize_case_u_schema(schema=prepared.template, materializer=owner)
        assert not owner._pending
    finally:
        owner.close()


def test_genuine_u_template_owner_preserves_installed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual admitted U schemas retain the original installed economic values."""
    owner = _owner()
    completed = _use_actual_u_template_owner(monkeypatch=monkeypatch, owner=owner)
    try:
        model = _model_u()
        assert completed
        roots = jax.tree.leaves(owner.solve_input_roots())
        for prepared in completed:
            leaves = jax.tree.leaves(prepared.template)
            assert all(isinstance(leaf, jax.Array) for leaf in leaves)
            assert all(any(leaf is root for root in roots) for leaf in leaves)
        solution = model.solve(params=_params(), log_level="off", runtime_checks=True)
        liquid = completed[-1].context.state_action_space.states["wealth"]
        np.testing.assert_array_almost_equal(
            solution.values[1]["child"],
            2 * jnp.log(liquid / 2) + liquid / 2,
            decimal=DECIMAL_PRECISION,
        )
    finally:
        owner.close()


def _local_numpy_shape_factory(*, value: FloatND) -> FloatND:
    from numpy import zeros  # noqa: PLC0415

    return value + zeros(1)[0]


def _local_jax_shape_context(*, value: FloatND) -> FloatND:
    from jax import ensure_compile_time_eval  # noqa: PLC0415

    with ensure_compile_time_eval():
        offset = jnp.zeros(())
    return value + offset


@pytest.mark.parametrize(
    "callback", [_local_numpy_shape_factory, _local_jax_shape_context]
)
def test_genuine_u_shape_guard_refuses_local_import_authorities(
    *, callback: Callable[..., FloatND]
) -> None:
    """Function-local imports cannot hide producers from pretrace admission."""
    from _lcm.solution.current_iid_case_u_build import (  # noqa: PLC0415
        guard_case_u_shape,
    )

    model = _model_u()
    flat, generation = model._prepare_call(params=_params())
    prepared = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=1
    )
    owner = _owner(roots=generation.roots)
    try:
        with pytest.raises(UnsupportedOperationError, match="local import"):
            guard_case_u_shape(
                function=callback,
                context=CoreBuildContext(
                    state_action_space=prepared.context.state_action_space,
                    next_regime_to_V_arr={},
                    next_regime_to_continuation={},
                    flat_params=flat,
                    period=1,
                    ages=prepared.context.ages,
                ),
                fixed_flat_params=MappingProxyType({}),
                materializer=owner,
            )
        assert not owner._pending
    finally:
        owner.close()


def _assert_original_u_future_producer(
    *,
    kernel: NativeCaseUPeriodKernel,
    payload: ArtifactPayload,
    done: ArtifactPayload,
) -> EGMCarry:
    """The actual retained bundle has a typed producer and original child data."""
    assert isinstance(done, EGMCarry)
    graph = core_program_graph(kernel=kernel)
    assert (
        retained_artifact_payload_type(graph=graph, key=CASE_U_FUTURE_REPLAY)
        is CaseUReplayChildren
    )
    assert tuple(select_programs(graph=graph, retain_replay=False)) == ("main",)
    assert isinstance(payload, CaseUReplayChildren)
    for retained, actual_child in zip(
        jax.tree.leaves(payload.children["done"]), jax.tree.leaves(done), strict=True
    ):
        np.testing.assert_array_equal(retained, actual_child)
    return done


def _assert_original_u_template(*, prepared: PreparedCaseUDate) -> None:
    """Concrete original date placeholders grant no coverage or extra axes."""
    assert prepared.program.requirements.axes == ()
    assert prepared.program.requirements.host_buffers == {}
    assert prepared.program.donation_candidates == ()
    assert all(
        isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(prepared.template)
    )
    assert not bool(prepared.date.template.valid)
    assert prepared.template.output.projected is prepared.date.template
    assert all(
        isinstance(role, StateAxesLeading) and role.shape == leaf.shape
        for role, leaf in zip(
            jax.tree.leaves(prepared.program.output_roles),
            jax.tree.leaves(prepared.template),
            strict=True,
        )
    )


def _assert_original_u_retention(
    *, generation: CanonicalModelGeneration, flat: FlatParams
) -> None:
    """Actual authority recognizes the canonical U spec and refuses its clone."""
    execution = generation.execution
    regimes = execution._regimes
    child = regimes["child"]
    spec = child.solution.continuation_spec
    assert isinstance(spec, ProjectedContinuationSpec)
    ref = ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    original = build_solution_authority(
        regimes=regimes,
        flat_params=flat,
        ages=execution._time,
        process_grid_resolver=None,
    )
    assert (
        original.artifacts[ref].descriptor.persistence
        is PersistencePolicy.MODEL_VERIFIABLE
    )
    detached_child = replace(
        child, solution=replace(child.solution, continuation_spec=replace(spec))
    )
    detached = build_solution_authority(
        regimes=MappingProxyType({**regimes, "child": detached_child}),
        flat_params=flat,
        ages=execution._time,
        process_grid_resolver=None,
    )
    assert (
        detached.artifacts[ref].descriptor.persistence
        is PersistencePolicy.NOT_PERSISTED
    )
