"""The original Euler consumer closes a native exact-only child demand bank.

This internal seam keeps the public current-IID DCEGM guard closed. Future
continuations are captured from an ordinary canonical solve; no successful
resources row, child value, or reader result is supplied by the fixture.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType
from typing import Never

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import ProjectedContinuationSpec
from _lcm.egm.current_iid_declarations import projected_producer_declarations
from _lcm.egm.current_iid_demands import ParentResourcesDemand
from _lcm.egm.current_iid_native import NativeCaseRCore
from _lcm.egm.preferences import BoundUtilityOfAction
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.egm.step import _EGMStep
from _lcm.egm.step_core import (
    OwnedResourcesRow,
    _EgmKernelPieces,
    _get_compute_node,
    _SolveOneCombo,
)
from _lcm.egm.upper_envelope import get_upper_envelope
from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope
from _lcm.engine import Regime as CanonicalRegime
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionRequirements,
    MaterializedCoreProgram,
    materialize_core_program,
)
from _lcm.regime_building import processing
from _lcm.regime_building.processing import _SolutionBuild
from _lcm.solution.current_iid_case_r import (
    CaseRArguments,
    CaseRProgramFunction,
    NativeCaseRPeriodKernel,
    case_r_core_programs,
    native_case_r_template,
)
from _lcm.solution.current_iid_parent_demands import (
    _parent_pieces,
    close_case_r_parent_demands,
)
from _lcm.solution.dcegm import _BoundDCEGM, _DCEGMArgumentBuilder, _DCEGMPeriodKernel
from _lcm.solution.feasibility import BoundedResult, validate_query_input_status
from _lcm.typing import (
    FlatEdgeParams,
    FlatParams,
    FlatRegimeParams,
    PytreeValue,
    ShapeDtypePytree,
)
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    InitialNodes,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidValueFunctionError, UnsupportedOperationError
from lcm.shocks import Normal, ShockGroup, UserNodes
from lcm.solvers import DCEGM, ExactEnvelope, GridSearch
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ReferenceName,
    RegimeName,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION
from tests.solution._nbegm_direct_oracle import OracleContext, ride_along_kernel


@categorical(ordered=False)
class _RegimeId:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _resources(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _current_resources(*, wealth: ContinuousState, eps: FloatND) -> FloatND:
    return wealth + eps


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _landing(
    *, savings: FloatND, slope: FloatND, period: ScalarInt, age: FloatND | ScalarInt
) -> ContinuousState:
    return slope * savings + 0.5 + 0.125 * period + 0.0625 * age


def _identity(*, savings: FloatND, slope: FloatND) -> ContinuousState:
    return slope * savings


def _terminal(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _model(*, current_child: bool) -> Model:
    """Canonical baseline and separately declared current-node law model."""
    grid = IrregSpacedGrid(points=np.array([1.0, 4.0]))
    # The parent's required upper saving lands at 8.5, or 12.5 after rebinding
    # its slope. Each log/log child optimum is half its resources (the landing
    # plus current income), so both endpoint optima remain explicit savings nodes.
    child_grid = IrregSpacedGrid(points=np.array([1.0, 12.5]))
    child_nodes = IrregSpacedGrid(
        points=np.array(
            [
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.375,
                1.5,
                1.75,
                1.875,
                2.0,
                2.25,
                2.5,
                4.25,
                4.75,
                6.25,
                6.75,
                12.5,
            ]
        )
    )
    functions = {
        "utility": _utility,
        "resources": _resources,
        "savings": _savings,
    }
    if not current_child:
        functions["inverse_marginal_utility"] = _inverse
    margin = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    action = {"consumption": IrregSpacedGrid(points=np.array([0.25, 2.0]))}
    parent = ConsumptionSavingsRegime(
        states={"wealth": grid},
        actions=action,
        functions=functions,
        state_transitions={"wealth": {"child": _landing}},
        liquid=margin,
        solver=GridSearch()
        if current_child
        else DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.5, 0.75, 1.0, 4.0])),
            envelope=ExactEnvelope(),
        ),
    )
    child = ConsumptionSavingsRegime(
        states={"wealth": child_grid},
        actions=action,
        functions={
            **functions,
            "resources": _current_resources if current_child else _resources,
        },
        state_transitions={"wealth": {"done": _identity}},
        liquid=margin,
        solver=GridSearch()
        if current_child
        else DCEGM(savings_grid=child_nodes, envelope=ExactEnvelope()),
        shocks=("income",) if current_child else (),
    )
    done = Regime(states={"wealth": child_nodes}, functions={"utility": _terminal})
    return Model(
        regimes={"parent": parent, "child": child, "done": done},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "parent"}),
        edges={
            "parent": {"child": (0,)},
            "child": {"done": (1,)},
        },
        regime_id_class=_RegimeId,
        shocks={
            "income": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.array([[0.0], [1.0]]), weights=np.array([0.5, 0.5])
                ),
            )
        },
    )


def _context(context: OracleContext) -> CoreBuildContext:
    return CoreBuildContext(**context)


def _ordinary_kernels() -> tuple[
    Model, _DCEGMPeriodKernel, OracleContext, _DCEGMPeriodKernel, OracleContext
]:
    """Capture genuine future carries and original consumers from the baseline."""
    model = _model(current_child=False)
    params = {
        "discount_factor": 1.0,
        "parent": {"next_wealth": {"slope": 2.0}},
        "child": {"next_wealth": {"slope": 1.0}},
    }
    child, child_context = ride_along_kernel(
        model=model, params=params, regime_name="child", period=1
    )
    parent, parent_context = ride_along_kernel(
        model=model, params=params, regime_name="parent", period=0
    )
    assert isinstance(child, _DCEGMPeriodKernel)
    assert isinstance(parent, _DCEGMPeriodKernel)
    assert isinstance(child.demand_step, _EGMStep)
    assert isinstance(parent.demand_step, _EGMStep)
    return model, child, child_context, parent, parent_context


def _prepare_projected_parent_pieces(
    *, parent_build: _SolutionBuild, spec: ProjectedContinuationSpec
) -> _EgmKernelPieces:
    """Bind the real parent date and query slots through its original builder."""
    # Use the original builder so date, resource binding and mandatory target
    # slots are installed together, exactly as in the real dated parent route.
    assert isinstance(parent_build.solver, _BoundDCEGM)
    child_date = spec.date_at(period=1)
    prepared_parent = parent_build.solver.prepare_period_steps(
        context=parent_build.context,
        periods=(0,),
        projected_dates={"child": child_date},
    )
    parent_step = prepared_parent.steps[0]
    assert isinstance(parent_step, _EGMStep)
    parent_pieces = parent_step.pieces
    read = parent_pieces.continuation_plan.child_reads["child"]
    assert read.projected_date is child_date
    assert read.query_target == "child"
    assert parent_pieces.continuation_plan.query_targets == ("child",)
    return parent_pieces


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_real_parent_law_consumes_native_child_exact_demands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compile the native child and feed its output to the parent Euler row."""
    builds: dict[RegimeName, _SolutionBuild] = {}
    _record_original_contexts(monkeypatch=monkeypatch, captures=builds)
    model, child, child_context, parent, parent_context = _ordinary_kernels()
    law_model = _model(current_child=True)
    plan = law_model._regimes["child"].solution.current_iid_plans[1]
    assert plan is not None
    solver = model.user_regimes["child"].solver
    assert isinstance(solver, _BoundDCEGM)
    assert isinstance(solver.envelope, ExactEnvelope)
    spec = projected_producer_declarations(
        user_regimes=model.user_regimes,
        bindings={
            ("child", "solve", name): binding for name, binding, _ in plan.bindings
        },
        state_grids={"child": model._regimes["child"].solution.grids},
        active_periods={"child": (1,)},
    )["child"]
    assert isinstance(spec, ProjectedContinuationSpec)
    parent_pieces = _prepare_projected_parent_pieces(
        parent_build=builds["parent"], spec=spec
    )
    demand = ParentResourcesDemand(
        source="parent",
        period=0,
        target="child",
        spec=spec,
        pieces=parent_pieces,
        transition_targets=("child",),
    )
    pieces = replace(
        _parent_pieces(kernel=child),
        own_resources_func=_current_resources,
        collect_continuation_read_counts=True,
    )
    core = NativeCaseRCore(
        pieces=pieces,
        plan=plan,
        coefficient=Fraction(1),
        refine=get_upper_envelope(
            solver=solver, n_refined=pieces.n_pad, publish_provenance=True
        ),
        owner_capacity=solver.envelope.max_runs,
        projection_capacity=solver.current_iid_projection_capacity,
        parent_demands=(demand,),
    )
    arguments = CaseRArguments(
        base=_DCEGMArgumentBuilder(
            regime_name="child",
            stateful_targets=frozenset({"done"}),
            transition_target_names=("done",),
        ),
        plan=plan,
        parent_demands=(demand,),
    )
    template = native_case_r_template(
        spec=spec,
        period=1,
        liquid_shape=(2,),
        resources_capacity=pieces.n_pad,
        child_slots=0,
        current_nodes=plan.indices.shape[0],
    )
    programs = case_r_core_programs(
        core=core,
        arguments=arguments,
        template=template,
        requirements=CoreExecutionRequirements(),
        child_slots=0,
    )
    program = materialize_core_program(
        program=programs["replay"], context=_context(child_context)
    )
    compiled = jax.jit(program.function).lower(**program.arguments).compile()
    kernel = NativeCaseRPeriodKernel(
        programs=programs,
        spec=spec,
        period=1,
        query_input_authorities=(("child", 1, spec.authority_at(period=1)),),
        regime_name="child",
        transition_target_names=("done",),
    )
    _assert_modelwide_closure(
        kernel=kernel, parent=parent, demand=demand, builds=builds
    )
    _assert_closed_law_signatures(demand=demand)
    output = kernel(
        compiled_cores={"replay": compiled},
        **child_context,
        logger=logging.getLogger(__name__),
    )
    validate_query_input_status(
        statuses=output.query_status, authorities=kernel.query_input_authorities
    )
    projected = output.continuations[PROJECTED_EGM_CONTINUATION]
    assert isinstance(projected, ProjectedEGMCarry)
    queries = np.array([1.5, 2.0, 2.5, 8.5])
    retained = np.asarray(projected.grid)[: int(projected.valid_length)]
    for query in queries:
        assert np.count_nonzero(retained == query) == 1
    assert not bool(jnp.any(projected.link_valid))
    parent_arguments = parent.core_programs()["replay"].argument_builder(
        _context(parent_context)
    )
    parent_pool = _array_pool(
        arguments={
            name: value
            for name, value in parent_arguments.items()
            if name
            not in (
                *parent_context["state_action_space"].states,
                "next_regime_to_continuation",
            )
        }
    )
    run_parent = _SolveOneCombo(
        pieces=parent_pieces,
        pool=parent_pool,
        state_grid=parent_context["state_action_space"].states["wealth"],
        next_regime_to_continuation=MappingProxyType({"child": projected}),
        savings_point_width=None,
        stochastic_node_width=None,
        envelope_cell_width=1,
        resolved_process_grids=MappingProxyType({}),
    )
    row = run_parent.owned_resources_row(
        combo_values=(), resources_domain=jnp.asarray([1.0, 4.0]), refine=core.refine
    )
    assert not bool(row.construction_invalid)
    assert row.child_status is not None
    validate_query_input_status(
        statuses=row.child_status,
        authorities=(("child", 1, spec.authority_at(period=1)),),
    )
    _assert_parent_euler_oracle(
        row=row,
        savings=parent_pieces.savings_nodes,
        slope=2.0,
        resources_domain=(1.0, 4.0),
    )
    # Child log/log Euler optima are exactly on the original native savings grid.
    expected = 0.5 * (2.0 * np.log(queries / 2.0)) + 0.5 * (
        2.0 * np.log((queries + 1.0) / 2.0)
    )
    slots = np.searchsorted(retained, queries)
    np.testing.assert_array_almost_equal(
        np.asarray(projected.value)[slots], expected, decimal=DECIMAL_PRECISION
    )
    # Economic chain channel is the child marginal times the nonunit law slope.
    expected_chain = 2.0 * (1.0 / queries + 1.0 / (queries + 1.0))
    np.testing.assert_array_almost_equal(
        np.asarray(projected.economic_marginal)[slots] * 2.0,
        expected_chain,
        decimal=DECIMAL_PRECISION,
    )

    _assert_rebinding_and_domain_failure(
        source_params=_context(parent_context).flat_params["parent"],
        compiled=compiled,
        program=program,
        kernel=kernel,
        demand=demand,
        parent=run_parent,
        refine=core.refine,
    )
    _assert_fixed_parent_namespace(
        kernel=kernel,
        context=_context(child_context),
        parent=run_parent,
        refine=core.refine,
        demand=demand,
    )
    _assert_zero_discount_read_failure(
        parent=run_parent,
        demand=demand,
        source_params=_context(parent_context).flat_params["parent"],
    )


def _array_pool(
    *, arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
) -> Mapping[ReferenceName, jax.Array]:
    """Narrow this fixture's canonical scalar operands without coercing them."""
    result: dict[ReferenceName, jax.Array] = {}
    for name, value in arguments.items():
        assert isinstance(value, jax.Array)
        result[name] = value
    return MappingProxyType(result)


def _assert_parent_euler_oracle(
    *,
    row: OwnedResourcesRow,
    savings: FloatND,
    slope: float,
    resources_domain: tuple[float, float],
) -> None:
    """Observe the actual parent candidates, not a manually chained child read."""
    nodes = np.asarray(savings)
    landing = slope * nodes + 0.5
    expected_value = np.log(landing / 2.0) + np.log((landing + 1.0) / 2.0)
    chain = slope * (1.0 / landing + 1.0 / (landing + 1.0))
    consumption = 1.0 / chain
    resources = nodes + consumption
    # Upper savings coverage is required even when that Euler candidate's cash
    # lies outside this explicitly clipped parent row. Keep every original
    # interior anchor and exclude only that analytically out-of-domain point.
    interior = np.flatnonzero(
        (resources >= resources_domain[0]) & (resources <= resources_domain[1])
    )
    interior = interior[interior != 0]
    np.testing.assert_array_equal(interior, [1, 2])
    assert resources[-1] > resources_domain[1]
    # The first saving is the raw borrowing-floor continuation. Interior
    # candidates are above the declared current resources lower endpoint.
    np.testing.assert_array_almost_equal(
        row.expected_floor_value, expected_value[0], decimal=DECIMAL_PRECISION
    )
    active = np.arange(row.envelope.grid.size) < int(row.envelope.n_kept)
    native = np.asarray(row.envelope.grid)[active]
    slots = np.argmin(np.abs(native[:, None] - resources[None, interior]), axis=0)
    for actual, expected in (
        (native[slots], resources[interior]),
        (np.asarray(row.envelope.policy)[slots], consumption[interior]),
        (np.asarray(row.envelope.savings)[slots], nodes[interior]),
        (np.asarray(row.marginal)[slots], chain[interior]),
        (
            np.asarray(row.envelope.value)[slots],
            (np.log(consumption) + expected_value)[interior],
        ),
    ):
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=DECIMAL_PRECISION
        )


def _assert_rebinding_and_domain_failure(
    *,
    source_params: FlatRegimeParams | FlatEdgeParams,
    compiled: Callable[..., PytreeValue],
    program: MaterializedCoreProgram,
    kernel: NativeCaseRPeriodKernel,
    demand: ParentResourcesDemand,
    parent: _SolveOneCombo,
    refine: Callable[..., OwnedEnvelope],
) -> None:
    """Rebind the same executable's actual parent parameter operands."""
    pools = program.arguments["_lcm_parent_demand_pools"]
    assert isinstance(pools, tuple)
    assert len(pools) == 1
    assert isinstance(pools[0], Mapping)
    pool = _array_pool(arguments=pools[0])
    slope_names = tuple(name for name in source_params if name.endswith("__slope"))
    assert len(slope_names) == 1
    np.testing.assert_array_equal(pool[slope_names[0]], source_params[slope_names[0]])
    np.testing.assert_array_equal(parent.pool[slope_names[0]], 2.0)
    assert int(pool["period"]) == demand.period
    np.testing.assert_array_equal(pool["age"], parent.pool["age"])
    changed_pool = {**pool, slope_names[0]: jnp.asarray(3.0)}
    changed = compiled(
        **{**program.arguments, "_lcm_parent_demand_pools": (changed_pool,)}
    )
    assert isinstance(changed, BoundedResult)
    assert isinstance(changed.output, tuple)
    changed_projected = changed.output[1]
    assert isinstance(changed_projected, ProjectedEGMCarry)
    changed_queries = np.array([2.0, 2.75, 3.5, 12.5])
    changed_grid = np.asarray(changed_projected.grid)[
        : int(changed_projected.valid_length)
    ]
    assert bool(changed_projected.valid)
    for query in changed_queries:
        assert np.count_nonzero(changed_grid == query) == 1
    changed_slots = np.searchsorted(changed_grid, changed_queries)
    np.testing.assert_array_almost_equal(
        np.asarray(changed_projected.value)[changed_slots],
        np.log(changed_queries / 2.0) + np.log((changed_queries + 1.0) / 2.0),
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        np.asarray(changed_projected.economic_marginal)[changed_slots],
        1.0 / changed_queries + 1.0 / (changed_queries + 1.0),
        decimal=DECIMAL_PRECISION,
    )
    changed_parent = replace(
        parent,
        pool={**parent.pool, slope_names[0]: jnp.asarray(3.0)},
        next_regime_to_continuation=MappingProxyType({"child": changed_projected}),
    ).owned_resources_row(
        combo_values=(), resources_domain=jnp.asarray([1.0, 4.0]), refine=refine
    )
    _assert_parent_euler_oracle(
        row=changed_parent,
        savings=demand.pieces.savings_nodes,
        slope=3.0,
        resources_domain=(1.0, 4.0),
    )

    # Domain corruption remains mandatory before any parent discount mask.
    bad_pool = {**pool, slope_names[0]: jnp.asarray(20.0)}
    bad = compiled(**{**program.arguments, "_lcm_parent_demand_pools": (bad_pool,)})
    assert isinstance(bad, BoundedResult)
    with pytest.raises(InvalidValueFunctionError):
        validate_query_input_status(
            statuses=bad.query_status, authorities=kernel.query_input_authorities
        )

    with pytest.raises(
        ValueError, match=r"^No projected producer is declared at period 2\.$"
    ):
        replace(demand, period=1)
    with pytest.raises(
        TypeError, match=r"^Incoming parent omitted its projected child read\.$"
    ):
        replace(demand, target="done")


def _record_original_contexts(
    *, monkeypatch: pytest.MonkeyPatch, captures: dict[RegimeName, _SolutionBuild]
) -> None:
    """Observe the real hook while preserving its original model construction."""
    original = processing._with_declared_continuation_reads

    def record(
        *,
        canonical_regimes: dict[RegimeName, CanonicalRegime],
        solution_builds: Mapping[RegimeName, _SolutionBuild],
    ) -> MappingProxyType[RegimeName, CanonicalRegime]:
        if isinstance(
            solution_builds["parent"].kernels.period_kernels.get(0), _DCEGMPeriodKernel
        ):
            captures.update(solution_builds)
        return original(
            canonical_regimes=canonical_regimes, solution_builds=solution_builds
        )

    monkeypatch.setattr(processing, "_with_declared_continuation_reads", record)


def _assert_modelwide_closure(
    *,
    kernel: NativeCaseRPeriodKernel,
    parent: _DCEGMPeriodKernel,
    demand: ParentResourcesDemand,
    builds: Mapping[RegimeName, _SolutionBuild],
) -> None:
    """Exercise actual model contexts and reject queries even without an edge."""
    assert builds
    assert isinstance(parent.demand_step, _EGMStep)
    contexts = {name: build.context for name, build in builds.items()}
    kernels = {name: build.kernels for name, build in builds.items()}
    kernels["child"] = replace(
        kernels["child"], period_kernels={1: kernel}, continuation_spec=demand.spec
    )
    kernels["parent"] = replace(
        kernels["parent"],
        period_kernels={
            0: replace(
                parent, demand_step=replace(parent.demand_step, pieces=demand.pieces)
            )
        },
    )
    closed = close_case_r_parent_demands(kernels=kernels, contexts=contexts)
    closed_kernel = closed["child"].period_kernels[1]
    assert isinstance(closed_kernel, NativeCaseRPeriodKernel)
    closed_function = closed_kernel.programs["main"].function
    assert isinstance(closed_function, CaseRProgramFunction)
    actual = closed_function.core.parent_demands
    assert actual is not None
    assert len(actual) == 1
    assert actual[0].pieces is demand.pieces
    assert actual[0].spec is demand.spec
    # A conditional/shadow same-period read remains outside the finite class,
    # even if ordinary transition consumers have been removed from the graph.
    graph = contexts["parent"].solution_reachability
    no_parent = replace(
        graph,
        targets_by_period=tuple(
            MappingProxyType(
                {
                    source: tuple(target for target in targets if target != "child")
                    for source, targets in period.items()
                }
            )
            for period in graph.targets_by_period
        ),
    )
    no_parent_contexts = {
        name: replace(context, solution_reachability=no_parent)
        for name, context in contexts.items()
    }
    no_parent_contexts["parent"] = replace(
        no_parent_contexts["parent"], same_period_ref_regimes=("child",)
    )
    with pytest.raises(UnsupportedOperationError, match="same-period"):
        close_case_r_parent_demands(kernels=kernels, contexts=no_parent_contexts)
    with pytest.raises(TypeError, match="every source kernel"):
        close_case_r_parent_demands(
            kernels={
                name: value for name, value in kernels.items() if name != "parent"
            },
            contexts=contexts,
        )


def _variadic_law(*args: FloatND, **kwargs: FloatND) -> dict[str, FloatND]:
    del args
    return {"next_wealth": kwargs["savings"]}


def _stateful_law(*, savings: FloatND, wealth: FloatND) -> dict[str, FloatND]:
    return {"next_wealth": savings + wealth}


class _UninspectableLaw:
    @property
    def __signature__(self) -> Never:
        raise ValueError("This callable has no closed dependency signature.")

    def __call__(self, **kwargs: FloatND) -> dict[str, FloatND]:
        return {"next_wealth": kwargs["savings"]}


def _assert_closed_law_signatures(*, demand: ParentResourcesDemand) -> None:
    read = demand.pieces.continuation_plan.child_reads["child"]
    for law in (_variadic_law, _stateful_law, _UninspectableLaw()):
        plan = replace(
            demand.pieces.continuation_plan,
            child_reads=MappingProxyType(
                {"child": replace(read, euler_state_func=law)}
            ),
        )
        with pytest.raises(UnsupportedOperationError):
            replace(demand, pieces=replace(demand.pieces, continuation_plan=plan))


def _assert_fixed_parent_namespace(
    *,
    kernel: NativeCaseRPeriodKernel,
    context: CoreBuildContext,
    parent: _SolveOneCombo,
    refine: Callable[..., OwnedEnvelope],
    demand: ParentResourcesDemand,
) -> None:
    """Bind distinct actual parent/child fixed namespaces through the native adapter."""
    source_keys = tuple(
        name for name in context.flat_params["parent"] if name.endswith("__slope")
    )
    target_keys = tuple(
        name for name in context.flat_params["child"] if name.endswith("__slope")
    )
    assert len(source_keys) == len(target_keys) == 1
    fixed: FlatParams = MappingProxyType(
        {
            "parent": MappingProxyType({source_keys[0]: jnp.asarray(3.0)}),
            "child": MappingProxyType({target_keys[0]: jnp.asarray(1.0)}),
        }
    )
    free = {
        name: MappingProxyType(
            {
                key: value
                for key, value in values.items()
                if key not in fixed.get(name, {})
            }
        )
        for name, values in context.flat_params.items()
    }
    fixed_kernel = kernel.with_fixed_params(fixed_flat_params=fixed)
    materialized = materialize_core_program(
        program=fixed_kernel.programs["replay"],
        context=replace(context, flat_params=free),
    )
    pools = materialized.arguments["_lcm_parent_demand_pools"]
    assert isinstance(pools, tuple)
    assert len(pools) == 1
    assert isinstance(pools[0], Mapping)
    pool = _array_pool(arguments=pools[0])
    np.testing.assert_array_equal(pool[source_keys[0]], 3.0)
    np.testing.assert_array_equal(pool[f"child__{target_keys[0]}"], 1.0)
    evaluated = (
        jax.jit(materialized.function)
        .lower(**materialized.arguments)
        .compile()(**materialized.arguments)
    )
    assert isinstance(evaluated, BoundedResult)
    assert isinstance(evaluated.output, tuple)
    projected = evaluated.output[1]
    assert isinstance(projected, ProjectedEGMCarry)
    row = replace(
        parent,
        pool={**parent.pool, source_keys[0]: pool[source_keys[0]]},
        next_regime_to_continuation=MappingProxyType({"child": projected}),
    ).owned_resources_row(
        combo_values=(), resources_domain=jnp.asarray([1.0, 4.0]), refine=refine
    )
    _assert_parent_euler_oracle(
        row=row,
        savings=demand.pieces.savings_nodes,
        slope=3.0,
        resources_domain=(1.0, 4.0),
    )


def _assert_zero_discount_read_failure(
    *,
    parent: _SolveOneCombo,
    demand: ParentResourcesDemand,
    source_params: FlatRegimeParams | FlatEdgeParams,
) -> None:
    """The real parent Euler node must retain a missing read after discounting."""
    slope_keys = tuple(name for name in source_params if name.endswith("__slope"))
    discount_keys = tuple(
        name for name in source_params if name.endswith("__discount_factor")
    )
    assert len(slope_keys) == len(discount_keys) == 1
    np.testing.assert_array_equal(
        parent.pool[slope_keys[0]], source_params[slope_keys[0]]
    )
    np.testing.assert_array_equal(
        parent.pool[discount_keys[0]], source_params[discount_keys[0]]
    )
    np.testing.assert_array_equal(parent.pool[discount_keys[0]], 1.0)
    pool = {
        **parent.pool,
        slope_keys[0]: jnp.asarray(3.125),
        discount_keys[0]: jnp.asarray(0.0),
    }
    prepared, valid = demand.prepare(
        pool=pool, savings_nodes=demand.pieces.savings_nodes
    )
    assert bool(valid)
    bank = parent.next_regime_to_continuation["child"]
    assert isinstance(bank, ProjectedEGMCarry)
    assert bool((prepared[1] >= bank.domain_lower) & (prepared[1] <= bank.domain_upper))
    assert not bool(jnp.any(prepared[1] == bank.grid))
    discount = tuple(parent.pieces.build_W_kwargs(pool).values())
    assert len(discount) == 1
    np.testing.assert_array_equal(discount[0], 0.0)
    assert isinstance(discount[0], jax.Array)
    node = _get_compute_node(
        pieces=parent.pieces,
        combo_pool=pool,
        discount_factor=discount[0],
        utility_of_action=BoundUtilityOfAction(
            utility_func=parent.pieces.utility_func,
            action_name=parent.pieces.action_name,
            bound=pool,
        ),
        next_regime_to_continuation=parent.next_regime_to_continuation,
        dtype=parent.state_grid.dtype,
        stochastic_node_width=None,
        resolved_process_grids=MappingProxyType({}),
    )(parent.pieces.savings_nodes[1])
    assert isinstance(node, BoundedResult)
    with pytest.raises(InvalidValueFunctionError):
        validate_query_input_status(
            statuses=node.query_status,
            authorities=(("child", 1, demand.spec.authority_at(period=1)),),
        )
