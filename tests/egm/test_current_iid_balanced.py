"""Connected private balanced native production and guarded parent reads.

The genuine canonical producer and parent builder supply acceptance economics.
Literal adversarial rows below discriminate locator failures separately; they
are not supplied as successful producer outputs or substitute reference cores.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import ProjectedContinuationSpec
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_balanced import (
    BalancedProjectedEGMCarry,
    BalancedResourcesProgram,
    _mapping_path_key,
    _matches_catalog,
    _semantic_native_key,
    build_resources_catalog,
)
from _lcm.egm.current_iid_declarations import projected_producer_declarations
from _lcm.egm.current_iid_demands import ParentResourcesDemand
from _lcm.egm.current_iid_evaluation import CurrentResourcesEvaluator
from _lcm.egm.current_iid_native import (
    NativeCaseRCore,
    NativeCaseRInputs,
    NativeCaseROutput,
)
from _lcm.egm.current_iid_resource_query import (
    read_resources_candidate,
    read_resources_candidate_details,
)
from _lcm.egm.interp import materialize_hermite_shadow
from _lcm.egm.projected_child import ProjectedChildReader
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.egm.step import _EGMStep
from _lcm.egm.upper_envelope import get_upper_envelope
from _lcm.egm.upper_envelope._exact_affine.hermite_critical import (
    hermite_critical_bernstein_signs,
)
from _lcm.execution.core_program import (
    CoreExecutionRequirements,
    _value_read_argument_leaf,
    materialize_core_program,
)
from _lcm.regime_building.processing import _SolutionBuild
from _lcm.solution.current_iid_case_r import (
    CaseRArguments,
    case_r_core_programs,
    install_balanced_case_r,
    native_case_r_template,
)
from _lcm.solution.dcegm import _BoundDCEGM, _DCEGMArgumentBuilder, _DCEGMPeriodKernel
from _lcm.solution.feasibility import BoundedResult
from _lcm.typing import EconFunctionKwargs, PytreeValue, ShapeDtypePytree
from lcm.solvers import ExactEnvelope
from lcm.typing import Float1D, ReferenceName, RegimeName, ScalarBool, ScalarFloat
from tests.egm.test_current_iid_parent_demands import (
    _array_pool,
    _context,
    _current_resources,
    _model,
    _ordinary_kernels,
    _record_original_contexts,
)
from tests.egm.test_current_iid_projection_domain import _reference
from tests.solution._nbegm_direct_oracle import OracleContext


@dataclass(frozen=True, kw_only=True)
class _BalancedFixture:
    produce: Callable[[EconFunctionKwargs], NativeCaseROutput]
    pool: Mapping[ReferenceName, jax.Array]
    spec: ProjectedContinuationSpec
    parent: _DCEGMPeriodKernel
    parent_context: OracleContext
    native_output: BoundedResult[tuple[PytreeValue, ...]]


def _native_balanced(monkeypatch: pytest.MonkeyPatch) -> _BalancedFixture:
    """Install actual core-derived schemas before building the genuine parent."""
    builds: dict[RegimeName, _SolutionBuild] = {}
    _record_original_contexts(monkeypatch=monkeypatch, captures=builds)
    model, child, child_context, _, parent_context = _ordinary_kernels()
    original_parent = builds["parent"]
    law_model = _model(current_child=True)
    plan = law_model._regimes["child"].solution.current_iid_plans[1]
    assert plan is not None
    assert isinstance(child.demand_step, _EGMStep)
    pieces = replace(child.demand_step.pieces, own_resources_func=_current_resources)
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
    core = NativeCaseRCore(
        pieces=pieces,
        plan=plan,
        coefficient=Fraction(1),
        refine=get_upper_envelope(
            solver=solver, n_refined=pieces.n_pad, publish_provenance=True
        ),
        owner_capacity=solver.envelope.max_runs,
        projection_capacity=solver.current_iid_projection_capacity,
    )
    base_arguments = child.core_programs()["replay"].argument_builder(
        _context(child_context)
    )
    pool = _array_pool(
        arguments={
            name: value
            for name, value in base_arguments.items()
            if name not in ("wealth", "next_regime_to_continuation")
        }
    )
    liquid = child_context["state_action_space"].states["wealth"]
    native_arguments: NativeCaseRInputs = {
        "liquid_grid": liquid,
        "pool": pool,
        "next_regime_to_continuation": _carry_arguments(arguments=base_arguments),
        "law_arguments": {**pool, "wealth": liquid[0]},
        "group_params": {name: {} for name, _, _ in plan.bindings},
        "node_operands": plan.node_operands,
        "indices": plan.indices,
        "savings_point_width": None,
        "stochastic_node_width": None,
        "envelope_cell_width": 1,
    }
    template = native_case_r_template(
        spec=spec,
        period=1,
        liquid_shape=liquid.shape,
        resources_capacity=pieces.n_pad,
        child_slots=0,
        current_nodes=plan.indices.shape[0],
    )
    core, spec, template = install_balanced_case_r(
        core=core,
        spec=spec,
        period=1,
        template=template,
        native_arguments=native_arguments,
    )
    # This is the genuine parent builder with the very same immutable spec,
    # rather than a test-built reader or a replacement successful KernelOutput.
    parent_kernels = original_parent.solver.build_period_kernels(
        context=replace(
            original_parent.context, projected_continuation_declarations={"child": spec}
        )
    )
    parent_kernels = original_parent.solver.declare_continuation_reads(
        kernels=parent_kernels,
        context=replace(
            original_parent.context,
            continuation_specs=MappingProxyType({"child": spec}),
        ),
    )
    parent = parent_kernels.period_kernels[0]
    assert isinstance(parent, _DCEGMPeriodKernel)
    assert isinstance(parent.demand_step, _EGMStep)
    demand = ParentResourcesDemand(
        source="parent",
        period=0,
        target="child",
        spec=spec,
        pieces=parent.demand_step.pieces,
        transition_targets=("child",),
    )
    core = replace(core, parent_demands=(demand,))
    arguments = CaseRArguments(
        base=_DCEGMArgumentBuilder(
            regime_name="child",
            stateful_targets=frozenset({"done"}),
            transition_target_names=("done",),
        ),
        plan=plan,
        parent_demands=(demand,),
    )
    graphs = case_r_core_programs(
        core=core,
        arguments=arguments,
        template=template,
        requirements=CoreExecutionRequirements(),
        child_slots=0,
    )
    materialized = materialize_core_program(
        program=graphs["main"], context=_context(child_context)
    )
    native_output = materialized.function(**materialized.arguments)
    assert isinstance(native_output, BoundedResult)
    assert isinstance(native_output.output, tuple)
    assert isinstance(native_output.output[0], jax.Array)
    demand_pools, demand_savings = _parent_operands(arguments=materialized.arguments)

    def produce(bound_pool: EconFunctionKwargs) -> NativeCaseROutput:
        return core(
            liquid_grid=liquid,
            pool=bound_pool,
            next_regime_to_continuation=native_arguments["next_regime_to_continuation"],
            law_arguments={**bound_pool, "wealth": liquid[0]},
            group_params=native_arguments["group_params"],
            node_operands=plan.node_operands,
            indices=plan.indices,
            savings_point_width=None,
            stochastic_node_width=None,
            envelope_cell_width=1,
            demand_pools=demand_pools,
            demand_savings=demand_savings,
        )

    return _BalancedFixture(
        produce=produce,
        pool=pool,
        spec=spec,
        parent=parent,
        parent_context=parent_context,
        native_output=native_output,
    )


def _carry_arguments(
    *, arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
) -> MappingProxyType[RegimeName, EGMCarry | ProjectedEGMCarry]:
    """Recover this fixture's genuine future carries without reconstructing them."""
    supplied = arguments["next_regime_to_continuation"]
    assert isinstance(supplied, Mapping)
    result: dict[RegimeName, EGMCarry | ProjectedEGMCarry] = {}
    for name, carry in supplied.items():
        assert isinstance(carry, EGMCarry | ProjectedEGMCarry)
        result[name] = carry
    return MappingProxyType(result)


def _parent_operands(
    *, arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
) -> tuple[tuple[EconFunctionKwargs, ...], tuple[Float1D, ...]]:
    """Keep the actual materialized parent pool and saving-array identity."""
    pools = arguments["_lcm_parent_demand_pools"]
    savings = arguments["_lcm_parent_demand_savings"]
    assert isinstance(pools, tuple)
    assert isinstance(savings, tuple)
    assert len(pools) == len(savings) == 1
    assert isinstance(pools[0], Mapping)
    assert isinstance(savings[0], jax.Array)
    return (_array_pool(arguments=pools[0]),), (savings[0],)


def _coverage_missing(
    result: BoundedResult[tuple[ScalarFloat, ScalarFloat]],
) -> ScalarBool:
    """Read the mandatory integrity slot, refusing an absent status record."""
    assert result.query_status is not None
    return result.query_status[0].coverage_missing


def _identity(savings: ScalarFloat) -> ScalarFloat:
    return savings


def _reader(
    *, fixture: _BalancedFixture, carry: BalancedProjectedEGMCarry
) -> ProjectedChildReader:
    return ProjectedChildReader.bind(
        spec=fixture.spec, target="child", period=1, carry=carry, next_liquid=_identity
    )


def _guard_program(fixture: _BalancedFixture) -> BalancedResourcesProgram:
    program = fixture.spec.date_at(period=1).guard_program
    assert isinstance(program, BalancedResourcesProgram)
    return program


def _smooth_query(carry: BalancedProjectedEGMCarry) -> tuple[int, ScalarFloat]:
    links = np.flatnonzero(np.asarray(carry.link_valid))
    assert links.size > 0
    index = int(
        links[
            np.argmax(np.asarray(carry.grid)[links + 1] - np.asarray(carry.grid)[links])
        ]
    )
    return index, carry.grid[index] + (
        carry.grid[index + 1] - carry.grid[index]
    ) * 0.375


def _run_parent(
    *,
    fixture: _BalancedFixture,
    carry: BalancedProjectedEGMCarry,
    zero_discount: bool = False,
    landing: ScalarFloat | None = None,
) -> BoundedResult[PytreeValue]:
    context = replace(
        _context(fixture.parent_context),
        next_regime_to_continuation=MappingProxyType({"child": carry}),
    )
    program = materialize_core_program(
        program=fixture.parent.core_programs()["replay"], context=context
    )
    arguments = dict(program.arguments)
    if zero_discount:
        for name, value in arguments.items():
            if name.endswith("discount_factor"):
                assert isinstance(value, jax.Array)
                arguments[name] = jnp.zeros_like(value)
    if landing is not None:
        # The genuine dated parent law is slope*savings+0.5 at period/age zero;
        # select its floor landing through its actual runtime slope operand.
        for name in arguments:
            if name.endswith("slope"):
                arguments[name] = 2.0 * (landing - 0.5)
    result = program.function(**arguments)
    assert isinstance(result, BoundedResult)
    return result


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_balanced_native_producer_reaches_actual_guarded_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real parent reads give the Hermite level and the linear economic marginal.

    On a smooth link the level is the cubic Hermite interpolant of the stored
    values and value slopes; the economic marginal interpolates linearly.
    """
    fixture = _native_balanced(monkeypatch)
    output = fixture.produce(fixture.pool)
    row = output.projected
    assert isinstance(row, BalancedProjectedEGMCarry)
    assert bool(row.valid)
    # The actual main native graph retains the guard even without replay rows.
    assert isinstance(fixture.native_output.output[1], BalancedProjectedEGMCarry)
    assert fixture.native_output.output[2] is None
    index, query = _smooth_query(row)
    result = _reader(fixture=fixture, carry=row)(query)
    assert not bool(_coverage_missing(result))
    t, width = 0.375, row.grid[index + 1] - row.grid[index]
    expected_level = (
        (2 * t**3 - 3 * t**2 + 1) * row.value[index]
        + (t**3 - 2 * t**2 + t) * width * row.value_slope[index]
        + (-2 * t**3 + 3 * t**2) * row.value[index + 1]
        + (t**3 - t**2) * width * row.value_slope[index + 1]
    )
    expected_marginal = row.economic_marginal[index] + 0.375 * (
        row.economic_marginal[index + 1] - row.economic_marginal[index]
    )
    np.testing.assert_allclose(result.output, (expected_level, expected_marginal))
    pwa_slope = (row.value[index + 1] - row.value[index]) / (
        row.grid[index + 1] - row.grid[index]
    )
    assert not bool(jnp.array_equal(expected_marginal, pwa_slope))
    actual_parent = _run_parent(fixture=fixture, carry=row)
    assert actual_parent.query_status is not None
    assert actual_parent.query_status
    assert all(
        not bool(status.coverage_missing) for status in actual_parent.query_status
    )
    # Every guard leaf participates in the same declared acquisition vocabulary.
    assert any(path[0] == "guard" for path in row.leaves())
    declared = {
        read.target.leaf_path
        for read in fixture.parent.core_programs()["replay"].requirements.value_reads
    }
    assert {path for path in row.leaves() if path[0] == "guard"} <= declared
    # Resolve genuine declared consumer addresses through the production resolver.
    # Semantic publication names must still select each original numeric object.
    materialized = materialize_core_program(
        program=fixture.parent.core_programs()["replay"],
        context=replace(
            _context(fixture.parent_context),
            next_regime_to_continuation=MappingProxyType({"child": row}),
        ),
    )
    leaves = row.leaves()
    for read in materialized.requirements.value_reads:
        if read.target.regime != "child":
            continue
        expected = leaves[read.target.leaf_path]
        actual = _value_read_argument_leaf(program=materialized, read=read)
        assert actual is expected
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
    flat, structure = jax.tree.flatten(row)
    restored = jax.tree.unflatten(structure, flat)
    assert isinstance(restored, BalancedProjectedEGMCarry)
    assert tuple(restored.leaves()) == tuple(leaves)
    assert all(restored.leaves()[path] is leaf for path, leaf in leaves.items())
    assert jax.tree.structure(row) == jax.tree.structure(
        fixture.spec.template_at(period=1)
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_changed_traced_parameters_rebuild_original_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _native_balanced(monkeypatch)
    original = fixture.produce(fixture.pool).projected
    assert isinstance(original, BalancedProjectedEGMCarry)
    changed = {
        name: (value * 0.75 if name.endswith("discount_factor") else value)
        for name, value in fixture.pool.items()
    }
    rebuilt = jax.jit(fixture.produce)(changed).projected
    assert isinstance(rebuilt, BalancedProjectedEGMCarry)
    assert bool(rebuilt.valid)
    assert not bool(
        jnp.array_equal(
            rebuilt.guard.row.envelope.value, original.guard.row.envelope.value
        )
    )
    assert not bool(
        jnp.array_equal(
            rebuilt.guard.row.discount_factor, original.guard.row.discount_factor
        )
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_owner_mismatch_is_flagged_ex_post_not_at_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read interpolates stored channels; the guard program checks ex post."""
    fixture = _native_balanced(monkeypatch)
    row = fixture.produce(fixture.pool).projected
    assert isinstance(row, BalancedProjectedEGMCarry)
    _, query = _smooth_query(row)
    catalog = row.guard.catalog
    corrupted = replace(
        row,
        guard=replace(
            row.guard,
            catalog=replace(
                catalog,
                branch=1 - catalog.branch,
                exact_owner=catalog.exact_owner + 1000,
            ),
        ),
    )
    np.testing.assert_array_equal(corrupted.value, row.value)
    assert not bool(_coverage_missing(_reader(fixture=fixture, carry=corrupted)(query)))
    program = _guard_program(fixture)
    assert bool(program.validate(carry=row, query=query))
    assert not bool(program.validate(carry=corrupted, query=query))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_hole_fails_at_read_and_invalid_source_fails_ex_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored hole refuses the read; source corruption is an ex-post finding."""
    fixture = _native_balanced(monkeypatch)
    row = fixture.produce(fixture.pool).projected
    assert isinstance(row, BalancedProjectedEGMCarry)
    index, query = _smooth_query(row)
    hole = replace(row, link_valid=row.link_valid.at[index].set(False))
    assert bool(_coverage_missing(_reader(fixture=fixture, carry=hole)(query)))
    parent = _run_parent(fixture=fixture, carry=hole, zero_discount=True, landing=query)
    assert parent.query_status is not None
    assert any(bool(status.coverage_missing) for status in parent.query_status)
    poisoned = replace(
        row,
        guard=replace(
            row.guard,
            row_payload=tuple(
                jax.tree.leaves(
                    row.guard.row._replace(
                        marginal=jnp.full_like(row.guard.row.marginal, jnp.inf)
                    )
                )
            ),
        ),
    )
    program = _guard_program(fixture)
    assert not bool(program.validate(carry=poisoned, query=query))
    bad_law = replace(
        row, guard=replace(row.guard, law_valid=jnp.zeros((), dtype=jnp.bool_))
    )
    assert not bool(program.validate(carry=bad_law, query=row.grid[0]))


def _log_row(*, gap: float) -> tuple[CurrentResourcesEvaluator, ScalarFloat]:
    """Literal adversarial reader row; never a successful native-producer fixture."""
    reference = _reference(liquid_grid=jnp.asarray([1.0, 2.0]), capacity=128)
    row = reference.evaluator.row
    grid = row.envelope.grid
    center = jnp.asarray(9.0 / 8.0, dtype=grid.dtype)
    value = grid - center - gap
    envelope = row.envelope._replace(
        value=value, savings=jnp.full_like(grid, 0.25), policy=grid - 0.25
    )
    row = row._replace(
        envelope=envelope,
        marginal=jnp.ones_like(grid),
        borrowing_limit=center - 1.0,
    )
    evaluator = replace(reference.evaluator, row=row, utility_of_action=jnp.log)
    return evaluator, center


@pytest.mark.coverage(backends=("cpu",), precisions="both")
# A binary gap keeps the tangent representable: a rounded decimal moves it off center.
@pytest.mark.parametrize("gap", [0.0, 2.0**-10])
def test_hidden_corner_island_is_linked_and_flagged_ex_post(gap: float) -> None:
    """Ordinary links miss an island between probes; the ex-post check finds it."""
    evaluator, center = _log_row(gap=gap)
    probes = jnp.asarray([1.0, 1.25, 1.5, 1.75, 2.0], dtype=center.dtype)
    selected = jax.vmap(
        lambda resources: read_resources_candidate(
            row=evaluator.row,
            resources=resources,
            utility_of_action=evaluator.utility_of_action,
            owner_capacity=1,
        )
    )(probes)
    assert bool(jnp.all(~selected.corner))
    at_center = read_resources_candidate(
        row=evaluator.row,
        resources=center,
        utility_of_action=evaluator.utility_of_action,
        owner_capacity=1,
    )
    if gap > 0:
        assert bool(at_center.corner)
    catalog = build_resources_catalog(
        evaluator=evaluator,
        scale=jnp.ones_like(center),
        required=jnp.empty((0,), dtype=center.dtype),
        capacity=128,
    )
    touches = (
        (catalog.points[:-1] <= center)
        & (center <= catalog.points[1:])
        & (jnp.arange(127) < catalog.length - 1)
    )
    # Both ends of the covering cell select the same interior owner.
    assert bool(jnp.any(touches & catalog.smooth))
    selected, legal, piece = read_resources_candidate_details(
        row=evaluator.row,
        resources=center,
        utility_of_action=evaluator.utility_of_action,
        owner_capacity=1,
    )
    matched = _matches_catalog(
        catalog=catalog,
        row=evaluator.row,
        resources=center,
        owner=selected.native_owner,
        side=selected.side_id,
        corner=selected.corner,
        feasible=selected.feasible,
        legal=legal,
        native_piece=piece,
    )
    assert bool(matched) is (gap == 0)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_log_authentication_reads_actual_scale_and_refuses_hidden_action_terms() -> (
    None
):
    """The full unbound graph proves the class before dynamic coefficient reads."""
    from _lcm.egm.current_iid_log_utility import LogUtilityProgram  # noqa: PLC0415
    from lcm.exceptions import UnsupportedOperationError  # noqa: PLC0415

    def utility(
        *, consumption: ScalarFloat, scale: ScalarFloat, shift: ScalarFloat
    ) -> ScalarFloat:
        return scale * jnp.log(consumption) + shift

    abstract = {
        name: jax.ShapeDtypeStruct((), jnp.asarray(1.0).dtype)
        for name in ("consumption", "scale", "shift")
    }
    program = LogUtilityProgram.authenticate(
        function=utility, action_name="consumption", abstract_pool=abstract
    )

    def coefficients(
        scale: ScalarFloat,
    ) -> tuple[ScalarFloat, ScalarFloat, ScalarBool]:
        return program.coefficients(
            pool={"scale": scale, "shift": jnp.asarray(0.25)}, dtype=scale.dtype
        )

    scale, shift, valid = jax.jit(coefficients)(jnp.asarray(2.0))
    np.testing.assert_array_equal(scale, jnp.asarray(2.0))
    np.testing.assert_array_equal(shift, jnp.asarray(0.25))
    assert bool(valid)
    assert not bool(coefficients(jnp.asarray(-1.0))[2])

    def invalid(*, consumption: ScalarFloat) -> ScalarFloat:
        return jnp.log(consumption) + jax.lax.stop_gradient(consumption * consumption)

    with pytest.raises(UnsupportedOperationError, match="action-dependent"):
        LogUtilityProgram.authenticate(
            function=invalid,
            action_name="consumption",
            abstract_pool={"consumption": abstract["consumption"]},
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_flattened_mapping_path_keeps_registered_insertion_order() -> None:
    """Ordinal keys identify registered mapping children, never sorted aliases."""
    first = jnp.asarray(2.0)
    second = jnp.asarray(3.0)
    mapping = MappingProxyType({"zeta": first, "alpha": second})
    for index, name in enumerate(("zeta", "alpha")):
        key = _mapping_path_key(mapping=mapping, index=index)
        assert key == jax.tree_util.DictKey(name)
        assert mapping[key.key] is (first, second)[index]
    for invalid in (-1, len(mapping), True):
        with pytest.raises(TypeError, match="index"):
            _mapping_path_key(mapping=mapping, index=invalid)
    with pytest.raises(TypeError, match="string-key"):
        _mapping_path_key(mapping=MappingProxyType({1: first}), index=0)
    with pytest.raises(TypeError, match="registered mapping"):
        _semantic_native_key(node=first, key=jax.tree_util.FlattenedIndexKey(0))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_exact_log_tangent_fixture_has_native_zero_endpoint_controls() -> None:
    """Literal dyadic data make R(m)=m-9/8, including its exact endpoint zero."""
    evaluator, center = _log_row(gap=0.0)
    row = evaluator.row
    grid = row.envelope.grid
    h, a, b, applicable = materialize_hermite_shadow(
        xp_lower=grid[0],
        xp_upper=grid[1],
        fp_lower=row.envelope.value[0],
        fp_upper=row.envelope.value[1],
        slope_lower=row.marginal[0],
        slope_upper=row.marginal[1],
    )
    assert bool(applicable)
    np.testing.assert_array_equal((h, a, b), (1.0, 0.0, 0.0))
    np.testing.assert_array_equal(row.envelope.value, (-0.125, 0.875))
    np.testing.assert_array_equal(
        evaluator.utility_of_action(center - row.borrowing_limit), 0.0
    )
    for lower, upper, expected in (
        (grid[0], center, (-1, -1, -1, 0)),
        (center, grid[1], (0, 1, 1, 1)),
    ):
        signs, status = hermite_critical_bernstein_signs(
            x0=grid[0],
            h=h,
            v0=row.envelope.value[0],
            v1=row.envelope.value[1],
            a=a,
            b=b,
            L=row.borrowing_limit,
            A=jnp.ones_like(center),
            lower=lower,
            upper=upper,
        )
        np.testing.assert_array_equal(status, 0)
        np.testing.assert_array_equal(signs, expected)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_balanced_guard_has_closed_detached_artifact_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot the genuine projection and restore every original native row leaf."""
    from lcm._solver_api.authority import (  # noqa: PLC0415
        _snapshot_artifact_template_once,
    )

    fixture = _native_balanced(monkeypatch)
    original = fixture.produce(fixture.pool).projected
    assert isinstance(original, BalancedProjectedEGMCarry)
    snapshot, _ = _snapshot_artifact_template_once(
        template=original, payload_runtime_type=BalancedProjectedEGMCarry
    )
    detached = snapshot.payload
    assert isinstance(detached, BalancedProjectedEGMCarry)
    assert jax.tree.structure(detached.guard.row) == jax.tree.structure(
        original.guard.row
    )
    for expected, actual in zip(
        jax.tree.leaves(original.guard.row),
        jax.tree.leaves(detached.guard.row),
        strict=True,
    ):
        assert actual is not expected
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    assert tuple(detached.guard.pool) == tuple(original.guard.pool)
    assert tuple(detached.guard.physical) == tuple(original.guard.physical)
    assert tuple(detached.leaves()) == tuple(original.leaves())
