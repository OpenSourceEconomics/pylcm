"""Original resources rows produce actions under the same addressed physical frame."""

import pickle
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_replay import (
    CASE_R_REPLAY_PROGRAM,
    CaseRReplayInput,
    CaseRReplayOutput,
    CaseRReplayPeriod,
    CaseRReplaySpec,
    _unflatten_case_r_replay_output,
)
from _lcm.egm.current_iid_resources_artifact import (
    resources_row_leaf_specs,
    resources_row_payload,
    resources_row_reads,
    restore_resources_row,
)
from _lcm.egm.query_data import QueryNodeKind
from _lcm.egm.step_core import OwnedResourcesRow
from _lcm.egm.upper_envelope.segment_envelope import OwnedEnvelope
from _lcm.engine import Regime
from _lcm.params.edges import regime_kernel_params
from _lcm.shock_overrides import CurrentIIDOverride
from _lcm.simulation.case_r_programs import (
    _case_r_output_roles,
    declare_case_r_replay_programs,
)
from _lcm.simulation.case_r_replay import (
    case_r_arguments,
    case_r_profile_output,
    case_r_status,
)
from _lcm.simulation.chunk_admission import prepare_simulation_chunks
from _lcm.simulation.chunk_profiles import (
    _period_copy_reservation,
    profile_simulation_chunk,
)
from _lcm.simulation.compile import bind_simulation_runtime
from _lcm.simulation.current_iid import prepare_current_iid_keys
from _lcm.simulation.current_iid_overrides import CurrentIIDOverrides
from _lcm.simulation.period_inputs import unit_value_reads
from _lcm.simulation.program_types import SUBJECT_WIDTH_KEYWORD
from _lcm.simulation.random import create_simulation_key
from _lcm.simulation.replay_inputs import replay_payload_reads
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.simulate import simulate
from _lcm.simulation.value_reads import PeriodSimulationReads
from _lcm.solution.current_iid_case_r import CASE_R_RESOURCES_REPLAY
from _lcm.solution.feasibility import (
    BoundInputStatus,
    ContinuationReadCounts,
    QueryInputStatus,
)
from _lcm.solution.model_authority import _authority_from_template
from _lcm.utils.logging import get_logger
from lcm import ExecutionConfig, InitialNodes, LinSpacedGrid, Model, Periods
from lcm import Regime as UserRegime
from lcm._solver_api.authority import _canonicalize_artifact_payload
from lcm.exceptions import ExecutionPlanningError, InvalidSimulationInputError
from lcm.result import SimulationResult
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactChannel, PersistencePolicy
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, RegimeName
from tests.conftest import DECIMAL_PRECISION
from tests.simulation.initial_conditions.test_public_api import (
    _tagged_current_entry_model,
)
from tests.simulation.test_budget_lifecycle import _LifecycleRegimeId


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_r_output_preserves_roles_and_abstract_reconstruction() -> None:
    """Role and abstract trees reuse the numeric schema without checked construction."""
    floating = jnp.asarray(0.0)
    integer = jnp.zeros((), dtype=jnp.int32)
    boolean = jnp.zeros((), dtype=jnp.bool_)
    numeric = CaseRReplayOutput(
        actions={"consumption": floating},
        value=floating,
        physical={"eps": floating},
        law_invalid=boolean,
        invalid=boolean,
        feasible=boolean,
        native_owner=integer,
        side_id=integer,
        corner=boolean,
    )
    abstract = jax.eval_shape(lambda: numeric)
    assert jax.tree.structure(abstract) == jax.tree.structure(numeric)
    assert all(
        isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(abstract)
    )
    profile = case_r_profile_output(output=abstract)
    assert profile.actions["consumption"] is abstract.actions["consumption"]
    assert profile.physical["eps"] is abstract.physical["eps"]
    for name in (
        "value",
        "law_invalid",
        "invalid",
        "feasible",
        "native_owner",
        "side_id",
        "corner",
    ):
        assert getattr(profile, name) is getattr(abstract, name)
    with pytest.raises(ExecutionPlanningError, match="incompatible output schema"):
        case_r_profile_output(output=())
    restored = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), abstract)
    np.testing.assert_array_equal(restored.native_owner, 0)
    assert restored.invalid.dtype == jnp.bool_
    restore, (metadata, children) = numeric.__reduce__()
    retained = restore(metadata, children)
    assert retained.actions["consumption"] is numeric.actions["consumption"]
    assert retained.physical["eps"] is numeric.physical["eps"]
    with pytest.raises(TypeError, match="exact fields"):
        _unflatten_case_r_replay_output(metadata=None, children=())

    roles = _case_r_output_roles(action_name="consumption", physical_names=("eps",))
    assert jax.tree.structure(roles) == jax.tree.structure(numeric)
    expected = {
        "actions": "action",
        "value": "decision_value",
        "physical": "current_shock",
        "law_invalid": "current_iid_invalid",
        "invalid": "current_iid_invalid",
        "feasible": "action_feasible",
        "native_owner": "candidate_owner",
        "side_id": "candidate_side",
        "corner": "borrowing_corner",
    }
    for path, label in jax.tree_util.tree_flatten_with_path(roles)[0]:
        assert isinstance(path[0], jax.tree_util.GetAttrKey)
        assert label == expected[path[0].name]


def _resources(*, wealth: FloatND, eps: FloatND) -> FloatND:
    return wealth + eps


def _utility(*, consumption: FloatND, **_kwargs: FloatND) -> FloatND:
    return jnp.log(consumption)


def _row() -> OwnedResourcesRow:
    """An original owned flat-value row and its literal borrowing corner.

    The row saves 0.5, strictly above the 0.25 borrowing limit, so the row and
    the corner are distinct actions.
    """
    grid = jnp.asarray([1.0, 2.0])
    codes = jnp.zeros((2,), dtype=jnp.int32)
    return OwnedResourcesRow(
        envelope=OwnedEnvelope(
            grid=grid,
            policy=grid - 0.5,
            value=jnp.ones_like(grid),
            n_kept=jnp.asarray(2, dtype=jnp.int32),
            savings=jnp.full_like(grid, 0.5),
            owner_id=codes,
            side_id=codes,
            link_valid=jnp.ones((1,), dtype=jnp.bool_),
            source_left_index=codes,
            source_right_index=jnp.ones_like(codes),
        ),
        marginal=jnp.zeros_like(grid),
        node_kind=jnp.full((2,), QueryNodeKind.FINITE, dtype=jnp.int32),
        borrowing_limit=jnp.asarray(0.25),
        upper_savings=jnp.asarray(2.0),
        expected_floor_value=jnp.asarray(0.0),
        discount_factor=jnp.asarray(1.0),
        discounted_floor_value=jnp.asarray(0.0),
        resources_domain=grid,
        construction_invalid=jnp.asarray(False),  # noqa: FBT003
        floor_status=None,
        child_status=None,
        continuation_read_counts=None,
    )


def _declaration(
    *, regime: RegimeName = "alive", row: OwnedResourcesRow | None = None
) -> CaseRReplayPeriod:
    original = _row() if row is None else row
    return CaseRReplayPeriod(
        period=0,
        liquid_state="wealth",
        action_name="consumption",
        resources=_resources,
        utility=_utility,
        coefficient=Fraction(1),
        owner_capacity=1,
        row_reads=resources_row_reads(
            row=original,
            key=CASE_R_RESOURCES_REPLAY,
            period=0,
            regime=regime,
            core=CASE_R_REPLAY_PROGRAM,
        ),
        row_tree=jax.tree.structure(original),
        row_leaf_specs=resources_row_leaf_specs(row=original),
    )


def _private_route() -> tuple[
    Model, MappingProxyType[RegimeName, Regime], CaseRReplayInput
]:
    """Declare private original-row replay while public admission stays closed."""
    model = _tagged_current_entry_model(positive_shock=True, enable_jit=True)
    row = _row()
    declaration = _declaration(row=row)
    original = model._regimes["alive"]
    private = replace(
        original,
        simulation=replace(
            original.simulation,
            case_r_replay_spec=CaseRReplaySpec(
                regime="alive",
                artifact_key=CASE_R_RESOURCES_REPLAY,
                periods={0: declaration},
            ),
        ),
    )
    declared = declare_case_r_replay_programs(regime=private)
    regimes = bind_simulation_runtime(
        regimes=MappingProxyType({"alive": declared}),
        execution=model._execution,
        enable_jit=True,
    )
    return (
        model,
        regimes,
        CaseRReplayInput(
            regime="alive", period=0, key=CASE_R_RESOURCES_REPLAY, row=row
        ),
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("supplied", [False, True])
def test_case_r_action_replay_retains_the_actual_physical_frame(
    *, supplied: bool
) -> None:
    """A continuous row action follows actual draws, without an action-grid read."""
    model, regimes, replay = _private_route()
    regime = regimes["alive"]
    plan = regime.simulation.current_iid_draw_plan
    assert plan is not None
    keys = prepare_current_iid_keys(
        plan=plan,
        key=create_simulation_key(seed=17),
        period=0,
        subject_slice=slice(0, 2),
        original_n_subjects=2,
        memory=None,
    )
    overrides = (
        {
            "eps": CurrentIIDOverride(
                kind="physical",
                values=jnp.asarray([0.3, 0.7]),
                active=jnp.ones((2,), dtype=jnp.bool_),
            )
        }
        if supplied
        else {}
    )
    program = regime.simulation.programs.forward_decision[0]
    reads = unit_value_reads(
        regime=regime,
        name="alive",
        period=0,
        values={},
        flags={},
        policy=None,
        reader=None,
        current_iid_replay_input=replay,
    )
    assert reads == program.requirements.value_reads
    executor = regime.simulation.programs.executor
    assert isinstance(executor, SimulationRuntime)
    owner = PeriodSimulationReads(
        period=0,
        devices=executor.subject_devices,
        reads_by_unit={"alive": reads},
        release_enabled=False,
    )
    spec = regime.simulation.case_r_replay_spec
    assert spec is not None
    placed = replay.acquire(core=program.name, owner=owner, spec=spec)
    output = cast(
        "CaseRReplayOutput",
        program.function(
            **case_r_arguments(
                regime=regime,
                replay=replay,
                row=placed,
                states={"wealth": jnp.ones((2,))},
                keys=keys,
                overrides=overrides,
                flat_params=model._process_params({}),
                period=0,
                period_value=jnp.asarray(0, dtype=jnp.int32),
                age=jnp.asarray(0, dtype=jnp.int32),
            ),
            **{SUBJECT_WIDTH_KEYWORD: 1},
        ),
    )
    owner.commit(unit="alive", outputs=output)
    owner.finish()
    if supplied:
        np.testing.assert_array_almost_equal(
            output.physical["eps"], [0.3, 0.7], decimal=DECIMAL_PRECISION
        )
    np.testing.assert_array_almost_equal(
        output.actions["consumption"],
        output.physical["eps"] + 0.5,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(output.value, [1.0, 1.0])
    np.testing.assert_array_equal(output.native_owner, [0, 0])
    np.testing.assert_array_equal(output.corner, [False, False])
    assert not np.any(output.invalid)
    assert not np.any(output.law_invalid)
    assert np.all(output.feasible)


def test_case_r_replay_inputs_preserve_key_date_and_original_buffer_owners() -> None:
    """Addressed row storage is separate from NNBEGM's simulation-policy key."""
    row = _row()
    entry = CaseRReplayInput(
        regime="saving", period=0, key=CASE_R_RESOURCES_REPLAY, row=row
    )
    spec = CaseRReplaySpec(
        regime="saving",
        artifact_key=CASE_R_RESOURCES_REPLAY,
        periods={0: _declaration(regime="saving", row=row)},
    )
    assert entry.require(spec=spec, period=0) is row
    assert {id(leaf) for leaf in jax.tree.leaves(entry)} == {
        id(leaf) for leaf in jax.tree.leaves(row)
    }
    reads = entry.reads(core=CASE_R_REPLAY_PROGRAM)
    assert all(read.target.artifact_key == CASE_R_RESOURCES_REPLAY for read in reads)
    assert all(read.source.argument == "row" for read in reads)
    with pytest.raises(InvalidSimulationInputError, match="producer"):
        entry.require(spec=spec, period=1)


def test_case_r_integrity_is_checked_before_action_feasibility() -> None:
    bad_law, bad_row, empty = case_r_status(
        law_invalid=jnp.asarray([False, True]),
        invalid=jnp.asarray([True, True]),
        feasible=jnp.asarray([False, False]),
        in_regime=jnp.asarray([True, False]),
    )
    assert not bool(bad_law)
    assert bool(bad_row)
    assert bool(empty)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_case_r_chunk_profile_retains_original_row_frame_action_and_status() -> None:
    """A compiled chunk inventories original rows and action-valued outputs."""
    model, regimes, replay = _private_route()
    regime = regimes["alive"]
    runtime = regime.simulation.programs.executor
    assert isinstance(runtime, SimulationRuntime)
    flat = model._process_params({})
    spaces = MappingProxyType(
        {
            name: item.solution.state_action_space(
                regime_params=regime_kernel_params(flat, regime_name=name)
            )
            for name, item in regimes.items()
        }
    )
    initial = {
        "wealth": jnp.ones((2,)),
        "age": jnp.zeros((2,)),
        "regime_id": jnp.zeros((2,), dtype=jnp.int32),
    }
    rows = {0: {"alive": replay}}
    copies = _period_copy_reservation(
        regimes=regimes,
        values={},
        period=0,
        devices=runtime.subject_devices,
        current_iid_replay_inputs=rows,
    )
    # These canonical originals already live on the executing device; the
    # retained footprint owns them, so no independent aligned copy is charged.
    assert copies == {}
    profile = profile_simulation_chunk(
        runtime=runtime,
        regimes=regimes,
        flat_params=flat,
        base_spaces=spaces,
        values={},
        ages=model._time,
        initial_conditions=initial,
        regime_names_to_ids=model.regime_names_to_ids,
        n_subjects=2,
        population=2,
        original_population=2,
        widths={"subject": 1},
        independent_taste=False,
        log_level="off",
        current_iid_replay_inputs=rows,
        max_compilation_workers=1,
    )
    assert profile.n_subjects == 2
    core = regime.simulation.programs.forward_decision[0]
    assert core.name == CASE_R_REPLAY_PROGRAM
    assert all(
        read.target.artifact_key == CASE_R_RESOURCES_REPLAY
        for read in core.requirements.value_reads
    )
    assert all(value >= 0 for value in profile.fixed_reservation.values())
    assert profile.output_reservation


# The action remains a physical value throughout canonical downstream DAGs.
def _next_wealth(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _flow(*, consumption: ContinuousAction) -> FloatND:
    return consumption


def _terminal(*, wealth: ContinuousState) -> FloatND:
    return wealth


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_grid_policy_lookup_retains_the_resolved_solution_schema() -> None:
    """An authenticated ordinary solution still selects its literal grid optimum."""
    model = Model(
        regimes={
            "alive": UserRegime(
                actions={"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _flow},
            ),
            "done": UserRegime(functions={"utility": _terminal}),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
        state_transitions={"wealth": _terminal},
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_LifecycleRegimeId,
        enable_jit=True,
    )
    params = {"discount_factor": 0.0}
    solution = model.solve(params=params, log_level="off")
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states={"wealth": jnp.asarray([1.0, 2.0])},
    )
    assert lookup.actions is not None
    np.testing.assert_array_equal(lookup.actions["consumption"], [1.0, 1.0])


def _record_frame(*, consumption: ContinuousAction, eps: FloatND) -> FloatND:
    return consumption + 2 * eps


def _affordable(*, resources: FloatND, consumption: ContinuousAction) -> BoolND:
    return consumption <= resources


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("invalid_row", [False, True])
def test_case_r_admitted_chunk_keeps_frame_through_transition_and_report(
    *,
    supplied: bool,
    invalid_row: bool,
) -> None:
    """Private routing-unit actions reach next wealth and report DAGs under one frame.

    Explicit original-row and ordinary continuation fixtures test placement and
    downstream custody; they do not qualify a public native-producer solve.
    """
    model = Model(
        regimes={
            "alive": UserRegime(
                actions={"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={
                    "utility": _flow,
                    "resources": _resources,
                    "record_frame": _record_frame,
                },
                constraints={"affordable": _affordable},
                output_only_functions=("record_frame",),
                shocks=("entry",),
            ),
            "done": UserRegime(functions={"utility": _terminal}),
        },
        states={"wealth": LinSpacedGrid(start=0.5, stop=2, n_points=2)},
        state_transitions={"wealth": _next_wealth},
        shocks={
            "entry": ShockGroup(
                shocks={"eps": Uniform(start=0.25, stop=0.75)},
                nodes=UserNodes(
                    nodes=jnp.asarray([[0.0], [0.5], [1.0]]),
                    weights=jnp.asarray([0.25, 0.5, 0.25]),
                ),
            )
        },
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_LifecycleRegimeId,
        execution_config=ExecutionConfig(
            axis_widths={"subject": 1}, device_memory_bytes=2**30
        ),
        enable_jit=True,
    )
    row = _row()._replace(construction_invalid=jnp.asarray(invalid_row))
    spec = CaseRReplaySpec(
        regime="alive",
        artifact_key=CASE_R_RESOURCES_REPLAY,
        periods={0: _declaration(row=row)},
    )
    alive = model._regimes["alive"]
    declared = declare_case_r_replay_programs(
        regime=replace(
            alive,
            simulation=replace(alive.simulation, case_r_replay_spec=spec),
        )
    )
    regimes = bind_simulation_runtime(
        regimes=MappingProxyType({**model._regimes, "alive": declared}),
        execution=model._execution,
        enable_jit=True,
    )
    replay = CaseRReplayInput(
        regime="alive", period=0, key=CASE_R_RESOURCES_REPLAY, row=row
    )
    rows = {0: {"alive": replay}}
    flat = model._process_params({"discount_factor": 0.0})
    initial = {
        "wealth": jnp.ones((2,)),
        "age": jnp.zeros((2,)),
        "regime_id": jnp.zeros((2,), dtype=jnp.int32),
    }
    values = MappingProxyType(
        {
            0: MappingProxyType({"alive": jnp.ones((2,))}),
            1: MappingProxyType({"done": jnp.asarray([0.5, 2.0])}),
        }
    )
    overrides = CurrentIIDOverrides(
        entries={"eps": PhysicalValues(values=jnp.asarray([0.3, 0.7]))}
        if supplied
        else {},
        paths={},
        starts=np.zeros((2,), dtype=np.int32),
    )
    retained = measure_buffer_footprint(tree=(row, values, initial, flat))
    prepared = prepare_simulation_chunks(
        regimes=regimes,
        flat_params=flat,
        values=values,
        flags={},
        ages=model._time,
        initial_conditions=initial,
        regime_names_to_ids=model.regime_names_to_ids,
        original_population=2,
        retained_footprint=retained,
        independent_taste=False,
        log_level="off",
        current_iid_replay_inputs=rows,
        current_iid_override_schema=overrides.schema,
        max_compilation_workers=1,
    )

    def run() -> SimulationResult:
        return simulate(
            regimes=regimes,
            flat_params=flat,
            initial_conditions=initial,
            regime_names_to_ids=model.regime_names_to_ids,
            logger=get_logger(log_level="off"),
            period_to_regime_to_V_arr=values,
            ages=model._time,
            simulation_output_dtypes=model.simulation_output_dtypes,
            seed=17,
            current_iid_replay_inputs=rows,
            current_iid_overrides=overrides,
            retained_footprint=retained,
            prepared_chunks=prepared,
            subject_batch_size=prepared.plan.profile.n_subjects,
        )

    if invalid_row:
        with pytest.raises(ExecutionPlanningError, match="row integrity"):
            run()
        return
    result = run()
    record = result.raw_results["alive"][0]
    frame = record.shocks["eps"]
    if supplied:
        np.testing.assert_array_almost_equal(
            frame, [0.3, 0.7], decimal=DECIMAL_PRECISION
        )
    np.testing.assert_array_almost_equal(
        record.actions["consumption"], frame + 0.5, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][1].states["wealth"],
        [0.5, 0.5],
        decimal=DECIMAL_PRECISION,
    )
    table = result.to_dataframe(additional_targets=["record_frame"])
    observed = table.loc[table["period"] == 0, "record_frame"]
    np.testing.assert_array_almost_equal(
        observed, 0.5 + 3 * np.asarray(frame), decimal=DECIMAL_PRECISION
    )


def test_case_r_closed_artifact_restores_original_row_and_leaf_addresses() -> None:
    """Snapshot storage is closed while native buffers retain their original view."""
    row = _row()._replace(
        floor_status=BoundInputStatus(
            jnp.zeros((), dtype=jnp.bool_), jnp.zeros((), dtype=jnp.bool_)
        ),
        child_status=(
            QueryInputStatus(
                jnp.zeros((), dtype=jnp.bool_),
                jnp.zeros((), dtype=jnp.bool_),
                jnp.ones((), dtype=jnp.bool_),
            ),
        ),
        continuation_read_counts=ContinuationReadCounts(
            jnp.asarray(1, dtype=jnp.int32), jnp.asarray(2, dtype=jnp.int32)
        ),
    )
    payload = resources_row_payload(row=row)
    declaration = _declaration(row=row)
    assert type(payload) is tuple
    assert all(a is b for a, b in zip(payload, jax.tree.leaves(row), strict=True))
    authority = _authority_from_template(
        key=CASE_R_RESOURCES_REPLAY,
        channel=ArtifactChannel.REPLAY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_runtime_type=tuple,
        template=payload,
        applicable=True,
        required=True,
    )
    assert all(
        runtime_type is tuple
        for runtime_type in authority.container_runtime_types.values()
    )
    detached = authority.template
    assert type(detached) is tuple
    for original, copied in zip(payload, detached, strict=True):
        assert copied is not original
        np.testing.assert_array_equal(copied, original)
    checked = _canonicalize_artifact_payload(payload=payload, authority=authority)
    restored = restore_resources_row(
        payload=checked,
        tree=declaration.row_tree,
        leaf_specs=declaration.row_leaf_specs,
    )
    assert type(restored) is OwnedResourcesRow
    assert all(
        a is b
        for a, b in zip(
            jax.tree.leaves(restored), jax.tree.leaves(checked), strict=True
        )
    )
    saved = pickle.loads(pickle.dumps(payload))
    roundtrip = restore_resources_row(
        payload=saved, tree=declaration.row_tree, leaf_specs=declaration.row_leaf_specs
    )
    compiled = jax.jit(
        lambda values: restore_resources_row(
            payload=values,
            tree=declaration.row_tree,
            leaf_specs=declaration.row_leaf_specs,
        )
    )(payload)
    abstract = jax.eval_shape(lambda native: resources_row_payload(row=native), row)
    for source, pickled, actual, shape in zip(
        payload,
        jax.tree.leaves(roundtrip),
        jax.tree.leaves(compiled),
        abstract,
        strict=True,
    ):
        np.testing.assert_array_equal(pickled, source)
        np.testing.assert_array_equal(actual, source)
        assert shape.shape == source.shape
        assert shape.dtype == source.dtype
    tuple_reads = replay_payload_reads(
        payload=payload,
        key=CASE_R_RESOURCES_REPLAY,
        period=0,
        regime="alive",
        core=CASE_R_REPLAY_PROGRAM,
        argument="row",
    )
    for mapped, stored in zip(declaration.row_reads, tuple_reads, strict=True):
        assert mapped.source == stored.source
        assert mapped.target == stored.target
    abstract_row = jax.tree.unflatten(declaration.row_tree, declaration.row_leaf_specs)
    assert (
        resources_row_reads(
            row=abstract_row,
            key=CASE_R_RESOURCES_REPLAY,
            period=0,
            regime="alive",
            core=CASE_R_REPLAY_PROGRAM,
        )
        == declaration.row_reads
    )


def test_case_r_foreign_row_schema_refuses_before_placement() -> None:
    """A foreign leaf shape cannot borrow a matching source/date/key."""
    row = _row()
    declaration = _declaration(row=row)
    spec = CaseRReplaySpec(
        regime="alive", artifact_key=CASE_R_RESOURCES_REPLAY, periods={0: declaration}
    )
    wrong = row._replace(marginal=jnp.zeros((3,), dtype=row.marginal.dtype))
    entry = CaseRReplayInput(
        regime="alive", period=0, key=CASE_R_RESOURCES_REPLAY, row=wrong
    )
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices("cpu")[0],),
        reads_by_unit={"alive": declaration.row_reads},
        release_enabled=False,
    )
    with pytest.raises(InvalidSimulationInputError, match="model-owned row schema"):
        entry.acquire(spec=spec, core=CASE_R_REPLAY_PROGRAM, owner=owner)
    assert not owner._source_values
    with pytest.raises(InvalidSimulationInputError, match="leaf shape or dtype"):
        restore_resources_row(
            payload=resources_row_payload(row=wrong),
            tree=declaration.row_tree,
            leaf_specs=declaration.row_leaf_specs,
        )
    with pytest.raises(InvalidSimulationInputError, match="foreign row schema"):
        restore_resources_row(
            payload=resources_row_payload(row=row)[:-1],
            tree=declaration.row_tree,
            leaf_specs=declaration.row_leaf_specs,
        )
