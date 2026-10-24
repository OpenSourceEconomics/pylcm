"""Simulation value-copy ownership on actual ordered four-device CPU layouts."""

import dataclasses
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Never, Unpack, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.arrival_queries import ArrivalQueryArguments
from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.execution.core_program import ValueRead
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.scheduler import shares_a_buffer
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.simulation.exit_replay import _query_arguments
from _lcm.simulation.exits import OriginalExitOutput, ParticipatingExitOutput
from _lcm.simulation.replay_inputs import place_replay_payload, replay_payload_reads
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.value_reads import PeriodSimulationReads
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.typing import PytreeValue
from lcm import AgeGrid, CollectiveUtility, ExecutionConfig, Model
from lcm.solver_api import SIMULATION_POLICY, ResultRetention
from lcm.typing import DiscreteAction, FloatND
from tests.conftest import DECIMAL_PRECISION, ApplyValueTransferKwargs

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _TOPOLOGY_UNAVAILABLE = False
except RuntimeError:
    _TOPOLOGY_UNAVAILABLE = True

# This fixture defines categoricals and arrays at import, after topology setup.
from _lcm.simulation import chunk_profiles, value_reads
from _lcm.simulation.chunk_profiles import _period_copy_reservation
from _lcm.simulation.residency import (
    measure_buffer_footprint,
    require_transfer_headroom,
    union_buffer_footprints,
)
from lcm.exceptions import ExecutionPlanningError
from tests.conformance_solver import ReferenceReplayRoute, ReferenceSolver
from tests.regime_building.test_collective_regime_simulate import (
    _DISSOLUTION_PARAMS,
    DissolutionRegimeId,
    _make_dissolution_regimes,
    _married_dissolution_transition,
)
from tests.simulation.test_finite_policy_budget import _inputs as _finite_inputs
from tests.test_external_solver_conformance import _PARAMS, _model, _solve

pytestmark = pytest.mark.skipif(
    _TOPOLOGY_UNAVAILABLE, reason="Four-device topology requires an isolated process"
)


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("aliased", [False, True])
def test_finite_profile_reserves_addressed_policy_copies_once_on_ordered_submesh(
    *,
    discrete: bool,
    aliased: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real prepare/rank declarations share copies without collapsing artifact slots."""
    model, params, _ = _finite_inputs(discrete=discrete, budget=None)
    solution = model.solve(params=params, log_level="off")
    view = cast("OwnedSolutionView", solution._engine_view)
    source_device = jax.devices()[2]
    devices = (jax.devices()[3], jax.devices()[1])
    original_policy = view.simulation_policies[0]["alive"]
    policy = jax.tree.map(
        lambda leaf: jax.device_put(leaf, source_device), original_policy
    )
    if aliased:
        policy = dataclasses.replace(
            policy, candidate_value=policy.candidate_inner_action
        )
    values = jax.tree.map(lambda leaf: jax.device_put(leaf, source_device), view.values)
    regime = model._runtime_regimes_for_shape(compile_batch_size=2)["alive"]
    reads = (
        *regime.simulation.programs.policy_prepare[0].requirements.value_reads,
        *regime.simulation.programs.policy_rank[0].requirements.value_reads,
    )
    leaves = jax.tree.leaves(policy)
    policy_reads = replay_payload_reads(
        payload=policy,
        key=SIMULATION_POLICY,
        period=0,
        regime="alive",
        core="reference",
    )
    sources = {
        read.target: leaf for read, leaf in zip(policy_reads, leaves, strict=True)
    }
    sources.update(
        {
            read.target: values[read.target.period][read.target.regime]
            for read in reads
            if read.target.kind is ValueArtifactKind.REGIME_VALUE
        }
    )
    assert len(sources) == len({read.target for read in reads})
    logical_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in sources.values())
    with monkeypatch.context() as guard:

        def forbid(*_args: Never, **_kwargs: Never) -> Never:
            raise AssertionError("A copy reservation allocated a device buffer")

        guard.setattr(jax, "device_put", forbid)
        reservation = _period_copy_reservation(
            regimes={"alive": regime},
            values=values,
            period=0,
            devices=devices,
            policies={0: {"alive": policy}},
        )
    assert reservation == {
        source_device: logical_bytes,
        devices[0]: 2 * logical_bytes,
        devices[1]: 2 * logical_bytes,
    }
    owner = PeriodSimulationReads(
        period=0,
        devices=devices,
        reads_by_unit={"alive": reads},
        release_enabled=True,
    )
    actual_transfer = value_reads.apply_value_transfer
    copied_addresses = []

    def observe_copy(**call: Unpack[ApplyValueTransferKwargs]) -> jax.Array:
        copied_addresses.append(call["transfer"].target)
        return actual_transfer(**call)

    monkeypatch.setattr(value_reads, "apply_value_transfer", observe_copy)
    copies = {}
    for read in reads:
        source = sources[read.target]
        placed = owner.read(unit="alive", read=read, value=source)
        if read.target in copies:
            assert placed is copies[read.target]
        copies[read.target] = placed
        assert isinstance(placed.sharding, jax.NamedSharding)
        assert tuple(placed.sharding.mesh.devices.flat) == devices
        np.testing.assert_array_equal(placed, source)
    assert len(copied_addresses) == len(sources)
    assert len(set(copied_addresses)) == len(sources)
    if aliased:
        assert len({id(leaf) for leaf in sources.values()}) < len(sources)
    _close_finite_copy_owner(owner=owner, sources=sources, copies=copies)


def test_finite_profile_keeps_host_policy_scratch_outside_the_device_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy retained on the host stages its copy scratch in host RAM.

    The solve retains finite policies on the host CPU. On GPU that device is not a
    budgeted chunk device, so charging its source scratch made every budgeted
    finite replay refuse with "names an unbudgeted device". Device 2 stands in for
    the host on the four-device CPU layout.
    """
    model, params, _ = _finite_inputs(discrete=False, budget=None)
    solution = model.solve(params=params, log_level="off")
    view = cast("OwnedSolutionView", solution._engine_view)
    host = jax.devices()[2]
    devices = (jax.devices()[3], jax.devices()[1])
    policy = jax.tree.map(
        lambda leaf: jax.device_put(leaf, host), view.simulation_policies[0]["alive"]
    )
    values = jax.tree.map(lambda leaf: jax.device_put(leaf, host), view.values)
    regime = model._runtime_regimes_for_shape(compile_batch_size=2)["alive"]
    monkeypatch.setattr(
        chunk_profiles,
        "_host_outside_device_ceiling",
        lambda *, devices: host,  # noqa: ARG005
        raising=False,
    )
    reservation = _period_copy_reservation(
        regimes={"alive": regime},
        values=values,
        period=0,
        devices=devices,
        policies={0: {"alive": policy}},
    )
    assert set(reservation) == set(devices)


def _close_finite_copy_owner(
    *,
    owner: PeriodSimulationReads,
    sources: Mapping[ValueArtifactAddress, jax.Array],
    copies: Mapping[ValueArtifactAddress, jax.Array],
) -> None:
    """Compare real sources after the period releases every addressed destination."""
    expected_sources = {target: np.asarray(leaf) for target, leaf in sources.items()}
    owner.commit(unit="alive", outputs=())
    owner.finish()
    assert all(leaf.is_deleted() for leaf in copies.values())
    for target, source in sources.items():
        assert not source.is_deleted()
        np.testing.assert_array_equal(source, expected_sources[target])


def _read(*, unit: str) -> ValueRead:
    """Read one solved target value from the active regime's decision family."""
    return ValueRead(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="target"
        ),
        source=ValueConsumerAddress(
            source_period=0,
            source_regime=unit,
            core_key="argmax",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            path=("target",),
        ),
    )


def test_finite_policy_transfer_refuses_before_its_ordered_copy_allocates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual period owner admits retained policy plus destination and scratch."""
    model, params, _ = _finite_inputs(discrete=False, budget=None)
    solution = model.solve(params=params, log_level="off")
    policy = cast("OwnedSolutionView", solution._engine_view).simulation_policies[0][
        "alive"
    ]
    assert isinstance(policy, NNBEGMSimPolicy)
    source = jax.device_put(policy.candidate_inner_action, jax.devices()[2])
    regime = model._runtime_regimes_for_shape(compile_batch_size=2)["alive"]
    read = regime.simulation.programs.policy_prepare[0].requirements.value_reads[0]
    live = measure_buffer_footprint(tree=source)
    observed = []

    def before_transfer(
        *, transfer: ResolvedValueTransfer, live_values: tuple[jax.Array, ...]
    ) -> None:
        observed.append(transfer)
        destination = transfer.source_sharding.device_set
        require_transfer_headroom(
            live=union_buffer_footprints(
                footprints=(
                    live,
                    measure_buffer_footprint(tree=live_values),
                )
            ),
            destination_bytes=dict.fromkeys(destination, source.nbytes),
            scratch_bytes=dict.fromkeys(destination | source.devices(), source.nbytes),
            budget_bytes=2 * source.nbytes - 1,
            devices=tuple(jax.devices()),
        )

    def forbid_copy(**_call: Unpack[ApplyValueTransferKwargs]) -> jax.Array:
        raise AssertionError("The refused policy transfer allocated")

    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[3], jax.devices()[1]),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
        before_transfer=before_transfer,
    )
    monkeypatch.setattr(value_reads, "apply_value_transfer", forbid_copy)
    with pytest.raises(ExecutionPlanningError):
        owner.read(unit="alive", read=read, value=source)
    assert len(observed) == 1
    assert tuple(observed[0].source_sharding.mesh.devices.flat) == (
        jax.devices()[3],
        jax.devices()[1],
    )
    with pytest.raises(ExecutionPlanningError, match="uncommitted units"):
        owner.finish()
    assert not source.is_deleted()
    np.testing.assert_array_equal(source, policy.candidate_inner_action)


@pytest.mark.parametrize("subject_ids", [(3,), (3, 1)])
def test_host_replay_has_no_allocation_on_an_excluded_default_device(
    *,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    subject_ids: tuple[int, ...],
) -> None:
    """A NumPy artifact moves directly to the requested ordered destination."""
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    original = np.array([2.0, 5.0, 9.0], dtype=dtype)
    reads = replay_payload_reads(
        payload=original, key=SIMULATION_POLICY, period=0, regime="alive", core="replay"
    )
    devices = tuple(jax.devices()[index] for index in subject_ids)
    owner = PeriodSimulationReads(
        period=0, devices=devices, reads_by_unit={"alive": reads}, release_enabled=True
    )
    allocations: list[tuple[int, ...]] = []
    actual_put = jax.device_put

    def observe_put[**P](func: Callable[P, jax.Array]) -> Callable[P, jax.Array]:
        def observe(*args: P.args, **kwargs: P.kwargs) -> jax.Array:
            placed = func(*args, **kwargs)
            if isinstance(placed.sharding, jax.NamedSharding):
                allocations.append(
                    tuple(device.id for device in placed.sharding.mesh.devices.flat)
                )
            else:
                allocations.append(tuple(device.id for device in placed.devices()))
            return placed

        return observe

    monkeypatch.setattr(jax, "device_put", observe_put(actual_put))
    placed = cast(
        "jax.Array",
        place_replay_payload(
            payload=original,
            key=SIMULATION_POLICY,
            period=0,
            regime="alive",
            core="replay",
            owner=owner,
        ),
    )
    request.node.user_properties.append(("allocation_stages", repr(allocations)))
    assert allocations == [subject_ids]
    if isinstance(placed.sharding, jax.NamedSharding):
        assert tuple(device.id for device in placed.sharding.mesh.devices.flat) == (
            subject_ids
        )
    else:
        assert tuple(device.id for device in placed.devices()) == subject_ids
    np.testing.assert_array_equal(placed, [2.0, 5.0, 9.0])
    owner.commit(unit="alive", outputs=placed + 1)
    owner.finish()
    assert placed.is_deleted()
    np.testing.assert_array_equal(original, [2.0, 5.0, 9.0])


@pytest.mark.parametrize("subject_ids", [(3,), (3, 1)])
def test_external_reader_consumes_the_exact_validated_placed_cell(
    *, subject_ids: tuple[int, ...]
) -> None:
    """Keep authority while payload and node copies use the reader's device order."""
    model = _model(solver=ReferenceSolver())
    solution = _solve(model=model)
    prepared = model._resolve_solution_result(
        solution=solution, flat_params=model._process_params(_PARAMS)
    )[3][0]["active"]
    devices = tuple(jax.devices()[index] for index in subject_ids)
    owner = PeriodSimulationReads(
        period=0,
        devices=devices,
        reads_by_unit={"active": prepared.reads()},
        release_enabled=True,
    )
    reader = prepared.build(owner=owner, devices=devices)
    assert callable(reader)
    route = prepared.route
    assert isinstance(route, ReferenceReplayRoute)
    audit = route.audit
    assert audit.validated_snapshots[-1] == audit.reader_snapshots[-1]
    assert audit.validation_contexts[-1] is audit.reader_contexts[-1]
    original = audit.validated_contents[0]
    placed = audit.validated_contents[-1]
    consumed = audit.reader_contents[-1]
    assert original.metadata_identity == placed.metadata_identity
    assert placed.metadata_identity == consumed.metadata_identity
    assert original.authority_identities == placed.authority_identities
    assert placed.authority_identities == consumed.authority_identities
    assert placed.device_ids == consumed.device_ids == subject_ids
    assert original.values.dtype == placed.values.dtype == consumed.values.dtype
    np.testing.assert_array_equal(original.values, placed.values)
    np.testing.assert_array_equal(placed.values, consumed.values)
    context = audit.reader_contexts[-1]
    for old_nodes, new_nodes in (
        (prepared.context.state_nodes, context.state_nodes),
        (prepared.context.action_nodes, context.action_nodes),
    ):
        assert tuple(old_nodes) == tuple(new_nodes)
        for name, value in new_nodes.items():
            np.testing.assert_array_equal(value, old_nodes[name])
            assert value.dtype == old_nodes[name].dtype
            actual = (
                tuple(device.id for device in value.sharding.mesh.devices.flat)
                if isinstance(value.sharding, jax.NamedSharding)
                else tuple(device.id for device in value.devices())
            )
            assert actual == subject_ids
    owner.commit(unit="active", outputs=())
    owner.finish()


def test_repeated_host_replay_occurrences_reuse_the_same_addressed_copy() -> None:
    """Distinct core reads of one NumPy leaf share one period-owned conversion."""
    original = np.array([True, False, True])
    reads = tuple(
        read
        for core in ("replay", "transition")
        for read in replay_payload_reads(
            payload=original, key=SIMULATION_POLICY, period=0, regime="alive", core=core
        )
    )
    owner = PeriodSimulationReads(
        period=0,
        devices=(jax.devices()[3], jax.devices()[1]),
        reads_by_unit={"alive": reads},
        release_enabled=True,
    )
    copied = cast(
        "jax.Array",
        place_replay_payload(
            payload=original,
            key=SIMULATION_POLICY,
            period=0,
            regime="alive",
            core="replay",
            owner=owner,
        ),
    )
    again = place_replay_payload(
        payload=original,
        key=SIMULATION_POLICY,
        period=0,
        regime="alive",
        core="transition",
        owner=owner,
    )
    assert again is copied
    owner.commit(unit="alive", outputs=~copied)
    owner.finish()
    assert copied.is_deleted()
    np.testing.assert_array_equal(original, [True, False, True])


@pytest.mark.parametrize("subject_ids", [(0, 1, 2, 3), (3, 1)])
def test_a_physical_copy_uses_the_ordered_subject_devices_and_last_regime_lifetime(
    *, subject_ids: tuple[int, ...]
) -> None:
    """Full gathers and proper-submesh copies preserve originals and share once."""
    devices = jax.devices()
    stored_devices = devices if len(subject_ids) == 4 else [devices[0], devices[2]]
    mesh = jax.sharding.Mesh(np.asarray(stored_devices), ("stored",))
    stored = jax.device_put(jnp.arange(8), jax.NamedSharding(mesh, jax.P("stored")))
    first = _read(unit="first")
    second = _read(unit="second")
    subject_devices = tuple(devices[index] for index in subject_ids)
    owner = PeriodSimulationReads(
        period=0,
        devices=subject_devices,
        reads_by_unit={"first": (first,), "second": (second,)},
        release_enabled=True,
    )
    copied = owner.read(unit="first", read=first, value=stored)
    assert isinstance(copied.sharding, jax.NamedSharding)
    assert tuple(copied.sharding.mesh.devices.flat) == subject_devices
    assert not shares_a_buffer(first=copied, second=stored)
    np.testing.assert_array_equal(copied, np.arange(8))
    owner.commit(unit="first", outputs=copied + 1)
    assert not copied.is_deleted()
    assert owner.read(unit="second", read=second, value=stored) is copied
    owner.commit(unit="second", outputs=copied + 2)
    owner.finish()

    assert copied.is_deleted()
    np.testing.assert_array_equal(stored, np.arange(8))


def test_a_copy_that_reuses_an_original_shard_is_never_explicitly_deleted() -> None:
    """A wider replica can own fresh shards while still aliasing its stored source."""
    devices = jax.devices()
    stored = jax.device_put(
        jnp.arange(8), jax.sharding.SingleDeviceSharding(devices[1])
    )
    read = _read(unit="alive")
    owner = PeriodSimulationReads(
        period=0,
        devices=(devices[3], devices[1], devices[2], devices[0]),
        reads_by_unit={"alive": (read,)},
        release_enabled=True,
    )
    copied = owner.read(unit="alive", read=read, value=stored)
    assert copied is not stored
    assert len(copied.addressable_shards) == 4
    assert shares_a_buffer(first=copied, second=stored)
    owner.commit(unit="alive", outputs=copied + 1)
    owner.finish()

    np.testing.assert_array_equal(copied, np.arange(8))
    np.testing.assert_array_equal(stored, np.arange(8))


def _query_copy_terminal_wage(*, wage: FloatND) -> FloatND:
    return wage


def _query_copy_collective_terminal_wage(
    *, wage: FloatND, work: DiscreteAction
) -> FloatND:
    return wage + 0.0 * work


def _query_copy_leaf(
    *, arguments: Mapping[str, PytreeValue], path: tuple[str | int, ...]
) -> FloatND:
    node: PytreeValue = arguments
    for step in path:
        if isinstance(node, ArrivalQueryArguments):
            assert step == "by_target"
            node = node.by_target
        else:
            assert isinstance(node, Mapping)
            assert isinstance(step, str)
            node = node[step]
    assert isinstance(node, jax.Array)
    return node


def test_collective_queries_share_original_value_copies_on_an_ordered_submesh(  # noqa: PLR0915 - keep the actual owner lifecycle in one witness
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current and arrival queries share live future-value copies until completion."""
    declarations = _make_dissolution_regimes()
    declarations["married_terminal"] = declarations["married_terminal"].replace(
        functions={
            "utility": CollectiveUtility(
                utilities={
                    "f": _query_copy_collective_terminal_wage,
                    "m": _query_copy_collective_terminal_wage,
                }
            )
        }
    )
    for target in ("single_f_terminal", "single_m_terminal"):
        declarations[target] = declarations[target].replace(
            functions={"utility": _query_copy_terminal_wage}
        )
    model = Model(
        regimes=declarations,
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=DissolutionRegimeId,
        initial_nodes={0: "married"},
        edges={
            "married": _married_dissolution_transition(),
            "married_ir": {"married_terminal": 1},
            "single_f": {"single_f_terminal": 1},
            "single_m": {"single_m_terminal": (0, 1)},
        },
        execution_config=ExecutionConfig(devices=(0,)),
    )
    solution = model.solve(
        params=_DISSOLUTION_PARAMS, retention=ResultRetention.VALUES, log_level="off"
    )
    owned = solution._engine_view
    assert isinstance(owned, OwnedSolutionView)
    devices = jax.devices()
    subject_devices = (devices[3], devices[1])
    originals = {
        period: {
            target: jax.device_put(value, devices[2])
            for target, value in values.items()
        }
        for period, values in owned.values.items()
    }
    before = jax.tree.map(np.asarray, originals)
    regime = model._regimes["married_ir"]
    plan = regime.simulation.exit_queries[1]
    assert plan.full is not None
    families = (
        ("exit_full", plan.full, regime.simulation.programs.exit_full[1]),
        (
            "exit_participating",
            plan.participating["married_ir"],
            regime.simulation.programs.exit_participating[1],
        ),
        (
            "arrival_entry:married",
            plan.arrivals["married"].queries,
            regime.simulation.programs.arrival_program(
                family="arrival_entry:married", period=1
            ),
        ),
    )
    addresses = [
        {read.target for read in program.requirements.value_reads}
        for _, _, program in families
    ]
    assert set.intersection(*addresses)
    reads = tuple(
        read for _, _, program in families for read in program.requirements.value_reads
    )
    assert reads
    assert all(read.target.kind is ValueArtifactKind.REGIME_VALUE for read in reads)
    assert all(read.target.period == 2 for read in reads)
    owner = PeriodSimulationReads(
        period=1,
        devices=subject_devices,
        reads_by_unit={"married_ir": reads},
        release_enabled=True,
    )
    native_transfer = value_reads.apply_value_transfer
    transferred = []

    def observe(**call: Unpack[ApplyValueTransferKwargs]) -> FloatND:
        transferred.append(call["transfer"].target)
        return native_transfer(**call)

    monkeypatch.setattr(value_reads, "apply_value_transfer", observe)
    runtime = SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(3, 1),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({}),
            device_memory_bytes=None,
        ),
        enable_jit=True,
        subject_devices=subject_devices,
    )
    states = {"wage": jnp.asarray([1.0, 2.0, 3.0, 1.0])}
    params = model._process_params(_DISSOLUTION_PARAMS)
    copies = {}
    outputs = []
    for family, queries, program in families:
        query_arguments = _query_arguments(
            plan=plan,
            queries=queries,
            states=states,
            flat_params=params,
            period=1,
            ages=model._time,
            values=originals,
            owner=owner,
            unit="married_ir",
            family=family,
            regime=regime,
        )
        for read in program.requirements.value_reads:
            copied = _query_copy_leaf(arguments=query_arguments, path=read.source.path)
            original = originals[read.target.period][read.target.regime]
            assert copied.dtype == original.dtype
            assert isinstance(copied.sharding, jax.NamedSharding)
            assert tuple(copied.sharding.mesh.devices.flat) == subject_devices
            assert not shares_a_buffer(first=copied, second=original)
            np.testing.assert_array_equal(copied, original)
            assert np.any(np.asarray(original) != 0)
            if read.target in copies:
                assert copies[read.target] is copied
            copies[read.target] = copied
        arguments = {
            "states": states,
            "keys": {},
            "group_params": {},
            "query_arguments": query_arguments,
            "_holders": jnp.ones(4, dtype=bool),
            "taste_keys": {},
            "overrides": {},
        }
        if family == "exit_participating":
            arguments.update(held_bases={}, produced={})
        output = runtime.dispatch(
            program=program, arguments=arguments, period=1, n_subjects=4
        )
        jax.block_until_ready(output)
        answer = cast("OriginalExitOutput | ParticipatingExitOutput", output)[0]
        np.testing.assert_array_equal(answer[3], np.ones(4, dtype=bool))
        if family == "exit_participating":
            np.testing.assert_array_equal(answer[2], [True, False, True, True])
            np.testing.assert_allclose(
                answer[0][jnp.asarray([0, 2, 3])],
                [[2.95, 1.95], [8.85, 5.85], [2.95, 1.95]],
                rtol=10.0**-DECIMAL_PRECISION,
            )
        else:
            np.testing.assert_array_equal(answer[2], np.ones(4, dtype=bool))
            np.testing.assert_allclose(
                answer[0],
                [[2.95, 1.95], [7.4, 2.9], [8.85, 5.85], [2.95, 1.95]],
                rtol=10.0**-DECIMAL_PRECISION,
            )
            exits = tuple(queries.definitions[queries.root].exit_targets)
            assert exits
            rejected = (
                np.asarray(answer[1]) == 1
                if family == "arrival_entry:married"
                else np.isin(np.asarray(answer[1]), exits)
            )
            np.testing.assert_array_equal(rejected, [False, True, False, False])
        outputs.append(output)
        assert all(not copied.is_deleted() for copied in copies.values())
    assert len(transferred) == len(copies)
    assert set(transferred) == set(copies)
    owner.commit(unit="married_ir", outputs=tuple(outputs))
    owner.finish()
    assert all(copied.is_deleted() for copied in copies.values())
    for period, values in originals.items():
        for target, original in values.items():
            assert not original.is_deleted()
            assert original.devices() == {devices[2]}
            np.testing.assert_array_equal(original, before[period][target])
