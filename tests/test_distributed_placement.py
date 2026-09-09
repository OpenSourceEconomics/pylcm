"""A solve places every regime on a submesh of the visible devices.

Runs on a four-CPU-device topology pinned at import; the file skips wholesale
in a process whose backend is already initialized, so it runs in its own
process. Placement is a partition of the solve, never a change to it: the
values two placements of one model publish name the same real numbers, and a
simulation reads them off the canonical layout either way.
"""

import dataclasses
import functools
import logging
import subprocess
import sys
import weakref
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from _lcm.execution import value_transfer as transfers_module
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    ResolvedCoreProgram,
    ValueRead,
)
from _lcm.execution.eager_core import make_eager_core
from _lcm.execution.footprint import (
    ResidentInventory,
    concrete_device_bytes,
)
from _lcm.execution.output_layout import (
    VALUE,
    assert_output_layout,
    resolve_output_layout,
)
from _lcm.execution.runtime_sharding import runtime_shardings_match
from _lcm.execution.scheduler import (
    BufferRegistry,
    PeriodTransferCache,
    ReleaseRecord,
    shares_a_buffer,
)
from _lcm.execution.value_transfer import (
    MaterializedTransferObserver,
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
)
from _lcm.execution.workspace_planning import compiler_peak_bytes, plan_workspace
from _lcm.grids import categorical
from _lcm.grids.continuous import LinSpacedGrid
from _lcm.grids.discrete import DiscreteGrid
from _lcm.regime_building import processing
from _lcm.simulation.initial_conditions import build_initial_states
from _lcm.simulation.process_grids import SimulationProcessGrids
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from _lcm.solution import backward_induction
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.solution.v_topology import _get_regime_V_shapes_and_shardings
from _lcm.typing import RegimeName
from _lcm.utils.logging import LogLevel
from lcm import fixed_transition
from lcm.ages import AgeGrid
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.model import Model
from lcm.regime import Regime as UserRegime
from lcm.solver_api import ContinuationReader
from lcm.solvers import GridSearch, Solver
from lcm.typing import Float1D, ScalarFloat, ScalarInt
from tests.conftest import assert_agrees_to_ulp
from tests.execution.test_eager_core import eager_program, internal_eager_program

# Run these tests on a four-CPU-device topology. The pin only applies in a
# process whose JAX backends are not yet initialized; otherwise the tests skip.
# The device-count update is attempted FIRST because it is the one that raises
# after initialization, which keeps the pin atomic.
try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _PYTEST_PARALLEL = False
except RuntimeError:
    _PYTEST_PARALLEL = True

_skip_pytest_parallel = pytest.mark.skipif(
    _PYTEST_PARALLEL, reason="Can't set num cpus in pytest paralellel"
)


@_skip_pytest_parallel
@pytest.mark.parametrize("partitioned", [False, True])
def test_eager_internal_input_preserves_its_ordered_producer_layout(
    *, monkeypatch: pytest.MonkeyPatch, partitioned: bool
) -> None:
    """A layout-free internal template never turns a producer shard into a replica."""
    devices = (jax.devices()[3], jax.devices()[1])
    mesh = jax.sharding.Mesh(np.array(devices), ("state",))
    target = jax.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec("state")
        if partitioned
        else jax.sharding.PartitionSpec(),
    )
    source = jax.device_put(np.arange(8, dtype=np.int32), target)
    original_put = jax.device_put
    destinations: list[tuple[jax.Device, ...]] = []

    # keyword-only-exempt: library-callback=jax.device_put
    def record_put(value: object, device: object = None, **kwargs: Any) -> object:
        assert isinstance(device, jax.sharding.Sharding)
        destinations.append(tuple(device.device_set))
        return original_put(value, device, **kwargs)

    adapter = make_eager_core(
        program=internal_eager_program(function=lambda produced: produced),
        internal_input_templates={"produced": jax.ShapeDtypeStruct((8,), np.int32)},
        execution_sharding=target,
    )
    with monkeypatch.context() as probe:
        probe.setattr(jax, "device_put", record_put)
        output = adapter(produced=source)
    assert isinstance(output, jax.Array)
    assert runtime_shardings_match(actual=output.sharding, expected=target, ndim=1)
    assert isinstance(output.sharding, jax.NamedSharding)
    assert tuple(output.sharding.mesh.devices.flat) == devices
    assert all(set(destination) == set(devices) for destination in destinations)
    assert len(destinations) <= 1
    np.testing.assert_array_equal(output, np.arange(8))
    np.testing.assert_array_equal(source, np.arange(8))
    assert not source.is_deleted()


_PARAMS = {"discount_factor": 0.95}


def _ordered_eager_sharding(*, explicit: bool = False) -> jax.NamedSharding:
    """Build an actual ordered submesh, bypassing model device-set sorting."""
    mesh = jax.sharding.Mesh(
        np.asarray([jax.devices()[i] for i in (3, 1, 2)]),
        ("kind",),
        axis_types=(
            jax.sharding.AxisType.Explicit if explicit else jax.sharding.AxisType.Auto,
        ),
    )
    return jax.NamedSharding(mesh, jax.P("kind", None))


@_skip_pytest_parallel
@pytest.mark.parametrize("operand", ["uncommitted", "committed"])
@pytest.mark.parametrize("constant", [False, True])
def test_eager_ordered_mesh_preserves_owners_and_outputs_at_birth(
    *, operand: str, constant: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _ordered_eager_sharding()
    source = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    if operand == "committed":
        source = jax.device_put(source, expected)
    before = np.asarray(source).copy()
    fixed = jnp.asarray(2.0)
    observed: list[object] = []
    body_finished = False
    original_put = jax.device_put

    def observed_put(*args: Any, **kwargs: Any) -> object:
        assert not body_finished, "eager adapter repaired an already produced output"
        return original_put(*args, **kwargs)

    def body(*, value: jax.Array, offset: jax.Array) -> object:
        nonlocal body_finished
        assert offset is fixed
        result = (
            jax.vmap(lambda _row: jnp.full((2,), 7.0))(value)
            if constant
            else value + offset
        )
        tree = {"payload": (result, None)}
        observed.append(tree)
        body_finished = True
        return tree

    function = functools.partial(body, offset=fixed)
    arguments: dict[str, object] = {
        "value": jax.ShapeDtypeStruct(source.shape, source.dtype, sharding=expected)
    }
    adapter = make_eager_core(
        program=eager_program(function=function, arguments=arguments),
        execution_sharding=expected,
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(jax, "device_put", observed_put)
        result = adapter(value=source)
    assert result is observed[0]
    value = cast("dict[str, tuple[jax.Array, None]]", result)["payload"][0]
    assert isinstance(value.sharding, jax.NamedSharding)
    assert tuple(value.sharding.mesh.devices.flat) == tuple(
        jax.devices()[i] for i in (3, 1, 2)
    )
    assert value.sharding.mesh.devices.shape == (3,)
    assert value.sharding.mesh.axis_names == ("kind",)
    assert value.sharding.memory_kind == expected.memory_kind
    assert value.sharding.devices_indices_map(
        value.shape
    ) == expected.devices_indices_map((3, 2))
    assert {shard.data.shape for shard in value.addressable_shards} == {(1, 2)}
    assert jax.devices()[0] not in value.devices()
    np.testing.assert_array_equal(
        value, np.full((3, 2), 7.0) if constant else before + 2
    )
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(fixed, np.asarray(2.0))
    assert not source.is_deleted()


@_skip_pytest_parallel
def test_eager_committed_operand_is_not_silently_moved() -> None:
    expected = _ordered_eager_sharding()
    source = jax.device_put(jnp.ones((3, 2)), jax.devices()[0])

    def forbidden(*, value: object) -> object:
        pytest.fail(f"a misplaced committed operand reached the body: {value!r}")

    adapter = make_eager_core(
        program=eager_program(
            function=forbidden,
            arguments={
                "value": jax.ShapeDtypeStruct(
                    source.shape, source.dtype, sharding=expected
                )
            },
        ),
        execution_sharding=expected,
    )
    with pytest.raises(ExecutionPlanningError, match="committed eager operand"):
        adapter(value=source)
    assert source.devices() == {jax.devices()[0]}
    np.testing.assert_array_equal(source, np.ones((3, 2)))


@_skip_pytest_parallel
def test_eager_nested_aliases_and_context_restoration() -> None:
    expected = _ordered_eager_sharding()
    source = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    original = MappingProxyType({"second": source, "first": (source, None)})
    descriptor = jax.ShapeDtypeStruct(source.shape, source.dtype, sharding=expected)
    descriptors = MappingProxyType({"second": descriptor, "first": (descriptor, None)})
    outside_mesh = jax.get_mesh()
    outside_device = jax.config.jax_default_device

    def body(*, values: Mapping[str, Any]) -> object:
        assert type(values) is MappingProxyType
        assert tuple(values) == tuple(original)
        assert jax.tree.structure(values) == jax.tree.structure(original)
        assert shares_a_buffer(first=values["second"], second=values["first"][0])
        assert tuple(jax.get_mesh().devices.flat) == tuple(expected.mesh.devices.flat)
        assert jax.config.jax_default_device == jax.devices()[3]
        raise RuntimeError("body failure after observing placed aliases")

    adapter = make_eager_core(
        program=eager_program(function=body, arguments={"values": descriptors}),
        execution_sharding=expected,
    )
    with pytest.raises(
        RuntimeError, match="body failure after observing placed aliases"
    ):
        adapter(values=original)
    assert original["second"] is original["first"][0]
    assert jax.get_mesh() == outside_mesh
    assert jax.config.jax_default_device == outside_device
    np.testing.assert_array_equal(source, np.arange(6).reshape(3, 2))


@_skip_pytest_parallel
def test_eager_repeated_original_keeps_distinct_declared_layouts() -> None:
    sharded = _ordered_eager_sharding()
    replicated = jax.NamedSharding(sharded.mesh, jax.P())
    source = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)

    def body(*, partitioned: jax.Array, whole: jax.Array) -> object:
        assert partitioned is not whole
        assert partitioned.sharding.devices_indices_map(
            source.shape
        ) == sharded.devices_indices_map(source.shape)
        assert whole.sharding.devices_indices_map(
            source.shape
        ) == replicated.devices_indices_map(source.shape)
        np.testing.assert_array_equal(partitioned, source)
        np.testing.assert_array_equal(whole, source)
        return partitioned, whole

    adapter = make_eager_core(
        program=eager_program(
            function=body,
            arguments={
                "partitioned": jax.ShapeDtypeStruct(
                    source.shape, source.dtype, sharding=sharded
                ),
                "whole": jax.ShapeDtypeStruct(
                    source.shape, source.dtype, sharding=replicated
                ),
            },
        ),
        execution_sharding=sharded,
    )
    adapter(partitioned=source, whole=source)
    np.testing.assert_array_equal(source, np.arange(6).reshape(3, 2))


@_skip_pytest_parallel
@pytest.mark.parametrize(
    "difference",
    [
        "wrong_device",
        "reordered",
        "replicated",
        "axis_names",
        "mesh_shape",
        "memory_kind",
    ],
)
def test_physical_layout_guard_rejects_actual_placement_changes(
    *, difference: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _ordered_eager_sharding()
    actual = jax.NamedSharding(expected.mesh, jax.P("kind"))
    assert runtime_shardings_match(actual=actual, expected=expected, ndim=2)
    if difference == "wrong_device":
        actual = jax.NamedSharding(
            jax.sharding.Mesh(np.asarray(jax.devices()[:3]), ("kind",)), jax.P("kind")
        )
    elif difference == "reordered":
        actual = jax.NamedSharding(
            jax.sharding.Mesh(
                np.asarray([jax.devices()[i] for i in (1, 3, 2)]), ("kind",)
            ),
            jax.P("kind"),
        )
    elif difference == "replicated":
        actual = jax.NamedSharding(expected.mesh, jax.P())
    elif difference == "axis_names":
        actual = jax.NamedSharding(
            jax.sharding.Mesh(expected.mesh.devices, ("other",)), jax.P("other")
        )
    elif difference == "mesh_shape":
        expected = jax.NamedSharding(
            jax.sharding.Mesh(np.asarray(jax.devices()).reshape(2, 2), ("a", "b")),
            jax.P(),
        )
        actual = jax.NamedSharding(
            jax.sharding.Mesh(np.asarray(jax.devices()).reshape(4, 1), ("a", "b")),
            jax.P(),
        )
        assert actual.is_equivalent_to(expected, 2)
    else:
        # CPU metadata mutation: this backend need not expose a second memory kind.
        monkeypatch.setattr(
            jax.NamedSharding,
            "memory_kind",
            property(lambda self: "pinned_host" if self is actual else "device"),
        )
    assert not runtime_shardings_match(actual=actual, expected=expected, ndim=2)


@_skip_pytest_parallel
def test_equivalent_eager_output_flows_through_declared_transfer_unchanged() -> None:
    planned = _ordered_eager_sharding()
    template = jax.device_put(jnp.arange(6, dtype=jnp.float32).reshape(3, 2), planned)
    layout = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=VALUE,
    )
    actual_sharding = jax.NamedSharding(
        _ordered_eager_sharding(explicit=True).mesh, jax.P("kind")
    )
    output = jax.device_put(template, actual_sharding)
    transfer = transfers_module.resolve_value_transfer(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="working"
        ),
        source=ValueConsumerAddress(
            source_period=0,
            source_regime="working",
            core_key="main",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            path=("working",),
        ),
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_template=template,
        source_sharding=planned,
    )
    original_key = transfer.specialization_key
    assert_output_layout(output=output, layout=layout)
    assert (
        transfers_module.apply_value_transfer(value=output, transfer=transfer) is output
    )
    assert transfer.specialization_key == original_key
    assert transfer.stored_sharding == planned
    assert transfer.source_sharding == planned
    wrong = jax.device_put(template, jax.NamedSharding(planned.mesh, jax.P()))
    with pytest.raises(AssertionError, match="output sharding"):
        assert_output_layout(output=wrong, layout=layout)
    with pytest.raises(ValueError, match="sharding"):
        transfers_module.apply_value_transfer(value=wrong, transfer=transfer)
    np.testing.assert_array_equal(output, template)


@categorical(ordered=False)
class _ThreeTypeRegimeId:
    """Regime vocabulary of the three-valued-type model."""

    working: ScalarInt
    retired: ScalarInt


@categorical(ordered=True)
class _Type:
    """A three-valued preference type; its extent is the mesh size."""

    low: ScalarInt
    mid: ScalarInt
    high: ScalarInt


def _constant_retired_value(*, wealth: jax.Array, type1: jax.Array) -> jax.Array:
    """Declare both state axes without reading either in the numerical body."""
    del wealth, type1
    return jnp.asarray(2.0)


def _make_three_type_model(
    *,
    distributed: bool,
    sharded: tuple[str, ...] = (),
    devices: tuple[int, ...] | None = None,
    solver: Solver | None = None,
    budget_bytes: int | None = None,
    enable_jit: bool = True,
    constant_retired: bool = False,
) -> Model:
    """A working regime over a three-valued type beside a single-device terminal one.

    Both regimes are active before the final age and read nothing of each other
    within a period, so on four devices the working regime runs on three and the
    terminal one on the fourth. `sharded` names the same axis through
    `ExecutionConfig`; either spelling places the regime the same way.
    `devices` restricts the model to a subset of the four.
    """
    working = UserRegime(
        active=lambda age: age < 4,
        solver=GridSearch() if solver is None else solver,
        functions={
            "utility": lambda wealth, consumption, type1: (
                (jnp.log(consumption) + wealth * 0.001) * (type1 + 1)
            ),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=12)},
        state_transitions={"wealth": lambda wealth, consumption: wealth - consumption},
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 3, _ThreeTypeRegimeId.retired, _ThreeTypeRegimeId.working
        ),
    )
    retired = UserRegime(
        transition=None,
        functions={
            "utility": (
                _constant_retired_value
                if constant_retired
                else (lambda wealth: wealth * 0.5)
            )
        },
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=12)},
    )
    return Model(
        regimes={"working": working, "retired": retired},
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=_ThreeTypeRegimeId,
        enable_jit=enable_jit,
        states={"type1": DiscreteGrid(category_class=_Type)},
        state_transitions={"type1": fixed_transition("type1")},
        execution_config=ExecutionConfig(
            device_memory_bytes=budget_bytes,
            sharded_states=tuple(
                dict.fromkeys((*sharded, *(("type1",) if distributed else ())))
            ),
            devices=devices,
        ),
    )


@_skip_pytest_parallel
@pytest.mark.parametrize(
    ("budget_bytes", "enable_jit", "devices"),
    [
        (None, True, None),
        (128 * 1024 * 1024, True, None),
        (None, True, (3, 1, 2, 0)),
        (128 * 1024 * 1024, True, (3, 1, 2, 0)),
        (None, False, (0,)),
    ],
)
def test_solve_planning_keeps_descriptors_instead_of_transferred_buffers(
    *,
    monkeypatch: pytest.MonkeyPatch,
    budget_bytes: int | None,
    devices: tuple[int, ...] | None,
    enable_jit: bool,
) -> None:
    """All width candidates resolve before admission without copying real values."""
    model = _make_three_type_model(
        distributed=enable_jit,
        budget_bytes=budget_bytes,
        devices=devices,
        enable_jit=enable_jit,
    )
    original_resolve = backward_induction._resolve_output_layouts_and_lowering_keys
    original_transfer = transfers_module.apply_value_transfer
    original_peak = backward_induction.compiler_peak_bytes
    planning = False
    peak_calls: list[object] = []
    planning_copies: list[tuple[weakref.ReferenceType[jax.Array], int]] = []
    runtime_copies: list[ResolvedValueTransfer] = []
    runtime_reads: list[ResolvedValueTransfer] = []
    candidate_counts: list[int] = []

    def observe_peak(**kwargs: Any) -> int:
        peak_calls.append(kwargs["compiled"])
        return original_peak(**kwargs)

    def observe_transfer(
        *,
        value: object,
        transfer: ResolvedValueTransfer,
        on_materialized: MaterializedTransferObserver | None = None,
    ) -> jax.Array:
        copied = original_transfer(
            value=value, transfer=transfer, on_materialized=on_materialized
        )
        if not planning:
            assert isinstance(value, jax.Array)
            runtime_reads.append(transfer)
        if transfer.kind is not ValueTransferKind.ALIGNED_LOCAL:
            assert isinstance(value, jax.Array)
            if planning:
                assert not shares_a_buffer(first=value, second=copied)
                assert copied.sharding == transfer.source_sharding
                planning_copies.append((weakref.ref(copied), copied.nbytes))
            else:
                runtime_copies.append(transfer)
        return copied

    def observe_planning(**kwargs: Any) -> Any:
        nonlocal planning
        planning = True
        try:
            result = original_resolve(**kwargs)
        finally:
            planning = False
        programs = result[2]
        candidate_counts.append(len(programs))
        assert not peak_calls, "Compiler admission must follow candidate resolution."
        _assert_only_planning_descriptors(programs=programs, copies=planning_copies)
        return result

    monkeypatch.setattr(backward_induction, "compiler_peak_bytes", observe_peak)
    monkeypatch.setattr(transfers_module, "apply_value_transfer", observe_transfer)
    monkeypatch.setattr(
        backward_induction,
        "_resolve_output_layouts_and_lowering_keys",
        observe_planning,
    )

    solution = model.solve(params=_PARAMS, log_level="off")

    assert candidate_counts
    assert candidate_counts[0] > 1
    assert runtime_reads, "Real dispatch must still consume concrete value operands."
    if enable_jit:
        assert runtime_copies, "The compiled solve still needs cross-mesh value copies."
    for values in solution.values.values():
        if "retired" in values:
            expected = np.linspace(1.0, 100.0, 12) * 0.5
            assert_agrees_to_ulp(got=values["retired"], expected=expected, n_ulp=8)


def _assert_only_planning_descriptors[Key: Hashable](
    *,
    programs: Mapping[Key, ResolvedCoreProgram],
    copies: list[tuple[weakref.ReferenceType[jax.Array], int]],
) -> None:
    """Check the real pre-admission candidate tree and any copied buffers."""
    retained = {
        id(leaf)
        for program in programs.values()
        for leaf in jax.tree.leaves(program.arguments)
        if isinstance(leaf, jax.Array)
    }
    live = [(ref(), size) for ref, size in copies]
    assert all(array is not None and id(array) in retained for array, _ in live)
    assert not live, (
        f"Solve planning allocated and retained {len(live)} concrete copies "
        f"({sum(size for _, size in live)} global bytes) across "
        f"{len(programs)} candidates before width admission."
    )
    assert all(
        isinstance(leaf, jax.ShapeDtypeStruct)
        for program in programs.values()
        for leaf in jax.tree.leaves(program.arguments)
    )


@_skip_pytest_parallel
@pytest.mark.parametrize("devices", [None, (1, 2, 3)])
@pytest.mark.parametrize("constant_retired", [False, True])
def test_eager_solve_respects_planned_regime_layouts(
    *, devices: tuple[int, ...] | None, constant_retired: bool
) -> None:
    """Eager computations, including constant bodies, obey actual regime placement."""
    eager = _make_three_type_model(
        distributed=True,
        enable_jit=False,
        devices=devices,
        constant_retired=constant_retired,
    ).solve(params=_PARAMS, log_level="off")
    compiled = _make_three_type_model(
        distributed=True, devices=devices, constant_retired=constant_retired
    ).solve(params=_PARAMS, log_level="off")

    assert tuple(eager.values) == tuple(compiled.values)
    for period, values in eager.values.items():
        assert tuple(values) == tuple(compiled.values[period])
        for regime, value in values.items():
            expected = compiled.values[period][regime]
            assert value.shape == expected.shape
            assert value.dtype == expected.dtype
            assert value.sharding.is_equivalent_to(expected.sharding, value.ndim)
            assert value.sharding.memory_kind == expected.sharding.memory_kind
            assert value.sharding.devices_indices_map(value.shape) == (
                expected.sharding.devices_indices_map(expected.shape)
            )
            assert tuple(shard.data.shape for shard in value.addressable_shards) == (
                tuple(shard.data.shape for shard in expected.addressable_shards)
            )
            if devices is not None:
                assert value.devices() <= {jax.devices()[index] for index in devices}
                assert jax.devices()[0] not in value.devices()
            assert_agrees_to_ulp(got=value, expected=expected, n_ulp=8)


@_skip_pytest_parallel
def test_sharded_state_from_execution_config_places_the_regime_on_the_submesh() -> None:
    """Declaring a state in `sharded_states` places its regime's nodes on a submesh."""
    model = _make_three_type_model(distributed=False, sharded=("type1",))

    assert model._regimes["working"].solution.submesh_device_ids == (0, 1, 2)


@_skip_pytest_parallel
def test_sharded_state_from_execution_config_shards_the_value() -> None:
    """A state named in `sharded_states` carries the same device axis as the field."""
    solution = _make_three_type_model(distributed=False, sharded=("type1",)).solve(
        params=_PARAMS, log_level="off"
    )
    mesh = solution.values[0]["working"].sharding.mesh  # ty: ignore[unresolved-attribute]

    assert tuple(device.id for device in mesh.devices.flat) == (0, 1, 2)


@_skip_pytest_parallel
def test_execution_devices_reports_the_configured_device_ids() -> None:
    """`Model.execution_devices` names exactly the ids the configuration gave."""
    model = _make_three_type_model(distributed=False, devices=(2, 3))

    assert model.execution_devices == (2, 3)


@_skip_pytest_parallel
def test_the_planner_partitions_the_models_own_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The submesh planner is given the model's device count, not JAX's."""
    recorded: list[int] = []
    original = processing.plan_submesh_placement

    def _recording(*, requests: Any, n_devices: int) -> Any:
        recorded.append(n_devices)
        return original(requests=requests, n_devices=n_devices)

    monkeypatch.setattr(processing, "plan_submesh_placement", _recording)
    _make_three_type_model(distributed=False, devices=(2, 3))

    assert recorded == [2]


@_skip_pytest_parallel
def test_a_sharded_regime_is_placed_on_the_configured_device_ids() -> None:
    """The planner's block positions are read back as the model's own device ids."""
    model = _make_three_type_model(
        distributed=False, sharded=("type1",), devices=(1, 2, 3)
    )

    assert model._regimes["working"].solution.submesh_device_ids == (1, 2, 3)


@_skip_pytest_parallel
def test_a_model_restricted_to_two_devices_publishes_every_value_on_them() -> None:
    """Every value a solve publishes lives on a device the configuration named."""
    solution = _make_three_type_model(distributed=False, devices=(2, 3)).solve(
        params=_PARAMS, log_level="off"
    )

    published_device_ids = {
        device.id
        for by_regime in solution.values.values()
        for value in by_regime.values()
        for device in value.sharding.device_set
    }

    assert published_device_ids <= {2, 3}


@_skip_pytest_parallel
def test_the_regime_beside_a_sharded_one_stays_on_the_configured_devices() -> None:
    """A single-device regime of a restricted model keeps off the excluded devices."""
    solution = _make_three_type_model(
        distributed=False, sharded=("type1",), devices=(1, 2, 3)
    ).solve(params=_PARAMS, log_level="off")
    value = solution.values[0]["retired"]

    assert {device.id for device in value.sharding.device_set} <= {1, 2, 3}


@_skip_pytest_parallel
def test_seeded_subject_states_of_a_restricted_model_stay_on_its_devices() -> None:
    """Per-subject simulate arrays are seeded on a device the configuration named."""
    model = _make_three_type_model(distributed=False, devices=(2, 3))
    states_per_regime = build_initial_states(
        initial_states={
            "wealth": jnp.full(4, 50.0),
            "type1": jnp.asarray([0, 1, 2, 0]),
        },
        regimes=model._regimes,
        device_ids=model.execution_devices,
    )

    seeded_device_ids = {
        device.id
        for regime_states in states_per_regime.values()
        for array in regime_states.values()
        for device in array.sharding.device_set
    }

    assert seeded_device_ids <= {2, 3}


@_skip_pytest_parallel
def test_a_three_valued_type_is_sharded_over_three_devices() -> None:
    """The working regime's value lives on a three-device mesh."""
    solution = _make_three_type_model(distributed=True).solve(
        params=_PARAMS, log_level="off"
    )
    value = solution.values[0]["working"]

    mesh = value.sharding.mesh  # ty: ignore[unresolved-attribute]

    assert tuple(device.id for device in mesh.devices.flat) == (0, 1, 2)


@_skip_pytest_parallel
def test_the_single_device_regime_takes_the_idle_device() -> None:
    """The terminal regime's value lives on the device the mesh leaves idle."""
    solution = _make_three_type_model(distributed=True).solve(
        params=_PARAMS, log_level="off"
    )
    value = solution.values[0]["retired"]

    assert value.sharding == jax.sharding.SingleDeviceSharding(jax.devices()[3])


@_skip_pytest_parallel
@pytest.mark.parametrize("regime", ["working", "retired"])
def test_two_placements_of_one_model_publish_the_same_values(
    *, regime: RegimeName
) -> None:
    """Placement partitions a solve without changing what it computes.

    Sharding the type axis over three devices and keeping the whole regime on
    one are the same arithmetic in a different partition, and XLA vectorizes
    each at its own width, so the two runs name the same real number rather
    than the same bit pattern.
    """
    placed = _make_three_type_model(distributed=True).solve(
        params=_PARAMS, log_level="off"
    )
    canonical = _make_three_type_model(distributed=False).solve(
        params=_PARAMS, log_level="off"
    )

    expected_roster = {
        period: {"working", "retired"} if period < 4 else {"retired"}
        for period in range(5)
    }
    assert {period: set(values) for period, values in placed.values.items()} == (
        expected_roster
    )
    assert {period: set(values) for period, values in canonical.values.items()} == (
        expected_roster
    )
    for period, active in expected_roster.items():
        if regime not in active:
            continue
        assert_agrees_to_ulp(
            got=np.asarray(placed.values[period][regime]),
            expected=np.asarray(canonical.values[period][regime]),
            n_ulp=8,
            err_msg=f"regime {regime!r}, period {period}",
        )


@_skip_pytest_parallel
def test_independent_regimes_of_one_period_share_one_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both regimes of a period are dispatched together in the period's first wave."""
    units_by_period: dict[int, int] = {}
    recorder = _WavePlanRecorder(
        planner=backward_induction.plan_period_waves, units_by_period=units_by_period
    )
    monkeypatch.setattr(backward_induction, "plan_period_waves", recorder)
    _make_three_type_model(distributed=True).solve(params=_PARAMS, log_level="off")

    assert units_by_period == {0: 2, 1: 2, 2: 2, 3: 2, 4: 1}


class _WavePlanRecorder:
    """Call the real wave planner and record each period's first wave width."""

    def __init__(self, *, planner: object, units_by_period: dict[int, int]) -> None:
        """Keep the planner to delegate to and the mapping to record into."""
        self._planner = planner
        self._units_by_period = units_by_period

    def __call__(self, **kwargs: object) -> object:
        """Plan the period's waves and record how many units the first one holds."""
        waves = self._planner(**kwargs)  # ty: ignore[call-non-callable]
        self._units_by_period[waves[0][0].period] = len(waves[0])
        return waves


@_skip_pytest_parallel
def test_a_model_with_one_regime_per_period_is_placed_on_device_zero() -> None:
    """Where nothing is co-active, a single-device regime keeps today's placement."""
    from tests.test_distributed import (  # noqa: PLC0415
        _make_correct_distributed_model,
    )

    solution = _make_correct_distributed_model(distributed=False).solve(
        params=_PARAMS, log_level="off"
    )
    value = solution.values[5]["retirement"]

    assert value.sharding.device_set == {jax.devices()[0]}


@_skip_pytest_parallel
def test_simulating_a_submesh_placed_solution_uses_the_subject_devices() -> None:
    """Subjects span four devices while the original three-device value survives."""
    model = _make_three_type_model(distributed=True)
    solution = model.solve(params=_PARAMS, log_level="off")
    view = solution._engine_view
    assert isinstance(view, OwnedSolutionView)
    original = view.values[0]["working"]
    original_values = np.asarray(original).copy()
    assert original.sharding.device_set == set(jax.devices()[:3])

    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.array([10.0, 20.0, 30.0, 40.0]),
            "type1": jnp.array([0, 1, 2, 1]),
            "age": jnp.zeros(4),
            "regime_id": jnp.array([0, 0, 0, 0]),
        },
        solution=solution,
        log_level="off",
        seed=42,
    )
    assert result.n_subjects == 4
    assert result.raw_results["working"][0].V_arr.sharding.device_set == (
        set(jax.devices())
    )
    np.testing.assert_array_equal(np.asarray(original), original_values)


@_skip_pytest_parallel
def test_simulating_co_active_single_device_regimes_matches_one_device(
    tmp_path: Path,
) -> None:
    """Values placed off device zero are brought back before simulation."""
    code = (
        "import jax; jax.config.update('jax_num_cpu_devices', 1); "
        "import sys; import numpy as np; "
        "from tests.test_distributed_placement import _simulate_two_single_regimes; "
        "np.save(sys.argv[1], _simulate_two_single_regimes())"
    )
    reference = tmp_path / "one-device.npy"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(reference)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0, result.stderr

    np.testing.assert_array_equal(_simulate_two_single_regimes(), np.load(reference))


def _simulate_two_single_regimes() -> np.ndarray:
    """Solve and simulate the two-regime out-of-tree model; return its wealth.

    Its `alive` and `dead` regimes are co-active in every period but the last,
    so on four devices `alive` sits on device 0 and `dead` on device 1.
    """
    from tests.test_solver_api_out_of_tree import (  # noqa: PLC0415
        WealthSolver,
        _two_regime_model,
    )

    model = _two_regime_model(solver=WealthSolver())
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": jnp.array([1.0, 2.0, 3.0, 4.0]),
            "age": jnp.zeros(4),
            "regime_id": jnp.array([0, 0, 0, 0]),
        },
        log_level="off",
    )
    return np.asarray(result.to_dataframe()["wealth"])


@categorical(ordered=False)
class _TwoMeshRegimeId:
    """Regime vocabulary of the two-sharded-regime model."""

    alpha: ScalarInt
    beta: ScalarInt
    retired: ScalarInt


def _make_two_mesh_model() -> Model:
    """Two co-active regimes over one three-valued type beside a terminal one.

    Both sharded regimes take the same three-device block, and the terminal
    regime — which does not read the type — takes the device that block leaves
    idle. Each sharded regime therefore reads the terminal value across
    disjoint devices, into the one replicated layout their shared mesh
    defines.
    """

    def _worker() -> UserRegime:
        return UserRegime(
            functions={
                "utility": lambda wealth, consumption, type1: (
                    jnp.log(consumption) + wealth * 0.001 * (type1 + 1)
                ),
            },
            states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=8)},
            state_transitions={
                "wealth": lambda wealth, consumption: wealth - consumption
            },
            actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=6)},
            transition=lambda age: jnp.where(
                age >= 0, _TwoMeshRegimeId.retired, _TwoMeshRegimeId.alpha
            ),
        )

    return Model(
        regimes={
            "alpha": _worker(),
            "beta": _worker(),
            "retired": UserRegime(
                transition=None,
                functions={"utility": lambda wealth: wealth * 0.5},
                states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=8)},
            ),
        },
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_TwoMeshRegimeId,
        states={"type1": DiscreteGrid(category_class=_Type)},
        execution_config=ExecutionConfig(sharded_states=("type1",)),
        state_transitions={"type1": fixed_transition("type1")},
    )


class _SharedCopyLog:
    """Registered shared-transfer copies and their liveness after every commit."""

    def __init__(self) -> None:
        """Start with no copy recorded and no commit observed."""
        self.copies: list[jax.Array] = []
        self.deleted_after_commit: list[bool] = []


class _RecordingTransferCache(PeriodTransferCache):
    """A period transfer cache that reports what it registers and releases."""

    __slots__ = ("log",)

    def put(
        self,
        *,
        transfer: ResolvedValueTransfer,
        array: jax.Array,
        stored: jax.Array,
    ) -> None:
        """Cache the copy, recording it when it occupies a buffer of its own."""
        super().put(transfer=transfer, array=array, stored=stored)
        if not shares_a_buffer(first=array, second=stored):
            self.log.copies.append(array)

    def commit_consumer(
        self, *, key: tuple[Hashable, Hashable]
    ) -> tuple[ReleaseRecord, ...]:
        """Commit, then record whether the period's copy is already deleted."""
        records = super().commit_consumer(key=key)
        if self.log.copies:
            self.log.deleted_after_commit.append(self.log.copies[-1].is_deleted())
        return records


class _RecordingTransferCacheFactory:
    """Build recording caches that all report into one log."""

    def __init__(self, *, log: _SharedCopyLog) -> None:
        """Keep the log every cache this factory builds reports into."""
        self._log = log

    def __call__(self, **kwargs: Any) -> _RecordingTransferCache:
        """Build one period's recording cache."""
        cache = _RecordingTransferCache(**kwargs)
        cache.log = self._log
        return cache


def _record_shared_copy_lifetime(*, monkeypatch: pytest.MonkeyPatch) -> _SharedCopyLog:
    """Solve the two-mesh model, recording each registered copy's liveness.

    Every commit of a shared-transfer key appends whether the copy the period
    registered is already deleted, so the recorded sequence says exactly when
    the copy was freed relative to its declared consumers.
    """
    log = _SharedCopyLog()
    monkeypatch.setattr(
        backward_induction,
        "PeriodTransferCache",
        _RecordingTransferCacheFactory(log=log),
    )
    _make_two_mesh_model().solve(params=_PARAMS, log_level="off")
    return log


@_skip_pytest_parallel
def test_a_cross_device_regime_value_copy_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading a value from off the reader's mesh makes a copy the engine owns."""
    log = _record_shared_copy_lifetime(monkeypatch=monkeypatch)

    assert log.copies


@_skip_pytest_parallel
def test_a_shared_copy_survives_its_first_consumers_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy stays alive while a second declared consumer has not committed."""
    log = _record_shared_copy_lifetime(monkeypatch=monkeypatch)

    assert log.deleted_after_commit[0] is False


@_skip_pytest_parallel
def test_a_shared_copy_is_deleted_after_its_last_consumers_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy is released once every declared consumer has committed."""
    log = _record_shared_copy_lifetime(monkeypatch=monkeypatch)

    assert log.deleted_after_commit[1] is True


@_skip_pytest_parallel
def test_two_co_active_sharded_regimes_take_the_same_block() -> None:
    """Blocks follow declaration order, so two equal meshes may overlap."""
    solution = _make_two_mesh_model().solve(params=_PARAMS, log_level="off")

    assert solution.values[0]["alpha"].sharding == solution.values[0]["beta"].sharding


@categorical(ordered=False)
class _TwoBlockRegimeId:
    """Regime vocabulary of the two-block model."""

    first: ScalarInt
    second: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _TwoValuedType:
    """A two-valued preference type; its extent is each block's mesh size."""

    low: ScalarInt
    high: ScalarInt


def _make_two_block_model(*, distributed: bool) -> Model:
    """Two sharded regimes over a two-valued type; the first enters the second.

    Both regimes carry the type, so each takes a two-device block, and on four
    devices the blocks are disjoint. The first regime's continuation therefore
    reads the second regime's value from devices its own mesh does not hold.
    """

    def _utility(*, wealth: Any, consumption: Any, type1: Any) -> Any:
        return (jnp.log(consumption) + wealth * 0.001) * (type1 + 1)

    def _next_wealth(*, wealth: Any, consumption: Any) -> Any:
        return wealth - consumption

    def _worker(*, transition: Any, active: Any) -> UserRegime:
        return UserRegime(
            functions={"utility": _utility},
            states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
            state_transitions={"wealth": _next_wealth},
            actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
            transition=transition,
            active=active,
        )

    first = _worker(
        transition=lambda age: jnp.where(
            age >= 1, _TwoBlockRegimeId.second, _TwoBlockRegimeId.first
        ),
        active=lambda age: age < 3,
    )
    second = _worker(
        transition=lambda age: jnp.where(
            age >= 3, _TwoBlockRegimeId.dead, _TwoBlockRegimeId.second
        ),
        active=lambda _age: True,
    )
    dead = UserRegime(
        transition=None,
        functions={"utility": lambda wealth, type1: 0.0 * wealth * type1},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        active=lambda age: age >= 4,
    )
    return Model(
        regimes={"first": first, "second": second, "dead": dead},
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=_TwoBlockRegimeId,
        states={"type1": DiscreteGrid(category_class=_TwoValuedType)},
        state_transitions={"type1": fixed_transition("type1")},
        execution_config=ExecutionConfig(
            sharded_states=("type1",) if distributed else ()
        ),
    )


@_skip_pytest_parallel
def test_a_value_read_across_disjoint_blocks_is_a_cross_mesh_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first block's regime reads the second block's value by cross-mesh copy."""
    captured: list[Any] = []
    original = backward_induction._attach_resolved_output_layout

    def capture(**kwargs: Any) -> Any:
        core = original(**kwargs)
        captured.append(core)
        return core

    monkeypatch.setattr(backward_induction, "_attach_resolved_output_layout", capture)
    _make_two_block_model(distributed=True).solve(params=_PARAMS, log_level="off")
    kinds = {
        transfer.kind
        for core in captured
        for transfer in core.input_transfer_plan
        if transfer.target.regime == "second"
    }

    assert ValueTransferKind.CROSS_MESH_COPY in kinds


@_skip_pytest_parallel
@pytest.mark.parametrize("regime", ["first", "second"])
def test_a_two_block_solve_publishes_the_single_device_values(
    *, regime: RegimeName
) -> None:
    """A cross-mesh copy delivers the stored values unchanged."""
    placed = _make_two_block_model(distributed=True).solve(
        params=_PARAMS, log_level="off"
    )
    reference = _make_two_block_model(distributed=False).solve(
        params=_PARAMS, log_level="off"
    )

    for period in placed.values:
        if regime in placed.values[period]:
            np.testing.assert_array_equal(
                np.asarray(placed.values[period][regime]),
                np.asarray(reference.values[period][regime]),
            )


def _nbegm_toy(*, distributed_kind: bool) -> Model:
    """The NB-EGM ride-along toy at its smallest grids.

    Its terminal `dead` regime does not read the ride-along type, so on four
    devices the sharded `alive` regime takes a two-device block and `dead` is
    placed on a device that block leaves free.
    """
    from tests.test_models import nbegm_ride_along_toy  # noqa: PLC0415

    return nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=4,
        n_liquid=24,
        n_savings=32,
        distributed_kind=distributed_kind,
    )


def _nbegm_toy_params() -> dict[str, float]:
    """The toy's parameters."""
    from tests.test_models import nbegm_ride_along_toy  # noqa: PLC0415

    return nbegm_ride_along_toy.build_params()


def _carry_template_devices(
    *, model: Model, regime_name: RegimeName
) -> set[jax.Device]:
    """Return every device one regime's continuation template leaves sit on."""
    template = model._regimes[regime_name].solution.continuation_template
    assert isinstance(template, ContinuationReader)
    return {
        device
        for leaf in template.leaves().values()
        for device in leaf.sharding.device_set
    }


@_skip_pytest_parallel
def test_the_terminal_egm_regime_beside_a_sharded_one_is_placed_off_device_zero() -> (
    None
):
    """The block the sharded regime takes leaves the terminal regime elsewhere."""
    model = _nbegm_toy(distributed_kind=True)

    assert model._regimes["dead"].solution.submesh_device_ids != (0,)


@_skip_pytest_parallel
def test_a_placed_single_device_regime_keeps_its_carry_template_on_its_device() -> None:
    """An EGM carry template lives on the devices its regime was placed on."""
    model = _nbegm_toy(distributed_kind=True)
    placed = set(model._regimes["dead"].solution.placed_devices())

    assert _carry_template_devices(model=model, regime_name="dead") == placed


@_skip_pytest_parallel
@pytest.mark.parametrize("regime", ["alive", "dead"])
def test_the_nbegm_toy_publishes_the_same_values_under_both_placements(
    *, regime: RegimeName, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Co-mapped NB-EGM keeps ordinary ownership and the same solved values."""
    params = _nbegm_toy_params()
    run = backward_induction._run_period_kernel
    nominations: list[tuple[str, ...]] = []

    def observe(**kwargs: Any) -> Any:
        if kwargs["regime_name"] == "alive":
            nominations.extend(
                core.donated_arguments for core in kwargs["compiled_cores"].values()
            )
        return run(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backward_induction, "_run_period_kernel", observe)
        placed = _nbegm_toy(distributed_kind=True).solve(params=params, log_level="off")
    assert nominations
    assert not any(nominations)
    canonical = _nbegm_toy(distributed_kind=False).solve(params=params, log_level="off")

    for period in placed.values:
        if regime not in placed.values[period]:
            continue
        assert_agrees_to_ulp(
            got=np.asarray(placed.values[period][regime]),
            expected=np.asarray(canonical.values[period][regime]),
            n_ulp=8,
            err_msg=f"regime {regime!r}, period {period}",
        )


@_skip_pytest_parallel
def test_simulation_topology_reads_a_proper_submesh_on_all_subject_devices() -> None:
    """The three-type value keeps its shape in a four-device replicated read."""
    model = _make_three_type_model(distributed=True)
    topologies = {
        phase: _get_regime_V_shapes_and_shardings(
            regimes=model._regimes,
            flat_params=model._process_params(_PARAMS),
            phase=phase,
        )
        for phase in ("solve", "simulate")
    }
    stored = topologies["solve"]["working"]
    read = topologies["simulate"]["working"]
    assert stored.shape == read.shape == (3, 12)
    assert {device.id for device in stored.sharding.device_set} == {0, 1, 2}
    assert {device.id for device in read.sharding.device_set} == {0, 1, 2, 3}
    assert read.sharding.is_fully_replicated


def _shape_only_transfer_inputs(*, values: Mapping[str, jax.Array]) -> jax.Array:
    """Both runtime copies are pruned; only the declared shape affects output."""
    return jnp.arange(values["first"].size, dtype=values["first"].dtype)


@_skip_pytest_parallel
@pytest.mark.parametrize("shared", [False, True])
def test_pruned_transfer_destinations_remain_budgeted_before_dispatch(
    *, monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    """Device3 originals and distinct device2 copies survive compiler pruning."""
    payload = 1024 * 1024
    dtype = jnp.zeros(()).dtype
    stored = jax.sharding.SingleDeviceSharding(jax.devices()[3])
    required = jax.sharding.SingleDeviceSharding(jax.devices()[2])
    source = jax.device_put(np.full(payload // dtype.itemsize, -3, dtype=dtype), stored)
    lowering_value = jax.device_put(source, required)
    address = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="done"
    )
    transfers = tuple(
        ResolvedValueTransfer(
            target=address,
            source=ValueConsumerAddress(
                source_period=0,
                source_regime="acting",
                core_key="main",
                channel=ValueInputChannel.NEXT_REGIME_VALUE,
                argument="values",
                path=(name,),
            ),
            kind=ValueTransferKind.COPY_TO_SOURCE_LAYOUT,
            stored_sharding=stored,
            source_sharding=required,
            expected_shape=source.shape,
            expected_dtype=source.dtype,
            reused_by_several_consumers=shared,
        )
        for name in ("first", "second")
    )
    program = ResolvedCoreProgram(
        name="main",
        function=_shape_only_transfer_inputs,
        arguments={
            "values": MappingProxyType(
                {"first": lowering_value, "second": lowering_value}
            )
        },
        static_kwargs={},
        requirements=CoreExecutionRequirements(
            value_reads=tuple(
                ValueRead(target=address, source=transfer.source)
                for transfer in transfers
            )
        ),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
        tile_widths={},
        specialization_key=(),
        input_transfer_plan=transfers,
    )
    metadata = backward_induction._ProgramExecutionMetadata(
        requirements=program.requirements,
        disposition=program.disposition,
        scope=program.scope,
        input_transfer_plan=transfers,
    )
    copies = backward_induction._period_copy_reservations(
        period=0,
        metadata={("acting", 0, "main"): metadata},
    )
    inventory = ResidentInventory(
        device_ids=(2,),
        live={},
        peer_bytes={2: 0},
        declared_inputs=(),
        shared_copies=copies,
    )
    layout = resolve_output_layout(
        core_key="main",
        value_template=lowering_value,
        state_order=("wealth",),
        output_roles=VALUE,
    )
    triple = ("acting", 0, "main")
    candidate = (triple, ())
    compiled: dict[Hashable, jax.stages.Compiled] = {}
    backward_induction._lower_and_compile_wave(
        new_lowerings={"transferred": candidate},
        resolved_programs={candidate: program},
        all_layouts={triple: layout},
        internal_templates={candidate: {}},
        donations={candidate: ()},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        n_triples_per_lowering={"transferred": 1},
        log_kernel_memory=False,
        n_workers=1,
        logger=logging.getLogger(__name__),
        compiled=compiled,
        labels={},
    )
    executable = compiled["transferred"]
    assert all(
        value is None for value in executable.input_shardings[1]["values"].values()
    )
    peak = compiler_peak_bytes(compiled=executable, widths={})
    copy_count = 1 if shared else 2

    def resident(comp: jax.stages.Compiled) -> int:
        return backward_induction._candidate_resident_bytes(
            compiled=comp,
            program=program,
            internal_arguments={},
            inventory=inventory,
        )

    assert resident(executable) == copy_count * payload
    copies_made: list[jax.Array] = []
    apply = transfers_module.apply_value_transfer

    def observe(
        *,
        value: object,
        transfer: ResolvedValueTransfer,
        on_materialized: MaterializedTransferObserver | None = None,
    ) -> jax.Array:
        result = apply(value=value, transfer=transfer, on_materialized=on_materialized)
        copies_made.append(result)
        return result

    monkeypatch.setattr(transfers_module, "apply_value_transfer", observe)
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        plan_workspace(
            axes=(),
            compile_candidate=lambda _widths: executable,
            budget_bytes=peak + copy_count * payload - 1,
            resident_bytes_for=resident,
        )
    assert copies_made == []
    generous = plan_workspace(
        axes=(),
        compile_candidate=lambda _widths: executable,
        budget_bytes=peak + copy_count * payload,
        resident_bytes_for=resident,
    )
    core = backward_induction._attach_resolved_output_layout(
        compiled=generous.compiled,
        layout=layout,
        tile_widths={},
        input_transfer_plan=transfers,
        name="main",
    )
    if shared:
        core = dataclasses.replace(
            core,
            transfer_cache=PeriodTransferCache(
                registry=BufferRegistry(),
                consumer_counts={(address, required): 1},
            ),
        )
    output = core(values={"first": source, "second": source})
    assert isinstance(output, jax.Array)
    jax.block_until_ready((output, copies_made))
    assert len(copies_made) == copy_count
    assert concrete_device_bytes(tree=copies_made)[2] == copy_count * payload
    assert (
        concrete_device_bytes(tree=(output, copies_made))[2]
        == (copy_count + 1) * payload
    )
    assert output.devices() == {jax.devices()[2]}
    assert source.devices() == {jax.devices()[3]}
    assert not source.is_deleted()
    np.testing.assert_array_equal(source, np.full(source.shape, -3))
    np.testing.assert_array_equal(output, np.arange(source.size))


@_skip_pytest_parallel
@pytest.mark.parametrize(
    ("sharded", "supplied"), [(False, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_uniform_entry_grid_uses_selected_device_and_keeps_source_owners(
    *,
    monkeypatch: pytest.MonkeyPatch,
    sharded: bool,
    log_level: LogLevel,
    supplied: bool,
) -> None:
    """Uniform support and its live sources are charged on every consuming device."""

    selected = (1, 3) if sharded else (2,)
    model = _uniform_placement_model(selected=selected, sharded=sharded)
    params = {
        "alive": {
            "income": {"start": jnp.asarray(1.0), "stop": jnp.asarray(3.0)},
            "koopmans_aggregator": {"discount_factor": 0.9},
        },
        "done": {},
    }
    initial = {
        "income": jnp.asarray([2.0, 2.0]),
        "kind": jnp.asarray([0, 1]),
        "age": jnp.asarray([0.0, 0.0]),
        "regime_id": jnp.asarray([0, 0]),
    }
    values = model.solve(params=params, log_level=log_level) if supplied else None
    sources = measure_buffer_footprint(tree=(params, initial, values))
    assert jax.devices()[0] in sources.spans
    grids: list[Float1D] = []
    monkeypatch.setattr(
        SimulationProcessGrids,
        "_produce",
        functools.partialmethod(
            _observe_selected_uniform_grid,
            original=SimulationProcessGrids._produce,
            selected=selected,
            sources=sources,
            grids=grids,
        ),
    )
    result = model.simulate(
        params=params, initial_conditions=initial, solution=values, log_level=log_level
    )
    assert len(grids) == 1
    np.testing.assert_array_equal(grids[0], [1.0, 1.5, 2.0, 2.5, 3.0])
    assert result.raw_results["alive"][0].V_arr.sharding.device_set == {
        jax.devices()[device] for device in selected
    }
    rows = result.to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        rows.loc[
            rows["regime_name"] == "alive", ["income", "saving", "value"]
        ].to_numpy(),
        [[2.0, 1.0, 3.0], [2.0, 1.0, 3.0]],
    )


# keyword-only-exempt: library-callback=functools.partialmethod
def _observe_selected_uniform_grid(
    self: SimulationProcessGrids,
    *,
    original: Callable[..., Float1D],
    selected: tuple[int, ...],
    sources: DeviceBufferFootprint,
    grids: list[Float1D],
    **kwargs: Any,
) -> Float1D:

    missing = resident_bytes_by_device(
        live=sources, arguments=self.snapshot(), devices=tuple(sources.spans)
    )
    assert not any(missing.values()), (
        "Original source buffers were omitted before grid allocation"
    )
    grid = original(self, **kwargs)
    assert grid.sharding.device_set == {jax.devices()[device] for device in selected}
    grids.append(grid)
    return grid


@categorical(ordered=False)
class _UniformPlacementRegimeId:
    alive: ScalarInt
    done: ScalarInt


def _uniform_placement_utility(
    *, income: ScalarFloat, saving: ScalarFloat, kind: ScalarInt
) -> ScalarFloat:
    return income + saving + 0 * kind


def _uniform_placement_terminal(*, kind: ScalarInt) -> ScalarFloat:
    return 0.0 * kind


def _uniform_placement_transition() -> ScalarInt:
    return _UniformPlacementRegimeId.done


def _uniform_placement_initial_age(age: float) -> bool:
    return age == 0


def _uniform_placement_terminal_age(age: float) -> bool:
    return age == 1


def _uniform_placement_model(*, selected: tuple[int, ...], sharded: bool) -> Model:
    from lcm import UniformIIDProcess  # noqa: PLC0415

    return Model(
        regimes={
            "alive": UserRegime(
                transition=_uniform_placement_transition,
                active=_uniform_placement_initial_age,
                states={"income": UniformIIDProcess(n_points=5)},
                actions={"saving": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _uniform_placement_utility},
            ),
            "done": UserRegime(
                transition=None,
                active=_uniform_placement_terminal_age,
                functions={"utility": _uniform_placement_terminal},
            ),
        },
        states={"kind": DiscreteGrid(_TwoValuedType)},
        state_transitions={"kind": fixed_transition("kind")},
        regime_id_class=_UniformPlacementRegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(
            device_memory_bytes=2**28,
            devices=selected,
            sharded_states=("kind",) if sharded else (),
        ),
    )
