"""Shared simulation operands follow the actual subject devices in both modes.

Run this module alone so its four-device CPU topology precedes initialization.
"""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Never

import jax
import jax._src.core
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ValueRead,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.policy_programs import ReplayPayload
from _lcm.simulation.program_types import subject_axis
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import SimulationRuntime, _materialize_abstract
from _lcm.typing import ParticipationQueryArguments
from lcm.exceptions import ExecutionPlanningError
from tests.conftest import assert_agrees_to_ulp

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _TOPOLOGY_AVAILABLE = True
except RuntimeError:
    _TOPOLOGY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TOPOLOGY_AVAILABLE, reason="Requires a fresh four-CPU-device process"
)


def _subject_with_shared_operands(
    *,
    state: jax.Array,
    action_grid: jax.Array,
    params: Mapping[str, jax.Array],
    age: jax.Array,
    period: jax.Array,
    reference: Mapping[str, jax.Array],
) -> jax.Array:
    """Use the full shared arrays even when their size equals the population."""
    return (
        state
        + action_grid.sum()
        + params["weights"].sum()
        + age
        + period
        + reference["done"][0]
    )


def _program() -> CoreProgram:
    """Declare a real subject-tiled body and one already-placed value read."""
    name = "simulate_transition"
    return CoreProgram(
        name=name,
        function=_SubjectTiled(
            func=_subject_with_shared_operands, subject_arg_names=("state",)
        ),
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name=name, subject_arg_names=("state",)
        ),
        requirements=CoreExecutionRequirements(
            tiled_axes=(subject_axis(state_names=("state",)),),
            value_reads=(
                ValueRead(
                    target=ValueArtifactAddress(
                        kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="done"
                    ),
                    source=ValueConsumerAddress(
                        source_period=0,
                        source_regime="alive",
                        core_key=name,
                        channel=ValueInputChannel.NEXT_REGIME_VALUE,
                        argument="reference",
                        path=("done",),
                    ),
                ),
            ),
        ),
        output_roles="next_state",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


@pytest.mark.parametrize("prewarm", [False, True])
def test_runtime_places_shared_operands_on_subject_devices(*, prewarm: bool) -> None:
    """Lazy and AOT calls exclude device zero and keep equal-size grids shared."""
    devices = jax.devices()
    subject_devices = (devices[3], devices[1], devices[2])
    mesh = jax.make_mesh(
        (3,), ("X",), (jax.sharding.AxisType.Auto,), devices=subject_devices
    )
    subject_sharding = jax.NamedSharding(mesh, jax.P("X"))
    shared_sharding = jax.NamedSharding(mesh, jax.P())
    state = jax.device_put(jnp.arange(6, dtype=float), subject_sharding)
    action_grid = jax.device_put(jnp.arange(6, dtype=float), devices[0])
    reference = jax.device_put(jnp.full(6, 4.0), shared_sharding)
    arguments = {
        "state": state,
        "action_grid": action_grid,
        "params": MappingProxyType(
            {"weights": jax.device_put(jnp.arange(1, 7, dtype=float), devices[0])}
        ),
        "age": jax.device_put(jnp.asarray(2.0), devices[0]),
        "period": jax.device_put(jnp.int32(1), devices[0]),
        "reference": MappingProxyType({"done": reference}),
    }
    runtime = SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(1, 2, 3),
            sharded_states=frozenset({"state"}),
            axis_widths=MappingProxyType({"subject": 3}),
            device_memory_bytes=None,
        ),
        enable_jit=True,
        subject_devices=subject_devices,
    )
    program = _program()
    if prewarm:
        runtime.prepare(program=program, arguments=arguments, period=0, n_subjects=6)
    result = runtime.dispatch(
        program=program, arguments=arguments, period=0, n_subjects=6
    )

    assert isinstance(result, jax.Array)
    assert_agrees_to_ulp(
        got=np.asarray(result), expected=np.arange(43.0, 49.0), n_ulp=0
    )
    result_sharding = result.sharding
    assert isinstance(result_sharding, jax.NamedSharding)
    assert tuple(result_sharding.mesh.devices.flat) == subject_devices
    assert arguments["state"] is state
    assert state.sharding == subject_sharding
    assert action_grid.sharding.device_set == {devices[0]}
    assert not reference.is_deleted()
    assert reference.sharding == shared_sharding


@pytest.mark.parametrize("budget_bytes", [None, 1_000_000])
def test_operand_placement_preserves_subjects_and_addressed_values(
    *, budget_bytes: int | None
) -> None:
    """Exact names distinguish equal-size shared grids, subject keys and V leaves."""
    devices = jax.devices()
    subject_devices = (devices[3], devices[1], devices[2])
    mesh = jax.make_mesh(
        (3,), ("X",), (jax.sharding.AxisType.Auto,), devices=subject_devices
    )
    subject_sharding = jax.NamedSharding(mesh, jax.P("X"))
    shared_sharding = jax.NamedSharding(mesh, jax.P())
    state = jax.device_put(jnp.arange(6.0), subject_sharding)
    keys = jax.device_put(jax.random.split(jax.random.key(0), 6), devices[0])
    value = jax.device_put(jnp.arange(6.0), devices[0])
    placed = place_simulation_arguments(
        arguments={
            "state": state,
            "key": keys,
            "action_grid": jax.device_put(jnp.arange(6.0), devices[0]),
            "reference": {"done": value, "scale": 2.0},
        },
        subject_arg_names=("state", "key"),
        value_reads=_program().requirements.value_reads,
        devices=subject_devices,
        budget_bytes=budget_bytes,
        live_footprint=measure_buffer_footprint(tree=(state, keys, value)),
        budget_devices=tuple(devices),
    )

    assert placed["state"] is state
    assert state.sharding == subject_sharding
    placed_keys = placed["key"]
    assert isinstance(placed_keys, jax.Array)
    assert placed_keys.sharding == subject_sharding
    np.testing.assert_array_equal(
        jax.random.key_data(placed_keys), jax.random.key_data(keys)
    )
    action_grid = placed["action_grid"]
    assert isinstance(action_grid, jax.Array)
    assert action_grid.sharding == shared_sharding
    reference = placed["reference"]
    assert isinstance(reference, Mapping)
    assert reference["done"] is value
    scale = reference["scale"]
    assert isinstance(scale, jax.Array)
    assert scale.sharding == shared_sharding
    assert not value.is_deleted()


def _refuse_operand_copy[Ignored](*args: Ignored, **kwargs: Ignored) -> Never:
    del args, kwargs
    raise AssertionError("A shared operand was copied before checking headroom")


def test_shared_operand_copy_checks_source_destination_and_scratch_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proper-subset replica is refused before any destination allocation."""
    devices = tuple(jax.devices())
    source = jax.device_put(jnp.arange(6.0), devices[0])
    live = measure_buffer_footprint(tree=source)
    monkeypatch.setattr(jax, "device_put", _refuse_operand_copy)
    with pytest.raises(ExecutionPlanningError, match="budget"):
        place_simulation_arguments(
            arguments={"action_grid": source},
            subject_arg_names=(),
            value_reads=(),
            devices=(devices[3], devices[1], devices[2]),
            budget_bytes=2 * source.nbytes - 1,
            live_footprint=live,
            budget_devices=devices,
        )


def _describe_array(array: jax.Array) -> jax.ShapeDtypeStruct:
    """Preserve the actual leaf's layout and weak type in its shape descriptor."""
    return jax.ShapeDtypeStruct(
        array.shape, array.dtype, sharding=array.sharding, weak_type=array.weak_type
    )


def test_abstract_materialization_preserves_container_order_and_value_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shape-only placement keeps dynamic key order and protected value layouts."""
    devices = tuple(jax.devices())
    subject_devices = (devices[3], devices[1], devices[2])
    mesh = jax.make_mesh(
        (3,), ("X",), (jax.sharding.AxisType.Auto,), devices=subject_devices
    )
    subject = jax.NamedSharding(mesh, jax.P("X"))
    shared = jax.NamedSharding(mesh, jax.P())
    protected = jax.ShapeDtypeStruct(
        (6,), float, sharding=jax.sharding.SingleDeviceSharding(devices[0])
    )
    weights = jax.ShapeDtypeStruct((6,), float, sharding=shared)
    scalar = jax.ShapeDtypeStruct((), float, sharding=shared, weak_type=True)
    arguments = {
        "state": jax.ShapeDtypeStruct((6,), float, sharding=subject),
        "action_grid": weights,
        "params": {"zeta": scalar, "weights": weights, "alpha": scalar},
        "age": scalar,
        "period": jax.ShapeDtypeStruct((), jnp.int32, sharding=shared),
        "reference": {"done": protected},
    }
    runtime = SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(1, 2, 3),
            sharded_states=frozenset({"state"}),
            axis_widths=MappingProxyType({"subject": 3}),
            device_memory_bytes=None,
        ),
        enable_jit=True,
        subject_devices=subject_devices,
    )
    concrete_value = jax.device_put(jnp.arange(6.0), devices[0])
    concrete = ParticipationQueryArguments(
        by_target={
            "done": {
                "query": {
                    "zeta": concrete_value,
                    "wealth": concrete_value,
                    "alpha": concrete_value,
                }
            }
        }
    )
    carrier = jax.tree.map(_describe_array, concrete)
    assert isinstance(carrier, ParticipationQueryArguments)
    query = carrier.by_target["done"]["query"]
    protected_query_value = query["wealth"]
    assert isinstance(protected_query_value, jax.ShapeDtypeStruct)
    with pytest.raises(BeartypeCallHintParamViolation):
        ParticipationQueryArguments(by_target={"done": {"query": query}})
    value_read = _program().requirements.value_reads[0]
    query_read = replace(
        value_read,
        source=replace(
            value_read.source,
            argument="participation_arguments",
            path=("by_target", "done", "query", "wealth"),
        ),
    )
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _refuse_operand_copy)
        guard.setattr(
            jax._src.core.EvalTrace, "process_primitive", _refuse_operand_copy
        )
        with pytest.raises(AssertionError, match="before checking headroom"):
            jnp.zeros(3)
        placed_carrier = place_simulation_arguments(
            arguments={"participation_arguments": carrier},
            subject_arg_names=(),
            value_reads=(query_read,),
            devices=subject_devices,
        )["participation_arguments"]
        materialized = _materialize_abstract(
            runtime=runtime,
            program=_program(),
            arguments=arguments,
            period=0,
            n_subjects=6,
        )
    placed = materialized.arguments
    params = placed["params"]
    reference = placed["reference"]
    assert isinstance(params, MappingProxyType)
    assert tuple(params) == ("zeta", "weights", "alpha")
    assert all(
        params[name] is value
        for name, value in (("weights", weights), ("zeta", scalar), ("alpha", scalar))
    )
    assert scalar.weak_type
    assert isinstance(reference, MappingProxyType)
    assert reference["done"] is protected
    assert protected.sharding.device_set == {devices[0]}
    assert placed["state"] is arguments["state"]
    assert placed["action_grid"] is weights
    assert not runtime.cache
    assert isinstance(placed_carrier, ParticipationQueryArguments)
    placed_query = placed_carrier.by_target["done"]["query"]
    assert isinstance(placed_query, MappingProxyType)
    assert tuple(placed_query) == ("zeta", "wealth", "alpha")
    assert placed_query["wealth"] is protected_query_value
    assert placed_query["zeta"] is query["zeta"]
    assert placed_query["alpha"] is query["alpha"]
    assert protected_query_value.sharding.device_set == {devices[0]}


def test_replay_payload_placement_preserves_abstract_and_protected_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the replay tree, addressed arrays and strict runtime boundary."""
    devices = tuple(jax.devices())
    subject_devices = (devices[3], devices[1], devices[2])
    mesh = jax.make_mesh(
        (3,), ("X",), (jax.sharding.AxisType.Auto,), devices=subject_devices
    )
    shared = jax.NamedSharding(mesh, jax.P())
    concrete_value = jax.device_put(jnp.arange(6.0), devices[0])
    value_read = _program().requirements.value_reads[0]
    replay = ReplayPayload(
        arrays=(concrete_value, concrete_value),
        structure=jax.tree_util.tree_structure(
            {"zeta": concrete_value, "alpha": concrete_value}
        ),
    )
    abstract_replay = jax.tree.map(_describe_array, replay)
    assert isinstance(abstract_replay, ReplayPayload)
    with pytest.raises(BeartypeCallHintParamViolation):
        ReplayPayload(arrays=abstract_replay.arrays, structure=replay.structure)
    replay_read = replace(
        value_read,
        source=replace(value_read.source, argument="payload", path=("arrays", 0)),
    )
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _refuse_operand_copy)
        guard.setattr(
            jax._src.core.EvalTrace, "process_primitive", _refuse_operand_copy
        )
        placed_replay = place_simulation_arguments(
            arguments={"payload": abstract_replay},
            subject_arg_names=(),
            value_reads=(replay_read,),
            devices=subject_devices,
        )["payload"]
    assert isinstance(placed_replay, ReplayPayload)
    assert placed_replay.structure is replay.structure
    assert all(
        a is b
        for a, b in zip(placed_replay.arrays, abstract_replay.arrays, strict=True)
    )
    concrete_replay = place_simulation_arguments(
        arguments={"payload": replay},
        subject_arg_names=(),
        value_reads=(replay_read,),
        devices=subject_devices,
    )["payload"]
    assert isinstance(concrete_replay, ReplayPayload)
    assert concrete_replay.structure is replay.structure
    assert concrete_replay.arrays[0] is concrete_value
    assert concrete_replay.arrays[1].sharding == shared
    np.testing.assert_array_equal(concrete_replay.arrays[1], np.arange(6.0))
