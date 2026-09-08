"""Shared simulation operands follow the actual subject devices in both modes.

Run this module alone so its four-device CPU topology precedes initialization.
"""

from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from _lcm.simulation.program_types import subject_axis
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.runtime import SimulationRuntime
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


def _refuse_operand_copy(*args: object, **kwargs: object) -> object:
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
