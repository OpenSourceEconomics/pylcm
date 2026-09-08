"""Numeric entry staging uses the first selected device before subject sharding.

Run alone in a fresh four-CPU-device process.
"""

from types import MappingProxyType

import jax
import numpy as np
import pytest

from _lcm.dtypes import safe_to_int_dtype
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.simulation.initial_conditions import canonicalize_initial_conditions
from _lcm.simulation.operand_placement import place_simulation_arguments
from lcm.exceptions import ExecutionPlanningError

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _TOPOLOGY_AVAILABLE = True
except RuntimeError:
    _TOPOLOGY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TOPOLOGY_AVAILABLE, reason="Requires a fresh four-CPU-device process"
)


@pytest.mark.parametrize("order", [(3, 1, 2), (2, 3, 1)])
def test_numeric_entry_stages_and_pads_on_the_first_actual_device(
    order: tuple[int, ...],
) -> None:
    """A seven-row source remains readable while staged padding creates nine rows."""
    devices = tuple(jax.devices()[index] for index in order)
    source = jax.device_put(np.arange(7, dtype=np.int32), device=jax.devices()[0])
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=(source,)),
        solution=None,
        model_roots=(),
        devices=devices,
        budget_bytes=2**20,
    )
    initial = canonicalize_initial_conditions(
        initial_conditions={"regime_id": source},
        regimes=MappingProxyType({}),
        array_writer=owner,
    )
    owner.publish(stage="initial", tree=initial)
    assert initial["regime_id"].devices() == {devices[0]}
    padded, count = owner.pad(initial_conditions=initial, multiple=3)
    assert count == 7
    assert padded["regime_id"].devices() == {devices[0]}
    np.testing.assert_array_equal(padded["regime_id"], [0, 1, 2, 3, 4, 5, 6, 6, 6])
    owner.publish(stage="initial", tree=padded)
    placed = place_simulation_arguments(
        arguments=padded,
        subject_arg_names=("regime_id",),
        value_reads=(),
        devices=devices,
        budget_bytes=owner.budget_bytes,
        budget_devices=tuple(jax.devices()),
        live_footprint=owner.snapshot(),
    )["regime_id"]
    assert isinstance(placed, jax.Array)
    assert isinstance(placed.sharding, jax.NamedSharding)
    assert tuple(placed.sharding.mesh.devices.flat) == devices
    assert placed.sharding.spec == jax.P("X")
    assert all(shard.data.shape == (3,) for shard in placed.addressable_shards)
    np.testing.assert_array_equal(placed, [0, 1, 2, 3, 4, 5, 6, 6, 6])
    np.testing.assert_array_equal(source, np.arange(7, dtype=np.int32))


def test_retained_source_on_an_excluded_device_still_consumes_its_budget() -> None:
    """A small destination cannot hide a larger retained bank on source device zero."""
    source = jax.device_put(np.arange(16, dtype=np.int32), device=jax.devices()[0])
    assert source.nbytes == 64
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=(source,)),
        solution=None,
        model_roots=(),
        devices=tuple(jax.devices()[1:]),
        budget_bytes=63,
    )
    with pytest.raises(ExecutionPlanningError, match="budget"):
        safe_to_int_dtype(
            value=np.array([1], dtype=np.int64), name="small", array_writer=owner
        )
    np.testing.assert_array_equal(source, np.arange(16, dtype=np.int32))
