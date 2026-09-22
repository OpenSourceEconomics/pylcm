"""Addressed taste keys retain the actual ordered subject-device layout.

Run this module alone in a fresh four-CPU-device process.
"""

from collections.abc import Callable
from functools import partialmethod
from typing import Any

import jax
import numpy as np
import pytest

from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import measure_buffer_footprint
from _lcm.simulation.taste_stream import (
    create_taste_shock_key,
    generate_taste_shock_keys,
)

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _TOPOLOGY_AVAILABLE = True
except RuntimeError:
    _TOPOLOGY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TOPOLOGY_AVAILABLE, reason="Requires a fresh four-CPU-device process"
)


def _oracle(*, start: int, count: int, population: int) -> np.ndarray:
    key = jax.random.key(2**40 + 17, impl="threefry2x32")
    for word in range(8):
        key = jax.random.fold_in(key, np.uint32(word))
    rows = []
    for row in range(start, start + count):
        bounded = min(row, population - 1)
        addressed = jax.random.fold_in(key, np.uint32(bounded >> 32))
        addressed = jax.random.fold_in(addressed, np.uint32(bounded & (2**32 - 1)))
        rows.append(np.asarray(jax.random.key_data(addressed)))
    return np.stack(rows)


# keyword-only-exempt: library-callback=functools.partialmethod
def _record_output(
    self: jax.stages.Compiled,
    *,
    original: Callable[..., object],
    calls: list[tuple[jax.stages.Compiled, object]],
    **arguments: Any,
) -> object:
    result = original(self, **arguments)
    calls.append((self, result))
    return result


@pytest.mark.parametrize("device_ids", [(3,), (3, 1, 2), (2, 3, 1), (0, 1, 2, 3)])
def test_addressed_key_profile_dispatches_directly_on_ordered_subject_devices(
    *, device_ids: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No excluded default device or post-dispatch reshard changes the output bank."""
    all_devices = tuple(jax.devices())
    devices = tuple(all_devices[index] for index in device_ids)
    memory = SimulationMemory(
        budget_bytes=2**24,
        devices=all_devices,
        subject_devices=devices,
        operations=ProfiledSimulationOperations(),
        inputs=measure_buffer_footprint(tree=()),
    )
    calls: list[tuple[jax.stages.Compiled, object]] = []
    monkeypatch.setattr(
        jax.stages.Compiled,
        "__call__",
        partialmethod(
            _record_output, original=jax.stages.Compiled.__call__, calls=calls
        ),
    )
    key = create_taste_shock_key(seed=2**40 + 17, memory=memory)
    assert isinstance(key, jax.Array)
    assert key.sharding.device_set == set(devices)
    start = 2**32 - 2
    result = generate_taste_shock_keys(
        key=key,
        address_words=tuple(range(8)),
        subject_slice=slice(start, start + 12),
        original_n_subjects=start + 7,
        memory=memory,
    )
    assert len(calls) == 2
    executable, dispatched = calls[-1]
    assert result is dispatched
    assert result.sharding.device_set == set(devices)
    assert result.sharding.is_equivalent_to(executable.output_shardings, ndim=1)
    if len(devices) > 1:
        assert isinstance(result.sharding, jax.NamedSharding)
        assert tuple(result.sharding.mesh.devices.flat) == devices
        assert result.sharding.spec == jax.P("X")
    assert {shard.device for shard in result.addressable_shards} == set(devices)
    assert all(
        shard.data.shape == (12 // len(devices),) for shard in result.addressable_shards
    )
    np.testing.assert_array_equal(
        jax.random.key_data(result),
        _oracle(start=start, count=12, population=start + 7),
    )
    np.testing.assert_array_equal(
        jax.random.key_data(key),
        jax.random.key_data(jax.random.key(2**40 + 17, impl="threefry2x32")),
    )


def test_ordered_taste_layouts_have_distinct_cache_entries() -> None:
    """The same address and shape never reuse an executable on another device order."""
    all_devices = tuple(jax.devices())
    operations = ProfiledSimulationOperations()
    outputs = []
    for ids in ((3, 1, 2), (2, 3, 1), (3, 1, 2)):
        devices = tuple(all_devices[index] for index in ids)
        memory = SimulationMemory(
            budget_bytes=2**24,
            devices=all_devices,
            subject_devices=devices,
            operations=operations,
            inputs=measure_buffer_footprint(tree=()),
        )
        key = create_taste_shock_key(seed=47, memory=memory)
        assert isinstance(key, jax.Array)
        result = generate_taste_shock_keys(
            key=key,
            address_words=tuple(range(8)),
            subject_slice=slice(0, 6),
            original_n_subjects=5,
            memory=memory,
        )
        outputs.append(np.array(jax.random.key_data(result), copy=True))
        assert isinstance(result.sharding, jax.NamedSharding)
        assert tuple(result.sharding.mesh.devices.flat) == devices
        memory.close_unit()
    assert len(operations.cache) == 4
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[0], outputs[2])
